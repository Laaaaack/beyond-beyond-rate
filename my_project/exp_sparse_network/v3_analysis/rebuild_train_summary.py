"""Rebuild a v3 training summary from the checkpoints on disk.

Why this exists
---------------
``sn_train_{withDelay,noDelay}_v3.py`` write one summary file per arm, keyed by run
tag, and each invocation writes the whole file. So a run launched with a reduced
``SEEDS`` list — as the delay arm's seed-44 extension was — replaces the summary
rather than extending it, and the earlier seeds' rows are lost even though their
checkpoints are not.

Every quantity in the summary is a function of the checkpoint and the fixed test
split, so a lost row can be recovered from the checkpoint alone. This script does
that: it enumerates ``sn_data/sparse_whole_{arm}_v3_k*_floor*_seed*.pt`` and re-runs
the training script's own ``evaluate_clean_and_activity`` on any checkpoint the
summary has no row for.

**Rows already present are kept verbatim, never re-measured.** Re-measuring is not
quite bit-exact: the training script's ``set_seed`` disables cuDNN, so a rebuild takes
a different convolution path, and near a spike threshold a float-level difference
flips a spike. Measured over the 32 rows that did survive, that moves ``clean_acc`` by
at most .002 (three test samples) and both sparsity axes by less than .0003 — harmless,
but there is no reason to import it into rows that are already correct. Running this
script when nothing is missing is therefore a no-op.

It is eval-only — no training, no checkpoint is modified — and it imports the
measurement function from the training script rather than reimplementing it, so the
recovered rows cannot drift from the ones a fresh training run would have written.

Writes ``sn_log/sparse_whole_{arm}_v3_train_summary.json``.
"""

import json
import re
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
CKPT_DIR = EXP_DIR / "sn_data"
LOG_DIR = EXP_DIR / "sn_log"

sys.path.insert(0, str(EXP_DIR))

import sn_train_noDelay_v3 as nodelay_train  # noqa: E402
import sn_train_withDelay_v3 as delay_train  # noqa: E402

# Arm name -> the training module that defines its network and its constants.
ARMS: tuple[tuple[str, object], ...] = (
    ("delay", delay_train),
    ("nodelay", nodelay_train),
)

# Real-run checkpoints only: `_probe`, `_ceilcal_*` and the v3L2/v3L12 layer-2 variants
# all carry a suffix, so anchoring the seed at the end of the stem excludes them.
RUN_TAG_PATTERN = re.compile(
    r"^sparse_whole_(?P<arm>delay|nodelay)_v3_"
    r"k(?P<k>[0-9.]+)_floor(?P<floor>[0-9.]+)_seed(?P<seed>[0-9]+)$"
)


def parse_run_tag(stem: str) -> dict | None:
    """Return the design variables encoded in a checkpoint file name.

    Args:
        stem: Checkpoint file name without its ``.pt`` extension.

    Returns:
        Dict with ``arm``, ``ceiling_k``, ``floor_strength`` and ``seed``, or None if
        the name is not a real-run v3 checkpoint.
    """
    match = RUN_TAG_PATTERN.match(stem)
    if match is None:
        return None
    return {
        "arm": match.group("arm"),
        "ceiling_k": float(match.group("k")),
        "floor_strength": float(match.group("floor")),
        "seed": int(match.group("seed")),
    }


def find_checkpoints(arm: str) -> list[tuple[Path, dict]]:
    """Return this arm's real-run v3 checkpoints, ordered by (k, floor, seed)."""
    found = []
    for path in CKPT_DIR.glob(f"sparse_whole_{arm}_v3_k*_floor*_seed*.pt"):
        design = parse_run_tag(path.stem)
        if design is not None and design["arm"] == arm:
            found.append((path, design))
    found.sort(key=lambda item: (item[1]["ceiling_k"], item[1]["floor_strength"],
                                 item[1]["seed"]))
    return found


def rebuild_arm(arm: str, train_module) -> dict[str, dict]:
    """Return one arm's summary, measuring only the rows that are missing.

    Args:
        arm: ``"delay"`` or ``"nodelay"``.
        train_module: The arm's training script module, used for its network class,
            its ``evaluate_clean_and_activity`` and its grid constants.

    Returns:
        The summary dict, run tag -> recorded metrics, ordered by (k, floor, seed).
    """
    summary_path = LOG_DIR / f"sparse_whole_{arm}_v3_train_summary.json"
    existing = {}
    if summary_path.exists():
        with open(summary_path) as handle:
            existing = json.load(handle)

    checkpoints = find_checkpoints(arm)
    missing = [item for item in checkpoints if item[0].stem not in existing]
    print(f"\n=== {arm}: {len(checkpoints)} checkpoints, "
          f"{len(existing)} rows present, {len(missing)} to measure ===")
    if not missing:
        return {path.stem: existing[path.stem] for path, _ in checkpoints}

    features, labels = train_module.load_shd_data(
        train_module.MAT_FILE, target_T=train_module.SIM_PARAMS["tSample"])
    _, _, test_loader = train_module.build_dataloaders(
        features, labels, batch_size=train_module.BATCH_SIZE, seed=42)

    for path, design in missing:
        net = train_module.SparseSHDNetwork(train_module.INPUT_DIM).to(
            train_module.device)
        net.load_state_dict(torch.load(path, map_location=train_module.device))

        metrics = train_module.evaluate_clean_and_activity(
            net, test_loader, design["ceiling_k"])
        existing[path.stem] = {
            "ceiling_k": design["ceiling_k"],
            "ceiling_strength": float(train_module.CEILING_STRENGTH),
            "floor_strength": design["floor_strength"],
            "theta_margin": float(train_module.THETA_MARGIN),
            "warmup_epochs": int(train_module.WARMUP_EPOCHS),
            "epochs": int(train_module.EPOCHS),
            "seed": design["seed"],
            "recovered_from_checkpoint": True,
            **{name: float(value) for name, value in metrics.items()},
        }
        print(f"{path.stem:<46} acc={metrics['clean_acc']:.4f} "
              f"a={metrics['spikes_per_active_neuron']:.2f} "
              f"s={metrics['silent_fraction']:.2%} "
              f"sp/neu={metrics['spikes_per_neuron']:.2f}", flush=True)

    return {path.stem: existing[path.stem] for path, _ in checkpoints}


def main() -> None:
    """Fill any missing summary rows for every configured arm."""
    for arm, train_module in ARMS:
        summary = rebuild_arm(arm, train_module)
        out_path = LOG_DIR / f"sparse_whole_{arm}_v3_train_summary.json"
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"{len(summary)} rows -> {out_path}")


if __name__ == "__main__":
    main()
