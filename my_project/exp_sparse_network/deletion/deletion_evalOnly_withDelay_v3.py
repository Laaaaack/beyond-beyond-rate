"""v3 Phase 1, WITH-DELAY arm: 1st-hidden-layer deletion sweep, eval-only.

Companion to the training script
[sn_train_withDelay_v3.py](../sn_train_withDelay_v3.py), which trained the v3
**factorial** — a per-pair ceiling ``relu(count - k)`` setting temporal sparsity
``a`` along ``CEILING_K = [1, 2, 4, 8]``, crossed with a membrane-potential floor
setting selectivity ``s`` along ``FLOOR_STRENGTH = [0, 1]``, at seeds 42 and 43 —
and saved one checkpoint per ``(k, floor, seed)``. This script performs **no
training**. It loads each of those 16 checkpoints and measures how much test
accuracy degrades as the **1st hidden layer's** spikes are deleted at evaluation
only.

Why the v3 checkpoints rather than v1's: in v1 the two sparsity axes were
confounded at ``rho(a, s) = -0.455`` in this arm and ``-0.943`` in the other, so no
sweep over those models could attribute a trend to one rather than the other. The
v3 grid breaks that by construction — ``rho(a, s) = -0.053`` (p = .85) across the
16 models — which is what makes a deletion curve read against ``a`` interpretable.
See document 5 §2b and §4 Phase 1.

The no-delay v3 arm is calibrated but not yet trained, so it has no sibling script
yet; when it is, it belongs beside this one as
``deletion_evalOnly_noDelay_v3.py``. The arms are kept as separate scripts (rather
than one script with a flag) for the same reason the training scripts are: this
network carries the learnable axonal delays ``delay1``/``delay2``, which are
themselves a timing mechanism, so its deletion curve measures sparsity's effect *on
top of* whatever the delays contribute.

Relation to [phase1_measure.py](../v3_analysis/phase1_measure.py): that script
measures only the ``p_d = 0.8`` endpoint of this sweep, because that single number
is the general-robustness covariate the Phase 1 regressions control on. This script
sweeps the full grid, which shows the *shape* of the degradation rather than one
point of it.

Where the delays sit relative to the perturbation site: ``delay1`` is the first
operation *after* the 1st hidden layer's spikes, so deletion is applied upstream of
it — exactly as in training, where the sparsity penalty also acts on the
pre-``delay1`` spike tensor.

Protocol rule (do not break): sparsity is applied during *training* on clean data;
deletion is applied here at *evaluation* only. The loaded checkpoint is never
modified, no gradients are taken, and the delays are used exactly as trained (the
adaptive clamping schedule is a training-time device and has no role here).

Per-spike deletion: each 1st-hidden spike is dropped independently with probability
``p_d``; survivors keep their exact times and no spikes are added. This is the one
perturbation here that is *not* rate-preserving — it is the milestone's
rate/robustness control, and its curve is read against the timing perturbations
(jitter, shift) rather than as a timing measure on its own.

What this script does, for every checkpoint listed in the training summary (the
authoritative live-checkpoint list — read rather than globbed, so we evaluate
exactly the models the milestone locked and reuse their recorded metadata):

- loads the SGD-delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``PD_VALUES`` on the 1st hidden layer, ``NUM_REPEATS`` times per p_d
  for error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_delay_v3_deletion_eval.json`` mapping each run tag to
  its ``acc(p_d)`` sweep, alongside the training summary's design knobs
  (``ceiling_k``, ``floor_strength``, ``seed``) and *measured* sparsity metrics, so
  the file is self-contained for the analysis.

The downstream analysis turns each ``acc(p_d)`` curve into a chance-corrected,
baseline-normalised ``temporal_score``. Plot it against the *measured* axes ``a``
(``spikes_per_active_neuron``) and ``s`` (``silent_fraction``) separately — never
against ``spikes_per_neuron`` alone, which is their product and moves with both, and
never against the penalty knobs, whose map to achieved sparsity is nonlinear and
seed-dependent (document 5 §8 rule 1). Clean accuracy across this grid is .707-.859,
so raw accuracy drops are not comparable across cells; the normalised score is.

Architecture: Input(700) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA), with
learnable delays after each hidden layer.
Sweep (eval only): p_d in {0.0, 0.2, 0.4, 0.6, 0.8}.

v3 note — this sweep needs no support-window correction, unlike relocation and
jitter. Deletion removes spikes and never chooses a destination bin, so there is no
window to confine and the temporal support is irrelevant to it. Its role in v3 is
unchanged: it is the general-robustness control the timing probes are read against,
and Phase 0 showed that control is not optional — it tracked sparsity at +0.936 /
+0.713, nearly as strongly as the timing probes did, so much of what looks like a
timing effect is a network simply being robust to hidden interference of any kind.
See ``v3_analysis/temporal_support.py`` and document 5 §2a, §4 Phase 0.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Directory of this script. Dataset, checkpoint, and log paths are anchored here so
# the script can be launched from any working directory. The sparsity training is
# shared across the perturbation experiments, so its checkpoints and summaries live
# one level up in exp_sparse_network/; only this deletion eval's own output is local.
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / ".." / "sn_data"        # trained checkpoints read from here
TRAIN_LOG_DIR = SCRIPT_DIR / ".." / "sn_log"    # training summary read from here
LOG_DIR = SCRIPT_DIR / "log"                    # deletion eval results written here
SHD_DATA_DIR = SCRIPT_DIR / ".." / "shd" / "shd_data"  # SHD .mat source

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Quick pipeline check. True evaluates only the first MAX_QUICK_CHECKPOINTS models
# on a coarse p_d grid with a single repeat, and suffixes the output file with
# RUN_SUFFIX so a probe can never overwrite the real eval results. False = the real
# sweep over every checkpoint in the training summary.
QUICK_TEST: bool = False

# Suffix appended to the results filename when running a probe (see QUICK_TEST).
# Empty for the real run. Mirrors the convention in the training scripts.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Checkpoints evaluated when QUICK_TEST is set (in training-summary order).
MAX_QUICK_CHECKPOINTS: int = 2

# --- Milestone scope (matches the training script) ---
DATASET_KEY: str = "whole"
DELAY_TAG: str = "delay"      # this arm; tags the summary and results files
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# Training generation to evaluate. "v3" selects the 16-model factorial grid trained
# by sn_train_withDelay_v3.py; it matches that script's own VERSION_TAG, and it tags
# both the summary read here and the results written below, so v3 sweeps can never
# collide with v1's files in log/.
VERSION_TAG: str = "v3"

# The training summary enumerating this arm's live v3 checkpoints. Its keys are the
# run tags (``sparse_whole_delay_v3_k{k}_floor{F}_seed{seed}``) and
# ``sn_data/{run_tag}.pt`` is the matching checkpoint. The 4-corner calibration probe
# is kept separately as ..._train_summary_probe.json and is NOT evaluated here.
TRAIN_SUMMARY_FILE: str = (
    f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_train_summary.json"
)

# Training-summary fields carried through into the results file, so the analysis can
# plot temporal score against both design knobs and both *measured* sparsity axes
# without re-joining the summary. ``spikes_per_active_neuron`` (= ``a``) and
# ``silent_fraction`` (= ``s``) are the two axes the v3 factorial separates; v1's
# summaries lacked the former, which is how the confound went unnoticed for two
# versions. ``over_k_fraction`` records how hard the ceiling actually bound.
SUMMARY_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "ceiling_k",
    "ceiling_strength",
    "floor_strength",
    "seed",
    "clean_acc",
    "firing_rate",
    "spikes_per_neuron",
    "spikes_per_active_neuron",
    "silent_fraction",
    "over_k_fraction",
)

# --- SLAYER neuron and simulation descriptors (identical to training) ---
SIM_PARAMS = {"Ts": 1, "tSample": 200}
LIF_PARAMS = {
    "type": "SRMALPHA",
    "theta": 10,
    "tauSr": 1,
    "tauRho": 0.1,
    "tauRef": 2,
    "scaleRef": 2,
    "scaleRho": 0.1,
}

# --- Data split ratios (identical to training, so the test set matches) ---
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# --- Network / evaluation hyper-parameters ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
BATCH_SIZE: int = 128
SEED: int = 42                # base seed for the deletion repeats
MAX_DELAY: int = 64           # recorded for parity with training; unused at eval

# --- Deletion sweep: probability of dropping each spike. 0 = clean baseline. ---
# The grid the earlier fixed-weight deletion experiments used, so this milestone's
# curves can be read beside them.
PD_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8]

# --- Repeats per p_d, for error bars (the deletion mask is stochastic). ---
NUM_REPEATS: int = 3


def resolve_eval_config() -> tuple[list[float], int, int | None]:
    """Return (pd_values, num_repeats, max_checkpoints) for this run.

    Collapses to a tiny, fast sweep when ``QUICK_TEST`` is set, so the eval
    pipeline can be validated before committing to every checkpoint.

    Returns:
        Tuple of (pd_values, num_repeats, max_checkpoints), where
        ``max_checkpoints`` is None for the real run (evaluate all of them).
    """
    if QUICK_TEST:
        return [0.0, 0.4, 0.8], 1, MAX_QUICK_CHECKPOINTS
    return PD_VALUES, NUM_REPEATS, None


def load_shd_data(mat_path: str, target_T: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Load an SHD dataset from a .mat file and pad the time dimension.

    Args:
        mat_path: Path to the .mat file containing 'X' and 'Y'.
        target_T: Target time dimension (pad with zeros if shorter).

    Returns:
        Tuple of (X, Y) where X has shape (N, neurons, target_T).
    """
    data = loadmat(mat_path)
    X = data["X"]
    Y = data["Y"].ravel()

    n_samples, n_neurons, T = X.shape
    if T < target_T:
        padded = np.zeros((n_samples, n_neurons, target_T), dtype=X.dtype)
        padded[:, :, :T] = X
        X = padded
        print(f"Padded time dimension from {T} to {target_T}")

    print(f"Loaded {mat_path}: X={X.shape}, Y={Y.shape}, classes={len(np.unique(Y))}")
    return X, Y


class SpikeDataset(Dataset):
    """Wrap numpy spike trains and labels into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


def get_split_indices(
    split_range: tuple[float, float],
    total: int,
) -> np.ndarray:
    """Return the index array for a given fractional range of the dataset."""
    start = int(total * split_range[0])
    end = int(total * split_range[1])
    return np.arange(start, end)


def build_test_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build the test DataLoader over the fixed test split.

    The split is a fixed fractional range, so the test set is identical to the one
    the training script held out — and identical across every checkpoint and both
    arms — which is what makes the deletion curves comparable.

    Args:
        X: Full dataset features, shape (N, neurons, T).
        Y: Full dataset labels, shape (N,).
        batch_size: Batch size for the loader.

    Returns:
        The test DataLoader (unshuffled).
    """
    test_idx = get_split_indices(TEST_RANGE, len(Y))
    test_ds = SpikeDataset(X[test_idx], Y[test_idx])
    print(f"Test set: {len(test_ds)} samples")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def delete_hidden_batch(
    hidden_spikes: torch.Tensor,
    p_d: float,
) -> torch.Tensor:
    """Vectorised GPU-side per-spike deletion (eval-only helper).

    Each spike is kept independently with probability ``1 - p_d`` via a Bernoulli
    mask drawn directly on the input device. Surviving spike times are unchanged
    and no new spikes are created.

    Unlike jitter and shift, this perturbation is deliberately **not**
    rate-preserving: it removes roughly a fraction ``p_d`` of the layer's spikes
    outright. It is the milestone's rate/robustness control rather than a timing
    measure. A network whose accuracy rests on a few precisely-placed spikes should
    degrade faster than one whose accuracy rests on a redundant population rate, so
    this curve is read *against* the timing perturbations — an already-sparse layer
    has fewer spikes to lose, which is itself part of what the comparison probes.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        p_d: Per-spike deletion probability in [0, 1]. 0 means no deletion.

    Returns:
        Spike tensor with the same shape, dtype and device.
    """
    if p_d <= 0:
        return hidden_spikes
    if p_d >= 1:
        return torch.zeros_like(hidden_spikes)

    is_spike = hidden_spikes > 0.5
    keep = torch.rand_like(hidden_spikes) > p_d
    return (is_spike & keep).to(hidden_spikes.dtype)


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with learnable delays, for the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_train_withDelay.py](../sn_train_withDelay.py) — ``fc1``/``fc2``/``fc3``
    weight-norm parameters plus ``delay1``/``delay2`` — so this class loads those
    checkpoints directly. The training script's adaptive delay-clamping schedule is
    deliberately absent: it shapes delays *during* training, and the loaded values
    are used here exactly as saved.

    ``forward_with_hidden_perturbation`` deletes 1st hidden layer spikes before
    the readout. It is eval-only; nothing here is ever trained.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = HIDDEN_UNITS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer

        self.fc1 = nn.utils.weight_norm(
            slayer.dense(input_dim, hidden_units), name="weight"
        )
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(hidden_units, hidden_units), name="weight"
        )
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(hidden_units, num_classes), name="weight"
        )

        # delay1 sits at the start of _second_hidden_and_output, i.e. immediately
        # after the perturbation site; delay2 stays between the fc2 spike and fc3.
        self.delay1 = slayer.delay(hidden_units)
        self.delay2 = slayer.delay(hidden_units)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the input is 5-D NCHWT on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(device)

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike -> 1st hidden spikes (strictly binary)."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> delay1 -> fc2 -> spike -> delay2 -> fc3 -> spike."""
        x = self.delay1(hidden1)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean forward pass (equivalent to the p_d = 0 baseline)."""
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        p_d: float = 0.0,
    ) -> torch.Tensor:
        """Eval-only: delete 1st hidden layer spikes before the readout.

        Must be called inside ``torch.no_grad()`` — ``delete_hidden_batch`` is not
        autograd-safe. ``p_d = 0`` reduces to the clean forward pass.

        Args:
            x: Input spike trains.
            p_d: Per-spike deletion probability in [0, 1].

        Returns:
            The output spike tensor.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if p_d > 0:
            hidden1 = delete_hidden_batch(hidden1, p_d)
        return self._second_hidden_and_output(hidden1)


def load_checkpoint(
    checkpoint_path: Path,
    input_dim: int = INPUT_DIM,
) -> SparseSHDNetwork:
    """Load a trained checkpoint into a fresh delay network in eval mode.

    Args:
        checkpoint_path: Path to the saved state_dict (.pt).
        input_dim: Number of input neurons (matches the dataset variant).

    Returns:
        The network in eval mode with the checkpoint weights loaded.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = SparseSHDNetwork(input_dim, HIDDEN_UNITS, NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


@torch.no_grad()
def test_at_pd(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    p_d: float,
) -> float:
    """Return test accuracy with spikes deleted from the 1st hidden layer at ``p_d``.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        p_d: Per-spike deletion probability; 0 is the clean baseline.

    Returns:
        Fraction of correctly classified test samples.
    """
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        outputs = net.forward_with_hidden_perturbation(x_batch, p_d=p_d)
        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)
    return correct / max(1, total)


def test_with_repeats(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    p_d: float,
    num_repeats: int,
) -> dict:
    """Repeat the thinned evaluation ``num_repeats`` times for error bars.

    Each repeat re-seeds torch (which seeds the CUDA generator the deletion mask draws
    from) so the perturbation differs run to run but the whole sweep is
    reproducible. ``p_d = 0`` involves no draw, so its repeats are identical and
    its std must come out at 0 — a useful check that no other randomness leaks into
    evaluation.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        p_d: Per-spike deletion probability in [0, 1].
        num_repeats: Number of repeats.

    Returns:
        Dict with keys ``mean``, ``std``, ``values``.
    """
    accuracies = []
    for repeat in range(num_repeats):
        torch.manual_seed(SEED + repeat)
        np.random.seed(SEED + repeat)
        accuracies.append(test_at_pd(net, test_loader, p_d))
    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": [float(acc) for acc in accuracies],
    }


def load_train_summary() -> dict:
    """Load this arm's training summary (the authoritative checkpoint list).

    Returns:
        Dict mapping run tag -> the training script's recorded metrics.

    Raises:
        FileNotFoundError: If the summary is missing (train the arm first).
    """
    summary_path = TRAIN_LOG_DIR / TRAIN_SUMMARY_FILE
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Training summary not found: {summary_path}. Run "
            f"sn_train_withDelay_v3.py first — this script only evaluates existing "
            f"checkpoints."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_deletion_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only 1st-layer deletion over every checkpoint and save the results.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The results dict written to disk (run tag -> metadata + p_d sweep).
    """
    pd_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print("# v3 Phase 1 (eval-only deletion, 1st hidden layer)")
    print(f"# arm: SGD-delay | dataset: SHD {DATASET_KEY} | grid: {VERSION_TAG} | "
          f"QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | p_d: {pd_values} | "
          f"repeats: {num_repeats}")
    print(f"{'#' * 70}")

    results: dict[str, dict] = {}
    for index, run_tag in enumerate(run_tags, start=1):
        row = train_summary[run_tag]
        print(f"\n[{index}/{len(run_tags)}] {run_tag}")

        net = load_checkpoint(CKPT_DIR / f"{run_tag}.pt")

        pd_sweep = {}
        for p_d in pd_values:
            sweep_result = test_with_repeats(net, test_loader, p_d, num_repeats)
            pd_sweep[str(p_d)] = sweep_result
            print(
                f"  p_d={p_d:<4} acc={sweep_result['mean']:.4f} "
                f"+/- {sweep_result['std']:.4f}"
            )

        results[run_tag] = {
            "use_delay": True,
            # Copied verbatim rather than cast, so integer knobs (``seed``) stay
            # integers and a downstream join on them cannot go wrong.
            **{
                field: row[field]
                for field in SUMMARY_PASSTHROUGH_FIELDS
                if field in row
            },
            "pd_sweep": pd_sweep,
        }

    results_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_deletion_eval"
          f"{RUN_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nDeletion sweep saved to {results_path}")

    print_summary_table(results, pd_values)
    return results


def print_summary_table(results: dict, pd_values: list[float]) -> None:
    """Print one row per checkpoint: both sparsity axes vs clean and thinned accuracy.

    ``retention`` is the chance-corrected fraction of accuracy surviving the
    heaviest deletion; the analysis's control score is ``1 - retention``. It is
    printed here only for eyeballing — the scoring, the regression on ``a`` and ``s``
    and the plotting belong to the analysis step.

    ``a`` and ``s`` are both shown, and ``sp/neu`` (their product) last, because the
    whole point of the v3 grid is that the first two move independently: reading a
    trend off the product alone is exactly the mistake that made v1 uninterpretable
    (document 5 §8 rule 1).

    Args:
        results: The results dict produced by ``run_deletion_sweep``.
        pd_values: The p_d grid that was swept, in order.
    """
    chance = 1.0 / NUM_CLASSES
    pd_min, pd_max = str(pd_values[0]), str(pd_values[-1])

    print(
        f"\n{'run_tag':<40} {'a':>6} {'s':>7} {'sp/neu':>7} {'acc(0)':>8} "
        f"{'acc(' + pd_max + ')':>9} {'retention':>10}"
    )
    for run_tag, row in results.items():
        clean = row["pd_sweep"][pd_min]["mean"]
        worst = row["pd_sweep"][pd_max]["mean"]
        headroom = clean - chance
        retention = (worst - chance) / headroom if headroom > 0 else float("nan")
        print(
            f"{run_tag:<40} {row['spikes_per_active_neuron']:>6.2f} "
            f"{row['silent_fraction']:>7.3f} {row['spikes_per_neuron']:>7.2f} "
            f"{clean:>8.4f} {worst:>9.4f} {retention:>10.3f}"
        )


def main() -> None:
    """Load the shared test set once, then sweep deletion over the with-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_deletion_sweep(test_loader)


if __name__ == "__main__":
    main()
