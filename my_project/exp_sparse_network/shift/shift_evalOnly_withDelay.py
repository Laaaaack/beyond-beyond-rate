"""First milestone (Step 3), WITH-DELAY arm: 1st-hidden-layer shift sweep, eval-only.

Companion to the training script
[sn_train_withDelay.py](../sn_train_withDelay.py), which induced the sparsity grid
and saved one checkpoint per ``(target, strength, seed)``. This script performs
**no training**. It loads each of those checkpoints and measures how much test
accuracy degrades as the **1st hidden layer's** spikes are shifted in time at
evaluation only — the operational definition of "temporal processing" for this
milestone.

The no-delay arm is evaluated by the sibling script
[shift_evalOnly_noDelay.py](shift_evalOnly_noDelay.py). The two are kept separate
(rather than one script with a flag) for the same reason the training scripts are:
this network carries the learnable axonal delays ``delay1``/``delay2``, which are
themselves a timing mechanism, so its shift curve measures sparsity's effect *on
top of* whatever the delays contribute. The gap between the two arms' temporal
scores is how much of the temporal processing the delays were doing.

Where the delays sit relative to the perturbation site: ``delay1`` is the first
operation *after* the 1st hidden layer's spikes, so shift is injected upstream of
it — exactly as in training, where the sparsity penalty also acts on the
pre-``delay1`` spike tensor.

Protocol rule (do not break): sparsity is applied during *training* on clean data;
shift is applied here at *evaluation* only. The loaded checkpoint is never
modified, no gradients are taken, and the delays are used exactly as trained (the
adaptive clamping schedule is a training-time device and has no role here).

Per-neuron shift: one offset is drawn from ``N(0, sigma)`` per (sample, neuron) and
*all* of that neuron's spikes move together, clipped to ``[0, T-1]``. Unlike shift,
each neuron's internal spike pattern survives intact — only its alignment to the
other neurons and to stimulus onset is destroyed. Spike count is preserved except
where end-of-window clipping merges spikes, so this too is a timing perturbation
rather than a rate one.

What this script does, for every checkpoint listed in the training summary (the
authoritative live-checkpoint list — read rather than globbed, so we evaluate
exactly the models the milestone locked and reuse their recorded metadata):

- loads the SGD-delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``SIGMA_VALUES`` on the 1st hidden layer, ``NUM_REPEATS`` times per sigma
  for error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_delay_shift_eval.json`` mapping each run tag to its
  ``acc(sigma)`` sweep, alongside the training summary's ``penalty_strength``,
  ``seed`` and *measured* sparsity metrics so the file is self-contained for Step 4.

The downstream analysis (Step 4) turns each ``acc(sigma)`` curve into a
chance-corrected, baseline-normalised ``temporal_score`` and plots it against the
*measured* firing rate — never against the penalty strength, whose map to achieved
sparsity is nonlinear and seed-dependent. The two arms must be compared on that
normalised score rather than on raw accuracy drops, since this arm's clean accuracy
is far higher (78-89% vs the no-delay arm's 49-59%).

Architecture: Input(700) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA), with
learnable delays after each hidden layer.
Sweep (eval only): sigma in {0, 1, 3, 5, 10, 17, 25} time steps (ms).

v3 note — this sweep needs no support-window correction, unlike relocation and
jitter. A shift is a *rigid translation* of one neuron's whole spike train, so it
preserves the population's instantaneous spike density exactly and never dilutes it
into the zero-padded tail; the rate insult that the other two probes acquire from
the padding simply does not arise here. Clipping targets to the measured support
``[0, 88)`` would instead pile spikes up at the support edge and merge them, which
*would* destroy per-neuron count — so the full-window clip below is the correct
choice for this perturbation, not an oversight. See
``v3_analysis/temporal_support.py`` and document 5 §2a.
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
# one level up in exp_sparse_network/; only this shift eval's own output is local.
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / ".." / "sn_data"        # trained checkpoints read from here
TRAIN_LOG_DIR = SCRIPT_DIR / ".." / "sn_log"    # training summary read from here
LOG_DIR = SCRIPT_DIR / "log"                    # shift eval results written here
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
# on a coarse sigma grid with a single repeat, and suffixes the output file with
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

# The training summary enumerating this arm's live checkpoints. Its keys are the run
# tags, and ``sn_data/{run_tag}.pt`` is the matching checkpoint. Note this is the
# re-tuned (2026-07-27) 12-model summary; the superseded, re-densified first run is
# kept beside it as ..._train_summary_redensified_1250ep.json and is NOT evaluated.
TRAIN_SUMMARY_FILE: str = f"sparse_{DATASET_KEY}_{DELAY_TAG}_train_summary.json"

# Training-summary fields carried through into the results file, so Step 4 can plot
# temporal score against measured sparsity without re-joining the summary.
SUMMARY_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "target_rate",
    "penalty_strength",
    "seed",
    "clean_acc",
    "firing_rate",
    "spikes_per_neuron",
    "silent_fraction",
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
SEED: int = 42                # base seed for the shift repeats
MAX_DELAY: int = 64           # recorded for parity with training; unused at eval

# --- Shift sweep: sigma in time steps (ms). 0 = clean baseline. ---
# The same grid the jitter sweep uses, so the two timing perturbations share one
# x-axis; it also matches the earlier fixed-weight perturbation results.
SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]

# --- Repeats per sigma, for error bars (the shift draw is stochastic). ---
NUM_REPEATS: int = 3


def resolve_eval_config() -> tuple[list[int], int, int | None]:
    """Return (sigma_values, num_repeats, max_checkpoints) for this run.

    Collapses to a tiny, fast sweep when ``QUICK_TEST`` is set, so the eval
    pipeline can be validated before committing to every checkpoint.

    Returns:
        Tuple of (sigma_values, num_repeats, max_checkpoints), where
        ``max_checkpoints`` is None for the real run (evaluate all of them).
    """
    if QUICK_TEST:
        return [0, 5, 25], 1, MAX_QUICK_CHECKPOINTS
    return SIGMA_VALUES, NUM_REPEATS, None


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
    arms — which is what makes the shift curves comparable.

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
def shift_hidden_batch(
    hidden_spikes: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Vectorised GPU-side per-neuron Gaussian shift (eval-only helper).

    For each ``(sample, neuron)`` a single integer offset ``round(N(0, sigma))`` is
    drawn and every spike of that neuron is moved by it, then clipped to
    ``[0, T - 1]``. Spikes that collide after clipping are merged (logical OR).

    This is the coarser sibling of the shift perturbation. Shift moves every spike
    independently and so destroys a neuron's *internal* spike pattern; a shift
    translates that pattern intact and destroys only its alignment relative to the
    other neurons and to stimulus onset. A network reading absolute or cross-neuron
    spike timing loses accuracy here; one reading each neuron's own inter-spike
    structure, or just its rate, does not.

    Spike count is preserved except where end-of-window clipping merges spikes,
    which is why the shift grid stays small relative to ``T = 200``.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        sigma: Per-neuron shift std dev in time steps (ms). 0 means no shift.

    Returns:
        Shifted tensor with the same shape, dtype and device.
    """
    if sigma <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = (x > 0.5).to(x.dtype)

    # One integer offset per (sample, neuron); broadcast over the time axis.
    offset = (torch.randn(B, C, 1, device=x.device) * sigma).round().long()
    t_idx = torch.arange(T, device=x.device).view(1, 1, T)
    target = (t_idx + offset).clamp(0, T - 1).expand(B, C, T).contiguous()

    # Scatter each source bin's spike onto its shifted target; summing then
    # thresholding merges colliding spikes as a logical OR.
    new_x = torch.zeros_like(x)
    new_x.scatter_add_(-1, target, is_spike)
    new_x = (new_x > 0.5).to(hidden_spikes.dtype)

    return new_x.view(B, C, H, W, T)


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with learnable delays, for the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_train_withDelay.py](../sn_train_withDelay.py) — ``fc1``/``fc2``/``fc3``
    weight-norm parameters plus ``delay1``/``delay2`` — so this class loads those
    checkpoints directly. The training script's adaptive delay-clamping schedule is
    deliberately absent: it shapes delays *during* training, and the loaded values
    are used here exactly as saved.

    ``forward_with_hidden_perturbation`` shifts the 1st hidden layer before
    ``delay1`` and the readout. It is eval-only; nothing here is ever trained.
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
        """Clean forward pass (equivalent to the sigma = 0 baseline)."""
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Eval-only: shift the 1st hidden layer's spikes before the readout.

        Must be called inside ``torch.no_grad()`` — ``shift_hidden_batch`` is not
        autograd-safe. ``sigma = 0`` reduces to the clean forward pass.

        Args:
            x: Input spike trains.
            sigma: Shift std dev in time steps (ms).

        Returns:
            The output spike tensor.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if sigma > 0:
            hidden1 = shift_hidden_batch(hidden1, sigma)
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
def test_at_sigma(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    sigma: float,
) -> float:
    """Return test accuracy with the 1st hidden layer shifted at ``sigma``.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        sigma: Shift std dev in time steps (ms); 0 is the clean baseline.

    Returns:
        Fraction of correctly classified test samples.
    """
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        outputs = net.forward_with_hidden_perturbation(x_batch, sigma=sigma)
        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)
    return correct / max(1, total)


def test_with_repeats(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    sigma: float,
    num_repeats: int,
) -> dict:
    """Repeat the shifted evaluation ``num_repeats`` times for error bars.

    Each repeat re-seeds torch (which seeds the CUDA generator the shift draws
    from) so the perturbation differs run to run but the whole sweep is
    reproducible. ``sigma = 0`` involves no draw, so its repeats are identical and
    its std must come out at 0 — a useful check that no other randomness leaks into
    evaluation.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        sigma: Shift std dev in time steps (ms).
        num_repeats: Number of repeats.

    Returns:
        Dict with keys ``mean``, ``std``, ``values``.
    """
    accuracies = []
    for repeat in range(num_repeats):
        torch.manual_seed(SEED + repeat)
        np.random.seed(SEED + repeat)
        accuracies.append(test_at_sigma(net, test_loader, sigma))
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
            f"Training summary not found: {summary_path}. Run sn_train_withDelay.py "
            f"first — this script only evaluates existing checkpoints."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_shift_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only 1st-layer shift over every checkpoint and save the results.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The results dict written to disk (run tag -> metadata + sigma sweep).
    """
    sigma_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print("# Sparse-network milestone Step 3 (eval-only shift, 1st hidden layer)")
    print(f"# arm: SGD-delay | dataset: SHD {DATASET_KEY} | QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | sigmas: {sigma_values} | "
          f"repeats: {num_repeats}")
    print(f"{'#' * 70}")

    results: dict[str, dict] = {}
    for index, run_tag in enumerate(run_tags, start=1):
        row = train_summary[run_tag]
        print(f"\n[{index}/{len(run_tags)}] {run_tag}")

        net = load_checkpoint(CKPT_DIR / f"{run_tag}.pt")

        sigma_sweep = {}
        for sigma in sigma_values:
            sweep_result = test_with_repeats(net, test_loader, sigma, num_repeats)
            sigma_sweep[str(sigma)] = sweep_result
            print(
                f"  sigma={sigma:<3} acc={sweep_result['mean']:.4f} "
                f"+/- {sweep_result['std']:.4f}"
            )

        results[run_tag] = {
            "use_delay": True,
            **{
                field: float(row[field])
                for field in SUMMARY_PASSTHROUGH_FIELDS
                if field in row
            },
            "sigma_sweep": sigma_sweep,
        }

    results_path = (
        LOG_DIR / f"sparse_{DATASET_KEY}_{DELAY_TAG}_shift_eval{RUN_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nShift sweep saved to {results_path}")

    print_summary_table(results, sigma_values)
    return results


def print_summary_table(results: dict, sigma_values: list[int]) -> None:
    """Print one row per checkpoint: measured sparsity vs clean and shifted accuracy.

    ``retention`` is the chance-corrected fraction of accuracy surviving the
    strongest shift; Step 4's ``temporal_score`` is ``1 - retention``. It is
    printed here only for eyeballing (H1 predicts retention falling as
    spikes/neuron falls) — the scoring and plotting belong to the analysis step.

    Args:
        results: The results dict produced by ``run_shift_sweep``.
        sigma_values: The sigma grid that was swept, in order.
    """
    chance = 1.0 / NUM_CLASSES
    sigma_min, sigma_max = str(sigma_values[0]), str(sigma_values[-1])

    print(
        f"\n{'run_tag':<44} {'sp/neuron':>10} {'acc(0)':>8} "
        f"{'acc(' + sigma_max + ')':>9} {'retention':>10}"
    )
    for run_tag, row in results.items():
        clean = row["sigma_sweep"][sigma_min]["mean"]
        worst = row["sigma_sweep"][sigma_max]["mean"]
        headroom = clean - chance
        retention = (worst - chance) / headroom if headroom > 0 else float("nan")
        print(
            f"{run_tag:<44} {row['spikes_per_neuron']:>10.2f} {clean:>8.4f} "
            f"{worst:>9.4f} {retention:>10.3f}"
        )


def main() -> None:
    """Load the shared test set once, then sweep shift over the with-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_shift_sweep(test_loader)


if __name__ == "__main__":
    main()
