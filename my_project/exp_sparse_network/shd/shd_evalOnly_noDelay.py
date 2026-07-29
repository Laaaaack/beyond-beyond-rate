"""First milestone (Step 3), NO-DELAY arm: 1st-hidden-layer spike-relocation sweep, eval-only.

Companion to the training script
[sn_train_noDelay.py](../sn_train_noDelay.py), which induced the sparsity grid and
saved one checkpoint per ``(target, strength, seed)``. This script performs **no
training**. It loads each of those checkpoints and measures how much test accuracy
degrades as the **1st hidden layer's** spikes are relocated in time at evaluation
only — the operational definition of "temporal processing" for this milestone.

The with-delay arm is evaluated by the sibling script
[shd_evalOnly_withDelay.py](shd_evalOnly_withDelay.py). The two are kept
separate (rather than one script with a flag) for the same reason the training
scripts are: the network here has no ``delay1``/``delay2`` at all, so any surviving
sparsity -> timing-dependence trend must come from the SRM neurons' own membrane
dynamics rather than from learnable axonal delays.

Protocol rule (do not break): sparsity is applied during *training* on clean data;
the relocation is applied here at *evaluation* only. The loaded checkpoint is never
modified and no gradients are taken.

Partial spike relocation: a fraction ``f`` of each neuron's 1st-hidden spikes are
removed and re-placed at randomly chosen unoccupied bins, leaving the rest
untouched. Per-neuron spike count is preserved exactly, so ``f`` interpolates
between the clean layer (0) and one retaining only each neuron's spike *count* (1)
— destroying spike *timing* while holding *rate* fixed.

v3 correction: destinations are drawn from the layer's *measured* temporal support,
``[0, SUPPORT_BINS)``, not from the whole 200-bin simulation window. See the
``SUPPORT_BINS`` comment below — the uncorrected version scattered most relocated
spikes into the zero-padded tail, which thinned the population's spike density and
so mixed a rate insult into a timing-only probe.

What this script does, for every checkpoint listed in the training summary (the
authoritative live-checkpoint list — read rather than globbed, so we evaluate
exactly the models the milestone locked and reuse their recorded metadata):

- loads the no-delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``F_VALUES`` on the 1st hidden layer, ``NUM_REPEATS`` times per f
  for error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_nodelay_shd_eval.json`` mapping each run tag to its
  ``acc(f)`` sweep, alongside the training summary's ``penalty_strength``,
  ``seed`` and *measured* sparsity metrics so the file is self-contained for Step 4.

The downstream analysis (Step 4) turns each ``acc(f)`` curve into a
chance-corrected, baseline-normalised ``temporal_score`` and plots it against the
*measured* firing rate — never against the penalty strength, whose map to achieved
sparsity is nonlinear and seed-dependent.

Architecture: Input(700) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA), no delays.
Sweep (eval only): f in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
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
# one level up in exp_sparse_network/; only this relocation eval's own output is local.
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / ".." / "sn_data"        # trained checkpoints read from here
TRAIN_LOG_DIR = SCRIPT_DIR / ".." / "sn_log"    # training summary read from here
LOG_DIR = SCRIPT_DIR / "log"                    # relocation eval results written here
SHD_DATA_DIR = SCRIPT_DIR / "shd_data"          # SHD .mat source (local)

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Quick pipeline check. True evaluates only the first MAX_QUICK_CHECKPOINTS models
# on a coarse f grid with a single repeat, and suffixes the output file with
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
DELAY_TAG: str = "nodelay"    # this arm; tags the summary and results files
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# The training summary enumerating this arm's live checkpoints. Its keys are the run
# tags, and ``sn_data/{run_tag}.pt`` is the matching checkpoint.
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
SEED: int = 42                # base seed for the relocation repeats

# --- Destination window for relocated spikes (v3 correction) ---
# shd_whole.mat holds 100 time bins and load_shd_data zero-pads them to the
# simulator's 200, so the 1st hidden layer's spikes never occupy a bin beyond 87
# (measured over all 27 checkpoints by v3_analysis/temporal_support.py). Drawing
# destinations from the full 200 therefore sends ~57% of relocated spikes into a
# region where no hidden spike ever naturally occurs, thinning the population's
# instantaneous spike density by roughly 5x — a *rate* insult riding on a probe
# that is supposed to hold rate fixed and destroy only timing.
#
# Set to None to reproduce v1's full-window behaviour exactly; the two runs write
# to different files (see WINDOW_SUFFIX), so neither can overwrite the other.
SUPPORT_BINS: int | None = 88

# Appended to the results filename so the corrected sweep sits alongside v1's
# full-window results rather than replacing them.
WINDOW_SUFFIX: str = "" if SUPPORT_BINS is None else "_v3window"

# --- Relocation sweep: fraction of each neuron's spikes moved. 0 = clean. ---
# The grid the earlier Beyond Rate realistic-SHD runs used, so this milestone's
# curves can be read beside them. f = 1 keeps only each neuron's spike count.
F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# --- Repeats per f, for error bars (the relocation draw is stochastic). ---
NUM_REPEATS: int = 3


def resolve_eval_config() -> tuple[list[float], int, int | None]:
    """Return (f_values, num_repeats, max_checkpoints) for this run.

    Collapses to a tiny, fast sweep when ``QUICK_TEST`` is set, so the eval
    pipeline can be validated before committing to every checkpoint.

    Returns:
        Tuple of (f_values, num_repeats, max_checkpoints), where
        ``max_checkpoints`` is None for the real run (evaluate all of them).
    """
    if QUICK_TEST:
        return [0.0, 0.4, 1.0], 1, MAX_QUICK_CHECKPOINTS
    return F_VALUES, NUM_REPEATS, None


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
    arms — which is what makes the relocation curves comparable.

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
def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float = 0.0,
) -> torch.Tensor:
    """Vectorised GPU-side partial spike relocation (eval-only helper).

    For each ``(sample, neuron)``, a fraction ``f`` of the existing spikes are
    removed and replaced by the same number of spikes placed at randomly chosen,
    previously-unoccupied time bins. Spike count per neuron is preserved exactly.

    This is the "realistic" SHD perturbation of the Beyond Rate line of work, and it
    differs from jitter in *reach* rather than in kind: jitter displaces every spike
    by a local Gaussian offset, whereas this relocates a chosen fraction of spikes
    to anywhere in the window while leaving the rest untouched. ``f`` therefore
    interpolates between the clean layer (0) and one retaining only each neuron's
    spike *count* (1), which makes it the most direct test of "is this network
    reading timing or rate?" available at a fixed firing rate.

    Destinations are confined to ``[0, SUPPORT_BINS)``, the layer's measured
    temporal support, so that relocation cannot dilute the population's spike
    density by scattering spikes into the zero-padded tail. Per-neuron spike count
    is preserved exactly either way; the support always has room, since a neuron's
    spikes all originate inside it.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).

    Returns:
        Perturbed tensor with the same shape, dtype and device.
    """
    if f <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    # Count spikes per (sample, neuron) and compute how many to move.
    n_spikes = is_spike.sum(dim=-1, keepdim=True)          # (B, C, 1)
    num_to_move = (n_spikes.float() * f).floor().long()    # (B, C, 1)

    # --- 1. Choose which existing spikes to remove ---
    # Random key per time bin; non-spike bins get a large key so they sort last.
    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    # rank[b, c, t] = position of t in the per-(b, c) ascending sort of `key`.
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move                       # (B, C, T)

    keep_mask = is_spike & ~remove_mask

    # --- 2. Place the same number of spikes in currently-unoccupied bins ---
    available = ~keep_mask      # everything except positions we are keeping
    if SUPPORT_BINS is not None:
        available[:, :, SUPPORT_BINS:] = False    # stay inside the measured support
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move    # disjoint from keep_mask by construction

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype)
    return new_spikes.view(B, C, H, W, T)


class SparseSHDNetworkNoDelay(nn.Module):
    """2-hidden-layer SLAYER SNN (no delays) for evaluating the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_train_noDelay.py](../sn_train_noDelay.py) — ``fc1``/``fc2``/``fc3``
    weight-norm parameters and nothing else — so this class loads those checkpoints
    directly. Spikes propagate straight from one dense layer to the next, with no
    ``delay1``/``delay2``, so the only timing machinery is the SRM neurons' own
    membrane dynamics.

    ``forward_with_hidden_perturbation`` relocates 1st hidden layer spikes before
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
        """hidden1 -> fc2 -> spike -> fc3 -> spike."""
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean forward pass (equivalent to the f = 0 baseline)."""
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        f: float = 0.0,
    ) -> torch.Tensor:
        """Eval-only: relocate 1st hidden layer spikes before the readout.

        Must be called inside ``torch.no_grad()`` — ``perturb_hidden_batch`` is not
        autograd-safe. ``f = 0`` reduces to the clean forward pass.

        Args:
            x: Input spike trains.
            f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).

        Returns:
            The output spike tensor.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if f > 0:
            hidden1 = perturb_hidden_batch(hidden1, f)
        return self._second_hidden_and_output(hidden1)


def load_checkpoint(
    checkpoint_path: Path,
    input_dim: int = INPUT_DIM,
) -> SparseSHDNetworkNoDelay:
    """Load a trained checkpoint into a fresh no-delay network in eval mode.

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

    net = SparseSHDNetworkNoDelay(input_dim, HIDDEN_UNITS, NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


@torch.no_grad()
def test_at_f(
    net: SparseSHDNetworkNoDelay,
    test_loader: DataLoader,
    f: float,
) -> float:
    """Return test accuracy with the 1st hidden layer's spikes relocated at ``f``.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        f: Fraction of spikes relocated; 0 is the clean baseline.

    Returns:
        Fraction of correctly classified test samples.
    """
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        outputs = net.forward_with_hidden_perturbation(x_batch, f=f)
        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)
    return correct / max(1, total)


def test_with_repeats(
    net: SparseSHDNetworkNoDelay,
    test_loader: DataLoader,
    f: float,
    num_repeats: int,
) -> dict:
    """Repeat the perturbed evaluation ``num_repeats`` times for error bars.

    Each repeat re-seeds torch (which seeds the CUDA generator the relocation draws
    from) so the perturbation differs run to run but the whole sweep is
    reproducible. ``f = 0`` involves no draw, so its repeats are identical and
    its std must come out at 0 — a useful check that no other randomness leaks into
    evaluation.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).
        num_repeats: Number of repeats.

    Returns:
        Dict with keys ``mean``, ``std``, ``values``.
    """
    accuracies = []
    for repeat in range(num_repeats):
        torch.manual_seed(SEED + repeat)
        np.random.seed(SEED + repeat)
        accuracies.append(test_at_f(net, test_loader, f))
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
            f"Training summary not found: {summary_path}. Run sn_train_noDelay.py "
            f"first — this script only evaluates existing checkpoints."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_perturbation_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only 1st-layer spike relocation over every checkpoint and save the results.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The results dict written to disk (run tag -> metadata + f sweep).
    """
    f_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print("# Sparse-network milestone Step 3 (eval-only spike relocation, 1st hidden layer)")
    print(f"# arm: SGD no-delay | dataset: SHD {DATASET_KEY} | QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | f: {f_values} | "
          f"repeats: {num_repeats}")
    print(f"{'#' * 70}")

    results: dict[str, dict] = {}
    for index, run_tag in enumerate(run_tags, start=1):
        row = train_summary[run_tag]
        print(f"\n[{index}/{len(run_tags)}] {run_tag}")

        net = load_checkpoint(CKPT_DIR / f"{run_tag}.pt")

        f_sweep = {}
        for f_val in f_values:
            sweep_result = test_with_repeats(net, test_loader, f_val, num_repeats)
            f_sweep[str(f_val)] = sweep_result
            print(
                f"  f={f_val:<4} acc={sweep_result['mean']:.4f} "
                f"+/- {sweep_result['std']:.4f}"
            )

        results[run_tag] = {
            "use_delay": False,
            **{
                field: float(row[field])
                for field in SUMMARY_PASSTHROUGH_FIELDS
                if field in row
            },
            "f_sweep": f_sweep,
        }

    results_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_shd_eval{RUN_SUFFIX}{WINDOW_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nRelocation sweep saved to {results_path}")

    print_summary_table(results, f_values)
    return results


def print_summary_table(results: dict, f_values: list[float]) -> None:
    """Print one row per checkpoint: measured sparsity vs clean and perturbed accuracy.

    ``retention`` is the chance-corrected fraction of accuracy surviving the
    strongest relocation; Step 4's ``temporal_score`` is ``1 - retention``. It is
    printed here only for eyeballing (H1 predicts retention falling as
    spikes/neuron falls) — the scoring and plotting belong to the analysis step.

    Args:
        results: The results dict produced by ``run_perturbation_sweep``.
        f_values: The f grid that was swept, in order.
    """
    chance = 1.0 / NUM_CLASSES
    f_min, f_max = str(f_values[0]), str(f_values[-1])

    print(
        f"\n{'run_tag':<44} {'sp/neuron':>10} {'acc(0)':>8} "
        f"{'acc(' + f_max + ')':>9} {'retention':>10}"
    )
    for run_tag, row in results.items():
        clean = row["f_sweep"][f_min]["mean"]
        worst = row["f_sweep"][f_max]["mean"]
        headroom = clean - chance
        retention = (worst - chance) / headroom if headroom > 0 else float("nan")
        print(
            f"{run_tag:<44} {row['spikes_per_neuron']:>10.2f} {clean:>8.4f} "
            f"{worst:>9.4f} {retention:>10.3f}"
        )


def main() -> None:
    """Load the shared test set once, then sweep spike relocation over the no-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_perturbation_sweep(test_loader)


if __name__ == "__main__":
    main()
