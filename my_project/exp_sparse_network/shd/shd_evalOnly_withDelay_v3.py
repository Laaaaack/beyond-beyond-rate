"""v3 Phase 1, WITH-DELAY arm: 1st-hidden-layer spike-relocation sweep, eval-only.

Companion to the training script
[sn_train_withDelay_v3.py](../sn_train_withDelay_v3.py), which trained the v3
**factorial** — a per-pair ceiling ``relu(count - k)`` setting temporal sparsity
``a`` along ``CEILING_K = [1, 2, 4, 8]``, crossed with a membrane-potential floor
setting selectivity ``s`` along ``FLOOR_STRENGTH = [0, 1]``, at seeds 42, 43 and 44
— and saved one checkpoint per ``(k, floor, seed)``. This script performs **no
training**. It loads each of those 24 checkpoints and measures how much test
accuracy degrades as the **1st hidden layer's** spikes are relocated in time at
evaluation only — the *usage* measure of document 5 §3a.

Why the v3 checkpoints rather than v1's: in v1 the two sparsity axes were
confounded at ``rho(a, s) = -0.455`` in this arm and ``-0.943`` in the other, so no
sweep over those models could attribute a trend to one rather than the other. The
v3 grid breaks that by construction — ``rho(a, s) = -0.087`` (p = .69) across the
24 models — which is what turns this sweep from a correlation into a test of H1'.
See document 5 §2b and §4 Phase 1.

The no-delay arm is now trained as well, and its sibling is
[shd_evalOnly_noDelay_v3.py](shd_evalOnly_noDelay_v3.py). The arms are kept as
separate scripts (rather than one script with a flag) for the same reason the
training scripts are: this network carries the learnable axonal delays
``delay1``/``delay2``, which are themselves a timing mechanism, so its relocation
curve measures sparsity's effect *on top of* whatever the delays contribute, where
the sibling's measures it with membrane dynamics as the only timing machinery in the
model.

**This is the primary arm for interpretation**, on two Phase 0 findings the factorial
cannot fix because they concern the dependent variable this sweep produces: its
readout leaves only **+.026** of accuracy unextracted relative to a linear decoder on
its own layer 1 (the no-delay arm: +.300, on every checkpoint), and its deletion
control is essentially orthogonal to selectivity, ``rho(s, control) = +0.030`` against
the no-delay arm's **-0.818**. The no-delay arm carries the larger coefficients; this
one carries the ones that can be attributed.

Relation to [phase1_measure.py](../v3_analysis/phase1_measure.py): that script
measures only the ``f = 1`` endpoint of this sweep, because that single number is
the ``usage`` dependent variable of the Phase 1 regressions. This script sweeps the
full ``f`` grid, which shows the *shape* of the degradation rather than one point
of it. The two use the same support window and the same relocation algorithm, so
the ``f = 1`` column here should reproduce that script's ``acc_relocated``.

Where the delays sit relative to the perturbation site: ``delay1`` is the first
operation *after* the 1st hidden layer's spikes, so the relocation is applied upstream of
it — exactly as in training, where the sparsity penalty also acts on the
pre-``delay1`` spike tensor.

Protocol rule (do not break): sparsity is applied during *training* on clean data;
the relocation is applied here at *evaluation* only. The loaded checkpoint is never
modified, no gradients are taken, and the delays are used exactly as trained (the
adaptive clamping schedule is a training-time device and has no role here).

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

- loads the SGD-delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``F_VALUES`` on the 1st hidden layer, ``NUM_REPEATS`` times per f
  for error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_delay_v3_shd_eval.json`` with two sections:
  ``per_setup``, one **seed-averaged** row per ``(k, floor)`` cell — the
  headline, since the three seeds are replicates of one cell rather than three
  conditions — and ``per_checkpoint``, the raw per-seed ``acc(f)`` sweeps
  it was computed from. Both carry the training summary's design knobs
  (``ceiling_k``, ``floor_strength``, ``seed``) and *measured* sparsity metrics,
  so the file is self-contained for the analysis.

The downstream analysis turns each ``acc(f)`` curve into a chance-corrected,
baseline-normalised ``temporal_score``. Plot it against the *measured* axes ``a``
(``spikes_per_active_neuron``) and ``s`` (``silent_fraction``) separately — never
against ``spikes_per_neuron`` alone, which is their product and moves with both, and
never against the penalty knobs, whose map to achieved sparsity is nonlinear and
seed-dependent (document 5 §8 rule 1). Clean accuracy across this grid is .707-.859,
so raw accuracy drops are not comparable across cells; the normalised score is. Read
the result against the deletion sweep as well: Phase 0 found the general-robustness
control tracking sparsity almost as strongly as the timing probe did.

Architecture: Input(700) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA), with
learnable delays after each hidden layer.
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
DELAY_TAG: str = "delay"      # this arm; tags the summary and results files
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# Training generation to evaluate. "v3" selects the 24-model factorial grid trained
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

# Per-checkpoint fields averaged across seeds in the per-setup summary. The seeds of
# one ``(k, floor)`` cell are replicates of the same factorial cell, so these are the
# cell's achieved values; their across-seed spread is reported alongside, because a
# mean that hides its spread is how seed noise gets read as structure.
AVERAGED_SUMMARY_FIELDS: tuple[str, ...] = (
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
SEED: int = 42                # base seed for the relocation repeats
MAX_DELAY: int = 64           # recorded for parity with training; unused at eval

# --- Destination window for relocated spikes (v3 correction) ---
# shd_whole.mat holds 100 time bins and load_shd_data zero-pads them to the
# simulator's 200, so the 1st hidden layer's spikes never occupy a bin beyond 87
# (measured over all 27 checkpoints by v3_analysis/temporal_support.py). Drawing
# destinations from the full 200 therefore sends ~57% of relocated spikes into a
# region where no hidden spike ever naturally occurs, thinning the population's
# instantaneous spike density by roughly 5x — a *rate* insult riding on a probe
# that is supposed to hold rate fixed and destroy only timing.
#
# Set to None to reproduce the uncorrected full-window behaviour exactly; the two
# runs write to different files (see WINDOW_SUFFIX), so neither can overwrite the
# other.
SUPPORT_BINS: int | None = 88

# Appended to the results filename. The corrected window is what a v3 run means, so
# it takes the bare name and the full-window variant is the one that gets marked.
WINDOW_SUFFIX: str = "_fullwindow" if SUPPORT_BINS is None else ""

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


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with learnable delays, for the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_train_withDelay_v3.py](../sn_train_withDelay_v3.py) — ``fc1``/``fc2``/``fc3``
    weight-norm parameters plus ``delay1``/``delay2`` — so this class loads those
    checkpoints directly. The training script's adaptive delay-clamping schedule is
    deliberately absent: it shapes delays *during* training, and the loaded values
    are used here exactly as saved.

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
def test_at_f(
    net: SparseSHDNetwork,
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
    net: SparseSHDNetwork,
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
            f"Training summary not found: {summary_path}. Run "
            f"sn_train_withDelay_v3.py first — this script only evaluates existing "
            f"checkpoints."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_perturbation_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only 1st-layer spike relocation over every checkpoint and save the results.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The payload written to disk: ``per_setup`` (one seed-averaged row per
        ``(k, floor)`` cell) and ``per_checkpoint`` (the raw per-seed rows).
    """
    f_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print("# v3 Phase 1 (eval-only spike relocation, 1st hidden layer)")
    print(f"# arm: SGD-delay | dataset: SHD {DATASET_KEY} | grid: {VERSION_TAG} | "
          f"QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | f: {f_values} | "
          f"repeats: {num_repeats} | window: "
          f"{'full' if SUPPORT_BINS is None else f'[0,{SUPPORT_BINS})'}")
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
            "use_delay": True,
            # Copied verbatim rather than cast, so integer knobs (``seed``) stay
            # integers and a downstream join on them cannot go wrong.
            **{
                field: row[field]
                for field in SUMMARY_PASSTHROUGH_FIELDS
                if field in row
            },
            "f_sweep": f_sweep,
        }

    # The per-setup average is the headline; the per-checkpoint rows it was
    # computed from are kept beside it, because they are the raw measurement and
    # because the across-seed spread cannot be recovered from a mean alone.
    aggregated = aggregate_over_seeds(results, "f_sweep", f_values)
    payload = {"per_setup": aggregated, "per_checkpoint": results}

    results_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_shd_eval"
          f"{RUN_SUFFIX}{WINDOW_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nRelocation sweep saved to {results_path} "
          f"({len(results)} checkpoints -> {len(aggregated)} setups)")

    print_summary_table(aggregated, f_values)
    return payload


def aggregate_over_seeds(results: dict, sweep_field: str,
                         grid_values: list) -> dict:
    """Collapse the per-seed checkpoints into one row per ``(k, floor)`` setup.

    The seeds are replicates of the same factorial cell, so the *cell* is the unit
    the design is about, and a per-seed table invites reading noise as structure —
    the three seeds of one cell differ by up to .08 in clean accuracy here.

    Two quantities are averaged, and the two averages are **not** interchangeable:

    - ``acc(f)`` at each grid point is averaged directly across seeds, which is
      what a curve should show;
    - ``retention`` is computed **per seed, against that seed's own clean accuracy**,
      and only then averaged. The score exists to normalise away each model's
      differing headroom above chance, so deriving it from an already-averaged curve
      divided by an averaged baseline would put exactly that variation back in.

    Args:
        results: Per-checkpoint results, run tag -> metadata + sweep.
        sweep_field: Key holding the sweep dict in each per-checkpoint row.
        grid_values: The swept grid, in order. Its two ends define ``retention``.

    Returns:
        Dict keyed ``k{k}_floor{f}`` -> the cell's averaged metrics, ordered by
        ``(ceiling_k, floor_strength)``. Every averaged field is accompanied by its
        across-seed standard deviation and by the per-seed values it came from.
    """
    chance = 1.0 / NUM_CLASSES
    grid_keys = [str(value) for value in grid_values]

    cells: dict[str, list[dict]] = {}
    for row in results.values():
        setup_tag = f"k{row['ceiling_k']:g}_floor{row['floor_strength']:g}"
        cells.setdefault(setup_tag, []).append(row)

    aggregated: dict[str, dict] = {}
    for setup_tag in sorted(cells, key=lambda tag: (cells[tag][0]["ceiling_k"],
                                                    cells[tag][0]["floor_strength"])):
        rows = sorted(cells[setup_tag], key=lambda row: row["seed"])

        sweep = {}
        for key in grid_keys:
            per_seed = [row[sweep_field][key]["mean"] for row in rows]
            sweep[key] = {
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed)),
                "values": [float(value) for value in per_seed],
            }

        retentions = []
        for row in rows:
            clean = row[sweep_field][grid_keys[0]]["mean"]
            worst = row[sweep_field][grid_keys[-1]]["mean"]
            headroom = clean - chance
            retentions.append((worst - chance) / headroom
                              if headroom > 0 else float("nan"))

        averaged = {}
        for field in AVERAGED_SUMMARY_FIELDS:
            if field not in rows[0]:
                continue
            per_seed = [row[field] for row in rows]
            averaged[field] = float(np.mean(per_seed))
            averaged[f"{field}_std"] = float(np.std(per_seed))

        aggregated[setup_tag] = {
            "ceiling_k": rows[0]["ceiling_k"],
            "floor_strength": rows[0]["floor_strength"],
            "n_seeds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "retention": float(np.mean(retentions)),
            "retention_std": float(np.std(retentions)),
            "retention_per_seed": [float(value) for value in retentions],
            **averaged,
            sweep_field: sweep,
        }
    return aggregated


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Render one averaged quantity as ``mean+-std`` for the printed table."""
    return f"{mean:.{decimals}f}+-{std:.{decimals}f}"


def print_summary_table(aggregated: dict, f_values: list[float]) -> None:
    """Print one row per ``(k, floor)`` setup, averaged over its seeds.

    ``retention`` is the chance-corrected fraction of accuracy surviving the
    strongest relocation; the analysis's ``usage`` score is ``1 - retention``. It is
    printed here only for eyeballing — the scoring, the regression on ``a`` and ``s``
    and the plotting belong to the analysis step.

    ``a`` and ``s`` are both shown, and ``sp/neu`` (their product) last, because the
    whole point of the v3 grid is that the first two move independently: reading a
    trend off the product alone is exactly the mistake that made v1 uninterpretable
    (document 5 §8 rule 1). Every column carries its across-seed spread, so a cell
    whose seeds disagree cannot be mistaken for a tight one.

    Args:
        aggregated: The per-setup dict produced by ``aggregate_over_seeds``.
        f_values: The f grid that was swept, in order.
    """
    f_min, f_max = str(f_values[0]), str(f_values[-1])

    print(
        f"\n{'setup':<14} {'n':>2} {'a':>13} {'s':>13} {'sp/neu':>7} "
        f"{'acc(' + f_min + ')':>15} {'acc(' + f_max + ')':>15} "
        f"{'retention':>15}"
    )
    for setup_tag, row in aggregated.items():
        sweep = row["f_sweep"]
        print(
            f"{setup_tag:<14} {row['n_seeds']:>2} "
            f"{format_mean_std(row['spikes_per_active_neuron'], row['spikes_per_active_neuron_std'], 2):>13} "
            f"{format_mean_std(row['silent_fraction'], row['silent_fraction_std'], 3):>13} "
            f"{row['spikes_per_neuron']:>7.2f} "
            f"{format_mean_std(sweep[f_min]['mean'], sweep[f_min]['std'], 4):>15} "
            f"{format_mean_std(sweep[f_max]['mean'], sweep[f_max]['std'], 4):>15} "
            f"{format_mean_std(row['retention'], row['retention_std'], 3):>15}"
        )
    print("\n  Means over seeds; +- is the across-seed standard deviation. "
          "Per-seed rows are kept in the results file under 'per_checkpoint'.")


def main() -> None:
    """Load the shared test set once, then sweep spike relocation over the with-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_perturbation_sweep(test_loader)


if __name__ == "__main__":
    main()
