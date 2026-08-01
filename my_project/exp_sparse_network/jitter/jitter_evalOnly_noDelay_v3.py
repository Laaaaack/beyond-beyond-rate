"""v3 Phase 1, NO-DELAY arm: 1st-hidden-layer jitter sweep, eval-only.

Companion to the training script
[sn_train_noDelay_v3.py](../sn_train_noDelay_v3.py), which trained the v3
**factorial** — a per-pair ceiling ``relu(count - k)`` setting temporal sparsity
``a`` along ``CEILING_K = [1, 2, 4, 8]``, crossed with a membrane-potential floor
setting selectivity ``s`` along ``FLOOR_STRENGTH = [0, 1]``, at seeds 42, 43 and 44
— and saved one checkpoint per ``(k, floor, seed)``. This script performs **no
training**. It loads each of those 24 checkpoints and measures how much test
accuracy degrades as the **1st hidden layer's** spikes are jittered in time at
evaluation only.

Why the v3 checkpoints rather than v1's: in v1 this arm's two sparsity axes were
confounded at ``rho(a, s) = -0.943`` — the worst figure in the project — so no sweep
over those models could attribute a trend to one rather than the other. The v3 grid
breaks that by construction: ``rho(a, s) = -0.342`` (p = .10) across the 24 models,
which is what makes a jitter curve read against ``a`` interpretable. See document 5
§2b and §4 Phase 1.

Which arm this is, and what it is for. The sibling is
[jitter_evalOnly_withDelay_v3.py](jitter_evalOnly_withDelay_v3.py). The arms are kept
as separate scripts (rather than one script with a flag) for the same reason the
training scripts are: that one carries the learnable axonal delays
``delay1``/``delay2``, which are themselves a timing mechanism, so its curve measures
sparsity's effect *on top of* whatever the delays contribute. **Here there are no
delays at all**, so any timing effect this sweep finds is attributable to the SRM
neurons' membrane dynamics alone — which is what makes this arm worth running as a
replication rather than a duplicate.

Two caveats specific to this arm, both from Phase 0 and neither fixable by the
factorial, because they concern the *dependent* variable rather than the independent
ones:

- this arm's readout leaves **+.300** of accuracy unextracted relative to a linear
  decoder on its own 1st hidden layer, on **every** checkpoint (delay arm: +.026), so
  a ``temporal_score`` here is as much a statement about ``fc2``/``fc3`` as about
  layer 1;
- its deletion control is strongly tied to selectivity, ``rho(s, control) = -0.818``
  against the delay arm's +0.030, so a trend attributed to ``a`` here is on much
  firmer ground than one attributed to ``s``.

Relation to [phase1_measure.py](../v3_analysis/phase1_measure.py): that script
measures only the *endpoints* the Phase 1 regressions need (relocation ``f = 1``
and the ``p_d = 0.8`` deletion control) on these same checkpoints. This script
sweeps the full jitter grid, which the regressions do not use but which shows the
*shape* of the degradation rather than a single number.

Where the perturbation site sits: the 1st hidden layer's spikes feed ``fc2``
directly, with no delay line in between, so jitter is injected exactly where the
training penalties acted.

Protocol rule (do not break): sparsity is applied during *training* on clean data;
jitter is applied here at *evaluation* only. The loaded checkpoint is never
modified and no gradients are taken.

Per-spike jitter: each 1st-hidden spike is independently shifted by an offset drawn
from ``N(0, sigma)``, clipped to the layer's measured temporal support
``[0, SUPPORT_BINS)`` and placed at the nearest free bin. Per-neuron spike count is
preserved, so the perturbation destroys spike *timing*, not *rate* — a network that
leans on timing loses accuracy; a pure rate coder does not.

v3 correction: the clip is to the layer's measured support rather than to ``T - 1``.
The zero-padded tail holds no hidden spikes, so jittering into it thins the
population's spike density and mixes a rate insult into a timing-only probe.

What this script does, for every checkpoint listed in the training summary (the
authoritative live-checkpoint list — read rather than globbed, so we evaluate
exactly the models the milestone locked and reuse their recorded metadata):

- loads the no-delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``SIGMA_VALUES`` on the 1st hidden layer, ``NUM_REPEATS`` times per sigma
  for error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_nodelay_v3_jitter_eval.json`` with two sections:
  ``per_setup``, one **seed-averaged** row per ``(k, floor)`` cell — the headline,
  since the three seeds are replicates of one cell rather than three conditions — and
  ``per_checkpoint``, the raw per-seed ``acc(sigma)`` sweeps it was computed from.
  Both carry the training summary's design knobs (``ceiling_k``, ``floor_strength``,
  ``seed``) and *measured* sparsity metrics, so the file is self-contained for the
  analysis.

The downstream analysis turns each ``acc(sigma)`` curve into a chance-corrected,
baseline-normalised ``temporal_score``. Plot it against the *measured* axes ``a``
(``spikes_per_active_neuron``) and ``s`` (``silent_fraction``) separately — never
against ``spikes_per_neuron`` alone, which is their product and moves with both, and
never against the penalty knobs, whose map to achieved sparsity is nonlinear and
seed-dependent (document 5 §8 rule 1). Clean accuracy across this grid is .465-.550,
so raw accuracy drops are not comparable across cells; the normalised score is.

Architecture: Input(700) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA), with
no learnable delays anywhere.
Sweep (eval only): sigma in {0, 1, 3, 5, 10, 17, 25} time steps (ms).
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
# one level up in exp_sparse_network/; only this jitter eval's own output is local.
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / ".." / "sn_data"        # trained checkpoints read from here
TRAIN_LOG_DIR = SCRIPT_DIR / ".." / "sn_log"    # training summary read from here
LOG_DIR = SCRIPT_DIR / "log"                    # jitter eval results written here
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
DELAY_TAG: str = "nodelay"    # this arm; tags the summary and results files
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# Training generation to evaluate. "v3" selects the 24-model factorial grid trained
# by sn_train_noDelay_v3.py; it matches that script's own VERSION_TAG, and it tags
# both the summary read here and the results written below, so v3 sweeps can never
# collide with v1's files in log/.
VERSION_TAG: str = "v3"

# The training summary enumerating this arm's live v3 checkpoints. Its keys are the
# run tags (``sparse_whole_nodelay_v3_k{k}_floor{F}_seed{seed}``) and
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
SEED: int = 42                # base seed for the jitter repeats

# --- Jitter destination window (v3 correction) ---
# shd_whole.mat holds 100 time bins and load_shd_data zero-pads them to the
# simulator's 200, so the 1st hidden layer's spikes never occupy a bin beyond 87
# (measured over all 27 checkpoints by v3_analysis/temporal_support.py). Clipping
# jitter targets to T - 1 = 199 therefore lets the largest sigmas push spikes into
# a region where no hidden spike ever naturally occurs, which thins the
# population's instantaneous spike density — a *rate* insult riding on a probe
# that is supposed to hold rate fixed and destroy only timing.
#
# Clipping to the support instead concentrates the overflow at the support edge.
# That is the lesser distortion: the collision retry below preserves each neuron's
# spike count exactly either way, so the edge pile-up costs alignment, not rate.
#
# Set to None to reproduce the uncorrected full-window behaviour exactly; the two
# runs write to different files (see WINDOW_SUFFIX), so neither can overwrite the
# other.
SUPPORT_BINS: int | None = None

# Appended to the results filename. The corrected window is what a v3 run means, so
# it takes the bare name and the full-window variant is the one that gets marked.
WINDOW_SUFFIX: str = "_fullwindow" if SUPPORT_BINS is None else ""

# --- Jitter sweep: sigma in time steps (ms). 0 = clean baseline. ---
# Copied unchanged from the existing jitter scripts so this milestone's curves are
# comparable to the earlier fixed-weight perturbation results.
SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]

# --- Repeats per sigma, for error bars (the jitter draw is stochastic). ---
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
    arms — which is what makes the jitter curves comparable.

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
def jitter_hidden_batch(
    hidden_spikes: torch.Tensor,
    sigma: float,
    max_attempts: int = 50,
) -> torch.Tensor:
    """Vectorised GPU-side per-spike Gaussian jitter (eval-only helper).

    For each spike, draw an iid Gaussian offset ``~ N(0, sigma)``, shift the spike
    by ``round(offset)`` and clip to ``[0, SUPPORT_BINS)``, the layer's measured
    temporal support. Two spikes landing in the same bin are resolved by a
    random-priority tiebreaker; the loser is retried with a
    fresh offset for up to ``max_attempts`` outer iterations, and any spike still
    unplaced falls back to its original bin. Per-neuron spike count is therefore
    preserved and only timing is destroyed.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        sigma: Jitter std dev in time steps (ms). 0 means no jitter.
        max_attempts: Outer retry budget per spike before fallback.

    Returns:
        Jittered tensor with the same shape, dtype and device.
    """
    if sigma <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    new_spikes = torch.zeros_like(is_spike)
    unplaced = is_spike.clone()

    t_idx = torch.arange(T, device=x.device).view(1, 1, T)
    inf_tensor = torch.full_like(x, float("inf"))

    for _ in range(max_attempts):
        if not unplaced.any():
            break

        offsets = torch.randn_like(x) * sigma
        upper_bin = (T if SUPPORT_BINS is None else SUPPORT_BINS) - 1
        target = (t_idx + offsets).round().long().clamp(0, upper_bin)

        priority = torch.where(unplaced, torch.rand_like(x), inf_tensor)
        min_priority = inf_tensor.clone()
        min_priority.scatter_reduce_(
            -1, target, priority, reduce="amin", include_self=True,
        )

        target_min = min_priority.gather(-1, target)
        target_free = ~new_spikes.gather(-1, target)
        wins = unplaced & (priority == target_min) & target_free

        scatter_out = torch.zeros((B, C, T), device=x.device, dtype=torch.uint8)
        scatter_out.scatter_add_(-1, target, wins.to(torch.uint8))
        new_spikes = new_spikes | (scatter_out > 0)

        unplaced = unplaced & ~wins

    new_spikes = new_spikes | unplaced

    return new_spikes.to(hidden_spikes.dtype).view(B, C, H, W, T)


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN without delays, for the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_train_noDelay_v3.py](../sn_train_noDelay_v3.py) — ``fc1``/``fc2``/``fc3``
    weight-norm parameters and nothing else — so this class loads those checkpoints
    directly.

    Spikes propagate straight from one dense layer to the next, so the only timing
    machinery in the whole model is the SRM neurons' own membrane dynamics. That is
    what this arm is for: whatever survives the sweep cannot be credited to a
    learnable delay line, because there is none.

    ``forward_with_hidden_perturbation`` jitters the 1st hidden layer before the
    readout. It is eval-only; nothing here is ever trained.
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
        """hidden1 -> fc2 -> spike -> fc3 -> spike (no delays anywhere)."""
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

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
        """Eval-only: jitter the 1st hidden layer's spikes before the readout.

        Must be called inside ``torch.no_grad()`` — ``jitter_hidden_batch`` is not
        autograd-safe. ``sigma = 0`` reduces to the clean forward pass.

        Args:
            x: Input spike trains.
            sigma: Jitter std dev in time steps (ms).

        Returns:
            The output spike tensor.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if sigma > 0:
            hidden1 = jitter_hidden_batch(hidden1, sigma)
        return self._second_hidden_and_output(hidden1)


def load_checkpoint(
    checkpoint_path: Path,
    input_dim: int = INPUT_DIM,
) -> SparseSHDNetwork:
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
    """Return test accuracy with the 1st hidden layer jittered at ``sigma``.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        sigma: Jitter std dev in time steps (ms); 0 is the clean baseline.

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
    """Repeat the jittered evaluation ``num_repeats`` times for error bars.

    Each repeat re-seeds torch (which seeds the CUDA generator the jitter draws
    from) so the perturbation differs run to run but the whole sweep is
    reproducible. ``sigma = 0`` involves no draw, so its repeats are identical and
    its std must come out at 0 — a useful check that no other randomness leaks into
    evaluation.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        sigma: Jitter std dev in time steps (ms).
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
            f"Training summary not found: {summary_path}. Run "
            f"sn_train_noDelay_v3.py first — this script only evaluates existing "
            f"checkpoints."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_jitter_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only 1st-layer jitter over every checkpoint and save the results.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The payload written to disk: ``per_setup`` (one seed-averaged row per
        ``(k, floor)`` cell) and ``per_checkpoint`` (the raw per-seed rows).
    """
    sigma_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print("# v3 Phase 1 (eval-only jitter, 1st hidden layer)")
    print(f"# arm: no-delay | dataset: SHD {DATASET_KEY} | grid: {VERSION_TAG} | "
          f"QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | sigmas: {sigma_values} | "
          f"repeats: {num_repeats} | window: "
          f"{'full' if SUPPORT_BINS is None else f'[0,{SUPPORT_BINS})'}")
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
            "use_delay": False,
            # Copied verbatim rather than cast, so integer knobs (``seed``) stay
            # integers and a downstream join on them cannot go wrong.
            **{
                field: row[field]
                for field in SUMMARY_PASSTHROUGH_FIELDS
                if field in row
            },
            "sigma_sweep": sigma_sweep,
        }

    # The per-setup average is the headline; the per-checkpoint rows it was computed
    # from are kept beside it, because they are the raw measurement and because the
    # across-seed spread cannot be recovered from a mean alone.
    aggregated = aggregate_over_seeds(results, "sigma_sweep", sigma_values)
    payload = {"per_setup": aggregated, "per_checkpoint": results}

    results_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_jitter_eval"
          f"{RUN_SUFFIX}{WINDOW_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nJitter sweep saved to {results_path} "
          f"({len(results)} checkpoints -> {len(aggregated)} setups)")

    print_summary_table(aggregated, sigma_values)
    return payload


def aggregate_over_seeds(results: dict, sweep_field: str,
                         grid_values: list) -> dict:
    """Collapse the per-seed checkpoints into one row per ``(k, floor)`` setup.

    The seeds are replicates of the same factorial cell, so the *cell* is the unit
    the design is about, and a per-seed table invites reading noise as structure —
    the three seeds of one cell differ by up to .08 in clean accuracy here.

    Two quantities are averaged, and the two averages are **not** interchangeable:

    - ``acc(sigma)`` at each grid point is averaged directly across seeds, which is
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


def print_summary_table(aggregated: dict, sigma_values: list[int]) -> None:
    """Print one row per ``(k, floor)`` setup, averaged over its seeds.

    ``retention`` is the chance-corrected fraction of accuracy surviving the
    strongest jitter; the analysis's ``temporal_score`` is ``1 - retention``. It is
    printed here only for eyeballing — the scoring, the regression on ``a`` and ``s``
    and the plotting belong to the analysis step.

    ``a`` and ``s`` are both shown, and ``sp/neu`` (their product) last, because the
    whole point of the v3 grid is that the first two move independently: reading a
    trend off the product alone is exactly the mistake that made v1 uninterpretable
    (document 5 §8 rule 1). Every column carries its across-seed spread, so a cell
    whose seeds disagree cannot be mistaken for a tight one.

    Args:
        aggregated: The per-setup dict produced by ``aggregate_over_seeds``.
        sigma_values: The sigma grid that was swept, in order.
    """
    sigma_min, sigma_max = str(sigma_values[0]), str(sigma_values[-1])

    print(
        f"\n{'setup':<14} {'n':>2} {'a':>13} {'s':>13} {'sp/neu':>7} "
        f"{'acc(' + sigma_min + ')':>15} {'acc(' + sigma_max + ')':>15} "
        f"{'retention':>15}"
    )
    for setup_tag, row in aggregated.items():
        sweep = row["sigma_sweep"]
        print(
            f"{setup_tag:<14} {row['n_seeds']:>2} "
            f"{format_mean_std(row['spikes_per_active_neuron'], row['spikes_per_active_neuron_std'], 2):>13} "
            f"{format_mean_std(row['silent_fraction'], row['silent_fraction_std'], 3):>13} "
            f"{row['spikes_per_neuron']:>7.2f} "
            f"{format_mean_std(sweep[sigma_min]['mean'], sweep[sigma_min]['std'], 4):>15} "
            f"{format_mean_std(sweep[sigma_max]['mean'], sweep[sigma_max]['std'], 4):>15} "
            f"{format_mean_std(row['retention'], row['retention_std'], 3):>15}"
        )
    print("\n  Means over seeds; +- is the across-seed standard deviation. "
          "Per-seed rows are kept in the results file under 'per_checkpoint'.")


def main() -> None:
    """Load the shared test set once, then sweep jitter over the no-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_jitter_sweep(test_loader)


if __name__ == "__main__":
    main()
