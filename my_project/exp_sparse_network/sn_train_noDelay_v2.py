"""First milestone v2.2 (training), NO-DELAY arm: top-k truncated hidden layer on SHD.

Successor to [sn_train_noDelay.py](sn_train_noDelay.py). Same network, data,
optimiser and schedule; **only the sparsity mechanism differs**.

The problem every previous version failed
------------------------------------------
H1 says a sparse layer is forced onto a *timing* code because *count* stops carrying
information. Testing that requires the count and identity channels — the ones every
rate-preserving perturbation leaves intact — to actually be closed. Three penalty
designs failed to close them, each in the same way:

===========  ==================================  ==========================
version      mechanism                           how the network escaped
===========  ==================================  ==========================
v1           hinge on batch-mean rate            went silent on some samples
v2           two-sided band [lo, hi]             varied count inside the band
v2.1         L1 exact-count |count - k|          kept a heavy tail (3.6-9.2%
                                                 of pairs above k+2)
===========  ==================================  ==========================

Measured decode of the label from the per-neuron count vector, which is what has to
fall to chance: v1 .687-.749, v2 .707-.754, v2.1 .659-.664. Barely moved. At k=1 the
count channel decoded at **.66 while the network itself scored .55** — counts still
beat the network, so no count-preserving perturbation could be guaranteed to hurt it.

**Why tuning cannot fix this.** Clipping v2.1's measured counts to progressively
tighter sets showed the requirement is effectively binary:

======================  ==========  =====  ========
count representation    at-target   std    decode
======================  ==========  =====  ========
as measured (L1)        59.8%       1.50   .661
clip to {0,1,2}         59.8%       0.63   .605
clip to {0,1}           78.9%       0.41   .528
pin to exactly 1        100%        0.00   **.050 = chance**
======================  ==========  =====  ========

Going 60% -> 79% at-target moved the decode only .66 -> .53. Nothing short of ~100%
collapses it. Since the count channel is worth more to the network (.66) than its own
accuracy (.55), it will exploit whatever slack a soft penalty leaves.

What v2.2 changes
-----------------
Stop *penalising* count variation and make it **impossible**:

1. **Hard top-k truncation in the forward pass.** After the 1st hidden layer spikes,
   only each neuron's first ``k`` spikes are passed on (``truncate_to_k_spikes``).
   The layer's output count is then exactly ``k`` for every neuron that fired at
   least ``k`` times — by construction, with nothing to calibrate.
2. **A one-sided floor penalty** ``relu(k - count)`` replaces the two-sided one. Its
   only job is to stop a neuron firing *fewer* than ``k`` times, which is the sole
   remaining way the count could vary. This is the *easy* direction: the v2 probe's
   [8,12] band reached 1.6% silent using nothing but its lower arm.

With no neuron under-firing, the count vector is constant, the identity vector is
constant, and the only thing left varying across samples is **when** the 128 x k
spikes occur — a pure latency code, which is H1's premise made true rather than
approximated. ``under_target_fraction`` is the single number that says whether this
holds; when it is 0, ``count_std`` is 0 and the count decode is chance by construction.

Sweeping ``k`` sweeps the layer's output firing rate exactly, while the
perturbation-immune channel capacity stays constant (nil) at every point — removing
the co-variation between sparsity and immune-channel capacity that made v1's headline
correlation uninterpretable.

Two consequences worth knowing:

- **No anneal is needed.** The dead zone (SLAYER's surrogate gradient vanishing below
  threshold) was a hazard of pushing firing *down*; v2.2 only ever pushes it *up*, and
  truncation masks a neuron's output without weakening its drive. That matters most
  for this arm, which went fully silent for 26 epochs under v1's hinge and still sat
  at 21-40% silent under v2/v2.1. ``ANNEAL_EPOCHS`` is kept but defaults to 0.
- **Raw firing is unconstrained above k** and is logged separately
  (``raw_spikes_per_neuron``). Only the truncated output reaches ``fc2``, is measured,
  and is perturbed at eval, so the experiment is well defined regardless; the raw rate
  is a diagnostic for how hard the neurons are being driven.

What v2.1 got right and v2.2 keeps
-----------------------------------
The v2.1 probe was not a wasted run — it established two things this design depends on:
the constraint can be made to bind (``sp/neuron`` landed at 1.19 and 3.13 against
targets 1 and 3, versus 3.08 for v2's [1,2] band), and **k = 1 is learnable** (clean
accuracy .552 here, .772 with delays). The regime is feasible; only the pinning
mechanism was wrong.

What differs from the with-delay v2.2 script
--------------------------------------------
Only the network, exactly as in v1: ``delay1``/``delay2``, the adaptive
delay-clamping schedule and the ``delay_mean`` log field are removed outright rather
than switched off by a flag, so the class is ``SparseSHDNetworkNoDelay`` and spikes go
straight ``fc1 -> (truncate) -> fc2 -> fc3``. Any surviving sparsity ->
timing-dependence trend must therefore come from the SRM neurons' own membrane
dynamics rather than from learnable axonal delays. Everything else is held identical
to the with-delay arm, including the warm-up.

Expect **lower clean accuracy** than the with-delay arm throughout (v1: ~.49-.59 vs
~.78-.89; v2.1 probe: .50-.55 vs .77-.78). That is not a failure of the sparsity
mechanism — it is why the analysis uses a chance-corrected, baseline-normalised score
and compares the arms at matched firing rate, never on raw accuracy drops.

Outputs are tagged ``_v2_2_`` and cannot collide with v1, the v2 band probe, or the
v2.1 exact-count probe.

This is a **training-only** script. The perturbations that measure temporal processing
are applied **only at evaluation**, by the sibling experiments under
``exp_sparse_network/{jitter,shift,shd,deletion}/`` — never switch one on here. Note
those eval scripts carry their own copy of the network class and **must be given the
same truncation** before they are run against v2.2 checkpoints. See:

- ``my_project/docs/progress/sparse_network_1stLayer_results.md`` — v1 results.
- ``my_project/docs/progress/sparse_network_test_progress_v2.md`` — v2 checklist.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.

What this script produces, per ``(target_count, seed)``:

- a checkpoint ``sn_data/sparse_whole_nodelay_v2_2_k{k}_str{s}_seed{seed}.pt``;
- a training-curve log ``sn_log/..._training_log.json`` including ``train_under_k``,
  ``train_raw_rate`` and ``train_count_std``;
- a row in ``sn_log/sparse_whole_nodelay_v2_2_train_summary.json`` recording clean
  test accuracy and **``under_target_fraction``** — the manipulation check.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Directory of this script. Dataset, checkpoint, and log paths are anchored here
# so the script can be launched from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "sn_data"       # checkpoints written here
LOG_DIR = SCRIPT_DIR / "sn_log"         # training logs + summary written here
SHD_DATA_DIR = SCRIPT_DIR / "shd/shd_data"  # SHD .mat source (this experiment's own copy)

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Quick pipeline check / probe. v2.2's truncation is structural, so the count channel
# closes by construction -- there is no calibration that can get it wrong. What the
# probe must establish is whether the network can still do the task under it. Check,
# in priority order:
#   (1) under_target_fraction -> 0? The single manipulation check. It counts pairs
#       firing FEWER than k, the only remaining route to count variation. At 0 the
#       count vector is constant, count_std is 0, and the count decode is chance. If
#       it stays high, raise PENALTY_STRENGTH (pushing firing UP has no dead-zone
#       risk, so this is safe to do aggressively). This arm is the likelier of the two
#       to need it -- it fires ~half as much as the delay net intrinsically and
#       reached the higher silent fraction under every previous version.
#   (2) clean_acc above chance at k=1? THE REAL RISK, and sharper here than in the
#       delay arm. With count and identity carrying nothing, the task must be solved
#       from spike times alone. v2.1 showed k=1 is learnable (.552) when the mean was
#       pinned; truncation is stricter. If accuracy collapses to chance, H1 is
#       UNTESTABLE in this architecture -- a legitimate finding, not a bug (see the
#       v2 checklist §5).
#   (3) raw_spikes_per_neuron -- diagnostic only. Unconstrained above k by design.
#       Watch it for a pathological blow-up, which would mean the first k spikes are
#       all crowding into the earliest bins and carrying no timing information.
# Probe artifacts carry the _probe suffix (RUN_SUFFIX) so they can never clobber a
# real run.
QUICK_TEST: bool = True

# Suffix appended to every output name (checkpoints, per-model logs, summary) when
# running a probe, so a QUICK_TEST probe can never overwrite real-run artifacts that
# share the same (target_count, seed). Empty for the real run.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Marks every v2.2 artifact so it can never collide with a v1 checkpoint/summary, the
# superseded v2 band probe, or the superseded v2.1 exact-count probe.
VERSION_TAG: str = "v2_2"

# --- Milestone scope (deliberately narrow; see progress doc §3) ---
DATASET_KEY: str = "whole"
DELAY_TAG: str = "nodelay"    # tags checkpoints/logs apart from the delay run
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# --- Sparsity sweep: sweep the truncation level k ---
# Every neuron emits exactly k spikes per sample (truncation), so k IS the layer's
# output firing rate -- an exactly controlled independent variable rather than an
# achieved one. At every k the count vector and the identity vector are constant, so
# the perturbation-immune channel capacity is held at nil across the whole sweep. That
# uniformity is the point: in v1 and v2 the immune capacity co-varied with sparsity,
# which is what made the headline correlation impossible to interpret.
#
# k = 1 is the sharpest test: 128 spikes per sample, all information in their times.
# Held identical to the with-delay arm's grid so the two are directly comparable.
TARGET_COUNTS: list[float] = [1.0, 2.0, 3.0, 5.0, 8.0]
SEEDS: list[int] = [42, 43, 44]

# Floor-penalty coefficient, on relu(k - count). Its only job is to stop neurons
# firing fewer than k times; everything above k is handled by truncation. Pushing
# firing UP carries no dead-zone risk (truncation masks output without weakening a
# neuron's drive), so this can be set aggressively. 10 continues v2.1's value; the v2
# probe reached 1.6% silent in this direction with only strength 3, so there is margin.
# If under_target_fraction stays high, raise it; if clean accuracy suffers, lower it.
PENALTY_STRENGTH: float = 10.0

# --- Anneal schedule for k (retained, but OFF by default) ---
# v2.1 needed this because a tight target pushed firing down into the dead zone, where
# SLAYER's surrogate gradient vanishes. v2.2 never pushes firing down -- truncation
# masks a neuron's output without touching its drive, and the floor penalty only
# pushes up -- so the hazard is gone and no curriculum is required. That is a bigger
# deal for this arm than the other: it is the one that went fully silent for 26 epochs
# under v1. Kept because annealing the truncation level is a sensible fallback if the
# k=1 models train poorly from a cold start. ANNEAL_EPOCHS = 0 disables it.
ANNEAL_FROM: float = 8.0
ANNEAL_EPOCHS: int = 0

# Clean warm-up before the penalty engages, in epochs. 0 for BOTH v2.2 arms. Note the
# truncation itself is always on, warm-up or not: it defines the layer, it is not a
# regulariser. Only the floor penalty is gated by this. v1 ran this arm at 15 as a
# collapse guard; the guard is unnecessary now that nothing pushes firing down, and
# dropping it keeps the two arms directly comparable.
WARMUP_EPOCHS: int = 0

# --- SLAYER neuron and simulation descriptors ---
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

# --- Data split ratios (identical to v1 and the rest of the project) ---
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# --- Training hyper-parameters (held identical to v1 for comparability) ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
EARLY_STOP_PATIENCE: int = 300

# --- Acceptance threshold, used only to annotate the printed summary table ---
# Fraction of (sample, neuron) pairs firing FEWER than k spikes. Truncation guarantees
# no pair exceeds k, so this is the only remaining source of count variation and the
# complete manipulation check.
MAX_UNDER_TARGET: float = 0.02


def resolve_run_config() -> tuple[list[float], float, list[int], int, int, int]:
    """Return (target_counts, penalty_strength, seeds, epochs, warmup, anneal_epochs).

    Collapses to a small, fast probe grid when ``QUICK_TEST`` is set so v2.2 can be
    checked cheaply before the full sweep.

    Returns:
        Tuple of (target_counts, penalty_strength, seeds, epochs, warmup_epochs,
        anneal_epochs).
    """
    if QUICK_TEST:
        # v2.2 VALIDATION PROBE. Two targets, one seed, 400 epochs (~35 min/model on
        # this arm, measured from the v2 probe). k=1 is the decisive one and runs
        # first so a failure surfaces early; k=3 confirms behaviour across the sweep.
        # Matches the v2.1 probe grid so the two are directly comparable.
        #
        # Unlike previous probes this is NOT a calibration of the manipulation --
        # truncation closes the count channel structurally. It is a feasibility check
        # on whether the task survives, plus a check that the floor penalty is strong
        # enough to drive under_target_fraction to 0.
        return [1.0, 3.0], PENALTY_STRENGTH, [42], 400, 0, ANNEAL_EPOCHS
    return (TARGET_COUNTS, PENALTY_STRENGTH, SEEDS, EPOCHS, WARMUP_EPOCHS,
            ANNEAL_EPOCHS)


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


def build_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split data and build train/val/test DataLoaders.

    The splits are fixed fractional ranges, so the test set is identical across
    every model and identical to v1's — a requirement for comparing their
    perturbation curves. Only the training order is shuffled.

    Args:
        X: Full dataset features, shape (N, neurons, T).
        Y: Full dataset labels, shape (N,).
        batch_size: Batch size for all loaders.
        seed: Random seed for the one-off train-index shuffle.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    N = len(Y)
    train_idx = get_split_indices(TRAIN_RANGE, N)
    val_idx = get_split_indices(VAL_RANGE, N)
    test_idx = get_split_indices(TEST_RANGE, N)

    np.random.seed(seed)
    np.random.shuffle(train_idx)

    train_ds = SpikeDataset(X[train_idx], Y[train_idx])
    val_ds = SpikeDataset(X[val_idx], Y[val_idx])
    test_ds = SpikeDataset(X[test_idx], Y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def truncate_to_k_spikes(
    hidden_spikes: torch.Tensor,
    k: float | None,
) -> torch.Tensor:
    """Keep only each (sample, neuron)'s first ``k`` spikes; drop the rest.

    This is v2.2's central mechanism, and the reason it succeeds where three penalty
    designs failed. Rather than making count variation *costly*, it makes count
    variation *impossible*: after truncation no neuron can emit more than ``k``
    spikes, so combined with a floor penalty preventing fewer, the count vector is
    constant across samples and carries no information at all.

    Implementation: the running spike index along time is ``cumsum`` of the binary
    train, so the n-th spike sits where the cumulative sum equals n. Keeping bins
    whose cumulative sum is ``<= k`` therefore keeps exactly the first ``k`` spikes.
    The comparison produces a non-differentiable boolean mask, so gradients flow
    straight through the surviving spikes and are simply absent for the dropped ones
    — the network gets no signal to stop producing spikes above ``k``, which is
    correct, since those spikes never leave the layer.

    Args:
        hidden_spikes: Binary 1st hidden layer spikes, shape (B, C, 1, 1, T).
        k: Maximum spikes to keep per (sample, neuron). ``None`` disables truncation,
            which reproduces the pre-v2.2 behaviour for loading older checkpoints.

    Returns:
        Tensor of the same shape, dtype and device, with at most ``k`` spikes per
        (sample, neuron).
    """
    if k is None:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    flat = hidden_spikes.view(B, C, T)
    spike_index = flat.cumsum(dim=-1)          # n-th spike -> cumulative sum n
    keep = (spike_index <= k).to(flat.dtype)   # boolean -> constant, no gradient
    return (flat * keep).view(B, C, H, W, T)


class SparseSHDNetworkNoDelay(nn.Module):
    """2-hidden-layer SLAYER SNN (no delays), for the v2.2 experiment.

    The parameter set is identical to v1's class of the same name, so checkpoints are
    interchangeable — but the *forward pass* now differs: the 1st hidden layer's
    output is truncated to each neuron's first ``truncate_k`` spikes before it reaches
    ``fc2``. That truncation is part of the layer's definition, not a regulariser, so
    anything evaluating a v2.2 checkpoint **must apply it too**; otherwise it is
    measuring a different network. ``truncate_k = None`` disables it and reproduces
    the v1/v2/v2.1 forward pass exactly.

    Spikes propagate straight from one dense layer to the next, with no
    ``delay1``/``delay2``, so the only timing machinery is the SRM neurons' own
    membrane dynamics.

    Training uses the clean ``forward`` (no perturbation). The eval-only perturbations
    of the truncated layer live under
    ``exp_sparse_network/{jitter,shift,shd,deletion}/``, not here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 128,
        num_classes: int = 20,
        truncate_k: float | None = None,
    ):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.truncate_k = truncate_k

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

    def _first_hidden_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike, BEFORE truncation (diagnostic use)."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The 1st hidden layer's actual output: raw spikes truncated to ``k``.

        This is the tensor that reaches ``fc2``, that the sparsity metrics measure,
        and that the eval scripts perturb.
        """
        return truncate_to_k_spikes(self._first_hidden_raw(x), self.truncate_k)

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> fc2 -> spike -> fc3 -> spike."""
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Clean forward pass (used during training and clean evaluation).

        Args:
            x: Input spike trains.
            return_hidden: If True, also return the truncated 1st hidden layer spikes
                and the raw (pre-truncation) ones, so the caller can apply the floor
                penalty and log how hard the neurons are being driven.

        Returns:
            The output spike tensor, or ``(output, hidden1, hidden1_raw)`` if
            ``return_hidden`` is True.
        """
        x = self._prepare_input(x)
        hidden1_raw = self._first_hidden_raw(x)
        hidden1 = truncate_to_k_spikes(hidden1_raw, self.truncate_k)
        out = self._second_hidden_and_output(hidden1)
        return (out, hidden1, hidden1_raw) if return_hidden else out


def hidden_mean_rate(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """Mean spikes per hidden neuron per sample (for logging).

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).

    Returns:
        Scalar tensor (mean spikes per neuron per sample).
    """
    return hidden_spikes.sum(dim=-1).mean()


def hidden_activity_stats(
    hidden_spikes: torch.Tensor,
    target_count: float,
) -> tuple[float, float, float]:
    """Return (under_target_fraction, silent_fraction, count_std) for one batch.

    ``under_target_fraction`` is v2.2's manipulation check, and supersedes the
    count_std proxy v2.1 used. Truncation guarantees no (sample, neuron) pair exceeds
    ``target_count``, so pairs firing *fewer* than it are the only remaining source of
    count variation. When this reaches 0 the count vector is constant across samples,
    ``count_std`` is 0, every neuron is active, and the label cannot be decoded from
    counts or from neuron identity at all.

    Args:
        hidden_spikes: TRUNCATED 1st hidden layer spikes, shape (B, C, 1, 1, T).
        target_count: The truncation level ``k`` these spikes were produced under.

    Returns:
        Tuple of (under_target_fraction, silent_fraction, count_std).
    """
    counts = hidden_spikes.sum(dim=-1).detach()
    n = max(1, counts.numel())
    under_target = (counts < target_count).sum().item() / n
    silent_fraction = (counts == 0).sum().item() / n
    count_std = counts.std().item() if counts.numel() > 1 else 0.0
    return under_target, silent_fraction, count_std


def annealed_target(
    target_count: float,
    epoch: int,
    anneal_from: float = ANNEAL_FROM,
    anneal_epochs: int = ANNEAL_EPOCHS,
) -> float:
    """Return the truncation level in force at ``epoch``, ramping linearly.

    ``k`` starts at ``anneal_from`` and reaches ``target_count`` at ``anneal_epochs``,
    holding there afterwards. Disabled by default in v2.2 (``ANNEAL_EPOCHS = 0``): the
    dead-zone hazard it existed to avoid was a consequence of pushing firing down,
    which v2.2 never does. Retained as a fallback in case cold-starting the k=1 models
    trains poorly.

    Args:
        target_count: Final truncation level.
        epoch: Current epoch index (0-based).
        anneal_from: Level in force at epoch 0.
        anneal_epochs: Epochs over which to ramp. 0 disables annealing.

    Returns:
        The truncation level to use this epoch.
    """
    if anneal_epochs <= 0 or epoch >= anneal_epochs:
        return float(target_count)
    fraction = epoch / anneal_epochs
    return float(anneal_from + fraction * (target_count - anneal_from))


def hidden_min_count_penalty(
    hidden_spikes: torch.Tensor,
    target_count: float,
) -> torch.Tensor:
    """One-sided floor penalty: charge each (sample, neuron) for firing under ``k``.

    ::

        per_sample_rate = hidden_spikes.sum(dim=-1)          # (B, C, 1, 1)
        penalty = relu(target_count - per_sample_rate).mean()

    Truncation already caps the count at ``k``, so this penalty's sole job is to stop
    neurons falling *below* it — the one remaining way the count could vary across
    samples. Together they pin the count exactly, with no calibration.

    Unlike every previous version's penalty, this one pushes firing **up** only. That
    matters especially for this arm: the dead zone that left it fully silent for 26
    epochs under v1, and at 21-40% silent under v2/v2.1, is a hazard of pushing firing
    *down* past threshold, where SLAYER's surrogate gradient vanishes and nothing can
    revive the neuron. Pushing up is the safe direction, and the v2 probe confirmed it
    empirically: the [8,12] band reached 1.6% silent using only its lower arm at
    strength 3.

    Computing this on the raw (pre-truncation) count or the truncated count gives
    identical values *and* identical gradients, since truncation only bites when the
    count already exceeds ``k``, where the penalty and its gradient are both zero.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).
        target_count: The floor each neuron must reach (the swept axis).

    Returns:
        Scalar penalty tensor (mean shortfall below ``target_count``).
    """
    per_sample_rate = hidden_spikes.sum(dim=-1)
    return torch.relu(target_count - per_sample_rate).mean()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = False


def build_loss_and_optimizer(
    net: SparseSHDNetworkNoDelay,
    lr: float = 0.1,
) -> tuple:
    """Build the NumSpikes loss, Nadam optimizer, and LR scheduler.

    ``NumSpikes`` pins the *output* firing rate to fixed targets, which keeps the
    sparsity intervention localised to the hidden layer.

    Args:
        net: The network to optimize.
        lr: Base learning rate.

    Returns:
        Tuple of (loss_fn, optimizer, scheduler).
    """
    error_cfg = {
        "neuron": LIF_PARAMS,
        "simulation": SIM_PARAMS,
        "training": {
            "error": {
                "type": "NumSpikes",
                "tgtSpikeRegion": {"start": 0, "stop": 200},
                "tgtSpikeCount": {True: 40, False: 4},
            }
        },
    }
    loss_fn = snn.spikeLoss.spikeLoss(error_cfg)
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300], gamma=0.1
    )
    return loss_fn, optimizer, scheduler


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_count: float,
    seed: int,
    epochs: int,
    penalty_strength: float = PENALTY_STRENGTH,
    warmup_epochs: int = WARMUP_EPOCHS,
    anneal_from: float = ANNEAL_FROM,
    anneal_epochs: int = ANNEAL_EPOCHS,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[SparseSHDNetworkNoDelay, dict]:
    """Train one clean model with a top-k truncated hidden layer and a floor penalty.

    The forward pass is clean (no perturbation) and always truncated — truncation
    defines the layer rather than regularising it, so ``warmup_epochs`` gates only the
    floor penalty. After the warm-up the total loss is the NumSpikes task loss plus
    ``penalty_strength`` times the shortfall below ``target_count``.

    Best-model selection and early stopping use the *task* validation loss and are
    reset once any anneal completes, so the saved checkpoint comes from the fully
    constrained regime.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        target_count: Truncation level and floor (the swept axis).
        seed: Random seed (controls init and shuffle order).
        epochs: Maximum training epochs.
        penalty_strength: Floor-penalty coefficient.
        warmup_epochs: Clean epochs before the floor penalty engages (0 in v2.2).
        anneal_from: Truncation level in force at epoch 0.
        anneal_epochs: Epochs over which the level ramps. 0 disables annealing.
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        lr: Learning rate.
        patience: Early stopping patience (on task val loss).

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = SparseSHDNetworkNoDelay(
        input_dim, hidden_units, num_classes, truncate_k=target_count
    ).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    # Epoch from which the checkpoint may be selected: the first epoch at which the
    # penalty is engaged AND the truncation level has finished annealing.
    selection_start = max(warmup_epochs, anneal_epochs)

    log = {
        "epoch": [],
        "train_loss": [],        # total (task + penalty)
        "train_task_loss": [],
        "train_rate": [],        # spikes / neuron / sample AFTER truncation
        "train_raw_rate": [],    # spikes / neuron / sample BEFORE truncation
        "train_under_k": [],     # manipulation check: 0 => count carries nothing
        "train_silent": [],      # fraction of (sample, neuron) pairs with no spikes
        "train_count_std": [],   # 0 whenever train_under_k is 0
        "train_target_k": [],    # the truncation level in force this epoch
        "val_loss": [],          # task only
        "val_acc": [],
    }

    desc = f"Train nodelay k={target_count:g} seed={seed}"
    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=desc) as pbar:
        for epoch in range(epochs):
            # Reset best-model tracking once the truncation level is final, so the
            # saved checkpoint is the best *fully constrained* model.
            if epoch == selection_start and selection_start > 0:
                best_val_loss = float("inf")
                best_model_state = None
                early_stop_counter = 0

            penalty_on = epoch >= warmup_epochs
            target_k = annealed_target(
                target_count, epoch, anneal_from, anneal_epochs
            )
            net.truncate_k = target_k

            # --- Train (clean truncated forward + floor penalty) ---
            net.train()
            batch_total_losses = []
            batch_task_losses = []
            batch_rates = []
            batch_raw_rates = []
            batch_under_k = []
            batch_silent = []
            batch_count_std = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs, hidden1, hidden1_raw = net(x_batch, return_hidden=True)
                task_loss = loss_fn.numSpikes(outputs, target)
                rate = hidden_mean_rate(hidden1)
                raw_rate = hidden_mean_rate(hidden1_raw)
                under_k, silent, count_std = hidden_activity_stats(hidden1, target_k)
                if penalty_on:
                    penalty = hidden_min_count_penalty(hidden1_raw, target_k)
                    loss = task_loss + penalty_strength * penalty
                else:
                    loss = task_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_total_losses.append(loss.item())
                batch_task_losses.append(task_loss.item())
                batch_rates.append(rate.item())
                batch_raw_rates.append(raw_rate.item())
                batch_under_k.append(under_k)
                batch_silent.append(silent)
                batch_count_std.append(count_std)
                pbar.update(1)

            # --- Validate (clean; task loss only) ---
            net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = (
                        x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                    )
                    y_batch = y_batch.to(device).long()

                    target = torch.zeros(
                        (len(y_batch), num_classes, 1, 1, 1), device=device
                    )
                    target.scatter_(
                        1, y_batch[:, None, None, None, None], 1.0
                    )

                    outputs = net(x_batch)
                    val_loss += loss_fn.numSpikes(outputs, target).item()

                    pred = snn.predict.getClass(outputs)
                    correct += (pred.cpu() == y_batch.cpu()).sum().item()
                    total += len(y_batch)

            val_loss /= max(1, len(val_loader))
            val_acc = correct / max(1, total)
            train_loss = float(np.mean(batch_total_losses))
            train_task_loss = float(np.mean(batch_task_losses))
            train_rate = float(np.mean(batch_rates))
            train_raw_rate = float(np.mean(batch_raw_rates))
            train_under_k = float(np.mean(batch_under_k))
            train_silent = float(np.mean(batch_silent))
            train_count_std = float(np.mean(batch_count_std))

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["train_task_loss"].append(train_task_loss)
            log["train_rate"].append(train_rate)
            log["train_raw_rate"].append(train_raw_rate)
            log["train_under_k"].append(train_under_k)
            log["train_silent"].append(train_silent)
            log["train_count_std"].append(train_count_std)
            log["train_target_k"].append(float(target_k))
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))

            pbar.set_postfix(
                epoch=epoch + 1,
                k=f"{target_k:.2f}",
                task=f"{train_task_loss:.3f}",
                raw=f"{train_raw_rate:.2f}",
                under=f"{train_under_k:.1%}",
                acc=f"{val_acc:.2%}",
            )
            scheduler.step()

            # Early stopping on task val loss (only once the level is final)
            if epoch < selection_start:
                continue
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.clone() for k, v in net.state_dict().items()
                }
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

    net.truncate_k = target_count
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


@torch.no_grad()
def evaluate_clean_and_activity(
    net: SparseSHDNetworkNoDelay,
    test_loader: DataLoader,
    target_count: float,
) -> dict:
    """Measure clean test accuracy and 1st hidden layer activity in one pass.

    All sparsity statistics are computed on the **truncated** layer output — the
    tensor that actually reaches ``fc2`` and that the eval scripts perturb — with the
    raw pre-truncation rate reported alongside as a diagnostic.

    ``under_target_fraction`` is the manipulation check: truncation caps every count
    at ``target_count``, so pairs below it are the only remaining count variation.
    At 0, ``count_std`` is 0 and neither counts nor neuron identity carry any label
    information.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.
        target_count: The truncation level the model was trained at.

    Returns:
        Dict with keys ``clean_acc``, ``firing_rate``, ``spikes_per_neuron``,
        ``raw_spikes_per_neuron``, ``spikes_per_active_neuron``, ``silent_fraction``,
        ``under_target_fraction``, ``count_std``, ``count_at_target``.
    """
    net.eval()
    correct = 0
    total = 0
    total_spikes = 0.0
    total_raw_spikes = 0.0
    total_sq = 0.0
    total_slots = 0
    total_neuron_samples = 0
    silent_neuron_samples = 0.0
    under_target_samples = 0.0
    at_target_samples = 0.0

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
        y_batch = y_batch.to(device)

        outputs, hidden1, hidden1_raw = net(x_batch, return_hidden=True)

        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)

        B, C, _, _, T = hidden1.shape
        counts = hidden1.sum(dim=-1).view(B, C)  # (sample, neuron), truncated
        total_spikes += counts.sum().item()
        total_raw_spikes += hidden1_raw.sum().item()
        total_sq += (counts ** 2).sum().item()
        total_slots += B * C * T
        total_neuron_samples += B * C
        silent_neuron_samples += (counts == 0).sum().item()
        under_target_samples += (counts < target_count).sum().item()
        at_target_samples += (counts == target_count).sum().item()

    active_neuron_samples = total_neuron_samples - silent_neuron_samples
    n = max(1, total_neuron_samples)
    mean_count = total_spikes / n
    variance = max(0.0, total_sq / n - mean_count ** 2)

    return {
        "clean_acc": correct / max(1, total),
        "firing_rate": total_spikes / max(1, total_slots),
        "spikes_per_neuron": mean_count,
        "raw_spikes_per_neuron": total_raw_spikes / n,
        "spikes_per_active_neuron": total_spikes / max(1.0, active_neuron_samples),
        "silent_fraction": silent_neuron_samples / n,
        "under_target_fraction": under_target_samples / n,
        "count_std": variance ** 0.5,
        "count_at_target": at_target_samples / n,
    }


def run_milestone() -> None:
    """Train the (target_count, seed) grid and record sparsity + clean accuracy."""
    (target_counts, penalty_strength, seeds, epochs, warmup_epochs,
     anneal_epochs) = resolve_run_config()

    n_models = len(target_counts) * len(seeds)
    print(f"{'#' * 70}")
    print("# Sparse-network milestone v2.2 (training): SHD whole, SGD no-delay")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs} | warmup={warmup_epochs}")
    print("# mechanism=hard top-k truncation + floor penalty relu(k-count)")
    print(f"# strength: {penalty_strength} | anneal_epochs: {anneal_epochs}")
    print(f"# target counts: {target_counts}")
    print(f"# seeds: {seeds}  ->  {n_models} models")
    print(f"{'#' * 70}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data and build loaders once; the split (and hence the test set) is
    # fixed, so all models are comparable — including against v1.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    train_loader, val_loader, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=42,
    )

    summary: dict[str, dict] = {}

    for target_count in target_counts:
        for seed in seeds:
            run_tag = (
                f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_"
                f"k{target_count:g}_str{penalty_strength:g}_"
                f"seed{seed}{RUN_SUFFIX}"
            )
            print(f"\n{'=' * 60}")
            print(f"  Training {run_tag}")
            print(f"{'=' * 60}")

            net, training_log = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                target_count=target_count,
                seed=seed,
                epochs=epochs,
                penalty_strength=penalty_strength,
                warmup_epochs=warmup_epochs,
                anneal_from=ANNEAL_FROM,
                anneal_epochs=anneal_epochs,
            )

            ckpt_path = DATA_DIR / f"{run_tag}.pt"
            torch.save(net.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

            metrics = evaluate_clean_and_activity(net, test_loader, target_count)
            closed = metrics["under_target_fraction"] < MAX_UNDER_TARGET
            print(
                f"  clean_acc={metrics['clean_acc']:.4f} | "
                f"under_k={metrics['under_target_fraction']:.2%} | "
                f"count_std={metrics['count_std']:.3f} | "
                f"silent={metrics['silent_fraction']:.2%} | "
                f"raw_rate={metrics['raw_spikes_per_neuron']:.2f} | "
                f"{'CHANNEL CLOSED' if closed else 'STILL LEAKING'}"
            )

            summary[run_tag] = {
                "target_count": float(target_count),
                "penalty_strength": float(penalty_strength),
                "anneal_from": float(ANNEAL_FROM),
                "anneal_epochs": int(anneal_epochs),
                "truncate_k": float(target_count),
                "seed": int(seed),
                **{k: float(v) for k, v in metrics.items()},
            }

            # Persist per-model training curve.
            log_path = LOG_DIR / f"{run_tag}_training_log.json"
            log_serialisable = {
                k: [float(v) for v in vals]
                for k, vals in training_log.items()
            }
            with open(log_path, "w") as fp:
                json.dump(log_serialisable, fp, indent=2)
            print(f"  Training log saved to {log_path}")

    # Persist the sweep summary (the calibration/analysis table).
    summary_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_"
          f"train_summary{RUN_SUFFIX}.json"
    )
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Final table. v2.2's acceptance criteria, in priority order:
    #   under_k   -> 0        counts and identity carry nothing (THE manipulation)
    #   clean_acc >> chance   the task is still solvable from spike times alone
    #   raw_rate              diagnostic; watch for a pathological blow-up
    print(
        f"\n{'run_tag':<58} {'clean_acc':>9} {'k':>4} {'under_k':>9} "
        f"{'count_std':>10} {'silent':>8} {'raw_rate':>9}"
    )
    for run_tag, row in summary.items():
        print(
            f"{run_tag:<58} {row['clean_acc']:>9.4f} {row['target_count']:>4g} "
            f"{row['under_target_fraction']:>9.2%} {row['count_std']:>10.3f} "
            f"{row['silent_fraction']:>8.2%} {row['raw_spikes_per_neuron']:>9.2f}"
        )


if __name__ == "__main__":
    run_milestone()
