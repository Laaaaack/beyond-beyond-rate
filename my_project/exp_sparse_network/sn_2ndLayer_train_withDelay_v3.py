"""2nd-layer milestone v3 (training), WITH-DELAY arm: the a x s dissociation factorial.

Sibling of [sn_train_withDelay_v3.py](sn_train_withDelay_v3.py), which runs the
identical factorial against the **1st** hidden layer. Same network, data, splits,
optimiser, schedule and logging; the only difference is **which layer the two penalties
act on**, and therefore which layer's code the experiment is about.

Why repeat the experiment one layer deeper
-------------------------------------------
The 1st-layer factorial answered H1' for layer 1 and got a null: with ``a`` and ``s``
decorrelated by construction (rho = -0.053), beta_a's point estimate ran *against* H1'
and the residual signal was selectivity plus general robustness. One reading of that
null is that it is a fact about temporal coding. Another is that it is a fact about
**layer 1 specifically** — the first layer sits directly on the input, its job is
largely to re-represent a 700-channel spike train, and it may simply not be where the
network's timing computation lives. In this arm that reading has teeth: ``delay1`` sits
immediately *after* layer 1, so the learnable-delay machinery this arm exists to study
does its work between layer 1 and layer 2, and layer 2 is the first place its output is
visible.

The baseline measurement says the two layers are very different objects here — far more
so than in the no-delay arm (``v3_analysis/layer2_baseline.py``, v1's 12 checkpoints):

=========================  =================  =================
                           layer 1            layer 2
=========================  =================  =================
``spikes_per_neuron``      2.12 - 11.66       14.12 - 23.93
``a`` (per active neuron)  3.82 - 15.47       20.71 - 30.44
``s`` (silent fraction)    24.6% - 55.1%      4.4% - 39.6%
``rho(a, s)`` across v1    **-0.455**         **+0.427**
temporal support           ``[0, 88)``        ``[0, 160)``
=========================  =================  =================

Three things to take from that table:

1. **Layer 2 fires about twice as hard as layer 1** (``a`` of 20-30 against 4-15).
   The ceiling has correspondingly more work to do at every ``k``, and the row grid had
   to be extended upward — see ``CEILING_K``.
2. **``rho(a, s)`` flips sign.** At layer 1, v1's penalty drove ``a`` down while pushing
   ``s`` up. At layer 2, which v1 never penalised directly, the two axes move
   *together*. At ``+0.427`` this arm's layer-2 confound is milder than the no-delay
   arm's ``+0.829`` and sits just inside the acceptance threshold, but on n = 12 at
   p = .17 that is a power statement, not a clean bill of health.
3. **The temporal support nearly doubles**, from ``[0, 88)`` to ``[0, 160)``, because
   ``delay1`` (up to 64 bins) shifts layer 1's spikes later before they ever reach
   layer 2. Any relocation or jitter perturbation measured against this layer must use
   the wider window; re-using layer 1's ``[0, 88)`` would throw away more than a third
   of the layer's own support and mix a large rate insult into a timing probe.

The two terms
-------------
Unchanged in form from the 1st-layer script; only the tensor they are applied to moves.

1. **Ceiling — sets ``a``.** ``relu(count - k)`` charged per ``(sample, neuron)``
   pair of the **2nd** hidden layer, not on the batch mean. Per-pair is the whole
   point: a batch-mean hinge lets a neuron satisfy the penalty by going *silent* on
   some samples and firing freely on others, which is exactly how ``a`` and ``s``
   became confounded in v1.
2. **Floor — sets ``s``.** ``relu(THETA_MARGIN - max_t u2(t))`` on the 2nd layer's
   **membrane potential**, not on its spike count.

The floor's mechanism argument survives the move, and it was re-checked rather than
assumed. Two things had to hold, and both were measured:

- **Silent pairs must be out of reach of a count-based penalty**, or the potential
  floor buys nothing. Silent *layer-2* pairs sit at a median peak potential of
  **0.65-1.26** against ``theta = 10`` in this arm — as deeply silent as layer 1's, so
  the dead zone that defeated every count-based floor in v1/v2/v2.1/v2.2 is present
  here too and the potential floor is still the right instrument.
- **The floor's gradient must not itself need the surrogate.** At layer 1,
  ``potential1 = fc1(psp(x))`` is a convolution of the *input*, so ``du/dW1`` never
  vanishes. At layer 2, ``d potential2 / d W2 = psp(delay1(hidden1))``, which is
  likewise surrogate-free — **but it is zero for any sample whose entire layer 1 is
  silent**, and such a pair could only be reached back through ``fc1``, i.e. through
  the very surrogate the floor exists to avoid. This failure mode has no layer-1
  analogue (layer 1's input is data and is never silent). Measured across all 27
  checkpoints: **0.00% of samples** have an entirely silent layer 1, in either arm.
  The risk is real in principle, absent in this network, and worth re-checking if the
  ceiling is ever driven hard enough to empty layer 1.

The floor's **strength is the column knob**: 0 in one column, ``FLOOR_STRENGTH`` in
the other. ``k`` is the row knob. Crossing them is the factorial.

What is calibrated, and what is only inherited
-----------------------------------------------
**Measured for layer 2 in this arm** — ``NATURAL_SPIKES_PER_NEURON`` (23.93, against
layer 1's 11.66), the temporal support ``[0, 160)``, and the ``CEILING_K`` grid, which
was extended to anchor the top of the row axis near the natural rate.

**Inherited from the 1st-layer arm and NOT yet validated here** —
``CEILING_STRENGTH = 10``, ``FLOOR_STRENGTH = 1`` and ``WARMUP_EPOCHS = 20``. Each was
calibrated against layer 1's activity, and the layer-1 experiment's own history is that
these constants do not transfer for free: ``CEILING_STRENGTH`` had to go 3 -> 10 when
the row axis came back at 1.4x, and ``WARMUP_EPOCHS = 0`` was unsurvivable. Run
``QUICK_TEST`` before committing to the grid. See
``sparse_network_2ndLayer_test_progress_v3.md``.

Reading the probe
-----------------
``QUICK_TEST`` runs the 4 corners ``k in {1, 16} x floor in {0, F}`` at 400 epochs — the
extremes of the extended row axis. It validates the **manipulation**, which is a
property of the constraint and readable early. Check, in priority order:

1. ``silent_fraction`` roughly constant as ``k`` varies *within* a column;
2. ``silent_fraction`` differing by **>= 20 points** *across* columns at matched ``k``
   — if not, raise ``FLOOR_STRENGTH``;
3. ``spikes_per_neuron`` not inflated beyond this layer's natural 23.93;
4. clean accuracy comfortably above chance in every corner.

**Do not read achieved firing rates or accuracy levels off the probe.** Every v1 model
was still improving at epoch 1250 (best val loss at epochs 1168-1243), so 400 epochs is
a different regime, not a scaled-down one. Calibrate nothing from it.

Design acceptance check
------------------------
After the full grid, ``rho(a, s)`` across the trained models is printed. **If
|rho| > 0.5 the factorial has failed to break the confound** and the regression it
feeds is not interpretable — report that and stop. The layer-2 observational baseline
this must improve on is **+0.427** in this arm.

Outputs are tagged ``_v3L2_`` and cannot collide with the 1st-layer v3 grid, with v1,
or with any v2 probe.

This is a **training-only** script. The perturbations that measure temporal processing
are applied **only at evaluation**. Note that the existing eval scripts under
``exp_sparse_network/{jitter,shift,shd,deletion}/`` all inject at the **1st** hidden
layer and confine relocation/jitter to ``[0, 88)``; measuring this experiment needs
them retargeted to the 2nd layer at ``[0, 160)``. See:

- ``my_project/docs/progress/sparse_network_2ndLayer_test_progress_v3.md`` — this
  experiment's design, calibration and status.
- ``my_project/docs/progress/sparse_network_test_progress_v3.md`` — the 1st-layer v3
  design, Phase 0 and Phase 1 result this is the sibling of.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.

What this script produces, per ``(k, floor_strength, seed)``:

- a checkpoint ``sn_data/sparse_whole_delay_v3L2_k{k}_floor{f}_seed{seed}.pt``;
- a training-curve log ``sn_log/..._training_log.json`` carrying the trajectories of
  **both** sparsity axes, not just ``spikes_per_neuron``;
- a row in ``sn_log/sparse_whole_delay_v3L2_train_summary.json`` recording clean test
  accuracy and both axes.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Directory of this script. Dataset, checkpoint, and log paths are anchored here
# so the script can be launched from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "sn_data"       # checkpoints written here
LOG_DIR = SCRIPT_DIR / "sn_log"         # training logs + summary written here
SHD_DATA_DIR = SCRIPT_DIR / "shd/shd_data"  # SHD .mat source (this experiment's copy)

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Corner probe. True runs k in {1, 16} x floor in {0, FLOOR_STRENGTH} at 400 epochs,
# one seed -- 4 models, ~2.5 h on this arm. Its only job is to validate the
# manipulation; see "Reading the probe" in the module docstring. Probe artifacts carry
# RUN_SUFFIX so they can never clobber a real run.
#
# Leave this True until the three inherited constants have been checked against
# LAYER 2 in this arm. None of them was calibrated here.
QUICK_TEST: bool = False

# Suffix appended to every output name (checkpoints, per-model logs, summary) when
# running a probe, so a QUICK_TEST probe can never overwrite real-run artifacts that
# share the same (k, floor_strength, seed). Empty for the real run.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Marks every artifact of the 2nd-layer factorial. Distinct from the 1st-layer v3 tag
# ("v3"), so the two grids can sit side by side in sn_data/ and sn_log/ without any
# chance of one overwriting the other -- they share dataset, arm, k, floor and seed,
# and would otherwise produce identical filenames.
VERSION_TAG: str = "v3L2"

# --- Milestone scope ---
DATASET_KEY: str = "whole"
INPUT_DIM: int = 700          # SHD whole
USE_DELAY: bool = True        # SGD-delay
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# Which hidden layer the two penalties act on. This is the whole point of the script;
# it exists as a named constant so that the many places that say "layer 2" in prose can
# be checked against one value.
TARGET_LAYER: int = 2

# --- The factorial ---
# Row knob: the per-pair spike ceiling on the 2nd hidden layer. This is the H1 axis --
# it sets `a`, spikes per ACTIVE neuron, by making counts above k costly for every
# (sample, neuron) pair individually.
#
# EXTENDED FOR THIS ARM, on measurement rather than by analogy. Layer 2's natural `a`
# here is 20.71-30.44 (v1's 12 checkpoints) -- roughly twice layer 1's 3.82-15.47. With
# the 1st-layer grid [1, 2, 4, 8], even the loosest row would be a ~3x cut below the
# natural rate, so the ceiling would bind in every cell and the top of the `a` axis
# would sit unanchored well below where the layer naturally runs. That compresses the
# row axis, which is precisely the failure the 1st-layer experiment diagnosed and could
# not fix mid-flight: its own progress note recommends "adding a no-ceiling row (or
# k = 16) to anchor the top of the `a` axis near the natural rate", left unapplied only
# because that grid was already running.
#
# Nothing is running here yet, so it is applied. k = 16 sits just under the natural
# range and should leave the ceiling barely binding, anchoring the top of the axis.
# Cost: 20 models rather than 16, ~46 h rather than ~37 h at 1250 epochs.
#
# The no-delay arm keeps [1, 2, 4, 8] -- its layer-2 natural `a` is 7.49-12.46, so k = 8
# already sits at the top of the natural range there and needs no extension.
CEILING_K: list[float] = [1.0, 2.0, 4.0, 8.0, 16.0]

# Column knob: the membrane-potential floor. 0 lets neurons fall silent freely (high
# `s`); nonzero forbids it (low `s`). This is the H2 axis, and it is what makes the
# floor-on vs floor-off contrast at matched `k` a genuine controlled comparison rather
# than a correlation.
FLOOR_STRENGTH: list[float] = [0.0, 1.0]

SEEDS: list[int] = [42, 43, 44]

# Coefficient on the ceiling term. Fixed across the whole grid so that `k` is the only
# thing varying along the row axis.
#
# INHERITED FROM THE 1st-LAYER ARM, NOT YET VALIDATED AT LAYER 2. There, 3.0 (v2's
# value) proved too weak -- it left `a` spanning only 1.24-1.42x across k = 1 -> 8,
# because the penalty reaches equilibrium far above k and stops pushing -- and 10.0
# roughly doubled the row-axis span at an accuracy cost that stayed usable:
#
#   k=1     a      s      sp/neu   over_k   clean_acc     (LAYER 1, this arm)
#   ---------------------------------------------------
#   str=3,  floor=0   6.17  62.7%   2.30    27%     .818
#   str=3,  floor=1   4.90  11.0%   4.36    67%     .824
#   str=10, floor=0   4.00  72.1%   1.12    17%     .685
#   str=10, floor=1   2.66  22.6%   2.06    41%     .771
#
# Two reasons it may need re-tuning here rather than transferring, both pushing toward
# a LARGER value:
#   - layer 2 starts about twice as dense in `a` (20.71-30.44 against 3.82-15.47), so
#     the ceiling has much further to push at every k, and at layer 1 even strength 10
#     left 41% of pairs above k at k = 1;
#   - the ceiling's gradient at layer 2 reaches fc2 directly but reaches fc1 only
#     through the surrogate, so the same coefficient does not buy the same pressure.
# The probe's `over_k` column is what says whether it binds. If it stalls high, raise
# this before widening the k grid further.
CEILING_STRENGTH: float = 10.0

# The floor's threshold on peak membrane potential, as a multiple of theta = 10. 1.1 x
# theta so that a satisfied pair is not left sitting exactly on the boundary, where a
# small weight change would drop it back into silence.
THETA_MARGIN: float = 11.0

# Clean warm-up before either penalty engages, in epochs. **Required, not optional.**
#
# The ceiling relu(count - k) is one-sided and is minimised at count = 0, so in the
# floor-OFF column nothing but the task loss opposes silence -- and total silence is an
# *absorbing* state, because once every pair sits at u ~ 0 against theta = 10 the only
# route back is SLAYER's surrogate gradient, which carries no signal there. Measured at
# layer 1: warmup = 0 left the layer 100% silent by epoch 5 with val_acc pinned at
# chance for all 60 epochs, unrecoverable; warmup = 20 survived with 40-48% silent and
# val_acc 74.5%.
#
# INHERITED, AND THE ABSORBING STATE IS THE ONE FAILURE THIS GUARDS. There is a reason
# to think layer 2 is *more* exposed, not less: silencing layer 2 cuts the output layer
# off from the input entirely, and the recovery path runs back through the surrogate at
# BOTH fc2 and fc1. The probe's k = 1, floor = 0 corner is the cell that would show it.
# If val_acc sits at chance (5%) with silent -> 100%, raise this before anything else.
WARMUP_EPOCHS: int = 20

# --- SLAYER neuron and simulation descriptors (identical to v1 and v2) ---
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
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

# --- Acceptance thresholds, used only to annotate the printed summary ---
# Separation in silent_fraction the floor column must produce at matched k.
MIN_SILENT_SEPARATION: float = 0.20
# The natural firing rate of the 2nd hidden layer in this arm, measured on v1's 12
# checkpoints (range 14.12-23.93; this is the densest). The ceiling must not drive the
# layer ABOVE this; v2.2 inflated layer 1 from ~6.3 to 12.5-21.9 and stopped being an
# instance of "sparser-activity networks" as a result.
#
# This is a LAYER-2 figure and differs sharply from the 1st-layer script's 11.66 — this
# layer naturally runs about twice as hard. The no-delay arm's layer 2 sits at 5.53, so
# the constant must never be carried across either a layer or an arm boundary.
NATURAL_SPIKES_PER_NEURON: float = 23.93
# The design acceptance check on the factorial itself.
MAX_AXIS_CORRELATION: float = 0.5
# What the factorial has to improve on: rho(a, s) at LAYER 2 across v1's gradient in
# this arm. Note the sign -- at layer 2 the two axes move together, where at layer 1
# they opposed each other (-0.455).
V1_LAYER2_AXIS_CORRELATION: float = +0.427


def resolve_run_config() -> tuple[list[float], list[float], list[int], int, int]:
    """Return (ceiling_k, floor_strengths, seeds, epochs, warmup_epochs) for this run.

    Collapses to the 4-corner manipulation probe when ``QUICK_TEST`` is set. The
    corners are the extremes of both axes, which is what makes them a test of the
    manipulation: ``k`` at the ends of ``CEILING_K`` brackets the whole row axis, and
    the floor at 0 and ``FLOOR_STRENGTH`` is the entire column axis.

    Returns:
        Tuple of (ceiling_k, floor_strengths, seeds, epochs, warmup_epochs).
    """
    if QUICK_TEST:
        return [CEILING_K[0], CEILING_K[-1]], FLOOR_STRENGTH, [42], 400, WARMUP_EPOCHS
    return CEILING_K, FLOOR_STRENGTH, SEEDS, EPOCHS, WARMUP_EPOCHS


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

    The splits are fixed fractional ranges, so the test set is identical across every
    model and identical to v1's — a requirement for comparing their perturbation
    curves and for reusing the measurement layer unchanged. Only the training order is
    shuffled.

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


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with learnable delays, for the 2nd-layer factorial.

    The parameter set is identical to v1's class of the same name — ``fc1``/``fc2``/
    ``fc3`` weight-norm parameters plus ``delay1``/``delay2`` and nothing else — so
    these checkpoints load directly into the existing eval scripts and into
    ``v3_analysis/``. There is no truncation, so the forward pass is v1's exactly and a
    checkpoint needs no special handling anywhere downstream.

    The one difference from the 1st-layer script is which tensors ``forward`` hands
    back: the **2nd** hidden layer's spikes and the membrane potential they were
    thresholded from, because that is the layer both penalties now act on. Both tensors
    are computed by the forward pass regardless; returning them costs nothing.

    Note where the delays sit relative to the constrained layer. ``delay1`` is
    **upstream** of layer 2 — it acts on layer 1's spikes before ``fc2`` sees them — so
    it shapes the very activity the penalties are charging, and it is why this layer's
    temporal support runs to bin 159 rather than layer 1's 87. ``delay2`` is downstream
    and only feeds the output.

    Training uses the clean ``forward`` (no perturbation). The eval-only perturbations
    live under ``exp_sparse_network/{jitter,shift,shd,deletion}/``, not here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = HIDDEN_UNITS,
        num_classes: int = NUM_CLASSES,
        use_delay: bool = USE_DELAY,
        max_delay: int = MAX_DELAY,
    ):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.use_delay = use_delay
        self.max_delay = max_delay

        self.fc1 = nn.utils.weight_norm(
            slayer.dense(input_dim, hidden_units), name="weight"
        )
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(hidden_units, hidden_units), name="weight"
        )
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(hidden_units, num_classes), name="weight"
        )

        # delay1 sits between hidden1 and fc2, i.e. upstream of the constrained layer;
        # delay2 stays between the fc2 spike and fc3.
        if use_delay:
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
        """input -> fc1 -> spike."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden(
        self,
        hidden1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden1 -> (delay1) -> fc2 -> spike, returning the potential too.

        The potential is what the floor penalty acts on, so it is returned rather than
        discarded. ``d potential2 / d W2 = psp(delay1(hidden1))`` is an ordinary
        convolution gradient — no surrogate — which is what lets the floor reach pairs
        that sit far below threshold.

        Args:
            hidden1: 1st hidden layer spikes, shape (B, C, 1, 1, T).

        Returns:
            Tuple of (2nd hidden layer spikes, its membrane potential), same shape.
        """
        routed = self.delay1(hidden1) if self.use_delay else hidden1
        potential2 = self.fc2(self.slayer.psp(routed))
        return self.slayer.spike(potential2), potential2

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> (delay2) -> fc3 -> spike."""
        x = self.delay2(hidden2) if self.use_delay else hidden2
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Clean forward pass (used during training and clean evaluation).

        Args:
            x: Input spike trains.
            return_hidden: If True, also return the **2nd** hidden layer's spikes and
                the membrane potential they were thresholded from, so the caller can
                apply both penalties.

        Returns:
            The output spike tensor, or ``(output, hidden2, potential2)`` if
            ``return_hidden`` is True.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2, potential2 = self._second_hidden(hidden1)
        out = self._output(hidden2)
        return (out, hidden2, potential2) if return_hidden else out

    def clamp_delays(self, max1: int = 64, max2: int = 64) -> None:
        """Clamp delay parameters to [0, max]."""
        if not self.use_delay:
            return
        self.delay1.delay.data.clamp_(0, max1)
        self.delay2.delay.data.clamp_(0, max2)

    def get_delays(self) -> dict[str, np.ndarray]:
        """Return current delay values as a dict."""
        delays = {}
        if self.use_delay:
            delays["delay1"] = self.delay1.delay.data.cpu().numpy()
            delays["delay2"] = self.delay2.delay.data.cpu().numpy()
        return delays


def ceiling_penalty(hidden_spikes: torch.Tensor, k: float) -> torch.Tensor:
    """Per-pair spike ceiling: charge each ``(sample, neuron)`` for exceeding ``k``.

    ::

        counts  = hidden_spikes.sum(dim=-1)        # (B, C, 1, 1)
        penalty = relu(counts - k).mean()

    **The per-pair charge is the point.** v1 applied its hinge to the batch *mean*
    rate, which a neuron could satisfy by going silent on some samples while firing
    freely on others — building the identity code that made sparse networks look
    perturbation-robust, and confounding ``a`` with ``s``. Charging each pair
    individually removes that escape route: silence on one sample buys no licence to
    over-fire on another.

    This is the axis H1 is actually about. It is deliberately one-sided: nothing here
    stops a neuron falling silent, because whether silence is permitted is the *other*
    knob's job.

    Args:
        hidden_spikes: **2nd** hidden layer spikes, shape (B, C, 1, 1, T).
        k: Spikes per (sample, neuron) above which the penalty engages.

    Returns:
        Scalar penalty tensor (mean excess above ``k``).
    """
    return torch.relu(hidden_spikes.sum(dim=-1) - k).mean()


def floor_penalty(potential: torch.Tensor,
                  theta_margin: float = THETA_MARGIN) -> torch.Tensor:
    """Membrane-potential floor: charge each pair for never approaching threshold.

    ::

        peak    = potential.max(dim=-1).values      # (B, C, 1, 1)
        penalty = relu(theta_margin - peak).mean()

    This acts on the **potential**, not on the spike count, and that distinction is
    the whole reason it can work where four count-based floors could not. Measured on
    v1's checkpoints, silent 2nd-layer (sample, neuron) pairs sit at a median peak
    potential of **0.65-1.26** against ``theta = 10`` in this arm — they are not
    marginally sub-threshold, they receive essentially no drive at all. A count-based
    penalty reaches them only through SLAYER's surrogate gradient, which at ``u ~ 0``
    against ``theta = 10`` carries no signal; that dead zone silenced v1's no-delay arm
    for 26 epochs, left v2/v2.1 at 21-40% silent, and defeated v2.2's
    ``relu(k - count)`` even at strength 10.

    ``d potential2 / d W2 = psp(delay1(hidden1))`` is an ordinary convolution gradient
    and does not vanish — with one caveat that is specific to acting on a *hidden*
    layer rather than on the first: it **is** zero for a sample whose entire layer 1 is
    silent, and such a pair is reachable only back through ``fc1``'s surrogate.
    Measured at 0.00% of samples across all 27 v1 checkpoints, so this is a documented
    edge rather than a live problem, but it is the thing to check first if the floor
    ever stops biting.

    Asking for one suprathreshold crossing rather than for more spikes is also what
    should keep raw firing off v2.2's blow-up: it requests the minimum sufficient
    condition for the neuron to be non-silent.

    Args:
        potential: **2nd** hidden layer membrane potential, shape (B, C, 1, 1, T).
        theta_margin: Peak potential each pair must reach. Default ``1.1 x theta``, so
            a satisfied pair is not left sitting on the boundary.

    Returns:
        Scalar penalty tensor (mean shortfall below ``theta_margin``).
    """
    return torch.relu(theta_margin - potential.max(dim=-1).values).mean()


def hidden_activity_stats(
    hidden_spikes: torch.Tensor,
    potential: torch.Tensor,
    k: float,
    theta_margin: float = THETA_MARGIN,
) -> tuple[float, float, float, float, float]:
    """Return both sparsity axes and both manipulation checks for one batch.

    **Never report ``spikes_per_neuron`` alone.** It is the product ``(1 - s) * a`` of
    two variables that push the code in opposite directions, and reporting only the
    product is how the confound went unnoticed for the whole of v1 and v2. All five
    numbers are logged every epoch.

    Args:
        hidden_spikes: **2nd** hidden layer spikes, shape (B, C, 1, 1, T).
        potential: The membrane potential those spikes came from, same shape.
        k: The ceiling in force, for the ``over_k`` check.
        theta_margin: The floor in force, for the ``sub_threshold`` check.

    Returns:
        Tuple of (spikes_per_neuron, spikes_per_active_neuron ``a``,
        silent_fraction ``s``, over_k_fraction, sub_threshold_fraction). The last two
        say whether each penalty is actually binding.
    """
    counts = hidden_spikes.sum(dim=-1).detach()
    peak = potential.max(dim=-1).values.detach()
    n_pairs = max(1, counts.numel())
    total_spikes = counts.sum().item()

    silent = counts == 0
    n_active = max(1, int((~silent).sum().item()))
    return (
        total_spikes / n_pairs,
        total_spikes / n_active,
        silent.sum().item() / n_pairs,
        (counts > k).sum().item() / n_pairs,
        (peak < theta_margin).sum().item() / n_pairs,
    )


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


def build_loss_and_optimizer(net: SparseSHDNetwork,
                             lr: float = LEARNING_RATE) -> tuple:
    """Build the NumSpikes loss, Nadam optimizer, and LR scheduler.

    ``NumSpikes`` pins the *output* firing rate to fixed targets, which keeps the
    sparsity intervention localised to the hidden layer being constrained. Held
    identical to v1 and v2 so the arms and the two layer variants remain comparable.

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
    k: float,
    floor_strength: float,
    seed: int,
    epochs: int,
    ceiling_strength: float = CEILING_STRENGTH,
    warmup_epochs: int = WARMUP_EPOCHS,
    theta_margin: float = THETA_MARGIN,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    use_delay: bool = USE_DELAY,
    max_delay: int = MAX_DELAY,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[SparseSHDNetwork, dict]:
    """Train one cell of the factorial: ceiling ``k`` crossed with ``floor_strength``.

    Both penalties act on the **2nd** hidden layer. The forward pass is clean (no
    perturbation). After the warm-up the total loss is the NumSpikes task loss, plus
    ``ceiling_strength`` times the per-pair excess above ``k``, plus ``floor_strength``
    times the per-pair shortfall of peak membrane potential below ``theta_margin``.
    ``floor_strength = 0`` skips the floor term entirely, which is the control column.

    Best-model selection and early stopping use the *task* validation loss, so the
    saved checkpoint is chosen on the task rather than on how well the constraint is
    satisfied.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        k: Per-pair spike ceiling (the row axis, sets ``a``).
        floor_strength: Coefficient on the potential floor (the column axis, sets
            ``s``). 0 disables the floor.
        seed: Random seed (controls init and shuffle order).
        epochs: Maximum training epochs.
        ceiling_strength: Coefficient on the ceiling term, fixed across the grid.
        warmup_epochs: Clean epochs before either penalty engages.
        theta_margin: Peak potential the floor requires.
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        use_delay: Whether to use learnable delays.
        max_delay: Maximum delay in time steps.
        lr: Learning rate.
        patience: Early stopping patience (on task val loss).

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = SparseSHDNetwork(
        input_dim, hidden_units, num_classes, use_delay, max_delay,
    ).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    # Adaptive delay clamping state
    update1 = 0
    update2 = 0
    thea1 = max_delay
    thea2 = max_delay

    # Both sparsity axes are logged per epoch, not just their product -- this is
    # document 3's recommendation 3, unimplemented through the whole of v1 and v2.
    log = {
        "epoch": [],
        "train_loss": [],             # total (task + ceiling + floor)
        "train_task_loss": [],
        "train_ceiling_penalty": [],
        "train_floor_penalty": [],
        "train_rate": [],             # spikes / neuron / sample  = (1 - s) * a
        "train_rate_active": [],      # `a`: spikes / ACTIVE neuron / sample
        "train_silent": [],           # `s`: fraction of (sample, neuron) pairs silent
        "train_over_k": [],           # ceiling bind check: pairs firing above k
        "train_sub_threshold": [],    # floor bind check: pairs never reaching margin
        "val_loss": [],               # task only
        "val_acc": [],
        "delay_mean": [],
    }

    stat_names = ("total", "task", "ceiling", "floor", "rate", "rate_active",
                  "silent", "over_k", "sub_threshold")

    desc = f"Train L2 k={k:g} floor={floor_strength:g} seed={seed}"
    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=desc) as pbar:
        for epoch in range(epochs):
            penalty_on = epoch >= warmup_epochs

            net.train()
            batch_stats: dict[str, list[float]] = {name: [] for name in stat_names}

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs, hidden2, potential2 = net(x_batch, return_hidden=True)
                task_loss = loss_fn.numSpikes(outputs, target)

                ceiling_term = ceiling_penalty(hidden2, k)
                floor_term = (floor_penalty(potential2, theta_margin)
                              if floor_strength > 0
                              else torch.zeros((), device=device))
                if penalty_on:
                    loss = (task_loss
                            + ceiling_strength * ceiling_term
                            + floor_strength * floor_term)
                else:
                    loss = task_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rate, rate_active, silent, over_k, sub_threshold = (
                    hidden_activity_stats(hidden2, potential2, k, theta_margin))
                for name, value in zip(stat_names, (
                        loss.item(), task_loss.item(), ceiling_term.item(),
                        floor_term.item(), rate, rate_active, silent, over_k,
                        sub_threshold)):
                    batch_stats[name].append(value)
                pbar.update(1)

            # --- Adaptive delay clamping (SGD-delay only) ---
            if use_delay:
                if epoch <= 250:
                    net.clamp_delays(max_delay, max_delay)
                else:
                    update1 += 1
                    update2 += 1
                    for name, param in net.named_parameters():
                        if "delay1.delay" in name and update1 > 150:
                            sorted_ = torch.sort(
                                torch.floor(param.detach().flatten())
                            )[0]
                            thea1_val = torch.max(sorted_)
                            if sorted_[108] > (thea1_val - 5):
                                thea1 = int(thea1_val.item()) + 1
                                update1 = 0
                        elif "delay2.delay" in name and update2 > 150:
                            sorted_ = torch.sort(
                                torch.floor(param.detach().flatten())
                            )[0]
                            thea2_val = torch.max(sorted_)
                            if sorted_[108] > (thea2_val - 5):
                                thea2 = int(thea2_val.item()) + 1
                                update2 = 0
                    net.clamp_delays(thea1, thea2)

            # --- Validate (clean; task loss only) ---
            net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                    y_batch = y_batch.to(device).long()

                    target = torch.zeros(
                        (len(y_batch), num_classes, 1, 1, 1), device=device
                    )
                    target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                    outputs = net(x_batch)
                    val_loss += loss_fn.numSpikes(outputs, target).item()

                    pred = snn.predict.getClass(outputs)
                    correct += (pred.cpu() == y_batch.cpu()).sum().item()
                    total += len(y_batch)

            val_loss /= max(1, len(val_loader))
            val_acc = correct / max(1, total)
            means = {name: float(np.mean(values))
                     for name, values in batch_stats.items()}

            delays = net.get_delays()
            avg_delay = (
                np.mean([np.mean(d) for d in delays.values() if len(d) > 0])
                if delays else 0.0
            )

            log["epoch"].append(epoch)
            log["train_loss"].append(means["total"])
            log["train_task_loss"].append(means["task"])
            log["train_ceiling_penalty"].append(means["ceiling"])
            log["train_floor_penalty"].append(means["floor"])
            log["train_rate"].append(means["rate"])
            log["train_rate_active"].append(means["rate_active"])
            log["train_silent"].append(means["silent"])
            log["train_over_k"].append(means["over_k"])
            log["train_sub_threshold"].append(means["sub_threshold"])
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))
            log["delay_mean"].append(float(avg_delay))

            pbar.set_postfix(
                epoch=epoch + 1,
                task=f"{means['task']:.3f}",
                a=f"{means['rate_active']:.2f}",
                s=f"{means['silent']:.1%}",
                over=f"{means['over_k']:.1%}",
                acc=f"{val_acc:.2%}",
            )
            scheduler.step()

            # Early stopping on task val loss.
            if epoch < warmup_epochs:
                continue
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    name: value.clone() for name, value in net.state_dict().items()
                }
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


@torch.no_grad()
def evaluate_clean_and_activity(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    k: float,
    theta_margin: float = THETA_MARGIN,
) -> dict:
    """Measure clean test accuracy and the 2nd layer's two sparsity axes in one pass.

    The summary this feeds carries ``spikes_per_active_neuron`` and
    ``silent_fraction`` alongside ``spikes_per_neuron``, which v1's summaries did not
    — the omission that let the ``rho(a, s)`` confound go unnoticed through two whole
    versions.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.
        k: The ceiling the model was trained at, for the ``over_k`` check.
        theta_margin: The floor threshold, for the ``sub_threshold`` check.

    Returns:
        Dict with keys ``clean_acc``, ``firing_rate``, ``spikes_per_neuron``,
        ``spikes_per_active_neuron``, ``silent_fraction``, ``over_k_fraction``,
        ``sub_threshold_fraction``, ``count_std``, ``peak_potential_silent_median``,
        ``peak_potential_active_median`` — all of the 2nd hidden layer.
    """
    net.eval()
    correct = 0
    total = 0
    total_spikes = 0.0
    total_sq = 0.0
    total_slots = 0
    total_pairs = 0
    silent_pairs = 0.0
    over_k_pairs = 0.0
    sub_threshold_pairs = 0.0
    silent_peaks: list[torch.Tensor] = []
    active_peaks: list[torch.Tensor] = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
        y_batch = y_batch.to(device)

        outputs, hidden2, potential2 = net(x_batch, return_hidden=True)

        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)

        batch, channels, _, _, time = hidden2.shape
        counts = hidden2.sum(dim=-1).view(batch, channels)
        peak = potential2.max(dim=-1).values.view(batch, channels)
        silent = counts == 0

        total_spikes += counts.sum().item()
        total_sq += (counts ** 2).sum().item()
        total_slots += batch * channels * time
        total_pairs += batch * channels
        silent_pairs += silent.sum().item()
        over_k_pairs += (counts > k).sum().item()
        sub_threshold_pairs += (peak < theta_margin).sum().item()
        silent_peaks.append(peak[silent].cpu())
        active_peaks.append(peak[~silent].cpu())

    n_pairs = max(1, total_pairs)
    active_pairs = total_pairs - silent_pairs
    mean_count = total_spikes / n_pairs
    variance = max(0.0, total_sq / n_pairs - mean_count ** 2)
    silent_peak = torch.cat(silent_peaks)
    active_peak = torch.cat(active_peaks)

    return {
        "clean_acc": correct / max(1, total),
        "firing_rate": total_spikes / max(1, total_slots),
        "spikes_per_neuron": mean_count,
        "spikes_per_active_neuron": total_spikes / max(1.0, active_pairs),
        "silent_fraction": silent_pairs / n_pairs,
        "over_k_fraction": over_k_pairs / n_pairs,
        "sub_threshold_fraction": sub_threshold_pairs / n_pairs,
        "count_std": variance ** 0.5,
        "peak_potential_silent_median": (float(silent_peak.median())
                                         if silent_peak.numel() else float("nan")),
        "peak_potential_active_median": (float(active_peak.median())
                                         if active_peak.numel() else float("nan")),
    }


def report_manipulation_check(summary: dict[str, dict]) -> None:
    """Print whether the floor column actually separated the two selectivity levels.

    The probe's job is this and nothing else. Reported per ``k`` so that a floor that
    bites at one end of the row axis but not the other is visible rather than averaged
    away.

    Args:
        summary: The run summary, run tag -> recorded metrics.
    """
    by_k: dict[float, dict[float, float]] = {}
    for row in summary.values():
        by_k.setdefault(row["ceiling_k"], {})[row["floor_strength"]] = (
            row["silent_fraction"])

    print(f"\n{'k':>5} {'floor=0 silent':>15} {'floor=on silent':>16} "
          f"{'separation':>11}")
    for k in sorted(by_k):
        levels = by_k[k]
        floor_off = levels.get(0.0)
        floor_on = next(
            (value for strength, value in levels.items() if strength > 0), None)
        if floor_off is None or floor_on is None:
            print(f"{k:>5g} {'incomplete cell':>32}")
            continue
        separation = floor_off - floor_on
        flag = "OK" if separation >= MIN_SILENT_SEPARATION else "TOO SMALL"
        print(f"{k:>5g} {floor_off:>14.2%} {floor_on:>15.2%} "
              f"{separation:>+10.2%}  {flag}")
    print(f"\n  Target separation >= {MIN_SILENT_SEPARATION:.0%} at matched k. "
          f"If too small, raise FLOOR_STRENGTH.")


def report_design_acceptance(summary: dict[str, dict]) -> None:
    """Print the design acceptance check on the factorial itself.

    This is a check on the **design**, not on the network: it asks whether the
    factorial actually decorrelated the 2nd layer's two sparsity axes. If it did not,
    the regression the grid feeds cannot separate H1' from H2 and must not be read as
    though it could — the correct response is to report the failure and stop.

    Unlike v2's acceptance test this one is achievable: it asks for two variables to
    be decorrelated, not for one to be driven to zero.

    Args:
        summary: The run summary, run tag -> recorded metrics.
    """
    if len(summary) < 4:
        print("\n[design check] Fewer than 4 models — rho(a, s) not meaningful yet.")
        return

    a_axis = [row["spikes_per_active_neuron"] for row in summary.values()]
    s_axis = [row["silent_fraction"] for row in summary.values()]
    result = spearmanr(a_axis, s_axis)
    verdict = ("PASS — the factorial broke the confound"
               if abs(result.statistic) <= MAX_AXIS_CORRELATION
               else "FAIL — a and s are still confounded; the regression is NOT "
                    "interpretable as a test of H1'")
    print(f"\n{'=' * 70}")
    print(f"DESIGN ACCEPTANCE CHECK — LAYER {TARGET_LAYER} (n={len(summary)})")
    print(f"  rho(a, s) = {result.statistic:+.3f}  (p = {result.pvalue:.4g}), "
          f"threshold |rho| <= {MAX_AXIS_CORRELATION}")
    print(f"  {verdict}")
    print(f"  Observational baseline to beat: rho(a, s) at layer {TARGET_LAYER} across "
          f"v1's gradient in this arm was {V1_LAYER2_AXIS_CORRELATION:+.3f}.")
    print("  Note the sign: at layer 2 the two axes move TOGETHER, where at layer 1 "
          "they opposed (-0.455).")
    print(f"{'=' * 70}")


def run_milestone() -> None:
    """Train the (k, floor_strength, seed) factorial and record both sparsity axes."""
    ceiling_k, floor_strengths, seeds, epochs, warmup_epochs = resolve_run_config()

    n_models = len(ceiling_k) * len(floor_strengths) * len(seeds)
    print(f"{'#' * 70}")
    print(f"# Sparse-network milestone v3 (training): SHD whole, SGD-delay, "
          f"LAYER {TARGET_LAYER}")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs} | warmup={warmup_epochs}")
    print(f"# mechanism = per-pair ceiling relu(count-k) + potential floor "
          f"relu(margin - max_t u), both on hidden layer {TARGET_LAYER}")
    print(f"# ceiling k: {ceiling_k}  (strength {CEILING_STRENGTH:g})")
    print(f"# floor strengths: {floor_strengths}  (margin {THETA_MARGIN:g}, "
          f"theta {LIF_PARAMS['theta']})")
    print(f"# seeds: {seeds}  ->  {n_models} models")
    print(f"{'#' * 70}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data and build loaders once; the split (and hence the test set) is fixed, so
    # all models are comparable — including against v1 and against the 1st-layer grid.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    train_loader, val_loader, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=42,
    )

    summary: dict[str, dict] = {}

    for k in ceiling_k:
        for floor_strength in floor_strengths:
            for seed in seeds:
                run_tag = (
                    f"sparse_{DATASET_KEY}_delay_{VERSION_TAG}_"
                    f"k{k:g}_floor{floor_strength:g}_seed{seed}{RUN_SUFFIX}"
                )
                print(f"\n{'=' * 60}")
                print(f"  Training {run_tag}")
                print(f"{'=' * 60}")

                net, training_log = train_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    k=k,
                    floor_strength=floor_strength,
                    seed=seed,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                )

                ckpt_path = DATA_DIR / f"{run_tag}.pt"
                torch.save(net.state_dict(), ckpt_path)
                print(f"Model saved to {ckpt_path}")

                metrics = evaluate_clean_and_activity(net, test_loader, k)
                print(
                    f"  clean_acc={metrics['clean_acc']:.4f} | "
                    f"a={metrics['spikes_per_active_neuron']:.2f} | "
                    f"s={metrics['silent_fraction']:.2%} | "
                    f"sp/neuron={metrics['spikes_per_neuron']:.2f} | "
                    f"over_k={metrics['over_k_fraction']:.2%}"
                )

                summary[run_tag] = {
                    "target_layer": int(TARGET_LAYER),
                    "ceiling_k": float(k),
                    "ceiling_strength": float(CEILING_STRENGTH),
                    "floor_strength": float(floor_strength),
                    "theta_margin": float(THETA_MARGIN),
                    "warmup_epochs": int(warmup_epochs),
                    "epochs": int(epochs),
                    "seed": int(seed),
                    **{name: float(value) for name, value in metrics.items()},
                }

                log_path = LOG_DIR / f"{run_tag}_training_log.json"
                with open(log_path, "w") as fp:
                    json.dump({name: [float(v) for v in values]
                               for name, values in training_log.items()},
                              fp, indent=2)
                print(f"  Training log saved to {log_path}")

    summary_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_delay_{VERSION_TAG}_train_summary{RUN_SUFFIX}.json"
    )
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Final table. Both axes always, never their product alone.
    print(
        f"\n{'run_tag':<60} {'clean_acc':>9} {'a':>7} {'s':>8} {'sp/neu':>8} "
        f"{'over_k':>8} {'sub_th':>8}"
    )
    for run_tag, row in summary.items():
        inflated = "" if row["spikes_per_neuron"] <= NATURAL_SPIKES_PER_NEURON else " !"
        print(
            f"{run_tag:<60} {row['clean_acc']:>9.4f} "
            f"{row['spikes_per_active_neuron']:>7.2f} "
            f"{row['silent_fraction']:>8.2%} {row['spikes_per_neuron']:>8.2f} "
            f"{row['over_k_fraction']:>8.2%} "
            f"{row['sub_threshold_fraction']:>8.2%}{inflated}"
        )
    print(f"\n  ! = spikes_per_neuron above layer {TARGET_LAYER}'s natural "
          f"{NATURAL_SPIKES_PER_NEURON} in this arm — v2.2's failure mode.")

    report_manipulation_check(summary)
    report_design_acceptance(summary)


if __name__ == "__main__":
    run_milestone()
