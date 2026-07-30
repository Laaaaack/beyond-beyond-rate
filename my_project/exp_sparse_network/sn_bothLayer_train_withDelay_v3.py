"""Both-layer milestone v3 (training), WITH-DELAY arm: the a x s dissociation factorial.

Third sibling of [sn_train_withDelay_v3.py](sn_train_withDelay_v3.py) (penalties on the
**1st** hidden layer) and
[sn_2ndLayer_train_withDelay_v3.py](sn_2ndLayer_train_withDelay_v3.py) (penalties on the
**2nd**). Same network, data, splits, optimiser, schedule and logging; the only
difference is that both penalties are charged against **both hidden layers
simultaneously**, so the manipulated quantity is the sparsity of the hidden network
rather than the sparsity of one layer inside it.

Why constrain both layers at once
----------------------------------
The two single-layer grids each answer "what happens when *this* layer is made sparse".
Neither answers "what happens when the *network* is made sparse", and the layer-2
baseline measurement says those are not the same question. Measured across v1's 12
delay checkpoints — where only layer 1 was ever penalised:

===============================  ==========  ==============================
correlation across v1's gradient  value       what it says
===============================  ==========  ==============================
``rho(a1, s1)``                   **-0.455**  layer 1's own confound
``rho(a2, s2)``                   **+0.427**  layer 2 carries its own
                                              confound, running the *other*
                                              way
``rho(s1, s2)``                   **+0.203**  essentially unrelated: the
                                              layer-1 penalty does not make
                                              the network selective
``rho(a1, a2)``                   **+0.839**  the two layers' activity levels
                                              are near-inseparable
``rho(a_net, s_net)`` pooled      **+0.042**  network-wide, the two axes are
                                              already decorrelated here
===============================  ==========  ==============================

Two things follow, and they differ from the no-delay arm's story:

- **The pooled acceptance check is not the hurdle in this arm.** At ``+0.042`` the
  network-wide axes are already decorrelated observationally, where in the no-delay arm
  they sit at ``-0.879``. What this design buys here is therefore not decorrelation but
  **manipulation**: the floor-on vs floor-off contrast at matched ``a`` is a controlled
  comparison, where v1's gradient was only ever a correlation. That is the same reason
  the 1st-layer factorial was worth running in this arm at ``rho = -0.455``.
- **The per-layer figures are power statements, not clean bills of health.** ``-0.455``
  and ``+0.427`` on n = 12 at p = .14 / .17 are inside the ``|rho| <= 0.5`` threshold by
  a margin narrower than their own uncertainty. The acceptance check is still run at all
  three levels.

The fourth row is what this design deliberately does **not** fix — see below.

What this design can and cannot separate
-----------------------------------------
It manipulates **network-wide** ``a`` and **network-wide** ``s``, and it deliberately
does **not** separate layer 1's ``a`` from layer 2's ``a``. Both knobs act on both
layers, so ``a1`` and ``a2`` are moved together by construction — exactly as
``rho(a1, a2) = +0.839`` already is observationally. That is a property of the question,
not a defect: the layer-1-only and layer-2-only grids are what attribute an effect to a
particular layer, and this grid is what tests the effect of sparsifying the network.

The design acceptance check is therefore run at three levels — layer 1, layer 2 and
pooled — and the cross-layer correlations are printed as **diagnostics that are expected
to be high**, never as a pass/fail.

The two terms
-------------
Unchanged in form. Each is now charged **once per layer**, at the same coefficient the
single-layer scripts used, so each layer receives exactly the pressure its own sibling
script applied and the three grids stay comparable cell by cell::

    ceiling_term = relu(count1 - k1).mean() + relu(count2 - k2).mean()
    floor_term   = relu(margin - max_t u1).mean() + relu(margin - max_t u2).mean()

1. **Ceiling — sets ``a``.** ``relu(count - k)`` charged per ``(sample, neuron)`` pair,
   not on the batch mean. Per-pair is the whole point: a batch-mean hinge lets a neuron
   satisfy the penalty by going *silent* on some samples and firing freely on others,
   which is exactly how ``a`` and ``s`` became confounded in v1.
2. **Floor — sets ``s``.** ``relu(THETA_MARGIN - max_t u(t))`` on the **membrane
   potential**, not on the spike count. Silent pairs sit at a median peak potential of
   0.00-2.13 (layer 1) and 0.65-1.26 (layer 2) against ``theta = 10``; a count-based
   penalty reaches them only through SLAYER's surrogate gradient, which carries no
   signal there. ``du/dW`` is an ordinary convolution gradient at both layers.

The floor's **strength is the column knob**: 0 in one column, ``FLOOR_STRENGTH`` in the
other, applied to both layers together. ``k`` is the row knob. Crossing them is the
factorial.

The row axis: one knob, two per-layer budgets
----------------------------------------------
``k`` is set per layer rather than shared, because in this arm the two layers
emphatically do not run at the same natural rate: the measured natural ``a`` is
3.82-15.47 at layer 1 against **20.71-30.44** at layer 2, roughly a factor of two. A
shared absolute budget would be a mild cut for layer 1 and a savage one for layer 2 in
the same cell.
``CEILING_K`` lists layer 1's levels; layer 2's are ``CEILING_K_LAYER2_RATIO = 2.0``
times them, so a row is an equal *relative* cut at both layers.

This also settles a question the 2nd-layer script had to solve with an extra grid row.
That script kept a shared budget and added ``k = 16`` to anchor the top of layer 2's
axis near its natural rate, at a cost of 4 extra models (~9 h). Here the ratio does the
same job inside 4 rows: the top row is ``k1 = 8, k2 = 16``, which sits just under each
layer's own natural range. 16 models, not 20.

Where the delays sit relative to the constrained layers
--------------------------------------------------------
``delay1`` is **between** the two constrained layers — it acts on layer 1's spikes
before ``fc2`` sees them. So it is downstream of the layer-1 penalty and upstream of the
layer-2 one, and it shapes the very activity the layer-2 terms charge. That is why this
arm's layer-2 temporal support runs to bin 159 where layer 1's stops at 87, and it is
the reason any eval-time probe of layer 2 must apply ``delay1`` (see
``v3_analysis/layer2_baseline.py`` for the reference implementation). ``delay2`` is
downstream of both and only feeds the output.

Where the floor's layer-2 reachability edge goes
-------------------------------------------------
``d potential2 / d W2 = psp(delay1(hidden1))`` is surrogate-free, **but it is exactly
zero for any sample whose entire layer 1 is silent** — such a pair is reachable only
back through ``fc1``'s surrogate. Currently 0.00% of samples across all 27 v1
checkpoints.

This design changes the risk in both directions, and both are worth stating:

- **In the floor-ON column it is now actively guarded.** The layer-1 floor forbids
  layer-1 silence, which is the precondition for the edge case. Neither single-layer
  script had that protection: the 2nd-layer script left layer 1 unconstrained.
- **In the floor-OFF column it is worse.** The ceiling now pushes *both* layers toward
  silence with nothing opposing either, so emptying layer 1 is more reachable here than
  in any previous script. That corner — ``k`` at its lowest, floor 0 — is the one the
  probe exists to watch.

What is calibrated, and what is only inherited
-----------------------------------------------
**Measured for this arm, per layer** — ``NATURAL_SPIKES_PER_NEURON`` (11.66 at layer 1,
23.93 at layer 2), the per-layer natural ``a`` that sets ``CEILING_K_LAYER2_RATIO``, and
the three observational ``rho(a, s)`` baselines the acceptance check reports against.

**Inherited and NOT yet validated in this configuration** — ``CEILING_STRENGTH = 10``,
``FLOOR_STRENGTH = 1`` and ``WARMUP_EPOCHS = 20``. Every one was calibrated against a
*single* layer's activity, and this script charges both penalties twice. Two specific
reasons to expect them to behave differently here:

- **``fc1`` now receives ceiling pressure from two directions** — directly from the
  layer-1 term, and indirectly from the layer-2 term through the surrogate. The same
  coefficient therefore buys more pressure on layer 1 than the 1st-layer script's did.
- **The absorbing all-silent state is easier to reach**, per the floor-OFF note above,
  and ``WARMUP_EPOCHS`` is the only thing guarding it.

Run ``QUICK_TEST`` before committing to the grid. See
``sparse_network_bothLayer_test_progress.md``.

Reading the probe
-----------------
``QUICK_TEST`` runs the 4 corners ``k in {1, 8} x floor in {0, F}`` at 400 epochs (layer
2 seeing ``{2, 16}``). It validates the **manipulation**, which is a property of the
constraint and readable early. Check, in priority order:

1. **Both layers survive the ``k = 1``, floor = 0 corner.** ``val_acc`` at chance (5%)
   with ``silent -> 100%`` at either layer means the warm-up is too short. This check is
   new to this script and comes first because it is the one failure that cannot be
   recovered from;
2. ``silent_fraction`` roughly constant as ``k`` varies *within* a column, at each
   layer;
3. ``silent_fraction`` differing by **>= 20 points** *across* columns at matched ``k``,
   **at both layers** — if the floor bites at one layer and not the other, the column
   knob is not doing what the design needs;
4. ``spikes_per_neuron`` not inflated beyond each layer's natural rate (11.66 / 23.93);
5. clean accuracy comfortably above chance in every corner.

**Do not read achieved firing rates or accuracy levels off the probe.** Every v1 model
was still improving at epoch 1250 (best val loss at epochs 1168-1243), so 400 epochs is
a different regime, not a scaled-down one. Calibrate nothing from it.

Design acceptance check
------------------------
After the full grid, ``rho(a, s)`` is printed at layer 1, at layer 2 and pooled over
both. **If |rho| > 0.5 at the level being analysed, the factorial has failed to break
that confound** and the regression it feeds is not interpretable there — report it and
stop. The observational baselines this must improve on in this arm are **-0.455**
(layer 1), **+0.427** (layer 2) and **+0.042** (network); note that the last already
passes, so in this arm the value added is the controlled contrast rather than
decorrelation.

Outputs are tagged ``_v3L12_`` and cannot collide with the 1st-layer grid (``v3``), the
2nd-layer grid (``v3L2``), v1, or any v2 probe.

This is a **training-only** script. The perturbations that measure temporal processing
are applied **only at evaluation**. Note that the existing eval scripts under
``exp_sparse_network/{jitter,shift,shd,deletion}/`` all inject at the **1st** hidden
layer and confine relocation/jitter to ``[0, 88)``; measuring this experiment needs them
run at both layers, at that layer's own window (``[0, 88)`` for layer 1 and
``[0, 160)`` for layer 2 in this arm). See:

- ``my_project/docs/progress/sparse_network_bothLayer_test_progress.md`` — this
  experiment's design, calibration and status.
- ``my_project/docs/progress/sparse_network_test_progress_v3.md`` — the 1st-layer v3
  design, Phase 0 and Phase 1 result.
- ``my_project/docs/progress/sparse_network_2ndLayer_test_progress_v3.md`` — the
  2nd-layer sibling's calibration.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.

What this script produces, per ``(k, floor_strength, seed)``:

- a checkpoint ``sn_data/sparse_whole_delay_v3L12_k{k}_floor{f}_seed{seed}.pt``;
- a training-curve log ``sn_log/..._training_log.json`` carrying the trajectories of
  both sparsity axes **at both layers**, plus the network-wide pool;
- a row in ``sn_log/sparse_whole_delay_v3L12_train_summary.json`` recording clean test
  accuracy and every axis at every level.
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
# Corner probe. True runs k in {1, 8} x floor in {0, FLOOR_STRENGTH} at 400 epochs,
# one seed -- 4 models, ~3 h on this arm. Its only job is to validate the manipulation;
# see "Reading the probe" in the module docstring. Probe artifacts carry RUN_SUFFIX so
# they can never clobber a real run.
#
# Leave this True until the three inherited constants have been checked in the
# BOTH-LAYER configuration. None of them was calibrated with two penalties running.
QUICK_TEST: bool = True

# Suffix appended to every output name (checkpoints, per-model logs, summary) when
# running a probe, so a QUICK_TEST probe can never overwrite real-run artifacts that
# share the same (k, floor_strength, seed). Empty for the real run.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Marks every artifact of the both-layer factorial. Distinct from the 1st-layer tag
# ("v3") and the 2nd-layer tag ("v3L2"), so all three grids can sit side by side in
# sn_data/ and sn_log/ without any chance of one overwriting another -- they share
# dataset, arm, k, floor and seed, and would otherwise produce identical filenames.
VERSION_TAG: str = "v3L12"

# --- Milestone scope ---
DATASET_KEY: str = "whole"
INPUT_DIM: int = 700          # SHD whole
USE_DELAY: bool = True        # SGD-delay
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# Which hidden layers the two penalties act on. This is the whole point of the script;
# it exists as a named constant so that the many places that say "both layers" in prose
# can be checked against one value.
TARGET_LAYERS: tuple[int, ...] = (1, 2)

# --- The factorial ---
# Row knob: the per-pair spike ceiling. This is the H1 axis -- it sets `a`, spikes per
# ACTIVE neuron, by making counts above k costly for every (sample, neuron) pair
# individually. These are LAYER 1's levels; layer 2's are these times the ratio below.
#
# Layer 1's natural `a` in this arm is 3.82-15.47 (measured on v1's 12 checkpoints), so
# k = 8 sits inside the top of that range and k = 1 is an order of magnitude under it.
#
# The 2nd-layer script needed a 5th row (k = 16) to anchor the top of layer 2's axis,
# because it charged a shared absolute budget against a layer whose natural `a` is
# 20.71-30.44. Here the per-layer ratio does that job instead: the top row is
# k1 = 8, k2 = 16, just under each layer's own natural range. 16 models rather than 20.
CEILING_K: list[float] = [1.0, 2.0, 4.0, 8.0]

# Layer 2's ceiling as a multiple of layer 1's, so that one row knob delivers an equal
# RELATIVE cut at both layers rather than an equal absolute budget.
#
# Set from the measured natural `a` per layer in THIS arm: 3.82-15.47 at layer 1 against
# 20.71-30.44 at layer 2 -- layer 2 runs about twice as hard, hence 2.0. This constant
# is the arm's own, not a shared default: the no-delay arm measures 3.06-11.49 against
# 7.49-12.46, where the ranges overlap and the ratio is 1.0. Carrying this value across
# would cut that arm's layer 2 twice as loosely as its layer 1.
CEILING_K_LAYER2_RATIO: float = 2.0

# Column knob: the membrane-potential floor, applied to BOTH layers at the same
# strength. 0 lets neurons fall silent freely (high `s`); nonzero forbids it (low `s`).
# This is the H2 axis, and it is what makes the floor-on vs floor-off contrast at
# matched `k` a genuine controlled comparison rather than a correlation.
FLOOR_STRENGTH: list[float] = [0.0, 1.0]

SEEDS: list[int] = [42, 43]

# Coefficient on the ceiling term, applied to EACH layer's term separately (the two are
# summed). Charging each layer at the coefficient its own single-layer script used is
# what keeps the three grids comparable cell by cell. Fixed across the whole grid so
# that `k` is the only thing varying along the row axis.
#
# INHERITED FROM THE SINGLE-LAYER ARMS, NOT YET VALIDATED WITH BOTH PENALTIES RUNNING.
# At layer 1 in this arm, 3.0 (v2's value) proved too weak -- it left `a` spanning only
# 1.24-1.42x across k = 1 -> 8, because the penalty reaches equilibrium far above k and
# stops pushing -- and 10.0 roughly doubled the row-axis span at an accuracy cost that
# stayed usable:
#
#   k=1     a      s      sp/neu   over_k   clean_acc     (LAYER 1 ONLY, this arm)
#   ---------------------------------------------------
#   str=3,  floor=0   6.17  62.7%   2.30    27%     .818
#   str=3,  floor=1   4.90  11.0%   4.36    67%     .824
#   str=10, floor=0   4.00  72.1%   1.12    17%     .685
#   str=10, floor=1   2.66  22.6%   2.06    41%     .771
#
# Why it may not transfer here, and the two reasons point in OPPOSITE directions:
#   - fc1 now receives ceiling pressure twice, directly from the layer-1 term and
#     through the surrogate from the layer-2 term, so layer 1 is squeezed harder than
#     the 1st-layer script squeezed it at the same coefficient -- an argument for a
#     SMALLER value;
#   - layer 2 in this arm starts about twice as dense as layer 1, so its term has much
#     further to push even at the scaled k, and at layer 1 even strength 10 left 41% of
#     pairs above k at k = 1 -- an argument for a LARGER one.
# The probe's two `over_k` columns are what say whether it binds at each layer. If one
# stalls high and the other does not, that asymmetry is the thing to fix, not the
# overall level.
CEILING_STRENGTH: float = 10.0

# The floor's threshold on peak membrane potential, as a multiple of theta = 10. 1.1 x
# theta so that a satisfied pair is not left sitting exactly on the boundary, where a
# small weight change would drop it back into silence. Same value at both layers.
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
# INHERITED, AND THIS SCRIPT IS THE MOST EXPOSED OF THE THREE. The ceiling now pushes
# both layers toward silence at once with nothing opposing either in the floor-off
# column, and emptying layer 1 additionally kills the layer-2 floor's gradient path
# (d potential2 / d W2 = psp(delay1(hidden1)) = 0), so the two failures compound. The
# probe's k = 1, floor = 0 corner is the cell that would show it. If val_acc sits at
# chance (5%) with silent -> 100% at either layer, raise this before anything else.
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
# Separation in silent_fraction the floor column must produce at matched k. Required at
# BOTH layers here, not just on average: a floor that bites at one layer and not the
# other leaves the column knob doing half its job.
MIN_SILENT_SEPARATION: float = 0.20

# The natural firing rate of each hidden layer in this arm, measured on v1's 12
# checkpoints (layer 1 spans 2.12-11.66, layer 2 spans 14.12-23.93; these are the
# densest). The ceiling must not drive either layer ABOVE its own figure; v2.2 inflated
# layer 1 from ~6.3 to 12.5-21.9 and stopped being an instance of "sparser-activity
# networks" as a result.
#
# These are per-LAYER and per-ARM figures. The no-delay arm measures 7.85 and 5.53 --
# note that its layer 2 is *sparser* than its layer 1 while this arm's is twice as dense
# -- so the constant must never be carried across either boundary.
NATURAL_SPIKES_PER_NEURON: dict[int, float] = {1: 11.66, 2: 23.93}

# The design acceptance check on the factorial itself.
MAX_AXIS_CORRELATION: float = 0.5

# What the factorial has to improve on: rho(a, s) across v1's gradient in this arm, at
# each level the acceptance check reports.
#
# Unlike the no-delay arm, the network-wide baseline here (+0.042) ALREADY passes the
# threshold, and the two per-layer figures sit inside it on n = 12 at p = .14 / .17 --
# margins narrower than their own uncertainty. So in this arm the design's value is the
# controlled floor-on vs floor-off contrast at matched `a`, not decorrelation. That is
# the same reason the 1st-layer factorial was worth running here at -0.455.
V1_AXIS_CORRELATION: dict[str, float] = {
    "layer1": -0.455,
    "layer2": +0.427,
    "network": +0.042,
}

# Cross-layer correlations across v1's gradient, reported as DIAGNOSTICS rather than as
# acceptance criteria. This design moves both layers with one pair of knobs, so it is
# expected to leave rho(a1, a2) high -- that is what "constrain the network" means, and
# separating the layers is what the two single-layer grids are for. rho(s1, s2) at
# +0.203 says v1's layer-1-only penalty left the two layers' selectivity essentially
# unrelated; this design should pull it clearly positive.
V1_CROSS_LAYER_CORRELATION: dict[str, float] = {"a": +0.839, "s": +0.203}


def layer2_ceiling(k: float, ratio: float = CEILING_K_LAYER2_RATIO) -> float:
    """Return layer 2's spike ceiling for a given layer-1 ceiling.

    One row knob drives both layers, scaled by the ratio of their measured natural
    firing rates so that a row is an equal *relative* cut rather than an equal absolute
    budget. See ``CEILING_K_LAYER2_RATIO``.

    Args:
        k: Layer 1's per-pair spike ceiling (a level of ``CEILING_K``).
        ratio: Layer 2's ceiling as a multiple of layer 1's.

    Returns:
        Layer 2's per-pair spike ceiling.
    """
    return k * ratio


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
    """2-hidden-layer SLAYER SNN with learnable delays, for the both-layer factorial.

    The parameter set is identical to v1's class of the same name — ``fc1``/``fc2``/
    ``fc3`` weight-norm parameters plus ``delay1``/``delay2`` and nothing else — so
    these checkpoints load directly into the existing eval scripts and into
    ``v3_analysis/``. There is no truncation, so the forward pass is v1's exactly and a
    checkpoint needs no special handling anywhere downstream.

    The one difference from the single-layer scripts is which tensors ``forward`` hands
    back: **both** hidden layers' spikes and the membrane potentials they were
    thresholded from, because both penalties are now charged at both layers. All four
    tensors are computed by the forward pass regardless; returning them costs nothing.

    Note where the delays sit relative to the constrained layers. ``delay1`` is
    **between** them — downstream of the layer-1 penalty, upstream of the layer-2 one —
    so it shapes the activity the layer-2 terms charge, and it is why this arm's layer-2
    temporal support runs to bin 159 rather than layer 1's 87. ``delay2`` is downstream
    of both and only feeds the output.

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

        # delay1 sits between the two constrained layers; delay2 between the fc2 spike
        # and fc3.
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

    def _first_hidden(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input -> fc1 -> spike, returning the potential as well as the spikes.

        ``d potential1 / d W1 = psp(x)`` is a convolution of the *input*, so it never
        vanishes however far below threshold a pair sits. That is what lets the floor
        reach deeply silent pairs at this layer.

        Args:
            x: Prepared input spike trains, shape (B, C, 1, 1, T).

        Returns:
            Tuple of (1st hidden layer spikes, its membrane potential).
        """
        potential1 = self.fc1(self.slayer.psp(x))
        return self.slayer.spike(potential1), potential1

    def _second_hidden(
        self,
        hidden1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden1 -> (delay1) -> fc2 -> spike, returning the potential too.

        ``d potential2 / d W2 = psp(delay1(hidden1))`` is likewise an ordinary
        convolution gradient — no surrogate — with the one caveat that it is zero for a
        sample whose entire layer 1 is silent. In this design the layer-1 floor guards
        that precondition in the floor-on column; the floor-off column is where it is
        worth watching.

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
            return_hidden: If True, also return **both** hidden layers' spikes and the
                membrane potentials they were thresholded from, so the caller can apply
                both penalties at both layers.

        Returns:
            The output spike tensor, or
            ``(output, hidden1, potential1, hidden2, potential2)`` if ``return_hidden``
            is True.
        """
        x = self._prepare_input(x)
        hidden1, potential1 = self._first_hidden(x)
        hidden2, potential2 = self._second_hidden(hidden1)
        out = self._output(hidden2)
        if return_hidden:
            return out, hidden1, potential1, hidden2, potential2
        return out

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

    Called once per hidden layer, with that layer's own ``k``.

    Args:
        hidden_spikes: One hidden layer's spikes, shape (B, C, 1, 1, T).
        k: Spikes per (sample, neuron) above which the penalty engages, for that layer.

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
    v1's checkpoints, silent (sample, neuron) pairs sit at a median peak potential of
    0.00-2.13 at layer 1 and 0.65-1.26 at layer 2 against ``theta = 10`` in this arm —
    they are not marginally sub-threshold, they receive essentially no drive at all. A
    count-based penalty reaches them only through SLAYER's surrogate gradient, which at
    ``u ~ 0`` against ``theta = 10`` carries no signal; that dead zone silenced v1's
    no-delay arm for 26 epochs, left v2/v2.1 at 21-40% silent, and defeated v2.2's
    ``relu(k - count)`` even at strength 10.

    Both layers' potentials are ordinary convolution gradients in their own weights, so
    neither term needs the surrogate — with the layer-2 caveat noted on
    ``_second_hidden``.

    Asking for one suprathreshold crossing rather than for more spikes is also what
    should keep raw firing off v2.2's blow-up: it requests the minimum sufficient
    condition for the neuron to be non-silent.

    Called once per hidden layer.

    Args:
        potential: One hidden layer's membrane potential, shape (B, C, 1, 1, T).
        theta_margin: Peak potential each pair must reach. Default ``1.1 x theta``, so
            a satisfied pair is not left sitting on the boundary.

    Returns:
        Scalar penalty tensor (mean shortfall below ``theta_margin``).
    """
    return torch.relu(theta_margin - potential.max(dim=-1).values).mean()


def layer_activity_stats(
    hidden_spikes: torch.Tensor,
    potential: torch.Tensor,
    k: float,
    theta_margin: float = THETA_MARGIN,
) -> dict[str, float]:
    """Return both sparsity axes and both manipulation checks for one layer, one batch.

    **Never report ``spikes_per_neuron`` alone.** It is the product ``(1 - s) * a`` of
    two variables that push the code in opposite directions, and reporting only the
    product is how the confound went unnoticed for the whole of v1 and v2. Every number
    below is logged every epoch, at each layer.

    Args:
        hidden_spikes: One hidden layer's spikes, shape (B, C, 1, 1, T).
        potential: The membrane potential those spikes came from, same shape.
        k: The ceiling in force at this layer, for the ``over_k`` check.
        theta_margin: The floor in force, for the ``sub_threshold`` check.

    Returns:
        Dict with keys ``rate`` (spikes per neuron), ``rate_active`` (``a``), ``silent``
        (``s``), ``over_k``, ``sub_threshold`` and ``n_pairs``. The two fraction keys
        say whether each penalty is actually binding; ``n_pairs`` lets
        :func:`pool_layer_stats` weight the layers exactly.
    """
    counts = hidden_spikes.sum(dim=-1).detach()
    peak = potential.max(dim=-1).values.detach()
    n_pairs = max(1, counts.numel())
    total_spikes = counts.sum().item()

    silent = counts == 0
    n_active = max(1, int((~silent).sum().item()))
    return {
        "rate": total_spikes / n_pairs,
        "rate_active": total_spikes / n_active,
        "silent": silent.sum().item() / n_pairs,
        "over_k": (counts > k).sum().item() / n_pairs,
        "sub_threshold": (peak < theta_margin).sum().item() / n_pairs,
        "n_pairs": float(n_pairs),
    }


def pool_layer_stats(per_layer: dict[int, dict[str, float]]) -> dict[str, float]:
    """Pool per-layer statistics into the network-wide sparsity axes.

    This is the level the both-layer factorial actually manipulates, so it is the level
    the design acceptance check and the regression are read at. Pooling is over
    ``(sample, neuron)`` pairs across both layers, weighted by each layer's pair count,
    which makes ``rate_active`` the exact network-wide spikes-per-active-neuron rather
    than an average of two ratios::

        rate_active_net = (spikes1 + spikes2) / (active1 + active2)

    Args:
        per_layer: Layer index -> the dict returned by :func:`layer_activity_stats`.

    Returns:
        Dict with the pooled ``rate``, ``rate_active`` and ``silent``. The two binding
        checks are deliberately absent: ``over_k`` is measured against a different ``k``
        at each layer, so a pooled version would not mean anything.
    """
    total_pairs = sum(stats["n_pairs"] for stats in per_layer.values())
    total_spikes = sum(stats["rate"] * stats["n_pairs"] for stats in per_layer.values())
    silent_pairs = sum(stats["silent"] * stats["n_pairs"]
                       for stats in per_layer.values())
    active_pairs = max(1.0, total_pairs - silent_pairs)
    return {
        "rate": total_spikes / max(1.0, total_pairs),
        "rate_active": total_spikes / active_pairs,
        "silent": silent_pairs / max(1.0, total_pairs),
    }


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
    sparsity intervention localised to the hidden layers being constrained. Held
    identical to v1 and v2 so the arms and the three layer variants remain comparable.

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


def build_training_log() -> dict[str, list]:
    """Return the empty per-epoch log, with both axes at both layers and pooled.

    v1's logs recorded only ``spikes_per_neuron`` for one layer, which is how the
    ``rho(a, s)`` confound went unnoticed for two whole versions. This log carries
    ``a``, ``s``, and both binding checks separately for layer 1 and layer 2, plus the
    network-wide pool that the factorial is actually manipulating.

    Returns:
        Dict of empty lists, one per logged quantity.
    """
    log: dict[str, list] = {
        "epoch": [],
        "train_loss": [],                 # total (task + both ceilings + both floors)
        "train_task_loss": [],
        "val_loss": [],                   # task only
        "val_acc": [],
        "delay_mean": [],
    }
    for layer in TARGET_LAYERS:
        log[f"train_ceiling_penalty_l{layer}"] = []
        log[f"train_floor_penalty_l{layer}"] = []
        log[f"train_rate_l{layer}"] = []          # spikes / neuron  = (1 - s) * a
        log[f"train_rate_active_l{layer}"] = []   # `a`: spikes / ACTIVE neuron
        log[f"train_silent_l{layer}"] = []        # `s`: fraction of pairs silent
        log[f"train_over_k_l{layer}"] = []        # ceiling bind check
        log[f"train_sub_threshold_l{layer}"] = []  # floor bind check
    log["train_rate_net"] = []
    log["train_rate_active_net"] = []
    log["train_silent_net"] = []
    return log


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
    k_layer2_ratio: float = CEILING_K_LAYER2_RATIO,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    use_delay: bool = USE_DELAY,
    max_delay: int = MAX_DELAY,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[SparseSHDNetwork, dict]:
    """Train one cell of the factorial: ceiling ``k`` crossed with ``floor_strength``.

    Both penalties act on **both** hidden layers. After the warm-up the total loss is
    the NumSpikes task loss, plus ``ceiling_strength`` times each layer's per-pair
    excess above its own ceiling, plus ``floor_strength`` times each layer's per-pair
    shortfall of peak membrane potential below ``theta_margin``. Each layer is charged
    at the full coefficient rather than at half, so it receives exactly the pressure its
    own single-layer script applied. ``floor_strength = 0`` skips both floor terms,
    which is the control column.

    Layer 2's ceiling is ``k * k_layer2_ratio``, so one row knob delivers an equal
    relative cut at both layers.

    Best-model selection and early stopping use the *task* validation loss, so the
    saved checkpoint is chosen on the task rather than on how well the constraints are
    satisfied.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        k: Layer 1's per-pair spike ceiling (the row axis, sets ``a``).
        floor_strength: Coefficient on each potential floor (the column axis, sets
            ``s``). 0 disables both floors.
        seed: Random seed (controls init and shuffle order).
        epochs: Maximum training epochs.
        ceiling_strength: Coefficient on each ceiling term, fixed across the grid.
        warmup_epochs: Clean epochs before any penalty engages.
        theta_margin: Peak potential the floors require, at both layers.
        k_layer2_ratio: Layer 2's ceiling as a multiple of layer 1's.
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

    ceilings = {1: k, 2: layer2_ceiling(k, k_layer2_ratio)}

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    # Adaptive delay clamping state
    update1 = 0
    update2 = 0
    thea1 = max_delay
    thea2 = max_delay

    log = build_training_log()
    scalar_names = ["total", "task"]
    for layer in TARGET_LAYERS:
        scalar_names += [f"ceiling_l{layer}", f"floor_l{layer}",
                         f"rate_l{layer}", f"rate_active_l{layer}",
                         f"silent_l{layer}", f"over_k_l{layer}",
                         f"sub_threshold_l{layer}"]
    scalar_names += ["rate_net", "rate_active_net", "silent_net"]

    desc = (f"Train L1+L2 k={k:g}/{ceilings[2]:g} "
            f"floor={floor_strength:g} seed={seed}")
    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=desc) as pbar:
        for epoch in range(epochs):
            penalty_on = epoch >= warmup_epochs

            net.train()
            batch_stats: dict[str, list[float]] = {name: [] for name in scalar_names}

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs, hidden1, potential1, hidden2, potential2 = net(
                    x_batch, return_hidden=True)
                task_loss = loss_fn.numSpikes(outputs, target)

                spikes = {1: hidden1, 2: hidden2}
                potentials = {1: potential1, 2: potential2}

                # One ceiling and one floor per layer, summed. Each layer is charged at
                # the full coefficient, which is what makes a cell here comparable to
                # the same cell in either single-layer grid.
                ceiling_terms = {
                    layer: ceiling_penalty(spikes[layer], ceilings[layer])
                    for layer in TARGET_LAYERS
                }
                floor_terms = {
                    layer: (floor_penalty(potentials[layer], theta_margin)
                            if floor_strength > 0
                            else torch.zeros((), device=device))
                    for layer in TARGET_LAYERS
                }
                if penalty_on:
                    loss = task_loss
                    for layer in TARGET_LAYERS:
                        loss = (loss
                                + ceiling_strength * ceiling_terms[layer]
                                + floor_strength * floor_terms[layer])
                else:
                    loss = task_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                per_layer = {
                    layer: layer_activity_stats(
                        spikes[layer], potentials[layer], ceilings[layer], theta_margin)
                    for layer in TARGET_LAYERS
                }
                pooled = pool_layer_stats(per_layer)

                batch_stats["total"].append(loss.item())
                batch_stats["task"].append(task_loss.item())
                for layer in TARGET_LAYERS:
                    batch_stats[f"ceiling_l{layer}"].append(
                        ceiling_terms[layer].item())
                    batch_stats[f"floor_l{layer}"].append(floor_terms[layer].item())
                    for stat in ("rate", "rate_active", "silent", "over_k",
                                 "sub_threshold"):
                        batch_stats[f"{stat}_l{layer}"].append(per_layer[layer][stat])
                for stat in ("rate", "rate_active", "silent"):
                    batch_stats[f"{stat}_net"].append(pooled[stat])
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
            for layer in TARGET_LAYERS:
                log[f"train_ceiling_penalty_l{layer}"].append(
                    means[f"ceiling_l{layer}"])
                log[f"train_floor_penalty_l{layer}"].append(means[f"floor_l{layer}"])
                for stat in ("rate", "rate_active", "silent", "over_k",
                             "sub_threshold"):
                    log[f"train_{stat}_l{layer}"].append(means[f"{stat}_l{layer}"])
            for stat in ("rate", "rate_active", "silent"):
                log[f"train_{stat}_net"].append(means[f"{stat}_net"])
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))
            log["delay_mean"].append(float(avg_delay))

            # Both layers on the bar: the failure this run is most exposed to is one
            # layer collapsing while the other looks healthy.
            pbar.set_postfix(
                epoch=epoch + 1,
                task=f"{means['task']:.3f}",
                a1=f"{means['rate_active_l1']:.2f}",
                s1=f"{means['silent_l1']:.0%}",
                a2=f"{means['rate_active_l2']:.2f}",
                s2=f"{means['silent_l2']:.0%}",
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
    k_layer2_ratio: float = CEILING_K_LAYER2_RATIO,
) -> dict:
    """Measure clean test accuracy and both layers' sparsity axes in one pass.

    Keys are suffixed ``_l1`` / ``_l2`` per layer and ``_net`` for the pool. The three
    canonical unsuffixed names — ``spikes_per_neuron``, ``spikes_per_active_neuron``,
    ``silent_fraction`` — are **also** emitted, carrying the **network-wide** values, so
    that analysis code written against the single-layer summaries reads the quantity
    this experiment actually manipulates rather than one layer of it by accident.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.
        k: Layer 1's ceiling, for its ``over_k`` check. Layer 2's is derived.
        theta_margin: The floor threshold, for the ``sub_threshold`` checks.
        k_layer2_ratio: Layer 2's ceiling as a multiple of layer 1's.

    Returns:
        Dict of clean accuracy plus every axis at layer 1, layer 2 and pooled.
    """
    net.eval()
    ceilings = {1: k, 2: layer2_ceiling(k, k_layer2_ratio)}

    correct = 0
    total = 0
    totals = {
        layer: {"spikes": 0.0, "sq": 0.0, "slots": 0, "pairs": 0,
                "silent": 0.0, "over_k": 0.0, "sub_threshold": 0.0}
        for layer in TARGET_LAYERS
    }
    silent_peaks: dict[int, list[torch.Tensor]] = {l: [] for l in TARGET_LAYERS}
    active_peaks: dict[int, list[torch.Tensor]] = {l: [] for l in TARGET_LAYERS}

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
        y_batch = y_batch.to(device)

        outputs, hidden1, potential1, hidden2, potential2 = net(
            x_batch, return_hidden=True)

        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)

        spikes = {1: hidden1, 2: hidden2}
        potentials = {1: potential1, 2: potential2}
        for layer in TARGET_LAYERS:
            batch, channels, _, _, time = spikes[layer].shape
            counts = spikes[layer].sum(dim=-1).view(batch, channels)
            peak = potentials[layer].max(dim=-1).values.view(batch, channels)
            silent = counts == 0

            bucket = totals[layer]
            bucket["spikes"] += counts.sum().item()
            bucket["sq"] += (counts ** 2).sum().item()
            bucket["slots"] += batch * channels * time
            bucket["pairs"] += batch * channels
            bucket["silent"] += silent.sum().item()
            bucket["over_k"] += (counts > ceilings[layer]).sum().item()
            bucket["sub_threshold"] += (peak < theta_margin).sum().item()
            silent_peaks[layer].append(peak[silent].cpu())
            active_peaks[layer].append(peak[~silent].cpu())

    metrics: dict[str, float] = {"clean_acc": correct / max(1, total)}
    net_spikes = 0.0
    net_pairs = 0
    net_silent = 0.0

    for layer in TARGET_LAYERS:
        bucket = totals[layer]
        n_pairs = max(1, bucket["pairs"])
        active_pairs = bucket["pairs"] - bucket["silent"]
        mean_count = bucket["spikes"] / n_pairs
        variance = max(0.0, bucket["sq"] / n_pairs - mean_count ** 2)
        silent_peak = torch.cat(silent_peaks[layer])
        active_peak = torch.cat(active_peaks[layer])

        metrics.update({
            f"ceiling_k_l{layer}": float(ceilings[layer]),
            f"firing_rate_l{layer}": bucket["spikes"] / max(1, bucket["slots"]),
            f"spikes_per_neuron_l{layer}": mean_count,
            f"spikes_per_active_neuron_l{layer}": (
                bucket["spikes"] / max(1.0, active_pairs)),
            f"silent_fraction_l{layer}": bucket["silent"] / n_pairs,
            f"over_k_fraction_l{layer}": bucket["over_k"] / n_pairs,
            f"sub_threshold_fraction_l{layer}": bucket["sub_threshold"] / n_pairs,
            f"count_std_l{layer}": variance ** 0.5,
            f"peak_potential_silent_median_l{layer}": (
                float(silent_peak.median()) if silent_peak.numel() else float("nan")),
            f"peak_potential_active_median_l{layer}": (
                float(active_peak.median()) if active_peak.numel() else float("nan")),
        })
        net_spikes += bucket["spikes"]
        net_pairs += bucket["pairs"]
        net_silent += bucket["silent"]

    net_pairs = max(1, net_pairs)
    net_active = max(1.0, net_pairs - net_silent)
    metrics.update({
        "spikes_per_neuron_net": net_spikes / net_pairs,
        "spikes_per_active_neuron_net": net_spikes / net_active,
        "silent_fraction_net": net_silent / net_pairs,
    })

    # Canonical unsuffixed keys alias the NETWORK-WIDE pool, not layer 1. This is the
    # level the both-layer factorial manipulates, so it is what a downstream script
    # reading `spikes_per_active_neuron` should get. Deliberate, and documented in the
    # module docstring -- do not repoint these at a single layer.
    metrics.update({
        "spikes_per_neuron": metrics["spikes_per_neuron_net"],
        "spikes_per_active_neuron": metrics["spikes_per_active_neuron_net"],
        "silent_fraction": metrics["silent_fraction_net"],
    })
    return metrics


def report_manipulation_check(summary: dict[str, dict]) -> None:
    """Print whether the floor column separated the selectivity levels, at each layer.

    The probe's job is this and nothing else. Reported per ``k`` **and per layer**, so
    that a floor which bites at one layer but not the other — or at one end of the row
    axis but not the other — is visible rather than averaged away. A column knob that
    works at only one layer leaves this design doing half its job.

    Args:
        summary: The run summary, run tag -> recorded metrics.
    """
    levels = ("l1", "l2", "net")
    by_k: dict[float, dict[float, dict[str, float]]] = {}
    for row in summary.values():
        by_k.setdefault(row["ceiling_k"], {})[row["floor_strength"]] = {
            level: row[f"silent_fraction_{level}"] for level in levels
        }

    print(f"\n{'k':>5} {'level':>7} {'floor=0 silent':>15} {'floor=on silent':>16} "
          f"{'separation':>11}")
    for k in sorted(by_k):
        cells = by_k[k]
        floor_off = cells.get(0.0)
        floor_on = next(
            (value for strength, value in cells.items() if strength > 0), None)
        if floor_off is None or floor_on is None:
            print(f"{k:>5g} {'incomplete cell':>40}")
            continue
        for level in levels:
            separation = floor_off[level] - floor_on[level]
            flag = "OK" if separation >= MIN_SILENT_SEPARATION else "TOO SMALL"
            print(f"{k:>5g} {level:>7} {floor_off[level]:>14.2%} "
                  f"{floor_on[level]:>15.2%} {separation:>+10.2%}  {flag}")
    print(f"\n  Target separation >= {MIN_SILENT_SEPARATION:.0%} at matched k, "
          f"AT BOTH LAYERS.")
    print("  If it holds at one layer only, that layer's floor is doing all the work "
          "and the\n  column knob is not a network-wide selectivity manipulation. "
          "Raise FLOOR_STRENGTH.")


def report_design_acceptance(summary: dict[str, dict]) -> None:
    """Print the design acceptance check on the factorial itself.

    This is a check on the **design**, not on the network: it asks whether the
    factorial actually decorrelated the two sparsity axes. If it did not, the
    regression the grid feeds cannot separate H1' from H2 at that level and must not be
    read as though it could — the correct response is to report the failure and stop.

    Run at three levels, because this design has three. The network-wide row is the one
    the experiment is *about*; the two per-layer rows say whether a layer-resolved
    reading of the same grid is also interpretable. In this arm the network-wide
    observational baseline already passes at +0.042, so what the design adds here is the
    controlled contrast rather than decorrelation.

    The cross-layer correlations printed afterwards are **diagnostics, not criteria**.
    Both knobs act on both layers, so ``rho(a1, a2)`` is expected to stay high — that is
    what constraining the network means, and separating the layers is what the two
    single-layer grids are for. ``rho(s1, s2)`` is the informative one: under v1's
    layer-1-only penalty it sat at +0.203, i.e. the two layers' selectivity was
    essentially unrelated, and this design should pull it clearly positive.

    Args:
        summary: The run summary, run tag -> recorded metrics.
    """
    if len(summary) < 4:
        print("\n[design check] Fewer than 4 models — rho(a, s) not meaningful yet.")
        return

    rows = list(summary.values())
    print(f"\n{'=' * 70}")
    print(f"DESIGN ACCEPTANCE CHECK — LAYERS {'+'.join(map(str, TARGET_LAYERS))} "
          f"(n={len(rows)})")
    print(f"  threshold |rho| <= {MAX_AXIS_CORRELATION}")
    print(f"\n  {'level':>8} {'rho(a, s)':>12} {'p':>10} {'v1 baseline':>13}  verdict")
    for level, label in (("net", "network"), ("l1", "layer1"), ("l2", "layer2")):
        a_axis = [row[f"spikes_per_active_neuron_{level}"] for row in rows]
        s_axis = [row[f"silent_fraction_{level}"] for row in rows]
        result = spearmanr(a_axis, s_axis)
        verdict = ("PASS" if abs(result.statistic) <= MAX_AXIS_CORRELATION
                   else "FAIL — still confounded, NOT interpretable as a test of H1'")
        print(f"  {label:>8} {result.statistic:>+12.3f} {result.pvalue:>10.4g} "
              f"{V1_AXIS_CORRELATION[label]:>+13.3f}  {verdict}")
    print("\n  In this arm the v1 network-wide baseline (+0.042) already passed, so "
          "the value\n  added here is the controlled floor-on vs floor-off contrast at "
          "matched `a`,\n  not decorrelation.")

    print("\n  Cross-layer diagnostics (NOT acceptance criteria — this design moves "
          "both\n  layers with one pair of knobs, so a high rho(a1, a2) is expected):")
    for axis, key in (("a", "spikes_per_active_neuron"), ("s", "silent_fraction")):
        first = [row[f"{key}_l1"] for row in rows]
        second = [row[f"{key}_l2"] for row in rows]
        result = spearmanr(first, second)
        print(f"    rho({axis}1, {axis}2) = {result.statistic:+.3f} "
              f"(p = {result.pvalue:.4g}),  v1 baseline "
              f"{V1_CROSS_LAYER_CORRELATION[axis]:+.3f}")
    print("    rho(s1, s2) is the informative one: under v1's layer-1-only penalty the "
          "two\n    layers' selectivity was essentially unrelated (+0.203). This "
          "design should\n    pull it clearly positive.")
    print(f"{'=' * 70}")


def run_milestone() -> None:
    """Train the (k, floor_strength, seed) factorial and record every sparsity axis."""
    ceiling_k, floor_strengths, seeds, epochs, warmup_epochs = resolve_run_config()

    n_models = len(ceiling_k) * len(floor_strengths) * len(seeds)
    layer_label = "+".join(map(str, TARGET_LAYERS))
    print(f"{'#' * 70}")
    print(f"# Sparse-network milestone v3 (training): SHD whole, SGD-delay, "
          f"LAYERS {layer_label}")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs} | warmup={warmup_epochs}")
    print(f"# mechanism = per-pair ceiling relu(count-k) + potential floor "
          f"relu(margin - max_t u),")
    print(f"#             charged separately on hidden layers {layer_label}")
    print(f"# ceiling k (layer 1): {ceiling_k}  (strength {CEILING_STRENGTH:g} "
          f"per layer)")
    print(f"# ceiling k (layer 2): {[layer2_ceiling(k) for k in ceiling_k]}  "
          f"(ratio {CEILING_K_LAYER2_RATIO:g})")
    print(f"# floor strengths: {floor_strengths}  (margin {THETA_MARGIN:g}, "
          f"theta {LIF_PARAMS['theta']})")
    print(f"# seeds: {seeds}  ->  {n_models} models")
    print(f"{'#' * 70}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data and build loaders once; the split (and hence the test set) is fixed, so
    # all models are comparable — including against v1 and against both single-layer
    # grids.
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
                print(f"  ceilings: layer 1 k={k:g}, layer 2 k={layer2_ceiling(k):g}")
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
                    f"  clean_acc={metrics['clean_acc']:.4f}\n"
                    f"    layer 1: a={metrics['spikes_per_active_neuron_l1']:.2f} | "
                    f"s={metrics['silent_fraction_l1']:.2%} | "
                    f"sp/neuron={metrics['spikes_per_neuron_l1']:.2f} | "
                    f"over_k={metrics['over_k_fraction_l1']:.2%}\n"
                    f"    layer 2: a={metrics['spikes_per_active_neuron_l2']:.2f} | "
                    f"s={metrics['silent_fraction_l2']:.2%} | "
                    f"sp/neuron={metrics['spikes_per_neuron_l2']:.2f} | "
                    f"over_k={metrics['over_k_fraction_l2']:.2%}\n"
                    f"    network: a={metrics['spikes_per_active_neuron_net']:.2f} | "
                    f"s={metrics['silent_fraction_net']:.2%} | "
                    f"sp/neuron={metrics['spikes_per_neuron_net']:.2f}"
                )

                summary[run_tag] = {
                    "target_layers": list(TARGET_LAYERS),
                    "ceiling_k": float(k),
                    "ceiling_k_layer2": float(layer2_ceiling(k)),
                    "ceiling_k_layer2_ratio": float(CEILING_K_LAYER2_RATIO),
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

    # Final table. Both axes at both layers, never a product alone and never one layer
    # standing in for the network.
    print(
        f"\n{'run_tag':<56} {'acc':>7} "
        f"{'a1':>6} {'s1':>7} {'sp1':>6} {'ovr1':>6} "
        f"{'a2':>6} {'s2':>7} {'sp2':>6} {'ovr2':>6} "
        f"{'a_net':>6} {'s_net':>7}"
    )
    for run_tag, row in summary.items():
        flags = "".join(
            f" !L{layer}" if row[f"spikes_per_neuron_l{layer}"]
            > NATURAL_SPIKES_PER_NEURON[layer] else ""
            for layer in TARGET_LAYERS
        )
        print(
            f"{run_tag:<56} {row['clean_acc']:>7.4f} "
            f"{row['spikes_per_active_neuron_l1']:>6.2f} "
            f"{row['silent_fraction_l1']:>7.2%} "
            f"{row['spikes_per_neuron_l1']:>6.2f} "
            f"{row['over_k_fraction_l1']:>6.1%} "
            f"{row['spikes_per_active_neuron_l2']:>6.2f} "
            f"{row['silent_fraction_l2']:>7.2%} "
            f"{row['spikes_per_neuron_l2']:>6.2f} "
            f"{row['over_k_fraction_l2']:>6.1%} "
            f"{row['spikes_per_active_neuron_net']:>6.2f} "
            f"{row['silent_fraction_net']:>7.2%}{flags}"
        )
    print(f"\n  !L1 / !L2 = that layer's spikes_per_neuron above its natural rate "
          f"({NATURAL_SPIKES_PER_NEURON[1]} / {NATURAL_SPIKES_PER_NEURON[2]} "
          f"in this arm) — v2.2's failure mode.")

    report_manipulation_check(summary)
    report_design_acceptance(summary)


if __name__ == "__main__":
    run_milestone()
