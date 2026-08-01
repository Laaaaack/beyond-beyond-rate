"""First milestone v3 (training), WITH-DELAY arm: the a x s dissociation factorial.

Successor to [sn_train_withDelay_v2.py](sn_train_withDelay_v2.py). Same network, data,
splits, optimiser, schedule and logging; **only the sparsity mechanism differs**, and
the sweep is 2-D rather than 1-D.

Why this experiment exists
--------------------------
v1 and v2 both treated "sparsity" as one number. It is two, and they push the code in
opposite directions::

    spikes_per_neuron = (1 - s) * a

===================  ==========================================  ==================
variable             what it is                                  what it does
===================  ==========================================  ==================
``a`` temporal       spikes per **active** neuron per sample      fewer spikes per
sparsity                                                         neuron => count
                                                                 resolution falls
                                                                 => latency coding
``s`` selectivity    fraction of (sample, neuron) pairs silent    neurons silent on
                                                                 some stimuli => a
                                                                 labelled-line
                                                                 **identity** code,
                                                                 which every
                                                                 rate-preserving
                                                                 perturbation leaves
                                                                 intact
===================  ==========================================  ==================

**H1 is a claim about ``a`` alone.** v1's penalty moved both together, at
``rho(a, s) = -0.943`` in the no-delay arm and ``-0.455`` here, so no analysis of v1's
checkpoints can attribute its result to one rather than the other. This script's job
is to make ``a`` and ``s`` vary *independently*, which no previous version has done.

The two terms
-------------
1. **Ceiling — sets ``a``.** ``relu(count - k)`` charged per ``(sample, neuron)``
   pair, not on the batch mean. Per-pair is the whole point: v1's batch-mean hinge
   let a neuron satisfy the penalty by going *silent* on some samples and firing
   freely on others, which is exactly how ``a`` and ``s`` became confounded.
2. **Floor — sets ``s``.** ``relu(THETA_MARGIN - max_t u(t))`` on the **membrane
   potential**, not on the spike count. Two properties make this the right fix, both
   measured on v1's checkpoints (document 5 §2e):

   - ``max_t u >= theta`` guarantees at least one spike, so it is *sufficient* to
     prevent silence while asking for the minimum sufficient condition rather than
     for more spikes — which is what should keep raw firing off v2.2's blow-up.
   - ``du/dw`` is an ordinary convolution gradient: **no surrogate is involved**, so
     the gradient does not vanish however far below threshold the neuron sits. Silent
     pairs sit at ``u ~ 0`` against ``theta = 10``, which is why every count-based
     floor (v1, v2, v2.1, v2.2) could not reach them.

The floor's **strength is the column knob**: 0 in one column, ``FLOOR_STRENGTH`` in
the other. ``k`` is the row knob. Crossing them is the factorial.

Which arm, and why this one
----------------------------
The v3 design document nominated the no-delay arm. Phase 0 changed that call. Two of
its findings are properties of the arm that the factorial *cannot* fix:

- The no-delay readout leaves **+.300** of accuracy unextracted relative to a linear
  decoder on its own 1st hidden layer, on **every** checkpoint (delay: +.026). Its
  ``temporal_score`` is as much a statement about ``fc2``/``fc3`` as about layer 1.
- In the no-delay arm the deletion control correlates with sparsity at +0.936 against
  the timing probe's +0.939 — a pure *rate* insult and a pure *timing* insult are
  statistically indistinguishable there. In this arm they separate (+0.713 vs +0.881),
  and the usage effect survives partialling on both ``s`` and the deletion control
  (+0.716, p = .020) where the no-delay one does not (+0.238, p = .43).

A factorial with a compromised dependent variable yields a clean coefficient on a
number that does not mean what it needs to mean. The cost is ~2.3 h/model against
~1.8 h. See ``sparse_network_test_progress_v3.md`` §4.

What is deliberately NOT here
------------------------------
- **No ``truncate_to_k_spikes``.** v2.2's channel-closing programme is retired
  (document 5 §6): closing the immune channel costs the experiment's own validity,
  while *controlling* it only requires measuring it — which Phase 0 now does.
- **No anneal.** v2.2's was already off; the dead-zone hazard it existed to avoid is
  handled here by the potential floor, in the column where silence is not wanted.

Reading the probe
-----------------
``QUICK_TEST`` runs the 4 corners ``k in {1, 8} x floor in {0, F}`` at 400 epochs. It
validates the **manipulation**, which is a property of the constraint and readable
early. Check, in priority order:

1. ``silent_fraction`` roughly constant as ``k`` varies *within* a column;
2. ``silent_fraction`` differing by **>= 20 points** *across* columns at matched ``k``
   — if not, raise ``FLOOR_STRENGTH``;
3. ``spikes_per_neuron`` not inflated beyond the natural ~6.3 (v2.2's failure mode);
4. clean accuracy comfortably above chance in every corner.

**Do not read achieved firing rates or accuracy levels off the probe.** Every v1 model
was still improving at epoch 1250 (best val loss at epochs 1168-1243), so 400 epochs is
a different regime, not a scaled-down one. Calibrate nothing from it.

Design acceptance check
------------------------
After the full grid, ``rho(a, s)`` across the trained models is printed. **If
|rho| > 0.5 the factorial has failed to break the confound** and the regression it
feeds is not interpretable — report that and stop, exactly as v2 would have had to
report a failed manipulation check.

Outputs are tagged ``_v3_`` and cannot collide with v1, v2, v2.1 or v2.2.

This is a **training-only** script. The perturbations that measure temporal processing
are applied **only at evaluation**, by the sibling experiments under
``exp_sparse_network/{jitter,shift,shd,deletion}/`` — never switch one on here. The
availability measure lives in ``exp_sparse_network/v3_analysis/``. See:

- ``my_project/docs/progress/sparse_network_test_progress_v3.md`` — v3 design + Phase 0.
- ``my_project/docs/progress/sparse_network_1stLayer_results.md`` — v1 results.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.

What this script produces, per ``(k, floor_strength, seed)``:

- a checkpoint ``sn_data/sparse_whole_delay_v3_k{k}_floor{f}_seed{seed}.pt``;
- a training-curve log ``sn_log/..._training_log.json`` carrying the trajectories of
  **both** sparsity axes, not just ``spikes_per_neuron``;
- a row in ``sn_log/sparse_whole_delay_v3_train_summary.json`` recording clean test
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
# Corner probe. True runs k in {1, 8} x floor in {0, FLOOR_STRENGTH} at 400 epochs,
# one seed -- 4 models, ~3 h on this arm. Its only job is to validate the
# manipulation; see "Reading the probe" in the module docstring. Probe artifacts carry
# RUN_SUFFIX so they can never clobber a real run.
QUICK_TEST: bool = False

# Suffix appended to every output name (checkpoints, per-model logs, summary) when
# running a probe, so a QUICK_TEST probe can never overwrite real-run artifacts that
# share the same (k, floor_strength, seed). Empty for the real run.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Marks every v3 artifact so it can never collide with v1, the v2 band probe, the v2.1
# exact-count probe or the v2.2 truncation probe.
VERSION_TAG: str = "v3"

# --- Milestone scope ---
DATASET_KEY: str = "whole"
INPUT_DIM: int = 700          # SHD whole
USE_DELAY: bool = True        # SGD-delay; see "Which arm, and why this one" above
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# --- The factorial ---
# Row knob: the per-pair spike ceiling. This is the H1 axis -- it sets `a`, spikes per
# ACTIVE neuron, by making counts above k costly for every (sample, neuron) pair
# individually. Spans the natural rate (~6.3 in v1's dense models) down to a single
# spike, so `a` moves by roughly an order of magnitude.
CEILING_K: list[float] = [1.0, 2.0, 4.0, 8.0]

# Column knob: the membrane-potential floor. 0 lets neurons fall silent freely (high
# `s`); nonzero forbids it (low `s`). This is the H2 axis, and it is what makes the
# floor-on vs floor-off contrast at matched `k` a genuine controlled comparison rather
# than a correlation -- the first in this line of work.
FLOOR_STRENGTH: list[float] = [0.0, 1.0]

#SEEDS: list[int] = [42, 43]
SEEDS: list[int] = [44]

# Coefficient on the ceiling term. Fixed across the whole grid so that `k` is the only
# thing varying along the row axis.
#
# Calibrated at k = 1 on the delay arm, 400 epochs, both floor columns. 3.0 (v2's value)
# was too weak: it left `a` spanning only 1.24-1.42x across k = 1 -> 8, against v1's
# observational span of 4.1x in this arm, because the penalty reaches equilibrium far
# above k and stops pushing (67% of pairs still above the ceiling at k = 1).
#
#   k=1     a      s      sp/neu   over_k   clean_acc
#   ---------------------------------------------------
#   str=3,  floor=0   6.17  62.7%   2.30    27%     .818
#   str=3,  floor=1   4.90  11.0%   4.36    67%     .824
#   str=10, floor=0   4.00  72.1%   1.12    17%     .685
#   str=10, floor=1   2.66  22.6%   2.06    41%     .771
#
# 10.0 roughly doubles the row-axis span, holds the floor column's separation at ~49
# points (from ~52), keeps firing far below this arm's natural 11.66, does not re-enter
# the all-silent absorbing state, and costs accuracy that remains usable. It is also the
# safer side of the 400-epoch caveat: penalties calibrated at reduced epochs RE-DENSIFY
# by the full 1250, so the achieved `a` will sit above these numbers in the real run.
#
# Residual limitation, stated rather than hidden: even at 10 the ceiling does not bind
# hard -- 41% of pairs remain above k at k = 1, and `a` settles 2.7x above the target.
# The row axis is a soft, manipulated axis with real range, not a pinned one.
CEILING_STRENGTH: float = 10.0

# The floor's threshold on peak membrane potential, as a multiple of theta = 10. 1.1 x
# theta so that a satisfied pair is not left sitting exactly on the boundary, where a
# small weight change would drop it back into silence.
THETA_MARGIN: float = 11.0

# Clean warm-up before either penalty engages, in epochs. **Required, not optional.**
#
# v1 and v2.2 both used 0, but their penalties pushed firing toward a nonzero target.
# v3's ceiling relu(count - k) is one-sided and is minimised at count = 0, so in the
# floor-OFF column nothing but the task loss opposes silence -- and total silence is an
# *absorbing* state, because once every pair sits at u ~ 0 against theta = 10 the only
# route back is SLAYER's surrogate gradient, which carries no signal there.
#
# Measured on this arm at k = 1, floor = 0, full training set:
#
#   warmup=0   -> 100% silent by epoch 5, val_acc pinned at chance (5.1%) for all
#                 60 epochs. Dead layer, unrecoverable.
#   warmup=20  -> 40-48% silent, sp/neuron 8.6 -> 5.5 and still falling, val_acc 74.5%.
#
# 20 epochs is enough for the task representation to establish (val_acc was already
# 65.7% at epoch 19) before the ceiling starts pushing counts down, and the ceiling
# then binds gradually rather than catastrophically. Raise it if a tighter k still
# collapses; the probe's corner at k = 1, floor = 0 is what would show that.
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
# The natural firing rate of v1's densest models **in this arm**. The ceiling must not
# drive the layer ABOVE this; v2.2 inflated the no-delay layer from ~6.3 to 12.5-21.9
# and stopped being an instance of "sparser-activity networks" as a result.
#
# Note the two arms differ: v1's spikes_per_neuron spans 1.21-7.85 (no-delay) and
# 2.12-11.66 (delay), so the ~6.3 figure quoted in the v3 design document is the
# no-delay number and must not be applied here.
NATURAL_SPIKES_PER_NEURON: float = 11.66
# The design acceptance check on the factorial itself (document 5 §4b).
MAX_AXIS_CORRELATION: float = 0.5


def resolve_run_config() -> tuple[list[float], list[float], list[int], int, int]:
    """Return (ceiling_k, floor_strengths, seeds, epochs, warmup_epochs) for this run.

    Collapses to the 4-corner manipulation probe when ``QUICK_TEST`` is set. The
    corners are the extremes of both axes, which is what makes them a test of the
    manipulation: ``k`` at 1 and 8 brackets the whole row axis, and the floor at 0 and
    ``FLOOR_STRENGTH`` is the entire column axis.

    Returns:
        Tuple of (ceiling_k, floor_strengths, seeds, epochs, warmup_epochs).
    """
    if QUICK_TEST:
        return [1.0, 8.0], FLOOR_STRENGTH, [42], 400, WARMUP_EPOCHS
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
    curves and for reusing Phase 0's measurement layer unchanged. Only the training
    order is shuffled.

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
    """2-hidden-layer SLAYER SNN with learnable delays, for the v3 factorial.

    The parameter set is identical to v1's class of the same name — ``fc1``/``fc2``/
    ``fc3`` weight-norm parameters plus ``delay1``/``delay2`` and nothing else — so v3
    checkpoints load directly into the existing eval scripts and into
    ``v3_analysis/``. Unlike v2.2 there is **no truncation**, so the forward pass is
    v1's exactly and a v3 checkpoint needs no special handling anywhere downstream.

    The only addition is that ``forward`` can return the 1st hidden layer's **membrane
    potential** alongside its spikes, because v3's floor penalty acts on the potential
    rather than on the spike count. That tensor is computed by the forward pass
    regardless; returning it costs nothing.

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

        # delay1 sits at the start of _second_hidden_and_output (after the
        # perturbation site); delay2 stays between the fc2 spike and fc3.
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

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> (delay1) -> fc2 -> spike -> (delay2) -> fc3 -> spike."""
        x = hidden1
        if self.use_delay:
            x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay2(x)
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Clean forward pass (used during training and clean evaluation).

        Args:
            x: Input spike trains.
            return_hidden: If True, also return the 1st hidden layer's spikes and the
                membrane potential they were thresholded from, so the caller can apply
                both penalties.

        Returns:
            The output spike tensor, or ``(output, hidden1, potential1)`` if
            ``return_hidden`` is True.
        """
        x = self._prepare_input(x)
        potential1 = self.fc1(self.slayer.psp(x))
        hidden1 = self.slayer.spike(potential1)
        out = self._second_hidden_and_output(hidden1)
        return (out, hidden1, potential1) if return_hidden else out

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
    perturbation-robust, and confounding ``a`` with ``s`` at rho = -0.94. Charging
    each pair individually removes that escape route: silence on one sample buys no
    licence to over-fire on another.

    This is the axis H1 is actually about. It is deliberately one-sided: nothing here
    stops a neuron falling silent, because whether silence is permitted is the *other*
    knob's job.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).
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
    v1's checkpoints, silent (sample, neuron) pairs sit at a median peak potential of
    **0.00** against ``theta = 10`` — they are not marginally sub-threshold, they
    receive essentially no drive at all. A count-based penalty reaches them only
    through SLAYER's surrogate gradient, which at ``u ~ 0`` against ``theta = 10``
    carries no signal; that dead zone silenced v1's no-delay arm for 26 epochs, left
    v2/v2.1 at 21-40% silent, and defeated v2.2's ``relu(k - count)`` even at strength
    10. ``du/dw`` is an ordinary convolution gradient and never vanishes.

    Asking for one suprathreshold crossing rather than for more spikes is also what
    should keep raw firing off v2.2's blow-up (12.5-21.9 against a natural ~6.3): it
    requests the minimum sufficient condition for the neuron to be non-silent.

    Args:
        potential: 1st hidden layer membrane potential, shape (B, C, 1, 1, T).
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
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).
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
    sparsity intervention localised to the hidden layer. Held identical to v1 and v2
    so the arms remain comparable.

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

    The forward pass is clean (no perturbation). After the warm-up the total loss is
    the NumSpikes task loss, plus ``ceiling_strength`` times the per-pair excess above
    ``k``, plus ``floor_strength`` times the per-pair shortfall of peak membrane
    potential below ``theta_margin``. ``floor_strength = 0`` skips the floor term
    entirely, which is the control column.

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

    desc = f"Train k={k:g} floor={floor_strength:g} seed={seed}"
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

                outputs, hidden1, potential1 = net(x_batch, return_hidden=True)
                task_loss = loss_fn.numSpikes(outputs, target)

                ceiling_term = ceiling_penalty(hidden1, k)
                floor_term = (floor_penalty(potential1, theta_margin)
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
                    hidden_activity_stats(hidden1, potential1, k, theta_margin))
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
    """Measure clean test accuracy and both sparsity axes in one pass.

    The summary this feeds carries ``spikes_per_active_neuron`` and
    ``silent_fraction`` alongside ``spikes_per_neuron``, which v1's summaries did not
    — the omission that let the ``rho(a, s) = -0.94`` confound go unnoticed through
    two whole versions.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.
        k: The ceiling the model was trained at, for the ``over_k`` check.
        theta_margin: The floor threshold, for the ``sub_threshold`` check.

    Returns:
        Dict with keys ``clean_acc``, ``firing_rate``, ``spikes_per_neuron``,
        ``spikes_per_active_neuron``, ``silent_fraction``, ``over_k_fraction``,
        ``sub_threshold_fraction``, ``count_std``, ``peak_potential_silent_median``,
        ``peak_potential_active_median``.
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

        outputs, hidden1, potential1 = net(x_batch, return_hidden=True)

        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)

        batch, channels, _, _, time = hidden1.shape
        counts = hidden1.sum(dim=-1).view(batch, channels)
        peak = potential1.max(dim=-1).values.view(batch, channels)
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
    factorial actually decorrelated the two sparsity axes. If it did not, the
    regression the grid feeds cannot separate H1' from H2 and must not be read as
    though it could — the correct response is to report the failure and stop, exactly
    as v2 would have had to report a failed manipulation check.

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
    print(f"DESIGN ACCEPTANCE CHECK (n={len(summary)})")
    print(f"  rho(a, s) = {result.statistic:+.3f}  (p = {result.pvalue:.4g}), "
          f"threshold |rho| <= {MAX_AXIS_CORRELATION}")
    print(f"  {verdict}")
    print("  For reference, v1's observational values were -0.943 (no-delay) "
          "and -0.455 (delay).")
    print(f"{'=' * 70}")


def run_milestone() -> None:
    """Train the (k, floor_strength, seed) factorial and record both sparsity axes."""
    ceiling_k, floor_strengths, seeds, epochs, warmup_epochs = resolve_run_config()

    n_models = len(ceiling_k) * len(floor_strengths) * len(seeds)
    print(f"{'#' * 70}")
    print("# Sparse-network milestone v3 (training): SHD whole, SGD-delay")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs} | warmup={warmup_epochs}")
    print("# mechanism = per-pair ceiling relu(count-k) + potential floor "
          "relu(margin - max_t u)")
    print(f"# ceiling k: {ceiling_k}  (strength {CEILING_STRENGTH:g})")
    print(f"# floor strengths: {floor_strengths}  (margin {THETA_MARGIN:g}, "
          f"theta {LIF_PARAMS['theta']})")
    print(f"# seeds: {seeds}  ->  {n_models} models")
    print(f"{'#' * 70}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data and build loaders once; the split (and hence the test set) is fixed, so
    # all models are comparable — including against v1 and against Phase 0's measures.
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
        f"\n{'run_tag':<58} {'clean_acc':>9} {'a':>7} {'s':>8} {'sp/neu':>8} "
        f"{'over_k':>8} {'sub_th':>8}"
    )
    for run_tag, row in summary.items():
        inflated = "" if row["spikes_per_neuron"] <= NATURAL_SPIKES_PER_NEURON else " !"
        print(
            f"{run_tag:<58} {row['clean_acc']:>9.4f} "
            f"{row['spikes_per_active_neuron']:>7.2f} "
            f"{row['silent_fraction']:>8.2%} {row['spikes_per_neuron']:>8.2f} "
            f"{row['over_k_fraction']:>8.2%} "
            f"{row['sub_threshold_fraction']:>8.2%}{inflated}"
        )
    print(f"\n  ! = spikes_per_neuron above the natural "
          f"{NATURAL_SPIKES_PER_NEURON} — v2.2's failure mode.")

    report_manipulation_check(summary)
    report_design_acceptance(summary)


if __name__ == "__main__":
    run_milestone()
