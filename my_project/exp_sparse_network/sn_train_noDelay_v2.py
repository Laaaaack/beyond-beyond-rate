"""First milestone v2 (training), NO-DELAY arm: banded hidden-layer sparsity on SHD.

Successor to [sn_train_noDelay.py](sn_train_noDelay.py). Same network, data,
optimiser and schedule; **only the sparsity penalty differs** (plus the warm-up, see
below). v1's one-sided hinge produced a sparsity gradient, but the v1 perturbation
results (``docs/progress/sparse_network_1stLayer_results.md``) showed it could not
test H1:

- The hinge has **zero gradient at or below the target**, so a neuron can escape the
  penalty entirely by falling silent on a sample. Sparsity therefore arrived partly
  as **stimulus selectivity** (silent fraction .33 -> .54), which *strengthens* a
  population-identity code that every perturbation in the study leaves intact.
- The regime H1 actually needs — each active neuron firing 0-2 spikes so *count*
  carries no resolution — was never reached: even the sparsest v1 model fired
  **3.12 spikes per active neuron** (the headline ``spikes_per_neuron`` of 1.42 was
  diluted by silent neurons).

So v1 refuted "H1 as tested" while leaving "H1 as intended" untested: the network
always had a rich, perturbation-immune identity channel to fall back on. This arm
made the point most sharply — its sparsity *reduced* timing dependence even after
the deletion control was partialled out.

**What v2 changes.** The penalty becomes a **two-sided band applied per (sample,
neuron)** rather than a one-sided hinge on the batch-mean rate::

    per_sample_rate = hidden_spikes.sum(dim=-1)              # (B, C, 1, 1)
    penalty = (relu(per_sample_rate - band_hi)               # too many spikes
               + under_weight * relu(band_lo - per_sample_rate)).mean()

Two consequences, both deliberate:

1. **The lower arm closes the escape hatch.** Falling silent now *costs* ``band_lo``,
   so neurons are pushed to fire on every sample. With ``silent_fraction`` driven
   toward 0 the population-identity channel carries almost nothing, and with the
   count pinned inside a narrow band the count channel carries ~1 bit. **Timing
   becomes the only rich channel left** — H1's premise, true by construction.
2. **Per-sample, not batch-mean.** v1's ``per_neuron_rate.mean(dim=0)`` averaged over
   the batch first, so a neuron firing 10 spikes on 1 sample in 128 registered as
   rate 0.078 and drew *no* penalty at all. Bursty, highly selective neurons were
   free under v1; here each (sample, neuron) pair is charged on its own.

**The band is the swept axis.** Because both arms of the band bind, the band
*position* sets the achieved spikes-per-active-neuron directly — unlike v1, where the
target was inert and the penalty strength had to be swept. v2 therefore sweeps
``SPARSITY_BANDS`` at a fixed ``PENALTY_STRENGTH``. As always, analyse against the
**measured** firing statistics, never the band.

**What differs from the with-delay v2 script.** Only the network, exactly as in v1:
``delay1``/``delay2``, the adaptive delay-clamping schedule and the ``delay_mean``
log field are removed outright rather than switched off by a flag, so the class is
``SparseSHDNetworkNoDelay`` and spikes go straight ``fc1 -> fc2 -> fc3``. Any
surviving sparsity -> timing-dependence trend must therefore come from the SRM
neurons' own membrane dynamics rather than from learnable axonal delays. Everything
else is held identical to the with-delay arm — including, for the first time, the
warm-up (see ``WARMUP_EPOCHS``).

**Dead-neuron caveat.** The lower arm is a *preventive* force, not a reviving one:
SLAYER's surrogate gradient vanishes for a neuron whose membrane sits far below
threshold, so a deeply silent neuron may not be recoverable. This is why the penalty
runs from epoch 0 with no warm-up — neurons must never be allowed to go deeply silent
in the first place. This arm collapsed to silence under v1's hinge at warm-up 0 and
needed a 15-epoch guard; watch ``train_silent`` in the per-epoch log to confirm the
band's lower arm has replaced that guard successfully.

Expect **lower clean accuracy** than the with-delay arm throughout (v1: ~.49-.59 vs
~.78-.89). That is not a failure of the sparsity mechanism — it is why the analysis
uses a chance-corrected, baseline-normalised score and compares the arms at matched
firing rate, never on raw accuracy drops.

Outputs are tagged ``_v2_`` throughout and can never collide with v1 artifacts.

This is a **training-only** script. The perturbations that measure temporal
processing are applied **only at evaluation**, by the sibling experiments under
``exp_sparse_network/{jitter,shift,shd,deletion}/`` — never switch one on here. See:

- ``my_project/docs/progress/sparse_network_1stLayer_results.md`` — v1 results, and
  why v2 exists.
- ``my_project/docs/progress/sparse_network_test_progress.md`` — this milestone.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.

What this script produces, per ``(band, seed)``:

- a checkpoint ``sn_data/sparse_whole_nodelay_v2_band{lo}-{hi}_str{s}_seed{seed}.pt``;
- a training-curve log ``sn_log/..._training_log.json`` including ``train_silent``
  and ``train_sp_active``, the two diagnostics v1 lacked;
- a row in ``sn_log/sparse_whole_nodelay_v2_train_summary.json`` recording clean test
  accuracy and the achieved sparsity — including ``spikes_per_active_neuron``, the
  statistic whose absence hid v1's mechanism for the whole of Step 1.
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
# Quick pipeline check / probe. v2's band penalty is UNVALIDATED, and this is the
# riskier of the two arms: it fires ~half as much as the delay net intrinsically, and
# under v1's hinge it collapsed to silence at warm-up 0 (needing a 15-epoch guard).
# Run the probe FIRST (see resolve_run_config) and check four things before
# committing to the full run:
#   (1) does it train at all at the tight band [1, 2]?
#   (2) does silent_fraction actually fall toward 0 -- i.e. has the band's lower arm
#       replaced the warm-up guard? (if not, raise UNDER_WEIGHT, or re-add a short
#       warm-up);
#   (3) does spikes_per_active_neuron land inside the band? (if it sits above
#       band_hi, the band is not binding -> raise PENALTY_STRENGTH);
#   (4) does train_silent ever spike toward 1.0 early on (the v1 collapse signature)?
# Probe artifacts carry the _probe suffix (RUN_SUFFIX) so they can never clobber a
# real run.
QUICK_TEST: bool = True

# Suffix appended to every output name (checkpoints, per-model logs, summary) when
# running a probe, so a QUICK_TEST probe can never overwrite real-run artifacts that
# share the same (band, strength, seed). Empty for the real run.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Marks every v2 artifact so it can never collide with a v1 checkpoint or summary.
VERSION_TAG: str = "v2"

# --- Milestone scope (deliberately narrow; see progress doc §3) ---
DATASET_KEY: str = "whole"
DELAY_TAG: str = "nodelay"    # tags checkpoints/logs apart from the delay run
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# --- Sparsity sweep: sweep the BAND, hold the strength fixed ---
# This inverts v1's arrangement. Under v1's one-sided hinge the target was inert (the
# task loss parked firing above any target) so the penalty STRENGTH had to be swept.
# A two-sided band binds from both directions, so the band position sets the achieved
# spikes-per-active-neuron directly and becomes the natural swept axis.
#
# Each band is (lo, hi) in spikes per neuron per sample. [1, 2] is the regime H1 is
# actually about: count carries ~1 bit, identity carries ~nothing (silent_fraction
# should be near 0), so timing is the only rich channel left. The wider/denser bands
# provide the sparsity gradient the headline plot needs.
#
# Held identical to the with-delay arm's grid so the two are directly comparable.
# Note this arm's *natural* (unpenalised) firing is ~6.3 spikes/neuron, so the
# densest band [8, 12] sits above it and its lower arm will be the binding one --
# that is intended, and gives a genuine dense anchor rather than an inert point.
#
# Always analyse against the *measured* firing statistics, never the band.
SPARSITY_BANDS: list[tuple[float, float]] = [
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
    (5.0, 8.0),
    (8.0, 12.0),
]
SEEDS: list[int] = [42, 43, 44]

# Band penalty coefficient. FIXED in v2 (the band is the swept axis). It only has to
# be large enough for the band to bind; v1 evidence sets the scale -- at strength 10
# the v1 target became binding (spikes/active-neuron converged to target + 0.1), at
# strength 1 it did not (target 3, achieved 5.39). 3.0 is the starting point; the
# probe's job is to confirm the achieved rate lands inside the band. If it sits above
# band_hi, raise this; if clean accuracy collapses, lower it.
PENALTY_STRENGTH: float = 3.0

# Weight on the band's LOWER arm (the anti-silencing term) relative to the upper one.
# 1.0 keeps the band symmetric. Raise it if the probe shows silent_fraction refusing
# to fall; lower it if forcing every neuron active costs too much accuracy. This is
# the knob that trades off "kill the identity channel" against "keep the task
# solvable", and it is the one genuinely new hyper-parameter in v2. This arm is the
# likelier of the two to need it raised -- it is the weaker network and reached the
# higher silent fraction under v1.
UNDER_WEIGHT: float = 1.0

# Clean warm-up before the penalty engages, in epochs. 0 for BOTH v2 arms.
# This CHANGES this arm's v1 recipe, which used 15. That guard existed because v1's
# hinge had no lower arm: nothing stopped firing overshooting into silence, and at
# warm-up 0 this arm went fully silent for ~26 epochs and recovered degenerate. v2's
# band supplies the missing lower arm structurally, so the warm-up's safety role is
# redundant -- and dropping it removes the standing comparability caveat between the
# two arms (v1 ran this one at 15 and the delay arm at 0). It matters more than in v1
# that the penalty runs from epoch 0: the lower arm prevents silence but cannot
# reliably reverse it once the surrogate gradient has vanished. If the probe shows
# train_silent spiking toward 1.0 in the first epochs, re-add a short warm-up (10-15)
# here and match it in the delay arm.
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


def resolve_run_config() -> tuple[
    list[tuple[float, float]], float, list[int], int, int
]:
    """Return (bands, penalty_strength, seeds, epochs, warmup_epochs) for this run.

    Collapses to a small, fast probe grid when ``QUICK_TEST`` is set so v2's
    unvalidated band penalty can be checked cheaply before the full sweep.

    Returns:
        Tuple of (bands, penalty_strength, seeds, epochs, warmup_epochs).
    """
    if QUICK_TEST:
        # v2 VALIDATION PROBE. Three bands spanning the sweep, one seed, 400 epochs
        # (~1.25 h/model on this arm). This is a feasibility check, not a
        # calibration: reduced-epoch firing UNDER-ESTIMATES the full run (v1 lesson
        # `sparsity-penalty-redensifies-at-full-epochs`), so never lock a grid from
        # these numbers. Read off, per model:
        #   - silent_fraction -> did the lower arm close the identity escape hatch,
        #     and replace the warm-up guard this arm needed under v1?
        #   - spikes_per_active_neuron -> is the band binding (inside [lo, hi])?
        #   - clean_acc -> is the tight [1, 2] band still learnable, or has forcing
        #     every neuron active broken the task?
        # The tight band is the risky one; it is first so a failure shows up early.
        return [(1.0, 2.0), (3.0, 5.0), (8.0, 12.0)], PENALTY_STRENGTH, [42], 400, 0
    return SPARSITY_BANDS, PENALTY_STRENGTH, SEEDS, EPOCHS, WARMUP_EPOCHS


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


class SparseSHDNetworkNoDelay(nn.Module):
    """2-hidden-layer SLAYER SNN (no delays), for the v2 band experiment.

    Structurally identical to v1's class of the same name — only the training-loop
    penalty differs between versions — so the existing eval scripts load v2
    checkpoints unchanged. Spikes propagate straight from one dense layer to the
    next, with no ``delay1``/``delay2``, so the only timing machinery is the SRM
    neurons' own membrane dynamics.

    Training uses the clean ``forward`` (no perturbation). ``forward`` can optionally
    return the 1st hidden layer's spike tensor so the training loop can apply the
    band sparsity penalty to it. The eval-only perturbations of that same layer live
    under ``exp_sparse_network/{jitter,shift,shd,deletion}/``, not here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 128,
        num_classes: int = 20,
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

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Clean forward pass (used during training and clean evaluation).

        Args:
            x: Input spike trains.
            return_hidden: If True, also return the 1st hidden layer spikes so the
                caller can apply the sparsity penalty.

        Returns:
            The output spike tensor, or ``(output, hidden1)`` if ``return_hidden``
            is True.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        out = self._second_hidden_and_output(hidden1)
        return (out, hidden1) if return_hidden else out


def hidden_mean_rate(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """Mean spikes per hidden neuron per sample (raw firing, for logging).

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).

    Returns:
        Scalar tensor (mean spikes per neuron per sample).
    """
    return hidden_spikes.sum(dim=-1).mean()


def hidden_activity_stats(hidden_spikes: torch.Tensor) -> tuple[float, float]:
    """Return (silent_fraction, spikes_per_active_neuron) for one batch.

    These are the two diagnostics v1 lacked. ``spikes_per_neuron`` alone is diluted
    by silent neurons and hid the fact that v1's sparsest models still fired 3-4
    spikes per *active* neuron — far from the 0-2 regime H1 assumes. Watching both
    per epoch is how v2's band is confirmed to be doing what it claims, and how this
    arm's v1 collapse signature (train_silent spiking toward 1.0) is caught early.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).

    Returns:
        Tuple of (silent_fraction, spikes_per_active_neuron). The latter is 0.0 if
        the whole batch is silent.
    """
    counts = hidden_spikes.sum(dim=-1).detach()
    active = counts > 0
    n_active = int(active.sum().item())
    silent_fraction = 1.0 - (n_active / max(1, counts.numel()))
    sp_active = counts[active].mean().item() if n_active > 0 else 0.0
    return silent_fraction, sp_active


def hidden_rate_band(
    hidden_spikes: torch.Tensor,
    band_lo: float,
    band_hi: float,
    under_weight: float = UNDER_WEIGHT,
) -> torch.Tensor:
    """Two-sided band sparsity penalty on per-(sample, neuron) spike count.

    For every (sample, neuron) pair, charge the amount by which its spike count
    falls *outside* the band ``[band_lo, band_hi]``::

        per_sample_rate = hidden_spikes.sum(dim=-1)              # (B, C, 1, 1)
        penalty = (relu(per_sample_rate - band_hi)
                   + under_weight * relu(band_lo - per_sample_rate)).mean()

    This differs from v1's ``hidden_rate_hinge`` in two ways, both essential:

    **The lower arm closes the identity escape hatch.** v1's hinge had zero gradient
    at and below its target, so a neuron could dodge the penalty completely by going
    silent — and did, arriving at sparsity partly through stimulus selectivity that
    strengthened a perturbation-immune population-identity code. Here silence costs
    ``under_weight * band_lo``, so neurons are pushed to fire on every sample. Drive
    ``silent_fraction`` toward 0 and the identity channel carries almost nothing;
    pin the count inside a narrow band and the count channel carries ~1 bit; what
    remains for the network to use is *when* the spikes occur.

    **Charging per sample, not per batch-mean.** v1 averaged each neuron's count over
    the batch *before* the ReLU, so a neuron firing 10 spikes on 1 sample in 128
    registered as rate 0.078 and drew no penalty at all — bursty, highly selective
    neurons were free. Each (sample, neuron) pair is now charged on its own.

    SLAYER's spike function is surrogate-gradient differentiable, so the penalty
    propagates back to ``fc1``. Note the lower arm is *preventive*, not reviving: a
    neuron whose membrane has fallen far below threshold has no surrogate gradient
    left to push on, which is why v2 runs with no warm-up.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).
        band_lo: Lower edge of the target band, in spikes per neuron per sample.
        band_hi: Upper edge of the target band. Must be >= ``band_lo``.
        under_weight: Weight on the lower (anti-silencing) arm relative to the upper.

    Returns:
        Scalar penalty tensor (mean over (sample, neuron) pairs of the out-of-band
        excess).
    """
    per_sample_rate = hidden_spikes.sum(dim=-1)
    over = torch.relu(per_sample_rate - band_hi)
    under = torch.relu(band_lo - per_sample_rate)
    return (over + under_weight * under).mean()


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
    band: tuple[float, float],
    seed: int,
    epochs: int,
    penalty_strength: float = PENALTY_STRENGTH,
    under_weight: float = UNDER_WEIGHT,
    warmup_epochs: int = WARMUP_EPOCHS,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[SparseSHDNetworkNoDelay, dict]:
    """Train one clean model with the two-sided band hidden-sparsity penalty.

    The forward pass is clean (no perturbation). For the first ``warmup_epochs``
    the penalty is off; after that the total loss is the NumSpikes task loss plus
    ``penalty_strength`` times the band penalty. Best-model selection and early
    stopping use the *task* validation loss and are reset when the penalty engages,
    so the saved checkpoint is the most accurate *banded* model.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        band: ``(lo, hi)`` target band in spikes per neuron per sample.
        seed: Random seed (controls init and shuffle order).
        epochs: Maximum training epochs.
        penalty_strength: Band penalty coefficient.
        under_weight: Weight on the band's anti-silencing lower arm.
        warmup_epochs: Clean epochs before the penalty engages (0 in v2).
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        lr: Learning rate.
        patience: Early stopping patience (on task val loss).

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)
    band_lo, band_hi = band

    net = SparseSHDNetworkNoDelay(input_dim, hidden_units, num_classes).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    log = {
        "epoch": [],
        "train_loss": [],        # total (task + penalty)
        "train_task_loss": [],
        "train_rate": [],        # mean spikes / neuron / sample (raw firing)
        "train_silent": [],      # fraction of (sample, neuron) pairs with no spikes
        "train_sp_active": [],   # mean spikes among ACTIVE (sample, neuron) pairs
        "val_loss": [],          # task only
        "val_acc": [],
    }

    desc = f"Train nodelay band=[{band_lo:g},{band_hi:g}] seed={seed}"
    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=desc) as pbar:
        for epoch in range(epochs):
            # Reset best-model tracking when the penalty engages, so the saved
            # checkpoint is the best *banded* model, not the warm-up one.
            if epoch == warmup_epochs and warmup_epochs > 0:
                best_val_loss = float("inf")
                best_model_state = None
                early_stop_counter = 0

            penalty_on = epoch >= warmup_epochs

            # --- Train (clean forward + band sparsity penalty once warmed up) ---
            net.train()
            batch_total_losses = []
            batch_task_losses = []
            batch_rates = []
            batch_silent = []
            batch_sp_active = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs, hidden1 = net(x_batch, return_hidden=True)
                task_loss = loss_fn.numSpikes(outputs, target)
                rate = hidden_mean_rate(hidden1)
                silent, sp_active = hidden_activity_stats(hidden1)
                if penalty_on:
                    penalty = hidden_rate_band(
                        hidden1, band_lo, band_hi, under_weight
                    )
                    loss = task_loss + penalty_strength * penalty
                else:
                    loss = task_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_total_losses.append(loss.item())
                batch_task_losses.append(task_loss.item())
                batch_rates.append(rate.item())
                batch_silent.append(silent)
                batch_sp_active.append(sp_active)
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
            train_silent = float(np.mean(batch_silent))
            train_sp_active = float(np.mean(batch_sp_active))

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["train_task_loss"].append(train_task_loss)
            log["train_rate"].append(train_rate)
            log["train_silent"].append(train_silent)
            log["train_sp_active"].append(train_sp_active)
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))

            pbar.set_postfix(
                epoch=epoch + 1,
                pen="on" if penalty_on else "off",
                task=f"{train_task_loss:.3f}",
                rate=f"{train_rate:.2f}",
                act=f"{train_sp_active:.2f}",
                sil=f"{train_silent:.0%}",
                acc=f"{val_acc:.2%}",
            )
            scheduler.step()

            # Early stopping on task val loss
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

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


@torch.no_grad()
def evaluate_clean_and_activity(
    net: SparseSHDNetworkNoDelay,
    test_loader: DataLoader,
) -> dict:
    """Measure clean test accuracy and 1st hidden layer sparsity in one pass.

    Both are computed on the clean forward pass (no perturbation). Alongside v1's
    metrics this reports ``spikes_per_active_neuron`` — the statistic whose absence
    hid v1's mechanism, and the one that says whether the band actually bound.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.

    Returns:
        Dict with keys ``clean_acc``, ``firing_rate``, ``spikes_per_neuron``,
        ``spikes_per_active_neuron``, ``silent_fraction``.
    """
    net.eval()
    correct = 0
    total = 0
    total_spikes = 0.0
    total_slots = 0
    total_neuron_samples = 0
    silent_neuron_samples = 0.0

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
        y_batch = y_batch.to(device)

        outputs, hidden1 = net(x_batch, return_hidden=True)

        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)

        B, C, _, _, T = hidden1.shape
        per_neuron_counts = hidden1.sum(dim=-1).view(B, C)  # (sample, neuron)
        total_spikes += per_neuron_counts.sum().item()
        total_slots += B * C * T
        total_neuron_samples += B * C
        silent_neuron_samples += (per_neuron_counts == 0).sum().item()

    active_neuron_samples = total_neuron_samples - silent_neuron_samples
    return {
        "clean_acc": correct / max(1, total),
        "firing_rate": total_spikes / max(1, total_slots),
        "spikes_per_neuron": total_spikes / max(1, total_neuron_samples),
        "spikes_per_active_neuron": total_spikes / max(1.0, active_neuron_samples),
        "silent_fraction": silent_neuron_samples / max(1, total_neuron_samples),
    }


def run_milestone() -> None:
    """Train the (band, seed) grid and record sparsity + clean accuracy."""
    bands, penalty_strength, seeds, epochs, warmup_epochs = resolve_run_config()

    n_models = len(bands) * len(seeds)
    print(f"{'#' * 70}")
    print("# Sparse-network milestone v2 (training): SHD whole, SGD no-delay")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs} | warmup={warmup_epochs}")
    print(f"# penalty=two-sided band | strength: {penalty_strength} | "
          f"under_weight: {UNDER_WEIGHT}")
    print(f"# bands: {bands}")
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

    for band in bands:
        band_lo, band_hi = band
        for seed in seeds:
            run_tag = (
                f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_"
                f"band{band_lo:g}-{band_hi:g}_str{penalty_strength:g}_"
                f"seed{seed}{RUN_SUFFIX}"
            )
            print(f"\n{'=' * 60}")
            print(f"  Training {run_tag}")
            print(f"{'=' * 60}")

            net, training_log = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                band=band,
                seed=seed,
                epochs=epochs,
                penalty_strength=penalty_strength,
                under_weight=UNDER_WEIGHT,
                warmup_epochs=warmup_epochs,
            )

            ckpt_path = DATA_DIR / f"{run_tag}.pt"
            torch.save(net.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

            metrics = evaluate_clean_and_activity(net, test_loader)
            in_band = band_lo <= metrics["spikes_per_active_neuron"] <= band_hi
            print(
                f"  clean_acc={metrics['clean_acc']:.4f} | "
                f"spikes/neuron={metrics['spikes_per_neuron']:.2f} | "
                f"spikes/ACTIVE={metrics['spikes_per_active_neuron']:.2f} "
                f"({'in band' if in_band else 'OUT OF BAND'}) | "
                f"silent={metrics['silent_fraction']:.2%}"
            )

            summary[run_tag] = {
                "band_lo": float(band_lo),
                "band_hi": float(band_hi),
                "penalty_strength": float(penalty_strength),
                "under_weight": float(UNDER_WEIGHT),
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

    # Final table. The v2 acceptance criteria are: spikes/ACTIVE inside the band,
    # silent_fraction near 0 (the identity channel is closed), and clean_acc well
    # above chance. Read this table against all three, not against clean_acc alone.
    print(
        f"\n{'run_tag':<62} {'clean_acc':>9} {'sp/neuron':>10} "
        f"{'sp/ACTIVE':>10} {'silent':>8}"
    )
    for run_tag, row in summary.items():
        print(
            f"{run_tag:<62} {row['clean_acc']:>9.4f} "
            f"{row['spikes_per_neuron']:>10.2f} "
            f"{row['spikes_per_active_neuron']:>10.2f} "
            f"{row['silent_fraction']:>8.2%}"
        )


if __name__ == "__main__":
    run_milestone()
