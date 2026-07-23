"""First milestone (training), NO-DELAY variant: induce hidden-layer sparsity on SHD.

Companion to [jitter_test_withDelay.py](jitter_test_withDelay.py): same dataset,
same sparsity mechanism, same protocol — the only difference is the network, which
is plain SGD with **no learnable delays** (``delay1``/``delay2`` and the adaptive
delay clamping are removed entirely rather than switched off by a flag).

Why run it. Learnable axonal delays are themselves a timing mechanism, so in the
SGD-delay net the jitter test measures sparsity's effect *on top of* whatever the
delays contribute. Stripping the delays removes that confound: any surviving
sparsity -> timing-dependence trend must come from the neurons' own membrane
dynamics. Comparing the two runs also isolates how much of the temporal processing
the delays were responsible for.

Trains 2-hidden-layer SLAYER SNNs on SHD ``whole`` under the **fixed-weight**
protocol: training is on *clean* data (no jitter), with a one-sided **hinge**
penalty that pushes the 1st hidden layer's firing down toward a target rate
(applied from the start of training). A fresh model is trained for every
``(target_rate, penalty_strength, seed)`` combination and its checkpoint is saved.

A plain L1 penalty does not work here: under Adam/Nadam it is rescaled to
full-size suppression steps and silences the layer within one epoch regardless
of its coefficient (a coefficient of 1e-4 already drives firing to zero). The
hinge penalises only firing *above* ``target_rate`` and has zero gradient at or
below it, so it trims excess without driving neurons to silence. It is applied
from the start of training: an earlier variant delayed the penalty behind a clean
warm-up, but that parked the net in a dense solution the hinge could not escape,
so the warm-up was dropped (see the progress log).

The timing perturbation that measures "temporal processing" is applied **only at
evaluation** by the companion sigma-sweep script — it must never be switched on
here. See:

- ``my_project/docs/progress/sparse_network_test_progress.md`` — this milestone.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.
- ``my_project/docs/knowledge_bank/phase1_investigation.md`` — why the
  fixed-weight protocol (clean train / perturb at eval) is the correct lens.

What this script produces, per ``(target_rate, penalty_strength, seed)``:

- a checkpoint ``data/sparse_whole_nodelay_tgt{target}_str{strength}_seed{seed}.pt``;
- a training-curve log ``log/..._training_log.json``;
- a row in ``log/sparse_whole_nodelay_train_summary.json`` recording the achieved
  hidden sparsity (firing rate) and clean test accuracy — the two numbers used to
  confirm the strength grid still spans a usable sparsity range without delays.
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
DATA_DIR = SCRIPT_DIR / "data"          # checkpoints written here
LOG_DIR = SCRIPT_DIR / "log"            # training logs + summary written here
SHD_DATA_DIR = SCRIPT_DIR / "../shd/shd_data"  # SHD .mat source (this experiment's own copy)

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Quick pipeline check / calibration probe. Calibration is COMPLETE (probes 1-4,
# 2026-07-22/23; full narrative in the progress log). It took four probes because
# the no-delay net's strength->firing map does not transfer from the delay arm: it
# fires ~half as much, and at warm-up 0 its response is flat-then-cliff (weak
# penalties inert, strong ones collapse the layer to silence at init). The fix was a
# 15-ep warm-up guard (WARMUP_EPOCHS) plus a shifted-up strength range; probe 4 then
# measured a clean monotonic 5.8->1.2 sp/neuron spread, no collapse, all above
# chance, and that grid is now LOCKED into PENALTY_STRENGTHS. False = the real
# 15-model run (5 strengths x 3 seeds, EPOCHS). Set True only to re-enter a probe via
# resolve_run_config (its grid is now a pipeline smoke-test, not calibration).
QUICK_TEST: bool = False

# --- Milestone scope (deliberately narrow; see progress doc §3) ---
# Network: plain SGD, no learnable delays — that is the whole point of this file.
# Everything else is held identical to the with-delay run so the two are comparable.
DATASET_KEY: str = "whole"
DELAY_TAG: str = "nodelay"    # tags checkpoints/logs apart from the delay run
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# --- Sparsity sweep: hold the target FIXED, sweep the penalty STRENGTH ---
# Calibration (on the delay net) showed the target is NOT the binding control: the
# task loss pulls hidden firing up to a plateau ABOVE any target, and the hinge only
# lowers where that plateau sits (partly by silencing neurons). The penalty STRENGTH
# is what actually sets the achieved sparsity -- cleanly, monotonically, and without
# collapse -- so we fix one aggressive target and sweep strength (see
# PENALTY_STRENGTHS). Always analyse against the *measured* firing rate, since the
# achieved rate is seed-dependent.
SPARSITY_TARGETS: list[float] = [3.0]
SEEDS: list[int] = [42, 43, 44]

# Hinge penalty coefficient — the SWEPT sparsity axis (target held fixed).
# LOCKED (2026-07-23) from probe 4 (warm-up 15, seed 42, 400 ep), which measured a
# clean monotonic spread with no collapse (spikes/neuron, clean_acc): 1e-2 -> 5.80,
# 58% ; 5e-1 -> 4.19, 55% ; 1.0 -> 2.80, 53% ; 3.0 -> 1.84, 51% ; 10.0 -> 1.23, 47%.
# That is a ~4.7x firing spread (~5.1x including the ~6.3 str=0 anchor), every model
# well above chance, silent_fraction topping out at 64% (not degenerate). This grid
# differs completely from the delay arm's [1e-3..1.0]: the no-delay net fires ~half
# as much and its response is flat-then-cliff at warm-up 0, so it needs both the
# warm-up (WARMUP_EPOCHS) and this shifted-up strength range (probes 1-4, progress
# log). Always analyse against the *measured* firing rate, not these coefficients
# (the map is nonlinear and seed-dependent).
PENALTY_STRENGTHS: list[float] = [1e-2, 5e-1, 1.0, 3.0, 10.0]

# Clean warm-up before the hinge engages, in epochs. 15: the with-delay arm needs
# no warm-up (it fires ~11 sp/neuron and absorbs the penalty from epoch 0), but the
# no-delay net fires only ~6 and, at warm-up 0, its strength->firing response is
# flat-then-cliff: penalties <=1e-2 barely move firing while >=3e-2 crush the layer
# to silence at init (rate=0 for tens of epochs) with only partial, fragile recovery
# -- so no clean sparsity gradient exists (probes 1-2, progress log 2026-07-22/23).
# A SHORT warm-up lets the task loss first establish the ~6-spike pattern so a strong
# hinge then trims it gradually instead of crushing it at init. It must stay short:
# a long warm-up (100-200 ep) parks the net dense and the additive hinge can't move
# it (attempt 3). 15 is a starting guess -- confirm probe 3 gives a monotonic spread
# with no rate=0 transit, and lengthen only if the sparse end still collapses. Keep
# below EARLY_STOP_PATIENCE (300).
WARMUP_EPOCHS: int = 15

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

# --- Data split ratios (identical to the rest of the project) ---
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# --- Training hyper-parameters (identical to the with-delay run, for comparability) ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
EARLY_STOP_PATIENCE: int = 300


def resolve_run_config() -> tuple[list[float], list[float], list[int], int, int]:
    """Return (target_rates, penalty_strengths, seeds, epochs, warmup_epochs).

    Collapses to a tiny, fast grid when ``QUICK_TEST`` is set so the pipeline can
    be validated cheaply before the full milestone.

    Returns:
        Tuple of (target_rates, penalty_strengths, seeds, epochs, warmup_epochs).
    """
    if QUICK_TEST:
        # FOURTH probe: one seed, 400 epochs, WITH the 15-ep warm-up (WARMUP_EPOCHS).
        # Probe 3 (same warm-up, grid [1e-2..1.0]) showed the warm-up killed the
        # collapse (no run transits rate=0) but over-corrected into the attempt-3
        # dense-lock: penalties 1e-2..3e-1 (a 30x range) ALL settle at ~5.5-5.8
        # sp/neuron -- the task loss out-pulls the hinge in the dense basin -- and
        # only str=1.0 escapes, to 2.80 (cleanly, 54% silent, not degenerate). So
        # with the warm-up the penalty only starts biting near ~1.0, which makes
        # STRENGTH a clean monotone knob at the HIGH end. This probe moves the sweep
        # there: 1e-2 anchors the dense end (~5.8; the true str=0 anchor ~6.3 is
        # warm-up-independent and already known), 5e-1 fills the 5.8->2.8 transition,
        # and 1.0/3.0/10.0 map how far the sparse end goes while staying collapse-free.
        # Read spikes/neuron, clean_acc and the train_rate trajectory: we want a
        # monotonic spread with NO rate=0 transit, and watch whether 3.0/10.0 keep
        # descending (good) or saturate near 2.8 (that is then the sparse floor). If
        # it works, lock ~5 values (anchoring the dense end with str=0's ~6.3) into
        # PENALTY_STRENGTHS and set QUICK_TEST = False.
        return [3.0], [1e-2, 5e-1, 1.0, 3.0, 10.0], [42], 400, WARMUP_EPOCHS
    return SPARSITY_TARGETS, PENALTY_STRENGTHS, SEEDS, EPOCHS, WARMUP_EPOCHS


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
    every model (a requirement for comparing their perturbation curves, and for
    comparing this run against the with-delay one). Only the training order is
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


class JitterSHDNetworkNoDelay(nn.Module):
    """2-hidden-layer SLAYER SNN (no delays) for the fixed-weight sparsity experiment.

    Identical to the with-delay network minus ``delay1``/``delay2``: spikes
    propagate straight from one dense layer to the next, so the only timing
    machinery left is the SRM neurons' own membrane dynamics.

    Training uses the clean ``forward`` (no jitter). ``forward`` can optionally
    return the 1st hidden layer's spike tensor so the training loop can add a
    hinge sparsity penalty to it. ``forward_with_hidden_perturbation`` jitters that
    same 1st hidden layer and is used **only at evaluation** by the sweep script;
    it is never called during training.
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
            The output spike tensor, or ``(output, hidden1)`` if
            ``return_hidden`` is True.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        out = self._second_hidden_and_output(hidden1)
        return (out, hidden1) if return_hidden else out

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Eval-only: jitter the 1st hidden layer spikes before the readout.

        Call inside ``torch.no_grad()``. Never used during training — see the
        protocol rule in the module docstring.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if sigma > 0:
            hidden1 = jitter_hidden_batch(hidden1, sigma)
        return self._second_hidden_and_output(hidden1)


@torch.no_grad()
def jitter_hidden_batch(
    hidden_spikes: torch.Tensor,
    sigma: float,
    max_attempts: int = 50,
) -> torch.Tensor:
    """Vectorised GPU-side per-spike Gaussian jitter (eval-only helper).

    For each spike, draw an iid Gaussian offset ``~ N(0, sigma)``, shift the
    spike by ``round(offset)`` and clip to ``[0, T - 1]``. Collisions are
    resolved by a priority tiebreaker with a bounded retry budget; any spike
    still unplaced falls back to its original bin. Per-neuron spike count is
    preserved, so the perturbation destroys only timing, not rate. Copied
    verbatim from the fixed-weight jitter scripts; included here so the eval
    sweep can import a single, canonical network definition.

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
        target = (t_idx + offsets).round().long().clamp(0, T - 1)

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


def hidden_mean_rate(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """Mean spikes per hidden neuron per sample (raw firing, for logging).

    ``hidden_spikes`` is the binary 1st-hidden output of shape (B, C, 1, 1, T).
    Summing over time gives each neuron's spike count; averaging over batch and
    neurons gives a scalar. This is the quantity to watch per epoch to confirm
    firing settles at a plateau rather than collapsing to zero.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).

    Returns:
        Scalar tensor (mean spikes per neuron per sample).
    """
    return hidden_spikes.sum(dim=-1).mean()


def hidden_rate_hinge(hidden_spikes: torch.Tensor, target: float) -> torch.Tensor:
    """One-sided (hinge) sparsity penalty toward a target per-neuron firing rate.

    For each hidden neuron, take its mean spike count across the batch and
    penalise only the amount by which it *exceeds* ``target``::

        per_neuron_rate = hidden_spikes.sum(dim=-1).mean(dim=0)   # (C, 1, 1)
        penalty = relu(per_neuron_rate - target).mean()

    Because the ReLU has zero gradient at or below ``target``, a neuron already
    at/under the target gets no downward push and a silent neuron gets none at
    all. This structurally removes the drive-to-zero cascade that a plain L1
    penalty suffers under Adam/Nadam (which rescales the persistent L1 gradient
    to full-size suppression steps and silences the layer within one epoch).
    ``target`` acts as a floor the penalty stops pushing below, not as the achieved
    rate: the task loss holds firing at a plateau *above* it, and the penalty
    *strength* is what sets where that plateau lands (see the calibration in the
    progress log). SLAYER's spike function is surrogate-gradient differentiable, so
    the penalty propagates back to ``fc1``.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).
        target: Desired upper bound on mean spikes per neuron per sample.

    Returns:
        Scalar penalty tensor (mean over neurons of the above-target excess).
    """
    per_neuron_rate = hidden_spikes.sum(dim=-1).mean(dim=0)
    return torch.relu(per_neuron_rate - target).mean()


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
    net: JitterSHDNetworkNoDelay,
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
    target_rate: float,
    seed: int,
    epochs: int,
    penalty_strength: float = PENALTY_STRENGTHS[0],
    warmup_epochs: int = WARMUP_EPOCHS,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[JitterSHDNetworkNoDelay, dict]:
    """Train one clean no-delay model with a hinge hidden-sparsity penalty.

    The forward pass is clean (no jitter). For the first ``warmup_epochs`` the
    penalty is off so the task loss can establish firing; after that the total
    loss is the NumSpikes task loss plus ``penalty_strength`` times the hinge
    penalty toward ``target_rate``. ``warmup_epochs`` is 0 by default — the warm-up
    was found to prevent sparsity. Best-model selection and early stopping use
    the *task* validation loss and are reset when a nonzero warm-up ends, so the
    saved checkpoint is the most accurate *sparsified* model, not the denser
    warm-up one.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        target_rate: Target mean spikes per neuron per sample; the hinge only
            penalises firing above this.
        seed: Random seed (controls init and shuffle order).
        epochs: Maximum training epochs.
        penalty_strength: Hinge penalty coefficient — the binding sparsity control.
        warmup_epochs: Clean epochs before the penalty engages.
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        lr: Learning rate.
        patience: Early stopping patience (on task val loss).

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = JitterSHDNetworkNoDelay(input_dim, hidden_units, num_classes).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    log = {
        "epoch": [],
        "train_loss": [],       # total (task + penalty)
        "train_task_loss": [],
        "train_rate": [],       # mean spikes / neuron / sample (raw firing)
        "val_loss": [],         # task only
        "val_acc": [],
    }

    total_steps = epochs * len(train_loader)
    desc = f"Train nodelay tgt={target_rate:g} str={penalty_strength:g} seed={seed}"
    with tqdm(total=total_steps, desc=desc) as pbar:
        for epoch in range(epochs):
            # Reset best-model tracking when the penalty engages, so the saved
            # checkpoint is the best *sparsified* model, not the warm-up one.
            if epoch == warmup_epochs and warmup_epochs > 0:
                best_val_loss = float("inf")
                best_model_state = None
                early_stop_counter = 0

            penalty_on = epoch >= warmup_epochs

            # --- Train (clean forward + hinge sparsity penalty once warmed up) ---
            net.train()
            batch_total_losses = []
            batch_task_losses = []
            batch_rates = []

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
                if penalty_on:
                    penalty = hidden_rate_hinge(hidden1, target_rate)
                    loss = task_loss + penalty_strength * penalty
                else:
                    loss = task_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_total_losses.append(loss.item())
                batch_task_losses.append(task_loss.item())
                batch_rates.append(rate.item())
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

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["train_task_loss"].append(train_task_loss)
            log["train_rate"].append(train_rate)
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))

            pbar.set_postfix(
                epoch=epoch + 1,
                pen="on" if penalty_on else "off",
                task=f"{train_task_loss:.3f}",
                rate=f"{train_rate:.2f}",
                val=f"{val_loss:.3f}",
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
    net: JitterSHDNetworkNoDelay,
    test_loader: DataLoader,
) -> dict:
    """Measure clean test accuracy and 1st hidden layer sparsity in one pass.

    Both are computed on the clean forward pass (no jitter). Sparsity is the
    fraction of active (neuron, time-bin) slots; also reports spikes per neuron
    per sample and the fraction of neuron-samples that are completely silent.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.

    Returns:
        Dict with keys ``clean_acc``, ``firing_rate``, ``spikes_per_neuron``,
        ``silent_fraction``.
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

    return {
        "clean_acc": correct / max(1, total),
        "firing_rate": total_spikes / max(1, total_slots),
        "spikes_per_neuron": total_spikes / max(1, total_neuron_samples),
        "silent_fraction": silent_neuron_samples / max(1, total_neuron_samples),
    }


def run_milestone() -> None:
    """Train the (target, strength, seed) grid and record sparsity + clean accuracy."""
    target_rates, penalty_strengths, seeds, epochs, warmup_epochs = (
        resolve_run_config()
    )

    n_models = len(target_rates) * len(penalty_strengths) * len(seeds)
    print(f"{'#' * 70}")
    print("# Sparse-network milestone (training): SHD whole, SGD no-delay")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs} | warmup={warmup_epochs}")
    print(f"# penalty=hinge | targets: {target_rates} | strengths: {penalty_strengths}")
    print(f"# seeds: {seeds}  ->  {n_models} models")
    print(f"{'#' * 70}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data and build loaders once; the split (and hence the test set) is
    # fixed, so all models are comparable.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    train_loader, val_loader, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=42,
    )

    summary: dict[str, dict] = {}

    for target_rate in target_rates:
        for strength in penalty_strengths:
            for seed in seeds:
                run_tag = (
                    f"sparse_{DATASET_KEY}_{DELAY_TAG}_"
                    f"tgt{target_rate:g}_str{strength:g}_seed{seed}"
                )
                print(f"\n{'=' * 60}")
                print(f"  Training {run_tag}")
                print(f"{'=' * 60}")

                net, training_log = train_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    target_rate=target_rate,
                    seed=seed,
                    epochs=epochs,
                    penalty_strength=strength,
                    warmup_epochs=warmup_epochs,
                )

                ckpt_path = DATA_DIR / f"{run_tag}.pt"
                torch.save(net.state_dict(), ckpt_path)
                print(f"Model saved to {ckpt_path}")

                metrics = evaluate_clean_and_activity(net, test_loader)
                print(
                    f"  clean_acc={metrics['clean_acc']:.4f} | "
                    f"firing_rate={metrics['firing_rate']:.4f} | "
                    f"spikes/neuron={metrics['spikes_per_neuron']:.2f} | "
                    f"silent={metrics['silent_fraction']:.2%}"
                )

                summary[run_tag] = {
                    "target_rate": float(target_rate),
                    "penalty_strength": float(strength),
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
    summary_path = LOG_DIR / f"sparse_{DATASET_KEY}_{DELAY_TAG}_train_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Final table, for eyeballing calibration (aim: a wide spikes/neuron spread
    # across strengths with clean_acc well above chance everywhere).
    print(
        f"\n{'run_tag':<52} {'clean_acc':>9} {'firing_rate':>12} "
        f"{'spikes/neuron':>14}"
    )
    for run_tag, row in summary.items():
        print(
            f"{run_tag:<52} {row['clean_acc']:>9.4f} {row['firing_rate']:>12.4f} "
            f"{row['spikes_per_neuron']:>14.2f}"
        )


if __name__ == "__main__":
    run_milestone()
