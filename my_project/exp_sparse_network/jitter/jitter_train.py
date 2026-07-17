"""First milestone (training): induce hidden-layer sparsity on SHD.

Trains 2-hidden-layer SLAYER SNNs on SHD ``whole`` (SGD-delay) under the
**fixed-weight** protocol: training is on *clean* data (no jitter), with an L1
penalty on the 1st hidden layer's spikes to push that layer sparse. A fresh
model is trained for every ``(lambda, seed)`` pair and its checkpoint is saved.

The timing perturbation that measures "temporal processing" is applied **only at
evaluation** by the companion script (``jitter_eval.py`` / the sigma sweep) — it
must never be switched on here. See:

- ``my_project/docs/progress/sparse_network_progress.md`` — this milestone.
- ``my_project/docs/progress/sparse_network.md`` — full design and rationale.
- ``my_project/docs/knowledge_bank/phase1_investigation.md`` — why the
  fixed-weight protocol (clean train / perturb at eval) is the correct lens.

What this script produces, per ``(lambda, seed)``:

- a checkpoint ``data/sparse_whole_delay_lam{lam}_seed{seed}.pt``;
- a training-curve log ``log/..._training_log.json``;
- a row in ``log/sparse_whole_delay_train_summary.json`` recording the achieved
  hidden sparsity (firing rate) and clean test accuracy — the two numbers used
  to calibrate the ``lambda`` grid before committing to the full sweep.
"""

import os
import sys
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
# Quick pipeline check: when True, trains a tiny grid for few epochs so the whole
# pipeline (train -> checkpoint -> summary) can be validated in minutes before
# the real 15-model run. Set to False for the actual milestone.
QUICK_TEST: bool = False

# --- Milestone scope (deliberately narrow; see progress doc §3) ---
DATASET_KEY: str = "whole"
INPUT_DIM: int = 700          # SHD whole
USE_DELAY: bool = True        # SGD-delay
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# --- Sparsity sweep: L1 penalty strength on 1st hidden layer spikes ---
# Analyse results against the *measured* firing rate each lambda produces, never
# against lambda itself (the map is nonlinear and seed-dependent). These are
# starting guesses — calibrate with QUICK_TEST first (progress doc §1d).
LAM_VALUES: list[float] = [0.0, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS: list[int] = [42, 43, 44]

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

# --- Training hyper-parameters (copied from the fixed-weight jitter script) ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300


def resolve_run_config() -> tuple[list[float], list[int], int]:
    """Return the (lam_values, seeds, epochs) for this run.

    Collapses to a tiny, fast grid when ``QUICK_TEST`` is set so the pipeline can
    be validated cheaply before the full milestone.

    Returns:
        Tuple of (lam_values, seeds, epochs).
    """
    if QUICK_TEST:
        return [0.0, 1e-2], [42], 400
    return LAM_VALUES, SEEDS, EPOCHS


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
    every model (a requirement for comparing their perturbation curves). Only the
    training order is shuffled.

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


class JitterSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN for the fixed-weight sparsity experiment.

    Training uses the clean ``forward`` (no jitter). ``forward`` can optionally
    return the 1st hidden layer's spike tensor so the training loop can add an L1
    sparsity penalty to it. ``forward_with_hidden_perturbation`` jitters that
    same 1st hidden layer and is used **only at evaluation** by the sweep script;
    it is never called during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 128,
        num_classes: int = 20,
        use_delay: bool = True,
        max_delay: int = 64,
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

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike -> 1st hidden spikes (strictly binary)."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> (delay1) -> fc2 -> spike -> (delay2) -> fc3 -> spike."""
        x = hidden1
        if self.use_delay:
            x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay2(x)
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


def hidden_spike_penalty(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """L1 sparsity penalty: mean spikes per hidden neuron per sample.

    ``hidden_spikes`` is the binary 1st-hidden output of shape (B, C, 1, 1, T).
    Summing over time gives each neuron's spike count; averaging over batch and
    neurons gives a scalar that shrinks as the layer fires less. SLAYER's spike
    function is surrogate-gradient differentiable, so this term propagates back
    to ``fc1`` and genuinely drives the layer sparse.

    Args:
        hidden_spikes: 1st hidden layer spikes, shape (B, C, 1, 1, T).

    Returns:
        Scalar penalty tensor (mean spikes per neuron per sample).
    """
    return hidden_spikes.sum(dim=-1).mean()


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
    net: JitterSHDNetwork,
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
    lam: float,
    seed: int,
    epochs: int,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    use_delay: bool = USE_DELAY,
    max_delay: int = MAX_DELAY,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[JitterSHDNetwork, dict]:
    """Train one clean model with an L1 hidden-sparsity penalty.

    The forward pass is clean (no jitter). The total loss is the NumSpikes task
    loss plus ``lam`` times the 1st-hidden spike penalty. Early stopping and
    best-model selection use the *task* validation loss only, so we keep the most
    accurate model at whatever sparsity ``lam`` induces.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        lam: L1 penalty strength on 1st hidden layer spikes (0 = no penalty).
        seed: Random seed (controls init and shuffle order).
        epochs: Maximum training epochs.
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

    net = JitterSHDNetwork(
        input_dim, hidden_units, num_classes, use_delay, max_delay
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

    log = {
        "epoch": [],
        "train_loss": [],       # total (task + lam * reg)
        "train_task_loss": [],
        "train_reg": [],        # mean spikes / neuron / sample (unweighted)
        "val_loss": [],         # task only
        "val_acc": [],
        "delay_mean": [],
    }

    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=f"Train lam={lam:g} seed={seed}") as pbar:
        for epoch in range(epochs):
            # --- Train (clean forward + sparsity penalty) ---
            net.train()
            batch_total_losses = []
            batch_task_losses = []
            batch_regs = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs, hidden1 = net(x_batch, return_hidden=True)
                task_loss = loss_fn.numSpikes(outputs, target)
                reg = hidden_spike_penalty(hidden1)
                loss = task_loss + lam * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_total_losses.append(loss.item())
                batch_task_losses.append(task_loss.item())
                batch_regs.append(reg.item())
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
            train_reg = float(np.mean(batch_regs))

            delays = net.get_delays()
            avg_delay = (
                np.mean([
                    np.mean(d) for d in delays.values() if len(d) > 0
                ])
                if delays
                else 0.0
            )

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["train_task_loss"].append(train_task_loss)
            log["train_reg"].append(train_reg)
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))
            log["delay_mean"].append(float(avg_delay))

            pbar.set_postfix(
                epoch=epoch + 1,
                task=f"{train_task_loss:.3f}",
                reg=f"{train_reg:.2f}",
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
    net: JitterSHDNetwork,
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
    """Train the (lambda, seed) grid and record sparsity + clean accuracy."""
    lam_values, seeds, epochs = resolve_run_config()

    print(f"{'#' * 70}")
    print("# Sparse-network milestone (training): SHD whole, SGD-delay")
    print(f"# QUICK_TEST={QUICK_TEST} | epochs={epochs}")
    print(f"# lambda grid: {lam_values}")
    print(f"# seeds: {seeds}  ->  {len(lam_values) * len(seeds)} models")
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

    for lam in lam_values:
        for seed in seeds:
            run_tag = f"sparse_{DATASET_KEY}_delay_lam{lam:g}_seed{seed}"
            print(f"\n{'=' * 60}")
            print(f"  Training {run_tag}")
            print(f"{'=' * 60}")

            net, training_log = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                lam=lam,
                seed=seed,
                epochs=epochs,
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
                "lam": float(lam),
                "seed": int(seed),
                **{k: float(v) for k, v in metrics.items()},
            }

            # Persist per-model training curve.
            log_path = LOG_DIR / f"{run_tag}_training_log.json"
            log_serialisable = {
                k: [float(v) for v in vals] for k, vals in training_log.items()
            }
            with open(log_path, "w") as fp:
                json.dump(log_serialisable, fp, indent=2)
            print(f"  Training log saved to {log_path}")

    # Persist the sweep summary (the calibration/analysis table).
    summary_path = LOG_DIR / f"sparse_{DATASET_KEY}_delay_train_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Final table (dense -> sparse), for eyeballing calibration.
    print(f"\n{'run_tag':<40} {'clean_acc':>9} {'firing_rate':>12}")
    for run_tag, row in summary.items():
        print(f"{run_tag:<40} {row['clean_acc']:>9.4f} {row['firing_rate']:>12.4f}")


if __name__ == "__main__":
    run_milestone()
