"""Experiment 3A: Hidden Per-Spike Jitter — SHD.

Train 2-hidden-layer SNNs on SHD with per-spike Gaussian jitter applied at the
1st hidden layer output during training. A fresh model is trained from scratch
for each jitter level sigma (train-at-sigma / eval-at-sigma protocol), with the
jitter wired through a straight-through estimator so ``fc1`` keeps receiving
gradient.

This is the script (core training) version of ``jitter_train.ipynb`` with the
notebook's visualisation, diagnostic, and analysis cells removed. It loads the
data, trains one model per sigma for every requested dataset/delay variation,
and persists the checkpoints, sweep results, and training logs.
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

# Directory of this script. All dataset, SLAYER, checkpoint, and log paths are
# anchored here so the script can be launched from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "log"

# Add SLAYER to path
sys.path.append(str(SCRIPT_DIR / "../../../temporal_shd_project/code/src"))
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Batch mode: when True, the main run trains EVERY combination of
# (dataset_key, use_delay) listed in ALL_VARIATIONS case-by-case.
# When False, only the single (DATASET_KEY, USE_DELAY) pair below is run.
TRAIN_ALL_VARIATION: bool = False

# Network variant: True for SGD-delay, False for SGD (no delay).
# Ignored when TRAIN_ALL_VARIATION is True.
USE_DELAY: bool = True

# Dataset variant: "whole", "part", or "norm".
# Ignored when TRAIN_ALL_VARIATION is True.
DATASET_KEY: str = "whole"

# All (dataset_key, use_delay) pairs to iterate over in batch mode.
ALL_VARIATIONS: list[tuple[str, bool]] = [
    (dataset, delay)
    for dataset in ("norm", "part", "whole")
    for delay in (False, True)
]

# --- Dataset configurations ---
DATASET_CONFIGS = {
    "whole": {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_whole.mat"), "input_dim": 700},
    "part":  {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_part_new.mat"), "input_dim": 224},
    "norm":  {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_norm_new.mat"), "input_dim": 224},
}

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

# --- Data split ratios ---
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# --- Training hyper-parameters ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
SEED: int = 42
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

# --- Jitter sweep: sigma values in ms ---
SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]

# --- Evaluation ---
NUM_REPEATS: int = 3

# Default input dim for the single-variation case (train_model's default arg).
INPUT_DIM: int = DATASET_CONFIGS[DATASET_KEY]["input_dim"]


def load_shd_data(mat_path: str, target_T: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Load SHD dataset from a .mat file and pad time dimension.

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


@torch.no_grad()
def jitter_hidden_batch(
    hidden_spikes: torch.Tensor,
    sigma: float,
    max_attempts: int = 50,
) -> torch.Tensor:
    """Vectorised GPU-side per-spike Gaussian jitter.

    For each spike in ``hidden_spikes``, draw an iid Gaussian offset
    ``~ N(0, sigma)``, shift the spike by ``round(offset)`` and clip to
    ``[0, T - 1]``. Collisions (multiple spikes landing in the same bin)
    are resolved by a priority-based tiebreaker; losing spikes are retried
    with a fresh offset for up to ``max_attempts`` outer iterations. Any
    spike still unplaced after that falls back to its original position
    (matching the numpy reference). All ops stay on the input tensor's
    device, avoiding the CPU/numpy round-trip that previously dominated
    training cost (Section 7.1.3 of ``docs/progress/phase1to4_fixes.md``).

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
    is_spike = x > 0.5  # (B, C, T) bool

    new_spikes = torch.zeros_like(is_spike)
    unplaced = is_spike.clone()

    t_idx = torch.arange(T, device=x.device).view(1, 1, T)
    inf_tensor = torch.full_like(x, float("inf"))

    for _ in range(max_attempts):
        if not unplaced.any():
            break

        # Sample iid Gaussian targets for every position; positions that are
        # not currently unplaced are ignored downstream via the priority mask.
        offsets = torch.randn_like(x) * sigma
        target = (t_idx + offsets).round().long().clamp(0, T - 1)

        # Tiebreaker: each unplaced source gets a random priority; for every
        # target bin, the lowest-priority source wins. Non-unplaced sources
        # get +inf so they never win.
        priority = torch.where(unplaced, torch.rand_like(x), inf_tensor)
        min_priority = inf_tensor.clone()
        min_priority.scatter_reduce_(
            -1, target, priority, reduce="amin", include_self=True,
        )

        # A source wins iff its priority equals its target's min AND the
        # target bin is still free (not already placed in this or prior pass).
        target_min = min_priority.gather(-1, target)
        target_free = ~new_spikes.gather(-1, target)
        wins = unplaced & (priority == target_min) & target_free

        # Scatter winning sources onto their target positions.
        scatter_out = torch.zeros((B, C, T), device=x.device, dtype=torch.uint8)
        scatter_out.scatter_add_(-1, target, wins.to(torch.uint8))
        new_spikes = new_spikes | (scatter_out > 0)

        unplaced = unplaced & ~wins

    # Fallback: any source spike still unplaced after the retry budget stays
    # at its original time bin (collapses harmlessly under OR).
    new_spikes = new_spikes | unplaced

    return new_spikes.to(hidden_spikes.dtype).view(B, C, H, W, T)


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
    """Return index array for a given fractional range of the dataset."""
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

    Args:
        X: Full dataset features, shape (N, neurons, T).
        Y: Full dataset labels, shape (N,).
        batch_size: Batch size for all loaders.
        seed: Random seed for train shuffle.

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
    """2-hidden-layer SLAYER SNN with per-spike jitter at the 1st hidden layer.

    ``delay1`` is applied at the start of ``_second_hidden_and_output``
    (§3.4 Option B) so the perturbation hook sees the 1st hidden layer's
    binary spike output directly. ``forward(x, sigma)`` applies the jitter
    through a straight-through estimator so ``fc1`` still receives gradient
    when training at ``sigma > 0`` (§3.2 / §7.1 of the fixes doc).
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

        # delay1 lives at the start of _second_hidden_and_output (after the
        # perturbation hook); delay2 stays between fc2-spike and fc3.
        if use_delay:
            self.delay1 = slayer.delay(hidden_units)
            self.delay2 = slayer.delay(hidden_units)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 5-D NCHWT on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(device)

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike -> hidden1 spikes (strictly binary)."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> (delay1) -> PSP -> fc2 -> spike -> (delay2) -> PSP -> fc3 -> spike."""
        x = hidden1
        if self.use_delay:
            x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def _apply_jitter(
        self,
        hidden: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """STE wrapper around ``jitter_hidden_batch``.

        ``jitter_hidden_batch`` goes through numpy, so the returned tensor
        is a fresh leaf with no autograd edge. Without this wrapper,
        ``loss.backward()`` cannot propagate gradient back to ``fc1`` and
        the upstream layer stays at random init for the whole training
        run at any ``sigma > 0`` (Section 7.1 of the fixes doc — the
        weight-fingerprint identity in §7.1.3 came from this notebook).
        Forward value is the jittered tensor; backward gradient is the
        identity through ``hidden``.
        """
        if sigma <= 0:
            return hidden
        jittered = jitter_hidden_batch(hidden, sigma)
        return hidden + (jittered - hidden).detach()

    def forward(
        self,
        x: torch.Tensor,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with per-spike jitter at the 1st hidden layer.

        Args:
            x: Input spike trains.
            sigma: Jitter std dev in ms. 0 means no jitter.

        Returns:
            Output spike tensor.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden1 = self._apply_jitter(hidden1, sigma)
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
    """Build NumSpikes loss, Nadam optimizer, and LR scheduler.

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
    sigma: float,
    input_dim: int = INPUT_DIM,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    use_delay: bool = USE_DELAY,
    max_delay: int = MAX_DELAY,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[JitterSHDNetwork, dict]:
    """Train a JitterSHDNetwork with hidden-layer jitter.

    A new model is created and trained from scratch. Per-spike jitter
    with the given sigma is applied at the 1st hidden layer during
    every training forward pass.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        sigma: Jitter std dev applied at hidden layer during training.
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        use_delay: Whether to use learnable delays.
        max_delay: Maximum delay in time steps.
        epochs: Maximum training epochs.
        lr: Learning rate.
        seed: Random seed.
        patience: Early stopping patience.

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
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "delay_mean": [],
    }

    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=f"Training sigma={sigma}") as pbar:
        for epoch in range(epochs):
            # --- Train ---
            net.train()
            batch_losses = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                # Forward with hidden-layer jitter
                outputs = net(x_batch, sigma=sigma)
                loss = loss_fn.numSpikes(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                pbar.update(1)

            # --- Adaptive delay clamping ---
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

            # --- Validate (with same jitter) ---
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

                    outputs = net(x_batch, sigma=sigma)
                    val_loss += loss_fn.numSpikes(outputs, target).item()

                    pred = snn.predict.getClass(outputs)
                    correct += (pred.cpu() == y_batch.cpu()).sum().item()
                    total += len(y_batch)

            val_loss /= max(1, len(val_loader))
            val_acc = correct / max(1, total)
            train_loss = np.mean(batch_losses)

            # Log delay statistics
            delays = net.get_delays()
            avg_delay = (
                np.mean([
                    np.mean(d) for d in delays.values() if len(d) > 0
                ])
                if delays
                else 0.0
            )

            log["epoch"].append(epoch)
            log["train_loss"].append(float(train_loss))
            log["val_loss"].append(float(val_loss))
            log["val_acc"].append(float(val_acc))
            log["delay_mean"].append(float(avg_delay))

            pbar.set_postfix(
                epoch=epoch + 1,
                train=f"{train_loss:.3f}",
                val=f"{val_loss:.3f}",
                acc=f"{val_acc:.2%}",
                delay=f"{avg_delay:.1f}",
            )
            scheduler.step()

            # Early stopping
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


def test_with_jitter(
    net: JitterSHDNetwork,
    test_loader: DataLoader,
    sigma: float = 0.0,
) -> float:
    """Evaluate accuracy with hidden-layer jitter at level sigma.

    Args:
        net: Trained JitterSHDNetwork.
        test_loader: Test DataLoader.
        sigma: Jitter std dev applied to 1st hidden layer spikes.

    Returns:
        Test accuracy as a float in [0, 1].
    """
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device)

            outputs = net(x_batch, sigma=sigma)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: JitterSHDNetwork,
    test_loader: DataLoader,
    sigma: float,
    num_repeats: int = NUM_REPEATS,
) -> dict:
    """Evaluate with jitter multiple times for error bars.

    Args:
        net: Trained network.
        test_loader: Test DataLoader.
        sigma: Jitter std dev.
        num_repeats: Number of independent evaluations.

    Returns:
        Dict with "mean", "std", and "values".
    """
    accuracies = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        acc = test_with_jitter(net, test_loader, sigma=sigma)
        accuracies.append(acc)

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": accuracies,
    }


def run_variation_sweep(
    dataset_key: str,
    use_delay: bool,
) -> dict:
    """Train-at-sigma sweep for one (dataset_key, use_delay) variation.

    Loads the SHD variant, builds dataloaders, trains one fresh model per
    sigma in SIGMA_VALUES with hidden-layer jitter active during training
    (via the STE), evaluates each at the same sigma, and persists
    per-variation checkpoints + JSON.

    Args:
        dataset_key: One of "norm", "part", "whole".
        use_delay: Train the SGD-delay variant if True, else SGD.

    Returns:
        Dict with models / logs / results / test_loader / model_prefix
        / dataset_key / use_delay for the variation.
    """
    cfg = DATASET_CONFIGS[dataset_key]
    input_dim = cfg["input_dim"]
    mat_file = cfg["mat_file"]
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"jitter_{dataset_key}_{delay_tag}"

    print(f"\n{'#' * 70}")
    print(f"# Variation: dataset={dataset_key} | delay={delay_tag}")
    print(f"# Model prefix: {model_prefix}")
    print(f"{'#' * 70}")

    X, Y = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    train_loader, val_loader, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=SEED,
    )

    models: dict[int, JitterSHDNetwork] = {}
    logs: dict[int, dict] = {}
    results: dict[int, dict] = {}

    for sigma in SIGMA_VALUES:
        print(f"\n{'=' * 60}")
        print(f"  Training {model_prefix} with hidden jitter sigma={sigma} ms")
        print(f"{'=' * 60}")

        net, training_log = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            sigma=sigma,
            input_dim=input_dim,
            hidden_units=HIDDEN_UNITS,
            num_classes=NUM_CLASSES,
            use_delay=use_delay,
            max_delay=MAX_DELAY,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            seed=SEED,
            patience=EARLY_STOP_PATIENCE,
        )

        model_path = DATA_DIR / f"{model_prefix}_sigma{sigma}.pt"
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        test_result = test_with_repeats(net, test_loader, sigma=sigma)
        print(
            f"sigma={sigma} | Test Accuracy: "
            f"{test_result['mean']:.4f} +/- {test_result['std']:.4f}"
        )

        models[sigma] = net
        logs[sigma] = training_log
        results[sigma] = test_result

    # Persist sweep results
    sweep_serialisable = {
        str(sigma): {
            "mean": float(d["mean"]),
            "std": float(d["std"]),
            "values": [float(v) for v in d["values"]],
        }
        for sigma, d in results.items()
    }
    results_path = LOG_DIR / f"{model_prefix}_jitter_sweep_results.json"
    with open(results_path, "w") as fp:
        json.dump(sweep_serialisable, fp, indent=2)
    print(f"\nSweep results saved to {results_path}")

    # Persist per-sigma training logs
    for sigma, log in logs.items():
        log_path = LOG_DIR / f"{model_prefix}_sigma{sigma}_training_log.json"
        log_serialisable = {
            k: ([float(v) for v in vals] if isinstance(vals, list) else vals)
            for k, vals in log.items()
        }
        with open(log_path, "w") as fp:
            json.dump(log_serialisable, fp, indent=2)
        print(f"  Training log saved to {log_path}")

    return {
        "models": models,
        "logs": logs,
        "results": results,
        "test_loader": test_loader,
        "model_prefix": model_prefix,
        "dataset_key": dataset_key,
        "use_delay": use_delay,
    }


def main() -> None:
    """Run the configured train-at-sigma / eval-at-sigma sweep(s)."""
    if TRAIN_ALL_VARIATION:
        print(f"Batch mode: training {len(ALL_VARIATIONS)} variations:")
        for ds, ud in ALL_VARIATIONS:
            tag = "delay" if ud else "nodelay"
            print(f"  - jitter_{ds}_{tag}")
    else:
        tag = "delay" if USE_DELAY else "nodelay"
        print(f"Single variation: dataset=SHD {DATASET_KEY} | input_dim={INPUT_DIM}")
        print(f"  Network mode: {'SGD-delay' if USE_DELAY else 'SGD (no delay)'}")
        print(f"  Model prefix: jitter_{DATASET_KEY}_{tag}")
    print(f"Sigma sweep: {SIGMA_VALUES}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    variations_to_run = (
        ALL_VARIATIONS if TRAIN_ALL_VARIATION else [(DATASET_KEY, USE_DELAY)]
    )

    for ds_key, use_delay in variations_to_run:
        run_variation_sweep(ds_key, use_delay)


if __name__ == "__main__":
    main()
