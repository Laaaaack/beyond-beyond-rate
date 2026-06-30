"""Deeper SHD — 4-Hidden-Layer Per-Spike Jitter (train-all / eval-all).

The 4-hidden-layer counterpart of ``code/perturbation/jitter/jitter_train.py``.
The network stacks four spiking hidden layers (``h1``..``h4``) followed by a
spiking readout, and per-spike Gaussian jitter can be injected at the output of
any one hidden layer, selected by ``perturb_layer in {1, 2, 3, 4}``.

For each (perturbation site, jitter level ``sigma``) a fresh model is trained
end-to-end with the jitter active at that site on every batch (through a
straight-through estimator), then evaluated at the same ``sigma``. Sweeping the
site across all four hidden layers yields a depth trajectory of temporal
sensitivity (see ``code_moreLayers/README.md``).

SHD data is loaded from ``../../realistic/shd/shd_data``. Everything except
depth and perturbation site matches the 2-hidden-layer jitter script.
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

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directory of this script. All checkpoint and log paths are anchored here; the
# SHD dataset is loaded from the realistic/shd tree (see DATASET_CONFIGS).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "log"


# =====================================================================
# Global Configuration
# =====================================================================
# Batch mode: when True, the main run trains EVERY combination of
# (dataset_key, use_delay) listed in ALL_VARIATIONS case-by-case.
# When False, only the single (DATASET_KEY, USE_DELAY) pair below is run.
# In both cases the perturbation site is swept over PERTURB_LAYERS.
TRAIN_ALL_VARIATION: bool = True

# Network variant: True for SGD-delay, False for SGD (no delay).
# Ignored when TRAIN_ALL_VARIATION is True.
USE_DELAY: bool = True

# Dataset variant: "whole", "part", or "norm".
# Ignored when TRAIN_ALL_VARIATION is True.
DATASET_KEY: str = "norm"

# All (dataset_key, use_delay) pairs to iterate over in batch mode.
ALL_VARIATIONS: list[tuple[str, bool]] = [
    (dataset, delay)
    for dataset in ("norm", "part", "whole")
    for delay in (False, True)
]

# --- Dataset configurations (SHD data lives under realistic/shd/shd_data) ---
SHD_DATA_DIR = SCRIPT_DIR / "../../realistic/shd/shd_data"
DATASET_CONFIGS = {
    "whole": {"mat_file": str(SHD_DATA_DIR / "shd_whole.mat"), "input_dim": 700},
    "part":  {"mat_file": str(SHD_DATA_DIR / "shd_part_new.mat"), "input_dim": 224},
    "norm":  {"mat_file": str(SHD_DATA_DIR / "shd_norm_new.mat"), "input_dim": 224},
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
NUM_HIDDEN_LAYERS: int = 4
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
SEED: int = 42
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

# --- Hidden-perturbation sweep ---
# Hidden-layer sites at which to inject jitter, each in 1..NUM_HIDDEN_LAYERS.
PERTURB_LAYERS: list[int] = [1, 2, 3, 4]
# Per-spike jitter std dev sigma, in ms.
SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]
NUM_REPEATS: int = 3


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
    (matching the numpy reference). All ops stay on the input tensor's device.

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
    """4-hidden-layer SLAYER SNN with configurable per-spike jitter.

    The network stacks ``num_hidden_layers`` spiking hidden layers
    (``fc[0]``..``fc[L-1]`` -> spike, giving ``h1``..``hL``) followed by a
    spiking readout (``fc[L]``). For the SGD-delay variant a learnable delay is
    applied to each hidden spike train ``h_k`` before the next dense layer.

    ``forward(x, sigma, perturb_layer)`` injects per-spike Gaussian jitter of
    std dev ``sigma`` into the output of hidden layer ``perturb_layer``
    (1-indexed), directly on that layer's binary spike output and before its
    delay, through a straight-through estimator so the gradient path to the
    preceding layers stays intact during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 128,
        num_classes: int = 20,
        num_hidden_layers: int = 4,
        use_delay: bool = True,
        max_delay: int = 64,
    ):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.use_delay = use_delay
        self.max_delay = max_delay
        self.num_hidden_layers = num_hidden_layers

        # Dense layers: num_hidden_layers hidden transforms + 1 readout.
        dims = [input_dim] + [hidden_units] * num_hidden_layers + [num_classes]
        self.fc = nn.ModuleList([
            nn.utils.weight_norm(slayer.dense(dims[i], dims[i + 1]), name="weight")
            for i in range(num_hidden_layers + 1)
        ])

        # One learnable delay per hidden layer, applied to h_k before fc[k+1].
        if use_delay:
            self.delays = nn.ModuleList([
                slayer.delay(hidden_units) for _ in range(num_hidden_layers)
            ])

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 5-D NCHWT on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(device)

    def _apply_jitter(
        self,
        hidden: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """STE wrapper around ``jitter_hidden_batch``.

        Forward value is the jittered tensor; backward gradient is the
        identity through ``hidden``, so the layers before the perturbation
        site keep receiving gradient at any ``sigma > 0``.

        Args:
            hidden: Hidden-layer spike output, shape (B, C, 1, 1, T).
            sigma: Jitter std dev in ms.

        Returns:
            Jittered tensor that still carries the original gradient.
        """
        if sigma <= 0:
            return hidden
        jittered = jitter_hidden_batch(hidden, sigma)
        return hidden + (jittered - hidden).detach()

    def forward(
        self,
        x: torch.Tensor,
        sigma: float = 0.0,
        perturb_layer: int = 1,
    ) -> torch.Tensor:
        """Forward pass with per-spike jitter at one hidden-layer site.

        Args:
            x: Input spike trains, shape (B, C, T) or (B, C, 1, 1, T).
            sigma: Jitter std dev (ms) applied at the selected site.
            perturb_layer: Hidden layer whose spike output is jittered
                (1..num_hidden_layers).

        Returns:
            Output spike trains of shape (B, num_classes, 1, 1, T).
        """
        h = self._prepare_input(x)
        for i in range(self.num_hidden_layers):
            # Hidden layer i+1: PSP -> dense -> spike (strictly binary spikes).
            h = self.slayer.spike(self.fc[i](self.slayer.psp(h)))
            # Jitter this layer's spike output directly, before its delay.
            if (i + 1) == perturb_layer:
                h = self._apply_jitter(h, sigma)
            # Per-layer learnable delay feeding the next dense layer.
            if self.use_delay:
                h = self.delays[i](h)
        # Readout layer.
        return self.slayer.spike(self.fc[self.num_hidden_layers](self.slayer.psp(h)))

    def clamp_delays(self, maxima: list[int]) -> None:
        """Clamp each hidden-layer delay to [0, maxima[k]]."""
        if not self.use_delay:
            return
        for delay_mod, upper in zip(self.delays, maxima):
            delay_mod.delay.data.clamp_(0, upper)

    def get_delays(self) -> dict[str, np.ndarray]:
        """Return current delay values keyed by ``delay1``..``delayL``."""
        delays = {}
        if self.use_delay:
            for k, delay_mod in enumerate(self.delays):
                delays[f"delay{k + 1}"] = delay_mod.delay.data.cpu().numpy()
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
    """Build NumSpikes loss, Nadam optimizer, and LR scheduler."""
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
    input_dim: int,
    perturb_layer: int,
    sigma: float,
    hidden_units: int = 128,
    num_classes: int = 20,
    num_hidden_layers: int = 4,
    use_delay: bool = True,
    max_delay: int = 64,
    epochs: int = 1000,
    lr: float = 0.1,
    seed: int = 42,
    patience: int = 300,
) -> tuple[JitterSHDNetwork, dict]:
    """Train a JitterSHDNetwork with jitter at one hidden-layer site.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        input_dim: Number of input neurons.
        perturb_layer: Hidden layer jittered during training (1-indexed).
        sigma: Jitter std dev (ms) applied at the selected site.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        num_hidden_layers: Number of spiking hidden layers.
        use_delay: Whether to use learnable delays.
        max_delay: Maximum delay in time steps.
        epochs: Maximum training epochs.
        lr: Learning rate.
        seed: Random seed; re-seeded inside so each run starts from the same init.
        patience: Early stopping patience.

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = JitterSHDNetwork(
        input_dim, hidden_units, num_classes, num_hidden_layers, use_delay, max_delay
    ).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    # Adaptive delay clamping state, one counter/ceiling per hidden-layer delay.
    update = [0] * num_hidden_layers
    thea = [max_delay] * num_hidden_layers
    delay_ceiling_index = max(0, hidden_units - 20)

    log: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "delay_mean": [],
        "sigma": sigma,
        "perturb_layer": perturb_layer,
    }

    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=f"Train L{perturb_layer} sigma={sigma}") as pbar:
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

                outputs = net(x_batch, sigma=sigma, perturb_layer=perturb_layer)
                loss = loss_fn.numSpikes(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                pbar.update(1)

            # --- Adaptive delay clamping ---
            if use_delay:
                if epoch <= 250:
                    net.clamp_delays([max_delay] * num_hidden_layers)
                else:
                    for k, delay_mod in enumerate(net.delays):
                        update[k] += 1
                        if update[k] > 150:
                            sorted_ = torch.sort(
                                torch.floor(delay_mod.delay.detach().flatten())
                            )[0]
                            thea_val = torch.max(sorted_)
                            if sorted_[delay_ceiling_index] > (thea_val - 5):
                                thea[k] = int(thea_val.item()) + 1
                                update[k] = 0
                    net.clamp_delays(thea)

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

                    outputs = net(x_batch, sigma=sigma, perturb_layer=perturb_layer)
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
    perturb_layer: int,
    sigma: float = 0.0,
) -> float:
    """Evaluate accuracy with jitter at ``perturb_layer`` and level ``sigma``."""
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device)

            outputs = net(x_batch, sigma=sigma, perturb_layer=perturb_layer)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: JitterSHDNetwork,
    test_loader: DataLoader,
    perturb_layer: int,
    sigma: float,
    num_repeats: int = 3,
) -> dict:
    """Repeat ``test_with_jitter`` for mean ± std error bars."""
    accuracies: list[float] = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        torch.manual_seed(SEED + repeat)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + repeat)
        accuracies.append(
            test_with_jitter(net, test_loader, perturb_layer, sigma=sigma)
        )
    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": [float(a) for a in accuracies],
    }


def run_variation_sweep(
    perturb_layer: int,
    dataset_key: str,
    use_delay: bool,
) -> dict:
    """Train-all / eval-all sweep for one (site, dataset, delay) variation.

    Loads the SHD variant, builds dataloaders, trains one fresh model per
    sigma in SIGMA_VALUES with jitter active at ``perturb_layer`` during
    training, evaluates each at the same sigma, and persists per-variation
    checkpoints + JSON.

    Args:
        perturb_layer: Hidden layer jittered during this sweep (1-indexed).
        dataset_key: One of "norm", "part", "whole".
        use_delay: Train the SGD-delay variant if True, else SGD.

    Returns:
        Dict with models / logs / results / test_loader / model_prefix
        / perturb_layer / dataset_key / use_delay for the variation.
    """
    if not 1 <= perturb_layer <= NUM_HIDDEN_LAYERS:
        raise ValueError(
            f"perturb_layer must be in 1..{NUM_HIDDEN_LAYERS}, got {perturb_layer}"
        )

    cfg = DATASET_CONFIGS[dataset_key]
    input_dim = cfg["input_dim"]
    mat_file = cfg["mat_file"]
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"jitter_perturb{perturb_layer}_{dataset_key}_{delay_tag}"

    print(f"\n{'#' * 70}")
    print(
        f"# Layer-{perturb_layer} jitter: dataset={dataset_key} | delay={delay_tag}"
    )
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
        print(f"\n=== Training {model_prefix} with hidden jitter sigma={sigma} ms ===")
        net, training_log = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            perturb_layer=perturb_layer,
            sigma=sigma,
            hidden_units=HIDDEN_UNITS,
            num_classes=NUM_CLASSES,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            use_delay=use_delay,
            max_delay=MAX_DELAY,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            seed=SEED,
            patience=EARLY_STOP_PATIENCE,
        )

        model_path = DATA_DIR / f"{model_prefix}_sigma{sigma}.pt"
        torch.save(net.state_dict(), model_path)

        result = test_with_repeats(
            net, test_loader, perturb_layer, sigma=sigma, num_repeats=NUM_REPEATS
        )
        models[sigma] = net
        logs[sigma] = training_log
        results[sigma] = result
        print(
            f"sigma={sigma} | test acc = {result['mean']:.4f} ± {result['std']:.4f}"
            f" | checkpoint -> {model_path}"
        )

    # Per-variation JSON persistence.
    results_serialisable = {
        str(sigma): {
            "mean": float(d["mean"]),
            "std": float(d["std"]),
            "values": [float(v) for v in d["values"]],
        }
        for sigma, d in results.items()
    }
    results_path = LOG_DIR / f"{model_prefix}_jitter_sweep_results.json"
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")

    training_logs_serialisable = {
        str(sigma): {
            k: ([float(v) for v in vals] if isinstance(vals, list) else vals)
            for k, vals in log.items()
        }
        for sigma, log in logs.items()
    }
    log_path = LOG_DIR / f"{model_prefix}_training_log.json"
    with open(log_path, "w") as fp:
        json.dump(training_logs_serialisable, fp, indent=2)
    print(f"Training logs saved to {log_path}")

    return {
        "models": models,
        "logs": logs,
        "results": results,
        "test_loader": test_loader,
        "model_prefix": model_prefix,
        "perturb_layer": perturb_layer,
        "dataset_key": dataset_key,
        "use_delay": use_delay,
    }


def main() -> None:
    """Run the configured train-all / eval-all sweeps over sites and variations."""
    variations_to_run = (
        ALL_VARIATIONS if TRAIN_ALL_VARIATION else [(DATASET_KEY, USE_DELAY)]
    )

    print(
        f"Sweeping {len(PERTURB_LAYERS)} site(s) x {len(variations_to_run)} "
        f"variation(s) on a {NUM_HIDDEN_LAYERS}-hidden-layer network:"
    )
    for perturb_layer in PERTURB_LAYERS:
        for ds, ud in variations_to_run:
            tag = "delay" if ud else "nodelay"
            print(f"  - jitter_perturb{perturb_layer}_{ds}_{tag}")
    print(f"Sigma sweep: {SIGMA_VALUES}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    for perturb_layer in PERTURB_LAYERS:
        for ds_key, use_delay in variations_to_run:
            run_variation_sweep(perturb_layer, ds_key, use_delay)


if __name__ == "__main__":
    main()
