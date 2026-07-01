"""Deeper SHD — 4-Hidden-Layer Layer-wise Perturbation (train-all / eval-all).

This is the 4-hidden-layer counterpart of ``code/realistic/shd/shd_train.py``.
The network stacks four spiking hidden layers (``h1``..``h4``) followed by a
spiking readout, and the spike-relocation perturbation can be injected at the
output of any one hidden layer, selected by ``perturb_layer in {1, 2, 3, 4}``.

For each (perturbation site, level *f*) a fresh model is trained end-to-end
with the perturbation active at that site on every batch (through a
straight-through estimator), then evaluated at the same *f* — the same
train-all / eval-all protocol as the 2-hidden-layer experiments. Sweeping the
site across all four hidden layers yields a depth trajectory of temporal
sensitivity (see ``code_moreLayers/README.md``).

Everything except depth and perturbation site is held identical to the 2-layer
SHD script: SRMALPHA neurons, hidden width 128, learnable per-layer delays
(SGD-delay variant), NumSpikes loss, Nadam, and the adaptive delay-clamping
schedule.
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

# Directory of this script. All dataset, checkpoint, and log paths are anchored
# here so the script can be launched from any working directory.
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

# --- Dataset configurations ---
DATASET_CONFIGS = {
    "whole": {"mat_file": str(SCRIPT_DIR / "shd_data/shd_whole.mat"), "input_dim": 700},
    "part":  {"mat_file": str(SCRIPT_DIR / "shd_data/shd_part_new.mat"), "input_dim": 224},
    "norm":  {"mat_file": str(SCRIPT_DIR / "shd_data/shd_norm_new.mat"), "input_dim": 224},
}

# --- SLAYER neuron and simulation descriptors ---
SIM_PARAMS = {"Ts": 1, "tSample": 200}
LIF_PARAMS = {
    "type": "SRMALPHA",
    "theta": 2,  # lowered from 10: keeps spikes and gradient alive through all 4 hidden layers
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
EPOCHS: int = 150
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
SEED: int = 42
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

# --- Hidden-perturbation sweep ---
# Hidden-layer sites at which to inject perturbation, each in 1..NUM_HIDDEN_LAYERS.
PERTURB_LAYERS: list[int] = [1,2,3,4]
F_VALUES: list[float] = [0.0, 0.2]
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
def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float = 0.0,
) -> torch.Tensor:
    """Vectorised GPU-side partial spike relocation.

    For each (batch, neuron), a fraction *f* of the existing spikes are
    removed and replaced with the same number of spikes placed at randomly
    chosen previously-unoccupied time bins. Spike count per neuron is
    preserved exactly. All operations stay on the input tensor's device,
    avoiding the CPU/numpy round-trip that dominates training cost when
    perturbation runs on every batch (mirrors the ``isi_delay`` version).

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).

    Returns:
        Perturbed tensor with the same shape, dtype, and device.
    """
    if f <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    # Count spikes per (batch, neuron) and compute how many to move.
    n_spikes = is_spike.sum(dim=-1, keepdim=True)  # (B, C, 1)
    num_to_move = (n_spikes.float() * f).floor().long()  # (B, C, 1)

    # --- 1. Choose which existing spikes to remove ---
    # Random key per time bin; non-spike bins get +inf so they sort last.
    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    # rank[b, c, t] = position of t in the per-(b,c) ascending sort of `key`.
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move  # (B, C, T)

    keep_mask = is_spike & ~remove_mask

    # --- 2. Place the same number of spikes in currently-unoccupied bins ---
    available = ~keep_mask  # everything except positions we are keeping
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move  # disjoint from keep_mask by construction

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype)
    return new_spikes.view(B, C, H, W, T)


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


class SHDNetwork(nn.Module):
    """4-hidden-layer SLAYER SNN with configurable layer-wise perturbation.

    The network stacks ``num_hidden_layers`` spiking hidden layers
    (``fc[0]``..``fc[L-1]`` -> spike, giving ``h1``..``hL``) followed by a
    spiking readout (``fc[L]``). For the SGD-delay variant a learnable delay is
    applied to each hidden spike train ``h_k`` before the next dense layer,
    matching the 2-hidden-layer model's placement (a delay sits at the input of
    every non-first dense layer).

    ``forward(x, f, perturb_layer)`` injects the spike-relocation perturbation
    at level *f* into the output of hidden layer ``perturb_layer`` (1-indexed),
    directly on that layer's binary spike output and before its delay. The
    perturbation is wired through a straight-through estimator so the gradient
    path to the layers preceding the perturbation site stays intact during
    training.
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

        # Dense layers: num_hidden_layers hidden transforms + 1 readout, each
        # with weight normalisation. dims = [in, hid, hid, ..., hid, classes].
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

    def _apply_perturbation(
        self,
        hidden: torch.Tensor,
        f: float,
    ) -> torch.Tensor:
        """STE wrapper around ``perturb_hidden_batch``.

        Without this wrapper the perturbation would return a fresh leaf,
        blocking gradient flow to the layers before the perturbation site and
        freezing them at random init for any ``f > 0``.

        Args:
            hidden: Hidden-layer spike output, shape (B, C, 1, 1, T).
            f: Perturbation level.

        Returns:
            Perturbed tensor that still carries the original gradient.
        """
        if f <= 0:
            return hidden
        perturbed = perturb_hidden_batch(hidden, f)
        return hidden + (perturbed - hidden).detach()

    def forward(
        self,
        x: torch.Tensor,
        f: float = 0.0,
        perturb_layer: int = 1,
    ) -> torch.Tensor:
        """Forward pass with perturbation at one hidden-layer site.

        Args:
            x: Input spike trains, shape (B, C, T) or (B, C, 1, 1, T).
            f: Perturbation level applied at the selected site.
            perturb_layer: Hidden layer whose spike output is perturbed
                (1..num_hidden_layers).

        Returns:
            Output spike trains of shape (B, num_classes, 1, 1, T).
        """
        h = self._prepare_input(x)
        for i in range(self.num_hidden_layers):
            # Hidden layer i+1: PSP -> dense -> spike (strictly binary spikes).
            h = self.slayer.spike(self.fc[i](self.slayer.psp(h)))
            # Perturb this layer's spike output directly, before its delay.
            if (i + 1) == perturb_layer:
                h = self._apply_perturbation(h, f)
            # Per-layer learnable delay feeding the next dense layer.
            if self.use_delay:
                h = self.delays[i](h)
        # Readout layer.
        return self.slayer.spike(self.fc[self.num_hidden_layers](self.slayer.psp(h)))

    def clamp_delays(self, maxima: list[int]) -> None:
        """Clamp each hidden-layer delay to [0, maxima[k]].

        Args:
            maxima: Per-delay upper bounds, one entry per hidden layer.
        """
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
    net: SHDNetwork,
    lr: float = 0.1,
) -> tuple:
    """Build SpikeRate loss, Nadam optimizer, and LR scheduler."""
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
    hidden_units: int = 128,
    num_classes: int = 20,
    num_hidden_layers: int = 4,
    use_delay: bool = True,
    max_delay: int = 64,
    epochs: int = 1000,
    lr: float = 0.1,
    seed: int = 42,
    patience: int = 300,
    f: float = 0.0,
) -> tuple[SHDNetwork, dict]:
    """Train the SHDNetwork with perturbation at one hidden-layer site.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        input_dim: Number of input neurons.
        perturb_layer: Hidden layer perturbed during training (1-indexed).
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        num_hidden_layers: Number of spiking hidden layers.
        use_delay: Whether to use learnable delays.
        max_delay: Maximum delay in time steps.
        epochs: Maximum training epochs.
        lr: Learning rate.
        seed: Random seed; re-seeded inside so each run starts from the same init.
        patience: Early stopping patience.
        f: Hidden-layer perturbation level applied during forward passes.

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = SHDNetwork(
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
    # Heuristic index into the sorted per-layer delays (108 for 128 units): the
    # ceiling is raised once this high-percentile delay approaches the maximum.
    delay_ceiling_index = max(0, hidden_units - 20)

    log: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "delay_mean": [],
        "f": f,
        "perturb_layer": perturb_layer,
    }

    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc=f"Train L{perturb_layer} f={f}") as pbar:
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

                outputs = net(x_batch, f=f, perturb_layer=perturb_layer)
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

            # --- Validate (with same perturbation) ---
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

                    outputs = net(x_batch, f=f, perturb_layer=perturb_layer)
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


def test_with_hidden_perturbation(
    net: SHDNetwork,
    test_loader: DataLoader,
    perturb_layer: int,
    f: float = 0.0,
) -> float:
    """Evaluate accuracy with perturbation at ``perturb_layer`` and level *f*."""
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device)

            outputs = net(x_batch, f=f, perturb_layer=perturb_layer)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: SHDNetwork,
    test_loader: DataLoader,
    perturb_layer: int,
    f: float,
    num_repeats: int = 3,
) -> dict:
    """Repeat ``test_with_hidden_perturbation`` for mean ± std error bars."""
    accuracies: list[float] = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        torch.manual_seed(SEED + repeat)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + repeat)
        accuracies.append(
            test_with_hidden_perturbation(net, test_loader, perturb_layer, f=f)
        )
    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": [float(a) for a in accuracies],
    }


def save_training_curve(
    log: dict,
    out_path: Path,
    title: str = "",
) -> None:
    """Plot per-epoch training curves from a log dict and save them as a PNG.

    Renders training/validation loss and validation accuracy against epoch
    (with mean delay overlaid on a twin axis when the SGD-delay variant
    produced non-zero delays) and writes the figure to ``out_path``.

    Args:
        log: Training log dict with per-epoch lists "epoch", "train_loss",
            "val_loss", "val_acc", and "delay_mean".
        out_path: Destination PNG path.
        title: Figure super-title (e.g. the model prefix and level).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping training-curve plot.")
        return

    epochs = log.get("epoch", [])
    if not epochs:
        print(f"Empty training log; skipping curve for {out_path}")
        return

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, log["train_loss"], label="train loss")
    ax_loss.plot(epochs, log["val_loss"], label="val loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, log["val_acc"], color="tab:green", label="val acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_title("Validation accuracy")

    # Overlay mean delay on a twin axis when delays were actually learned.
    delay_mean = log.get("delay_mean", [])
    if delay_mean and max(delay_mean) > 0:
        ax_delay = ax_acc.twinx()
        ax_delay.plot(
            epochs, delay_mean, color="tab:orange", alpha=0.6, label="mean delay"
        )
        ax_delay.set_ylabel("mean delay (steps)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve saved to {out_path}")


def run_variation_sweep(
    perturb_layer: int,
    dataset_key: str,
    use_delay: bool,
) -> dict:
    """Train-all / eval-all sweep for one (site, dataset, delay) variation.

    Loads the dataset, builds dataloaders, trains one fresh model per f in
    F_VALUES with the perturbation active at ``perturb_layer`` during training,
    evaluates each at the same f, and persists per-variation checkpoints + JSON.

    Args:
        perturb_layer: Hidden layer perturbed during this sweep (1-indexed).
        dataset_key: One of "norm", "part", "whole".
        use_delay: Train the SGD-delay variant if True, else SGD.

    Returns:
        Dict with keys models / logs / results / test_loader / model_prefix
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
    model_prefix = f"shd_perturb{perturb_layer}_{dataset_key}_{delay_tag}"

    print(f"\n{'#' * 70}")
    print(
        f"# Layer-{perturb_layer} perturbation: dataset={dataset_key} | delay={delay_tag}"
    )
    print(f"# Model prefix: {model_prefix}")
    print(f"{'#' * 70}")

    X, Y = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    train_loader, val_loader, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=SEED,
    )

    models: dict[float, SHDNetwork] = {}
    logs: dict[float, dict] = {}
    results: dict[float, dict] = {}

    for f_val in F_VALUES:
        print(f"\n=== Training {model_prefix} at f={f_val} ===")
        net, training_log = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            perturb_layer=perturb_layer,
            hidden_units=HIDDEN_UNITS,
            num_classes=NUM_CLASSES,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            use_delay=use_delay,
            max_delay=MAX_DELAY,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            seed=SEED,
            patience=EARLY_STOP_PATIENCE,
            f=f_val,
        )

        model_path = DATA_DIR / f"{model_prefix}_f{f_val}.pt"
        torch.save(net.state_dict(), model_path)

        result = test_with_repeats(
            net, test_loader, perturb_layer, f=f_val, num_repeats=NUM_REPEATS
        )
        models[f_val] = net
        logs[f_val] = training_log
        results[f_val] = result
        print(
            f"f={f_val} | test acc = {result['mean']:.4f} ± {result['std']:.4f}"
            f" | checkpoint -> {model_path}"
        )

        # Save the per-epoch training curve as a PNG alongside the JSON logs.
        curve_path = LOG_DIR / f"{model_prefix}_f{f_val}_training_curve.png"
        save_training_curve(
            training_log, curve_path, title=f"{model_prefix} | f={f_val}"
        )

    # Per-variation JSON persistence.
    results_serialisable = {
        str(f_val): {
            "mean": float(d["mean"]),
            "std": float(d["std"]),
            "values": [float(v) for v in d["values"]],
        }
        for f_val, d in results.items()
    }
    results_path = LOG_DIR / f"{model_prefix}_hidden_perturbation_results.json"
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")

    training_logs_serialisable = {
        str(f_val): {
            k: ([float(v) for v in vals] if isinstance(vals, list) else vals)
            for k, vals in log.items()
        }
        for f_val, log in logs.items()
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
            print(f"  - shd_perturb{perturb_layer}_{ds}_{tag}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    for perturb_layer in PERTURB_LAYERS:
        for ds_key, use_delay in variations_to_run:
            run_variation_sweep(perturb_layer, ds_key, use_delay)


if __name__ == "__main__":
    main()
