"""Experiment 4B: SHD Time Reversal at 2nd Hidden Layer (train-at-f / eval-at-f).

Same as Experiment 4A but the perturbation and time reversal hook is placed on
the output spike train of the **2nd** hidden layer instead of the 1st.

Architecture:
    Input -> fc1 -> spike -> (delay1) -> fc2 -> spike -> hidden2 spikes
          -> [relocate fraction f] -> [reverse?]
          -> (delay2) -> fc3 -> spike -> output (20)
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "log"

sys.path.append(str(SCRIPT_DIR / "../../../temporal_shd_project/code/src"))
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================================
# Set to True to run all (dataset x delay) combinations automatically.
# When False, only the DATASET_KEY / USE_DELAY settings below are used.
# =====================================================================
TRAIN_ALL_VARIATIONS: bool = True

# =====================================================================
# Network variant: set to True for SGD-delay, False for SGD (no delay)
# =====================================================================
USE_DELAY: bool = False

# =====================================================================
# Dataset variant: "whole", "part", or "norm"
# =====================================================================
DATASET_KEY: str = "whole"

DATASET_CONFIGS = {
    "whole": {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_whole.mat"), "input_dim": 700},
    "part":  {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_part_new.mat"), "input_dim": 224},
    "norm":  {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_norm_new.mat"), "input_dim": 224},
}

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

TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
SEED: int = 42
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
REVERSAL_CONDITIONS: list[tuple[str, bool]] = [
    ("reversal", True),
]
NUM_REPEATS: int = 3


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Perturbation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def perturb_hidden_batch(hidden_spikes: torch.Tensor, f: float = 0.0) -> torch.Tensor:
    """Vectorised GPU-side partial spike relocation.

    For each (batch, neuron), a fraction f of the existing spikes are removed
    and replaced at randomly chosen previously-unoccupied time bins. Spike
    count per neuron is preserved exactly.

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

    n_spikes = is_spike.sum(dim=-1, keepdim=True)
    num_to_move = (n_spikes.float() * f).floor().long()

    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move

    keep_mask = is_spike & ~remove_mask

    available = ~keep_mask
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype)
    return new_spikes.view(B, C, H, W, T)


@torch.no_grad()
def reverse_hidden_batch(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """Vectorised GPU-side time reversal within each sample's active window.

    For every sample, the active window [t_start, t_end] spans the earliest
    and latest active time bin across all neurons. Each neuron's segment is
    flipped via t -> t_start + t_end - t. Samples with fewer than two spikes
    are left unchanged.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).

    Returns:
        Time-reversed tensor with the same shape, dtype, and device.
    """
    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    spike_any = is_spike.any(dim=1)  # (B, T)
    t_idx = torch.arange(T, device=x.device).view(1, T)
    tt = t_idx.expand(B, T)

    big = torch.full((B, T), T, device=x.device, dtype=torch.long)
    neg = torch.full((B, T), -1, device=x.device, dtype=torch.long)
    t_start = torch.where(spike_any, tt, big).min(dim=1).values  # (B,)
    t_end = torch.where(spike_any, tt, neg).max(dim=1).values    # (B,)

    n_spikes = is_spike.sum(dim=(1, 2))  # (B,)
    valid = n_spikes >= 2

    in_window = (
        valid.view(B, 1)
        & (tt >= t_start.view(B, 1))
        & (tt <= t_end.view(B, 1))
    )
    src = torch.where(in_window, (t_start + t_end).view(B, 1) - tt, tt)
    src = src.clamp(0, T - 1)

    src_exp = src.view(B, 1, T).expand(B, C, T).contiguous()
    reversed_x = torch.gather(x, 2, src_exp)
    return reversed_x.view(B, C, H, W, T)


# ---------------------------------------------------------------------------
# Dataset and DataLoaders
# ---------------------------------------------------------------------------

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


def get_split_indices(split_range: tuple[float, float], total: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class SHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with the perturbation/reversal hook at the 2nd hidden layer.

    Forward path:
        Input -> fc1 -> spike -> (delay1) -> fc2 -> spike -> hidden2
              -> [perturb f] -> [reverse?] -> (delay2) -> fc3 -> spike -> output
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

        self.fc1 = nn.utils.weight_norm(slayer.dense(input_dim, hidden_units), name="weight")
        self.fc2 = nn.utils.weight_norm(slayer.dense(hidden_units, hidden_units), name="weight")
        self.fc3 = nn.utils.weight_norm(slayer.dense(hidden_units, num_classes), name="weight")

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

    def _first_and_second_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> fc1 -> spike -> (delay1) -> fc2 -> spike -> hidden2 spikes."""
        h1 = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            h1 = self.delay1(h1)
        return self.slayer.spike(self.fc2(self.slayer.psp(h1)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> (delay2) -> PSP -> fc3 -> spike."""
        x = hidden2
        if self.use_delay:
            x = self.delay2(x)
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

    def _apply_perturbation(self, hidden: torch.Tensor, f: float) -> torch.Tensor:
        """STE wrapper around perturb_hidden_batch."""
        if f <= 0:
            return hidden
        perturbed = perturb_hidden_batch(hidden, f)
        return hidden + (perturbed - hidden).detach()

    def _apply_reversal(self, hidden: torch.Tensor) -> torch.Tensor:
        """STE wrapper around reverse_hidden_batch."""
        reversed_h = reverse_hidden_batch(hidden)
        return hidden + (reversed_h - hidden).detach()

    def forward(
        self,
        x: torch.Tensor,
        f: float = 0.0,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Forward pass with perturbation and optional reversal at the 2nd hidden layer.

        Args:
            x: Input spike trains.
            f: Fraction of 2nd hidden spikes to randomly relocate.
            reverse: Whether to time-reverse the 2nd hidden spikes.

        Returns:
            Output spike tensor.
        """
        x = self._prepare_input(x)
        hidden2 = self._first_and_second_hidden(x)
        hidden2 = self._apply_perturbation(hidden2, f)
        if reverse:
            hidden2 = self._apply_reversal(hidden2)
        return self._output(hidden2)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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


def build_loss_and_optimizer(net: SHDNetwork, lr: float = 0.1) -> tuple:
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)
    return loss_fn, optimizer, scheduler


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    hidden_units: int = 128,
    num_classes: int = 20,
    use_delay: bool = True,
    max_delay: int = 64,
    epochs: int = 1000,
    lr: float = 0.1,
    seed: int = 42,
    patience: int = 300,
    f: float = 0.0,
    reverse: bool = False,
) -> tuple[SHDNetwork, dict]:
    """Train the SHDNetwork with 2nd hidden perturbation/reversal on every batch.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        use_delay: Whether to use learnable delays.
        max_delay: Maximum delay in time steps.
        epochs: Maximum training epochs.
        lr: Learning rate.
        seed: Random seed.
        patience: Early stopping patience.
        f: 2nd hidden-layer relocation level applied during forward passes.
        reverse: Whether 2nd hidden time reversal is active during forward passes.

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = SHDNetwork(input_dim, hidden_units, num_classes, use_delay, max_delay).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    update1 = 0
    update2 = 0
    thea1 = max_delay
    thea2 = max_delay

    log: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "delay_mean": [],
        "f": f,
        "reverse": reverse,
    }

    total_steps = epochs * len(train_loader)
    desc = f"Train f={f} rev={reverse}"
    with tqdm(total=total_steps, desc=desc) as pbar:
        for epoch in range(epochs):
            net.train()
            batch_losses = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros((len(y_batch), num_classes, 1, 1, 1), device=device)
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs = net(x_batch, f=f, reverse=reverse)
                loss = loss_fn.numSpikes(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                pbar.update(1)

            if use_delay:
                if epoch <= 250:
                    net.clamp_delays(max_delay, max_delay)
                else:
                    update1 += 1
                    update2 += 1
                    for name, param in net.named_parameters():
                        if "delay1.delay" in name and update1 > 150:
                            sorted_ = torch.sort(torch.floor(param.detach().flatten()))[0]
                            thea1_val = torch.max(sorted_)
                            if sorted_[108] > (thea1_val - 5):
                                thea1 = int(thea1_val.item()) + 1
                                update1 = 0
                        elif "delay2.delay" in name and update2 > 150:
                            sorted_ = torch.sort(torch.floor(param.detach().flatten()))[0]
                            thea2_val = torch.max(sorted_)
                            if sorted_[108] > (thea2_val - 5):
                                thea2 = int(thea2_val.item()) + 1
                                update2 = 0
                    net.clamp_delays(thea1, thea2)

            net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                    y_batch = y_batch.to(device).long()

                    target = torch.zeros((len(y_batch), num_classes, 1, 1, 1), device=device)
                    target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                    outputs = net(x_batch, f=f, reverse=reverse)
                    val_loss += loss_fn.numSpikes(outputs, target).item()

                    pred = snn.predict.getClass(outputs)
                    correct += (pred.cpu() == y_batch.cpu()).sum().item()
                    total += len(y_batch)

            val_loss /= max(1, len(val_loader))
            val_acc = correct / max(1, total)
            train_loss = np.mean(batch_losses)

            delays = net.get_delays()
            avg_delay = (
                np.mean([np.mean(d) for d in delays.values() if len(d) > 0])
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in net.state_dict().items()}
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_with_repeats(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float,
    reverse: bool,
    num_repeats: int = 3,
) -> dict:
    """Repeat evaluation for mean +/- std error bars.

    Args:
        net: Trained SHDNetwork.
        test_loader: Test DataLoader.
        f: Perturbation fraction.
        reverse: Whether 2nd hidden time reversal is applied.
        num_repeats: Number of independent evaluations.

    Returns:
        Dict with "mean", "std", and "values".
    """
    net.eval()
    accuracies = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        torch.manual_seed(SEED + repeat)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + repeat)

        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device)
                outputs = net(x_batch, f=f, reverse=reverse)
                predicted = snn.predict.getClass(outputs)
                total += y_batch.size(0)
                correct += (predicted.cpu() == y_batch.cpu()).sum().item()
        accuracies.append(correct / total)

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": [float(a) for a in accuracies],
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(dataset_key: str, use_delay: bool) -> None:
    """Run the full reversal sweep for one (dataset, delay) combination.

    Args:
        dataset_key: One of "whole", "part", or "norm".
        use_delay: Whether to use SGD-delay (True) or SGD (False).
    """
    input_dim: int = DATASET_CONFIGS[dataset_key]["input_dim"]
    mat_file: str = DATASET_CONFIGS[dataset_key]["mat_file"]
    delay_tag: str = "delay" if use_delay else "nodelay"
    model_prefix: str = f"inverse_2ndLayer_{dataset_key}_{delay_tag}"

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_key} | Input dim: {input_dim} | Mode: {delay_tag}")
    print(f"{'=' * 70}")

    X_all, Y_all = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    train_loader, val_loader, test_loader = build_dataloaders(
        X_all, Y_all, batch_size=BATCH_SIZE, seed=SEED
    )

    sweep_results: dict[str, dict[float, dict]] = {
        cond: {} for cond, _ in REVERSAL_CONDITIONS
    }
    all_logs: dict[str, dict[float, dict]] = {
        cond: {} for cond, _ in REVERSAL_CONDITIONS
    }

    for cond_name, rev in REVERSAL_CONDITIONS:
        print(f"\n{'#' * 70}")
        print(f"# Condition: {cond_name} (reverse={rev})")
        print(f"{'#' * 70}")
        for f_val in F_VALUES:
            print(f"\n=== Training {model_prefix} | {cond_name} | f={f_val} ===")
            net, training_log = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                hidden_units=HIDDEN_UNITS,
                num_classes=NUM_CLASSES,
                use_delay=use_delay,
                max_delay=MAX_DELAY,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                seed=SEED,
                patience=EARLY_STOP_PATIENCE,
                f=f_val,
                reverse=rev,
            )

            model_path = DATA_DIR / f"{model_prefix}_{cond_name}_f{f_val}.pt"
            torch.save(net.state_dict(), model_path)

            result = test_with_repeats(
                net, test_loader, f=f_val, reverse=rev, num_repeats=NUM_REPEATS
            )
            sweep_results[cond_name][f_val] = result
            all_logs[cond_name][f_val] = training_log
            print(
                f"{cond_name} | f={f_val} | test acc = "
                f"{result['mean']:.4f} +/- {result['std']:.4f}"
                f" | checkpoint -> {model_path}"
            )

    # --- Save aggregate sweep results ---
    results_serialisable = {
        cond: {
            str(f_val): {
                "mean": float(d["mean"]),
                "std": float(d["std"]),
                "values": [float(v) for v in d["values"]],
            }
            for f_val, d in cond_results.items()
        }
        for cond, cond_results in sweep_results.items()
    }
    results_path = LOG_DIR / f"{model_prefix}_reversal_sweep_results.json"
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Save training logs ---
    logs_serialisable = {
        cond: {
            str(f_val): {
                k: ([float(v) for v in vals] if isinstance(vals, list) else vals)
                for k, vals in log.items()
            }
            for f_val, log in cond_logs.items()
        }
        for cond, cond_logs in all_logs.items()
    }
    log_path = LOG_DIR / f"{model_prefix}_training_log.json"
    with open(log_path, "w") as fp:
        json.dump(logs_serialisable, fp, indent=2)
    print(f"Training logs saved to {log_path}")

    # --- Print sweep summary ---
    print(f"\n=== 2nd Hidden-Layer Time Reversal Sweep (SHD {dataset_key}, {delay_tag}) ===")
    for cond_name, _ in REVERSAL_CONDITIONS:
        print(f"\n--- {cond_name} ---")
        for f_val in F_VALUES:
            r = sweep_results[cond_name][f_val]
            print(f"  f={f_val:.1f}:  accuracy = {r['mean']:.4f} +/- {r['std']:.4f}")


def main() -> None:
    """Dispatch single or all-variation sweeps depending on TRAIN_ALL_VARIATIONS."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if TRAIN_ALL_VARIATIONS:
        for dataset_key in DATASET_CONFIGS:
            for use_delay in (False, True):
                run_sweep(dataset_key, use_delay)
    else:
        run_sweep(DATASET_KEY, USE_DELAY)


if __name__ == "__main__":
    main()
