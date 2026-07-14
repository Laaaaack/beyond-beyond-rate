"""Experiment 2A: SHD - Hidden-Layer Perturbation.

Train a 2-hidden-layer SLAYER SNN on the Spiking Heidelberg Digits (SHD)
dataset with no perturbation (f=0), then evaluate it by applying spike-timing
perturbation at the output of the 1st hidden layer instead of the input.

Architecture: Input -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA)
Dataset variants: whole (700 input neurons), part (224), norm (224)
"""

import os
import json
import random

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Resolve all file paths relative to this script's location, not the CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# Global configuration
# =====================================================================

# When True, run every combination of {delay, no_delay} x {norm, part, whole}.
# When False, run only the single (USE_DELAY, DATASET_KEY) configuration below.
TRAIN_ALL_VARIATION: bool = False

# Network variant: True for SGD-delay, False for SGD (no delay)
USE_DELAY: bool = True

# Dataset variant: "whole", "part", or "norm"
DATASET_KEY: str = "norm"

DATASET_CONFIGS = {
    "whole": {"mat_file": "shd_data/shd_whole.mat", "input_dim": 700},
    "part":  {"mat_file": "shd_data/shd_part_new.mat", "input_dim": 224},
    "norm":  {"mat_file": "shd_data/shd_norm_new.mat", "input_dim": 224},
}

# SLAYER neuron and simulation descriptors
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

# Data split ratios
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# Training hyper-parameters
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
EPOCHS: int = 1250
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
SEED: int = 42
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

# Hidden-perturbation sweep
F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_REPEATS: int = 3

# All variations to sweep when TRAIN_ALL_VARIATION is True
ALL_DATASET_KEYS: list[str] = ["norm", "part", "whole"]
ALL_DELAY_OPTIONS: list[bool] = [True, False]


# =====================================================================
# Load SHD dataset
# =====================================================================

def load_shd_data(mat_path: str, target_T: int = 200) -> tuple[np.ndarray, np.ndarray]:
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


# =====================================================================
# Hidden-layer spike perturbation
# =====================================================================

def partial_randomize_spike_train(
    spike_train: np.ndarray,
    f: float = 0.0,
    max_attempts: int = 50,
) -> np.ndarray:
    if f <= 0:
        return spike_train

    num_neurons, T = spike_train.shape
    new_train = np.copy(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        for old_time in spike_times:
            if np.random.rand() < f:
                new_train[neuron_idx, old_time] = 0
                inserted = False
                attempts = 0
                while not inserted and attempts < max_attempts:
                    attempts += 1
                    new_t = np.random.randint(0, T)
                    if new_train[neuron_idx, new_t] == 0:
                        new_train[neuron_idx, new_t] = 1
                        inserted = True
    return new_train


def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float,
) -> torch.Tensor:
    dev = hidden_spikes.device
    spikes_np = hidden_spikes.cpu().numpy()
    B, C, H, W, T = spikes_np.shape

    for b in range(B):
        sample = spikes_np[b, :, 0, 0, :]  # (C, T)
        spikes_np[b, :, 0, 0, :] = partial_randomize_spike_train(sample, f)

    return torch.from_numpy(spikes_np).to(dev)


# =====================================================================
# Dataset and data splitting
# =====================================================================

class SpikeDataset(Dataset):

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
    start = int(total * split_range[0])
    end = int(total * split_range[1])
    return np.arange(start, end)


def build_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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


# =====================================================================
# Network architecture
# =====================================================================

class SHDNetwork(nn.Module):

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

        # Three FC layers with weight normalisation
        self.fc1 = nn.utils.weight_norm(
            slayer.dense(input_dim, hidden_units), name="weight"
        )
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(hidden_units, hidden_units), name="weight"
        )
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(hidden_units, num_classes), name="weight"
        )

        # Optional learnable delay modules
        if use_delay:
            self.delay1 = slayer.delay(hidden_units)
            self.delay2 = slayer.delay(hidden_units)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(device)

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # Input -> PSP -> fc1 -> spike. Returns raw binary hidden1 spikes (pre-delay).
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        # (delay1) -> PSP -> fc2 -> spike -> (delay2) -> PSP -> fc3 -> spike.
        if self.use_delay:
            hidden1 = self.delay1(hidden1)
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        f: float = 0.0,
    ) -> torch.Tensor:
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)

        if f > 0:
            hidden1 = perturb_hidden_batch(hidden1, f)

        return self._second_hidden_and_output(hidden1)

    def clamp_delays(self, max1: int = 64, max2: int = 64) -> None:
        if not self.use_delay:
            return
        self.delay1.delay.data.clamp_(0, max1)
        self.delay2.delay.data.clamp_(0, max2)

    def get_delays(self) -> dict[str, np.ndarray]:
        delays = {}
        if self.use_delay:
            delays["delay1"] = self.delay1.delay.data.cpu().numpy()
            delays["delay2"] = self.delay2.delay.data.cpu().numpy()
        return delays


# =====================================================================
# Training loop
# =====================================================================

def set_seed(seed: int) -> None:
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
    hidden_units: int = 128,
    num_classes: int = 20,
    use_delay: bool = True,
    max_delay: int = 64,
    epochs: int = 1000,
    lr: float = 0.1,
    seed: int = 42,
    patience: int = 300,
) -> tuple[SHDNetwork, dict]:
    set_seed(seed)

    net = SHDNetwork(
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
    with tqdm(total=total_steps, desc="Training") as pbar:
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

                outputs = net(x_batch)
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

            # --- Validate ---
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


# =====================================================================
# Testing with hidden-layer perturbation
# =====================================================================

def test_with_hidden_perturbation(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
) -> float:
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device)

            outputs = net.forward_with_hidden_perturbation(x_batch, f=f)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def run_hidden_perturbation_sweep(
    net: SHDNetwork,
    test_loader: DataLoader,
    f_values: list[float],
    num_repeats: int = 3,
) -> dict[float, dict]:
    results: dict[float, dict] = {}

    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            acc = test_with_hidden_perturbation(net, test_loader, f=f)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[f] = {
            "mean": mean_acc, "std": std_acc, "values": accuracies
        }
        print(f"  f={f:.1f}:  accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

    return results


# =====================================================================
# Run one variation
# =====================================================================

def run_variation(use_delay: bool, dataset_key: str) -> None:
    input_dim = DATASET_CONFIGS[dataset_key]["input_dim"]
    mat_file = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["mat_file"])
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"shd_{dataset_key}_{delay_tag}"

    data_dir = os.path.join(SCRIPT_DIR, "data")
    log_dir = os.path.join(SCRIPT_DIR, "log")

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_key} | Input dim: {input_dim}")
    print(f"Network mode: {'SGD-delay' if use_delay else 'SGD (no delay)'}")
    print(f"Model prefix: {model_prefix}")
    print(f"{'=' * 70}")

    # Load data
    X_all, Y_all = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])

    # Build data loaders (unperturbed)
    train_loader, val_loader, test_loader = build_dataloaders(
        X_all, Y_all, batch_size=BATCH_SIZE, seed=SEED
    )

    # Train on unperturbed data (f=0)
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
    )

    # Quick sanity check: accuracy on clean test set (f=0)
    clean_acc = test_with_hidden_perturbation(net, test_loader, f=0.0)
    print(f"\nClean test accuracy (f=0): {clean_acc:.4f}")

    # Save trained model
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, f"{model_prefix}_trained.pt")
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Hidden-perturbation sweep
    print(
        f"=== Hidden-Layer Perturbation Sweep "
        f"(SHD {dataset_key}, {delay_tag}) ==="
    )
    sweep_results = run_hidden_perturbation_sweep(
        net, test_loader, f_values=F_VALUES, num_repeats=NUM_REPEATS
    )

    # Save sweep results
    results_serialisable = {
        str(f_val): {
            "mean": float(data["mean"]),
            "std": float(data["std"]),
            "values": [float(v) for v in data["values"]],
        }
        for f_val, data in sweep_results.items()
    }

    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(
        log_dir, f"{model_prefix}_hidden_perturbation_results.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")

    # Save training log
    log_path = os.path.join(log_dir, f"{model_prefix}_training_log.json")
    training_log_serialisable = {
        k: [float(v) for v in vals] if isinstance(vals, list) else vals
        for k, vals in training_log.items()
    }
    with open(log_path, "w") as fp:
        json.dump(training_log_serialisable, fp, indent=2)
    print(f"Training log saved to {log_path}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if TRAIN_ALL_VARIATION:
        for dataset_key in ALL_DATASET_KEYS:
            for use_delay in ALL_DELAY_OPTIONS:
                run_variation(use_delay, dataset_key)
    else:
        run_variation(USE_DELAY, DATASET_KEY)


if __name__ == "__main__":
    main()