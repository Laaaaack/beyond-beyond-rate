"""SHD — Input-Layer Perturbation (eval-only, train-once / eval-all).

Loads a pretrained clean (no-perturbation) 2-hidden-layer SLAYER SNN
checkpoint — from the realistic SHD experiment
(my_project/code/realistic/shd/data) — and evaluates the frozen model by
applying spike-timing perturbation to the *input* spike trains across a sweep
of perturbation levels f. No training is performed here.

This mirrors the original Beyond Rate input-perturbation experiments but
follows the train-once / eval-all protocol: a single clean-trained model is
evaluated at every f. Reusing the hidden-layer experiment's checkpoint means
the input- and hidden-perturbation results come from identical weights.

Architecture: Input -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA)
Dataset variants: whole (700 input neurons), part (224), norm (224)
"""

import os
import json

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Resolve all file paths relative to this script's location, not the CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# Global configuration
# =====================================================================

# When True, evaluate every combination of {delay, no_delay} x {norm, part,
# whole}. When False, run only the single (USE_DELAY, DATASET_KEY) below.
EVAL_ALL_VARIATION: bool = True

# Network variant: True for SGD-delay, False for SGD (no delay)
USE_DELAY: bool = True

# Dataset variant: "whole", "part", or "norm"
DATASET_KEY: str = "norm"

DATASET_CONFIGS = {
    "whole": {"mat_file": "shd_data/shd_whole.mat", "input_dim": 700},
    "part":  {"mat_file": "shd_data/shd_part_new.mat", "input_dim": 224},
    "norm":  {"mat_file": "shd_data/shd_norm_new.mat", "input_dim": 224},
}

# --- Pretrained clean checkpoints (no perturbation) ---
# Loaded for evaluation instead of training a new model, so the input- and
# hidden-perturbation experiments share identical weights.
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "../../../../code/realistic/shd/data")
CHECKPOINT_TEMPLATE = "shd_{dataset_key}_{delay_tag}_trained.pt"

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

# Network / evaluation hyper-parameters
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
BATCH_SIZE: int = 128
SEED: int = 42
MAX_DELAY: int = 64

# Input-perturbation sweep
F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_REPEATS: int = 3

# All variations to sweep when EVAL_ALL_VARIATION is True
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
# Input-layer spike perturbation
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


def perturb_input_batch(
    input_spikes: torch.Tensor,
    f: float,
) -> torch.Tensor:
    """Apply spike-timing perturbation to a batch of input spike trains.

    Args:
        input_spikes: Input spike tensor of shape (B, num_neurons, T).
        f: Perturbation fraction.

    Returns:
        Perturbed input spike tensor of shape (B, num_neurons, T) on CPU.
    """
    spikes_np = input_spikes.detach().cpu().numpy()
    batch_size = spikes_np.shape[0]
    perturbed = np.zeros_like(spikes_np)

    for sample_idx in range(batch_size):
        perturbed[sample_idx] = partial_randomize_spike_train(spikes_np[sample_idx], f)

    return torch.from_numpy(perturbed).float()


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard forward pass: Input -> PSP -> fc1 -> spike -> (delay1)
        # -> PSP -> fc2 -> spike -> (delay2) -> PSP -> fc3 -> spike.
        # Input perturbation, when applied, perturbs the spike trains before
        # they reach this method, so the loaded checkpoint stays a clean
        # network.
        x = self._prepare_input(x)
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

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
# Pretrained model loading
# =====================================================================

def load_pretrained_model(
    checkpoint_path: str,
    input_dim: int,
    use_delay: bool,
) -> SHDNetwork:
    """Load a clean-trained checkpoint into a fresh network for evaluation.

    Args:
        checkpoint_path: Path to the saved state_dict (.pt).
        input_dim: Number of input neurons (matches the dataset variant).
        use_delay: Whether the checkpoint was trained with learnable delays.

    Returns:
        The network in eval mode with the checkpoint weights loaded.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = SHDNetwork(
        input_dim, HIDDEN_UNITS, NUM_CLASSES, use_delay, MAX_DELAY
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


# =====================================================================
# Testing with input-layer perturbation
# =====================================================================

def test_with_input_perturbation(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
) -> float:
    # Eval-only: perturb the input spike trains, then run the clean forward.
    # The numpy round-trip in perturb_input_batch is not autograd-safe, so this
    # must stay inside torch.no_grad().
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if f > 0:
                x_batch = perturb_input_batch(x_batch, f)
            y_batch = y_batch.to(device)

            outputs = net(x_batch)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def run_input_perturbation_sweep(
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
            acc = test_with_input_perturbation(net, test_loader, f=f)
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

    log_dir = os.path.join(SCRIPT_DIR, "log")

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_key} | Input dim: {input_dim}")
    print(f"Network mode: {'SGD-delay' if use_delay else 'SGD (no delay)'}")
    print(f"Model prefix: {model_prefix}")
    print(f"{'=' * 70}")

    # Load the pretrained clean checkpoint (no training performed; fail fast
    # before loading data if the checkpoint is missing).
    checkpoint_path = os.path.join(
        PRETRAINED_DIR,
        CHECKPOINT_TEMPLATE.format(dataset_key=dataset_key, delay_tag=delay_tag),
    )
    print(f"Loading clean checkpoint: {checkpoint_path}")
    net = load_pretrained_model(checkpoint_path, input_dim, use_delay)

    # Load data and build loaders (only the test split is used)
    X_all, Y_all = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    _, _, test_loader = build_dataloaders(
        X_all, Y_all, batch_size=BATCH_SIZE, seed=SEED
    )

    # Quick sanity check: accuracy on clean test set (f=0)
    clean_acc = test_with_input_perturbation(net, test_loader, f=0.0)
    print(f"\nClean test accuracy (f=0): {clean_acc:.4f}")

    # Input-perturbation sweep
    print(
        f"=== Input-Layer Perturbation Sweep "
        f"(SHD {dataset_key}, {delay_tag}) ==="
    )
    sweep_results = run_input_perturbation_sweep(
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
        log_dir, f"{model_prefix}_input_perturbation_results.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if EVAL_ALL_VARIATION:
        configs = [
            (use_delay, dataset_key)
            for dataset_key in ALL_DATASET_KEYS
            for use_delay in ALL_DELAY_OPTIONS
        ]
    else:
        configs = [(USE_DELAY, DATASET_KEY)]

    for use_delay, dataset_key in configs:
        try:
            run_variation(use_delay, dataset_key)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")


if __name__ == "__main__":
    main()
