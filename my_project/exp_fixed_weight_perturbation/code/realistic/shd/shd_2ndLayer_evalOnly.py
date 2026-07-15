"""Experiment 2A: SHD - 2nd-Hidden-Layer Perturbation (evaluation only).

Load pre-trained 2-hidden-layer SLAYER SNN checkpoints from `data/` and sweep
spike-timing perturbation applied at the output of the 2nd hidden layer. No
training is performed.

The network definition and test split mirror shd_2ndLayer_train.py, so the
checkpoints that script produces load without modification. Note the difference
from shd_evalOnly.py: delay1 is folded into the 1st hidden layer and delay2 into
the output routing, so the perturbation hook sees strictly binary
2nd-hidden-layer spikes.

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

# When True, run every combination of {delay, no_delay} x {norm, part, whole}.
# When False, run only the single (USE_DELAY, DATASET_KEY) configuration below.
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

# Data split ratios. Must match shd_2ndLayer_train.py, or the "test" set would
# overlap the data the checkpoints were fitted on.
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# Model hyper-parameters (must match the checkpoints)
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
BATCH_SIZE: int = 128
SEED: int = 42
MAX_DELAY: int = 64

# Hidden-perturbation sweep
F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_REPEATS: int = 3

# All variations to sweep when EVAL_ALL_VARIATION is True
ALL_DATASET_KEYS: list[str] = ["norm", "part", "whole"]
ALL_DELAY_OPTIONS: list[bool] = [True, False]

# Checkpoints in data/ are stored as "<model_prefix><CHECKPOINT_SUFFIX>".
CHECKPOINT_SUFFIX: str = "_f0.0.pt"


# =====================================================================
# Load SHD dataset
# =====================================================================

def load_shd_data(
    mat_path: str,
    target_T: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an SHD .mat file, zero-padding the time dimension if needed.

    Args:
        mat_path: Path to the .mat file holding "X" (spikes) and "Y" (labels).
        target_T: Number of timesteps to pad each sample out to.

    Returns:
        A tuple of (X, Y), where X has shape (n_samples, n_neurons, target_T).
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


# =====================================================================
# Hidden-layer spike perturbation
# =====================================================================

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
    avoiding the CPU/numpy round-trip that dominates cost when perturbation
    runs on every batch.

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
    # Random key per time bin; non-spike bins sort last.
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


# =====================================================================
# Dataset and data splitting
# =====================================================================

class SpikeDataset(Dataset):
    """Wraps spike arrays as a torch Dataset, casting to float on access."""

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
    """Return the contiguous sample indices covered by a split range."""
    start = int(total * split_range[0])
    end = int(total * split_range[1])
    return np.arange(start, end)


def build_test_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 128,
) -> DataLoader:
    """Build a loader over the test split only.

    Uses the same contiguous TEST_RANGE slice as shd_2ndLayer_train.py's
    build_dataloaders(); the train shuffle there does not affect which samples
    land in the test split.

    Args:
        X: All spike samples.
        Y: All labels.
        batch_size: Evaluation batch size.

    Returns:
        A DataLoader over the test split, unshuffled.
    """
    N = len(Y)
    test_idx = get_split_indices(TEST_RANGE, N)
    test_ds = SpikeDataset(X[test_idx], Y[test_idx])

    n_classes = len(np.unique(Y[test_idx]))
    print(f"Test: {len(test_ds)} samples | classes present: {n_classes}")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# =====================================================================
# Network architecture
# =====================================================================

class SHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with an optional learnable-delay variant.

    Structurally identical to the network in shd_2ndLayer_train.py so that its
    checkpoints load with strict key matching.
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
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        return x

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        # Compute block for hidden layer 2. Returns BINARY spikes.
        # delay2 is applied later (in _output), so the perturbation hook
        # sees a strictly 0/1 tensor.
        return self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        # Routing (delay2) + output layer.
        x = self.delay2(hidden2) if self.use_delay else hidden2
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)
        return self._output(hidden2)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        f: float = 0.0,
    ) -> torch.Tensor:
        """Run a forward pass, perturbing the 2nd hidden layer's spikes.

        Args:
            x: Input spikes, 3-D (batch, neurons, T) or SLAYER's 5-D format.
            f: Probability that any given hidden-layer spike is relocated.

        Returns:
            Output-layer spikes in SLAYER's 5-D format.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)

        if f > 0:
            hidden2 = perturb_hidden_batch(hidden2, f)

        return self._output(hidden2)

    def get_delays(self) -> dict[str, np.ndarray]:
        """Return the learned delay parameters, or an empty dict if unused."""
        delays = {}
        if self.use_delay:
            delays["delay1"] = self.delay1.delay.data.cpu().numpy()
            delays["delay2"] = self.delay2.delay.data.cpu().numpy()
        return delays


# =====================================================================
# Checkpoint loading
# =====================================================================

def load_trained_model(
    checkpoint_path: str,
    input_dim: int,
    use_delay: bool,
) -> SHDNetwork:
    """Build an SHDNetwork and restore its weights from a checkpoint.

    Args:
        checkpoint_path: Path to a state_dict saved by shd_2ndLayer_train.py.
        input_dim: Number of input neurons for the dataset variant.
        use_delay: Whether the checkpoint is a delay-enabled variant.

    Returns:
        The network in eval mode on `device`.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = SHDNetwork(
        input_dim, HIDDEN_UNITS, NUM_CLASSES, use_delay, MAX_DELAY
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return net


# =====================================================================
# Testing with hidden-layer perturbation
# =====================================================================

def test_with_hidden_perturbation(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
) -> float:
    """Measure test accuracy with the 2nd hidden layer perturbed.

    Args:
        net: A loaded SHDNetwork.
        test_loader: Loader over the test split.
        f: Probability that any given hidden-layer spike is relocated.

    Returns:
        Classification accuracy in [0, 1].
    """
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
    """Sweep perturbation strength, repeating each level with fresh RNG seeds.

    Args:
        net: A loaded SHDNetwork.
        test_loader: Loader over the test split.
        f_values: Perturbation probabilities to evaluate.
        num_repeats: Number of seeded repeats per f value.

    Returns:
        A dict mapping each f value to {"mean", "std", "values"}.
    """
    results: dict[float, dict] = {}

    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            torch.manual_seed(SEED + repeat)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED + repeat)
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

def run_variation(use_delay: bool, dataset_key: str) -> dict[float, dict]:
    """Evaluate one (delay x dataset) checkpoint across the perturbation sweep.

    Args:
        use_delay: True for the SGD-delay variant, False for plain SGD.
        dataset_key: One of "whole", "part", or "norm".

    Returns:
        The sweep results, as returned by run_hidden_perturbation_sweep().
    """
    input_dim = DATASET_CONFIGS[dataset_key]["input_dim"]
    mat_file = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["mat_file"])
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"shd_2ndLayer_{dataset_key}_{delay_tag}"

    data_dir = os.path.join(SCRIPT_DIR, "data")
    log_dir = os.path.join(SCRIPT_DIR, "log")
    checkpoint_path = os.path.join(
        data_dir, f"{model_prefix}{CHECKPOINT_SUFFIX}"
    )

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_key} | Input dim: {input_dim}")
    print(f"Network mode: {'SGD-delay' if use_delay else 'SGD (no delay)'}")
    print(f"Model prefix: {model_prefix}")
    print(f"{'=' * 70}")

    # Load the trained model
    net = load_trained_model(checkpoint_path, input_dim, use_delay)

    delays = net.get_delays()
    if delays:
        avg_delay = np.mean([np.mean(d) for d in delays.values() if len(d) > 0])
        print(f"Mean learned delay: {avg_delay:.1f}")

    # Load data and build the test loader
    X_all, Y_all = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X_all, Y_all, batch_size=BATCH_SIZE)

    # Hidden-perturbation sweep
    print(
        f"=== 2nd-Hidden-Layer Perturbation Sweep "
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

    return sweep_results


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    """Evaluate either every variation or the single configured one."""
    if EVAL_ALL_VARIATION:
        for dataset_key in ALL_DATASET_KEYS:
            for use_delay in ALL_DELAY_OPTIONS:
                run_variation(use_delay, dataset_key)
    else:
        run_variation(USE_DELAY, DATASET_KEY)


if __name__ == "__main__":
    main()
