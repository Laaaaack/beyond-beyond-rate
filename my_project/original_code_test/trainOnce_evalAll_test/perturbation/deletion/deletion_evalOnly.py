"""Input Spike Deletion — SHD (eval-only, train-once / eval-all).

Loads a pretrained clean (no-perturbation) 2-hidden-layer SLAYER SNN
checkpoint — from the hidden-layer deletion experiment
(my_project/code/perturbation/deletion/data) — and sweeps per-spike deletion
applied to the *input* spike trains during evaluation only. No training is
performed here.

Each input spike is independently dropped with probability p_d. This reduces
the total spike count while preserving the timing of surviving spikes.

Reusing the same clean checkpoint that the hidden-layer experiment evaluates —
only changing the perturbation site from the 1st hidden layer to the input —
ensures both experiments probe identical weights, enabling a fair comparison
between perturbing the dataset and perturbing the hidden layer.

Perturbation sweep (eval only):
    p_d      : 0.0, 0.2, 0.4, 0.6, 0.8
    Datasets : SHD whole, part, norm
    Network  : SGD (no delay) / SGD-delay
"""

# ============================================================
# 1. Imports and Setup
# ============================================================
import os
import sys
import json

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Resolve all file paths relative to this script's location, not the CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add SLAYER to path
sys.path.append(os.path.join(SCRIPT_DIR, "../../../temporal_shd_project/code/src"))
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Global Configuration
# ============================================================

# =====================================================================
# When True: evaluate every combination of dataset variant in
# DATASET_VARIANTS x use_delay in DELAY_OPTIONS — the full grid. The single
# (DATASET_KEY, USE_DELAY) below is used as the default config when this
# flag is False; under the grid it is ignored by the run loop.
# =====================================================================
EVAL_ALL_VARIATION: bool = True

# Grid option lists used when EVAL_ALL_VARIATION is True
DATASET_VARIANTS: list[str] = ["whole", "part", "norm"]
DELAY_OPTIONS: list[bool] = [False, True]

# =====================================================================
# Network variant: set to True for SGD-delay, False for SGD (no delay)
# =====================================================================
USE_DELAY: bool = True

# =====================================================================
# Dataset variant: "whole", "part", or "norm"
# =====================================================================
DATASET_KEY: str = "whole"

# --- Dataset configurations ---
DATASET_CONFIGS = {
    "whole": {"mat_file": "../../realistic/shd/shd_data/shd_whole.mat", "input_dim": 700},
    "part":  {"mat_file": "../../realistic/shd/shd_data/shd_part_new.mat", "input_dim": 224},
    "norm":  {"mat_file": "../../realistic/shd/shd_data/shd_norm_new.mat", "input_dim": 224},
}

# --- Pretrained clean checkpoints (no perturbation) ---
# Loaded for evaluation instead of training a new model, so the input- and
# hidden-perturbation experiments share identical weights. The deletion data
# dir holds the clean checkpoints under the 2nd-layer naming with p_d=0.0
# ("pd00"), which is a model trained without any deletion.
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "../../../../code/perturbation/deletion/data")
CHECKPOINT_TEMPLATE = "deletion_2ndLayer_{dataset_key}_{delay_tag}_pd00.pt"

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

# --- Network / evaluation hyper-parameters ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
BATCH_SIZE: int = 128
SEED: int = 42
MAX_DELAY: int = 64

# --- Deletion sweep: deletion probability p_d ---
PD_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8]

# --- Evaluation ---
NUM_REPEATS: int = 3


# ============================================================
# 3. Load SHD Dataset
# ============================================================
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


# ============================================================
# 4. Spike Deletion Utility
# ============================================================
def delete_spike_train(
    spike_train: np.ndarray,
    p_d: float = 0.0,
) -> np.ndarray:
    if p_d <= 0:
        return spike_train.copy()
    if p_d >= 1:
        return np.zeros_like(spike_train)

    num_neurons, T = spike_train.shape
    new_train = np.zeros_like(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        if len(spike_times) == 0:
            continue

        keep_mask = np.random.rand(len(spike_times)) > p_d
        kept_spikes = spike_times[keep_mask]

        new_train[neuron_idx, kept_spikes] = 1

    return new_train


def delete_input_batch(
    input_spikes: torch.Tensor,
    p_d: float,
) -> torch.Tensor:
    """Apply per-spike deletion to a batch of input spike trains.

    Args:
        input_spikes: Input spike tensor of shape (B, num_neurons, T).
        p_d: Per-spike deletion probability.

    Returns:
        Perturbed input spike tensor of shape (B, num_neurons, T) on CPU.
    """
    spikes_np = input_spikes.detach().cpu().numpy()
    batch_size = spikes_np.shape[0]
    perturbed = np.zeros_like(spikes_np)

    for sample_idx in range(batch_size):
        perturbed[sample_idx] = delete_spike_train(spikes_np[sample_idx], p_d)

    return torch.from_numpy(perturbed).float()


# ============================================================
# 5. Dataset and Data Splitting
# ============================================================
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


# ============================================================
# 6. Network Architecture
# ============================================================
class DeletionSHDNetwork(nn.Module):
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
        # Input deletion, when applied, perturbs the spike trains before they
        # reach this method, so the loaded checkpoint stays a clean network.
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


# ============================================================
# 7. Pretrained Model Loading
# ============================================================
def load_pretrained_model(
    checkpoint_path: str,
    input_dim: int,
    use_delay: bool,
) -> DeletionSHDNetwork:
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

    net = DeletionSHDNetwork(
        input_dim, HIDDEN_UNITS, NUM_CLASSES, use_delay, MAX_DELAY
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


# ============================================================
# 8. Testing with Input-Layer Spike Deletion
# ============================================================
def test_with_deletion(
    net: DeletionSHDNetwork,
    test_loader: DataLoader,
    p_d: float = 0.0,
) -> float:
    # Eval-only: delete spikes from the input spike trains, then run the clean
    # forward. The numpy round-trip in delete_input_batch is not autograd-safe,
    # so this must stay inside torch.no_grad().
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if p_d > 0:
                x_batch = delete_input_batch(x_batch, p_d)
            y_batch = y_batch.to(device)

            outputs = net(x_batch)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: DeletionSHDNetwork,
    test_loader: DataLoader,
    p_d: float,
    num_repeats: int = NUM_REPEATS,
) -> dict:
    accuracies = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        acc = test_with_deletion(net, test_loader, p_d=p_d)
        accuracies.append(acc)

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": accuracies,
    }


# ============================================================
# 9. Run: Load Checkpoint, then Sweep Deletion at Eval Time
# ============================================================
def run_variation(dataset_key: str, use_delay: bool) -> dict:
    # Derive per-config names and paths locally so different configs do not
    # overwrite each other's files.
    input_dim = DATASET_CONFIGS[dataset_key]["input_dim"]
    mat_file = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["mat_file"])
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"deletion_{dataset_key}_{delay_tag}"

    log_dir = os.path.join(SCRIPT_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"#  Configuration: dataset={dataset_key} | {delay_tag}")
    print(f"{'#'*70}")

    # Load dataset for this configuration (only the test split is used)
    X_cfg, Y_cfg = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    _, _, test_loader = build_dataloaders(
        X_cfg, Y_cfg, batch_size=BATCH_SIZE, seed=SEED
    )

    # Load the pretrained clean checkpoint (no training performed)
    checkpoint_path = os.path.join(
        PRETRAINED_DIR,
        CHECKPOINT_TEMPLATE.format(dataset_key=dataset_key, delay_tag=delay_tag),
    )
    print(f"\n  --- Loading clean checkpoint ---\n  {checkpoint_path}")
    net = load_pretrained_model(checkpoint_path, input_dim, use_delay)

    # Evaluate across the input-deletion sweep
    print(f"\n  --- Input-deletion sweep at evaluation ---")
    all_test_results: dict[float, dict] = {}
    for p_d in PD_VALUES:
        test_result = test_with_repeats(net, test_loader, p_d=p_d)
        all_test_results[p_d] = test_result
        print(
            f"    p_d={p_d:.1f} | "
            f"accuracy = {test_result['mean']:.4f} +/- {test_result['std']:.4f}"
        )

    # --- Save sweep results ---
    sweep_serialisable = {
        str(p_d): {
            "mean": data["mean"],
            "std": data["std"],
            "values": [float(v) for v in data["values"]],
        }
        for p_d, data in all_test_results.items()
    }
    results_path = os.path.join(log_dir, f"{model_prefix}_input_perturbation_results.json")
    with open(results_path, "w") as fp:
        json.dump(sweep_serialisable, fp, indent=2)
    print(f"  Sweep results saved to {results_path}")

    return {
        "net": net,
        "all_test_results": all_test_results,
        "dataset_key": dataset_key,
        "use_delay": use_delay,
        "delay_tag": delay_tag,
        "model_prefix": model_prefix,
        "checkpoint_path": checkpoint_path,
    }


def main() -> None:
    if EVAL_ALL_VARIATION:
        # Sweep every (dataset variant) x (delay/no-delay) combination.
        for dataset_key in DATASET_VARIANTS:
            for use_delay in DELAY_OPTIONS:
                run_variation(dataset_key, use_delay)
    else:
        # Single config picked by the global flags above.
        run_variation(DATASET_KEY, USE_DELAY)


if __name__ == "__main__":
    main()
