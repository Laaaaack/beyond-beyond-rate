"""Input Per-Spike Jitter — SHD (eval-only, train-once / eval-all).

Loads a pretrained clean (no-perturbation) 2-hidden-layer SLAYER SNN
checkpoint — produced by the hidden-layer jitter experiment
(my_project/code/perturbation/jitter/data) — and sweeps per-spike Gaussian
jitter applied to the *input* spike trains during evaluation only. No training
is performed here.

Per-spike jitter: each input spike is independently shifted in time by a
random offset drawn from N(0, sigma), clipped to [0, T-1] and placed at the
nearest unoccupied time bin. This disrupts precise spike timing while
approximately preserving spike count per neuron.

Reusing the same clean checkpoint that the hidden-layer experiment evaluates —
only changing the perturbation site from the 1st hidden layer to the input —
ensures both experiments probe identical weights, enabling a fair comparison
between perturbing the dataset and perturbing the hidden layer.

Architecture: Input(input_dim) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA)
Sweep (eval only): sigma in {0, 1, 3, 5, 10, 17, 25} ms

A USE_DELAY flag selects SGD-delay vs SGD (no delay). An EVAL_ALL_VARIATION
flag sweeps every (dataset-variant x delay) combination.
"""

# ===================================================================== #
#  Imports and setup
# ===================================================================== #
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


# ===================================================================== #
#  Global configuration
# ===================================================================== #

# When True: evaluate every combination of dataset-variant x use_delay (the
# full grid). When False: run only the single (DATASET_KEY, USE_DELAY)
# configuration selected below.
EVAL_ALL_VARIATION: bool = True

# Network variant: True for SGD-delay, False for SGD (no delay)
USE_DELAY: bool = False

# Dataset variant: "whole", "part", or "norm"
DATASET_KEY: str = "whole"

# Variation option lists swept when EVAL_ALL_VARIATION is True
DATASET_VARIATIONS: list[str] = ["whole", "part", "norm"]
DELAY_VARIATIONS: list[bool] = [False, True]

# --- Dataset configurations ---
DATASET_CONFIGS = {
    "whole": {"mat_file": "../../realistic/shd/shd_data/shd_whole.mat", "input_dim": 700},
    "part":  {"mat_file": "../../realistic/shd/shd_data/shd_part_new.mat", "input_dim": 224},
    "norm":  {"mat_file": "../../realistic/shd/shd_data/shd_norm_new.mat", "input_dim": 224},
}

# --- Pretrained clean checkpoints (no perturbation) ---
# Loaded for evaluation instead of training a new model. These are the clean
# models trained by the hidden-layer jitter experiment, so the input- and
# hidden-perturbation experiments share identical weights.
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "../../../../code/perturbation/jitter/data")
CHECKPOINT_TEMPLATE = "jitter_{dataset_key}_{delay_tag}_trained.pt"

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

# --- Jitter sweep: sigma values in ms ---
SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]

# --- Evaluation ---
NUM_REPEATS: int = 3


# ===================================================================== #
#  Dataset loading
# ===================================================================== #
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


# ===================================================================== #
#  Per-spike jitter utilities (core perturbation)
# ===================================================================== #
def jitter_spike_train(
    spike_train: np.ndarray,
    sigma: float = 0.0,
    max_attempts: int = 50,
) -> np.ndarray:
    if sigma <= 0:
        return spike_train.copy()

    num_neurons, T = spike_train.shape
    new_train = np.zeros_like(spike_train)

    for neuron_idx in range(num_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        if len(spike_times) == 0:
            continue

        for old_time in spike_times:
            inserted = False
            attempts = 0
            while not inserted and attempts < max_attempts:
                attempts += 1
                jittered_time = int(round(old_time + np.random.normal(0, sigma)))
                jittered_time = np.clip(jittered_time, 0, T - 1)

                if new_train[neuron_idx, jittered_time] == 0:
                    new_train[neuron_idx, jittered_time] = 1
                    inserted = True

            if not inserted:
                new_train[neuron_idx, old_time] = 1

    return new_train


def jitter_input_batch(
    input_spikes: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply per-spike jitter to a batch of input spike trains.

    Args:
        input_spikes: Input spike tensor of shape (B, num_neurons, T).
        sigma: Jitter standard deviation in ms.

    Returns:
        Perturbed input spike tensor of shape (B, num_neurons, T) on CPU.
    """
    spikes_np = input_spikes.detach().cpu().numpy()
    batch_size = spikes_np.shape[0]
    perturbed = np.zeros_like(spikes_np)

    for sample_idx in range(batch_size):
        perturbed[sample_idx] = jitter_spike_train(spikes_np[sample_idx], sigma)

    return torch.from_numpy(perturbed).float()


# ===================================================================== #
#  Dataset wrapper, splitting, and dataloader construction
# ===================================================================== #
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


# ===================================================================== #
#  Network architecture
# ===================================================================== #
class JitterSHDNetwork(nn.Module):

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
        # Input jitter, when applied, perturbs the spike trains before they
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


# ===================================================================== #
#  Pretrained model loading
# ===================================================================== #
def load_pretrained_model(
    checkpoint_path: str,
    input_dim: int,
    use_delay: bool,
) -> JitterSHDNetwork:
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

    net = JitterSHDNetwork(
        input_dim, HIDDEN_UNITS, NUM_CLASSES, use_delay, MAX_DELAY
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


# ===================================================================== #
#  Evaluation / jitter sweep
# ===================================================================== #
def test_with_jitter(
    net: JitterSHDNetwork,
    test_loader: DataLoader,
    sigma: float = 0.0,
) -> float:
    # Eval-only: jitter the input spike trains, then run the clean forward.
    # The numpy round-trip in jitter_input_batch is not autograd-safe, so this
    # must stay inside torch.no_grad().
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if sigma > 0:
                x_batch = jitter_input_batch(x_batch, sigma)
            y_batch = y_batch.to(device)

            outputs = net(x_batch)
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


# ===================================================================== #
#  Single-variation runner: load checkpoint, sweep jitter, save results
# ===================================================================== #
def run_variation(dataset_key: str, use_delay: bool) -> dict:
    # Derive names and output paths locally so configurations never
    # overwrite each other's files.
    input_dim = DATASET_CONFIGS[dataset_key]["input_dim"]
    mat_file = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["mat_file"])
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"jitter_{dataset_key}_{delay_tag}"

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

    # Evaluate across the input-jitter sweep
    print(f"\n  --- Input-jitter sweep at evaluation ---")
    all_test_results: dict[int, dict] = {}
    for sigma in SIGMA_VALUES:
        test_result = test_with_repeats(net, test_loader, sigma=sigma)
        all_test_results[sigma] = test_result
        print(
            f"    sigma={sigma:3d} ms | "
            f"accuracy = {test_result['mean']:.4f} +/- {test_result['std']:.4f}"
        )

    # Save sweep results
    sweep_serialisable = {
        str(sigma): {
            "mean": data["mean"],
            "std": data["std"],
            "values": [float(v) for v in data["values"]],
        }
        for sigma, data in all_test_results.items()
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


# ===================================================================== #
#  Main dispatcher
# ===================================================================== #
def main() -> None:
    print(f"Using device: {device}")

    if EVAL_ALL_VARIATION:
        config_grid = [
            (dk, ud) for dk in DATASET_VARIATIONS for ud in DELAY_VARIATIONS
        ]
    else:
        config_grid = [(DATASET_KEY, USE_DELAY)]

    print(f"Running {len(config_grid)} configuration(s): {config_grid}")

    all_runs: dict[tuple[str, bool], dict] = {}
    for dataset_key, use_delay in config_grid:
        all_runs[(dataset_key, use_delay)] = run_variation(dataset_key, use_delay)

    return all_runs


if __name__ == "__main__":
    main()
