"""SSC — Input-Layer Perturbation (eval-only, train-once / eval-all).

Loads a pretrained clean (no-perturbation) 2-hidden-layer SLAYER SNN
checkpoint — from the realistic SSC experiment
(my_project/code/realistic/ssc/data) — and evaluates the frozen model by
applying spike-timing perturbation to the *input* spike trains at test time,
across a sweep of perturbation levels f. No training is performed here.

This mirrors the original Beyond Rate input-perturbation experiments but
follows the train-once / eval-all protocol: a single clean-trained model is
evaluated at every f. Reusing the hidden-layer experiment's checkpoint means
the input- and hidden-perturbation results come from identical weights.
Supports the whole/part/norm dataset variants and SGD (no delay) / SGD-delay
modes.
"""

import os
import json

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Resolve all file paths relative to this script's location, not the CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# Global Configuration
# =====================================================================

# --- Eval-all switch ---
# When True, main() loops over every (delay x dataset-variant) combination.
# When False, main() runs the single config set by USE_DELAY / DATASET_KEY.
EVAL_ALL_VARIATION: bool = True

# Option lists used when EVAL_ALL_VARIATION is True.
DELAY_OPTIONS: list   = [True, False]
DATASET_VARIANTS: list = ["whole", "part", "norm"]

# --- Single-run flags (used when EVAL_ALL_VARIATION is False) ---
# Network variant: True for SGD-delay, False for SGD (no delay)
USE_DELAY: bool = True
# Dataset variant: "whole", "part", or "norm"
DATASET_KEY: str = "part"

# --- Dataset configurations ---
# whole: 700 input neurons (full SSC)
# part / norm: 285 input neurons (sub-sampled / rate-normalised)
DATASET_CONFIGS = {
    "whole": {"h5_file": "ssc_data/ssc_whole.h5", "input_dim": 700},
    "part":  {"h5_file": "ssc_data/ssc_part.h5",  "input_dim": 285},
    "norm":  {"h5_file": "ssc_data/ssc_norm.h5",  "input_dim": 285},
}

# --- Pretrained clean checkpoints (no perturbation) ---
# Loaded for evaluation instead of training a new model, so the input- and
# hidden-perturbation experiments share identical weights. Configs whose
# checkpoint is absent (e.g. "whole" if it has not been trained) are reported
# and skipped at run time.
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "../../../../code/realistic/ssc/data")
CHECKPOINT_TEMPLATE = "ssc_{dataset_key}_{delay_tag}_trained.pt"

# --- SLAYER neuron and simulation descriptors ---
# tSample=200 matches the original Beyond Rate training pipeline.
# SSC data has T=100; samples are zero-padded to T=200 on load.
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

# --- Data split ratios (applied to the combined h5 dataset) ---
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE   = (0.6, 0.75)
TEST_RANGE  = (0.75, 0.9)

# --- Network / evaluation hyper-parameters ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int  = 35   # SSC has 35 spoken-word classes
BATCH_SIZE: int   = 128
SEED: int         = 42
MAX_DELAY: int    = 64

# --- Input-perturbation sweep ---
F_VALUES: list   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_REPEATS: int = 3


# =====================================================================
# Dataset loading
# =====================================================================

def load_split_from_h5(
    h5_path: str,
    indices: np.ndarray,
    target_T: int = 200,
) -> tuple:
    with h5py.File(h5_path, "r") as hf:
        X = np.array(hf["X"][sorted(indices)], dtype=np.uint8)
        Y = np.array(hf["Y"][sorted(indices)]).astype(int).ravel()

    n, n_neurons, T = X.shape
    if T < target_T:
        padded = np.zeros((n, n_neurons, target_T), dtype=np.uint8)
        padded[:, :, :T] = X
        X = padded

    return X, Y


def probe_and_load_eval_data(
    h5_path: str,
    target_T: int = 200,
) -> tuple:
    with h5py.File(h5_path, "r") as hf:
        n_samples = hf["X"].shape[0]

    # Shuffle the sample order before splitting, using the same mechanism the
    # part/norm datasets are permuted with at generation time
    # (np.random.permutation). The whole dataset is stored in sorted
    # class-blocks, so without this the contiguous train/val/test ranges below
    # would see skewed, partially-missing classes. Seeded so the test split is
    # reproducible and matches the split the checkpoint was trained with.
    np.random.seed(SEED)
    shuffled_idx = np.random.permutation(n_samples)

    # Compute split indices over the shuffled order
    train_idx = shuffled_idx[
        int(n_samples * TRAIN_RANGE[0]):int(n_samples * TRAIN_RANGE[1])
    ]
    val_idx = shuffled_idx[
        int(n_samples * VAL_RANGE[0]):int(n_samples * VAL_RANGE[1])
    ]
    test_idx = shuffled_idx[
        int(n_samples * TEST_RANGE[0]):int(n_samples * TEST_RANGE[1])
    ]

    # Load val and test fully into memory (small enough)
    X_val, Y_val = load_split_from_h5(h5_path, val_idx, target_T)
    X_test, Y_test = load_split_from_h5(h5_path, test_idx, target_T)

    val_mem = X_val.nbytes / (1024 ** 3)
    test_mem = X_test.nbytes / (1024 ** 3)
    print(
        f"Dataset: {n_samples} samples total | "
        f"Train: {len(train_idx)} | "
        f"Val: {len(val_idx)} ({val_mem:.1f} GiB) | "
        f"Test: {len(test_idx)} ({test_mem:.1f} GiB)"
    )
    return n_samples, train_idx, (X_val, Y_val), (X_test, Y_test)


# =====================================================================
# Input-Layer Spike Perturbation
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
# Dataset and Data Splitting
# =====================================================================

class SpikeDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X  # uint8 to save memory
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx].astype(np.float32))
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


# =====================================================================
# Network Architecture
# =====================================================================

class SSCNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 128,
        num_classes: int = 35,
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

    # -----------------------------------------------------------------
    # Forward-pass building blocks
    # -----------------------------------------------------------------
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

    def get_delays(self) -> dict:
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
) -> SSCNetwork:
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

    net = SSCNetwork(
        input_dim, HIDDEN_UNITS, NUM_CLASSES, use_delay, MAX_DELAY
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


# =====================================================================
# Testing with Input-Layer Perturbation
# =====================================================================

def test_with_input_perturbation(
    net: SSCNetwork,
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
    net: SSCNetwork,
    test_loader: DataLoader,
    f_values: list,
    num_repeats: int = 3,
) -> dict:
    results = {}

    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            acc = test_with_input_perturbation(net, test_loader, f=f)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc  = np.std(accuracies)
        results[f] = {
            "mean": mean_acc, "std": std_acc, "values": accuracies
        }
        print(f"  f={f:.1f}:  accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

    return results


# =====================================================================
# Single-variation driver
# =====================================================================

def run_variation(use_delay: bool, dataset_key: str) -> dict:
    # --- Derived names / paths (local, so variations never collide) ---
    input_dim    = DATASET_CONFIGS[dataset_key]["input_dim"]
    h5_file      = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["h5_file"])
    delay_tag    = "delay" if use_delay else "nodelay"
    model_prefix = f"ssc_{dataset_key}_{delay_tag}"

    log_dir = os.path.join(SCRIPT_DIR, "log")

    print(f"\n{'=' * 70}")
    print(
        f"Variation: {model_prefix} | Input dim: {input_dim} | "
        f"Mode: {'SGD-delay' if use_delay else 'SGD (no delay)'}"
    )
    print(f"{'=' * 70}")

    # --- Load the pretrained clean checkpoint (no training performed; fail
    # fast before loading data if the checkpoint is missing). ---
    checkpoint_path = os.path.join(
        PRETRAINED_DIR,
        CHECKPOINT_TEMPLATE.format(dataset_key=dataset_key, delay_tag=delay_tag),
    )
    print(f"Loading clean checkpoint: {checkpoint_path}")
    net = load_pretrained_model(checkpoint_path, input_dim, use_delay)

    # --- Load eval data + probe split indices (only the test split is used) ---
    _, _, _, test_data = probe_and_load_eval_data(
        h5_file, target_T=SIM_PARAMS["tSample"]
    )

    # --- Build test loader from pre-loaded test data ---
    test_loader = DataLoader(
        SpikeDataset(test_data[0], test_data[1]),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # Sanity check: accuracy on clean test set (f=0)
    clean_acc = test_with_input_perturbation(net, test_loader, f=0.0)
    print(f"\nClean test accuracy (f=0): {clean_acc:.4f}")

    # --- Input-perturbation sweep ---
    print(
        f"=== Input-Layer Perturbation Sweep "
        f"(SSC {dataset_key}, {delay_tag}) ==="
    )
    sweep_results = run_input_perturbation_sweep(
        net, test_loader, f_values=F_VALUES, num_repeats=NUM_REPEATS
    )

    # --- Save results to JSON ---
    os.makedirs(log_dir, exist_ok=True)

    results_serialisable = {
        str(f_val): {
            "mean":   float(data["mean"]),
            "std":    float(data["std"]),
            "values": [float(v) for v in data["values"]],
        }
        for f_val, data in sweep_results.items()
    }
    results_path = os.path.join(
        log_dir, f"{model_prefix}_input_perturbation_results.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")

    return sweep_results


# =====================================================================
# Dispatcher
# =====================================================================

def main() -> None:
    if EVAL_ALL_VARIATION:
        configs = [
            (use_delay, dataset_key)
            for use_delay in DELAY_OPTIONS
            for dataset_key in DATASET_VARIANTS
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
