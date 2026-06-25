"""Input Time Reversal — SHD (eval-only, train-once / eval-all).

Loads a pretrained clean (no-perturbation) 2-hidden-layer SLAYER SNN
checkpoint — the SHD classifier produced by the realistic SHD experiment
(my_project/code/realistic/shd/data) — and, at evaluation only, time-reverses
each sample's *input* spike train within its active window, optionally combined
with per-spike relocation (fraction f). No training is performed here.

Two conditions are swept for every f:
  - no_reversal: input spike-timing perturbation only (fraction f relocated).
  - reversal:    the same f perturbation, then input time reversal.

Time reversal preserves spike counts, neuron identities, per-neuron ISIs, and
coincidence patterns, but flips temporal/causal order. Applying it at the input
(rather than the 1st hidden layer) and reusing the same clean checkpoint that
the hidden-layer reversal experiment evaluates ensures both experiments probe
identical weights — a fair comparison between reversing the stimulus and
reversing the internal representation.

Architecture: Input(input_dim) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA)
Sweep (eval only): f in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, reversal On/Off

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
# SHD classifiers trained by the realistic SHD experiment, the same models the
# hidden-layer reversal experiment evaluates, so the input- and hidden-site
# reversal experiments share identical weights.
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "../../../../code/realistic/shd/data")
CHECKPOINT_TEMPLATE = "shd_{dataset_key}_{delay_tag}_trained.pt"

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

# --- Reversal sweep: input perturbation fraction f ---
F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

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
#  Time reversal and perturbation utilities (core perturbation)
# ===================================================================== #
def partial_randomize_spike_train(
    spike_train: np.ndarray,
    f: float = 0.0,
    max_attempts: int = 50,
) -> np.ndarray:
    """Randomly relocate a fraction *f* of each neuron's spikes.

    For every neuron, *f* of its spikes are removed from their original
    positions and placed at uniformly random empty time bins. The total
    spike count per neuron is preserved.

    Args:
        spike_train: Binary array of shape (num_neurons, T).
        f: Fraction of spikes to relocate (0 = no change, 1 = full shuffle).
        max_attempts: Max tries to find an empty time bin per spike.

    Returns:
        Perturbed spike train with same shape and spike counts.
    """
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


def reverse_input_spike_train(spike_train: np.ndarray) -> np.ndarray:
    """Time-reverse a sample's input spike trains within its active window.

    The active window is defined as ``[t_start, t_end]`` where ``t_start``
    is the earliest spike across all neurons and ``t_end`` is the latest.
    Every neuron's segment in this window is flipped using the same
    transform ``t -> t_start + t_end - t``. This preserves spike count,
    per-neuron rate, per-neuron ISI distribution, **and cross-neuron
    coincidence patterns** (coincident spikes remain coincident, just
    mirrored in time). Only the temporal/causal direction is reversed.

    Args:
        spike_train: Binary array of shape (num_neurons, T).

    Returns:
        Time-reversed spike train with same shape.
    """
    reversed_train = np.copy(spike_train)
    spike_positions = np.where(spike_train == 1)
    if len(spike_positions[1]) < 2:
        return reversed_train

    t_start = int(spike_positions[1].min())
    t_end = int(spike_positions[1].max())
    segment = spike_train[:, t_start:t_end + 1]
    reversed_train[:, t_start:t_end + 1] = np.flip(segment, axis=1)

    return reversed_train


def reverse_input_batch(input_spikes: torch.Tensor) -> torch.Tensor:
    """Apply time reversal to a batch of input spike trains.

    Args:
        input_spikes: Input spike tensor of shape (B, num_neurons, T).

    Returns:
        Time-reversed input spike tensor of shape (B, num_neurons, T) on CPU.
    """
    spikes_np = input_spikes.detach().cpu().numpy()
    batch_size = spikes_np.shape[0]
    reversed_arr = np.zeros_like(spikes_np)

    for sample_idx in range(batch_size):
        reversed_arr[sample_idx] = reverse_input_spike_train(spikes_np[sample_idx])

    return torch.from_numpy(reversed_arr).float()


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
        # Input perturbation / reversal, when applied, perturbs the spike trains
        # before they reach this method, so the loaded checkpoint stays clean.
        x = self._prepare_input(x)
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        x = self.slayer.spike(self.fc2(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

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


# ===================================================================== #
#  Evaluation / input reversal sweep
# ===================================================================== #
def test_with_input_reversal(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
    reverse: bool = True,
) -> float:
    # Eval-only: perturb (and optionally reverse) the input spike trains, then
    # run the clean forward. The numpy round-trip is not autograd-safe, so this
    # must stay inside torch.no_grad().
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if f > 0:
                x_batch = perturb_input_batch(x_batch, f)
            if reverse:
                x_batch = reverse_input_batch(x_batch)
            y_batch = y_batch.to(device)

            outputs = net(x_batch)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_input_perturbation(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
) -> float:
    # Eval-only: perturb the input spike trains (no reversal), then run the
    # clean forward. Stays inside torch.no_grad() for the numpy round-trip.
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


def run_reversal_sweep(
    net: SHDNetwork,
    test_loader: DataLoader,
    f_values: list[float],
    num_repeats: int = NUM_REPEATS,
) -> dict[str, dict[float, dict]]:
    """Sweep over perturbation levels with and without input reversal.

    For each f value and each condition (no reversal / reversal), evaluation is
    repeated num_repeats times with different random seeds to obtain error bars.

    Args:
        net: Loaded SHDNetwork.
        test_loader: Test DataLoader.
        f_values: List of perturbation fractions to evaluate.
        num_repeats: Number of independent evaluations per setting.

    Returns:
        Dict with keys 'no_reversal' and 'reversal', each mapping f values to
        {"mean", "std", "values"}.
    """
    results: dict[str, dict[float, dict]] = {"no_reversal": {}, "reversal": {}}

    # --- No reversal (baseline input perturbation) ---
    print("  --- No reversal (baseline input perturbation) ---")
    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            acc = test_with_input_perturbation(net, test_loader, f=f)
            accuracies.append(acc)

        results["no_reversal"][f] = {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "values": accuracies,
        }
        print(
            f"    f={f:.1f} | "
            f"accuracy = {results['no_reversal'][f]['mean']:.4f} "
            f"+/- {results['no_reversal'][f]['std']:.4f}"
        )

    # --- With input-layer time reversal ---
    print("  --- With input-layer time reversal ---")
    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            acc = test_with_input_reversal(net, test_loader, f=f, reverse=True)
            accuracies.append(acc)

        results["reversal"][f] = {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "values": accuracies,
        }
        print(
            f"    f={f:.1f} | "
            f"accuracy = {results['reversal'][f]['mean']:.4f} "
            f"+/- {results['reversal'][f]['std']:.4f}"
        )

    return results


# ===================================================================== #
#  Single-variation runner: load checkpoint, sweep reversal, save results
# ===================================================================== #
def run_variation(dataset_key: str, use_delay: bool) -> dict:
    # Derive names and output paths locally so configurations never
    # overwrite each other's files.
    input_dim = DATASET_CONFIGS[dataset_key]["input_dim"]
    mat_file = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["mat_file"])
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"shd_{dataset_key}_{delay_tag}"

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

    # Evaluate across the input-reversal sweep (no reversal vs reversal)
    print(f"\n  --- Input-reversal sweep at evaluation ---")
    sweep_results = run_reversal_sweep(net, test_loader, F_VALUES, NUM_REPEATS)

    # Save sweep results
    results_serialisable = {
        condition: {
            str(f_val): {
                "mean": data["mean"],
                "std": data["std"],
                "values": [float(v) for v in data["values"]],
            }
            for f_val, data in cond_results.items()
        }
        for condition, cond_results in sweep_results.items()
    }
    results_path = os.path.join(
        log_dir, f"{model_prefix}_input_reversal_sweep_results.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"  Sweep results saved to {results_path}")

    return {
        "net": net,
        "sweep_results": sweep_results,
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
