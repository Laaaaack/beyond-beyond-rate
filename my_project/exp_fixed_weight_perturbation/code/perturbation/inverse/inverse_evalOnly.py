"""First-Hidden-Layer Time Reversal — SHD (eval-only, train-per-f / eval-per-f).

Reproduces the Beyond Rate reversal protocol at the **output of the 1st hidden
layer**. For every perturbation level f a *separate* 2-hidden-layer SLAYER SNN
checkpoint — one trained with 1st-hidden-layer perturbation at that same f (see
my_project/code/realistic/shd/shd_train.py) — is loaded and evaluated at that f.
Training the model at each f lets it adapt its code to the perturbation level
(relying more on rate as timing is destroyed), so the reversal curve recovers
the paper's rate-level plateau instead of collapsing to chance. The per-f
checkpoints live next to this script in ``data/``. No training is performed here.

The reversal and perturbation are applied to the spikes at the 1st hidden
layer's output (after delay1) — exactly where shd_train.py injects its
perturbation — so the loaded weights and the perturbation site match training.

Two conditions are swept for every f (each using the f-matched checkpoint):
  - no_reversal: 1st-hidden-layer spike-timing perturbation only (fraction f).
  - reversal:    1st-hidden-layer time reversal, then the same f perturbation.

Time reversal preserves spike counts, neuron identities, per-neuron ISIs, and
coincidence patterns, but flips temporal/causal order.

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

# --- Per-perturbation-level checkpoints ---
# One checkpoint per (dataset, delay, f): each was trained on data perturbed at
# that f. Evaluating the f-matched checkpoint at level f reproduces the Beyond
# Rate protocol, where the model adapts to each perturbation level. The files
# live in the local ``data/`` directory next to this script.
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "data")
CHECKPOINT_TEMPLATE = "shd_{dataset_key}_{delay_tag}_f{f:.1f}.pt"

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

# When False, skip the no-reversal baseline and report only reversal results.
RUN_NO_REVERSAL: bool = True


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


def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float,
) -> torch.Tensor:
    """Relocate a fraction *f* of spikes at a hidden layer's output.

    Mirrors the perturbation injected during training (see ``shd_train.py`` /
    ``shd_2ndLayer_train.py``): operates on the SLAYER hidden tensor of shape
    (B, C, 1, 1, T), perturbing each sample's (C, T) spike map independently
    while preserving per-channel spike counts.

    Args:
        hidden_spikes: Hidden spike tensor of shape (B, C, 1, 1, T).
        f: Perturbation fraction.

    Returns:
        Perturbed hidden spike tensor of the same shape, on the input device.
    """
    dev = hidden_spikes.device
    spikes_np = hidden_spikes.detach().cpu().numpy()
    batch_size = spikes_np.shape[0]

    for sample_idx in range(batch_size):
        sample = spikes_np[sample_idx, :, 0, 0, :]  # (C, T)
        spikes_np[sample_idx, :, 0, 0, :] = partial_randomize_spike_train(sample, f)

    return torch.from_numpy(spikes_np).to(dev)


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


def reverse_hidden_batch(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """Time-reverse each sample's hidden spike map within its active window.

    Operates on the SLAYER hidden tensor of shape (B, C, 1, 1, T); each
    sample's (C, T) map is reversed across all channels within the shared
    active window (see ``reverse_input_spike_train``).

    Args:
        hidden_spikes: Hidden spike tensor of shape (B, C, 1, 1, T).

    Returns:
        Time-reversed hidden spike tensor of the same shape, on the input device.
    """
    dev = hidden_spikes.device
    spikes_np = hidden_spikes.detach().cpu().numpy()
    batch_size = spikes_np.shape[0]

    for sample_idx in range(batch_size):
        sample = spikes_np[sample_idx, :, 0, 0, :]  # (C, T)
        spikes_np[sample_idx, :, 0, 0, :] = reverse_input_spike_train(sample)

    return torch.from_numpy(spikes_np).to(dev)


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

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # 1st hidden layer: PSP -> fc1 -> spike -> (delay1). This layer's
        # output is the perturbation site for this experiment, matching
        # shd_train.py.
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        return x

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        # Remaining forward: PSP -> fc2 -> spike -> (delay2) -> PSP -> fc3 -> spike.
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_perturbation(
        self,
        x: torch.Tensor,
        f: float = 0.0,
        reverse: bool = False,
    ) -> torch.Tensor:
        # Apply reversal and/or perturbation at the 1st hidden layer's output
        # (after delay1) — the same site shd_train.py perturbs during training.
        # Reversal is applied before perturbation (the two do not commute).
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if reverse:
            hidden1 = reverse_hidden_batch(hidden1)
        if f > 0:
            hidden1 = perturb_hidden_batch(hidden1, f)
        return self._second_hidden_and_output(hidden1)

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
    """Load a pretrained checkpoint into a fresh network for evaluation.

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
def test_with_reversal(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
    reverse: bool = True,
) -> float:
    # Eval-only: reverse and/or perturb at the 1st hidden layer's output, then
    # finish the forward. The numpy round-trip is not autograd-safe, so this
    # must stay inside torch.no_grad().
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_batch = y_batch.to(device)

            outputs = net.forward_with_perturbation(x_batch, f=f, reverse=reverse)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_perturbation(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
) -> float:
    # Eval-only: perturb at the 1st hidden layer's output (no reversal), then
    # finish the forward. Stays inside torch.no_grad() for the numpy round-trip.
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_batch = y_batch.to(device)

            outputs = net.forward_with_perturbation(x_batch, f=f, reverse=False)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def run_reversal_sweep(
    get_net,
    test_loader: DataLoader,
    f_values: list[float],
    num_repeats: int = NUM_REPEATS,
) -> dict[str, dict[float, dict]]:
    """Sweep over perturbation levels with and without reversal.

    For each f the f-matched checkpoint is loaded once via ``get_net(f)`` and
    evaluated under both conditions (no reversal / reversal). Each condition is
    repeated num_repeats times with different random seeds for error bars.

    Args:
        get_net: Callable mapping a perturbation level f to the loaded
            SHDNetwork that was trained at that f.
        test_loader: Test DataLoader.
        f_values: List of perturbation fractions to evaluate.
        num_repeats: Number of independent evaluations per setting.

    Returns:
        Dict with keys 'no_reversal' and 'reversal', each mapping f values to
        {"mean", "std", "values"}.
    """
    results: dict[str, dict[float, dict]] = {"no_reversal": {}, "reversal": {}}

    for f in f_values:
        # Load the checkpoint trained at this perturbation level (once per f).
        net = get_net(f)

        # --- No reversal (baseline input perturbation) ---
        if RUN_NO_REVERSAL:
            nr_acc = []
            for repeat in range(num_repeats):
                np.random.seed(SEED + repeat)
                nr_acc.append(test_with_perturbation(net, test_loader, f=f))
            results["no_reversal"][f] = {
                "mean": float(np.mean(nr_acc)),
                "std": float(np.std(nr_acc)),
                "values": nr_acc,
            }

        # --- With input-layer time reversal ---
        rev_acc = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            rev_acc.append(
                test_with_reversal(net, test_loader, f=f, reverse=True)
            )
        results["reversal"][f] = {
            "mean": float(np.mean(rev_acc)),
            "std": float(np.std(rev_acc)),
            "values": rev_acc,
        }

        if RUN_NO_REVERSAL:
            print(
                f"    f={f:.1f} | "
                f"no_reversal = {results['no_reversal'][f]['mean']:.4f} "
                f"+/- {results['no_reversal'][f]['std']:.4f} | "
                f"reversal = {results['reversal'][f]['mean']:.4f} "
                f"+/- {results['reversal'][f]['std']:.4f}"
            )
        else:
            print(
                f"    f={f:.1f} | "
                f"reversal = {results['reversal'][f]['mean']:.4f} "
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

    # Per-f checkpoint loader: each f uses the checkpoint trained at that f.
    def get_net(f: float) -> SHDNetwork:
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            CHECKPOINT_TEMPLATE.format(
                dataset_key=dataset_key, delay_tag=delay_tag, f=f
            ),
        )
        print(
            f"  Loading f={f:.1f} checkpoint: "
            f"{os.path.basename(checkpoint_path)}"
        )
        return load_pretrained_model(checkpoint_path, input_dim, use_delay)

    # Evaluate across the 1st-hidden-layer reversal sweep (no reversal vs
    # reversal), loading the f-matched checkpoint for each perturbation level.
    print(f"\n  --- 1st-hidden-layer reversal sweep (per-f checkpoints) ---")
    sweep_results = run_reversal_sweep(get_net, test_loader, F_VALUES, NUM_REPEATS)

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
        log_dir, f"{model_prefix}_hidden1_reversal_sweep_results.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"  Sweep results saved to {results_path}")

    return {
        "sweep_results": sweep_results,
        "dataset_key": dataset_key,
        "use_delay": use_delay,
        "delay_tag": delay_tag,
        "model_prefix": model_prefix,
        "checkpoint_dir": CHECKPOINT_DIR,
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
