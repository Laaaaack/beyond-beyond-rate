"""Experiment 2B: SSC — 2nd-Hidden-Layer Perturbation (evaluation only).

Load pre-trained 2-hidden-layer SLAYER SNN checkpoints from `data/` and sweep
spike-timing perturbation applied to the output of the 2nd hidden layer at test
time. No training is performed. Supports the whole/part/norm dataset variants
and SGD (no delay) / SGD-delay modes.

The network definition and perturbation hook mirror ssc_2ndLayer_train.py, so
the checkpoints that script produces load without modification. Note the
difference from ssc_evalOnly.py: delay1 is folded into the 1st hidden layer and
delay2 into the output routing, so the perturbation hook sees strictly binary
2nd-hidden-layer spikes.
"""

import json
import os

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
DELAY_OPTIONS: list    = [True, False]
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
# Must match the training script, or the "test" set would overlap the data the
# checkpoints were fitted on.
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE   = (0.6, 0.75)
TEST_RANGE  = (0.75, 0.9)

# --- Model hyper-parameters (must match the checkpoints) ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int  = 35   # SSC has 35 spoken-word classes
SEED: int         = 42
MAX_DELAY: int    = 64

# --- Hidden-perturbation sweep ---
# The low-f points matter: the no-delay arm's accuracy knee sits near f = 0.05-0.2,
# so the original 0.2-spaced grid rendered a steep-but-smooth decline as a cliff.
# The original six values are kept as a subset so old curves stay comparable.
F_VALUES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_REPEATS: int = 3

# Destinations for relocated spikes are confined to [0, SUPPORT_BINS).
#
# SSC holds 100 time bins and load_split_from_h5 zero-pads them to tSample=200, so
# the hidden layer never fires in the upper half of the window. Relocating across
# all 200 bins therefore drops drive from the live region and injects it into a
# region the next layer has never been driven in, which is a *rate* insult riding
# on a probe whose whole purpose is to destroy timing at fixed rate.
#
#   "auto" - measure the clean layer's own support, per batch (recommended)
#   int    - pin the window explicitly
#   None   - reproduce the uncorrected full-window behaviour
#
# "auto" rather than a constant is essential at this layer: delay1 is applied
# upstream of the probe, so measured over these checkpoints the 2nd hidden layer
# runs to bin 148 in the delay arm but stops at 80 in the no-delay arm. Pinning a
# single constant would re-create the very dead zone this setting exists to remove.
SUPPORT_BINS: int | str | None = "auto"

# --- Checkpoint naming ---
# Checkpoints in data/ are stored as "<model_prefix><CHECKPOINT_SUFFIX>".
CHECKPOINT_SUFFIX: str = "_f0.0.pt"

# --- Evaluation batch size ---
BATCH_SIZE: int = 128


# =====================================================================
# Dataset loading
# =====================================================================

def load_split_from_h5(
    h5_path: str,
    indices: np.ndarray,
    target_T: int = 200,
) -> tuple:
    """Load a subset of samples from an SSC h5 file, zero-padded in time.

    Args:
        h5_path: Path to the h5 file holding "X" (spikes) and "Y" (labels).
        indices: Sample indices to read.
        target_T: Number of timesteps to pad each sample out to.

    Returns:
        A tuple of (X, Y), where X has shape (n, n_neurons, target_T) as uint8
        and Y has shape (n,) as int.
    """
    with h5py.File(h5_path, "r") as hf:
        X = np.array(hf["X"][sorted(indices)], dtype=np.uint8)
        Y = np.array(hf["Y"][sorted(indices)]).astype(int).ravel()

    n, n_neurons, T = X.shape
    if T < target_T:
        padded = np.zeros((n, n_neurons, target_T), dtype=np.uint8)
        padded[:, :, :T] = X
        X = padded

    return X, Y


def load_test_data(h5_path: str, target_T: int = 200) -> tuple:
    """Load the test split of an SSC h5 file into memory.

    The sample order is shuffled with a seeded permutation before the
    contiguous TEST_RANGE slice is taken, reproducing the split used at
    training time. The `whole` dataset is stored in sorted class-blocks, so
    without this the contiguous slice would see skewed, partially-missing
    classes.

    Args:
        h5_path: Path to the h5 file.
        target_T: Number of timesteps to pad each sample out to.

    Returns:
        A tuple of (X_test, Y_test).
    """
    with h5py.File(h5_path, "r") as hf:
        n_samples = hf["X"].shape[0]

    np.random.seed(SEED)
    shuffled_idx = np.random.permutation(n_samples)

    test_idx = shuffled_idx[
        int(n_samples * TEST_RANGE[0]):int(n_samples * TEST_RANGE[1])
    ]

    X_test, Y_test = load_split_from_h5(h5_path, test_idx, target_T)

    test_mem = X_test.nbytes / (1024 ** 3)
    print(
        f"Dataset: {n_samples} samples total | "
        f"Test: {len(test_idx)} ({test_mem:.1f} GiB)"
    )
    return X_test, Y_test


# =====================================================================
# Hidden-Layer Spike Perturbation
# =====================================================================

def resolve_support_window(is_spike: torch.Tensor, num_bins: int) -> int:
    """Return the exclusive upper bin bound that relocated spikes may occupy.

    Args:
        is_spike: Boolean tensor of shape (B, C, T) marking the clean layer's spikes.
        num_bins: Total number of simulation bins, T.

    Returns:
        The window size, honouring the SUPPORT_BINS setting. For "auto" this is one
        past the last bin occupied anywhere in the batch, which self-calibrates to
        whichever arm and layer is being probed.
    """
    if SUPPORT_BINS is None:
        return num_bins
    if SUPPORT_BINS != "auto":
        return int(SUPPORT_BINS)

    occupied = is_spike.any(dim=0).any(dim=0).nonzero()
    if occupied.numel() == 0:
        return num_bins
    return int(occupied.max()) + 1


@torch.no_grad()
def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float = 0.0,
) -> torch.Tensor:
    """Vectorised GPU-side partial spike relocation.

    For each (batch, neuron), a fraction *f* of the existing spikes are removed and
    replaced with the same number of spikes placed at randomly chosen
    previously-unoccupied time bins. Spike count per neuron is preserved exactly.

    Destinations are confined to the layer's temporal support (see SUPPORT_BINS),
    so relocation cannot thin the population's spike density by scattering spikes
    into the zero-padded tail. Per-neuron spike count is preserved either way: the
    support always has room, since every spike being moved came out of it.

    This replaces an earlier per-spike numpy loop that drew its relocation count
    from Bernoulli(f) on the CPU. The count is now a deterministic floor(n * f),
    matching the SHD scripts exactly so the two datasets share one estimator, and
    the whole draw stays on-device. Note the draw is now a torch RNG consumer, so
    the sweep must seed torch as well as numpy.

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
    window = resolve_support_window(is_spike, T)

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
    available[:, :, window:] = False  # stay inside the layer's temporal support
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move  # disjoint from keep_mask by construction

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype)
    return new_spikes.view(B, C, H, W, T)


# =====================================================================
# Dataset
# =====================================================================

class SpikeDataset(Dataset):
    """Wraps uint8 spike arrays as a torch Dataset, casting to float on access."""

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
    """2-hidden-layer SLAYER SNN with an optional learnable-delay variant.

    Structurally identical to the network in ssc_2ndLayer_train.py so that its
    checkpoints load with strict key matching.
    """

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

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        return x

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        # Compute block for the 2nd hidden layer. Returns BINARY spikes.
        # delay2 is NOT applied here — it belongs in _output's routing step,
        # so the perturbation hook downstream operates on a strictly 0/1 tensor.
        return self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        # Routing (delay2) + output layer.
        x = self.delay2(hidden2) if self.use_delay else hidden2
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

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

    def get_delays(self) -> dict:
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
) -> SSCNetwork:
    """Build an SSCNetwork and restore its weights from a checkpoint.

    Args:
        checkpoint_path: Path to a state_dict saved by ssc_2ndLayer_train.py.
        input_dim: Number of input neurons for the dataset variant.
        use_delay: Whether the checkpoint is a delay-enabled variant.

    Returns:
        The network in eval mode on `device`.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = SSCNetwork(
        input_dim, HIDDEN_UNITS, NUM_CLASSES, use_delay, MAX_DELAY
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return net


# =====================================================================
# Testing with Hidden-Layer Perturbation
# =====================================================================

def test_with_hidden_perturbation(
    net: SSCNetwork,
    test_loader: DataLoader,
    f: float = 0.0,
) -> float:
    """Measure test accuracy with the 2nd hidden layer perturbed.

    Args:
        net: A loaded SSCNetwork.
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
    net: SSCNetwork,
    test_loader: DataLoader,
    f_values: list,
    num_repeats: int = 3,
) -> dict:
    """Sweep perturbation strength, repeating each level with fresh RNG seeds.

    Args:
        net: A loaded SSCNetwork.
        test_loader: Loader over the test split.
        f_values: Perturbation probabilities to evaluate.
        num_repeats: Number of seeded repeats per f value.

    Returns:
        A dict mapping each f value to {"mean", "std", "values"}.
    """
    results = {}

    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            # perturb_hidden_batch draws from the torch RNG, so seeding numpy
            # alone would leave the repeats identical rather than independent.
            np.random.seed(SEED + repeat)
            torch.manual_seed(SEED + repeat)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED + repeat)
            acc = test_with_hidden_perturbation(net, test_loader, f=f)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc  = np.std(accuracies)
        results[f] = {
            "mean": mean_acc, "std": std_acc, "values": accuracies
        }
        print(f"  f={f:.2f}:  accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

    return results


# =====================================================================
# Single-variation driver
# =====================================================================

def run_variation(use_delay: bool, dataset_key: str) -> dict:
    """Evaluate one (delay x dataset) checkpoint across the perturbation sweep.

    Args:
        use_delay: True for the SGD-delay variant, False for plain SGD.
        dataset_key: One of "whole", "part", or "norm".

    Returns:
        The sweep results, as returned by run_hidden_perturbation_sweep().
    """
    input_dim    = DATASET_CONFIGS[dataset_key]["input_dim"]
    h5_file      = os.path.join(SCRIPT_DIR, DATASET_CONFIGS[dataset_key]["h5_file"])
    delay_tag    = "delay" if use_delay else "nodelay"
    model_prefix = f"ssc_2ndLayer_{dataset_key}_{delay_tag}"

    data_dir = os.path.join(SCRIPT_DIR, "data")
    log_dir  = os.path.join(SCRIPT_DIR, "log")
    checkpoint_path = os.path.join(
        data_dir, f"{model_prefix}{CHECKPOINT_SUFFIX}"
    )

    print(f"\n{'=' * 70}")
    print(
        f"Variation: {model_prefix} | Input dim: {input_dim} | "
        f"Mode: {'SGD-delay' if use_delay else 'SGD (no delay)'}"
    )
    print(f"{'=' * 70}")

    # --- Load the trained model ---
    net = load_trained_model(checkpoint_path, input_dim, use_delay)

    delays = net.get_delays()
    if delays:
        avg_delay = np.mean([np.mean(d) for d in delays.values() if len(d) > 0])
        print(f"Mean learned delay: {avg_delay:.1f}")

    # --- Load test data ---
    X_test, Y_test = load_test_data(h5_file, target_T=SIM_PARAMS["tSample"])
    test_loader = DataLoader(
        SpikeDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    # --- Hidden-perturbation sweep ---
    print(
        f"=== 2nd-Hidden-Layer Perturbation Sweep "
        f"(SSC {dataset_key}, {delay_tag}) ==="
    )
    sweep_results = run_hidden_perturbation_sweep(
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
        log_dir, f"{model_prefix}_hidden_perturbation_results.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")

    return sweep_results


# =====================================================================
# Dispatcher
# =====================================================================

def main() -> None:
    """Evaluate either every variation or the single configured one."""
    if EVAL_ALL_VARIATION:
        for use_delay in DELAY_OPTIONS:
            for dataset_key in DATASET_VARIANTS:
                run_variation(use_delay, dataset_key)
    else:
        run_variation(USE_DELAY, DATASET_KEY)


if __name__ == "__main__":
    main()
