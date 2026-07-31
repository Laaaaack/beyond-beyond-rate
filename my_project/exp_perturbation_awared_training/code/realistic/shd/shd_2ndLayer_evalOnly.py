"""Experiment 2A: SHD — Eval-only Sweep (2nd Hidden Layer Perturbation).

For each perturbation level *f* the 2-hidden-layer SNN checkpoint that was
trained at that *f* (with 2nd hidden-layer perturbation, applied after the
fc2 spike) is loaded from ``data/`` and evaluated at the same *f*. No
training is performed.

This is the evaluation-only counterpart of the 2nd-layer train-at-f /
eval-at-f protocol: it reuses the persisted checkpoints and re-runs the
perturbed test sweep for every requested dataset/delay variation.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "log"


# =====================================================================
# Global Configuration
# =====================================================================
TRAIN_ALL_VARIATION: bool = False
USE_DELAY: bool = True
DATASET_KEY: str = "whole"

ALL_VARIATIONS: list[tuple[str, bool]] = [
    (dataset, delay)
    #for dataset in ("norm", "part", "whole")
    for dataset in ("whole",)
    for delay in (False, True)
]

DATASET_CONFIGS = {
    "whole": {"mat_file": str(SCRIPT_DIR / "shd_data/shd_whole.mat"), "input_dim": 700},
    "part":  {"mat_file": str(SCRIPT_DIR / "shd_data/shd_part_new.mat"), "input_dim": 224},
    "norm":  {"mat_file": str(SCRIPT_DIR / "shd_data/shd_norm_new.mat"), "input_dim": 224},
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

# Destinations for relocated spikes are confined to [0, SUPPORT_BINS).
#
# The .mat holds 100 time bins and load_shd_data zero-pads them to tSample=200, so the
# hidden layer never fires in the upper half of the window. Relocating across all
# 200 bins therefore drops drive from the live region and injects it into a region
# the next layer has never been driven in, which is a *rate* insult riding on a
# probe whose whole purpose is to destroy timing at fixed rate. Measured on the
# 1st hidden layer, ~57% of relocated spikes landed where no hidden spike ever
# naturally occurs.
#
#   "auto" - measure the clean layer's own support, per batch (recommended)
#   int    - pin the window explicitly
#   None   - reproduce the uncorrected full-window behaviour
#
# "auto" rather than a constant is essential at this layer: delay1 is applied
# upstream of the probe, so measured over the fixed-weight checkpoints the 2nd
# hidden layer runs to bin 158 in the delay arm but stops at 89 in the no-delay
# arm. Pinning one constant would re-create the very dead zone this removes.
#
# NOTE: unlike the fixed-weight experiment, perturbation here runs inside the
# *training* loop, so this is not an eval-only knob. Checkpoints in data/ predate
# this fix and were trained under the full-window behaviour; they must be
# retrained before their curves can be read against a corrected sweep. Set None to
# reproduce those checkpoints' regime exactly.
SUPPORT_BINS: int | str | None = "auto"
NUM_REPEATS: int = 3


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

    For each (batch, neuron), a fraction *f* of the existing spikes are
    removed and replaced with the same number of spikes placed at randomly
    chosen previously-unoccupied time bins. Spike count per neuron is
    preserved exactly. All operations stay on the input tensor's device.

    Destinations are confined to the layer's temporal support (see SUPPORT_BINS),
    so relocation cannot thin the population's spike density by scattering spikes
    into the zero-padded tail. Per-neuron spike count is preserved either way: the
    support always has room, since every spike being moved came out of it.
    """
    if f <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5
    window = resolve_support_window(is_spike, T)

    n_spikes = is_spike.sum(dim=-1, keepdim=True)
    num_to_move = (n_spikes.float() * f).floor().long()

    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move

    keep_mask = is_spike & ~remove_mask

    available = ~keep_mask
    available[:, :, window:] = False  # stay inside the layer's temporal support
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype)
    return new_spikes.view(B, C, H, W, T)


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


class SHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with perturbation on 2nd hidden layer output.

    Perturbation is applied after the spike output of fc2 (2nd hidden layer),
    before delay2. Forward pass applies spike-relocation perturbation through
    a straight-through estimator, keeping gradient flow to fc1 and fc2 intact.
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
        """Input -> PSP -> fc1 -> spike -> hidden1 spikes."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> (delay1) -> PSP -> fc2 -> spike -> hidden2 spikes."""
        x = hidden1
        if self.use_delay:
            x = self.delay1(x)
        return self.slayer.spike(self.fc2(self.slayer.psp(x)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> (delay2) -> PSP -> fc3 -> spike -> output."""
        x = hidden2
        if self.use_delay:
            x = self.delay2(x)
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

    def _apply_perturbation(
        self,
        hidden: torch.Tensor,
        f: float,
    ) -> torch.Tensor:
        """STE wrapper around ``perturb_hidden_batch``."""
        if f <= 0:
            return hidden
        perturbed = perturb_hidden_batch(hidden, f)
        return hidden + (perturbed - hidden).detach()

    def forward(self, x: torch.Tensor, f: float = 0.0) -> torch.Tensor:
        """Forward pass with 2nd hidden layer perturbation."""
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)
        hidden2 = self._apply_perturbation(hidden2, f)
        return self._output(hidden2)

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


def load_model(
    checkpoint_path: Path,
    input_dim: int,
    use_delay: bool,
    hidden_units: int = 128,
    num_classes: int = 20,
    max_delay: int = 64,
) -> SHDNetwork:
    """Instantiate an SHDNetwork and load weights from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` state-dict file.
        input_dim: Number of input neurons.
        use_delay: Whether the checkpoint was trained with learnable delays.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        max_delay: Maximum delay in time steps.

    Returns:
        The network in eval mode with the checkpoint weights loaded.
    """
    net = SHDNetwork(
        input_dim, hidden_units, num_classes, use_delay, max_delay
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


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

            outputs = net(x_batch, f=f)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: SHDNetwork,
    test_loader: DataLoader,
    f: float,
    num_repeats: int = 3,
) -> dict:
    accuracies: list[float] = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        torch.manual_seed(SEED + repeat)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + repeat)
        accuracies.append(test_with_hidden_perturbation(net, test_loader, f=f))
    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": [float(a) for a in accuracies],
    }


def run_variation_sweep(
    dataset_key: str,
    use_delay: bool,
) -> dict:
    """Eval-only sweep for 2nd hidden layer perturbation.

    Loads the dataset, builds dataloaders, then for each f in F_VALUES
    loads the pre-trained checkpoint (trained at that f) from ``data/`` and
    evaluates it at the same f. No training is performed.
    """
    cfg = DATASET_CONFIGS[dataset_key]
    input_dim = cfg["input_dim"]
    mat_file = cfg["mat_file"]
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"shd_2ndLayer_{dataset_key}_{delay_tag}"

    print(f"\n{'#' * 70}")
    print(f"# 2nd Hidden Layer Perturbation: dataset={dataset_key} | delay={delay_tag}")
    print(f"# Model prefix: {model_prefix}")
    print(f"{'#' * 70}")

    X, Y = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    _, _, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=SEED,
    )

    models: dict[float, SHDNetwork] = {}
    results: dict[float, dict] = {}

    for f_val in F_VALUES:
        model_path = DATA_DIR / f"{model_prefix}_f{f_val}.pt"
        if not model_path.exists():
            print(f"[skip] checkpoint not found: {model_path}")
            continue

        print(f"\n=== Evaluating {model_prefix} at f={f_val} ===")
        net = load_model(
            checkpoint_path=model_path,
            input_dim=input_dim,
            use_delay=use_delay,
            hidden_units=HIDDEN_UNITS,
            num_classes=NUM_CLASSES,
            max_delay=MAX_DELAY,
        )

        result = test_with_repeats(net, test_loader, f=f_val, num_repeats=NUM_REPEATS)
        models[f_val] = net
        results[f_val] = result
        print(
            f"f={f_val} | test acc = {result['mean']:.4f} ± {result['std']:.4f}"
            f" | checkpoint <- {model_path}"
        )

    results_serialisable = {
        str(f_val): {
            "mean": float(d["mean"]),
            "std": float(d["std"]),
            "values": [float(v) for v in d["values"]],
        }
        for f_val, d in results.items()
    }
    results_path = (
        LOG_DIR / f"{model_prefix}_hidden_perturbation_results_evalOnly.json"
    )
    with open(results_path, "w") as fp:
        json.dump(results_serialisable, fp, indent=2)
    print(f"Results saved to {results_path}")

    return {
        "models": models,
        "results": results,
        "test_loader": test_loader,
        "model_prefix": model_prefix,
        "dataset_key": dataset_key,
        "use_delay": use_delay,
    }


def main() -> None:
    if TRAIN_ALL_VARIATION:
        print(f"Batch mode: evaluating {len(ALL_VARIATIONS)} variations:")
        for ds, ud in ALL_VARIATIONS:
            tag = "delay" if ud else "nodelay"
            print(f"  - shd_2ndLayer_{ds}_{tag}")
    else:
        input_dim = DATASET_CONFIGS[DATASET_KEY]["input_dim"]
        tag = "delay" if USE_DELAY else "nodelay"
        print(f"Single variation: dataset={DATASET_KEY} | input_dim={input_dim}")
        print(f"  Network mode: {'SGD-delay' if USE_DELAY else 'SGD (no delay)'}")
        print(f"  Model prefix: shd_2ndLayer_{DATASET_KEY}_{tag}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    variations_to_run = (
        ALL_VARIATIONS if TRAIN_ALL_VARIATION else [(DATASET_KEY, USE_DELAY)]
    )

    for ds_key, use_delay in variations_to_run:
        run_variation_sweep(ds_key, use_delay)


if __name__ == "__main__":
    main()
