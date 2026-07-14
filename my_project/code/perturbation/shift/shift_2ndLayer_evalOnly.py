"""Experiment 3B: Hidden Per-Neuron Jitter (Shift) — 2nd Hidden Layer (eval-only).

Load 2-hidden-layer SNNs that were trained on the clean SHD dataset and sweep
per-neuron Gaussian shift applied at the 2nd hidden layer output at evaluation
time only (train-clean / eval-perturbed protocol). A single Gaussian offset is
drawn per neuron and applied to all of that neuron's spikes. **No training is
performed.**

This is the "train once / eval all" companion to ``shift_2ndLayer_train.py``.
Instead of training a fresh model per shift level, it loads the single
clean-trained checkpoint ``shift_2ndLayer_{dataset}_{delay_tag}_sigma0.pt`` for
each configuration and evaluates that one model across the whole sigma sweep.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR = SCRIPT_DIR / "log"

sys.path.append(str(SCRIPT_DIR / "../../../temporal_shd_project/code/src"))
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
EVAL_ALL_VARIATION: bool = True
USE_DELAY: bool = False
DATASET_KEY: str = "norm"

ALL_VARIATIONS: list[tuple[str, bool]] = [
    (dataset, delay)
    for dataset in ("norm", "part", "whole")
    for delay in (False, True)
]

DATASET_CONFIGS = {
    "whole": {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_whole.mat"), "input_dim": 700},
    "part":  {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_part_new.mat"), "input_dim": 224},
    "norm":  {"mat_file": str(SCRIPT_DIR / "../../realistic/shd/shd_data/shd_norm_new.mat"), "input_dim": 224},
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
BATCH_SIZE: int = 128
SEED: int = 42
MAX_DELAY: int = 64

SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]
NUM_REPEATS: int = 3

INPUT_DIM: int = DATASET_CONFIGS[DATASET_KEY]["input_dim"]


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


@torch.no_grad()
def shift_hidden_batch(
    hidden_spikes: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Vectorised GPU-side per-neuron Gaussian shift.

    For each (batch, neuron) a single integer offset ``round(N(0, sigma))``
    is drawn and every spike of that neuron is shifted by it, then clipped
    to ``[0, T - 1]``. Spikes that collide after clipping are merged
    via logical OR.
    """
    if sigma <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = (x > 0.5).to(x.dtype)

    offset = (torch.randn(B, C, 1, device=x.device) * sigma).round().long()
    t_idx = torch.arange(T, device=x.device).view(1, 1, T)
    target = (t_idx + offset).clamp(0, T - 1).expand(B, C, T).contiguous()

    new_x = torch.zeros_like(x)
    new_x.scatter_add_(-1, target, is_spike)
    new_x = (new_x > 0.5).to(hidden_spikes.dtype)

    return new_x.view(B, C, H, W, T)


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


class ShiftSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with per-neuron shift at 2nd hidden layer."""

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
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        x = hidden1
        if self.use_delay:
            x = self.delay1(x)
        return self.slayer.spike(self.fc2(self.slayer.psp(x)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        x = hidden2
        if self.use_delay:
            x = self.delay2(x)
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

    def _apply_shift(
        self,
        hidden: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Apply per-neuron shift to the 2nd hidden layer output (eval-only)."""
        if sigma <= 0:
            return hidden
        return shift_hidden_batch(hidden, sigma)

    def forward(
        self,
        x: torch.Tensor,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with per-neuron shift at 2nd hidden layer."""
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)
        hidden2 = self._apply_shift(hidden2, sigma)
        return self._output(hidden2)


def load_pretrained_model(
    model_path: Path,
    input_dim: int,
    hidden_units: int = HIDDEN_UNITS,
    num_classes: int = NUM_CLASSES,
    use_delay: bool = False,
    max_delay: int = MAX_DELAY,
) -> ShiftSHDNetwork:
    """Load a clean-trained ShiftSHDNetwork checkpoint ready for evaluation."""
    net = ShiftSHDNetwork(
        input_dim, hidden_units, num_classes, use_delay, max_delay
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    print(f"  Loaded model from {model_path}")
    print(f"    Architecture: {input_dim} -> {hidden_units} -> {hidden_units} -> {num_classes}")
    print(f"    Delay: {'Yes' if use_delay else 'No'}")
    return net


def test_with_shift(
    net: ShiftSHDNetwork,
    test_loader: DataLoader,
    sigma: float = 0.0,
) -> float:
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device)

            outputs = net(x_batch, sigma=sigma)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: ShiftSHDNetwork,
    test_loader: DataLoader,
    sigma: float,
    num_repeats: int = NUM_REPEATS,
) -> dict:
    accuracies = []
    for repeat in range(num_repeats):
        np.random.seed(SEED + repeat)
        torch.manual_seed(SEED + repeat)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + repeat)
        acc = test_with_shift(net, test_loader, sigma=sigma)
        accuracies.append(acc)

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": accuracies,
    }


def run_variation_sweep(
    dataset_key: str,
    use_delay: bool,
) -> dict:
    """Load the clean checkpoint and sweep 2nd hidden layer shift at eval time."""
    cfg = DATASET_CONFIGS[dataset_key]
    input_dim = cfg["input_dim"]
    mat_file = cfg["mat_file"]
    delay_tag = "delay" if use_delay else "nodelay"
    model_prefix = f"shift_2ndLayer_{dataset_key}_{delay_tag}"
    ckpt_path = DATA_DIR / f"{model_prefix}_sigma0.pt"

    print(f"\n{'#' * 70}")
    print(f"# 2nd Hidden Layer Shift (eval-only): dataset={dataset_key} | delay={delay_tag}")
    print(f"# Checkpoint: {ckpt_path}")
    print(f"{'#' * 70}")

    X, Y = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
    _, _, test_loader = build_dataloaders(
        X, Y, batch_size=BATCH_SIZE, seed=SEED,
    )

    net = load_pretrained_model(
        model_path=ckpt_path,
        input_dim=input_dim,
        use_delay=use_delay,
    )

    results: dict[int, dict] = {}
    for sigma in SIGMA_VALUES:
        result = test_with_repeats(net, test_loader, sigma=sigma)
        results[sigma] = result
        print(
            f"sigma={sigma} | test acc = {result['mean']:.4f} +/- {result['std']:.4f}"
        )

    sweep_serialisable = {
        str(sigma): {
            "mean": float(d["mean"]),
            "std": float(d["std"]),
            "values": [float(v) for v in d["values"]],
        }
        for sigma, d in results.items()
    }
    results_path = LOG_DIR / f"{model_prefix}_evalOnly_shift_sweep_results.json"
    with open(results_path, "w") as fp:
        json.dump(sweep_serialisable, fp, indent=2)
    print(f"Sweep results saved to {results_path}")

    return {
        "net": net,
        "results": results,
        "test_loader": test_loader,
        "model_prefix": model_prefix,
        "dataset_key": dataset_key,
        "use_delay": use_delay,
    }


def main() -> None:
    if EVAL_ALL_VARIATION:
        print(f"Batch mode: evaluating {len(ALL_VARIATIONS)} variations:")
        for ds, ud in ALL_VARIATIONS:
            tag = "delay" if ud else "nodelay"
            print(f"  - shift_2ndLayer_{ds}_{tag}")
        print(f"Sigma sweep: {SIGMA_VALUES}")
    else:
        tag = "delay" if USE_DELAY else "nodelay"
        print(f"Dataset: SHD {DATASET_KEY} | Input dim: {INPUT_DIM}")
        print(f"Network mode: {'SGD-delay' if USE_DELAY else 'SGD (no delay)'}")
        print(f"Model prefix: shift_2ndLayer_{DATASET_KEY}_{tag}")
        print(f"Sigma sweep: {SIGMA_VALUES}")

    os.makedirs(LOG_DIR, exist_ok=True)

    variations_to_run = (
        ALL_VARIATIONS if EVAL_ALL_VARIATION else [(DATASET_KEY, USE_DELAY)]
    )

    for ds_key, use_delay in variations_to_run:
        run_variation_sweep(ds_key, use_delay)


if __name__ == "__main__":
    main()
