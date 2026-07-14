"""Experiment 2B: SSC — Hidden-Layer Perturbation.

Train a 2-hidden-layer SLAYER SNN on the Spiking Speech Commands (SSC)
dataset with no perturbation (f=0), then evaluate it by applying spike-timing
perturbation to the output of the 1st hidden layer at test time. Supports the
whole/part/norm dataset variants and SGD (no delay) / SGD-delay modes.

Converted from ssc_train.ipynb: plotting and inspection-only code removed,
top-level run cells folded into run_variation(), with a main() dispatcher and
a TRAIN_ALL_VARIATION switch to sweep every delay x dataset combination.
"""

import os
import json
import random

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

# --- Train-all switch ---
# When True, main() loops over every (delay x dataset-variant) combination.
# When False, main() runs the single config set by USE_DELAY / DATASET_KEY.
TRAIN_ALL_VARIATION: bool = True

# Option lists used when TRAIN_ALL_VARIATION is True.
DELAY_OPTIONS: list   = [True, False]
DATASET_VARIANTS: list = ["whole", "part", "norm"]

# --- Single-run flags (used when TRAIN_ALL_VARIATION is False) ---
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
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE   = (0.6, 0.75)
TEST_RANGE  = (0.75, 0.9)

# --- Training hyper-parameters ---
HIDDEN_UNITS: int        = 128
NUM_CLASSES: int         = 35   # SSC has 35 spoken-word classes
EPOCHS: int              = 1250
BATCH_SIZE: int          = 128
LEARNING_RATE: float     = 0.1
SEED: int                = 42
MAX_DELAY: int           = 64
EARLY_STOP_PATIENCE: int = 300

# --- Hidden-perturbation sweep ---
F_VALUES: list   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_REPEATS: int = 3

# --- Chunked training (to fit large datasets in RAM) ---
# Training data is split into this many chunks; each chunk is loaded
# into memory one at a time.  Val/test are loaded fully.
# Set to 1 for part/norm (small enough to fit), 3+ for whole.
N_TRAIN_CHUNKS: int = 1


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
    # would see skewed, partially-missing classes. Seeded here for
    # reproducibility; train_model() re-seeds the global RNG afterwards.
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
        f"Train: {len(train_idx)} (chunked, {N_TRAIN_CHUNKS} chunks) | "
        f"Val: {len(val_idx)} ({val_mem:.1f} GiB) | "
        f"Test: {len(test_idx)} ({test_mem:.1f} GiB)"
    )
    return n_samples, train_idx, (X_val, Y_val), (X_test, Y_test)


# =====================================================================
# Hidden-Layer Spike Perturbation
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


def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float,
) -> torch.Tensor:
    # Converts to numpy, perturbs each sample, converts back.
    # Expects SLAYER's 5-D format: (batch, neurons, 1, 1, T).
    dev = hidden_spikes.device
    spikes_np = hidden_spikes.cpu().numpy()
    B, C, H, W, T = spikes_np.shape

    for b in range(B):
        sample = spikes_np[b, :, 0, 0, :]  # (C, T)
        spikes_np[b, :, 0, 0, :] = partial_randomize_spike_train(sample, f)

    return torch.from_numpy(spikes_np).to(dev)


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

    def forward(self, x):
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)
        return self._output(hidden2)

    def forward_with_hidden_perturbation(self, x, f=0.0):
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)
        if f > 0:
            hidden2 = perturb_hidden_batch(hidden2, f)
        return self._output(hidden2)

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
# Training Loop
# =====================================================================

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


def build_loss_and_optimizer(
    net: SSCNetwork,
    lr: float = 0.1,
) -> tuple:
    error_cfg = {
        "neuron": LIF_PARAMS,
        "simulation": SIM_PARAMS,
        "training": {
            "error": {
                "type": "NumSpikes",
                "tgtSpikeRegion": {"start": 0, "stop": 200},
                "tgtSpikeCount": {True: 40, False: 4},
            }
        },
    }
    loss_fn = snn.spikeLoss.spikeLoss(error_cfg)
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300], gamma=0.1
    )
    return loss_fn, optimizer, scheduler


def train_model(
    h5_path: str,
    train_indices: np.ndarray,
    val_data: tuple,
    target_T: int = 200,
    n_chunks: int = 3,
    input_dim: int = 700,
    hidden_units: int = 128,
    num_classes: int = 35,
    use_delay: bool = True,
    max_delay: int = 64,
    epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.1,
    seed: int = 42,
    patience: int = 300,
) -> tuple:
    set_seed(seed)

    net = SSCNetwork(
        input_dim, hidden_units, num_classes, use_delay, max_delay
    ).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    # Pre-build val loader (stays in memory)
    X_val, Y_val = val_data
    val_loader = DataLoader(
        SpikeDataset(X_val, Y_val), batch_size=batch_size, shuffle=False
    )

    # Pre-split training indices into chunk boundaries
    chunk_splits = np.array_split(np.arange(len(train_indices)), n_chunks)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    # Adaptive delay clamping state
    update1 = 0
    update2 = 0
    thea1 = max_delay
    thea2 = max_delay

    log = {
        "epoch":      [],
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
        "delay_mean": [],
    }

    for epoch in range(epochs):
        # --- Train (chunked) ---
        net.train()
        batch_losses = []

        # Shuffle training indices each epoch
        epoch_order = np.random.permutation(len(train_indices))
        epoch_indices = train_indices[epoch_order]

        for chunk_id, chunk_pos in enumerate(chunk_splits):
            # Load this chunk into memory
            chunk_idx = epoch_indices[chunk_pos]
            X_chunk, Y_chunk = load_split_from_h5(
                h5_path, chunk_idx, target_T
            )
            chunk_loader = DataLoader(
                SpikeDataset(X_chunk, Y_chunk),
                batch_size=batch_size, shuffle=True,
            )

            for x_batch, y_batch in chunk_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs = net(x_batch)
                loss = loss_fn.numSpikes(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            # Free chunk memory
            del X_chunk, Y_chunk, chunk_loader

        # --- Adaptive delay clamping ---
        if use_delay:
            if epoch <= 250:
                net.clamp_delays(max_delay, max_delay)
            else:
                update1 += 1
                update2 += 1
                for name, param in net.named_parameters():
                    if "delay1.delay" in name and update1 > 150:
                        sorted_ = torch.sort(
                            torch.floor(param.detach().flatten())
                        )[0]
                        thea1_val = torch.max(sorted_)
                        if sorted_[108] > (thea1_val - 5):
                            thea1 = int(thea1_val.item()) + 1
                            update1 = 0
                    elif "delay2.delay" in name and update2 > 150:
                        sorted_ = torch.sort(
                            torch.floor(param.detach().flatten())
                        )[0]
                        thea2_val = torch.max(sorted_)
                        if sorted_[108] > (thea2_val - 5):
                            thea2 = int(thea2_val.item()) + 1
                            update2 = 0
                net.clamp_delays(thea1, thea2)

        # --- Validate ---
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = (
                    x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                )
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(
                    1, y_batch[:, None, None, None, None], 1.0
                )

                outputs = net(x_batch)
                val_loss += loss_fn.numSpikes(outputs, target).item()

                pred = snn.predict.getClass(outputs)
                correct += (pred.cpu() == y_batch.cpu()).sum().item()
                total += len(y_batch)

        val_loss /= max(1, len(val_loader))
        val_acc = correct / max(1, total)
        train_loss = np.mean(batch_losses)

        # Log delay statistics
        delays = net.get_delays()
        avg_delay = (
            np.mean([
                np.mean(d) for d in delays.values() if len(d) > 0
            ])
            if delays
            else 0.0
        )

        log["epoch"].append(epoch)
        log["train_loss"].append(float(train_loss))
        log["val_loss"].append(float(val_loss))
        log["val_acc"].append(float(val_acc))
        log["delay_mean"].append(float(avg_delay))

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train {train_loss:.3f} | "
            f"Val {val_loss:.3f} | "
            f"Acc {val_acc:.2%} | "
            f"Delay {avg_delay:.1f}"
        )
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.clone() for k, v in net.state_dict().items()
            }
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


# =====================================================================
# Testing with Hidden-Layer Perturbation
# =====================================================================

def test_with_hidden_perturbation(
    net: SSCNetwork,
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
    results = {}

    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            acc = test_with_hidden_perturbation(net, test_loader, f=f)
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

    data_dir = os.path.join(SCRIPT_DIR, "data")
    log_dir = os.path.join(SCRIPT_DIR, "log")

    print(f"\n{'=' * 70}")
    print(
        f"Variation: {model_prefix} | Input dim: {input_dim} | "
        f"Mode: {'SGD-delay' if use_delay else 'SGD (no delay)'}"
    )
    print(f"{'=' * 70}")

    # --- Load eval data + probe split indices ---
    n_samples, train_indices, val_data, test_data = probe_and_load_eval_data(
        h5_file, target_T=SIM_PARAMS["tSample"]
    )

    # --- Train on unperturbed data (f=0) ---
    net, training_log = train_model(
        h5_path=h5_file,
        train_indices=train_indices,
        val_data=val_data,
        target_T=SIM_PARAMS["tSample"],
        n_chunks=N_TRAIN_CHUNKS,
        input_dim=input_dim,
        hidden_units=HIDDEN_UNITS,
        num_classes=NUM_CLASSES,
        use_delay=use_delay,
        max_delay=MAX_DELAY,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        seed=SEED,
        patience=EARLY_STOP_PATIENCE,
    )

    # --- Build test loader from pre-loaded test data ---
    test_loader = DataLoader(
        SpikeDataset(test_data[0], test_data[1]),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # Sanity check: accuracy on clean test set (f=0)
    clean_acc = test_with_hidden_perturbation(net, test_loader, f=0.0)
    print(f"\nClean test accuracy (f=0): {clean_acc:.4f}")

    # --- Save best model ---
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, f"{model_prefix}_trained.pt")
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # --- Hidden-perturbation sweep ---
    print(
        f"=== Hidden-Layer Perturbation Sweep "
        f"(SSC {dataset_key}, {delay_tag}) ==="
    )
    sweep_results = run_hidden_perturbation_sweep(
        net, test_loader, f_values=F_VALUES, num_repeats=NUM_REPEATS
    )

    # --- Save results + training log to JSON ---
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

    log_path = os.path.join(log_dir, f"{model_prefix}_training_log.json")
    training_log_serialisable = {
        k: [float(v) for v in vals] if isinstance(vals, list) else vals
        for k, vals in training_log.items()
    }
    with open(log_path, "w") as fp:
        json.dump(training_log_serialisable, fp, indent=2)
    print(f"Training log saved to {log_path}")

    return sweep_results


# =====================================================================
# Dispatcher
# =====================================================================

def main() -> None:
    if TRAIN_ALL_VARIATION:
        for use_delay in DELAY_OPTIONS:
            for dataset_key in DATASET_VARIANTS:
                run_variation(use_delay, dataset_key)
    else:
        run_variation(USE_DELAY, DATASET_KEY)


if __name__ == "__main__":
    main()