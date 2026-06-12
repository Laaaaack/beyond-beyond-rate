"""SSC dataset 2nd-hidden-layer spike-timing perturbation training script.

Train SNNs on the SSC dataset (no perturbation), then evaluate with
spike-timing perturbation applied at the 2nd hidden layer output.
Perturbation fraction swept over [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].

Supports training across all dataset variants (whole/part/norm) and network
modes (SGD/SGD-delay) via train_all_variation flag.
"""

import json
import os
import random
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

TRAIN_ALL_VARIATION: bool = True

USE_DELAY: bool = False

DATASET_KEY: str = "whole"

DATASET_CONFIGS = {
    "whole": {"h5_file": "ssc_data/ssc_whole.h5", "input_dim": 700},
    "part": {"h5_file": "ssc_data/ssc_part.h5", "input_dim": 285},
    "norm": {"h5_file": "ssc_data/ssc_norm.h5", "input_dim": 285},
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
NUM_CLASSES: int = 35
EPOCHS: int = 600
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.1
SEED: int = 42
MAX_DELAY: int = 64
EARLY_STOP_PATIENCE: int = 300

F_VALUES: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_REPEATS: int = 3
N_TRAIN_CHUNKS: int = 1

INPUT_DIM: int = DATASET_CONFIGS[DATASET_KEY]["input_dim"]
H5_FILE: str = DATASET_CONFIGS[DATASET_KEY]["h5_file"]
DELAY_TAG: str = "delay" if USE_DELAY else "nodelay"
MODEL_PREFIX: str = f"ssc_l2_{DATASET_KEY}_{DELAY_TAG}"


# ============================================================================
# DATA LOADING
# ============================================================================


def load_split_from_h5(
    h5_path: str, indices: np.ndarray, target_T: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a subset of samples from an HDF5 file into memory.

    Args:
        h5_path: Path to the HDF5 file with keys 'X' and 'Y'.
        indices: Sorted array of sample indices to load.
        target_T: Target time dimension (zero-pad if shorter).

    Returns:
        Tuple of (X, Y) where X has shape (N, neurons, target_T) and Y is
        class labels.
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


def probe_and_load_eval_data(
    h5_path: str, target_T: int = 200
) -> Tuple[int, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Probe dataset metadata and load val/test splits into memory.

    Training data is loaded in chunks during training to reduce memory usage.

    Args:
        h5_path: Path to the HDF5 dataset file.
        target_T: Target time dimension for zero-padding.

    Returns:
        Tuple of (n_samples, train_indices, val_data, test_data).
    """
    with h5py.File(h5_path, "r") as hf:
        n_samples = hf["X"].shape[0]

    train_idx = np.arange(
        int(n_samples * TRAIN_RANGE[0]), int(n_samples * TRAIN_RANGE[1])
    )
    val_idx = np.arange(
        int(n_samples * VAL_RANGE[0]), int(n_samples * VAL_RANGE[1])
    )
    test_idx = np.arange(
        int(n_samples * TEST_RANGE[0]), int(n_samples * TEST_RANGE[1])
    )

    X_val, Y_val = load_split_from_h5(h5_path, val_idx, target_T)
    X_test, Y_test = load_split_from_h5(h5_path, test_idx, target_T)

    val_mem = X_val.nbytes / (1024**3)
    test_mem = X_test.nbytes / (1024**3)
    print(
        f"Dataset: {n_samples} samples | "
        f"Train: {len(train_idx)} ({N_TRAIN_CHUNKS} chunks) | "
        f"Val: {len(val_idx)} ({val_mem:.1f} GiB) | "
        f"Test: {len(test_idx)} ({test_mem:.1f} GiB)"
    )
    return n_samples, train_idx, (X_val, Y_val), (X_test, Y_test)


# ============================================================================
# PERTURBATION
# ============================================================================


def partial_randomize_spike_train(
    spike_train: np.ndarray, f: float = 0.0, max_attempts: int = 50
) -> np.ndarray:
    """Randomly relocate a fraction f of each neuron's spikes.

    Args:
        spike_train: Binary array of shape (num_neurons, T).
        f: Fraction of spikes to relocate (0 = no change, 1 = full shuffle).
        max_attempts: Max tries to find an empty time bin per spike.

    Returns:
        Perturbed spike train with the same shape and spike counts.
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
                for _ in range(max_attempts):
                    new_t = np.random.randint(0, T)
                    if new_train[neuron_idx, new_t] == 0:
                        new_train[neuron_idx, new_t] = 1
                        inserted = True
                        break
    return new_train


def perturb_hidden_batch(hidden_spikes: torch.Tensor, f: float) -> torch.Tensor:
    """Apply perturbation to a batch of hidden spike tensors.

    Args:
        hidden_spikes: Spike tensor of shape (B, C, 1, 1, T).
        f: Perturbation fraction.

    Returns:
        Perturbed spike tensor with the same shape and device.
    """
    dev = hidden_spikes.device
    spikes_np = hidden_spikes.cpu().numpy()
    B, C, _, _, T = spikes_np.shape

    for b in range(B):
        sample = spikes_np[b, :, 0, 0, :]
        spikes_np[b, :, 0, 0, :] = partial_randomize_spike_train(sample, f)

    return torch.from_numpy(spikes_np).to(dev)


# ============================================================================
# DATASET
# ============================================================================


class SpikeDataset(Dataset):
    """In-memory dataset for spike trains."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """Initialize dataset.

        Args:
            X: Spike data of shape (N, neurons, T), dtype uint8.
            Y: Class labels of shape (N,).
        """
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (spike tensor, label tensor).
        """
        x = torch.from_numpy(self.X[idx].astype(np.float32))
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


# ============================================================================
# NETWORK
# ============================================================================


class SSCNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN for SSC classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 128,
        num_classes: int = 35,
        use_delay: bool = True,
        max_delay: int = 64,
    ):
        """Initialize network.

        Args:
            input_dim: Number of input neurons.
            hidden_units: Number of hidden units.
            num_classes: Number of output classes.
            use_delay: Whether to use learnable delays.
            max_delay: Maximum delay in time steps.
        """
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
        """Ensure input is 5-D NCHWT on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(device)

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike -> (delay1)."""
        x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if self.use_delay:
            x = self.delay1(x)
        return x

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> PSP -> fc2 -> spike -> (delay2) -> hidden2 spikes."""
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        if self.use_delay:
            x = self.delay2(x)
        return x

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> PSP -> fc3 -> spike -> output spikes."""
        return self.slayer.spike(self.fc3(self.slayer.psp(hidden2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (no perturbation)."""
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)
        return self._output(hidden2)

    def forward_with_hidden_perturbation(
        self, x: torch.Tensor, f: float = 0.0
    ) -> torch.Tensor:
        """Forward pass with perturbation at 2nd hidden layer.

        Args:
            x: Input spike trains.
            f: Perturbation fraction.

        Returns:
            Output spike tensor.
        """
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        hidden2 = self._second_hidden(hidden1)

        if f > 0:
            hidden2 = perturb_hidden_batch(hidden2, f)

        return self._output(hidden2)

    def clamp_delays(self, max1: int = 64, max2: int = 64) -> None:
        """Clamp delay parameters to [0, max]."""
        if not self.use_delay:
            return
        self.delay1.delay.data.clamp_(0, max1)
        self.delay2.delay.data.clamp_(0, max2)

    def get_delays(self) -> Dict[str, np.ndarray]:
        """Return current delay values as a dict."""
        delays = {}
        if self.use_delay:
            delays["delay1"] = self.delay1.delay.data.cpu().numpy()
            delays["delay2"] = self.delay2.delay.data.cpu().numpy()
        return delays


# ============================================================================
# TRAINING
# ============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
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


def build_loss_and_optimizer(net: SSCNetwork, lr: float = 0.1) -> Tuple[Any, Any, Any]:
    """Build loss function, optimizer, and scheduler.

    Args:
        net: The network to optimize.
        lr: Base learning rate.

    Returns:
        Tuple of (loss_fn, optimizer, scheduler).
    """
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
    val_data: Tuple[np.ndarray, np.ndarray],
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
) -> Tuple[SSCNetwork, Dict[str, Any]]:
    """Train the SSCNetwork with chunked data loading.

    Args:
        h5_path: Path to the HDF5 dataset file.
        train_indices: Array of training sample indices.
        val_data: Tuple (X_val, Y_val) already in memory.
        target_T: Target time dimension.
        n_chunks: Number of chunks to split training data into.
        input_dim: Number of input neurons.
        hidden_units: Hidden layer size.
        num_classes: Number of output classes.
        use_delay: Whether to use learnable delays.
        max_delay: Maximum delay in time steps.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seed: Random seed.
        patience: Early stopping patience.

    Returns:
        Tuple of (trained network, training log dict).
    """
    set_seed(seed)

    net = SSCNetwork(
        input_dim, hidden_units, num_classes, use_delay, max_delay
    ).to(device)
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

    X_val, Y_val = val_data
    val_loader = DataLoader(
        SpikeDataset(X_val, Y_val), batch_size=batch_size, shuffle=False
    )

    chunk_splits = np.array_split(np.arange(len(train_indices)), n_chunks)

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    update1 = 0
    update2 = 0
    thea1 = max_delay
    thea2 = max_delay

    log: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "delay_mean": [],
    }

    for epoch in range(epochs):
        net.train()
        batch_losses = []

        epoch_order = np.random.permutation(len(train_indices))
        epoch_indices = train_indices[epoch_order]

        for chunk_pos in chunk_splits:
            chunk_idx = epoch_indices[chunk_pos]
            X_chunk, Y_chunk = load_split_from_h5(h5_path, chunk_idx, target_T)
            chunk_loader = DataLoader(
                SpikeDataset(X_chunk, Y_chunk), batch_size=batch_size, shuffle=True
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

            del X_chunk, Y_chunk, chunk_loader

        if use_delay:
            if epoch <= 250:
                net.clamp_delays(max_delay, max_delay)
            else:
                update1 += 1
                update2 += 1
                for name, param in net.named_parameters():
                    if "delay1.delay" in name and update1 > 150:
                        sorted_vals = torch.sort(
                            torch.floor(param.detach().flatten())
                        )[0]
                        thea1_val = torch.max(sorted_vals)
                        if sorted_vals[108] > (thea1_val - 5):
                            thea1 = int(thea1_val.item()) + 1
                            update1 = 0
                    elif "delay2.delay" in name and update2 > 150:
                        sorted_vals = torch.sort(
                            torch.floor(param.detach().flatten())
                        )[0]
                        thea2_val = torch.max(sorted_vals)
                        if sorted_vals[108] > (thea2_val - 5):
                            thea2 = int(thea2_val.item()) + 1
                            update2 = 0
                net.clamp_delays(thea1, thea2)

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
                y_batch = y_batch.to(device).long()

                target = torch.zeros(
                    (len(y_batch), num_classes, 1, 1, 1), device=device
                )
                target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

                outputs = net(x_batch)
                val_loss += loss_fn.numSpikes(outputs, target).item()

                pred = snn.predict.getClass(outputs)
                correct += (pred.cpu() == y_batch.cpu()).sum().item()
                total += len(y_batch)

        val_loss /= max(1, len(val_loader))
        val_acc = correct / max(1, total)
        train_loss = np.mean(batch_losses)

        delays = net.get_delays()
        avg_delay = (
            np.mean([np.mean(d) for d in delays.values() if len(d) > 0])
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in net.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


# ============================================================================
# TESTING
# ============================================================================


def test_with_hidden_perturbation(
    net: SSCNetwork, test_loader: DataLoader, f: float = 0.0
) -> float:
    """Evaluate accuracy with 2nd-hidden-layer perturbation.

    Args:
        net: Trained SSCNetwork.
        test_loader: Test DataLoader.
        f: Perturbation fraction applied to 2nd hidden layer spikes.

    Returns:
        Test accuracy as a float in [0, 1].
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
    net: SSCNetwork, test_loader: DataLoader, f_values: list, num_repeats: int = 3
) -> Dict[float, Dict[str, Any]]:
    """Sweep over perturbation levels and collect accuracy statistics.

    Args:
        net: Trained SSCNetwork.
        test_loader: Test DataLoader.
        f_values: List of perturbation fractions to evaluate.
        num_repeats: Number of independent evaluations per f.

    Returns:
        Dict mapping each f to {"mean", "std", "values"}.
    """
    results: Dict[float, Dict[str, Any]] = {}

    for f in f_values:
        accuracies = []
        for repeat in range(num_repeats):
            np.random.seed(SEED + repeat)
            acc = test_with_hidden_perturbation(net, test_loader, f=f)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[f] = {"mean": mean_acc, "std": std_acc, "values": accuracies}
        print(f"  f={f:.1f}:  accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Main training and evaluation loop."""
    print(f"Using device: {device}")
    print(f"train_all_variation = {TRAIN_ALL_VARIATION}")

    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    if TRAIN_ALL_VARIATION:
        config_grid = [(dk, ud) for dk in DATASET_CONFIGS for ud in (False, True)]
    else:
        config_grid = [(DATASET_KEY, USE_DELAY)]

    print(f"Running {len(config_grid)} configuration(s): {config_grid}\n")

    for dataset_key, use_delay in config_grid:
        input_dim = DATASET_CONFIGS[dataset_key]["input_dim"]
        h5_file = DATASET_CONFIGS[dataset_key]["h5_file"]
        delay_tag = "delay" if use_delay else "nodelay"
        model_prefix = f"ssc_l2_{dataset_key}_{delay_tag}"

        print(f"\n{'#'*70}")
        print(f"#  Configuration: dataset={dataset_key} | {delay_tag}")
        print(f"{'#'*70}")

        n_samples, train_idx, val_data, test_data = probe_and_load_eval_data(
            h5_file, target_T=SIM_PARAMS["tSample"]
        )

        print(f"\n  --- Training (clean inputs, {delay_tag}) ---")
        net, training_log = train_model(
            h5_path=h5_file,
            train_indices=train_idx,
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

        test_loader = DataLoader(
            SpikeDataset(test_data[0], test_data[1]),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        clean_acc = test_with_hidden_perturbation(net, test_loader, f=0.0)
        print(f"\nClean test accuracy (f=0): {clean_acc:.4f}")

        model_path = f"data/{model_prefix}_trained.pt"
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        print(f"\n  --- 2nd-hidden-layer perturbation sweep at evaluation ---")
        sweep_results = run_hidden_perturbation_sweep(
            net, test_loader, f_values=F_VALUES, num_repeats=NUM_REPEATS
        )

        results_serialisable = {
            str(f_val): {
                "mean": float(data["mean"]),
                "std": float(data["std"]),
                "values": [float(v) for v in data["values"]],
            }
            for f_val, data in sweep_results.items()
        }

        results_path = f"log/{model_prefix}_hidden_perturbation_results.json"
        with open(results_path, "w") as fp:
            json.dump(results_serialisable, fp, indent=2)
        print(f"Results saved to {results_path}")

        log_path = f"log/{model_prefix}_training_log.json"
        training_log_serialisable = {
            k: [float(v) for v in vals] if isinstance(vals, list) else vals
            for k, vals in training_log.items()
        }
        with open(log_path, "w") as fp:
            json.dump(training_log_serialisable, fp, indent=2)
        print(f"Training log saved to {log_path}")

        print(f"\n=== Model Analysis: dataset={dataset_key}, {delay_tag} ===")
        delays = net.get_delays()
        if delays:
            for delay_name, delay_values in delays.items():
                if len(delay_values) > 0:
                    print(
                        f"  {delay_name}: mean={np.mean(delay_values):.2f}, "
                        f"std={np.std(delay_values):.2f}, "
                        f"min={np.min(delay_values):.2f}, "
                        f"max={np.max(delay_values):.2f}"
                    )
        else:
            print("  No delays (SGD mode)")


if __name__ == "__main__":
    main()