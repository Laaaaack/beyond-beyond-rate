"""Per-spike jitter perturbation training script.

Experiment 3A: Train SNNs on SHD dataset (clean inputs), then evaluate
with per-spike Gaussian jitter applied at the 1st hidden layer output.
Jitter sigma swept over [0, 1, 3, 5, 10, 17, 25] ms.
"""

import json
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_ALL_VARIATION: bool = True
USE_DELAY: bool = False
DATASET_KEY: str = "whole"

DATASET_CONFIGS = {
    "whole": {"mat_file": "../../realistic/shd/shd_data/shd_whole.mat", "input_dim": 700},
    "part": {"mat_file": "../../realistic/shd/shd_data/shd_part_new.mat", "input_dim": 224},
    "norm": {"mat_file": "../../realistic/shd/shd_data/shd_norm_new.mat", "input_dim": 224},
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

SIGMA_VALUES: list = [0, 1, 3, 5, 10, 17, 25]
NUM_REPEATS: int = 3

INPUT_DIM: int = DATASET_CONFIGS[DATASET_KEY]["input_dim"]
MAT_FILE: str = DATASET_CONFIGS[DATASET_KEY]["mat_file"]
DELAY_TAG: str = "delay" if USE_DELAY else "nodelay"
MODEL_PREFIX: str = f"jitter_{DATASET_KEY}_{DELAY_TAG}"


def load_shd_data(mat_path: str, target_T: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Load SHD dataset from mat file and pad time dimension."""
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


def jitter_spike_train(
    spike_train: np.ndarray, sigma: float = 0.0, max_attempts: int = 50
) -> np.ndarray:
    """Apply per-spike Gaussian jitter to a binary spike train."""
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
            for _ in range(max_attempts):
                jittered_time = int(round(old_time + np.random.normal(0, sigma)))
                jittered_time = np.clip(jittered_time, 0, T - 1)

                if new_train[neuron_idx, jittered_time] == 0:
                    new_train[neuron_idx, jittered_time] = 1
                    inserted = True
                    break

            if not inserted:
                new_train[neuron_idx, old_time] = 1

    return new_train


def jitter_hidden_batch(hidden_spikes: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply per-spike jitter to hidden spike batch."""
    dev = hidden_spikes.device
    spikes_np = hidden_spikes.detach().cpu().numpy()
    B, C, _, _, T = spikes_np.shape

    for b in range(B):
        sample = spikes_np[b, :, 0, 0, :]
        spikes_np[b, :, 0, 0, :] = jitter_spike_train(sample, sigma)

    return torch.from_numpy(spikes_np).to(dev)


class SpikeDataset(Dataset):
    """Spike train dataset wrapper."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


def build_dataloaders(
    X: np.ndarray, Y: np.ndarray, batch_size: int = 128, seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split data and build dataloaders."""
    N = len(Y)
    train_idx = np.arange(int(N * TRAIN_RANGE[0]), int(N * TRAIN_RANGE[1]))
    val_idx = np.arange(int(N * VAL_RANGE[0]), int(N * VAL_RANGE[1]))
    test_idx = np.arange(int(N * TEST_RANGE[0]), int(N * TEST_RANGE[1]))

    np.random.seed(seed)
    np.random.shuffle(train_idx)

    train_ds = SpikeDataset(X[train_idx], Y[train_idx])
    val_ds = SpikeDataset(X[val_idx], Y[val_idx])
    test_ds = SpikeDataset(X[test_idx], Y[test_idx])

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


class JitterNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with per-spike jitter."""

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

    def _second_hidden_and_output(self, hidden1: torch.Tensor) -> torch.Tensor:
        if self.use_delay:
            hidden1 = self.delay1(hidden1)
        x = self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))
        if self.use_delay:
            x = self.delay2(x)
        x = self.slayer.spike(self.fc3(self.slayer.psp(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_hidden_perturbation(
        self, x: torch.Tensor, sigma: float = 0.0
    ) -> torch.Tensor:
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)

        if sigma > 0:
            hidden1 = jitter_hidden_batch(hidden1, sigma)

        return self._second_hidden_and_output(hidden1)

    def clamp_delays(self, max1: int = 64, max2: int = 64) -> None:
        if not self.use_delay:
            return
        self.delay1.delay.data.clamp_(0, max1)
        self.delay2.delay.data.clamp_(0, max2)

    def get_delays(self) -> Dict[str, np.ndarray]:
        delays = {}
        if self.use_delay:
            delays["delay1"] = self.delay1.delay.data.cpu().numpy()
            delays["delay2"] = self.delay2.delay.data.cpu().numpy()
        return delays


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


def build_loss_and_optimizer(net: JitterNetwork, lr: float = 0.1) -> Tuple[Any, Any, Any]:
    """Build loss, optimizer, and scheduler."""
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int = 700,
    hidden_units: int = 128,
    num_classes: int = 20,
    use_delay: bool = False,
    max_delay: int = 64,
    epochs: int = 1250,
    lr: float = 0.1,
    seed: int = 42,
    patience: int = 300,
) -> Tuple[JitterNetwork, Dict[str, Any]]:
    """Train a JitterNetwork on unperturbed inputs."""
    set_seed(seed)

    net = JitterNetwork(input_dim, hidden_units, num_classes, use_delay, max_delay).to(
        device
    )
    loss_fn, optimizer, scheduler = build_loss_and_optimizer(net, lr=lr)
    loss_fn = loss_fn.to(device)

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

    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training (clean)")

    for epoch in range(epochs):
        net.train()
        batch_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device).long()

            target = torch.zeros((len(y_batch), num_classes, 1, 1, 1), device=device)
            target.scatter_(1, y_batch[:, None, None, None, None], 1.0)

            outputs = net(x_batch)
            loss = loss_fn.numSpikes(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            pbar.update(1)

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

                target = torch.zeros((len(y_batch), num_classes, 1, 1, 1), device=device)
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

        pbar.set_postfix(
            epoch=epoch + 1,
            train=f"{train_loss:.3f}",
            val=f"{val_loss:.3f}",
            acc=f"{val_acc:.2%}",
            delay=f"{avg_delay:.1f}",
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

    pbar.close()

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net, log


def test_with_jitter(net: JitterNetwork, test_loader: DataLoader, sigma: float = 0.0) -> float:
    """Evaluate accuracy with hidden-layer jitter."""
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            y_batch = y_batch.to(device)

            outputs = net.forward_with_hidden_perturbation(x_batch, sigma=sigma)
            predicted = snn.predict.getClass(outputs)

            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch.cpu()).sum().item()

    return correct / total


def test_with_repeats(
    net: JitterNetwork, test_loader: DataLoader, sigma: float, num_repeats: int = 3
) -> Dict[str, Any]:
    """Evaluate with jitter multiple times for error bars."""
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
        mat_file = DATASET_CONFIGS[dataset_key]["mat_file"]
        delay_tag = "delay" if use_delay else "nodelay"
        model_prefix = f"jitter_{dataset_key}_{delay_tag}"

        print(f"\n{'#'*70}")
        print(f"#  Configuration: dataset={dataset_key} | {delay_tag}")
        print(f"{'#'*70}")

        X_cfg, Y_cfg = load_shd_data(mat_file, target_T=SIM_PARAMS["tSample"])
        train_loader, val_loader, test_loader = build_dataloaders(
            X_cfg, Y_cfg, batch_size=BATCH_SIZE, seed=SEED
        )

        print(f"\n  --- Training (clean inputs, {delay_tag}) ---")
        net, training_log = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            use_delay=use_delay,
        )

        model_path = f"data/{model_prefix}_trained.pt"
        torch.save(net.state_dict(), model_path)
        print(f"\n  Model saved to {model_path}")

        print(f"\n  --- Hidden-jitter sweep at evaluation ---")
        cfg_test_results: Dict[int, Dict[str, Any]] = {}
        for sigma in SIGMA_VALUES:
            test_result = test_with_repeats(net, test_loader, sigma=sigma)
            cfg_test_results[sigma] = test_result
            print(
                f"    sigma={sigma:3d} ms | "
                f"accuracy = {test_result['mean']:.4f} +/- {test_result['std']:.4f}"
            )

        sweep_serialisable = {
            str(sigma): {
                "mean": data["mean"],
                "std": data["std"],
                "values": [float(v) for v in data["values"]],
            }
            for sigma, data in cfg_test_results.items()
        }
        results_path = f"log/{model_prefix}_jitter_sweep_results.json"
        with open(results_path, "w") as fp:
            json.dump(sweep_serialisable, fp, indent=2)
        print(f"Sweep results saved to {results_path}")

        log_path = f"log/{model_prefix}_training_log.json"
        log_serialisable = {
            k: [float(v) for v in vals] if isinstance(vals, list) else vals
            for k, vals in training_log.items()
        }
        with open(log_path, "w") as fp:
            json.dump(log_serialisable, fp, indent=2)
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
