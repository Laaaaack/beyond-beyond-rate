"""Firing statistics of the **non-sparse** baseline networks, for the v3L12 tables.

``result_visualization/bothLayer/firing_statistics.ipynb`` tabulates what the sparsity
penalties did to the hidden layers. Those numbers only mean something against the
network the penalties were removed from, and no training summary carries that row: the
baseline was trained in ``exp_fixed_weight_perturbation``, whose logs record accuracy
under perturbation and nothing about firing.

This script fills the gap. It loads the two unconstrained checkpoints

    exp_fixed_weight_perturbation/code/perturbation/jitter/data/
        jitter_whole_{nodelay,delay}_trained.pt

and measures them with the **same** procedure ``evaluate_firing_statistics()`` in
``sn_bothLayer_train_{noDelay,withDelay}_v3.py`` applies to the v3L12 checkpoints: the
same architecture, the same ``shd_whole.mat`` (byte-identical copies in both
experiments), the same fixed fractional test split ``[0.75, 0.9)``, and the same
per-(sample, neuron) reductions. The only field it cannot emit is ``over_k_fraction``,
which counts pairs above a ceiling the baseline does not have.

Eval-only: trains nothing, and writes ``v3_analysis/log/baseline_firing_statistics.json``
for the notebook to read, so the notebook itself needs no GPU.

Run from anywhere::

    python my_project/exp_sparse_network/v3_analysis/baseline_firing_statistics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "log"
MAT_FILE = EXP_DIR / "shd" / "shd_data" / "shd_whole.mat"
# The unconstrained networks, trained in the fixed-weight-perturbation experiment on
# this same data and split. `jitter_whole_*_trained.pt` is the sigma = 0 model, i.e.
# ordinary training with no perturbation and no sparsity penalty.
BASELINE_DIR = (EXP_DIR.parent / "exp_fixed_weight_perturbation" / "code"
                / "perturbation" / "jitter" / "data")

sys.path.insert(0, str(EXP_DIR))
import slayerSNN as snn  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (arm, use_delay), in the order the notebook lists them.
ARMS: tuple[tuple[str, bool], ...] = (("nodelay", False), ("delay", True))

# --- Architecture and simulation, identical to both training scripts ---
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
INPUT_DIM: int = 700
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
BATCH_SIZE: int = 128
TARGET_LAYERS: tuple[int, ...] = (1, 2)

# Fixed fractional splits, shared by every model in the project.
TEST_RANGE = (0.75, 0.9)
# theta_margin of the v3L12 floor. The baseline never saw this penalty; the fraction is
# reported so the `<theta%` column is comparable across the table.
THETA_MARGIN: float = 11.0


class BaselineNetwork(nn.Module):
    """The v1/baseline SLAYER SNN, returning both hidden layers' internals.

    Parameter names match ``SparseSHDNetwork`` and the fixed-weight-perturbation
    ``JitterSHDNetwork`` — ``fc1``/``fc2``/``fc3`` plus ``delay1``/``delay2`` — so the
    baseline checkpoints load into it unchanged.
    """

    def __init__(self, use_delay: bool):
        """Build the network.

        Args:
            use_delay: True for the with-delay arm, which adds the two delay layers.
        """
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.use_delay = use_delay
        self.fc1 = nn.utils.weight_norm(
            slayer.dense(INPUT_DIM, HIDDEN_UNITS), name="weight")
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, HIDDEN_UNITS), name="weight")
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, NUM_CLASSES), name="weight")
        if use_delay:
            self.delay1 = slayer.delay(HIDDEN_UNITS)
            self.delay2 = slayer.delay(HIDDEN_UNITS)

    def forward(self, x: torch.Tensor) -> tuple:
        """Run the clean forward pass, exposing both hidden layers.

        Args:
            x: Input spike trains, shape (B, 700, 1, 1, T).

        Returns:
            Tuple of (output spikes, hidden1, potential1, hidden2, potential2), the
            same five tensors ``SparseSHDNetwork.forward(return_hidden=True)`` returns.
        """
        potential1 = self.fc1(self.slayer.psp(x))
        hidden1 = self.slayer.spike(potential1)

        routed = self.delay1(hidden1) if self.use_delay else hidden1
        potential2 = self.fc2(self.slayer.psp(routed))
        hidden2 = self.slayer.spike(potential2)

        out = self.delay2(hidden2) if self.use_delay else hidden2
        outputs = self.slayer.spike(self.fc3(self.slayer.psp(out)))
        return outputs, hidden1, potential1, hidden2, potential2


def load_shd_data(mat_path: Path, target_T: int = 200) -> tuple:
    """Load the SHD dataset and pad its time dimension, as training does.

    Args:
        mat_path: Path to the ``.mat`` file holding ``X`` and ``Y``.
        target_T: Target time dimension; shorter recordings are zero-padded.

    Returns:
        Tuple of (X, Y) with X of shape (N, neurons, target_T).
    """
    data = loadmat(str(mat_path))
    features = data["X"]
    labels = data["Y"].ravel()

    n_samples, n_neurons, n_bins = features.shape
    if n_bins < target_T:
        padded = np.zeros((n_samples, n_neurons, target_T), dtype=features.dtype)
        padded[:, :, :n_bins] = features
        features = padded
    return features, labels


def build_test_loader() -> DataLoader:
    """Return the test split as a loader, identical to the training scripts'.

    Returns:
        A DataLoader over samples ``[0.75 N, 0.9 N)``, unshuffled.
    """
    features, labels = load_shd_data(MAT_FILE, SIM_PARAMS["tSample"])
    total = len(labels)
    index = np.arange(int(total * TEST_RANGE[0]), int(total * TEST_RANGE[1]))
    dataset = TensorDataset(torch.tensor(features[index], dtype=torch.float32),
                            torch.tensor(labels[index].astype(np.int64)))
    print(f"test split: {len(dataset)} samples of {total}")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def load_baseline(arm: str, use_delay: bool) -> BaselineNetwork:
    """Return one arm's trained baseline network, in eval mode on ``device``.

    Args:
        arm: ``"nodelay"`` or ``"delay"``.
        use_delay: Whether this arm has delay layers.

    Returns:
        The loaded network.
    """
    checkpoint = BASELINE_DIR / f"jitter_whole_{arm}_trained.pt"
    net = BaselineNetwork(use_delay).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    print(f"{arm}: loaded {checkpoint.name}")
    return net


def evaluate_firing_statistics(net: BaselineNetwork,
                               test_loader: DataLoader) -> dict:
    """Measure one baseline network's firing statistics on the test split.

    Mirrors ``evaluate_firing_statistics()`` in the v3 training scripts field for
    field, minus ``over_k_fraction``: the baseline has no ceiling to be over.

    Args:
        net: A loaded baseline network.
        test_loader: The test split loader.

    Returns:
        Dict of clean accuracy plus every firing axis at layer 1, layer 2 and pooled.
    """
    correct = 0
    total = 0
    totals = {
        layer: {"spikes": 0.0, "sq": 0.0, "slots": 0, "pairs": 0,
                "silent": 0.0, "sub_threshold": 0.0}
        for layer in TARGET_LAYERS
    }
    silent_peaks = {layer: [] for layer in TARGET_LAYERS}
    active_peaks = {layer: [] for layer in TARGET_LAYERS}

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.unsqueeze(2).unsqueeze(3).float().to(device)
            outputs, hidden1, potential1, hidden2, potential2 = net(x_batch)

            pred = snn.predict.getClass(outputs)
            correct += (pred.cpu() == y_batch.cpu()).sum().item()
            total += y_batch.size(0)

            spikes = {1: hidden1, 2: hidden2}
            potentials = {1: potential1, 2: potential2}
            for layer in TARGET_LAYERS:
                batch, channels, _, _, time = spikes[layer].shape
                counts = spikes[layer].sum(dim=-1).view(batch, channels)
                peak = potentials[layer].max(dim=-1).values.view(batch, channels)
                silent = counts == 0

                bucket = totals[layer]
                bucket["spikes"] += counts.sum().item()
                bucket["sq"] += (counts ** 2).sum().item()
                bucket["slots"] += batch * channels * time
                bucket["pairs"] += batch * channels
                bucket["silent"] += silent.sum().item()
                bucket["sub_threshold"] += (peak < THETA_MARGIN).sum().item()
                silent_peaks[layer].append(peak[silent].cpu())
                active_peaks[layer].append(peak[~silent].cpu())

    metrics = {"clean_acc": correct / max(1, total), "n_test": total}
    net_spikes = 0.0
    net_pairs = 0
    net_silent = 0.0

    for layer in TARGET_LAYERS:
        bucket = totals[layer]
        n_pairs = max(1, bucket["pairs"])
        active_pairs = bucket["pairs"] - bucket["silent"]
        mean_count = bucket["spikes"] / n_pairs
        variance = max(0.0, bucket["sq"] / n_pairs - mean_count ** 2)
        silent_peak = torch.cat(silent_peaks[layer])
        active_peak = torch.cat(active_peaks[layer])

        metrics.update({
            f"firing_rate_l{layer}": bucket["spikes"] / max(1, bucket["slots"]),
            f"spikes_per_neuron_l{layer}": mean_count,
            f"spikes_per_active_neuron_l{layer}": (
                bucket["spikes"] / max(1.0, active_pairs)),
            f"silent_fraction_l{layer}": bucket["silent"] / n_pairs,
            f"sub_threshold_fraction_l{layer}": bucket["sub_threshold"] / n_pairs,
            f"count_std_l{layer}": variance ** 0.5,
            f"peak_potential_silent_median_l{layer}": (
                float(silent_peak.median()) if silent_peak.numel() else float("nan")),
            f"peak_potential_active_median_l{layer}": (
                float(active_peak.median()) if active_peak.numel() else float("nan")),
        })
        net_spikes += bucket["spikes"]
        net_pairs += bucket["pairs"]
        net_silent += bucket["silent"]

    net_pairs = max(1, net_pairs)
    net_active = max(1.0, net_pairs - net_silent)
    metrics.update({
        "spikes_per_neuron_net": net_spikes / net_pairs,
        "spikes_per_active_neuron_net": net_spikes / net_active,
        "silent_fraction_net": net_silent / net_pairs,
    })
    return metrics


def main() -> None:
    """Measure both baseline arms and write the JSON the notebook reads."""
    if device.type != "cuda":
        raise SystemExit("slayerSNN requires CUDA; no GPU is visible.")

    test_loader = build_test_loader()
    results = {}
    for arm, use_delay in ARMS:
        net = load_baseline(arm, use_delay)
        metrics = evaluate_firing_statistics(net, test_loader)
        metrics["arm"] = arm
        metrics["checkpoint"] = f"jitter_whole_{arm}_trained.pt"
        metrics["theta_margin"] = THETA_MARGIN
        results[arm] = metrics
        print(f"  clean_acc {metrics['clean_acc']:.2%} | "
              f"L1 sp/neuron {metrics['spikes_per_neuron_l1']:.2f}, "
              f"silent {metrics['silent_fraction_l1']:.1%} | "
              f"L2 sp/neuron {metrics['spikes_per_neuron_l2']:.2f}, "
              f"silent {metrics['silent_fraction_l2']:.1%}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / "baseline_firing_statistics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
