"""v3 Phase 0: is v1's relocation endpoint distorted by the padding window?

The SHD ``.mat`` holds 100 time bins and ``load_shd_data`` zero-pads them to the
simulator's 200. Measured on trained checkpoints, 1st-hidden-layer spikes therefore
never occur beyond bin 86, and 90% of them fall inside the first ~39 bins. But
``perturb_hidden_batch`` in the four eval scripts relocates spikes uniformly over
all 200 bins, so at ``f = 1`` roughly 57% of relocated spikes land where no hidden
spike ever naturally occurs. That does not merely randomise timing — it thins the
population's instantaneous spike density by roughly 5x, which is a *rate*-like
insult riding on top of the intended timing-only one.

This script re-runs the ``f = 1`` endpoint twice per checkpoint — once with v1's
full-window relocation and once with destinations restricted to the measured
support — and recomputes the headline Spearman correlation under each, to establish
whether v1's conclusion depends on the artifact.

It is eval-only: checkpoints are loaded frozen, nothing is trained, everything runs
inside ``torch.no_grad()``.

Writes ``v3_analysis/log/relocation_window_check_{arm}.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import spearmanr
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
CKPT_DIR = EXP_DIR / "sn_data"
TRAIN_LOG_DIR = EXP_DIR / "sn_log"
LOG_DIR = SCRIPT_DIR / "log"
MAT_FILE = EXP_DIR / "shd/shd_data/shd_whole.mat"

sys.path.insert(0, str(EXP_DIR))
import slayerSNN as snn  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
TEST_RANGE = (0.75, 0.9)
CHANCE: float = 1.0 / NUM_CLASSES

# One past the last bin in which any hidden spike was observed, measured across all
# 27 checkpoints in both arms by temporal_support.py. (The 6-checkpoint spot check
# that motivated this script put the last occupied bin at 86; over the full set it
# is 87, so the bound is 88.)
SUPPORT_BINS: int = 88

NUM_REPEATS: int = 3


class SparseNetwork(nn.Module):
    """v1's architecture, with the relocation destination window exposed."""

    def __init__(self, delays: bool):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.delays = delays
        self.fc1 = nn.utils.weight_norm(
            slayer.dense(INPUT_DIM, HIDDEN_UNITS), name="weight")
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, HIDDEN_UNITS), name="weight")
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, NUM_CLASSES), name="weight")
        if delays:
            self.delay1 = slayer.delay(HIDDEN_UNITS)
            self.delay2 = slayer.delay(HIDDEN_UNITS)

    def forward(self, x: torch.Tensor, f: float = 0.0,
                window: int | None = None) -> torch.Tensor:
        """Forward pass, optionally relocating 1st-hidden-layer spikes.

        Args:
            x: Input spike trains.
            f: Fraction of each neuron's spikes to relocate. 0 leaves them alone.
            window: Exclusive upper bound on destination bins. ``None`` reproduces
                v1's behaviour of allowing the full simulation window.

        Returns:
            Output spike tensor.
        """
        hidden = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if f > 0:
            hidden = relocate(hidden, f, window)
        if self.delays:
            hidden = self.delay1(hidden)
        out = self.slayer.spike(self.fc2(self.slayer.psp(hidden)))
        if self.delays:
            out = self.delay2(out)
        return self.slayer.spike(self.fc3(self.slayer.psp(out)))


def relocate(hidden_spikes: torch.Tensor, f: float,
             window: int | None = None) -> torch.Tensor:
    """v1's partial spike relocation, with an optional destination window.

    Identical to ``perturb_hidden_batch`` in the four eval scripts except that
    ``window`` may confine relocated spikes to the layer's natural temporal
    support. Per-neuron spike count is preserved exactly either way.

    Args:
        hidden_spikes: Hidden spikes, shape (B, C, 1, 1, T).
        f: Fraction of each neuron's spikes to relocate.
        window: Exclusive upper bound on destination bins, or ``None`` for all of T.

    Returns:
        Perturbed tensor of the same shape, dtype and device.
    """
    batch, channels, height, width, time = hidden_spikes.shape
    flat = hidden_spikes.view(batch, channels, time)
    is_spike = flat > 0.5
    num_to_move = (is_spike.sum(dim=-1, keepdim=True).float() * f).floor().long()

    key = torch.where(is_spike, torch.rand_like(flat), torch.full_like(flat, 2.0))
    remove = key.argsort(dim=-1).argsort(dim=-1) < num_to_move
    keep = is_spike & ~remove

    available = ~keep
    if window is not None:
        available = available.clone()
        available[:, :, window:] = False
    key2 = torch.where(available, torch.rand_like(flat),
                       torch.full_like(flat, 2.0))
    add = key2.argsort(dim=-1).argsort(dim=-1) < num_to_move
    return (keep | add).to(hidden_spikes.dtype).view(
        batch, channels, height, width, time)


@torch.no_grad()
def accuracy(net: SparseNetwork, inputs: np.ndarray, labels: torch.Tensor,
             f: float, window: int | None, seed: int) -> float:
    """Test accuracy at one relocation setting."""
    torch.manual_seed(seed)
    correct = 0
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        predicted = snn.predict.getClass(net(batch, f=f, window=window)).cpu()
        correct += (predicted == labels[start:start + BATCH_SIZE]).sum().item()
    return correct / len(inputs)


def temporal_score(clean_acc: float, perturbed_acc: float) -> float:
    """Chance-corrected, baseline-normalised timing dependence (v1's definition)."""
    return 1.0 - (perturbed_acc - CHANCE) / max(1e-9, clean_acc - CHANCE)


def main() -> None:
    """Compare the two relocation windows across every v1 checkpoint."""
    print(f"Using device: {device}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw = loadmat(str(MAT_FILE))
    features, labels_all = raw["X"], raw["Y"].ravel()
    total = len(labels_all)
    idx = np.arange(int(total * TEST_RANGE[0]), int(total * TEST_RANGE[1]))
    test_inputs = np.zeros((len(idx), features.shape[1], SIM_PARAMS["tSample"]),
                           dtype=np.float32)
    test_inputs[:, :, :features.shape[2]] = features[idx]
    test_labels = torch.from_numpy(labels_all[idx].astype(np.int64))

    for arm, delays in [("nodelay", False), ("delay", True)]:
        summary_path = TRAIN_LOG_DIR / f"sparse_whole_{arm}_train_summary.json"
        with open(summary_path) as handle:
            summary = json.load(handle)

        rows = []
        for run_tag, meta in summary.items():
            net = SparseNetwork(delays).to(device)
            net.load_state_dict(
                torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
            net.eval()

            clean = accuracy(net, test_inputs, test_labels, 0.0, None, 0)
            full = float(np.mean([
                accuracy(net, test_inputs, test_labels, 1.0, None, seed)
                for seed in range(NUM_REPEATS)]))
            restricted = float(np.mean([
                accuracy(net, test_inputs, test_labels, 1.0, SUPPORT_BINS, seed)
                for seed in range(NUM_REPEATS)]))

            rows.append({
                "run_tag": run_tag,
                "spikes_per_neuron": meta["spikes_per_neuron"],
                "silent_fraction": meta["silent_fraction"],
                "clean_acc": clean,
                "acc_f1_full_window": full,
                "acc_f1_support_window": restricted,
                "temporal_score_full_window": temporal_score(clean, full),
                "temporal_score_support_window": temporal_score(clean, restricted),
            })
            print(f"  {run_tag:<44} clean={clean:.3f} "
                  f"f1_full={full:.3f} f1_support={restricted:.3f}", flush=True)

        rates = [row["spikes_per_neuron"] for row in rows]
        print(f"\n=== {arm} (n={len(rows)}) — Spearman rho(spikes/neuron, "
              f"temporal_score) ===")
        for key, label in [("temporal_score_full_window",
                            "relocate over [0,200)  (v1 as run)"),
                           ("temporal_score_support_window",
                            f"relocate over [0,{SUPPORT_BINS})  (support)")]:
            result = spearmanr(rates, [row[key] for row in rows])
            print(f"  {label:<42} rho={result.statistic:+.3f} "
                  f"p={result.pvalue:.4g}")

        out_path = LOG_DIR / f"relocation_window_check_{arm}.json"
        with open(out_path, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"Written to {out_path}\n")


if __name__ == "__main__":
    main()
