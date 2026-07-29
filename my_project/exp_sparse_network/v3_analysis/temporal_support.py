"""v3 Phase 0: measure the 1st hidden layer's temporal support.

Every timing perturbation in this project relocates or displaces hidden spikes
inside a window, and until now that window was the full simulation length,
``T = 200``. But ``shd_whole.mat`` holds only 100 time bins and ``load_shd_data``
zero-pads them to 200, so the hidden layer's activity occupies far less than the
window the perturbations draw from. Moving spikes into the unoccupied tail thins
the population's instantaneous spike density, which is a *rate* insult riding on
what is supposed to be a timing-only probe.

The other v3 scripts hard-code the support bound as a constant. This script is
where that constant comes from: it runs every checkpoint in both arms' training
summaries over the test split and reports, per checkpoint, the last bin in which
any 1st-hidden spike occurs together with the bins holding the 50th, 90th and 99th
percentile of the spike mass. The bound the perturbations should use is one past
the largest ``last_spike_bin`` observed anywhere.

Eval-only: checkpoints are loaded frozen and everything runs under
``torch.no_grad()``.

Writes ``v3_analysis/log/temporal_support.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
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

PERCENTILES: tuple[float, ...] = (50.0, 90.0, 99.0)


class FirstHiddenProbe(nn.Module):
    """v1's architecture, exposing only the 1st hidden layer's spikes.

    The parameter set matches the v1 training classes exactly, so v1 checkpoints
    load directly. ``delays`` builds the with-delay variant's ``delay1``/``delay2``
    so its checkpoints load, but they sit downstream of the probe site and are
    never applied here.
    """

    def __init__(self, delays: bool):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.fc1 = nn.utils.weight_norm(
            slayer.dense(INPUT_DIM, HIDDEN_UNITS), name="weight")
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, HIDDEN_UNITS), name="weight")
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, NUM_CLASSES), name="weight")
        if delays:
            self.delay1 = slayer.delay(HIDDEN_UNITS)
            self.delay2 = slayer.delay(HIDDEN_UNITS)

    def first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1st-hidden spikes of shape (B, C, T)."""
        spikes = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        batch, channels, _, _, time = spikes.shape
        return spikes.view(batch, channels, time)


def load_test_split() -> np.ndarray:
    """Return the zero-padded test-split input spike trains."""
    raw = loadmat(str(MAT_FILE))
    features, labels = raw["X"], raw["Y"].ravel()
    total = len(labels)
    idx = np.arange(int(total * TEST_RANGE[0]), int(total * TEST_RANGE[1]))
    padded = np.zeros((len(idx), features.shape[1], SIM_PARAMS["tSample"]),
                      dtype=np.float32)
    padded[:, :, :features.shape[2]] = features[idx]
    return padded


@torch.no_grad()
def spike_histogram(net: FirstHiddenProbe, inputs: np.ndarray) -> np.ndarray:
    """Return the total 1st-hidden spike count per time bin, shape (T,)."""
    histogram = torch.zeros(SIM_PARAMS["tSample"], device=device)
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        histogram += net.first_hidden(batch).sum(dim=(0, 1))
    return histogram.cpu().numpy()


def summarise(histogram: np.ndarray) -> dict:
    """Reduce a per-bin spike histogram to its support statistics.

    Args:
        histogram: Total spike count per time bin, shape (T,).

    Returns:
        Dict with the last occupied bin and the bins at each mass percentile.
    """
    total = histogram.sum()
    occupied = np.flatnonzero(histogram > 0)
    cumulative = np.cumsum(histogram) / max(1e-9, total)
    return {
        "total_spikes": float(total),
        "last_spike_bin": int(occupied[-1]) if len(occupied) else -1,
        **{f"bin_p{int(pct)}": int(np.searchsorted(cumulative, pct / 100.0))
           for pct in PERCENTILES},
    }


def main() -> None:
    """Measure the temporal support of every checkpoint in both arms."""
    print(f"Using device: {device}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    test_inputs = load_test_split()

    results: dict[str, dict] = {}
    for arm, delays in [("nodelay", False), ("delay", True)]:
        summary_path = TRAIN_LOG_DIR / f"sparse_whole_{arm}_train_summary.json"
        with open(summary_path) as handle:
            run_tags = list(json.load(handle))

        print(f"\n=== {arm}: {len(run_tags)} checkpoints ===")
        print(f"{'run_tag':<44} {'last_bin':>9} {'p50':>5} {'p90':>5} {'p99':>5}")
        for run_tag in run_tags:
            net = FirstHiddenProbe(delays).to(device)
            net.load_state_dict(
                torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
            net.eval()
            row = summarise(spike_histogram(net, test_inputs))
            row["arm"] = arm
            results[run_tag] = row
            print(f"{run_tag:<44} {row['last_spike_bin']:>9} "
                  f"{row['bin_p50']:>5} {row['bin_p90']:>5} "
                  f"{row['bin_p99']:>5}", flush=True)

    last_bins = [row["last_spike_bin"] for row in results.values()]
    print(f"\nMax last_spike_bin over all {len(results)} checkpoints: "
          f"{max(last_bins)}")
    print(f"=> perturbation destinations should be confined to "
          f"[0, {max(last_bins) + 1})")

    out_path = LOG_DIR / "temporal_support.json"
    with open(out_path, "w") as handle:
        json.dump({"per_checkpoint": results,
                   "max_last_spike_bin": int(max(last_bins)),
                   "recommended_support_bins": int(max(last_bins) + 1)},
                  handle, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
