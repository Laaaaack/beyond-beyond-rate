"""2nd-layer v3 calibration step 0: what layer 2 looks like before it is constrained.

The 2nd-layer factorial reuses the mechanism validated for layer 1 — a per-pair spike
ceiling ``relu(count - k)`` crossed with a membrane-potential floor
``relu(margin - max_t u)`` — but **none of its four calibrated constants can be
inherited**, because every one of them was measured against layer 1's activity. This
script measures the layer-2 equivalents on the frozen v1 checkpoints, so the 2nd-layer
grid is calibrated on evidence rather than on the assumption that the two layers behave
alike.

What it measures, and which constant each one sizes
---------------------------------------------------
=================================  =========================================
measurement                        what it settles
=================================  =========================================
``spikes_per_neuron`` at layer 2   ``NATURAL_SPIKES_PER_NEURON`` — the
                                   "not inflated past natural" acceptance
                                   check. Layer 1's 7.85 / 11.66 are the
                                   wrong numbers for layer 2.
``a`` and ``s`` at layer 2, and    the observational confound this factorial
``rho(a, s)`` across v1's grid     exists to break, and the reference the
                                   design acceptance check is read against
                                   (layer 1: -0.943 no-delay, -0.455 delay)
peak potential of silent vs        whether the §2e mechanism argument holds
active pairs                       here — the floor works *because* silent
                                   pairs sit far below threshold where no
                                   count-based penalty can reach them
temporal support                   the window bound for the relocation and
                                   jitter perturbations, which at layer 1 is
                                   ``[0, 88)``. Layer 2 is downstream of
                                   ``delay1`` (up to 64 bins in the delay
                                   arm), so its support must be measured, not
                                   assumed.
fraction of samples whose entire   a failure mode with **no layer-1
layer-1 activity is silent         analogue** — see below
=================================  =========================================

The one genuinely new risk: gradient reachability
--------------------------------------------------
The floor penalty works at layer 1 because ``potential1 = fc1(psp(x))`` is an ordinary
convolution of the *input*, so ``du/dW1`` never vanishes however far below threshold the
neuron sits. At layer 2::

    potential2 = fc2(psp(delay1(spike(potential1))))

``d potential2 / d W2 = psp(delay1(hidden1))``, which is still surrogate-free — so the
mechanism argument survives, and the floor can lift a silent layer-2 pair by growing
``fc2``'s weights. **But it is zero when that sample's entire layer 1 is silent.** Such
a pair is unreachable except through ``fc1``, i.e. through exactly the surrogate
gradient the floor was designed to avoid. This has no analogue at layer 1, where the
input is data and is never silent, so it is measured before it can surprise the grid.

Eval-only: reads the frozen v1 checkpoints, trains nothing. Writes
``v3_analysis/log/layer2_baseline_{arm}.json``.
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

# v1's 27-checkpoint sparsity gradient, both arms. These are the unconstrained-to-
# heavily-constrained models whose *layer 2* has never been looked at.
VERSION_TAG: str = ""
ARMS: tuple[tuple[str, bool], ...] = (("nodelay", False), ("delay", True))

# --- Architecture and simulation, identical to training ---
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

# Measured on the test split, matching the training scripts' own clean evaluation.
TEST_RANGE = (0.75, 0.9)

# Percentiles of the spike-time distribution to record, for the perturbation window.
SUPPORT_PERCENTILES: tuple[int, ...] = (50, 90, 99)


class TwoLayerProbeNetwork(nn.Module):
    """SLAYER SNN exposing **both** hidden layers' spikes and membrane potentials.

    Parameters are identical to the v1 training classes, so v1 checkpoints load
    directly.

    Note the delay handling, which differs from the 1st-layer probe. ``delay1`` sits
    between layer 1 and ``fc2``, so for a layer-2 measurement it is **upstream of the
    probe site and must be applied** — where a layer-1 probe correctly ignores it.
    ``delay2`` sits downstream of layer 2 and is never applied here.
    """

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

    def probe(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return both layers' spike trains and peak membrane potentials.

        Args:
            x: Input spike trains, shape (B, 700, 1, 1, T).

        Returns:
            Dict with ``spikes1``/``spikes2`` of shape (B, 128, T) and
            ``peak1``/``peak2`` of shape (B, 128).
        """
        potential1 = self.fc1(self.slayer.psp(x))
        spikes1 = self.slayer.spike(potential1)

        routed = self.delay1(spikes1) if self.delays else spikes1
        potential2 = self.fc2(self.slayer.psp(routed))
        spikes2 = self.slayer.spike(potential2)

        batch, channels, _, _, time = spikes1.shape
        flatten = lambda tensor: tensor.view(batch, channels, time)  # noqa: E731
        return {
            "spikes1": flatten(spikes1),
            "spikes2": flatten(spikes2),
            "peak1": flatten(potential1).max(dim=-1).values,
            "peak2": flatten(potential2).max(dim=-1).values,
        }


def load_split(features: np.ndarray, labels: np.ndarray,
               split_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return one zero-padded fractional split of the dataset."""
    total = len(labels)
    idx = np.arange(int(total * split_range[0]), int(total * split_range[1]))
    padded = np.zeros((len(idx), features.shape[1], SIM_PARAMS["tSample"]),
                      dtype=np.float32)
    padded[:, :, :features.shape[2]] = features[idx]
    return padded, labels[idx].astype(np.int64)


def summarise_layer(counts: np.ndarray, peak: np.ndarray,
                    bin_occupancy: np.ndarray) -> dict:
    """Reduce one layer's raw activity to the numbers that size the constraint.

    Args:
        counts: Per-(sample, neuron) spike counts, shape (N, 128).
        peak: Per-(sample, neuron) peak membrane potential, shape (N, 128).
        bin_occupancy: Total spikes falling in each time bin, shape (T,).

    Returns:
        Dict carrying both sparsity axes, the peak-potential split that sizes the
        floor, and the temporal support that bounds the perturbation window.
    """
    silent = counts == 0
    silent_fraction = float(silent.mean())
    spikes_per_neuron = float(counts.mean())

    occupied = np.flatnonzero(bin_occupancy > 0)
    cumulative = np.cumsum(bin_occupancy)
    total = cumulative[-1]
    percentiles = {
        f"support_p{percentile}": (
            int(np.searchsorted(cumulative, total * percentile / 100.0))
            if total > 0 else None)
        for percentile in SUPPORT_PERCENTILES
    }

    return {
        "spikes_per_neuron": spikes_per_neuron,
        "spikes_per_active_neuron": spikes_per_neuron / max(1e-9,
                                                            1.0 - silent_fraction),
        "silent_fraction": silent_fraction,
        "peak_potential_silent_median": (float(np.median(peak[silent]))
                                         if silent.any() else None),
        "peak_potential_active_median": (float(np.median(peak[~silent]))
                                         if (~silent).any() else None),
        "support_last_bin": int(occupied[-1]) if occupied.size else None,
        **percentiles,
    }


@torch.no_grad()
def analyse_checkpoint(run_tag: str, delays: bool,
                       test_inputs: np.ndarray) -> dict:
    """Measure both layers of one checkpoint over the test split.

    Args:
        run_tag: Checkpoint name, without extension.
        delays: Whether the checkpoint is from the with-delay arm.
        test_inputs: Padded test-split spike trains.

    Returns:
        Dict with a ``layer1`` and a ``layer2`` block, plus the gradient-reachability
        diagnostic that only applies to layer 2.
    """
    net = TwoLayerProbeNetwork(delays).to(device)
    net.load_state_dict(torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
    net.eval()

    counts = {"spikes1": [], "spikes2": []}
    peaks = {"peak1": [], "peak2": []}
    occupancy = {"spikes1": None, "spikes2": None}

    for start in range(0, len(test_inputs), BATCH_SIZE):
        batch = torch.from_numpy(test_inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        probed = net.probe(batch)
        for name in counts:
            counts[name].append(probed[name].sum(dim=-1).cpu().numpy())
            per_bin = probed[name].sum(dim=(0, 1)).cpu().numpy()
            occupancy[name] = (per_bin if occupancy[name] is None
                               else occupancy[name] + per_bin)
        for name in peaks:
            peaks[name].append(probed[name].cpu().numpy())

    counts = {name: np.concatenate(values) for name, values in counts.items()}
    peaks = {name: np.concatenate(values) for name, values in peaks.items()}

    # A layer-2 pair whose entire layer-1 input is silent receives no gradient through
    # fc2, so the floor cannot lift it without going back through the surrogate. Layer 1
    # has no analogue: its input is data.
    samples_with_silent_layer1 = int((counts["spikes1"].sum(axis=1) == 0).sum())

    return {
        "layer1": summarise_layer(counts["spikes1"], peaks["peak1"],
                                  occupancy["spikes1"]),
        "layer2": summarise_layer(counts["spikes2"], peaks["peak2"],
                                  occupancy["spikes2"]),
        "n_samples": int(len(counts["spikes1"])),
        "samples_with_silent_layer1": samples_with_silent_layer1,
        "fraction_samples_unreachable": (samples_with_silent_layer1
                                         / max(1, len(counts["spikes1"]))),
    }


def report_axis_correlation(results: dict[str, dict], arm: str) -> None:
    """Print rho(a, s) at both layers, i.e. the confound this factorial must break.

    Args:
        results: Per-checkpoint measurements for one arm.
        arm: Arm name, for the heading.
    """
    print(f"\n  rho(a, s) across v1's {arm} gradient (n={len(results)}):")
    for layer in ("layer1", "layer2"):
        a_axis = [row[layer]["spikes_per_active_neuron"] for row in results.values()]
        s_axis = [row[layer]["silent_fraction"] for row in results.values()]
        outcome = spearmanr(a_axis, s_axis)
        print(f"    {layer}: rho = {outcome.statistic:+.3f}  "
              f"(p = {outcome.pvalue:.4g})")


def main() -> None:
    """Measure layer 2 on every v1 checkpoint in both arms."""
    print(f"Using device: {device}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw = loadmat(str(MAT_FILE))
    features, labels = raw["X"], raw["Y"].ravel()
    test_inputs, _ = load_split(features, labels, TEST_RANGE)

    for arm, delays in ARMS:
        summary_path = (TRAIN_LOG_DIR
                        / f"sparse_whole_{arm}_{VERSION_TAG}train_summary.json")
        with open(summary_path) as handle:
            run_tags = list(json.load(handle))

        print(f"\n{'=' * 78}")
        print(f"=== {arm}: {len(run_tags)} checkpoints ===")
        print(f"{'run_tag':<40} {'L1 sp/neu':>9} {'L1 a':>6} {'L1 s':>7} "
              f"{'L2 sp/neu':>9} {'L2 a':>6} {'L2 s':>7} {'L2 last':>7}")

        results = {}
        for run_tag in run_tags:
            row = analyse_checkpoint(run_tag, delays, test_inputs)
            results[run_tag] = row
            layer1, layer2 = row["layer1"], row["layer2"]
            print(f"{run_tag:<40} {layer1['spikes_per_neuron']:>9.2f} "
                  f"{layer1['spikes_per_active_neuron']:>6.2f} "
                  f"{layer1['silent_fraction']:>7.2%} "
                  f"{layer2['spikes_per_neuron']:>9.2f} "
                  f"{layer2['spikes_per_active_neuron']:>6.2f} "
                  f"{layer2['silent_fraction']:>7.2%} "
                  f"{str(layer2['support_last_bin']):>7}", flush=True)

        report_axis_correlation(results, arm)

        # The constants the 2nd-layer training scripts need.
        layer2_rates = [row["layer2"]["spikes_per_neuron"] for row in results.values()]
        layer2_last = [row["layer2"]["support_last_bin"] for row in results.values()
                       if row["layer2"]["support_last_bin"] is not None]
        silent_peaks = [row["layer2"]["peak_potential_silent_median"]
                        for row in results.values()
                        if row["layer2"]["peak_potential_silent_median"] is not None]
        unreachable = [row["fraction_samples_unreachable"]
                       for row in results.values()]

        print(f"\n  NATURAL_SPIKES_PER_NEURON (layer 2) = {max(layer2_rates):.2f}  "
              f"[range {min(layer2_rates):.2f}-{max(layer2_rates):.2f}]")
        if layer2_last:
            print(f"  temporal support: last occupied bin = {max(layer2_last)}  "
                  f"-> window [0, {max(layer2_last) + 1})")
        if silent_peaks:
            print(f"  peak potential of SILENT layer-2 pairs: median "
                  f"{min(silent_peaks):.2f}-{max(silent_peaks):.2f} "
                  f"against theta = {LIF_PARAMS['theta']}")
        print(f"  samples with an entirely silent layer 1 (floor unreachable): "
              f"{max(unreachable):.2%} worst case")

        out_path = LOG_DIR / f"layer2_baseline_{arm}.json"
        with open(out_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
