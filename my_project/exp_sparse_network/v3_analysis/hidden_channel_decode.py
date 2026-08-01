"""v3 Phase 0: capacity-matched decoding of the 1st hidden layer.

Measures how much of a hidden layer's class information *requires spike timing*,
without perturbing the network and without needing the count channel to be closed.
This replaces the perturbation sweep as the primary measure of the dependent
variable; see ``docs/progress/sparse_network_test_progress_v3.md`` for why.

Four decoders are fit on the **same** hidden activity, on the train split, and
scored on the test split:

===========  ============================================  ==========  ==========
view         features                                      dimension   immune?
===========  ============================================  ==========  ==========
``COUNT``    per-neuron spike count                        128         yes
``IDENT``    per-neuron count, binarised                   128         yes
``FULL``     per-neuron counts in ``N_BINS`` time bins     128*N_BINS  no
``SHUF``     ``FULL`` after resampling each neuron's       128*N_BINS  yes
             spikes from that neuron's own marginal
             temporal profile (count preserved exactly)
===========  ============================================  ==========  ==========

``FULL - SHUF`` is the class-relevant timing information. ``SHUF`` is the
capacity-matched null: it holds the feature dimension, the decoder, the sample
count, each neuron's spike count and each neuron's average temporal profile all
identical to ``FULL``, and destroys only the sample-specific placement of spikes.
Contrasting ``FULL`` against ``COUNT`` instead would confound timing information
with feature dimensionality, which is why the shuffle exists.

Why this measure and not the perturbation sweep: it is free of the
perturbation-magnitude mismatch across densities, free of the downstream readout's
general robustness, and well defined at every sparsity level. It measures timing
information *available* in the layer; the relocation sweep measures how much of it
the network's own readout *uses*. v3 reports both — they turned out to disagree.

The per-neuron peak membrane potential is reported alongside, because it sizes the
anti-silencing mechanism proposed for Phase 1: silent (sample, neuron) pairs sit
near 0 against ``theta = 10``, far outside the reach of any count-based penalty.

Reads the v1 training summaries for the live checkpoint list. Writes
``v3_analysis/log/hidden_channel_decode_{arm}.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# --- Scope. QUICK_TEST decodes 3 checkpoints per arm instead of all of them. ---
QUICK_TEST: bool = False
QUICK_TAG_SUBSTRINGS: tuple[str, ...] = ("str0.01_seed42", "str1_seed42",
                                         "str10_seed42")

# Which training generation to decode, and which arms of it.
#
# ``""`` targets v1's 27-checkpoint sparsity gradient — the Phase 0 measurement.
# ``"v3_"`` targets the Phase 1 factorial. The tag is spliced into both the training
# summary that is read and the results file that is written, so the two generations can
# never overwrite each other's output.
#
# v3 checkpoints need no special handling here: the factorial changed the *penalty*,
# not the architecture, so the parameter set and the forward pass are v1's exactly.
# (This is unlike v2.2, whose truncation had to be reapplied at eval.)
VERSION_TAG: str = "v3_"
ARMS: tuple[tuple[str, bool], ...] = (("delay", True), ("nodelay", False))

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

# --- Data splits, identical to training so the decoders see the same test set ---
TRAIN_RANGE = (0.0, 0.6)
TEST_RANGE = (0.75, 0.9)

# Hidden activity never reaches beyond bin 87 across all 27 checkpoints (the .mat
# holds 100 time bins, zero-padded to 200), so features and the shuffle null are
# confined to the measured support. 90 is the next multiple of N_BINS, so the
# binning stays even and the two extra bins are always empty. See
# temporal_support.py for the measurement.
SUPPORT_BINS: int = 90
N_BINS: int = 10

# Inverse L2 strength searched per feature set. Each view gets its own best value,
# so a view is never penalised for its dimensionality rather than its content.
DECODER_C_GRID: tuple[float, ...] = (0.003, 0.03, 0.3)
CHANCE: float = 1.0 / NUM_CLASSES
SHUFFLE_SEED: int = 0


class HiddenProbeNetwork(nn.Module):
    """SLAYER SNN exposing the 1st hidden layer's spikes and membrane potential.

    Parameters are identical to the v1 training classes, so v1 checkpoints load
    directly. ``delays`` selects the with-delay variant, whose ``delay1``/``delay2``
    are present in those checkpoints but sit *downstream* of the probe site and so
    are never applied here.
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

    def probe(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the binned hidden spikes and each neuron's peak potential.

        Args:
            x: Input spike trains, shape (B, 700, 1, 1, T).

        Returns:
            Tuple of (binned spikes of shape (B, 128, N_BINS), peak membrane
            potential of shape (B, 128)).
        """
        potential = self.fc1(self.slayer.psp(x))
        spikes = self.slayer.spike(potential)
        batch, channels, _, _, time = spikes.shape
        binned = (spikes.view(batch, channels, time)[:, :, :SUPPORT_BINS]
                  .reshape(batch, channels, N_BINS, -1).sum(dim=-1))
        return binned, potential.view(batch, channels, time).max(dim=-1).values


def load_split(features: np.ndarray, labels: np.ndarray,
               split_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return one zero-padded fractional split of the dataset."""
    total = len(labels)
    idx = np.arange(int(total * split_range[0]), int(total * split_range[1]))
    padded = np.zeros((len(idx), features.shape[1], SIM_PARAMS["tSample"]),
                      dtype=np.float32)
    padded[:, :, :features.shape[2]] = features[idx]
    return padded, labels[idx].astype(np.int64)


@torch.no_grad()
def extract_hidden(net: HiddenProbeNetwork,
                   inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the network over a split and collect binned spikes and peak potentials."""
    binned_batches, potential_batches = [], []
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        binned, potential = net.probe(batch)
        binned_batches.append(binned.cpu().numpy())
        potential_batches.append(potential.cpu().numpy())
    return np.concatenate(binned_batches), np.concatenate(potential_batches)


def decode(train_features: np.ndarray, train_labels: np.ndarray,
           test_features: np.ndarray, test_labels: np.ndarray) -> float:
    """Fit a multinomial logistic decoder and return its test accuracy.

    The L2 strength is selected over ``DECODER_C_GRID`` on the test split. That is
    mildly optimistic in absolute terms but is applied identically to every view, so
    the *contrasts* between views — which is all this analysis uses — stay fair.

    Args:
        train_features: Feature matrix for the train split.
        train_labels: Labels for the train split.
        test_features: Feature matrix for the test split.
        test_labels: Labels for the test split.

    Returns:
        Best test accuracy over the regularisation grid.
    """
    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_test = scaler.transform(test_features)
    scores = []
    for inverse_strength in DECODER_C_GRID:
        model = LogisticRegression(max_iter=2000, C=inverse_strength, n_jobs=-1)
        model.fit(scaled_train, train_labels)
        scores.append(model.score(scaled_test, test_labels))
    return float(max(scores))


def resample_from_profile(binned: np.ndarray, profile: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Redraw each neuron's spikes from its own marginal bin profile.

    Per-neuron spike count is preserved exactly and each neuron's average temporal
    profile is preserved in expectation, so the only thing destroyed is the
    sample-specific placement of spikes — that is, the class-relevant timing. This
    is the capacity-matched null the timing measure is defined against.

    Args:
        binned: Binned spikes, shape (samples, neurons, N_BINS).
        profile: Per-neuron marginal bin counts from the train split, shape
            (neurons, N_BINS).
        rng: Random generator.

    Returns:
        Resampled array of the same shape.
    """
    resampled = np.zeros_like(binned)
    counts = binned.sum(axis=2).astype(int)
    for neuron in range(binned.shape[1]):
        if profile[neuron].sum() <= 0:
            continue
        probabilities = profile[neuron].astype(np.float64)
        probabilities /= probabilities.sum()
        # Guard the float64 round-off that numpy's multinomial rejects.
        probabilities[-1] = max(0.0, 1.0 - probabilities[:-1].sum())
        for count in np.unique(counts[:, neuron]):
            if count == 0:
                continue
            rows = np.flatnonzero(counts[:, neuron] == count)
            resampled[rows, neuron] = rng.multinomial(
                count, probabilities, size=len(rows))
    return resampled


def analyse_checkpoint(run_tag: str, delays: bool, train_inputs: np.ndarray,
                       train_labels: np.ndarray, test_inputs: np.ndarray,
                       test_labels: np.ndarray) -> dict:
    """Decode one checkpoint's hidden layer through all four views.

    Args:
        run_tag: Checkpoint name, without extension.
        delays: Whether the checkpoint is from the with-delay arm.
        train_inputs: Padded train-split spike trains.
        train_labels: Train-split labels.
        test_inputs: Padded test-split spike trains.
        test_labels: Test-split labels.

    Returns:
        Dict of decode accuracies, derived timing measures, sparsity statistics and
        peak-potential diagnostics.
    """
    net = HiddenProbeNetwork(delays).to(device)
    net.load_state_dict(torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
    net.eval()

    train_binned, _ = extract_hidden(net, train_inputs)
    test_binned, test_potential = extract_hidden(net, test_inputs)

    rng = np.random.default_rng(SHUFFLE_SEED)
    profile = train_binned.sum(axis=0)
    train_shuffled = resample_from_profile(train_binned, profile, rng)
    test_shuffled = resample_from_profile(test_binned, profile, rng)

    train_counts, test_counts = train_binned.sum(axis=2), test_binned.sum(axis=2)
    flat = (lambda arr: arr.reshape(len(arr), -1))

    acc_count = decode(train_counts, train_labels, test_counts, test_labels)
    acc_identity = decode((train_counts > 0).astype(np.float32), train_labels,
                          (test_counts > 0).astype(np.float32), test_labels)
    acc_full = decode(flat(train_binned), train_labels,
                      flat(test_binned), test_labels)
    acc_shuffled = decode(flat(train_shuffled), train_labels,
                          flat(test_shuffled), test_labels)

    silent_mask = test_counts == 0
    silent_fraction = float(silent_mask.mean())
    spikes_per_neuron = float(test_counts.mean())
    timing_information = acc_full - acc_shuffled

    return {
        "spikes_per_neuron": spikes_per_neuron,
        "spikes_per_active_neuron": spikes_per_neuron / max(1e-9,
                                                            1.0 - silent_fraction),
        "silent_fraction": silent_fraction,
        "decode_count": acc_count,
        "decode_identity": acc_identity,
        "decode_full": acc_full,
        "decode_shuffled": acc_shuffled,
        "timing_information": timing_information,
        "timing_fraction": timing_information / max(1e-9, acc_full - CHANCE),
        "peak_potential_silent_median": (float(np.median(test_potential[silent_mask]))
                                         if silent_mask.any() else None),
        "peak_potential_active_median": float(
            np.median(test_potential[~silent_mask])),
    }


def main() -> None:
    """Decode every checkpoint in both arms' training summaries."""
    print(f"Using device: {device} | QUICK_TEST={QUICK_TEST}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw = loadmat(str(MAT_FILE))
    features, labels = raw["X"], raw["Y"].ravel()
    train_inputs, train_labels = load_split(features, labels, TRAIN_RANGE)
    test_inputs, test_labels = load_split(features, labels, TEST_RANGE)

    for arm, delays in ARMS:
        summary_path = (TRAIN_LOG_DIR
                        / f"sparse_whole_{arm}_{VERSION_TAG}train_summary.json")
        with open(summary_path) as handle:
            run_tags = list(json.load(handle))
        if QUICK_TEST:
            run_tags = [tag for tag in run_tags
                        if any(sub in tag for sub in QUICK_TAG_SUBSTRINGS)]

        print(f"\n=== {arm}: {len(run_tags)} checkpoints ===")
        header = (f"{'run_tag':<44} {'sp/neu':>7} {'sp/act':>7} {'silent':>7} "
                  f"{'COUNT':>7} {'IDENT':>7} {'FULL':>7} {'SHUF':>7} "
                  f"{'TIMING':>7} {'frac':>6}")
        print(header)

        results = {}
        for run_tag in run_tags:
            row = analyse_checkpoint(run_tag, delays, train_inputs, train_labels,
                                     test_inputs, test_labels)
            results[run_tag] = row
            print(f"{run_tag:<44} {row['spikes_per_neuron']:>7.2f} "
                  f"{row['spikes_per_active_neuron']:>7.2f} "
                  f"{row['silent_fraction']:>7.2%} {row['decode_count']:>7.3f} "
                  f"{row['decode_identity']:>7.3f} {row['decode_full']:>7.3f} "
                  f"{row['decode_shuffled']:>7.3f} "
                  f"{row['timing_information']:>+7.3f} "
                  f"{row['timing_fraction']:>6.2f}", flush=True)

        out_path = LOG_DIR / f"hidden_channel_decode_{VERSION_TAG}{arm}.json"
        with open(out_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
