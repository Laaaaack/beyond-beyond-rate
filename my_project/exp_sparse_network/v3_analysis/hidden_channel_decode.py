"""v3 Phase 0/1: capacity-matched decoding of the constrained hidden layer(s).

Measures how much of a hidden layer's class information *requires spike timing*,
without perturbing the network and without needing the count channel to be closed.
This replaces the perturbation sweep as the primary measure of the dependent
variable; see ``docs/legacy/sn_progress/sparse_network_test_progress_v3.md`` for why.

Four decoders are fit on the **same** hidden activity, on the train split, and
scored on the test split:

===========  ============================================  ==========  ==========
view         features                                      dimension   immune?
===========  ============================================  ==========  ==========
``COUNT``    per-neuron spike count                        n_neurons   yes
``IDENT``    per-neuron count, binarised                   n_neurons   yes
``FULL``     per-neuron counts in ``N_BINS`` time bins     n_neurons    no
                                                           * N_BINS
``SHUF``     ``FULL`` after resampling each neuron's       n_neurons    yes
             spikes from that neuron's own marginal        * N_BINS
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

Which layers this measures, and why that became a configuration knob
--------------------------------------------------------------------
``GRID`` selects the training generation, and each generation constrained a
different set of hidden layers::

    ""       v1's 27-checkpoint observational gradient   layer 1
    "v3"     the 1st-layer factorial (document 5)        layer 1
    "v3L2"   the 2nd-layer factorial (document 6)        layer 2
    "v3L12"  the both-layer factorial (document 7)       layers 1 and 2

Availability has to be measured at the layers the penalty acted on, so the layer
set follows the grid rather than being fixed at layer 1. One script rather than
three forks, for a reason that is not merely convenience: document 7 §1 point 3's
payoff is a comparison of ``beta_a`` **across** the three grids, and that
comparison is only valid if the dependent variable was measured the same way in
all three. A forked script is a place for the three measurements to drift apart.

Two traps this handles structurally, both from document 7 §5 step 3:

- **``delay1`` moves from downstream to upstream depending on the probe site.**
  A layer-1 probe is right to ignore it; a layer-2 probe must apply it. Here that
  is structural — layer 2 is only reachable through the branch that applies it —
  rather than a matter of remembering.
  [layer2_baseline.py](layer2_baseline.py) is the reference implementation.
- **The two layers have different temporal supports and each needs its own
  binning window.** ``delay1`` (up to 64 bins) pushes layer 2's support out to
  bin 159 in the delay arm, where layer 1's stops at 87 in both arms. Binning a
  layer over a window it never occupies wastes bins on guaranteed zeros and
  coarsens the bins that carry signal; binning it over too *short* a window
  silently discards spikes. See ``SUPPORT_BINS``.

The pooled ``net`` view is an addition, not a replacement
----------------------------------------------------------
When more than one layer is constrained, a third ``net`` view decodes the two
layers' features **concatenated** (256 neurons rather than 128), and reports the
pooled ``a`` and ``s`` that document 7 §2 defines — the axes the both-layer
factorial actually manipulates. It is emitted *alongside* the per-layer views and
never in place of them, per document 7 §5 step 3's third trap.

Read it for what it is. Unlike the ``both`` perturbation site, which is a harsher
insult because layer 2 is damaged *after* being computed from an already-damaged
layer 1, concatenating features compounds nothing: layer 2 is computed from a
clean layer 1 either way. ``net`` is "what a decoder reading the whole hidden
network could recover", not "the network-level analogue of the ``both`` site".

Output schema, and the ``KeyError`` that is deliberate
-------------------------------------------------------
Each checkpoint's row carries one block per view, keyed ``l1`` / ``l2`` / ``net``.
For a **single-layer** grid that view's fields are *also* aliased flat at the top
of the row, so analysis written against the original single-layer output keeps
working unchanged. For a **multi-layer** grid there is no flat alias, so such code
raises ``KeyError`` here rather than silently reading a two-layer file as though
it were a one-layer one. That is the same choice the ``*_bothLayer_evalOnly_*``
sweeps made for their dependent variable, and for the same reason (document 7 §6
point 5).

Reads the selected generation's training summaries for the live checkpoint list.
Writes ``v3_analysis/log/hidden_channel_decode_{tag}{arm}.json``.
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

# --- Scope. QUICK_TEST decodes a few evenly spaced checkpoints per arm. ---
#
# Evenly spaced rather than the first few, because the first few rows of a factorial
# summary are the same cell at different seeds — a pipeline check that never leaves the
# k = 1, floor = 0 corner would miss exactly the failures the corner probe exists for.
QUICK_TEST: bool = False
MAX_QUICK_CHECKPOINTS: int = 4

# Which training generation to decode. The tag is spliced into both the training summary
# that is read and the results file that is written, so no two generations can overwrite
# each other's output — load-bearing now against three prior grids, since v3, v3L2 and
# v3L12 share dataset, arm, k, floor and seed and differ only in the constrained layers
# (document 7 §6 point 4).
#
# v3 checkpoints need no special handling here: the factorials changed the *penalty*,
# not the architecture, so the parameter set and the forward pass are v1's exactly.
# (This is unlike v2.2, whose truncation had to be reapplied at eval.)
GRID: str = "v3L12"

# The hidden layers each grid's penalties acted on, and therefore the layers whose
# availability is the dependent variable for it.
GRID_LAYERS: dict[str, tuple[int, ...]] = {
    "": (1,),        # v1's observational gradient — the Phase 0 measurement
    "v3": (1,),      # 1st-layer factorial, document 5
    "v3L2": (2,),    # 2nd-layer factorial, document 6
    "v3L12": (1, 2),  # both-layer factorial, document 7
}

VERSION_TAG: str = f"{GRID}_" if GRID else ""
LAYERS: tuple[int, ...] = GRID_LAYERS[GRID]

# One decode view per constrained layer, plus the pooled one when there is more than
# one. The pooled view is an addition rather than a replacement (document 7 §5 step 3).
VIEWS: tuple[str, ...] = (tuple(f"l{layer}" for layer in LAYERS)
                          + (("net",) if len(LAYERS) > 1 else ()))

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

# --- Binning window, PER LAYER and PER ARM (measured, then rounded up) ---
#
# shd_whole.mat holds 100 time bins zero-padded to the simulator's 200, so no hidden
# layer ever occupies the whole window and features confined to the measured support
# spend no bins on guaranteed zeros.
#
# Measured last occupied bin, over all 27 v1 checkpoints
# (temporal_support.py and layer2_baseline.py):
#
#     layer 1   87 in both arms                 -> support [0, 88)
#     layer 2   89 no-delay, 159 delay          -> support [0, 90) / [0, 160)
#
# Layer 2's delay-arm support runs to 159 because ``delay1`` shifts layer 1's spikes by
# up to 64 bins before ``fc2`` sees them. Inheriting one layer's bound for the other is
# document 7 §5 step 3's first trap, and in the delay arm the two differ by nearly 2x.
#
# The values below are each support rounded **up** to the next multiple of ``N_BINS``,
# so the reshape that bins them stays even and the extra bins are always empty. The
# perturbation scripts use the exact bounds (88 / 90 / 160) instead, because they place
# spikes rather than bin them and have no divisibility constraint.
SUPPORT_BINS: dict[str, dict[int, int]] = {
    "nodelay": {1: 90, 2: 90},
    "delay": {1: 90, 2: 160},
}
N_BINS: int = 10

# Inverse L2 strength searched per feature set. Each view gets its own best value,
# so a view is never penalised for its dimensionality rather than its content.
DECODER_C_GRID: tuple[float, ...] = (0.003, 0.03, 0.3)
CHANCE: float = 1.0 / NUM_CLASSES
SHUFFLE_SEED: int = 0


class HiddenProbeNetwork(nn.Module):
    """SLAYER SNN exposing each constrained layer's spikes and membrane potential.

    Parameters are identical to the v1 training classes, so v1, v2 and v3 checkpoints
    all load directly — the factorials changed the training penalty, not the
    architecture.

    Note the delay handling. ``delay1`` sits between layer 1 and ``fc2``: it is
    *downstream* of a layer-1 probe, which correctly ignores it, and *upstream* of a
    layer-2 probe, which must apply it. That is structural here rather than a matter of
    remembering, because layer 2 is only reachable through the branch that applies it.
    ``delay2`` is downstream of both probe sites and is never applied.

    Args:
        delays: Whether the checkpoint is from the with-delay arm.
        layers: The hidden layers to probe, a subset of ``(1, 2)``. Passed explicitly
            rather than defaulted to ``LAYERS``, so that a default frozen at
            class-definition time cannot survive a change to ``GRID``.
    """

    def __init__(self, delays: bool, layers: tuple[int, ...]):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.delays = delays
        self.layers = layers
        self.fc1 = nn.utils.weight_norm(
            slayer.dense(INPUT_DIM, HIDDEN_UNITS), name="weight")
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, HIDDEN_UNITS), name="weight")
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(HIDDEN_UNITS, NUM_CLASSES), name="weight")
        if delays:
            self.delay1 = slayer.delay(HIDDEN_UNITS)
            self.delay2 = slayer.delay(HIDDEN_UNITS)

    def probe(self, x: torch.Tensor,
              windows: dict[int, int]) -> dict[int, tuple[torch.Tensor,
                                                          torch.Tensor]]:
        """Return each probed layer's binned spikes and per-neuron peak potential.

        Args:
            x: Input spike trains, shape (B, 700, 1, 1, T).
            windows: This arm's binning window per layer.

        Returns:
            Dict mapping layer index to (binned spikes of shape (B, 128, N_BINS),
            peak membrane potential of shape (B, 128)).
        """
        potential1 = self.fc1(self.slayer.psp(x))
        spikes1 = self.slayer.spike(potential1)

        probed = {}
        if 1 in self.layers:
            probed[1] = (bin_spikes(spikes1, windows[1]), peak_potential(potential1))
        if 2 in self.layers:
            routed = self.delay1(spikes1) if self.delays else spikes1
            potential2 = self.fc2(self.slayer.psp(routed))
            probed[2] = (bin_spikes(self.slayer.spike(potential2), windows[2]),
                         peak_potential(potential2))
        return probed


def bin_spikes(spikes: torch.Tensor, support_bins: int) -> torch.Tensor:
    """Sum a layer's spikes into ``N_BINS`` equal bins over ``[0, support_bins)``.

    Args:
        spikes: SLAYER-format spike tensor, shape (B, C, 1, 1, T).
        support_bins: Exclusive upper bound of the layer's temporal support. Must be a
            multiple of ``N_BINS``, which ``validate_windows`` checks at start-up.

    Returns:
        Binned counts of shape (B, C, N_BINS).
    """
    batch, channels, _, _, time = spikes.shape
    return (spikes.view(batch, channels, time)[:, :, :support_bins]
            .reshape(batch, channels, N_BINS, -1).sum(dim=-1))


def peak_potential(potential: torch.Tensor) -> torch.Tensor:
    """Return each (sample, neuron)'s peak membrane potential over time."""
    batch, channels, _, _, time = potential.shape
    return potential.view(batch, channels, time).max(dim=-1).values


def validate_windows() -> None:
    """Fail at start-up if any configured window cannot be binned evenly.

    A window that is not a multiple of ``N_BINS`` would make the reshape in
    ``bin_spikes`` raise deep inside a batch loop, after minutes of forward passes.

    Raises:
        ValueError: If a window in use is not divisible by ``N_BINS``.
    """
    for arm, _ in ARMS:
        for layer in LAYERS:
            window = SUPPORT_BINS[arm][layer]
            if window % N_BINS:
                raise ValueError(
                    f"SUPPORT_BINS[{arm!r}][{layer}] = {window} is not a multiple of "
                    f"N_BINS = {N_BINS}; round the measured support up to one.")


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
def extract_hidden(net: HiddenProbeNetwork, inputs: np.ndarray,
                   windows: dict[int, int]) -> dict[int, tuple[np.ndarray,
                                                               np.ndarray]]:
    """Run the network over a split and collect binned spikes and peak potentials.

    Args:
        net: The probe network, already loaded and in eval mode.
        inputs: Padded spike trains for one split.
        windows: This arm's binning window per layer.

    Returns:
        Dict mapping layer index to (binned spikes, peak potentials) as numpy arrays.
    """
    collected: dict[int, tuple[list, list]] = {layer: ([], []) for layer in net.layers}
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        for layer, (binned, potential) in net.probe(batch, windows).items():
            collected[layer][0].append(binned.cpu().numpy())
            collected[layer][1].append(potential.cpu().numpy())
    return {layer: (np.concatenate(binned), np.concatenate(potential))
            for layer, (binned, potential) in collected.items()}


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
        model = LogisticRegression(max_iter=2000, C=inverse_strength)
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


def summarise_view(train_binned: np.ndarray, test_binned: np.ndarray,
                   train_shuffled: np.ndarray, test_shuffled: np.ndarray,
                   test_peak: np.ndarray, train_labels: np.ndarray,
                   test_labels: np.ndarray) -> dict:
    """Fit all four decoders on one feature set and derive the timing measures.

    Args:
        train_binned: Train-split binned spikes, shape (N, neurons, N_BINS).
        test_binned: Test-split binned spikes, same neuron count.
        train_shuffled: ``train_binned`` after the capacity-matched resample.
        test_shuffled: ``test_binned`` after the capacity-matched resample.
        test_peak: Test-split peak membrane potentials, shape (N, neurons).
        train_labels: Train-split labels.
        test_labels: Test-split labels.

    Returns:
        Dict of decode accuracies, derived timing measures, both sparsity axes and
        the peak-potential diagnostics.
    """
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
        "n_neurons": int(test_binned.shape[1]),
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
        "peak_potential_silent_median": (float(np.median(test_peak[silent_mask]))
                                         if silent_mask.any() else None),
        "peak_potential_active_median": (float(np.median(test_peak[~silent_mask]))
                                         if (~silent_mask).any() else None),
    }


def analyse_checkpoint(run_tag: str, delays: bool, windows: dict[int, int],
                       train_inputs: np.ndarray, train_labels: np.ndarray,
                       test_inputs: np.ndarray, test_labels: np.ndarray) -> dict:
    """Decode every configured view of one checkpoint through all four decoders.

    Args:
        run_tag: Checkpoint name, without extension.
        delays: Whether the checkpoint is from the with-delay arm.
        windows: This arm's binning window per layer.
        train_inputs: Padded train-split spike trains.
        train_labels: Train-split labels.
        test_inputs: Padded test-split spike trains.
        test_labels: Test-split labels.

    Returns:
        Dict with one block per view (``l1`` / ``l2`` / ``net``). Single-layer grids
        additionally alias that view's fields flat at the top of the row.
    """
    net = HiddenProbeNetwork(delays, LAYERS).to(device)
    net.load_state_dict(torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
    net.eval()

    train_probed = extract_hidden(net, train_inputs, windows)
    test_probed = extract_hidden(net, test_inputs, windows)

    # One rng for the whole checkpoint, seeded identically per checkpoint, so the
    # shuffle null is reproducible and independent of how many layers are probed.
    rng = np.random.default_rng(SHUFFLE_SEED)
    features = {}
    for layer in LAYERS:
        train_binned, _ = train_probed[layer]
        test_binned, test_peak = test_probed[layer]
        profile = train_binned.sum(axis=0)
        features[f"l{layer}"] = (
            train_binned, test_binned,
            resample_from_profile(train_binned, profile, rng),
            resample_from_profile(test_binned, profile, rng),
            test_peak,
        )

    if "net" in VIEWS:
        # The resample is independent per neuron, so shuffling each layer and then
        # concatenating is the same null as concatenating and then shuffling — and it
        # costs one pass rather than two.
        features["net"] = tuple(
            np.concatenate([features[f"l{layer}"][index] for layer in LAYERS], axis=1)
            for index in range(5))

    row = {"grid": GRID, "layers": list(LAYERS), "views": list(VIEWS),
           "support_bins": {f"l{layer}": windows[layer] for layer in LAYERS}}
    for view in VIEWS:
        row[view] = summarise_view(*features[view], train_labels, test_labels)

    # Single-layer grids keep the original flat schema as well, so analysis written
    # against it is unaffected. Multi-layer grids deliberately do not: a reader that
    # assumes the single-layer convention should raise KeyError rather than silently
    # read one layer's number as the network's (document 7 §6 point 5).
    if len(VIEWS) == 1:
        row.update(row[VIEWS[0]])
    return row


def print_checkpoint_row(run_tag: str, row: dict) -> None:
    """Print one line per view for a decoded checkpoint."""
    for view in VIEWS:
        measured = row[view]
        print(f"{run_tag:<44} {view:>4} {measured['spikes_per_neuron']:>7.2f} "
              f"{measured['spikes_per_active_neuron']:>7.2f} "
              f"{measured['silent_fraction']:>7.2%} {measured['decode_count']:>7.3f} "
              f"{measured['decode_identity']:>7.3f} {measured['decode_full']:>7.3f} "
              f"{measured['decode_shuffled']:>7.3f} "
              f"{measured['timing_information']:>+7.3f} "
              f"{measured['timing_fraction']:>6.2f}", flush=True)


def select_run_tags(run_tags: list[str]) -> list[str]:
    """Return the checkpoints to decode, thinned when ``QUICK_TEST`` is set.

    Evenly spaced rather than the first few: consecutive rows of a factorial summary
    are the same ``(k, floor)`` cell at different seeds, so a first-N probe would never
    leave one corner of the grid.

    Args:
        run_tags: Every checkpoint in the arm's training summary, in summary order.

    Returns:
        The subset to decode.
    """
    if not QUICK_TEST or len(run_tags) <= MAX_QUICK_CHECKPOINTS:
        return run_tags
    indices = np.linspace(0, len(run_tags) - 1, MAX_QUICK_CHECKPOINTS).round()
    return [run_tags[int(index)] for index in indices]


def main() -> None:
    """Decode every checkpoint of the selected grid, in both arms."""
    validate_windows()
    print(f"Using device: {device} | QUICK_TEST={QUICK_TEST}")
    print(f"grid: {GRID or 'v1'} | layers: {list(LAYERS)} | views: {list(VIEWS)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw = loadmat(str(MAT_FILE))
    features, labels = raw["X"], raw["Y"].ravel()
    train_inputs, train_labels = load_split(features, labels, TRAIN_RANGE)
    test_inputs, test_labels = load_split(features, labels, TEST_RANGE)

    for arm, delays in ARMS:
        summary_path = (TRAIN_LOG_DIR
                        / f"sparse_whole_{arm}_{VERSION_TAG}train_summary.json")
        with open(summary_path) as handle:
            run_tags = select_run_tags(list(json.load(handle)))
        windows = SUPPORT_BINS[arm]

        print(f"\n=== {arm}: {len(run_tags)} checkpoints ===")
        print("  windows: " + " | ".join(f"l{layer} [0,{windows[layer]})"
                                         for layer in LAYERS))
        print(f"{'run_tag':<44} {'view':>4} {'sp/neu':>7} {'sp/act':>7} {'silent':>7} "
              f"{'COUNT':>7} {'IDENT':>7} {'FULL':>7} {'SHUF':>7} "
              f"{'TIMING':>7} {'frac':>6}")

        results = {}
        for run_tag in run_tags:
            row = analyse_checkpoint(run_tag, delays, windows, train_inputs,
                                     train_labels, test_inputs, test_labels)
            results[run_tag] = row
            print_checkpoint_row(run_tag, row)

        out_path = LOG_DIR / f"hidden_channel_decode_{VERSION_TAG}{arm}.json"
        with open(out_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
