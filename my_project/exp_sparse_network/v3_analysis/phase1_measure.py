"""v3 Phase 1: the *usage* measure and its general-robustness control, per checkpoint.

Companion to [hidden_channel_decode.py](hidden_channel_decode.py), which supplies the
*availability* measure. Between them these are the two dependent variables of the
Phase 1 regressions; [phase1_regress.py](phase1_regress.py) joins them and fits.

Why this is a separate script from the v1 eval sweeps
-----------------------------------------------------
``{shd,deletion}/`` already sweep these perturbations, but they sweep the whole grid of
``f`` / ``p_d`` values over v1's checkpoint list. Phase 1 needs only the **endpoint** of
each sweep, on the v3 checkpoint list, and it needs both perturbations on the same
forward pass budget. Doing that here costs one pass over each checkpoint instead of two
script invocations, and keeps the v1 eval scripts untouched.

What is measured, per checkpoint and per injection site:

===================  =====================================================
``usage``            ``temporal_score`` at relocation ``f = 1``: 1 - the
                     chance-corrected fraction of accuracy surviving when
                     hidden spike *timing* is destroyed and per-neuron
                     *count* is preserved exactly. Higher = the readout
                     leans harder on timing.
``control``          the same score under ``p_d = 0.8`` spike **deletion**,
                     which destroys rate rather than timing. This is the
                     general-robustness covariate, and Phase 0 showed it is
                     not optional: in the no-delay arm it tracked sparsity
                     at +0.936 against the timing probe's +0.939.
===================  =====================================================

Where the perturbation is injected, and why that is now a configuration knob
-----------------------------------------------------------------------------
``GRID`` selects the training generation, and each generation constrained a different
set of hidden layers — so usage has to be measured at the layers the penalty acted on::

    "v3"     the 1st-layer factorial (document 5)     site "l1"
    "v3L2"   the 2nd-layer factorial (document 6)     site "l2"
    "v3L12"  the both-layer factorial (document 7)    sites "l1", "l2", "both"

One script rather than three forks, and not merely for convenience: document 7 §1
point 3's payoff is a comparison of ``beta_a`` **across** the three grids, which is
only valid if the dependent variable was measured the same way in all three.

**For the both-layer grid the pooled site is an addition, not a replacement**
(document 7 §5 step 3, third trap). Perturbing both layers at once is a *different* and
harsher insult than perturbing either — at the ``both`` site layer 2 is damaged after
being computed from an already-damaged layer 1 — and the sweeps confirmed the three
sites do not reduce to one another: relocation saturates (``both`` ~ ``l1``) while
deletion compounds hard (document 7 §6e). So ``l1`` and ``l2`` are reported alongside
``both`` rather than replaced by it, and this script must share the sweeps' injection
sites and per-layer windows for the two to agree.

Two traps this handles structurally, both from document 7 §5 step 3:

- **``delay1`` moves from downstream to upstream depending on the site.** It is applied
  inside ``_second_hidden``, which puts it downstream of a layer-1 injection and
  upstream of a layer-2 one. Getting this wrong silently measures the wrong tensor
  rather than raising.
- **The window correction is not uniform across perturbations.** Relocation chooses a
  destination bin and must respect the perturbed layer's own support; **deletion must
  be left alone**, because it chooses no destination. Clipping it would be meaningless
  at best. Each layer's support differs: ``[0, 88)`` at layer 1 in both arms against
  ``[0, 90)`` / ``[0, 160)`` at layer 2, since ``delay1`` shifts layer 1's spikes by up
  to 64 bins before ``fc2`` sees them. Drawing destinations from the full 200-bin
  window — as v1 did — sends most relocated spikes into the zero-padded tail, thinning
  the population's instantaneous spike density and mixing a *rate* insult into what
  must be a timing-only probe. See ``temporal_support.py`` for the layer-1 measurement,
  ``layer2_baseline.py`` for layer 2's, and document 5 §2a for the size of the artifact.

Output schema, and the ``KeyError`` that is deliberate
-------------------------------------------------------
Every score is keyed by its site — ``usage_l1``, ``control_deletion_both``, and so on.
For a **single-site** grid those fields are *also* aliased unsuffixed, so analysis
written against the original single-layer output keeps working unchanged. For a
**multi-site** grid there is no unsuffixed alias, so such code raises ``KeyError`` here
rather than silently reading a three-site file as though it were a one-site one. That
is the same choice the ``*_bothLayer_evalOnly_*`` sweeps made, and for the same reason
(document 7 §6 point 5).

Eval-only: checkpoints are loaded frozen, nothing is trained, everything runs inside
``torch.no_grad()``.

Writes ``v3_analysis/log/phase1_measure_{tag}{arm}.json``. The tag is load-bearing:
without it this script would overwrite the 1st-layer grid's measurements, since v3,
v3L2 and v3L12 share dataset, arm, ``k``, floor and seed (document 7 §6 point 4).
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
CHANCE: float = 1.0 / NUM_CLASSES

# Which training generation to measure, and therefore where to inject.
GRID: str = "v3L12"

# The injection sites each grid needs. "l1" and "l2" perturb one hidden layer; "both"
# perturbs each in turn, so the layer-2 insult lands on a layer that was itself computed
# from an already-perturbed layer 1. That is what makes "both" a harsher probe than
# either single site rather than their sum.
GRID_SITES: dict[str, tuple[str, ...]] = {
    "v3": ("l1",),                  # 1st-layer factorial, document 5
    "v3L2": ("l2",),                # 2nd-layer factorial, document 6
    "v3L12": ("l1", "l2", "both"),  # both-layer factorial, document 7
}

VERSION_TAG: str = f"{GRID}_" if GRID else ""
SITES: tuple[str, ...] = GRID_SITES[GRID]

# One line per site for the printed table, so a reader never has to infer what "both"
# did from the field name alone.
SITE_DESCRIPTIONS: dict[str, str] = {
    "l1": "injected at the 1st hidden layer only",
    "l2": "injected at the 2nd hidden layer only",
    "both": "injected at both layers -- layer 2 is perturbed after being computed "
            "from an already-perturbed layer 1",
}

ARMS: tuple[tuple[str, bool], ...] = (("delay", True), ("nodelay", False))

# --- Relocation destination windows, PER LAYER and PER ARM (measured) ---
#
# One past the last bin holding any spike at that layer, measured over all 27 v1
# checkpoints by temporal_support.py (layer 1) and layer2_baseline.py (layer 2):
# layer 1 stops at bin 87 in both arms, layer 2 at 89 no-delay and 159 delay. These are
# the exact bounds; hidden_channel_decode.py rounds each up to a multiple of its bin
# count, which is a binning constraint this script does not have.
#
# **Deletion deliberately ignores these.** It removes spikes where they are and chooses
# no destination, so there is nothing to confine.
SUPPORT_BINS: dict[str, dict[int, int]] = {
    "nodelay": {1: 88, 2: 90},
    "delay": {1: 88, 2: 160},
}

# Sweep endpoints. f = 1 relocates every spike; p_d = 0.8 deletes 80% of them. Both
# match the grids the v1 eval scripts used, so v3 scores are comparable to v1's.
RELOCATION_F: float = 1.0
DELETION_PD: float = 0.8

# Repeats per perturbed measurement; both draws are stochastic.
NUM_REPEATS: int = 3

# Training-summary fields carried through, so the regression file is self-contained and
# cannot be joined against the wrong summary. Written with ``if field in row`` because
# the three grids record different knobs: only v3L12 has ``ceiling_k_layer2`` and the
# ``_l1`` / ``_l2`` / ``_net`` axes, and only there do the unsuffixed keys alias the
# **network** pool rather than the single constrained layer (document 7 §6 point 5).
SUMMARY_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "target_layer",
    "target_layers",
    "ceiling_k",
    "ceiling_k_layer2",
    "ceiling_k_layer2_ratio",
    "ceiling_strength",
    "floor_strength",
    "seed",
    "spikes_per_neuron",
    "spikes_per_active_neuron",
    "silent_fraction",
    "spikes_per_neuron_l1",
    "spikes_per_active_neuron_l1",
    "silent_fraction_l1",
    "over_k_fraction_l1",
    "spikes_per_neuron_l2",
    "spikes_per_active_neuron_l2",
    "silent_fraction_l2",
    "over_k_fraction_l2",
    "spikes_per_neuron_net",
    "spikes_per_active_neuron_net",
    "silent_fraction_net",
)


class SparseNetwork(nn.Module):
    """v1's architecture with an eval-only perturbation hook at either hidden layer.

    The parameter set is v1's, so v1, v2 and v3 checkpoints all load directly — the v3
    factorials changed the training penalty, not the architecture.

    The forward pass is split into ``_first_hidden`` / ``_second_hidden`` / ``_output``
    so each injection site is a named stage rather than an inline expression. That is
    also what keeps ``delay1`` on the correct side of each site: it is applied inside
    ``_second_hidden``, downstream of a layer-1 injection and upstream of a layer-2 one.
    ``delay2`` is downstream of both sites and only feeds ``fc3``.

    Args:
        delays: Whether to build (and apply) the learnable axonal delays.
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

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike -> 1st hidden spikes."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> delay1 -> fc2 -> spike -> 2nd hidden spikes."""
        routed = self.delay1(hidden1) if self.delays else hidden1
        return self.slayer.spike(self.fc2(self.slayer.psp(routed)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> delay2 -> fc3 -> spike."""
        routed = self.delay2(hidden2) if self.delays else hidden2
        return self.slayer.spike(self.fc3(self.slayer.psp(routed)))

    def forward(self, x: torch.Tensor, perturbation: str = "none",
                magnitude: float = 0.0, site: str = "l1",
                windows: dict[int, int] | None = None) -> torch.Tensor:
        """Forward pass, optionally perturbing one or both hidden layers.

        Args:
            x: Input spike trains, shape (B, 700, 1, 1, T).
            perturbation: ``"none"``, ``"relocate"`` or ``"delete"``.
            magnitude: ``f`` for relocation, ``p_d`` for deletion.
            site: Injection site, one of ``SITES``.
            windows: This arm's relocation window per layer. Ignored by deletion.

        Returns:
            Output spike tensor.

        Raises:
            ValueError: If ``site`` is not one of ``SITES``.
        """
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}, got {site!r}")
        windows = windows or {}

        hidden1 = self._first_hidden(x)
        if magnitude > 0 and site in ("l1", "both"):
            hidden1 = perturb(hidden1, perturbation, magnitude, windows.get(1))
        hidden2 = self._second_hidden(hidden1)
        if magnitude > 0 and site in ("l2", "both"):
            hidden2 = perturb(hidden2, perturbation, magnitude, windows.get(2))
        return self._output(hidden2)


def perturb(hidden_spikes: torch.Tensor, perturbation: str, magnitude: float,
            window: int | None) -> torch.Tensor:
    """Apply one perturbation to one layer's spikes.

    The window is passed to relocation and withheld from deletion, which is the
    asymmetry document 7 §5 step 3 names as its first trap: relocation chooses a
    destination bin and must respect the layer's support, deletion chooses none.

    Args:
        hidden_spikes: Hidden spikes, shape (B, C, 1, 1, T).
        perturbation: ``"none"``, ``"relocate"`` or ``"delete"``.
        magnitude: ``f`` for relocation, ``p_d`` for deletion.
        window: Exclusive upper bound on relocation destinations, or None for all of T.

    Returns:
        Perturbed tensor of the same shape, dtype and device.

    Raises:
        ValueError: If ``perturbation`` is not a recognised name.
    """
    if perturbation == "none":
        return hidden_spikes
    if perturbation == "relocate":
        return relocate(hidden_spikes, magnitude, window)
    if perturbation == "delete":
        return delete(hidden_spikes, magnitude)
    raise ValueError(f"unknown perturbation {perturbation!r}")


def relocate(hidden_spikes: torch.Tensor, f: float,
             window: int | None) -> torch.Tensor:
    """Move a fraction ``f`` of each neuron's spikes to random free bins in ``window``.

    Per-neuron spike count is preserved **exactly**, which is what makes this a
    timing-only probe: at ``f = 1`` the layer retains nothing but each neuron's count.
    Confining destinations to the perturbed layer's own measured support is the v3
    correction; the support always has room, since every spike originated inside it.

    Args:
        hidden_spikes: Hidden spikes, shape (B, C, 1, 1, T).
        f: Fraction of each neuron's spikes to relocate.
        window: Exclusive upper bound on destination bins; ``None`` allows all of T.

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
        available[:, :, window:] = False
    key2 = torch.where(available, torch.rand_like(flat), torch.full_like(flat, 2.0))
    add = key2.argsort(dim=-1).argsort(dim=-1) < num_to_move
    return (keep | add).to(hidden_spikes.dtype).view(
        batch, channels, height, width, time)


def delete(hidden_spikes: torch.Tensor, deletion_probability: float) -> torch.Tensor:
    """Drop each spike independently with probability ``deletion_probability``.

    This destroys *rate* and leaves the surviving spikes where they were, so it is the
    complement of relocation and the control the timing probe has to be read against —
    a network that is simply robust to hidden interference scores highly on both.

    Args:
        hidden_spikes: Hidden spikes, shape (B, C, 1, 1, T).
        deletion_probability: Per-spike deletion probability.

    Returns:
        Perturbed tensor of the same shape, dtype and device.
    """
    keep = torch.rand_like(hidden_spikes) >= deletion_probability
    return ((hidden_spikes > 0.5) & keep).to(hidden_spikes.dtype)


@torch.no_grad()
def accuracy(net: SparseNetwork, inputs: np.ndarray, labels: torch.Tensor,
             perturbation: str, magnitude: float, site: str,
             windows: dict[int, int], seed: int) -> float:
    """Test accuracy at one perturbation setting and injection site."""
    torch.manual_seed(seed)
    correct = 0
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        predicted = snn.predict.getClass(
            net(batch, perturbation=perturbation, magnitude=magnitude, site=site,
                windows=windows)).cpu()
        correct += (predicted == labels[start:start + BATCH_SIZE]).sum().item()
    return correct / len(inputs)


def robustness_score(clean_acc: float, perturbed_acc: float) -> float:
    """Chance-corrected, baseline-normalised damage score (v1's ``temporal_score``).

    0 means the perturbation cost nothing; 1 means it drove the network to chance.
    Normalising by each network's own headroom above chance is what makes the score
    comparable across models whose clean accuracies differ — which they do here, by
    .712 to .856.

    Args:
        clean_acc: Unperturbed test accuracy.
        perturbed_acc: Test accuracy under the perturbation.

    Returns:
        The score, higher meaning more damage.
    """
    return 1.0 - (perturbed_acc - CHANCE) / max(1e-9, clean_acc - CHANCE)


def load_test_split() -> tuple[np.ndarray, torch.Tensor]:
    """Return the zero-padded test-split inputs and their labels."""
    raw = loadmat(str(MAT_FILE))
    features, labels_all = raw["X"], raw["Y"].ravel()
    total = len(labels_all)
    idx = np.arange(int(total * TEST_RANGE[0]), int(total * TEST_RANGE[1]))
    padded = np.zeros((len(idx), features.shape[1], SIM_PARAMS["tSample"]),
                      dtype=np.float32)
    padded[:, :, :features.shape[2]] = features[idx]
    return padded, torch.from_numpy(labels_all[idx].astype(np.int64))


def measure_checkpoint(run_tag: str, delays: bool, windows: dict[int, int],
                       inputs: np.ndarray, labels: torch.Tensor) -> dict:
    """Measure usage and the deletion control for one checkpoint, at every site.

    The clean accuracy is measured once and shared: it is the same forward pass
    whatever site would have been perturbed, and every site's score is normalised
    against it.

    Args:
        run_tag: Checkpoint name without extension.
        delays: Whether this is a with-delay checkpoint.
        windows: This arm's relocation window per layer.
        inputs: Padded test-split spike trains.
        labels: Test-split labels.

    Returns:
        Dict of clean accuracy, and per site both perturbed accuracies and both
        scores. Single-site grids additionally alias that site's fields unsuffixed.
    """
    net = SparseNetwork(delays).to(device)
    net.load_state_dict(torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
    net.eval()

    clean = accuracy(net, inputs, labels, "none", 0.0, SITES[0], windows, 0)
    row = {"clean_acc": clean, "sites": list(SITES)}

    for site in SITES:
        relocated = [accuracy(net, inputs, labels, "relocate", RELOCATION_F, site,
                              windows, seed) for seed in range(NUM_REPEATS)]
        deleted = [accuracy(net, inputs, labels, "delete", DELETION_PD, site,
                            windows, seed) for seed in range(NUM_REPEATS)]
        scores = {
            "acc_relocated": float(np.mean(relocated)),
            "acc_relocated_std": float(np.std(relocated)),
            "acc_deleted": float(np.mean(deleted)),
            "acc_deleted_std": float(np.std(deleted)),
            "usage": robustness_score(clean, float(np.mean(relocated))),
            "control_deletion": robustness_score(clean, float(np.mean(deleted))),
        }
        row.update({f"{name}_{site}": value for name, value in scores.items()})
        # Single-site grids keep the original flat schema as well, so analysis written
        # against it is unaffected. Multi-site grids deliberately do not: a reader that
        # assumes one site should raise KeyError rather than silently read the layer-1
        # score as the network's (document 7 §5 step 3).
        if len(SITES) == 1:
            row.update(scores)
    return row


def print_checkpoint_row(run_tag: str, row: dict) -> None:
    """Print one line per injection site for a measured checkpoint."""
    for site in SITES:
        print(f"{run_tag:<42} {site:>4} {row['clean_acc']:>7.3f} "
              f"{row[f'acc_relocated_{site}']:>7.3f} "
              f"{row[f'acc_deleted_{site}']:>7.3f} "
              f"{row[f'usage_{site}']:>7.3f} "
              f"{row[f'control_deletion_{site}']:>8.3f}", flush=True)


def main() -> None:
    """Measure usage and the deletion control across every checkpoint in each arm."""
    print(f"Using device: {device}")
    print(f"grid: {GRID} | sites: {list(SITES)} | "
          f"relocation f={RELOCATION_F:g} | deletion p_d={DELETION_PD:g} "
          f"(no window, by design) | {NUM_REPEATS} repeats")
    for site in SITES:
        print(f"  site {site:<5} {SITE_DESCRIPTIONS[site]}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    inputs, labels = load_test_split()

    for arm, delays in ARMS:
        summary_path = (TRAIN_LOG_DIR
                        / f"sparse_whole_{arm}_{VERSION_TAG}train_summary.json")
        with open(summary_path) as handle:
            summary = json.load(handle)
        windows = SUPPORT_BINS[arm]

        print(f"\n=== {arm}: {len(summary)} checkpoints ===")
        print("  relocation windows: "
              + " | ".join(f"l{layer} [0,{bound})"
                           for layer, bound in sorted(windows.items())))
        print(f"{'run_tag':<42} {'site':>4} {'clean':>7} {'reloc':>7} {'del':>7} "
              f"{'usage':>7} {'control':>8}")

        results = {}
        for run_tag, meta in summary.items():
            row = measure_checkpoint(run_tag, delays, windows, inputs, labels)
            row.update({field: meta[field] for field in SUMMARY_PASSTHROUGH_FIELDS
                        if field in meta})
            results[run_tag] = row
            print_checkpoint_row(run_tag, row)

        out_path = LOG_DIR / f"phase1_measure_{VERSION_TAG}{arm}.json"
        with open(out_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
