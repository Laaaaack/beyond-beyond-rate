"""v3 Phase 1 at BOTH LAYERS, WITH-DELAY: relocation sweep, eval-only.

Companion to the training script
[sn_bothLayer_train_withDelay_v3.py](../sn_bothLayer_train_withDelay_v3.py),
which trains the **both-layer** v3 factorial — a per-pair ceiling
``relu(count - k)`` setting temporal sparsity ``a``, crossed with a
membrane-potential floor setting selectivity ``s`` along
``FLOOR_STRENGTH = [0, 1]``, at seeds 42, 43 and 44. **Both penalties are
charged at both hidden layers**, each at the full coefficient its single-layer
sibling used, so a cell of this grid applies to layer 1 exactly the pressure
the 1st-layer grid applied and to layer 2 exactly what the 2nd-layer grid
applied. ``CEILING_K`` lists layer 1's levels; layer 2 gets twice those levels
(``CEILING_K_LAYER2_RATIO = 2.0``), because this arm's layer 2 runs about twice
as hard. This script performs **no training** and measures how much test
accuracy degrades when the trained network's hidden spikes are relocated at
evaluation only.

**The checkpoints do not exist yet, and that is expected.** Document 7 records
both both-layer grids as calibrated, corner-probed and ready but **not
launched** — they are queued for the remote server — so until
``sn_log/sparse_whole_delay_v3L12_train_summary.json`` exists this script
raises ``FileNotFoundError`` and does nothing else. It is written ahead of the
grids on purpose: retargeting the measurement layer is document 7 §5 step 3,
the largest piece of outstanding work, and the grids are uninterpretable
without it. This arm additionally carries ``SETTLE_EPOCHS = 150``, adopted
after a model-selection fault was found in its floor-on cells; that is a
training-side constant and has no role here, but it is why this arm's
checkpoints are constrained where an earlier probe's were not.

Why constrain both layers, and why this is a third question rather than a
rephrasing of the other two. The 1st- and 2nd-layer grids each constrain
**one** layer and leave the other free to compensate — and the calibration says
it does. Under v1's layer-1-only penalty the two layers' silent fractions are
correlated at **+0.203** (p = .53) in this arm: not opposed, but essentially
unrelated — v1's penalty simply had no consistent effect on layer 2's
selectivity, which is the same failure in a weaker form. The consequence is
that neither single-layer grid ever produces a sparse *network*: v1's no-delay
gradient spans 6.5x in layer-1 firing but only 2.8x network-wide, so the
manipulation is 57% weaker than the per-layer headline suggests. See document 7
§1 and §2.

What this arm's grid buys, specifically. Pooled over both layers,
``rho(a_net, s_net)`` across v1's gradient in this arm is **+0.042** (p = .90),
which already passes the ``|rho| <= 0.5`` threshold. So what this design adds
here is the **controlled** floor-on vs floor-off contrast at matched ``a``,
where v1's gradient was only a correlation, plus a direct comparison against
the completed 1st-layer delay grid (document 7 §2b point 3).

**This grid cannot attribute an effect to a layer, and neither can this sweep's
independent variables.** ``rho(a1, a2)`` is +0.839 observationally and both
knobs act on both layers, so the two layers' activity moves together *by
construction*. A coefficient estimated against these checkpoints is a
**network-level** one. Attribution is what the 1st- and 2nd-layer grids are
for, and any claim of the form "layer N's sparsity causes X" needs one of them
(document 7 §6 point 1). The printed tables and the results file carry the
per-layer axes anyway, because a cell where one layer collapsed and the other
did not must stay visible.

**Three injection sites, and the pooled one is an addition rather than a
replacement.** ``SITES = ("l1", "l2", "both")``, and every checkpoint is swept
at all three. The independent variables of this factorial are pooled, but the
dependent variable need not be and this script does not assume it should be:
perturbing both layers at once is a *different* and harsher insult than
perturbing either — at the ``both`` site layer 2 is perturbed after being
computed from an already-perturbed layer 1 — so per-layer results are reported
first (document 7 §5 step 3, third trap).

That choice has a deliberate consequence for the output format: **no unsuffixed
sweep or ``retention`` field is emitted.** Every curve is keyed
``acc_sweep_{site}`` and every score ``retention_{site}``, so a reader has to
name the site it means. The training script does alias its unsuffixed
*independent*-variable keys to the network-wide pool, and those aliases are
passed through unchanged — but for the dependent variable an analysis written
against a single-layer results file will raise ``KeyError`` here rather than
silently read a three-site file as though it were a one-site one, which is the
failure worth having.

Which arm this is. The sibling is
[the no-delay sibling](shd_bothLayer_evalOnly_noDelay_v3.py). The arms are kept
as separate scripts (rather than one script with a flag) for the same reason
the training scripts are. This network carries the learnable axonal delays
``delay1``/``delay2``, which are themselves a timing mechanism, so this sweep
measures sparsity's effect *on top of* whatever the delays contribute.

**Which arm is primary is an open question here, not an inherited one.**
Document 5's delay-arm decision rested on two Phase 0 measurements of the
*dependent* variable — the readout-efficiency gap and the deletion control
being indistinguishable from the timing probe in the no-delay arm — both
measured **at layer 1 only**. Deciding on them here would be inheriting a
constant across a layer boundary. Unlike documents 5 and 6 there is now a
consideration pointing the *other* way: the no-delay arm is the only one whose
pooled confound actually fails, so it is the only arm where this design has
something to break (document 7 §5 step 2, §6 point 8).

Relation to [phase1_measure.py](../v3_analysis/phase1_measure.py): that script
measures only the ``f = 1`` endpoint of this sweep, because that single number
is the ``usage`` dependent variable of the Phase 1 regressions. This script
sweeps the full ``f`` grid at every site, which shows the *shape* of the
degradation rather than one point of it. Note that ``phase1_measure.py`` is
itself still hardwired to layer 1 and is on document 7 §5 step 3's retarget
list; once retargeted it must share this file's injection sites and per-layer
windows for the two to agree.

Where the sites sit relative to the delays, and the trap that comes with it.
``delay1`` lives *between* the two constrained layers: it is **downstream** of
an injection at layer 1 and **upstream** of one at layer 2. A layer-1 probe is
right to ignore it; a layer-2 probe must apply it. Here that is structural
rather than a matter of remembering — layer 2 is only reachable through
``_second_hidden``, which applies ``delay1`` — but it is also why the two
layers need different windows, since ``delay1`` pushes layer 2's support out to
bin 159 where layer 1's stops at 87 (document 7 §5 step 3, second trap).
``delay2`` is downstream of both sites and only feeds ``fc3``.

Protocol rule (do not break): sparsity is applied during *training* on clean
data; relocation is applied here at *evaluation* only. The loaded checkpoint is
never modified and no gradients are taken.

Partial spike relocation: a fraction ``f`` of each neuron's spikes at the
injected layer are removed and re-placed at randomly chosen unoccupied bins,
leaving the rest untouched. Per-neuron spike count is preserved exactly, so
``f`` interpolates between the clean layer (0) and one retaining only each
neuron's spike *count* (1) — destroying spike *timing* while holding *rate*
fixed.

**Each layer uses its own measured window, and they differ.** Destinations are
drawn from ``SUPPORT_BINS[layer]`` — ``[0, 88)`` at layer 1 and ``[0, 160)`` at
layer 2 in this arm. Confining a layer's spikes to a region it never occupies
is the point; scattering them beyond it thins the population's instantaneous
spike density and mixes a rate insult into a timing-only probe. Inheriting one
layer's bound for the other is the mistake document 7 §5 step 3 names as its
first trap, and in the delay arm the two bounds differ by nearly 2x. See the
``SUPPORT_BINS`` block below.

What this script does, for every checkpoint listed in the training summary (the
authoritative live-checkpoint list — read rather than globbed, so we evaluate
exactly the models the milestone locked and reuse their recorded metadata):

- loads the delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``F_VALUES`` at each of ``SITES``, ``NUM_REPEATS`` times per f for
  error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_delay_v3L12_shd_eval.json`` with ``per_setup``, one
  **seed-averaged** row per ``(k, floor)`` cell carrying all three sites — the
  headline, since the three seeds are replicates of one cell rather than three
  conditions — and ``per_checkpoint``, the raw per-seed ``acc(f)`` sweeps it
  was computed from. Both carry the training summary's design knobs
  (``ceiling_k``, ``ceiling_k_layer2``, ``floor_strength``, ``seed``), the
  *measured* sparsity axes at both layers and pooled, and ``target_layers``, so
  the file is self-contained for the analysis and cannot be mistaken for a
  single-layer run.

The ``v3L12`` tag is load-bearing throughout, now against **two** prior grids
(document 7 §6 point 4). The 1st-layer (``v3``), 2nd-layer (``v3L2``) and
both-layer (``v3L12``) grids share dataset, arm, ``k``, floor and seed, so
without it this script would read another grid's summary and write over another
grid's results — the 1st-layer delay grid alone costs ~37 h to retrain.

The downstream analysis turns each ``acc(f)`` curve into a chance-corrected,
baseline-normalised ``usage`` score. Plot it against the *measured* axes ``a``
(``spikes_per_active_neuron_net``) and ``s`` (``silent_fraction_net``)
separately — never against ``spikes_per_neuron`` alone, which is their product
and moves with both, and never against the penalty knobs, whose map to achieved
sparsity is nonlinear and seed-dependent (document 5 §8 rule 1). Note that the
unsuffixed keys here mean the **network**, not layer 1 (document 7 §6 point 5);
use the explicit ``_l1`` / ``_l2`` keys when comparing against the single-layer
grids.

**Do not pre-fill a clean-accuracy range for this grid from the corner probe.**
The probe's .717-.806 came from four models at 400 epochs, and document 5
established that penalties calibrated at reduced epochs re-densify by the full
1250-epoch run. Read the range off the training summary once the grid has run.

One number to watch when the grid lands: ``rho(s1, s2)``. At **-0.829** under
v1's layer-1-only penalty the two layers' selectivity moved in *opposite*
directions. If this design does not pull it clearly positive, the column knob
is not producing a network-wide selectivity manipulation and the pooled
``s_net`` axis is really a statement about one layer (document 7 §6 point 3).
The training scripts print it against that baseline; it came back +1.000 in
both arms at the probe's n = 4, degenerate in magnitude but right in sign.

Architecture: Input(700) -> 128 hidden -> 128 hidden -> 20 output (SRMALPHA),
with the learnable axonal delays ``delay1`` (hidden1 -> fc2) and ``delay2``
(hidden2 -> fc3). Unchanged from v1 and from both single-layer v3 grids — only
the layers the penalties and this sweep act on moved, which is why these
checkpoints load with ``strict=True``.
Sweep (eval only): f in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, at each of the three
sites.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Directory of this script. Dataset, checkpoint, and log paths are anchored here so
# the script can be launched from any working directory. The sparsity training is
# shared across the perturbation experiments, so its checkpoints and summaries live
# one level up in exp_sparse_network/; only this relocation eval's own output is local.
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / ".." / "sn_data"        # trained checkpoints read from here
TRAIN_LOG_DIR = SCRIPT_DIR / ".." / "sn_log"    # training summary read from here
LOG_DIR = SCRIPT_DIR / "log"                    # relocation eval results written here
SHD_DATA_DIR = SCRIPT_DIR / "shd_data"          # SHD .mat source (local)

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Quick pipeline check. True evaluates only the first MAX_QUICK_CHECKPOINTS models
# on a coarse f grid with a single repeat, and suffixes the output file with
# RUN_SUFFIX so a probe can never overwrite the real eval results. False = the real
# sweep over every checkpoint in the training summary.
QUICK_TEST: bool = False

# Suffix appended to the results filename when running a probe (see QUICK_TEST).
# Empty for the real run. Mirrors the convention in the training scripts.
RUN_SUFFIX: str = "_probe" if QUICK_TEST else ""

# Checkpoints evaluated when QUICK_TEST is set (in training-summary order).
MAX_QUICK_CHECKPOINTS: int = 2

# --- Milestone scope (matches the training script) ---
DATASET_KEY: str = "whole"
DELAY_TAG: str = "delay"      # this arm; tags the summary and results files
INPUT_DIM: int = 700          # SHD whole
MAT_FILE: str = str(SHD_DATA_DIR / "shd_whole.mat")

# The hidden layers the training penalties acted on, and therefore the layers this
# sweep has to be able to reach. Matches the training scripts' constant of the same name.
TARGET_LAYERS: tuple[int, ...] = (1, 2)

# The injection sites swept. Every checkpoint is evaluated at all of them.
#
# **Per-layer first, pooled as an addition** -- document 7 §5 step 3, third trap. The
# both-layer factorial pools its *independent* variables, but the dependent variable need
# not be pooled and this design does not assume it should be: perturbing both layers at
# once is a different and harsher insult than perturbing either, not their sum. So "l1"
# and "l2" are reported alongside "both" rather than replaced by it.
#
# The cost is three sweeps per checkpoint rather than one. Trimming this tuple shortens a
# run; each site writes its own suffixed field, so a partial run stays readable.
SITES: tuple[str, ...] = ("l1", "l2", "both")

# One line per site for the printed tables, so a reader never has to infer what "both"
# did from the field name alone.
SITE_DESCRIPTIONS: dict[str, str] = {
    "l1": "injected at the 1st hidden layer only",
    "l2": "injected at the 2nd hidden layer only",
    "both": "injected at both layers -- layer 2 is perturbed after being computed "
            "from an already-perturbed layer 1",
}

# Training generation to evaluate. "v3L12" selects the 24-model **both-layer**
# factorial trained by sn_bothLayer_train_withDelay_v3.py; it matches that
# script's own VERSION_TAG, and it tags both the summary read here and the results
# written below.
#
# This tag is load-bearing against TWO prior grids (document 7 §6 point 4). The
# 1st-layer ("v3"), 2nd-layer ("v3L2") and both-layer ("v3L12") grids share
# dataset, arm, k, floor and seed and differ only in which layers the penalties
# acted on, so without a distinct tag this script would read another grid's summary
# and write over another grid's results.
VERSION_TAG: str = "v3L12"

# The training summary enumerating this arm's live v3L12 checkpoints. Its keys are
# the run tags (``sparse_whole_delay_v3L12_k{k}_floor{F}_seed{seed}``) and
# ``sn_data/{run_tag}.pt`` is the matching checkpoint. The 4-corner calibration probe
# is kept separately as ..._train_summary_probe.json and is NOT evaluated here.
TRAIN_SUMMARY_FILE: str = (
    f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_train_summary.json"
)

# Training-summary fields carried through into the results file, so the analysis can
# plot the perturbation scores against both design knobs and both *measured* sparsity
# axes without re-joining the summary.
#
# Both layers are carried, and so is the pool. The unsuffixed keys are the training
# script's deliberate aliases for the **network-wide** values, not for layer 1
# (document 7 §3 and §6 point 5) -- they are passed through unchanged so that the
# convention survives the join, and every per-layer value is available explicitly under
# ``_l1`` / ``_l2``. Use the suffixed keys when comparing against the single-layer grids.
SUMMARY_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "target_layers",
    "ceiling_k",
    "ceiling_k_layer2",
    "ceiling_k_layer2_ratio",
    "ceiling_strength",
    "floor_strength",
    "seed",
    "clean_acc",
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
    "spikes_per_neuron",
    "spikes_per_active_neuron",
    "silent_fraction",
)

# Per-checkpoint fields averaged across seeds in the per-setup summary. The seeds of one
# ``(k, floor)`` cell are replicates of the same factorial cell, so these are the cell's
# achieved values; their across-seed spread is reported alongside, because a mean that
# hides its spread is how seed noise gets read as structure.
AVERAGED_SUMMARY_FIELDS: tuple[str, ...] = (
    "clean_acc",
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
    "spikes_per_neuron",
    "spikes_per_active_neuron",
    "silent_fraction",
)

# --- SLAYER neuron and simulation descriptors (identical to training) ---
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

# --- Data split ratios (identical to training, so the test set matches) ---
TRAIN_RANGE = (0.0, 0.6)
VAL_RANGE = (0.6, 0.75)
TEST_RANGE = (0.75, 0.9)

# --- Network / evaluation hyper-parameters ---
HIDDEN_UNITS: int = 128
NUM_CLASSES: int = 20
BATCH_SIZE: int = 128
SEED: int = 42                # base seed for the relocation repeats
MAX_DELAY: int = 64           # recorded for parity with training; unused at eval

# --- Destination windows for relocated spikes, PER LAYER (measured, per arm) ---
# shd_whole.mat holds 100 time bins and load_shd_data zero-pads them to the simulator's
# 200, so neither hidden layer ever occupies the whole window. Drawing destinations from the full 200 sends most
# relocated spikes into a region where no hidden spike naturally occurs,
# thinning the population's instantaneous spike density -- a *rate* insult riding on a
# probe that is supposed to hold rate fixed and destroy only timing.
#
# **The two layers have different supports and each site must use its own** (document 7
# §5 step 3). Layer 1's last occupied bin is 87 in both arms; layer 2's is 159 in this arm, because ``delay1`` (up to 64 bins) shifts layer 1's
# spikes later before ``fc2`` sees them.
# Measured over all 27 v1 checkpoints by v3_analysis/temporal_support.py and
# v3_analysis/layer2_baseline.py.
#
# Set either to None to reproduce full-window behaviour at that layer.
SUPPORT_BINS_L1: int | None = 88
SUPPORT_BINS_L2: int | None = 160

# The window in force at each layer, looked up by layer index when a site is perturbed.
SUPPORT_BINS: dict[int, int | None] = {1: SUPPORT_BINS_L1, 2: SUPPORT_BINS_L2}

# Appended to the results filename. The corrected windows are what a v3 run means, so
# they take the bare name and any full-window variant is the one that gets marked.
WINDOW_SUFFIX: str = (
    "" if None not in SUPPORT_BINS.values() else "_fullwindow"
)

# --- Relocation sweep: fraction of each neuron's spikes moved. 0 = clean. ---
# The grid the earlier Beyond Rate realistic-SHD runs used, so this milestone's
# curves can be read beside them. f = 1 keeps only each neuron's spike count.
F_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# --- Repeats per f, for error bars (the relocation draw is stochastic). ---
NUM_REPEATS: int = 3


def resolve_eval_config() -> tuple[list[float], int, int | None]:
    """Return (f_values, num_repeats, max_checkpoints) for this run.

    Collapses to a tiny, fast sweep when ``QUICK_TEST`` is set, so the eval
    pipeline can be validated before committing to every checkpoint.

    Returns:
        Tuple of (f_values, num_repeats, max_checkpoints), where
        ``max_checkpoints`` is None for the real run (evaluate all of them).
    """
    if QUICK_TEST:
        return [0.0, 0.4, 1.0], 1, MAX_QUICK_CHECKPOINTS
    return F_VALUES, NUM_REPEATS, None


def load_shd_data(mat_path: str, target_T: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Load an SHD dataset from a .mat file and pad the time dimension.

    Args:
        mat_path: Path to the .mat file containing 'X' and 'Y'.
        target_T: Target time dimension (pad with zeros if shorter).

    Returns:
        Tuple of (X, Y) where X has shape (N, neurons, target_T).
    """
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


class SpikeDataset(Dataset):
    """Wrap numpy spike trains and labels into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


def get_split_indices(
    split_range: tuple[float, float],
    total: int,
) -> np.ndarray:
    """Return the index array for a given fractional range of the dataset."""
    start = int(total * split_range[0])
    end = int(total * split_range[1])
    return np.arange(start, end)


def build_test_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build the test DataLoader over the fixed test split.

    The split is a fixed fractional range, so the test set is identical to the one
    the training script held out — and identical across every checkpoint and both
    arms — which is what makes the relocation curves comparable.

    Args:
        X: Full dataset features, shape (N, neurons, T).
        Y: Full dataset labels, shape (N,).
        batch_size: Batch size for the loader.

    Returns:
        The test DataLoader (unshuffled).
    """
    test_idx = get_split_indices(TEST_RANGE, len(Y))
    test_ds = SpikeDataset(X[test_idx], Y[test_idx])
    print(f"Test set: {len(test_ds)} samples")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def perturb_hidden_batch(
    hidden_spikes: torch.Tensor,
    f: float = 0.0,
    support_bins: int | None = None,
) -> torch.Tensor:
    """Vectorised GPU-side partial spike relocation (eval-only helper).

    For each ``(sample, neuron)``, a fraction ``f`` of the existing spikes are
    removed and replaced by the same number of spikes placed at randomly chosen,
    previously-unoccupied time bins. Spike count per neuron is preserved exactly.

    This is the "realistic" SHD perturbation of the Beyond Rate line of work, and it
    differs from jitter in *reach* rather than in kind: jitter displaces every spike
    by a local Gaussian offset, whereas this relocates a chosen fraction of spikes
    to anywhere in the window while leaving the rest untouched. ``f`` therefore
    interpolates between the clean layer (0) and one retaining only each neuron's
    spike *count* (1), which makes it the most direct test of "is this network
    reading timing or rate?" available at a fixed firing rate.

    Destinations are confined to ``[0, SUPPORT_BINS)``, the layer's measured
    temporal support, so that relocation cannot dilute the population's spike
    density by scattering spikes into the zero-padded tail. Per-neuron spike count
    is preserved exactly either way; the support always has room, since a neuron's
    spikes all originate inside it.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).
        support_bins: Upper bound (exclusive) on destination bins, i.e. the
            perturbed layer's own measured temporal support. None draws from
            the whole simulation window.

    Returns:
        Perturbed tensor with the same shape, dtype and device.
    """
    if f <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    # Count spikes per (sample, neuron) and compute how many to move.
    n_spikes = is_spike.sum(dim=-1, keepdim=True)          # (B, C, 1)
    num_to_move = (n_spikes.float() * f).floor().long()    # (B, C, 1)

    # --- 1. Choose which existing spikes to remove ---
    # Random key per time bin; non-spike bins get a large key so they sort last.
    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    # rank[b, c, t] = position of t in the per-(b, c) ascending sort of `key`.
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move                       # (B, C, T)

    keep_mask = is_spike & ~remove_mask

    # --- 2. Place the same number of spikes in currently-unoccupied bins ---
    available = ~keep_mask      # everything except positions we are keeping
    if support_bins is not None:
        available[:, :, support_bins:] = False   # stay inside this layer's support
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move    # disjoint from keep_mask by construction

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype)
    return new_spikes.view(B, C, H, W, T)


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN with learnable delays, for the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_bothLayer_train_withDelay_v3.py](../sn_bothLayer_train_withDelay_v3.py)
    — ``fc1``/``fc2``/``fc3`` weight-norm parameters plus ``delay1``/``delay2``
    — so this class loads those checkpoints directly. It is also identical to
    v1's and to both single-layer v3 grids': only the layers the penalties
    target moved, never the architecture. The training script's adaptive
    delay-clamping schedule is deliberately absent: it shapes delays *during*
    training, and the loaded values are used here exactly as saved.

    ``forward_with_hidden_perturbation`` injects at ``site`` — the 1st hidden
    layer, the 2nd, or both — and both of those tensors are ones the training
    penalties charged. The forward pass is split into ``_first_hidden`` /
    ``_second_hidden`` / ``_output`` so each site is a named stage rather than an
    inline expression, which is also what keeps ``delay1`` on the correct side of
    each site: it is applied inside ``_second_hidden``, downstream of a layer-1
    injection and upstream of a layer-2 one. It is eval-only; nothing here is
    ever trained.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = HIDDEN_UNITS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer

        self.fc1 = nn.utils.weight_norm(
            slayer.dense(input_dim, hidden_units), name="weight"
        )
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(hidden_units, hidden_units), name="weight"
        )
        self.fc3 = nn.utils.weight_norm(
            slayer.dense(hidden_units, num_classes), name="weight"
        )

        # delay1 sits inside _second_hidden, i.e. BETWEEN the two constrained
        # layers: downstream of an injection at layer 1, upstream of one at
        # layer 2. That asymmetry is why this arm's layer-2 support runs to bin
        # 159 where layer 1's stops at 87. delay2 is downstream of both.
        self.delay1 = slayer.delay(hidden_units)
        self.delay2 = slayer.delay(hidden_units)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the input is 5-D NCHWT on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(device)

    def _first_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> PSP -> fc1 -> spike -> 1st hidden spikes (strictly binary)."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))

    def _second_hidden(self, hidden1: torch.Tensor) -> torch.Tensor:
        """hidden1 -> delay1 -> fc2 -> spike -> 2nd hidden spikes.

        ``delay1`` is applied here, which puts it downstream of an injection at
        layer 1 and upstream of one at layer 2. A layer-1 probe is right to
        ignore it; a layer-2 probe must apply it, and does so by construction
        because layer 2 is only reachable through this method (document 7 §5
        step 3, second trap).

        Args:
            hidden1: 1st hidden layer spikes, shape (B, C, 1, 1, T).

        Returns:
            The 2nd hidden layer's spikes, same shape.
        """
        return self.slayer.spike(
            self.fc2(self.slayer.psp(self.delay1(hidden1)))
        )

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> delay2 -> fc3 -> spike."""
        return self.slayer.spike(
            self.fc3(self.slayer.psp(self.delay2(hidden2)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean forward pass (equivalent to the f = 0 baseline)."""
        x = self._prepare_input(x)
        hidden2 = self._second_hidden(self._first_hidden(x))
        return self._output(hidden2)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        f: float = 0.0,
        site: str = "both",
    ) -> torch.Tensor:
        """Eval-only: relocate spikes at ``site`` before the readout.

        Must be called inside ``torch.no_grad()`` — ``perturb_hidden_batch`` is not
        autograd-safe. ``f = 0`` reduces to the clean forward pass.

        ``site`` is one of ``SITES``: ``"l1"`` perturbs the 1st hidden layer only,
        ``"l2"`` the 2nd only, and ``"both"`` perturbs each in turn -- so the
        layer-2 insult lands on a layer that was itself computed from an
        already-perturbed layer 1. That is what makes ``"both"`` a harsher probe
        than either single site rather than their sum (document 7 §5 step 3).

        Each layer is confined to **its own** measured support,
        ``SUPPORT_BINS[layer]`` -- the two differ, and in the delay arm they differ
        by nearly 2x.

        Args:
            x: Input spike trains.
            f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).

            site: Injection site, one of ``SITES``.

        Returns:
            The output spike tensor.

        Raises:
            ValueError: If ``site`` is not one of ``SITES``.
        """
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}, got {site!r}")

        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if f > 0 and site in ("l1", "both"):
            hidden1 = perturb_hidden_batch(hidden1, f, SUPPORT_BINS[1])
        hidden2 = self._second_hidden(hidden1)
        if f > 0 and site in ("l2", "both"):
            hidden2 = perturb_hidden_batch(hidden2, f, SUPPORT_BINS[2])
        return self._output(hidden2)


def window_label(support_bins: int | None) -> str:
    """Render one layer's perturbation window for the run banner."""
    return "full" if support_bins is None else f"[0,{support_bins})"


def load_checkpoint(
    checkpoint_path: Path,
    input_dim: int = INPUT_DIM,
) -> SparseSHDNetwork:
    """Load a trained checkpoint into a fresh delay network in eval mode.

    Args:
        checkpoint_path: Path to the saved state_dict (.pt).
        input_dim: Number of input neurons (matches the dataset variant).

    Returns:
        The network in eval mode with the checkpoint weights loaded.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = SparseSHDNetwork(input_dim, HIDDEN_UNITS, NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


@torch.no_grad()
def test_at_f(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    f: float,
    site: str,
) -> float:
    """Return test accuracy with ``site``'s spikes relocated at ``f``.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        f: Fraction of spikes relocated; 0 is the clean baseline.
        site: Injection site, one of ``SITES``.

    Returns:
        Fraction of correctly classified test samples.
    """
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        outputs = net.forward_with_hidden_perturbation(
            x_batch, f=f, site=site)
        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)
    return correct / max(1, total)


def test_with_repeats(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    f: float,
    num_repeats: int,
    site: str,
) -> dict:
    """Repeat the perturbed evaluation ``num_repeats`` times for error bars.

    Each repeat re-seeds torch (which seeds the CUDA generator the relocation draws
    from) so the perturbation differs run to run but the whole sweep is
    reproducible. ``f = 0`` involves no draw, so its repeats are identical and
    its std must come out at 0 — a useful check that no other randomness leaks into
    evaluation.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).
        num_repeats: Number of repeats.
        site: Injection site, one of ``SITES``.

    Returns:
        Dict with keys ``mean``, ``std``, ``values``.
    """
    accuracies = []
    for repeat in range(num_repeats):
        torch.manual_seed(SEED + repeat)
        np.random.seed(SEED + repeat)
        accuracies.append(test_at_f(net, test_loader, f, site))
    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "values": [float(acc) for acc in accuracies],
    }


def load_train_summary() -> dict:
    """Load this arm's training summary (the authoritative checkpoint list).

    Returns:
        Dict mapping run tag -> the training script's recorded metrics.

    Raises:
        FileNotFoundError: If the summary is missing (train the arm first).
    """
    summary_path = TRAIN_LOG_DIR / TRAIN_SUMMARY_FILE
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Training summary not found: {summary_path}. Run "
            f"sn_bothLayer_train_withDelay_v3.py first — this script only evaluates "
            f"existing checkpoints. Both both-layer grids were calibrated and "
            f"probed but not launched as of document 7, so this is the expected "
            f"failure until they are."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_perturbation_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only relocation at every site and checkpoint, and save.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The payload written to disk: ``per_setup`` (one seed-averaged row per
        ``(k, floor)`` cell) and ``per_checkpoint`` (the raw per-seed rows).
    """
    f_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print(f"# v3 Phase 1, layers {list(TARGET_LAYERS)} (eval-only relocation)")
    print(f"# arm: SGD-delay | dataset: SHD {DATASET_KEY} | grid: {VERSION_TAG} | "
          f"QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | f: {f_values} | "
          f"repeats: {num_repeats} | sites: {list(SITES)}")
    print(f"# windows: L1 {window_label(SUPPORT_BINS_L1)} | "
          f"L2 {window_label(SUPPORT_BINS_L2)}")
    print(f"{'#' * 70}")

    results: dict[str, dict] = {}
    for index, run_tag in enumerate(run_tags, start=1):
        row = train_summary[run_tag]
        print(f"\n[{index}/{len(run_tags)}] {run_tag}")

        net = load_checkpoint(CKPT_DIR / f"{run_tag}.pt")

        site_sweeps = {}
        for site in SITES:
            sweep = {}
            for f_val in f_values:
                sweep_result = test_with_repeats(
                    net, test_loader, f_val, num_repeats, site)
                sweep[str(f_val)] = sweep_result
                print(
                    f"  [{site:>4}] f={f_val:<4} "
                    f"acc={sweep_result['mean']:.4f} "
                    f"+/- {sweep_result['std']:.4f}"
                )
            site_sweeps[f"f_sweep_{site}"] = sweep

        results[run_tag] = {
            "use_delay": True,
            "target_layers": list(TARGET_LAYERS),
            "sites": list(SITES),
            # Copied verbatim rather than cast, so integer knobs (``seed``) stay
            # integers and a downstream join on them cannot go wrong.
            **{
                field: row[field]
                for field in SUMMARY_PASSTHROUGH_FIELDS
                if field in row
            },
            **site_sweeps,
        }

    # The per-setup average is the headline; the per-checkpoint rows it was
    # computed from are kept beside it, because they are the raw measurement and
    # because the across-seed spread cannot be recovered from a mean alone.
    aggregated = aggregate_sites(results, "f_sweep", f_values)
    payload = {
        "target_layers": list(TARGET_LAYERS),
        "sites": list(SITES),
        "per_setup": aggregated,
        "per_checkpoint": results,
    }

    results_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_shd_eval"
          f"{RUN_SUFFIX}{WINDOW_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nRelocation sweep saved to {results_path} "
          f"({len(results)} checkpoints -> {len(aggregated)} setups)")

    print_summary_table(aggregated, f_values)
    return payload


def aggregate_over_seeds(results: dict, sweep_field: str,
                         grid_values: list) -> dict:
    """Collapse the per-seed checkpoints into one row per ``(k, floor)`` setup.

    The seeds are replicates of the same factorial cell, so the *cell* is the unit
    the design is about, and a per-seed table invites reading noise as structure —
    the three seeds of one cell differ by up to .08 in clean accuracy here.

    Two quantities are averaged, and the two averages are **not** interchangeable:

    - ``acc(f)`` at each grid point is averaged directly across seeds, which is
      what a curve should show;
    - ``retention`` is computed **per seed, against that seed's own clean accuracy**,
      and only then averaged. The score exists to normalise away each model's
      differing headroom above chance, so deriving it from an already-averaged curve
      divided by an averaged baseline would put exactly that variation back in.

    Args:
        results: Per-checkpoint results, run tag -> metadata + sweep.
        sweep_field: Key holding the sweep dict in each per-checkpoint row.
        grid_values: The swept grid, in order. Its two ends define ``retention``.

    Returns:
        Dict keyed ``k{k}_floor{f}`` -> the cell's averaged metrics, ordered by
        ``(ceiling_k, floor_strength)``. Every averaged field is accompanied by its
        across-seed standard deviation and by the per-seed values it came from.
    """
    chance = 1.0 / NUM_CLASSES
    grid_keys = [str(value) for value in grid_values]

    cells: dict[str, list[dict]] = {}
    for row in results.values():
        setup_tag = f"k{row['ceiling_k']:g}_floor{row['floor_strength']:g}"
        cells.setdefault(setup_tag, []).append(row)

    aggregated: dict[str, dict] = {}
    for setup_tag in sorted(cells, key=lambda tag: (cells[tag][0]["ceiling_k"],
                                                    cells[tag][0]["floor_strength"])):
        rows = sorted(cells[setup_tag], key=lambda row: row["seed"])

        sweep = {}
        for key in grid_keys:
            per_seed = [row[sweep_field][key]["mean"] for row in rows]
            sweep[key] = {
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed)),
                "values": [float(value) for value in per_seed],
            }

        retentions = []
        for row in rows:
            clean = row[sweep_field][grid_keys[0]]["mean"]
            worst = row[sweep_field][grid_keys[-1]]["mean"]
            headroom = clean - chance
            retentions.append((worst - chance) / headroom
                              if headroom > 0 else float("nan"))

        averaged = {}
        for field in AVERAGED_SUMMARY_FIELDS:
            if field not in rows[0]:
                continue
            per_seed = [row[field] for row in rows]
            averaged[field] = float(np.mean(per_seed))
            averaged[f"{field}_std"] = float(np.std(per_seed))

        aggregated[setup_tag] = {
            "ceiling_k": rows[0]["ceiling_k"],
            "floor_strength": rows[0]["floor_strength"],
            "n_seeds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "retention": float(np.mean(retentions)),
            "retention_std": float(np.std(retentions)),
            "retention_per_seed": [float(value) for value in retentions],
            **averaged,
            sweep_field: sweep,
        }
    return aggregated


def aggregate_sites(results: dict, sweep_key: str, grid_values: list) -> dict:
    """Run ``aggregate_over_seeds`` once per injection site and merge the rows.

    Each site contributes its own ``{sweep_key}_{site}`` curve and its own
    ``retention_{site}``. The design metadata and the measured sparsity axes are
    identical across sites, so they are written once.

    **No unsuffixed ``retention`` or sweep field is emitted, deliberately.** The
    training script aliases its unsuffixed *independent*-variable keys to the
    network-wide pool, but document 7 §5 step 3 asks for the opposite treatment of the
    *dependent* variable: the pooled ``"both"`` measurement is an addition, not a
    replacement, so a reader has to name the site it means. Analysis code written
    against a single-layer results file therefore raises ``KeyError`` here rather than
    silently reading a three-site file as though it were a one-site one -- which is the
    failure worth having.

    Args:
        results: Per-checkpoint results, run tag -> metadata plus one sweep per site.
        sweep_key: Base name of the sweep field, without its site suffix.
        grid_values: The swept grid, in order.

    Returns:
        Dict keyed ``k{k}_floor{f}`` -> the cell's metrics, carrying every site.
    """
    site_specific = ("retention", "retention_std", "retention_per_seed")
    merged: dict[str, dict] = {}
    for site in SITES:
        per_site = aggregate_over_seeds(results, f"{sweep_key}_{site}", grid_values)
        for setup_tag, row in per_site.items():
            target = merged.setdefault(setup_tag, {})
            for name, value in row.items():
                target[f"{name}_{site}" if name in site_specific else name] = value
    return merged


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Render one averaged quantity as ``mean+-std`` for the printed table."""
    return f"{mean:.{decimals}f}+-{std:.{decimals}f}"


def print_summary_table(aggregated: dict, f_values: list[float]) -> None:
    """Print one table per injection site, one row per ``(k, floor)`` setup.

    ``retention`` is the chance-corrected fraction of accuracy surviving the strongest
    relocation; the analysis's ``usage`` score is ``1 - retention``. It is printed here
    only for eyeballing — the scoring, the regression on ``a`` and ``s`` and the
    plotting belong to the analysis step.

    **Three tables, not one, and per-layer columns in each.** The sites are not
    interchangeable: ``both`` is a harsher insult than either single layer rather than
    their sum, so it is reported alongside them rather than in place of them (document 7
    §5 step 3). And although this grid manipulates the *network*, so that ``a_net`` and
    ``s_net`` are the axes a coefficient off it belongs to, the per-layer columns stay
    visible because the failure mode this design is most exposed to is one layer
    collapsing while the other looks healthy (document 7 §6 point 6).

    Read a trend against ``a`` and ``s`` separately, never against ``sp/neuron``, which
    is their product and moves with both (document 5 §8 rule 1). Every averaged column
    carries its across-seed spread, so a cell whose seeds disagree cannot be mistaken
    for a tight one.

    Args:
        aggregated: The per-setup dict produced by ``aggregate_sites``.
        f_values: The f grid that was swept, in order.
    """
    grid_min, grid_max = str(f_values[0]), str(f_values[-1])

    for site in SITES:
        print(f"\n[site: {site}] {SITE_DESCRIPTIONS[site]}")
        print(
            f"{'setup':<14} {'n':>2} {'a_net':>12} {'s_net':>12} "
            f"{'a_l1':>6} {'s_l1':>7} {'a_l2':>6} {'s_l2':>7} "
            f"{'acc(' + grid_min + ')':>15} {'acc(' + grid_max + ')':>15} "
            f"{'retention':>15}"
        )
        for setup_tag, row in aggregated.items():
            sweep = row[f"f_sweep_{site}"]
            print(
                f"{setup_tag:<14} {row['n_seeds']:>2} "
                f"{format_mean_std(row['spikes_per_active_neuron_net'], row['spikes_per_active_neuron_net_std'], 2):>12} "
                f"{format_mean_std(row['silent_fraction_net'], row['silent_fraction_net_std'], 3):>12} "
                f"{row['spikes_per_active_neuron_l1']:>6.2f} "
                f"{row['silent_fraction_l1']:>7.1%} "
                f"{row['spikes_per_active_neuron_l2']:>6.2f} "
                f"{row['silent_fraction_l2']:>7.1%} "
                f"{format_mean_std(sweep[grid_min]['mean'], sweep[grid_min]['std'], 4):>15} "
                f"{format_mean_std(sweep[grid_max]['mean'], sweep[grid_max]['std'], 4):>15} "
                f"{format_mean_std(row[f'retention_{site}'], row[f'retention_std_{site}'], 3):>15}"
            )

    print("\n  Means over seeds; +- is the across-seed standard deviation. Per-seed "
          "rows are kept in the results file under 'per_checkpoint'.")
    print("  a_net / s_net are pooled over (sample, neuron) pairs across BOTH layers "
          "and are the axes this grid manipulates; a_l1 / a_l2 are shown so a cell "
          "where one layer collapsed stays visible.")
    print("  This grid moves both layers' activity together by construction, so a "
          "coefficient from it is a NETWORK-level one. Attributing an effect to a "
          "single layer needs the 1st- or 2nd-layer grid (document 7, section 6 "
          "point 1).")

def main() -> None:
    """Load the shared test set once, then sweep spike relocation over the with-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_perturbation_sweep(test_loader)


if __name__ == "__main__":
    main()
