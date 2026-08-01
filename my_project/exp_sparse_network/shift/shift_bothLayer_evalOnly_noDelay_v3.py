"""v3 Phase 1 at BOTH LAYERS, NO-DELAY: shift sweep, eval-only.

Companion to the training script
[sn_bothLayer_train_noDelay_v3.py](../sn_bothLayer_train_noDelay_v3.py), which
trains the **both-layer** v3 factorial — a per-pair ceiling ``relu(count - k)``
setting temporal sparsity ``a``, crossed with a membrane-potential floor
setting selectivity ``s`` along ``FLOOR_STRENGTH = [0, 1]``, at seeds 42, 43
and 44. **Both penalties are charged at both hidden layers**, each at the full
coefficient its single-layer sibling used, so a cell of this grid applies to
layer 1 exactly the pressure the 1st-layer grid applied and to layer 2 exactly
what the 2nd-layer grid applied. ``CEILING_K`` lists layer 1's levels; layer 2
gets the same levels (``CEILING_K_LAYER2_RATIO = 1.0``). This script performs
**no training** and measures how much test accuracy degrades when the trained
network's hidden spikes are shifted at evaluation only.

**The checkpoints do not exist yet, and that is expected.** Document 7 records
both both-layer grids as calibrated, corner-probed and ready but **not
launched** — they are queued for the remote server — so until
``sn_log/sparse_whole_nodelay_v3L12_train_summary.json`` exists this script
raises ``FileNotFoundError`` and does nothing else. It is written ahead of the
grids on purpose: retargeting the measurement layer is document 7 §5 step 3,
the largest piece of outstanding work, and the grids are uninterpretable
without it.

Why constrain both layers, and why this is a third question rather than a
rephrasing of the other two. The 1st- and 2nd-layer grids each constrain
**one** layer and leave the other free to compensate — and the calibration says
it does. Under v1's layer-1-only penalty the two layers' silent fractions are
correlated at **-0.829** (p = 1.4e-4) in this arm: as layer 1 is driven toward
silence, layer 2 becomes *less* silent — a layer-1 penalty does not make the
network selective, it trades layer 1's silence for layer 2's activity. The
consequence is that neither single-layer grid ever produces a sparse *network*:
v1's no-delay gradient spans 6.5x in layer-1 firing but only 2.8x network-wide,
so the manipulation is 57% weaker than the per-layer headline suggests. See
document 7 §1 and §2.

What this arm's grid buys, specifically. Pooled over both layers,
``rho(a_net, s_net)`` across v1's gradient in this arm is **-0.879** (p =
1.6e-5), which fails the ``|rho| <= 0.5`` acceptance threshold outright. So
what this design adds here is **breaking a pooled confound of -0.879** — this
is the only arm where the network-wide axes actually fail the threshold, so it
is where this design has something to break rather than merely something to
control (document 7 §2b point 3).

**This grid cannot attribute an effect to a layer, and neither can this sweep's
independent variables.** ``rho(a1, a2)`` is +0.961 observationally and both
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
[the with-delay sibling](shift_bothLayer_evalOnly_withDelay_v3.py). The arms
are kept as separate scripts (rather than one script with a flag) for the same
reason the training scripts are. **Here there are no delays at all**, so any
timing effect this sweep finds is attributable to the SRM neurons' membrane
dynamics alone.

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
measures only the *endpoints* the Phase 1 regressions need (relocation
``f = 1`` and the ``p_d = 0.8`` deletion control). Shift is not one of them —
it is kept here as a second, coarser timing perturbation to read the relocation
result against. Note that ``phase1_measure.py`` is itself still hardwired to
layer 1 and is on document 7 §5 step 3's retarget list; once retargeted it must
share this file's injection sites and per-layer windows for the two to agree.

Where the sites sit: ``hidden1 = spike(fc1(psp(x)))`` and
``hidden2 = spike(fc2(psp(hidden1)))``, exactly the two tensors the training
penalties charged. There is no delay line anywhere in this arm, so both sites
are unambiguous — but see the sibling, where ``delay1`` sits between them and
is downstream of one site and upstream of the other.

Protocol rule (do not break): sparsity is applied during *training* on clean
data; shift is applied here at *evaluation* only. The loaded checkpoint is
never modified and no gradients are taken.

Per-neuron shift: one offset is drawn from ``N(0, sigma)`` per (sample, neuron)
at the injected layer and *all* of that neuron's spikes move together, clipped
to ``[0, T-1]``. Unlike jitter, each neuron's internal spike pattern survives
intact — only its alignment to the other neurons and to stimulus onset is
destroyed. Spike count is preserved except where end-of-window clipping merges
spikes, so this too is a timing perturbation rather than a rate one.

**This sweep takes no support-window correction at either layer, and that is
deliberate.** A shift is a rigid translation of one neuron's whole spike train,
so it preserves the population's instantaneous spike density exactly and never
dilutes it into a region the layer does not occupy. Clipping targets to a
measured support would instead pile spikes up at the support edge and merge
them, which *would* destroy per-neuron count — so the full-window clip below is
the correct choice for this perturbation, not an oversight. Relocation and
jitter confine a destination bin and do take the per-layer windows; shift and
deletion do not (document 7 §5 step 3, first trap).

What this script does, for every checkpoint listed in the training summary (the
authoritative live-checkpoint list — read rather than globbed, so we evaluate
exactly the models the milestone locked and reuse their recorded metadata):

- loads the no-delay architecture and the checkpoint's weights, in eval mode;
- sweeps ``SIGMA_VALUES`` at each of ``SITES``, ``NUM_REPEATS`` times per sigma
  for error bars, all inside ``torch.no_grad()``;
- writes ``log/sparse_whole_nodelay_v3L12_shift_eval.json`` with ``per_setup``,
  one **seed-averaged** row per ``(k, floor)`` cell carrying all three sites —
  the headline, since the three seeds are replicates of one cell rather than
  three conditions — and ``per_checkpoint``, the raw per-seed ``acc(sigma)``
  sweeps it was computed from. Both carry the training summary's design knobs
  (``ceiling_k``, ``ceiling_k_layer2``, ``floor_strength``, ``seed``), the
  *measured* sparsity axes at both layers and pooled, and ``target_layers``, so
  the file is self-contained for the analysis and cannot be mistaken for a
  single-layer run.

The ``v3L12`` tag is load-bearing throughout, now against **two** prior grids
(document 7 §6 point 4). The 1st-layer (``v3``), 2nd-layer (``v3L2``) and
both-layer (``v3L12``) grids share dataset, arm, ``k``, floor and seed, so
without it this script would read another grid's summary and write over another
grid's results — the 1st-layer delay grid alone costs ~37 h to retrain.

The downstream analysis turns each ``acc(sigma)`` curve into a
chance-corrected, baseline-normalised ``temporal`` score. Plot it against the
*measured* axes ``a`` (``spikes_per_active_neuron_net``) and ``s``
(``silent_fraction_net``) separately — never against ``spikes_per_neuron``
alone, which is their product and moves with both, and never against the
penalty knobs, whose map to achieved sparsity is nonlinear and seed-dependent
(document 5 §8 rule 1). Note that the unsuffixed keys here mean the
**network**, not layer 1 (document 7 §6 point 5); use the explicit ``_l1`` /
``_l2`` keys when comparing against the single-layer grids.

**Do not pre-fill a clean-accuracy range for this grid from the corner probe.**
The probe's .441-.555 came from four models at 400 epochs, and document 5
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
with no learnable delays anywhere. Unchanged from v1 and from both single-layer
v3 grids — only the layers the penalties and this sweep act on moved, which is
why these checkpoints load with ``strict=True``.
Sweep (eval only): sigma in {0, 1, 3, 5, 10, 17, 25} time steps (ms), at each
of the three sites.
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
# one level up in exp_sparse_network/; only this shift eval's own output is local.
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / ".." / "sn_data"        # trained checkpoints read from here
TRAIN_LOG_DIR = SCRIPT_DIR / ".." / "sn_log"    # training summary read from here
LOG_DIR = SCRIPT_DIR / "log"                    # shift eval results written here
SHD_DATA_DIR = SCRIPT_DIR / ".." / "shd" / "shd_data"  # SHD .mat source

# slayerSNN is provided by the workspace venv (pip-installed egg); a plain import
# resolves it. No sys.path manipulation is needed here.
import slayerSNN as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# Global Configuration
# =====================================================================
# Quick pipeline check. True evaluates only the first MAX_QUICK_CHECKPOINTS models
# on a coarse sigma grid with a single repeat, and suffixes the output file with
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
DELAY_TAG: str = "nodelay"    # this arm; tags the summary and results files
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
# factorial trained by sn_bothLayer_train_noDelay_v3.py; it matches that
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
# the run tags (``sparse_whole_nodelay_v3L12_k{k}_floor{F}_seed{seed}``) and
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
SEED: int = 42                # base seed for the shift repeats

# --- Shift sweep: sigma in time steps (ms). 0 = clean baseline. ---
# The same grid the jitter sweep uses, so the two timing perturbations share one
# x-axis; it also matches the earlier fixed-weight perturbation results.
SIGMA_VALUES: list[int] = [0, 1, 3, 5, 10, 17, 25]

# --- Repeats per sigma, for error bars (the shift draw is stochastic). ---
NUM_REPEATS: int = 3


def resolve_eval_config() -> tuple[list[int], int, int | None]:
    """Return (sigma_values, num_repeats, max_checkpoints) for this run.

    Collapses to a tiny, fast sweep when ``QUICK_TEST`` is set, so the eval
    pipeline can be validated before committing to every checkpoint.

    Returns:
        Tuple of (sigma_values, num_repeats, max_checkpoints), where
        ``max_checkpoints`` is None for the real run (evaluate all of them).
    """
    if QUICK_TEST:
        return [0, 5, 25], 1, MAX_QUICK_CHECKPOINTS
    return SIGMA_VALUES, NUM_REPEATS, None


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
    arms — which is what makes the shift curves comparable.

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
def shift_hidden_batch(
    hidden_spikes: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Vectorised GPU-side per-neuron Gaussian shift (eval-only helper).

    For each ``(sample, neuron)`` a single integer offset ``round(N(0, sigma))`` is
    drawn and every spike of that neuron is moved by it, then clipped to
    ``[0, T - 1]``. Spikes that collide after clipping are merged (logical OR).

    This is the coarser sibling of the jitter perturbation. Jitter moves every spike
    independently and so destroys a neuron's *internal* spike pattern; a shift
    translates that pattern intact and destroys only its alignment relative to the
    other neurons and to stimulus onset. A network reading absolute or cross-neuron
    spike timing loses accuracy here; one reading each neuron's own inter-spike
    structure, or just its rate, does not.

    Spike count is preserved except where end-of-window clipping merges spikes,
    which is why the shift grid stays small relative to ``T = 200``.

    Args:
        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T).
        sigma: Per-neuron shift std dev in time steps (ms). 0 means no shift.

    Returns:
        Shifted tensor with the same shape, dtype and device.
    """
    if sigma <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = (x > 0.5).to(x.dtype)

    # One integer offset per (sample, neuron); broadcast over the time axis.
    offset = (torch.randn(B, C, 1, device=x.device) * sigma).round().long()
    t_idx = torch.arange(T, device=x.device).view(1, 1, T)
    target = (t_idx + offset).clamp(0, T - 1).expand(B, C, T).contiguous()

    # Scatter each source bin's spike onto its shifted target; summing then
    # thresholding merges colliding spikes as a logical OR.
    new_x = torch.zeros_like(x)
    new_x.scatter_add_(-1, target, is_spike)
    new_x = (new_x > 0.5).to(hidden_spikes.dtype)

    return new_x.view(B, C, H, W, T)


class SparseSHDNetwork(nn.Module):
    """2-hidden-layer SLAYER SNN without delays, for the sparse checkpoints.

    The parameter set is identical to the training class of the same name in
    [sn_bothLayer_train_noDelay_v3.py](../sn_bothLayer_train_noDelay_v3.py) —
    ``fc1``/``fc2``/``fc3`` weight-norm parameters and nothing else — so this
    class loads those checkpoints directly. It is also identical to v1's and to
    both single-layer v3 grids': only the layers the penalties target moved,
    never the architecture.

    Spikes propagate straight from one dense layer to the next, so the only timing
    machinery in the whole model is the SRM neurons' own membrane dynamics. That is
    what this arm is for: whatever survives the sweep cannot be credited to a
    learnable delay line, because there is none.

    ``forward_with_hidden_perturbation`` injects at ``site`` — the 1st hidden
    layer, the 2nd, or both — and both of those tensors are ones the training
    penalties charged. The forward pass is split into ``_first_hidden`` /
    ``_second_hidden`` / ``_output`` so each site is a named stage rather than an
    inline expression. It is eval-only; nothing here is ever trained.
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
        """hidden1 -> fc2 -> spike -> 2nd hidden spikes (no delays anywhere)."""
        return self.slayer.spike(self.fc2(self.slayer.psp(hidden1)))

    def _output(self, hidden2: torch.Tensor) -> torch.Tensor:
        """hidden2 -> fc3 -> spike (no delays anywhere)."""
        return self.slayer.spike(self.fc3(self.slayer.psp(hidden2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean forward pass (equivalent to the sigma = 0 baseline)."""
        x = self._prepare_input(x)
        hidden2 = self._second_hidden(self._first_hidden(x))
        return self._output(hidden2)

    def forward_with_hidden_perturbation(
        self,
        x: torch.Tensor,
        sigma: float = 0.0,
        site: str = "both",
    ) -> torch.Tensor:
        """Eval-only: shift spikes at ``site`` before the readout.

        Must be called inside ``torch.no_grad()`` — ``shift_hidden_batch`` is not
        autograd-safe. ``sigma = 0`` reduces to the clean forward pass.

        ``site`` is one of ``SITES``: ``"l1"`` perturbs the 1st hidden layer only,
        ``"l2"`` the 2nd only, and ``"both"`` perturbs each in turn -- so the
        layer-2 insult lands on a layer that was itself computed from an
        already-perturbed layer 1. That is what makes ``"both"`` a harsher probe
        than either single site rather than their sum (document 7 §5 step 3).

        Args:
            x: Input spike trains.
            sigma: Shift std dev in time steps (ms).

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
        if sigma > 0 and site in ("l1", "both"):
            hidden1 = shift_hidden_batch(hidden1, sigma)
        hidden2 = self._second_hidden(hidden1)
        if sigma > 0 and site in ("l2", "both"):
            hidden2 = shift_hidden_batch(hidden2, sigma)
        return self._output(hidden2)


def load_checkpoint(
    checkpoint_path: Path,
    input_dim: int = INPUT_DIM,
) -> SparseSHDNetwork:
    """Load a trained checkpoint into a fresh no-delay network in eval mode.

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
def test_at_sigma(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    sigma: float,
    site: str,
) -> float:
    """Return test accuracy with ``site`` shifted at ``sigma``.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        sigma: Shift std dev in time steps (ms); 0 is the clean baseline.
        site: Injection site, one of ``SITES``.

    Returns:
        Fraction of correctly classified test samples.
    """
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        outputs = net.forward_with_hidden_perturbation(
            x_batch, sigma=sigma, site=site)
        pred = snn.predict.getClass(outputs)
        correct += (pred.cpu() == y_batch.cpu()).sum().item()
        total += y_batch.size(0)
    return correct / max(1, total)


def test_with_repeats(
    net: SparseSHDNetwork,
    test_loader: DataLoader,
    sigma: float,
    num_repeats: int,
    site: str,
) -> dict:
    """Repeat the shifted evaluation ``num_repeats`` times for error bars.

    Each repeat re-seeds torch (which seeds the CUDA generator the shift draws
    from) so the perturbation differs run to run but the whole sweep is
    reproducible. ``sigma = 0`` involves no draw, so its repeats are identical and
    its std must come out at 0 — a useful check that no other randomness leaks into
    evaluation.

    Args:
        net: Loaded network in eval mode.
        test_loader: Test DataLoader.
        sigma: Shift std dev in time steps (ms).
        num_repeats: Number of repeats.
        site: Injection site, one of ``SITES``.

    Returns:
        Dict with keys ``mean``, ``std``, ``values``.
    """
    accuracies = []
    for repeat in range(num_repeats):
        torch.manual_seed(SEED + repeat)
        np.random.seed(SEED + repeat)
        accuracies.append(test_at_sigma(net, test_loader, sigma, site))
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
            f"sn_bothLayer_train_noDelay_v3.py first — this script only evaluates "
            f"existing checkpoints. Both both-layer grids were calibrated and "
            f"probed but not launched as of document 7, so this is the expected "
            f"failure until they are."
        )
    with open(summary_path) as fp:
        return json.load(fp)


def run_shift_sweep(test_loader: DataLoader) -> dict:
    """Sweep eval-only shift at every site and checkpoint, and save.

    Args:
        test_loader: The shared (fixed-split) test DataLoader.

    Returns:
        The payload written to disk: ``per_setup`` (one seed-averaged row per
        ``(k, floor)`` cell) and ``per_checkpoint`` (the raw per-seed rows).
    """
    sigma_values, num_repeats, max_checkpoints = resolve_eval_config()
    train_summary = load_train_summary()

    run_tags = list(train_summary)
    if max_checkpoints is not None:
        run_tags = run_tags[:max_checkpoints]

    print(f"\n{'#' * 70}")
    print(f"# v3 Phase 1, layers {list(TARGET_LAYERS)} (eval-only shift)")
    print(f"# arm: no-delay | dataset: SHD {DATASET_KEY} | grid: {VERSION_TAG} | "
          f"QUICK_TEST={QUICK_TEST}")
    print(f"# checkpoints: {len(run_tags)} | sigmas: {sigma_values} | "
          f"repeats: {num_repeats} | sites: {list(SITES)}")
    print(f"{'#' * 70}")

    results: dict[str, dict] = {}
    for index, run_tag in enumerate(run_tags, start=1):
        row = train_summary[run_tag]
        print(f"\n[{index}/{len(run_tags)}] {run_tag}")

        net = load_checkpoint(CKPT_DIR / f"{run_tag}.pt")

        site_sweeps = {}
        for site in SITES:
            sweep = {}
            for sigma in sigma_values:
                sweep_result = test_with_repeats(
                    net, test_loader, sigma, num_repeats, site)
                sweep[str(sigma)] = sweep_result
                print(
                    f"  [{site:>4}] sigma={sigma:<3} "
                    f"acc={sweep_result['mean']:.4f} "
                    f"+/- {sweep_result['std']:.4f}"
                )
            site_sweeps[f"sigma_sweep_{site}"] = sweep

        results[run_tag] = {
            "use_delay": False,
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
    aggregated = aggregate_sites(results, "sigma_sweep", sigma_values)
    payload = {
        "target_layers": list(TARGET_LAYERS),
        "sites": list(SITES),
        "per_setup": aggregated,
        "per_checkpoint": results,
    }

    results_path = (
        LOG_DIR
        / f"sparse_{DATASET_KEY}_{DELAY_TAG}_{VERSION_TAG}_shift_eval"
          f"{RUN_SUFFIX}.json"
    )
    with open(results_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nShift sweep saved to {results_path} "
          f"({len(results)} checkpoints -> {len(aggregated)} setups)")

    print_summary_table(aggregated, sigma_values)
    return payload


def aggregate_over_seeds(results: dict, sweep_field: str,
                         grid_values: list) -> dict:
    """Collapse the per-seed checkpoints into one row per ``(k, floor)`` setup.

    The seeds are replicates of the same factorial cell, so the *cell* is the unit
    the design is about, and a per-seed table invites reading noise as structure —
    the three seeds of one cell differ by up to .08 in clean accuracy here.

    Two quantities are averaged, and the two averages are **not** interchangeable:

    - ``acc(sigma)`` at each grid point is averaged directly across seeds, which is
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


def print_summary_table(aggregated: dict, sigma_values: list[int]) -> None:
    """Print one table per injection site, one row per ``(k, floor)`` setup.

    ``retention`` is the chance-corrected fraction of accuracy surviving the strongest
    shift; the analysis's ``temporal`` score is ``1 - retention``. It is printed here
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
        sigma_values: The sigma grid that was swept, in order.
    """
    grid_min, grid_max = str(sigma_values[0]), str(sigma_values[-1])

    for site in SITES:
        print(f"\n[site: {site}] {SITE_DESCRIPTIONS[site]}")
        print(
            f"{'setup':<14} {'n':>2} {'a_net':>12} {'s_net':>12} "
            f"{'a_l1':>6} {'s_l1':>7} {'a_l2':>6} {'s_l2':>7} "
            f"{'acc(' + grid_min + ')':>15} {'acc(' + grid_max + ')':>15} "
            f"{'retention':>15}"
        )
        for setup_tag, row in aggregated.items():
            sweep = row[f"sigma_sweep_{site}"]
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
    """Load the shared test set once, then sweep shift over the no-delay arm."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # The test split is fixed, so one loader serves every checkpoint.
    X, Y = load_shd_data(MAT_FILE, target_T=SIM_PARAMS["tSample"])
    test_loader = build_test_loader(X, Y, batch_size=BATCH_SIZE)

    run_shift_sweep(test_loader)


if __name__ == "__main__":
    main()
