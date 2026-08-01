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

What is measured, per checkpoint:

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

Relocation destinations are confined to ``[0, SUPPORT_BINS)``, the layer's measured
temporal support. Drawing them from the full 200-bin window — as v1 did — sends most
relocated spikes into the zero-padded tail, thinning the population's instantaneous
spike density and mixing a *rate* insult into what must be a timing-only probe. See
``temporal_support.py`` for the measurement and document 5 §2a for the size of the
artifact.

Eval-only: checkpoints are loaded frozen, nothing is trained, everything runs inside
``torch.no_grad()``.

Writes ``v3_analysis/log/phase1_measure_{arm}.json``.
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

# One past the last bin holding any 1st-hidden spike, measured over all 27 v1
# checkpoints by temporal_support.py.
SUPPORT_BINS: int = 88

# Sweep endpoints. f = 1 relocates every spike; p_d = 0.8 deletes 80% of them. Both
# match the grids the v1 eval scripts used, so v3 scores are comparable to v1's.
RELOCATION_F: float = 1.0
DELETION_PD: float = 0.8

# Repeats per perturbed measurement; both draws are stochastic.
NUM_REPEATS: int = 3

# Which training generation and arms to measure.
VERSION_TAG: str = "v3_"
ARMS: tuple[tuple[str, bool], ...] = (("delay", True), ("nodelay", False))


class SparseNetwork(nn.Module):
    """v1's architecture with an eval-only perturbation hook on the 1st hidden layer.

    The parameter set is v1's, so v1, v2 and v3 checkpoints all load directly — the v3
    factorial changed the training penalty, not the architecture.
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

    def forward(self, x: torch.Tensor, perturbation: str = "none",
                magnitude: float = 0.0) -> torch.Tensor:
        """Forward pass, optionally perturbing the 1st hidden layer.

        Args:
            x: Input spike trains, shape (B, 700, 1, 1, T).
            perturbation: ``"none"``, ``"relocate"`` or ``"delete"``.
            magnitude: ``f`` for relocation, ``p_d`` for deletion.

        Returns:
            Output spike tensor.
        """
        hidden = self.slayer.spike(self.fc1(self.slayer.psp(x)))
        if perturbation == "relocate" and magnitude > 0:
            hidden = relocate(hidden, magnitude, SUPPORT_BINS)
        elif perturbation == "delete" and magnitude > 0:
            hidden = delete(hidden, magnitude)
        if self.delays:
            hidden = self.delay1(hidden)
        out = self.slayer.spike(self.fc2(self.slayer.psp(hidden)))
        if self.delays:
            out = self.delay2(out)
        return self.slayer.spike(self.fc3(self.slayer.psp(out)))


def relocate(hidden_spikes: torch.Tensor, f: float,
             window: int | None = SUPPORT_BINS) -> torch.Tensor:
    """Move a fraction ``f`` of each neuron's spikes to random free bins in ``window``.

    Per-neuron spike count is preserved **exactly**, which is what makes this a
    timing-only probe: at ``f = 1`` the layer retains nothing but each neuron's count.
    Confining destinations to the measured support is the v3 correction; the support
    always has room, since every spike originated inside it.

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
        available = available.clone()
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
             perturbation: str, magnitude: float, seed: int) -> float:
    """Test accuracy at one perturbation setting."""
    torch.manual_seed(seed)
    correct = 0
    for start in range(0, len(inputs), BATCH_SIZE):
        batch = torch.from_numpy(inputs[start:start + BATCH_SIZE])
        batch = batch.unsqueeze(2).unsqueeze(3).to(device)
        predicted = snn.predict.getClass(
            net(batch, perturbation=perturbation, magnitude=magnitude)).cpu()
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


def measure_checkpoint(run_tag: str, delays: bool, inputs: np.ndarray,
                       labels: torch.Tensor) -> dict:
    """Measure usage and the deletion control for one checkpoint.

    Args:
        run_tag: Checkpoint name without extension.
        delays: Whether this is a with-delay checkpoint.
        inputs: Padded test-split spike trains.
        labels: Test-split labels.

    Returns:
        Dict of clean accuracy, both perturbed accuracies and both scores.
    """
    net = SparseNetwork(delays).to(device)
    net.load_state_dict(torch.load(CKPT_DIR / f"{run_tag}.pt", map_location=device))
    net.eval()

    clean = accuracy(net, inputs, labels, "none", 0.0, 0)
    relocated = [accuracy(net, inputs, labels, "relocate", RELOCATION_F, seed)
                 for seed in range(NUM_REPEATS)]
    deleted = [accuracy(net, inputs, labels, "delete", DELETION_PD, seed)
               for seed in range(NUM_REPEATS)]

    return {
        "clean_acc": clean,
        "acc_relocated": float(np.mean(relocated)),
        "acc_relocated_std": float(np.std(relocated)),
        "acc_deleted": float(np.mean(deleted)),
        "acc_deleted_std": float(np.std(deleted)),
        "usage": robustness_score(clean, float(np.mean(relocated))),
        "control_deletion": robustness_score(clean, float(np.mean(deleted))),
    }


def main() -> None:
    """Measure usage and the deletion control across every checkpoint in each arm."""
    print(f"Using device: {device}")
    print(f"relocation f={RELOCATION_F:g} over [0,{SUPPORT_BINS}) | "
          f"deletion p_d={DELETION_PD:g} | {NUM_REPEATS} repeats")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    inputs, labels = load_test_split()

    for arm, delays in ARMS:
        summary_path = (TRAIN_LOG_DIR
                        / f"sparse_whole_{arm}_{VERSION_TAG}train_summary.json")
        with open(summary_path) as handle:
            summary = json.load(handle)

        print(f"\n=== {arm}: {len(summary)} checkpoints ===")
        print(f"{'run_tag':<42} {'clean':>7} {'reloc':>7} {'del':>7} "
              f"{'usage':>7} {'control':>8}")
        results = {}
        for run_tag, meta in summary.items():
            row = measure_checkpoint(run_tag, delays, inputs, labels)
            # Carry the design variables through so the regression file is
            # self-contained and cannot be joined against the wrong summary.
            row.update({key: meta[key] for key in
                        ("ceiling_k", "floor_strength", "seed",
                         "spikes_per_neuron", "spikes_per_active_neuron",
                         "silent_fraction")})
            results[run_tag] = row
            print(f"{run_tag:<42} {row['clean_acc']:>7.3f} "
                  f"{row['acc_relocated']:>7.3f} {row['acc_deleted']:>7.3f} "
                  f"{row['usage']:>7.3f} {row['control_deletion']:>8.3f}",
                  flush=True)

        out_path = LOG_DIR / f"phase1_measure_{arm}.json"
        with open(out_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
