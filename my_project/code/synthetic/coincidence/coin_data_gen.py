"""Generate synthetic coincidence-detection spike train dataset.

Three neuron groups (g1, g2, g3) each of size GROUP_SIZE fire within
200 ms windows.  Class membership determines which two groups fire
in-phase (coincident) and which fires in anti-phase:

  Class A: g1 & g2 coincide, g3 anti-phase
  Class B: g1 & g3 coincide, g2 anti-phase
  Class C: g2 & g3 coincide, g1 anti-phase

Data are saved as per-lambda `.pt` files, then merged into a single
`coin_data.mat` file.
"""

import os

import numpy as np
import scipy.io
import torch
from scipy.stats import poisson

# ================== Configuration ==================
N_NEURONS_OVERLAP = 20      # Neurons used only when computing the lambda mapping.
GROUP_SIZE = 20             # Neurons per group.
N_NEURONS = GROUP_SIZE * 3  # Total neurons (3 groups).
N_TIMESTEPS = 4000
WINDOW_SIZE = 200
SPIKE_OFFSET_START = 50     # Spike window starts this many steps into each window.
SPIKE_OFFSET_END = 150      # Spike window ends this many steps into each window.

N_SAMPLES_PER_CLASS = 500
LAMBDA_PRIME_VALUES = [round(v, 1) for v in np.arange(0.0, 1.1, 0.1)]
CLASS_MAP = {"A": 0, "B": 1, "C": 2}

OUTPUT_DIR = "."
COMBINED_FILENAME = "coin_dataset.mat"


# ================== Probability Utilities ==================

def on_prob(k: int, rate: float) -> float:
    """Poisson PMF for the ON-state firing distribution.

    Args:
        k: Number of spikes.
        rate: Poisson rate parameter (mean).

    Returns:
        Probability mass at k.
    """
    return poisson.pmf(k, rate)


def off_prob(k: int, rate: float) -> float:
    """Poisson PMF for the OFF-state firing distribution.

    Args:
        k: Number of spikes.
        rate: Poisson rate parameter (mean).

    Returns:
        Probability mass at k.
    """
    return poisson.pmf(k, rate)


def normalize_probs(rate: float, state: str, max_k: int) -> np.ndarray:
    """Compute and normalise a Poisson PMF over [0, max_k].

    Args:
        rate: Poisson rate parameter.
        state: Either 'on' or 'off', selecting the probability function.
        max_k: Maximum spike count to include.

    Returns:
        Normalised probability array of length max_k + 1.

    Raises:
        ValueError: If *state* is not 'on' or 'off'.
    """
    if state == "on":
        raw = np.array([on_prob(k, rate) for k in range(max_k + 1)])
    elif state == "off":
        raw = np.array([off_prob(k, rate) for k in range(max_k + 1)])
    else:
        raise ValueError(f"state must be 'on' or 'off', got '{state}'.")
    return raw / raw.sum()


def compute_overlap(on_probs: np.ndarray, off_probs: np.ndarray) -> float:
    """Compute the probability overlap (intersection) between two distributions.

    Args:
        on_probs: Normalised ON-state probability array.
        off_probs: Normalised OFF-state probability array.

    Returns:
        Scalar overlap value in [0, 1].
    """
    return float(np.sum(np.minimum(on_probs, off_probs)))


# ================== Lambda Mapping ==================

def build_lambda_mapping(
    lambda_grid: np.ndarray,
    target_overlaps: np.ndarray,
    num_neurons: int,
) -> dict[float, float]:
    """Find the lambda value that achieves each target overlap level.

    Scans *lambda_grid* and records, for each target overlap, the lambda
    value whose resulting ON/OFF distribution overlap is closest to that target.

    Args:
        lambda_grid: 1-D array of candidate lambda values in [0, 1].
        target_overlaps: 1-D array of desired overlap levels.
        num_neurons: Maximum spike count used when normalising distributions.

    Returns:
        Dict mapping each target overlap to the best-matching lambda value.
        The key 0.0 is always mapped to 0.0.
    """
    best: dict[float, tuple[float, float]] = {
        v: (0.0, float("inf")) for v in target_overlaps
    }

    for lam in lambda_grid:
        on_mu = (1 - lam) * 12 + lam * 5
        off_mu = (1 - lam) * 2 + lam * 5
        on_probs = normalize_probs(on_mu, "on", num_neurons)
        off_probs = normalize_probs(off_mu, "off", num_neurons)
        overlap = compute_overlap(on_probs, off_probs)

        for target in target_overlaps:
            dist = abs(overlap - target)
            if dist < best[target][1]:
                best[target] = (lam, dist)

    mapping: dict[float, float] = {0.0: 0.0}
    for target, (lam, _) in best.items():
        mapping[round(float(target), 2)] = round(float(lam), 4)
    return mapping


# ================== Sample Generation ==================

def _group_distributions(
    class_type: str,
    on_probs: np.ndarray,
    off_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the firing distributions for (g1, g2, g3) given *class_type* and state.

    Args:
        class_type: One of 'A', 'B', or 'C'.
        on_probs: Normalised ON-state distribution.
        off_probs: Normalised OFF-state distribution.

    Returns:
        Tuple (g1_dist, g2_dist, g3_dist).

    Raises:
        ValueError: If *class_type* is not 'A', 'B', or 'C'.
    """
    if class_type == "A":
        g1_dist = on_probs
        g2_dist = on_probs
        g3_dist = off_probs
    elif class_type == "B":
        g1_dist = on_probs
        g2_dist = off_probs
        g3_dist = on_probs
    elif class_type == "C":
        g1_dist = off_probs
        g2_dist = on_probs
        g3_dist = on_probs
    else:
        raise ValueError(f"class_type must be 'A', 'B', or 'C', got '{class_type}'.")
    return g1_dist, g2_dist, g3_dist


def generate_sample(lam: float, class_type: str) -> np.ndarray:
    """Generate one spike-train sample for the given lambda and class.

    The trial is divided into non-overlapping windows of WINDOW_SIZE steps.
    Within each window, a random binary state is drawn; the two in-phase
    groups fire from the ON distribution and the anti-phase group fires from
    the OFF distribution (or vice versa, depending on the state).

    Args:
        lam: True lambda parameter controlling ON/OFF distribution overlap.
        class_type: One of 'A', 'B', or 'C'.

    Returns:
        Binary spike array of shape (N_NEURONS, N_TIMESTEPS).
    """
    on_mu = (1 - lam) * 12 + lam * 5
    off_mu = (1 - lam) * 2 + lam * 5
    on_probs = normalize_probs(on_mu, "on", GROUP_SIZE)
    off_probs = normalize_probs(off_mu, "off", GROUP_SIZE)

    spikes = np.zeros((N_NEURONS, N_TIMESTEPS), dtype=np.uint8)
    group1 = np.arange(0, 20)
    group2 = np.arange(20, 40)
    group3 = np.arange(40, 60)

    n_windows = N_TIMESTEPS // WINDOW_SIZE

    for window_idx in range(n_windows):
        window_start = window_idx * WINDOW_SIZE
        spike_start = window_start + SPIKE_OFFSET_START
        spike_end = min(window_start + SPIKE_OFFSET_END, N_TIMESTEPS)
        spike_window = np.arange(spike_start, spike_end)

        if len(spike_window) == 0 or spike_start >= N_TIMESTEPS:
            continue

        current_state = np.random.rand() < 0.5
        active_on = on_probs if current_state else off_probs
        active_off = off_probs if current_state else on_probs

        g1_dist, g2_dist, g3_dist = _group_distributions(
            class_type, active_on, active_off,
        )

        for group_neurons, dist in zip(
            [group1, group2, group3], [g1_dist, g2_dist, g3_dist]
        ):
            num_spikes = np.random.choice(np.arange(GROUP_SIZE + 1), p=dist)
            if num_spikes > 0:
                active = np.random.choice(GROUP_SIZE, size=num_spikes, replace=False)
                time_slots = np.random.choice(spike_window, size=num_spikes, replace=True)
                for neuron_idx, time_step in zip(group_neurons[active], time_slots):
                    spikes[neuron_idx, time_step] = 1

    return spikes


# ================== Dataset Generation ==================

def generate_and_save_per_lambda(
    lambda_prime_values: list[float],
    lambda_mapping: dict[float, float],
    output_dir: str,
) -> None:
    """Generate samples for each lambda value and save as individual .pt files.

    Args:
        lambda_prime_values: List of nominal lambda values (λ′).
        lambda_mapping: Dict mapping each λ′ to the true lambda used for generation.
        output_dir: Directory where per-lambda files are written.
    """
    for lam_prime in lambda_prime_values:
        lam_true = lambda_mapping[round(lam_prime, 1)]
        print(f"Generating for λ′ = {lam_prime:.1f} → λ = {lam_true:.4f}")

        samples_x: list[np.ndarray] = []
        samples_y: list[int] = []
        samples_lam: list[float] = []

        for class_type in ("A", "B", "C"):
            for _ in range(N_SAMPLES_PER_CLASS):
                spike = generate_sample(lam_true, class_type)
                samples_x.append(spike)
                samples_y.append(CLASS_MAP[class_type])
                samples_lam.append(lam_prime)

        x_array = np.array(samples_x, dtype=np.uint8)
        y_array = np.array(samples_y, dtype=np.int32)
        lam_array = np.array(samples_lam, dtype=np.float32)

        filename = os.path.join(
            output_dir, f"coin_data_lam{int(lam_prime * 10):02d}.pt"
        )
        torch.save({"X": x_array, "Y": y_array, "lambda": lam_array}, filename)
        print(
            f"  Saved {x_array.shape[0]} samples to {filename} "
            f"(X={x_array.shape}, Y={y_array.shape})"
        )


def combine_and_save(
    lambda_prime_values: list[float],
    output_dir: str,
    combined_filename: str,
) -> None:
    """Merge all per-lambda .pt files into a single .mat file.

    Args:
        lambda_prime_values: List of nominal lambda values used during generation.
        output_dir: Directory containing the per-lambda .pt files.
        combined_filename: Output path for the combined .mat file.
    """
    print("\nCombining all files into one dataset...")

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    lam_all: list[np.ndarray] = []

    for lam_prime in lambda_prime_values:
        filepath = os.path.join(
            output_dir, f"coin_data_lam{int(lam_prime * 10):02d}.pt"
        )
        data = torch.load(filepath, weights_only=False)
        x_all.extend(data["X"])
        y_all.extend(data["Y"])
        lam_all.extend(data["lambda"])

    x_combined = np.array(x_all, dtype=np.uint8)
    y_combined = np.array(y_all, dtype=np.int32)
    lam_combined = np.array(lam_all, dtype=np.float32)

    scipy.io.savemat(
        combined_filename,
        {"X": x_combined, "Y": y_combined, "lambda": lam_combined},
    )
    print(f"Combined dataset saved to {combined_filename}, X shape = {x_combined.shape}")


# ================== Main ==================

if __name__ == "__main__":
    lambda_grid = np.linspace(0, 1.0, 1001)
    target_overlaps = np.round(np.arange(0.1, 1.01, 0.1), 2)
    lambda_mapping = build_lambda_mapping(lambda_grid, target_overlaps, N_NEURONS_OVERLAP)

    generate_and_save_per_lambda(LAMBDA_PRIME_VALUES, lambda_mapping, OUTPUT_DIR)

    combined_path = os.path.join(OUTPUT_DIR, COMBINED_FILENAME)
    combine_and_save(LAMBDA_PRIME_VALUES, OUTPUT_DIR, combined_path)
