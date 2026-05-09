"""Generate synthetic ISI (Inter-Spike Interval) spike train dataset.

Produces a two-class dataset of spike trains where each sample is
characterised by a firing rate and an ISI value. Classes are separated
by a linear boundary in the (firing_rate, ISI) plane.  The result is
saved as an HDF5 file.
"""

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Set

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ================== Configuration ==================
MS = 1e-3

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

NUM_POINTS = 5000
PLANE_RADIUS = 10
MIN_DISTANCE_TO_BOUNDARY = 2.5
BOUNDARY_SLOPE = -1 / 2
BOUNDARY_INTERCEPT = 0
TIME_STEPS = 1000
NUM_NEURONS = 10
MAX_PLACEMENT_ATTEMPTS = 500
NUM_WORKERS = 4

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILENAME = "isi_dataset.h5"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)


# ================== Data Point Generation ==================

def generate_random_points(num_points: int, radius: float) -> np.ndarray:
    """Sample 2-D points uniformly within a square region.

    Args:
        num_points: Number of points to generate.
        radius: Half-width of the square (points lie in [-radius, radius]^2).

    Returns:
        Array of shape (num_points, 2).
    """
    return np.random.uniform(low=-radius, high=radius, size=(num_points, 2))


def compute_distances_to_line(
    points: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Compute perpendicular distances from *points* to the line y = slope*x + intercept.

    Args:
        points: Array of shape (N, 2).
        slope: Slope of the line.
        intercept: Y-intercept of the line.

    Returns:
        1-D array of distances, length N.
    """
    # Line in standard form: slope*x - y + intercept = 0
    numerator = np.abs(slope * points[:, 0] - points[:, 1] + intercept)
    denominator = np.sqrt(slope ** 2 + 1)
    return numerator / denominator


def assign_labels(
    points: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Assign binary class labels based on which side of the line points fall.

    Points above the line (y > slope*x + intercept) receive label 0;
    points below receive label 1.

    Args:
        points: Array of shape (N, 2).
        slope: Slope of the decision boundary.
        intercept: Y-intercept of the decision boundary.

    Returns:
        1-D uint8 array of labels, length N.
    """
    return np.where(
        points[:, 1] > slope * points[:, 0] + intercept, 0, 1
    ).astype(np.uint8)


# ================== Mapping Utilities ==================

def linear_map_to_steps(
    values: np.ndarray,
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
    step: float = 1.0,
) -> np.ndarray:
    """Linearly map *values* from [old_min, old_max] to [new_min, new_max], then snap to *step*.

    Args:
        values: Input array.
        old_min: Lower bound of the source range.
        old_max: Upper bound of the source range.
        new_min: Lower bound of the target range.
        new_max: Upper bound of the target range.
        step: Grid resolution to round to.

    Returns:
        Mapped and quantised array with the same shape as *values*.
    """
    mapped = (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    stepped = np.round(mapped / step) * step
    return np.clip(stepped, new_min, new_max)


# ================== Spike Train Generation ==================

def generate_spike_train(
    firing_rate: float,
    isi: float,
    num_neurons: int = 10,
    time_steps: int = 1000,
    max_attempts: int = 100,
) -> np.ndarray:
    """Generate a multi-neuron spike train with controlled ISI.

    Each neuron independently fires *pairs* of spikes separated by the
    given ISI.  The number of pairs per neuron is determined by the
    firing rate (in Hz) and the total duration.

    Args:
        firing_rate: Desired firing rate in Hz (may be fractional).
        isi: Desired inter-spike interval in ms.
        num_neurons: Number of independent neurons.
        time_steps: Duration of the spike train in ms.
        max_attempts: Maximum random placement attempts per pair.

    Returns:
        Binary array of shape (num_neurons, time_steps).
    """
    spike_trains = np.zeros((num_neurons, time_steps), dtype=np.uint8)

    duration_seconds = time_steps / 1000.0
    num_pairs = int(np.round(firing_rate * duration_seconds))
    isi_steps = max(1, int(round(isi)))

    for neuron_idx in range(num_neurons):
        occupied: Set[int] = set()

        for _ in range(num_pairs):
            placed = _try_place_spike_pair(
                spike_trains, neuron_idx, isi_steps, time_steps,
                max_attempts, occupied,
            )
            if not placed:
                continue

    return spike_trains


def _try_place_spike_pair(
    spike_trains: np.ndarray,
    neuron_idx: int,
    isi_steps: int,
    time_steps: int,
    max_attempts: int,
    occupied: Set[int],
) -> bool:
    """Attempt to place a pair of spikes for one neuron without conflicts.

    Args:
        spike_trains: Mutable spike-train array of shape (num_neurons, time_steps).
        neuron_idx: Index of the neuron to modify.
        isi_steps: Gap (in time-step units) between the two spikes in a pair.
        time_steps: Total number of time steps.
        max_attempts: How many random starts to try before giving up.
        occupied: Set of time indices already reserved (updated in place).

    Returns:
        True if the pair was successfully placed, False otherwise.
    """
    for _ in range(max_attempts):
        start_time = np.random.randint(0, time_steps - isi_steps)
        conflict_range = range(start_time - isi_steps, start_time + isi_steps + 1)

        if any(t in occupied for t in conflict_range):
            continue

        spike_trains[neuron_idx, start_time] = 1
        spike_trains[neuron_idx, start_time + isi_steps] = 1
        occupied.update(conflict_range)
        return True

    return False


# ================== Dataset Assembly ==================

def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate points, filter by distance, and compute firing rates / ISIs.

    Returns:
        Tuple of (points, labels, firing_rates, isis, col_min, col_max).
        col_min and col_max are the per-column extremes of the filtered points,
        needed to re-apply the same mapping elsewhere (e.g. for boundary plotting).
    """
    points = generate_random_points(NUM_POINTS, PLANE_RADIUS)

    distances = compute_distances_to_line(points, BOUNDARY_SLOPE, BOUNDARY_INTERCEPT)
    valid_mask = distances >= MIN_DISTANCE_TO_BOUNDARY
    points = points[valid_mask]

    labels = assign_labels(points, BOUNDARY_SLOPE, BOUNDARY_INTERCEPT)

    col_min = points.min(axis=0)
    col_max = points.max(axis=0)

    firing_rates = linear_map_to_steps(
        points[:, 0], col_min[0], col_max[0], 2, 10, step=2,
    )
    isis = linear_map_to_steps(
        points[:, 1], col_min[1], col_max[1], 1, 50, step=1,
    )

    return points, labels, firing_rates, isis, col_min, col_max


def plot_dataset(
    firing_rates: np.ndarray,
    isis: np.ndarray,
    labels: np.ndarray,
    col_min: np.ndarray,
    col_max: np.ndarray,
) -> None:
    """Show a scatter plot of the two-class dataset in (firing_rate, ISI) space.

    The decision boundary is mapped through the same linear transformation
    used to produce firing_rates and isis, so it aligns with the data.

    Args:
        firing_rates: 1-D array of firing rates.
        isis: 1-D array of ISI values.
        labels: 1-D array of binary class labels.
        col_min: Per-column minimum of the original filtered points (shape (2,)).
        col_max: Per-column maximum of the original filtered points (shape (2,)).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        firing_rates[labels == 0], isis[labels == 0],
        color="blue", label="Class 0",
    )
    plt.scatter(
        firing_rates[labels == 1], isis[labels == 1],
        color="red", label="Class 1",
    )

    # Sample the boundary in original space, then map both axes to match the data.
    # Use a continuous (unstepped) mapping so the line stays straight.
    x_raw = np.linspace(col_min[0], col_max[0], 200)
    y_raw = BOUNDARY_SLOPE * x_raw + BOUNDARY_INTERCEPT
    x_mapped = (x_raw - col_min[0]) / (col_max[0] - col_min[0]) * (10 - 2) + 2
    y_mapped = (y_raw - col_min[1]) / (col_max[1] - col_min[1]) * (50 - 1) + 1
    plt.plot(
        x_mapped, y_mapped, color="black", linestyle="--",
        label="Boundary line",
        #label=f"Boundary: y = {BOUNDARY_SLOPE}x + {BOUNDARY_INTERCEPT}",
    )

    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("ISI (ms)")
    plt.legend(loc="upper right")
    plt.title("Two-Class Dataset (Firing Rate & ISI Scaled)")
    plt.grid()
    plt.show()


def generate_all_spike_trains(
    firing_rates: np.ndarray,
    isis: np.ndarray,
) -> np.ndarray:
    """Generate spike trains for every sample using multiprocessing.

    Args:
        firing_rates: 1-D array of firing rates (Hz).
        isis: 1-D array of ISI values (ms).

    Returns:
        uint8 array of shape (num_samples, NUM_NEURONS, TIME_STEPS).
    """
    num_samples = len(firing_rates)

    print(
        f"Generating dataset with {num_samples} valid samples "
        f"using {NUM_WORKERS} workers..."
    )

    results: list[np.ndarray | None] = [None] * num_samples
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_worker_task, float(firing_rates[idx]), float(isis[idx])): idx
            for idx in range(num_samples)
        }
        for future in tqdm(
            as_completed(futures),
            total=num_samples,
            desc="Generating spike trains",
        ):
            sample_idx = futures[future]
            results[sample_idx] = future.result()

    return np.array(results, dtype=np.uint8)


def _worker_task(firing_rate: float, isi: float) -> np.ndarray:
    """Generate a single spike train (called inside a worker process).

    Args:
        firing_rate: Firing rate in Hz for this sample.
        isi: Inter-spike interval in ms for this sample.

    Returns:
        Spike train array of shape (NUM_NEURONS, TIME_STEPS).
    """
    return generate_spike_train(
        firing_rate,
        isi,
        num_neurons=NUM_NEURONS,
        time_steps=TIME_STEPS,
        max_attempts=MAX_PLACEMENT_ATTEMPTS,
    )


def save_dataset(
    filename: str,
    spike_trains: np.ndarray,
    labels: np.ndarray,
    firing_rates: np.ndarray,
    isis: np.ndarray,
) -> None:
    """Write the dataset to an HDF5 file with gzip compression.

    Args:
        filename: Output file path.
        spike_trains: Array of shape (N, NUM_NEURONS, TIME_STEPS).
        labels: 1-D array of class labels.
        firing_rates: 1-D array of firing rates.
        isis: 1-D array of ISI values.
    """
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("X", data=spike_trains, compression="gzip")
        hdf.create_dataset("Y", data=labels, compression="gzip")
        hdf.create_dataset("firing_rates", data=firing_rates, compression="gzip")
        hdf.create_dataset("isis", data=isis, compression="gzip")

    print(
        f"Dataset saved to {filename}, "
        f"X shape = {spike_trains.shape}, dtype={spike_trains.dtype}"
    )


# ================== Main ==================

if __name__ == "__main__":
    points, labels, firing_rates, isis, col_min, col_max = build_dataset()
    plot_dataset(firing_rates, isis, labels, col_min, col_max)

    spike_trains = generate_all_spike_trains(firing_rates, isis)
    save_dataset(OUTPUT_PATH, spike_trains, labels, firing_rates, isis)
