"""Visualise the CCISI spike-train dataset produced by ccisi_data_gen.py.

Two figures are produced:
  1. Scatter plot  — firing rate vs. ISI, coloured by class.
  2. Raster plots  — example spike trains for one sample per class,
                     with neuron pairs (a, b) highlighted in matching colours.
"""

import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), "ccisi_dataset.h5")

CLASS_COLORS = {0: "steelblue", 1: "tomato"}
CLASS_NAMES = {0: "Class 0", 1: "Class 1"}

# Colours cycling through neuron pairs in the raster plot.
PAIR_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62",
]


# ================== Data Loading ==================

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the HDF5 dataset from *path*.

    Args:
        path: Path to the HDF5 file.

    Returns:
        Tuple of (spike_trains, labels, firing_rates, isis).
    """
    with h5py.File(path, "r") as hdf:
        spike_trains = hdf["X"][:]
        labels = hdf["Y"][:]
        firing_rates = hdf["firing_rates"][:]
        isis = hdf["isis"][:]
    return spike_trains, labels, firing_rates, isis


# ================== Plot 1: Scatter ==================

def plot_scatter(
    firing_rates: np.ndarray,
    isis: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Scatter plot of firing rate vs. ISI coloured by class.

    Args:
        firing_rates: 1-D array of firing rates (Hz).
        isis: 1-D array of ISI values (ms).
        labels: 1-D array of binary class labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for class_id in (0, 1):
        mask = labels == class_id
        ax.scatter(
            firing_rates[mask], isis[mask],
            color=CLASS_COLORS[class_id],
            label=CLASS_NAMES[class_id],
            alpha=0.5,
            s=10,
        )

    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("ISI (ms)")
    ax.set_title("Dataset Overview: Firing Rate vs. ISI")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


# ================== Plot 2: Raster ==================

def plot_raster_examples(
    spike_trains: np.ndarray,
    labels: np.ndarray,
    firing_rates: np.ndarray,
    isis: np.ndarray,
    num_examples: int = 3,
) -> None:
    """Raster plots for a few example samples from each class.

    Neurons are drawn in (neuron_a, neuron_b) pairs using matching colours so
    the cross-neuron ISI delay is visually apparent.

    Args:
        spike_trains: Array of shape (N, num_neurons, time_steps).
        labels: 1-D array of binary class labels.
        firing_rates: 1-D array of firing rates (Hz).
        isis: 1-D array of ISI values (ms).
        num_examples: Number of samples to show per class.
    """
    num_neurons = spike_trains.shape[1]
    num_pairs = num_neurons // 2

    fig, axes = plt.subplots(
        num_examples, 2,
        figsize=(14, 3 * num_examples),
        sharex=True,
    )
    fig.suptitle("Example Spike Trains per Class (CCISI)", fontsize=13)

    col_titles = [CLASS_NAMES[0], CLASS_NAMES[1]]

    for col, class_id in enumerate((0, 1)):
        indices = np.where(labels == class_id)[0][:num_examples]
        for row, sample_idx in enumerate(indices):
            ax = axes[row, col]
            train = spike_trains[sample_idx]  # (num_neurons, time_steps)

            for pair_idx in range(num_pairs):
                neuron_a = pair_idx * 2
                neuron_b = pair_idx * 2 + 1
                color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]

                for neuron_row, neuron_idx in enumerate(
                    (neuron_a, neuron_b), start=neuron_a
                ):
                    spike_times = np.where(train[neuron_idx] == 1)[0]
                    ax.vlines(
                        spike_times, neuron_row + 0.5, neuron_row + 1.5,
                        color=color, linewidth=0.8,
                    )

            fr = firing_rates[sample_idx]
            isi = isis[sample_idx]
            ax.set_ylabel("Neuron")
            ax.set_yticks(range(1, num_neurons + 1))
            ax.set_yticklabels(
                [f"{'a' if i % 2 == 0 else 'b'}{i // 2}" for i in range(num_neurons)],
                fontsize=6,
            )
            ax.set_title(
                f"{col_titles[col]} — FR={fr:.1f} Hz, ISI={isi:.0f} ms",
                fontsize=9,
            )
            ax.set_ylim(0.5, num_neurons + 0.5)
            ax.grid(axis="x", linestyle=":", alpha=0.4)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")

    # Legend: one entry per pair colour.
    legend_handles = [
        mpatches.Patch(color=PAIR_COLORS[i % len(PAIR_COLORS)], label=f"Pair {i}")
        for i in range(num_pairs)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=num_pairs,
        fontsize=7,
        title="Neuron pairs",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    plt.show()


# ================== Main ==================

if __name__ == "__main__":
    spike_trains, labels, firing_rates, isis = load_dataset(DATASET_PATH)

    print(
        f"Loaded dataset: {spike_trains.shape[0]} samples, "
        f"{spike_trains.shape[1]} neurons ({spike_trains.shape[1] // 2} pairs), "
        f"{spike_trains.shape[2]} time steps"
    )
    print(f"  Class 0: {(labels == 0).sum()} samples")
    print(f"  Class 1: {(labels == 1).sum()} samples")

    plot_scatter(firing_rates, isis, labels)
    plot_raster_examples(spike_trains, labels, firing_rates, isis, num_examples=3)
