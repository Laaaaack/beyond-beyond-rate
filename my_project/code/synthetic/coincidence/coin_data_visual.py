"""Visualise the coincidence dataset produced by coin_data_gen.py.

Reproduces a figure similar to Figure 2(b): a 3×2 grid of raster plots
showing one example sample per class (A, B, C) at two lambda values.
Red horizontal lines mark group boundaries; alternating background bands
indicate which groups are ON (light) vs OFF (dark) for each class.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ================== Configuration ==================
OUTPUT_DIR = os.path.dirname(__file__)
CLASS_NAMES = ("A", "B", "C")
CLASS_LABELS = {0: "A", 1: "B", 2: "C"}
GROUP_SIZE = 20
GROUP_BOUNDARIES = [GROUP_SIZE, GROUP_SIZE * 2]   # Neuron indices of group splits.

# Which lambda values to display as columns (must match generated files).
DISPLAY_LAMBDAS = [0.0, 0.5]

# Background colours for the three groups, per class.
# ON groups get a light background; OFF groups get a slightly darker one.
# Layout: (class, group_index) → 'on' | 'off'
CLASS_GROUP_STATES = {
    "A": ["on",  "on",  "off"],
    "B": ["on",  "off", "on"],
    "C": ["off", "on",  "on"],
}
COLOR_ON  = "#d9d9d9"
COLOR_OFF = "#a0a0a0"
BOUNDARY_COLOR = "#e03030"
SPIKE_COLOR = "black"


# ================== Data Loading ==================

def load_lambda_file(output_dir: str, lam_prime: float) -> dict:
    """Load the .pt file for a given nominal lambda value.

    Args:
        output_dir: Directory containing the per-lambda .pt files.
        lam_prime: Nominal lambda value (e.g. 0.0, 0.5).

    Returns:
        Dict with keys 'X' (spike trains), 'Y' (labels), 'lambda'.
    """
    filename = os.path.join(
        output_dir, f"coin_data_lam{int(round(lam_prime * 10)):02d}.pt"
    )
    return torch.load(filename, weights_only=False)


def pick_example(
    data: dict,
    class_label: int,
    sample_index: int = 0,
) -> np.ndarray:
    """Return one spike train from *data* for the requested class.

    Args:
        data: Dict returned by load_lambda_file.
        class_label: Integer class label (0=A, 1=B, 2=C).
        sample_index: Which sample within the class to return.

    Returns:
        Binary array of shape (n_neurons, n_timesteps).
    """
    indices = np.where(np.array(data["Y"]) == class_label)[0]
    return np.array(data["X"][indices[sample_index]])


# ================== Plotting ==================

def _add_group_backgrounds(
    ax: plt.Axes,
    class_name: str,
    n_neurons: int,
) -> None:
    """Shade each neuron group background ON (light) or OFF (dark).

    Args:
        ax: Axes to draw on.
        class_name: One of 'A', 'B', 'C'.
        n_neurons: Total number of neurons.
    """
    states = CLASS_GROUP_STATES[class_name]
    group_edges = [0] + GROUP_BOUNDARIES + [n_neurons]

    for group_idx, (y_lo, y_hi) in enumerate(
        zip(group_edges[:-1], group_edges[1:])
    ):
        color = COLOR_ON if states[group_idx] == "on" else COLOR_OFF
        ax.axhspan(y_lo - 0.5, y_hi - 0.5, color=color, zorder=0)


def _draw_raster(
    ax: plt.Axes,
    spike_train: np.ndarray,
) -> None:
    """Draw spikes as vertical tick marks using vlines.

    Args:
        ax: Axes to draw on.
        spike_train: Binary array of shape (n_neurons, n_timesteps).
    """
    n_neurons = spike_train.shape[0]
    for neuron_idx in range(n_neurons):
        spike_times = np.where(spike_train[neuron_idx] == 1)[0]
        ax.vlines(
            spike_times,
            neuron_idx - 0.4,
            neuron_idx + 0.4,
            color=SPIKE_COLOR,
            linewidth=0.4,
            zorder=1,
        )


def _add_group_boundary_lines(ax: plt.Axes) -> None:
    """Draw red horizontal lines at group boundaries.

    Args:
        ax: Axes to draw on.
    """
    for boundary in GROUP_BOUNDARIES:
        ax.axhline(
            y=boundary - 0.5,
            color=BOUNDARY_COLOR,
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )


def plot_raster_grid(
    display_lambdas: list[float],
    output_dir: str,
    sample_index: int = 0,
) -> None:
    """Plot a 3-row × n-column raster grid (one column per lambda value).

    Args:
        display_lambdas: Lambda values to show as columns.
        output_dir: Directory containing the .pt files.
        sample_index: Which sample to pick from each class.
    """
    n_rows = len(CLASS_NAMES)
    n_cols = len(display_lambdas)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        sharex=True, sharey=True,
    )
    # Ensure axes is always 2-D.
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, lam_prime in enumerate(display_lambdas):
        data = load_lambda_file(output_dir, lam_prime)

        for row, class_name in enumerate(CLASS_NAMES):
            ax = axes[row, col]
            class_label = row   # A=0, B=1, C=2
            spike_train = pick_example(data, class_label, sample_index)

            n_neurons, n_timesteps = spike_train.shape

            _add_group_backgrounds(ax, class_name, n_neurons)
            _draw_raster(ax, spike_train)
            _add_group_boundary_lines(ax)

            ax.set_title(
                f"Class {class_name} | λ = {lam_prime:.1f}",
                fontsize=9, fontweight="bold",
            )
            ax.set_ylim(-0.5, n_neurons - 0.5)
            ax.set_xlim(0, n_timesteps)
            ax.set_ylabel("Neuron")
            ax.set_yticks([0, 20, 40, 60])

    for ax in axes[-1, :]:
        ax.set_xlabel("Time step (ms)")

    # Shared legend for group states.
    legend_handles = [
        mpatches.Patch(color=COLOR_ON,  label="ON group"),
        mpatches.Patch(color=COLOR_OFF, label="OFF group"),
        mpatches.Patch(color=BOUNDARY_COLOR, label="Group boundary"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Coincidence Dataset: Example Spike Trains per Class",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ================== Main ==================

if __name__ == "__main__":
    plot_raster_grid(DISPLAY_LAMBDAS, OUTPUT_DIR, sample_index=0)
