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
DISPLAY_LAMBDAS = [0.0, 0.5, 1.0]

# Which two groups coincide per class — the third is anti-phase.
# Index into groups [g1=0, g2=1, g3=2].
CLASS_COINCIDENT_PAIR = {
    "A": (0, 1),   # g1 & g2 coincide, g3 anti-phase
    "B": (0, 2),   # g1 & g3 coincide, g2 anti-phase
    "C": (1, 2),   # g2 & g3 coincide, g1 anti-phase
}
WINDOW_SIZE = 200
COLOR_ON  = "#e0e0e0"
COLOR_OFF = "#b0b0b0"
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

def _infer_window_states(
    spike_train: np.ndarray,
    class_name: str,
) -> np.ndarray:
    """Infer per-window ON/OFF state from spike counts.

    For each 200ms window, count total spikes in the coincident pair of
    groups.  If the coincident pair fires more than the anti-phase group,
    the window is ON (True); otherwise OFF (False).

    Args:
        spike_train: Binary array of shape (n_neurons, n_timesteps).
        class_name: One of 'A', 'B', 'C'.

    Returns:
        Boolean array of length n_windows (True = ON window).
    """
    n_timesteps = spike_train.shape[1]
    n_windows = n_timesteps // WINDOW_SIZE
    pair = CLASS_COINCIDENT_PAIR[class_name]
    anti = ({0, 1, 2} - set(pair)).pop()

    group_slices = [
        slice(0, GROUP_SIZE),
        slice(GROUP_SIZE, GROUP_SIZE * 2),
        slice(GROUP_SIZE * 2, GROUP_SIZE * 3),
    ]

    states = np.zeros(n_windows, dtype=bool)
    for w in range(n_windows):
        t0 = w * WINDOW_SIZE
        t1 = t0 + WINDOW_SIZE
        pair_count = (
            spike_train[group_slices[pair[0]], t0:t1].sum()
            + spike_train[group_slices[pair[1]], t0:t1].sum()
        )
        anti_count = spike_train[group_slices[anti], t0:t1].sum()
        # Coincident pair fires more on average → ON window.
        states[w] = pair_count / 2 > anti_count

    return states


def _add_group_backgrounds(
    ax: plt.Axes,
    class_name: str,
    spike_train: np.ndarray,
) -> None:
    """Shade each group per-window: ON windows light, OFF windows dark.

    The shading alternates every 200ms window, inferred from spike counts,
    so the visual matches what the data generator produced.

    Args:
        ax: Axes to draw on.
        class_name: One of 'A', 'B', 'C'.
        spike_train: Binary array of shape (n_neurons, n_timesteps).
    """
    ax.set_facecolor("white")
    n_neurons, n_timesteps = spike_train.shape
    n_windows = n_timesteps // WINDOW_SIZE
    group_edges = [0] + GROUP_BOUNDARIES + [n_neurons]

    pair = CLASS_COINCIDENT_PAIR[class_name]
    window_states = _infer_window_states(spike_train, class_name)

    for w in range(n_windows):
        x0 = w * WINDOW_SIZE
        x1 = x0 + WINDOW_SIZE
        for group_idx, (y_lo, y_hi) in enumerate(
            zip(group_edges[:-1], group_edges[1:])
        ):
            is_coincident = group_idx in pair
            # ON window: coincident groups are bright (ON), anti-phase is dark (OFF).
            # OFF window: reversed.
            if window_states[w]:
                color = COLOR_ON if is_coincident else COLOR_OFF
            else:
                color = COLOR_OFF if is_coincident else COLOR_ON
            ax.fill_between(
                [x0, x1], y_lo - 0.5, y_hi - 0.5,
                facecolor=color, edgecolor="none", zorder=0,
            )


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

            _add_group_backgrounds(ax, class_name, spike_train)
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
