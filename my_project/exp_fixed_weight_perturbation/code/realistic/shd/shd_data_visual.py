import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


def load_mat(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load X and Y from a .mat file."""
    data = io.loadmat(path)
    X = data["X"]
    Y = data["Y"].ravel()
    return X, Y


def plot_raster(ax: plt.Axes, spike_train: np.ndarray, title: str) -> None:
    """Plot a spike raster for a single sample (neurons x time)."""
    neuron_idxs, time_idxs = np.where(spike_train == 1)
    ax.scatter(time_idxs, neuron_idxs, s=0.3, c="black", marker=".")
    ax.set_title(title)
    ax.set_ylabel("Neuron Index")
    ax.set_xlim(0, spike_train.shape[1])
    ax.set_ylim(0, spike_train.shape[0])


def main() -> None:
    data_dir = "./shd_data"

    X_whole, Y_whole = load_mat(f"{data_dir}/shd_whole.mat")
    X_part, Y_part = load_mat(f"{data_dir}/shd_part_new.mat")
    X_norm, Y_norm = load_mat(f"{data_dir}/shd_norm_new.mat")

    print(f"Whole: X={X_whole.shape}, Y={Y_whole.shape}")
    print(f"Part:  X={X_part.shape},  Y={Y_part.shape}")
    print(f"Norm:  X={X_norm.shape},  Y={Y_norm.shape}")

    # Pick a random sample from each dataset
    idx_whole = np.random.randint(len(Y_whole))
    idx_part = np.random.randint(len(Y_part))
    idx_norm = np.random.randint(len(Y_norm))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plot_raster(axes[0], X_whole[idx_whole], f"Whole (raw)\nsample={idx_whole}, label={Y_whole[idx_whole]}")
    plot_raster(axes[1], X_part[idx_part], f"Part (filtered)\nsample={idx_part}, label={Y_part[idx_part]}")
    plot_raster(axes[2], X_norm[idx_norm], f"Norm (min-count)\nsample={idx_norm}, label={Y_norm[idx_norm]}")

    plt.tight_layout()
    plt.savefig(f"{data_dir}/shd_data_visual.png", dpi=150)
    plt.show()
    print(f"Saved to {data_dir}/shd_data_visual.png")


if __name__ == "__main__":
    main()
