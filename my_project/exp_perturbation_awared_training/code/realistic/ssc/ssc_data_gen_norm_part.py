import gc
from pathlib import Path

import numpy as np
import h5py

# Directory of this script; all data paths are anchored here so the script can
# be launched from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent


# ==== Load SSC Whole ====
# after running ssc_whole_data_gen.py and getting ssc_whole.h5
input_path = str(SCRIPT_DIR / "ssc_data" / "ssc_whole.h5")
with h5py.File(input_path, "r") as f:
    X_all = f["X"][:]  # (N, F, T)
    Y_all = f["Y"][:].ravel()

print(f"Loaded SSC whole: X.shape={X_all.shape}, Y.shape={Y_all.shape}")
N, F, T = X_all.shape


# ==== Min-count utils ====
def do_min_count_inplace(X: np.ndarray) -> None:
    """Trim spikes in-place so each neuron has at most min_count spikes per sample.

    Mutates X directly to avoid allocating a second full-size array.
    """
    _, n_f, _ = X.shape
    count_all = X.sum(axis=2)          # (N_kept, F_kept)
    min_counts = count_all.min(axis=0) # (F_kept,)

    for f_idx in range(n_f):
        mc = int(min_counts[f_idx])
        if mc == 0:
            X[:, f_idx, :] = 0
            continue
        col_counts = count_all[:, f_idx]
        excess_idxs = np.where(col_counts > mc)[0]
        for i_idx in excess_idxs:
            spike_times = np.where(X[i_idx, f_idx, :] == 1)[0]
            X[i_idx, f_idx, :] = 0
            chosen_times = np.random.choice(spike_times, size=mc, replace=False)
            X[i_idx, f_idx, chosen_times] = 1


# ==== Balance util ====
def balance_and_save(X_in: np.ndarray, Y_in: np.ndarray, output_path: str) -> None:
    """Class-balance dataset by under-sampling, shuffle, and save to h5."""
    n_samples, num_neurons, n_t = X_in.shape
    print(f"\nBalancing: X.shape=({n_samples},{num_neurons},{n_t}), Y.shape=({len(Y_in)})")

    unique_labels = np.unique(Y_in)
    counts_per_class = {c: np.sum(Y_in == c) for c in unique_labels}
    min_count = min(counts_per_class.values())

    print("--- Sample counts per class ---")
    for c in sorted(unique_labels):
        print(f"Class {c}: {counts_per_class[c]} samples")
    print(f"Using min_count = {min_count}")

    X_list = []
    Y_list = []
    for c in sorted(unique_labels):
        idxs_c = np.where(Y_in == c)[0]
        np.random.shuffle(idxs_c)
        selected = idxs_c[:min_count]
        X_list.append(X_in[selected])
        Y_list.append(Y_in[selected])

    X_bal = np.concatenate(X_list, axis=0)
    Y_bal = np.concatenate(Y_list, axis=0)

    perm = np.random.permutation(len(Y_bal))
    X_bal = X_bal[perm]
    Y_bal = Y_bal[perm]

    print(f"Final balanced shape: X={X_bal.shape}, Y={Y_bal.shape}")
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("X", data=X_bal, compression="gzip")
        f_out.create_dataset("Y", data=Y_bal, compression="gzip")
    print(f"Saved balanced dataset to: {output_path}")


# ============================================================
# Step 1: Compute keep_idxs and non_zero_mask from counts only
#         (no full-size copy needed)
# ============================================================
counts = X_all.sum(axis=2).astype(np.int16)  # (N, F), ~147 MB
min_counts_per_neuron = counts.min(axis=0)

neuron_threshold = 2
max_frac_for_neuron = 0.02
max_samples_to_remove = 20000

bad_neurons = np.where(min_counts_per_neuron < neuron_threshold)[0]
print(f"Found {len(bad_neurons)} neurons with min_count < {neuron_threshold}.")

keep_idxs = np.arange(N)
if len(bad_neurons) > 0:
    samples_to_remove = set()
    for f_idx in bad_neurons:
        neuron_counts = counts[:, f_idx]
        i_bad = np.where(neuron_counts < neuron_threshold)[0]
        frac = len(i_bad) / N
        if frac <= max_frac_for_neuron:
            samples_to_remove.update(i_bad)

    if 0 < len(samples_to_remove) < max_samples_to_remove:
        keep_idxs = np.setdiff1d(np.arange(N), list(samples_to_remove))
        print(f"Removing {len(samples_to_remove)} samples.")
    else:
        print("NOT removing any samples.")

# Recompute min_counts for kept samples to determine which neurons survive
filtered_counts = counts[keep_idxs]             # (N_kept, F)
min_counts_filtered = filtered_counts.min(axis=0)  # (F,)
non_zero_mask = min_counts_filtered > 0
neuron_idxs = np.where(non_zero_mask)[0]

print(f"Keeping {len(keep_idxs)} samples, {len(neuron_idxs)}/{F} neurons")
del counts, filtered_counts, min_counts_per_neuron, min_counts_filtered
gc.collect()

# ============================================================
# Step 2: PART — original spikes, kept samples, kept neurons
#         np.ix_ selects rows+cols in one shot (no intermediate copy)
# ============================================================
print("\n==== Creating PART version ====")
X_part = X_all[np.ix_(keep_idxs, neuron_idxs, np.arange(T))]
Y_part = Y_all[keep_idxs]
balance_and_save(X_part, Y_part, str(SCRIPT_DIR / "ssc_data" / "ssc_part.h5"))
del X_part, Y_part
gc.collect()

# ============================================================
# Step 3: NORM — min-count normalised spikes, kept samples, kept neurons
#         Slice first (smaller array), then min-count in-place
# ============================================================
print("\n==== Creating NORM version ====")
X_norm = X_all[np.ix_(keep_idxs, neuron_idxs, np.arange(T))]
del X_all  # free the 6.9 GiB array before min-count processing
gc.collect()

do_min_count_inplace(X_norm)
Y_norm = Y_all[keep_idxs]

print(f"After min-count + drop zero neurons => X_norm.shape = {X_norm.shape}")
balance_and_save(X_norm, Y_norm, str(SCRIPT_DIR / "ssc_data" / "ssc_norm.h5"))
