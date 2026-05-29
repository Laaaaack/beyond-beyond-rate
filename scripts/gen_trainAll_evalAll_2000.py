"""Generate the four trainAll_evalAll_test/2000_rate notebooks.

For each f in F_VALUES we train a fresh model with hidden-layer perturbation
applied during the forward pass at level f, then evaluate that same model at
the same f. Sample rate Ts = 0.5 ms.

The notebooks incorporate:
- Bug 1 fix from sample_rate.md: input scaled by 1/Ts in _prepare_input.
- Bug 2 fix from sample_rate.md: MAX_DELAY in milliseconds (15 ISI / 10 CCISI),
  not doubled when porting to Ts=0.5.
- STE pattern from how_to_perturb_hidden_layer_during_training.md: gradients
  flow through the perturbation operation back into the upstream layers.
- Vectorised GPU perturb_hidden_batch (no CPU round-trip per batch).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(r"d:\IC_2025\IRP\workspace\my_project\code\trainAll_evalAll_test\synthetic\diff_rate_test\2000_rate")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def cell_imports() -> dict:
    return code(
        'import os\n'
        'import sys\n'
        'import json\n'
        'import random\n'
        'from typing import Optional\n'
        '\n'
        'import numpy as np\n'
        'import matplotlib.pyplot as plt\n'
        'import torch\n'
        'from torch import nn\n'
        'import torch.nn.functional as F\n'
        'from torch.utils.data import Dataset, DataLoader\n'
        'from tqdm import tqdm\n'
        'import h5py\n'
        '\n'
        'import slayerSNN as snn\n'
        '\n'
        'MS = 1e-3  # Millisecond constant\n'
        '\n'
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
        'print(f"Device: {device}")'
    )


def cell_params(*, with_delay: bool, max_delay_ms: int, seed: int) -> dict:
    delay_line = ""
    if with_delay:
        delay_line = (
            f'MAX_DELAY = {max_delay_ms}  # delay budget in milliseconds '
            '(SLAYER delay param is in ms, not bins)\n'
        )
    return code(
        '# --- SLAYER neuron and simulation descriptors (2000-rate setup) ---\n'
        '# Ts = 0.5 ms / step => 2000 Hz sampling; tSample = 1000 ms window.\n'
        'SIM_PARAMS = {"Ts": 0.5, "tSample": 1000}\n'
        'LIF_PARAMS = {\n'
        '    "type": "SRMALPHA",\n'
        '    "theta": 1,\n'
        '    "tauSr": 1,\n'
        '    "tauRho": 1,\n'
        '    "tauRef": 1,\n'
        '    "scaleRef": 2,\n'
        '    "scaleRho": 1,\n'
        '}\n'
        '\n'
        '# --- Data split ratios ---\n'
        'TRAIN_RANGE = (0.0, 0.6)\n'
        'VAL_RANGE = (0.6, 0.75)\n'
        'TEST_RANGE = (0.75, 0.9)\n'
        '\n'
        '# --- Training hyper-parameters ---\n'
        'HIDDEN_UNITS = 100\n'
        'EPOCHS = 301\n'
        'BATCH_SIZE = 32\n'
        'LEARNING_RATE = 0.001\n'
        f'SEED = {seed}\n'
        f'{delay_line}'
        '\n'
        '# --- Hidden-perturbation sweep (train at f AND eval at f) ---\n'
        'F_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]\n'
        'NUM_REPEATS = 3  # repeated evaluations per f to get error bars'
    )


def cell_load_data(task: str) -> dict:
    if task == "isi":
        return code(
            'def load_isi_data(data_file: str = "isi_dataset.h5"):\n'
            '    """Load ISI dataset from HDF5 file.\n'
            '\n'
            '    Args:\n'
            '        data_file: Path to the HDF5 file.\n'
            '\n'
            '    Returns:\n'
            '        Tuple of (X, Y, firing_rates, isis) arrays.\n'
            '    """\n'
            '    with h5py.File(data_file, "r") as f:\n'
            '        X = f["X"][:]\n'
            '        Y = f["Y"][:].ravel()\n'
            '        firing_rates = f["firing_rates"][:] if "firing_rates" in f else None\n'
            '        isis = f["isis"][:] if "isis" in f else None\n'
            '\n'
            '    print(f"Loaded {data_file}: X={X.shape}, Y={Y.shape}")\n'
            '    print(f"Classes: {np.unique(Y)}, Time steps: {X.shape[2]}")\n'
            '    return X, Y, firing_rates, isis\n'
            '\n'
            '\n'
            'X_all, Y_all, firing_rates_all, isis_all = load_isi_data("isi_dataset.h5")\n'
            'NUM_NEURONS = X_all.shape[1]\n'
            'NUM_CLASSES = len(np.unique(Y_all))\n'
            'print(f"Network config: {NUM_NEURONS} input neurons, {NUM_CLASSES} classes")'
        )
    # ccisi
    return code(
        'def load_ccisi_data(data_file: str = "ccisi_dataset.h5"):\n'
        '    """Load CCISI dataset from HDF5 file.\n'
        '\n'
        '    Args:\n'
        '        data_file: Path to the HDF5 file.\n'
        '\n'
        '    Returns:\n'
        '        Tuple of (X, Y, firing_rates, isis) arrays.\n'
        '    """\n'
        '    with h5py.File(data_file, "r") as f:\n'
        '        X = f["X"][:]\n'
        '        Y = f["Y"][:].ravel()\n'
        '        firing_rates = f["firing_rates"][:] if "firing_rates" in f else None\n'
        '        isis = f["isis"][:] if "isis" in f else None\n'
        '\n'
        '    print(f"Loaded {data_file}: X={X.shape}, Y={Y.shape}")\n'
        '    print(f"Classes: {np.unique(Y)}, Time steps: {X.shape[2]}")\n'
        '    return X, Y, firing_rates, isis\n'
        '\n'
        '\n'
        'X_all, Y_all, firing_rates_all, isis_all = load_ccisi_data("ccisi_dataset.h5")\n'
        'NUM_NEURONS = X_all.shape[1]\n'
        'NUM_CLASSES = len(np.unique(Y_all))\n'
        'print(f"Network config: {NUM_NEURONS} input neurons, {NUM_CLASSES} classes")'
    )


def cell_perturb() -> dict:
    return code(
        'def partial_randomize_spike_train(\n'
        '    spike_train: np.ndarray,\n'
        '    f: float = 0.0,\n'
        '    max_attempts: int = 50,\n'
        ') -> np.ndarray:\n'
        '    """Randomly relocate a fraction *f* of each neuron\'s spikes (numpy reference).\n'
        '\n'
        '    Kept for documentation / numpy-only callers. The training and evaluation\n'
        '    paths use the vectorised GPU version ``perturb_hidden_batch`` below.\n'
        '\n'
        '    Args:\n'
        '        spike_train: Binary array of shape (num_neurons, T).\n'
        '        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).\n'
        '        max_attempts: Max placement retries per spike.\n'
        '\n'
        '    Returns:\n'
        '        Perturbed spike train with the same shape.\n'
        '    """\n'
        '    if f <= 0:\n'
        '        return spike_train\n'
        '\n'
        '    num_neurons, T = spike_train.shape\n'
        '    new_train = np.copy(spike_train)\n'
        '\n'
        '    for neuron_idx in range(num_neurons):\n'
        '        spike_times = np.where(spike_train[neuron_idx] == 1)[0]\n'
        '        num_to_move = int(len(spike_times) * f)\n'
        '        if num_to_move == 0:\n'
        '            continue\n'
        '\n'
        '        chosen = np.random.choice(spike_times, size=num_to_move, replace=False)\n'
        '        new_train[neuron_idx, chosen] = 0\n'
        '\n'
        '        placed = 0\n'
        '        for _ in range(max_attempts * num_to_move):\n'
        '            if placed >= num_to_move:\n'
        '                break\n'
        '            new_t = np.random.randint(0, T)\n'
        '            if new_train[neuron_idx, new_t] == 0:\n'
        '                new_train[neuron_idx, new_t] = 1\n'
        '                placed += 1\n'
        '\n'
        '    return new_train\n'
        '\n'
        '\n'
        '@torch.no_grad()\n'
        'def perturb_hidden_batch(\n'
        '    hidden_spikes: torch.Tensor,\n'
        '    f: float = 0.0,\n'
        ') -> torch.Tensor:\n'
        '    """Vectorised GPU-side partial spike relocation.\n'
        '\n'
        '    For each (batch, neuron), a fraction *f* of the existing spikes are\n'
        '    removed and replaced with the same number of spikes placed at randomly\n'
        '    chosen previously-unoccupied time bins. Spike count per neuron is\n'
        '    preserved exactly. All operations stay on the input tensor\'s device,\n'
        '    avoiding the CPU/numpy round-trip that dominates training cost when\n'
        '    perturbation runs on every batch.\n'
        '\n'
        '    Args:\n'
        '        hidden_spikes: SLAYER-format tensor of shape (B, C, 1, 1, T). At Ts < 1,\n'
        '            SLAYER emits spike values of 1/Ts; the > 0.5 threshold below works\n'
        '            for any sane Ts.\n'
        '        f: Fraction of spikes to relocate (0 = untouched, 1 = fully random).\n'
        '\n'
        '    Returns:\n'
        '        Perturbed tensor with the same shape, dtype, and device.\n'
        '    """\n'
        '    if f <= 0:\n'
        '        return hidden_spikes\n'
        '\n'
        '    B, C, H, W, T = hidden_spikes.shape\n'
        '    x = hidden_spikes.view(B, C, T)\n'
        '    is_spike = x > 0.5\n'
        '\n'
        '    n_spikes = is_spike.sum(dim=-1, keepdim=True)\n'
        '    num_to_move = (n_spikes.float() * f).floor().long()\n'
        '\n'
        '    key = torch.rand_like(x)\n'
        '    key = torch.where(is_spike, key, torch.full_like(key, 2.0))\n'
        '    rank = key.argsort(dim=-1).argsort(dim=-1)\n'
        '    remove_mask = rank < num_to_move\n'
        '    keep_mask = is_spike & ~remove_mask\n'
        '\n'
        '    available = ~keep_mask\n'
        '    key2 = torch.rand_like(x)\n'
        '    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))\n'
        '    rank2 = key2.argsort(dim=-1).argsort(dim=-1)\n'
        '    add_mask = rank2 < num_to_move\n'
        '\n'
        '    # Preserve SLAYER\'s 1/Ts spike amplitude (not just binary 1.0).\n'
        '    spike_value = x[is_spike].max() if is_spike.any() else torch.tensor(1.0, device=x.device)\n'
        '    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype) * spike_value\n'
        '    return new_spikes.view(B, C, H, W, T)'
    )


def cell_dataset_split() -> dict:
    return code(
        'class SpikeDataset(Dataset):\n'
        '    """Wrap numpy spike trains and labels into a PyTorch Dataset."""\n'
        '\n'
        '    def __init__(self, X: np.ndarray, Y: np.ndarray):\n'
        '        self.X = X\n'
        '        self.Y = Y\n'
        '\n'
        '    def __len__(self) -> int:\n'
        '        return len(self.Y)\n'
        '\n'
        '    def __getitem__(self, idx: int):\n'
        '        x = torch.tensor(self.X[idx], dtype=torch.float32)\n'
        '        y = torch.tensor(self.Y[idx], dtype=torch.long)\n'
        '        return x, y\n'
        '\n'
        '\n'
        'def get_split_indices(split_range: tuple[float, float], total: int) -> np.ndarray:\n'
        '    """Return integer indices for the given fractional range."""\n'
        '    return np.arange(int(split_range[0] * total), int(split_range[1] * total))\n'
        '\n'
        '\n'
        'def build_dataloaders(\n'
        '    X: np.ndarray,\n'
        '    Y: np.ndarray,\n'
        '    batch_size: int = 32,\n'
        ') -> tuple[DataLoader, DataLoader, DataLoader]:\n'
        '    """Split data and return train / val / test DataLoaders.\n'
        '\n'
        '    Inputs are stored unperturbed; the hidden-layer perturbation is applied\n'
        '    inside the network forward pass, parameterised by ``f``.\n'
        '    """\n'
        '    train_idx = get_split_indices(TRAIN_RANGE, len(X))\n'
        '    val_idx = get_split_indices(VAL_RANGE, len(X))\n'
        '    test_idx = get_split_indices(TEST_RANGE, len(X))\n'
        '\n'
        '    train_ds = SpikeDataset(X[train_idx], Y[train_idx])\n'
        '    val_ds = SpikeDataset(X[val_idx], Y[val_idx])\n'
        '    test_ds = SpikeDataset(X[test_idx], Y[test_idx])\n'
        '\n'
        '    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)\n'
        '    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)\n'
        '    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)\n'
        '\n'
        '    print(f"Split sizes -- Train: {len(train_ds)}, Val: {len(val_ds)}, '
        'Test: {len(test_ds)}")\n'
        '    return train_loader, val_loader, test_loader'
    )


def cell_network_tau(class_name: str) -> dict:
    return code(
        f'class {class_name}(nn.Module):\n'
        '    """SLAYER SNN with learnable PSP filter (tau), no delays.\n'
        '\n'
        '    Hidden-layer spike perturbation is built into ``forward`` via a\n'
        '    straight-through estimator so gradients flow through to the\n'
        '    upstream layers even when training at f > 0.\n'
        '    """\n'
        '\n'
        '    def __init__(\n'
        '        self,\n'
        '        num_neurons: int,\n'
        '        num_classes: int,\n'
        '        hidden_units: int = 100,\n'
        '    ):\n'
        '        super().__init__()\n'
        '        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)\n'
        '        self.slayer = slayer\n'
        '\n'
        '        self.fc1 = nn.utils.weight_norm(\n'
        '            slayer.dense(num_neurons, hidden_units), name="weight",\n'
        '        )\n'
        '        self.fc2 = nn.utils.weight_norm(\n'
        '            slayer.dense(hidden_units, num_classes), name="weight",\n'
        '        )\n'
        '\n'
        '        # filterLength = 100 at Ts=0.5 covers the same 50 ms window as\n'
        '        # filterLength = 50 at Ts=1.\n'
        '        self.psp_filter = slayer.pspFilter(\n'
        '            nFilter=1, filterLength=100, filterScale=1,\n'
        '        )\n'
        '        self._initialize_alpha_filter()\n'
        '\n'
        '    def _initialize_alpha_filter(self) -> None:\n'
        '        """Seed the learnable PSP filter with an alpha-function shape."""\n'
        '        tau = 50 * MS\n'
        '        Ts = self.slayer.simulation["Ts"] * MS\n'
        '        filt_len = self.psp_filter.weight.shape[-1]\n'
        '\n'
        '        alpha_kernel = np.array([\n'
        '            t / tau * np.exp(1 - t / tau)\n'
        '            for t in np.arange(0, filt_len * Ts, Ts)\n'
        '        ])\n'
        '        if np.max(np.abs(alpha_kernel)) > 0:\n'
        '            alpha_kernel /= np.max(np.abs(alpha_kernel))\n'
        '\n'
        '        with torch.no_grad():\n'
        '            self.psp_filter.weight.data = torch.FloatTensor(\n'
        '                np.flip(alpha_kernel).copy()\n'
        '            ).reshape(self.psp_filter.weight.shape)\n'
        '\n'
        '    def get_tau(self) -> torch.Tensor:\n'
        '        """Estimate the effective tau (peak position) of the learned filter."""\n'
        '        weights = self.psp_filter.weight.data.squeeze().cpu().numpy()\n'
        '        weights = np.flip(weights)\n'
        '        if len(weights) > 0:\n'
        '            peak_idx = np.argmax(np.abs(weights))\n'
        '            estimated_tau = 3 * peak_idx * self.slayer.simulation["Ts"] * MS\n'
        '            return torch.tensor(max(estimated_tau, 10 * MS))\n'
        '        return torch.tensor(50 * MS)\n'
        '\n'
        '    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:\n'
        '        """Ensure 5-D NCHWT on device and scale 0/1 to SLAYER\'s 1/Ts convention.\n'
        '\n'
        '        Fix for sample_rate.md Bug 1: our HDF5 datasets store binary 0/1\n'
        '        spike trains but SLAYER\'s internal spike value is 1/Ts. Without this\n'
        '        scaling, input-side PSPs come out at half magnitude at Ts=0.5.\n'
        '        """\n'
        '        if isinstance(x, np.ndarray):\n'
        '            x = torch.from_numpy(x)\n'
        '        if x.dim() == 3:\n'
        '            x = x.unsqueeze(2).unsqueeze(3)\n'
        '        return x.float().to(device) / self.slayer.simulation["Ts"]\n'
        '\n'
        '    def _first_layer(self, x: torch.Tensor) -> torch.Tensor:\n'
        '        """Input -> hidden spikes (learnable PSP + fc1 + spike)."""\n'
        '        x_filtered = self.psp_filter(x)\n'
        '        return self.slayer.spike(self.fc1(x_filtered))\n'
        '\n'
        '    def _second_layer(self, hidden_spikes: torch.Tensor) -> torch.Tensor:\n'
        '        """Hidden spikes -> output spikes (standard PSP + fc2 + spike)."""\n'
        '        return self.slayer.spike(self.fc2(self.slayer.psp(hidden_spikes)))\n'
        '\n'
        '    def _apply_perturbation(\n'
        '        self,\n'
        '        hidden_spikes: torch.Tensor,\n'
        '        f: float,\n'
        '    ) -> torch.Tensor:\n'
        '        """Straight-through estimator wrapper around perturb_hidden_batch.\n'
        '\n'
        '        Forward value = perturbed; backward gradient = identity through\n'
        '        ``hidden_spikes``. Without this, the perturbation breaks the\n'
        '        autograd graph and upstream layers (fc1, psp_filter) get zero\n'
        '        gradient during training.\n'
        '        """\n'
        '        if f <= 0:\n'
        '            return hidden_spikes\n'
        '        perturbed = perturb_hidden_batch(hidden_spikes, f)\n'
        '        return hidden_spikes + (perturbed - hidden_spikes).detach()\n'
        '\n'
        '    def forward(self, x: torch.Tensor, f: float = 0.0) -> torch.Tensor:\n'
        '        """Forward pass with hidden-layer perturbation at level *f*.\n'
        '\n'
        '        Args:\n'
        '            x: Input spike trains (numpy or tensor, 3-D or 5-D).\n'
        '            f: Perturbation fraction. 0 disables perturbation entirely.\n'
        '\n'
        '        Returns:\n'
        '            Output spike tensor.\n'
        '        """\n'
        '        x = self._prepare_input(x)\n'
        '        hidden_spikes = self._first_layer(x)\n'
        '        hidden_spikes = self._apply_perturbation(hidden_spikes, f)\n'
        '        return self._second_layer(hidden_spikes)'
    )


def cell_network_delay(class_name: str) -> dict:
    return code(
        f'class {class_name}(nn.Module):\n'
        '    """SLAYER SNN with learnable PSP filter (tau) AND learnable axonal delays.\n'
        '\n'
        '    Hidden-layer spike perturbation is built into ``forward`` via a\n'
        '    straight-through estimator so gradients flow through to the\n'
        '    upstream layers (fc1, psp_filter, delay1) even when training at f > 0.\n'
        '\n'
        '    delay2 is applied AFTER the perturbation hook so the binary spike\n'
        '    tensor handed to perturb_hidden_batch is strictly the output of\n'
        '    slayer.spike (no continuous-valued shifts).\n'
        '    """\n'
        '\n'
        '    def __init__(\n'
        '        self,\n'
        '        num_neurons: int,\n'
        '        num_classes: int,\n'
        '        hidden_units: int = 100,\n'
        '        max_delay: int = 15,\n'
        '    ):\n'
        '        super().__init__()\n'
        '        self.max_delay = max_delay\n'
        '        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)\n'
        '        self.slayer = slayer\n'
        '\n'
        '        self.fc1 = nn.utils.weight_norm(\n'
        '            slayer.dense(num_neurons, hidden_units), name="weight",\n'
        '        )\n'
        '        self.fc2 = nn.utils.weight_norm(\n'
        '            slayer.dense(hidden_units, num_classes), name="weight",\n'
        '        )\n'
        '\n'
        '        self.psp_filter = slayer.pspFilter(\n'
        '            nFilter=1, filterLength=100, filterScale=1,\n'
        '        )\n'
        '        self._initialize_alpha_filter()\n'
        '\n'
        '        # SLAYER delay parameter is in milliseconds (sample_rate.md Bug 2).\n'
        '        self.delay1 = slayer.delay(num_neurons)\n'
        '        self.delay2 = slayer.delay(hidden_units)\n'
        '        self._initialize_delays()\n'
        '\n'
        '    def _initialize_alpha_filter(self) -> None:\n'
        '        """Seed the learnable PSP filter with an alpha-function shape."""\n'
        '        tau = 50 * MS\n'
        '        Ts = self.slayer.simulation["Ts"] * MS\n'
        '        filt_len = self.psp_filter.weight.shape[-1]\n'
        '        alpha_kernel = np.array([\n'
        '            t / tau * np.exp(1 - t / tau)\n'
        '            for t in np.arange(0, filt_len * Ts, Ts)\n'
        '        ])\n'
        '        if np.max(np.abs(alpha_kernel)) > 0:\n'
        '            alpha_kernel /= np.max(np.abs(alpha_kernel))\n'
        '        with torch.no_grad():\n'
        '            self.psp_filter.weight.data = torch.FloatTensor(\n'
        '                np.flip(alpha_kernel).copy()\n'
        '            ).reshape(self.psp_filter.weight.shape)\n'
        '\n'
        '    def _initialize_delays(self) -> None:\n'
        '        """Initialize delay parameters uniformly in [0, max_delay] ms."""\n'
        '        with torch.no_grad():\n'
        '            if hasattr(self.delay1, "delay"):\n'
        '                self.delay1.delay.data.uniform_(0, self.max_delay)\n'
        '            if hasattr(self.delay2, "delay"):\n'
        '                self.delay2.delay.data.uniform_(0, self.max_delay)\n'
        '\n'
        '    def get_tau(self) -> torch.Tensor:\n'
        '        """Estimate the effective tau (peak position) of the learned filter."""\n'
        '        weights = self.psp_filter.weight.data.squeeze().cpu().numpy()\n'
        '        weights = np.flip(weights)\n'
        '        if len(weights) > 0:\n'
        '            peak_idx = np.argmax(np.abs(weights))\n'
        '            estimated_tau = 3 * peak_idx * self.slayer.simulation["Ts"] * MS\n'
        '            return torch.tensor(max(estimated_tau, 10 * MS))\n'
        '        return torch.tensor(50 * MS)\n'
        '\n'
        '    def get_delays(self) -> dict:\n'
        '        """Return delay parameters (in ms) as numpy arrays."""\n'
        '        delays = {}\n'
        '        if hasattr(self.delay1, "delay"):\n'
        '            delays["delay1"] = self.delay1.delay.data.cpu().numpy()\n'
        '        if hasattr(self.delay2, "delay"):\n'
        '            delays["delay2"] = self.delay2.delay.data.cpu().numpy()\n'
        '        return delays\n'
        '\n'
        '    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:\n'
        '        """Ensure 5-D NCHWT on device and scale 0/1 to SLAYER\'s 1/Ts convention.\n'
        '\n'
        '        Fix for sample_rate.md Bug 1: our HDF5 datasets store binary 0/1\n'
        '        spike trains but SLAYER\'s internal spike value is 1/Ts.\n'
        '        """\n'
        '        if isinstance(x, np.ndarray):\n'
        '            x = torch.from_numpy(x)\n'
        '        if x.dim() == 3:\n'
        '            x = x.unsqueeze(2).unsqueeze(3)\n'
        '        return x.float().to(device) / self.slayer.simulation["Ts"]\n'
        '\n'
        '    def _first_layer(self, x: torch.Tensor) -> torch.Tensor:\n'
        '        """Input -> hidden spikes (delay1 + learnable PSP + fc1 + spike)."""\n'
        '        x = self.delay1(x)\n'
        '        x_filtered = self.psp_filter(x)\n'
        '        return self.slayer.spike(self.fc1(x_filtered))\n'
        '\n'
        '    def _second_layer(self, hidden_spikes: torch.Tensor) -> torch.Tensor:\n'
        '        """Hidden spikes -> output spikes (delay2 + standard PSP + fc2 + spike)."""\n'
        '        x = self.delay2(hidden_spikes)\n'
        '        return self.slayer.spike(self.fc2(self.slayer.psp(x)))\n'
        '\n'
        '    def _apply_perturbation(\n'
        '        self,\n'
        '        hidden_spikes: torch.Tensor,\n'
        '        f: float,\n'
        '    ) -> torch.Tensor:\n'
        '        """Straight-through estimator wrapper around perturb_hidden_batch."""\n'
        '        if f <= 0:\n'
        '            return hidden_spikes\n'
        '        perturbed = perturb_hidden_batch(hidden_spikes, f)\n'
        '        return hidden_spikes + (perturbed - hidden_spikes).detach()\n'
        '\n'
        '    def forward(self, x: torch.Tensor, f: float = 0.0) -> torch.Tensor:\n'
        '        """Forward pass with hidden-layer perturbation at level *f*.\n'
        '\n'
        '        Args:\n'
        '            x: Input spike trains (numpy or tensor, 3-D or 5-D).\n'
        '            f: Perturbation fraction. 0 disables perturbation entirely.\n'
        '\n'
        '        Returns:\n'
        '            Output spike tensor.\n'
        '        """\n'
        '        x = self._prepare_input(x)\n'
        '        hidden_spikes = self._first_layer(x)\n'
        '        hidden_spikes = self._apply_perturbation(hidden_spikes, f)\n'
        '        return self._second_layer(hidden_spikes)'
    )


def cell_train_tau(class_name: str) -> dict:
    return code(
        'def train_model(\n'
        '    train_loader: DataLoader,\n'
        '    val_loader: DataLoader,\n'
        '    num_neurons: int,\n'
        '    num_classes: int,\n'
        '    train_f: float,\n'
        '    hidden_units: int = 100,\n'
        '    epochs: int = 301,\n'
        '    lr: float = 0.001,\n'
        '    seed: int = 42,\n'
        f') -> tuple["{class_name}", dict]:\n'
        '    """Train the network with hidden-layer perturbation at level ``train_f``.\n'
        '\n'
        '    Args:\n'
        '        train_loader: Training DataLoader (unperturbed inputs).\n'
        '        val_loader: Validation DataLoader (unperturbed inputs).\n'
        '        num_neurons: Number of input neurons.\n'
        '        num_classes: Number of output classes.\n'
        '        train_f: Perturbation fraction applied inside the network during training.\n'
        '        hidden_units: Hidden layer size.\n'
        '        epochs: Number of training epochs.\n'
        '        lr: Learning rate.\n'
        '        seed: Random seed for reproducibility.\n'
        '\n'
        '    Returns:\n'
        '        Tuple of (trained network, training log dict).\n'
        '    """\n'
        '    torch.manual_seed(seed)\n'
        '    np.random.seed(seed)\n'
        '    random.seed(seed)\n'
        '    if torch.cuda.is_available():\n'
        '        torch.cuda.manual_seed_all(seed)\n'
        '\n'
        f'    net = {class_name}(num_neurons, num_classes, hidden_units).to(device)\n'
        '\n'
        '    loss_fn = snn.spikeLoss.spikeLoss({\n'
        '        "neuron": LIF_PARAMS,\n'
        '        "simulation": SIM_PARAMS,\n'
        '        "training": {"error": {"type": "ProbSpikes"}},\n'
        '    }).to(device)\n'
        '\n'
        '    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=lr)\n'
        '    scheduler = torch.optim.lr_scheduler.MultiStepLR(\n'
        '        optimizer, milestones=[300], gamma=0.5,\n'
        '    )\n'
        '\n'
        '    best_val_loss = float("inf")\n'
        '    best_model_state = None\n'
        '    tau_history: list[float] = []\n'
        '\n'
        '    log = {"epoch": [], "train_loss": [], "val_loss": [], "tau": []}\n'
        '\n'
        '    total_steps = epochs * len(train_loader)\n'
        '    with tqdm(total=total_steps, desc=f"Train f={train_f:.2f}") as pbar:\n'
        '        for epoch in range(epochs):\n'
        '            net.train()\n'
        '            epoch_loss = 0.0\n'
        '            batch_count = 0\n'
        '\n'
        '            for x_batch, y_batch in train_loader:\n'
        '                if x_batch.dim() == 3:\n'
        '                    x_batch = x_batch.unsqueeze(2).unsqueeze(3)\n'
        '                x_batch = x_batch.to(device).float()\n'
        '                y_batch = y_batch.to(device).long()\n'
        '\n'
        '                outputs = net(x_batch, f=train_f)\n'
        '                loss = loss_fn.probSpikes(outputs, y_batch)\n'
        '                epoch_loss += loss.item()\n'
        '                batch_count += 1\n'
        '\n'
        '                optimizer.zero_grad()\n'
        '                loss.backward()\n'
        '                optimizer.step()\n'
        '                pbar.update(1)\n'
        '\n'
        '            net.eval()\n'
        '            val_loss = 0.0\n'
        '            with torch.no_grad():\n'
        '                for x_batch, y_batch in val_loader:\n'
        '                    if x_batch.dim() == 3:\n'
        '                        x_batch = x_batch.unsqueeze(2).unsqueeze(3)\n'
        '                    x_batch = x_batch.to(device).float()\n'
        '                    y_batch = y_batch.to(device).long()\n'
        '                    outputs = net(x_batch, f=train_f)\n'
        '                    val_loss += loss_fn.probSpikes(outputs, y_batch).item()\n'
        '\n'
        '            val_loss /= len(val_loader)\n'
        '            epoch_loss /= batch_count\n'
        '            tau_val = net.get_tau().item() / MS\n'
        '            tau_history.append(tau_val)\n'
        '\n'
        '            log["epoch"].append(epoch)\n'
        '            log["train_loss"].append(epoch_loss)\n'
        '            log["val_loss"].append(val_loss)\n'
        '            log["tau"].append(tau_val)\n'
        '\n'
        '            if val_loss < best_val_loss:\n'
        '                best_val_loss = val_loss\n'
        '                best_model_state = {k: v.clone() for k, v in net.state_dict().items()}\n'
        '\n'
        '            scheduler.step()\n'
        '            pbar.set_postfix(\n'
        '                epoch=epoch + 1,\n'
        '                val_loss=f"{val_loss:.4f}",\n'
        '                best=f"{best_val_loss:.4f}",\n'
        '                tau=f"{tau_val:.1f}ms",\n'
        '            )\n'
        '\n'
        '    if best_model_state is not None:\n'
        '        net.load_state_dict(best_model_state)\n'
        '\n'
        '    log["tau_history"] = tau_history\n'
        '    return net, log'
    )


def cell_train_delay(class_name: str) -> dict:
    return code(
        'def train_model(\n'
        '    train_loader: DataLoader,\n'
        '    val_loader: DataLoader,\n'
        '    num_neurons: int,\n'
        '    num_classes: int,\n'
        '    train_f: float,\n'
        '    hidden_units: int = 100,\n'
        '    max_delay: int = 15,\n'
        '    epochs: int = 301,\n'
        '    lr: float = 0.001,\n'
        '    seed: int = 42,\n'
        f') -> tuple["{class_name}", dict]:\n'
        '    """Train the delay network with hidden-layer perturbation at level ``train_f``.\n'
        '\n'
        '    Separate learning rates: regular weights at base LR, PSP filter (tau)\n'
        '    at LR x 10 for faster temporal adaptation, delays at LR x 5 for stable\n'
        '    delay learning.\n'
        '    """\n'
        '    torch.manual_seed(seed)\n'
        '    np.random.seed(seed)\n'
        '    random.seed(seed)\n'
        '    if torch.cuda.is_available():\n'
        '        torch.cuda.manual_seed_all(seed)\n'
        '\n'
        f'    net = {class_name}(num_neurons, num_classes, hidden_units, max_delay).to(device)\n'
        '\n'
        '    loss_fn = snn.spikeLoss.spikeLoss({\n'
        '        "neuron": LIF_PARAMS,\n'
        '        "simulation": SIM_PARAMS,\n'
        '        "training": {"error": {"type": "ProbSpikes"}},\n'
        '    }).to(device)\n'
        '\n'
        '    regular_params, tau_params, delay_params = [], [], []\n'
        '    for name, param in net.named_parameters():\n'
        '        if "delay" in name:\n'
        '            delay_params.append(param)\n'
        '        elif "psp_filter" in name:\n'
        '            tau_params.append(param)\n'
        '        else:\n'
        '            regular_params.append(param)\n'
        '\n'
        '    param_groups = [{"params": regular_params, "lr": lr}]\n'
        '    if tau_params:\n'
        '        param_groups.append({"params": tau_params, "lr": lr * 10})\n'
        '    if delay_params:\n'
        '        param_groups.append({"params": delay_params, "lr": lr * 5})\n'
        '\n'
        '    optimizer = snn.utils.optim.Nadam(param_groups)\n'
        '    scheduler = torch.optim.lr_scheduler.MultiStepLR(\n'
        '        optimizer, milestones=[300], gamma=0.5,\n'
        '    )\n'
        '\n'
        '    best_val_loss = float("inf")\n'
        '    best_model_state = None\n'
        '    tau_history: list[float] = []\n'
        '    delay_history: list[dict] = []\n'
        '\n'
        '    log = {\n'
        '        "epoch": [], "train_loss": [], "val_loss": [],\n'
        '        "tau": [], "delay_mean": [],\n'
        '    }\n'
        '\n'
        '    total_steps = epochs * len(train_loader)\n'
        '    with tqdm(total=total_steps, desc=f"Train f={train_f:.2f}") as pbar:\n'
        '        for epoch in range(epochs):\n'
        '            net.train()\n'
        '            epoch_loss = 0.0\n'
        '            batch_count = 0\n'
        '\n'
        '            for x_batch, y_batch in train_loader:\n'
        '                if x_batch.dim() == 3:\n'
        '                    x_batch = x_batch.unsqueeze(2).unsqueeze(3)\n'
        '                x_batch = x_batch.to(device).float()\n'
        '                y_batch = y_batch.to(device).long()\n'
        '\n'
        '                outputs = net(x_batch, f=train_f)\n'
        '                loss = loss_fn.probSpikes(outputs, y_batch)\n'
        '                epoch_loss += loss.item()\n'
        '                batch_count += 1\n'
        '\n'
        '                optimizer.zero_grad()\n'
        '                loss.backward()\n'
        '                optimizer.step()\n'
        '                pbar.update(1)\n'
        '\n'
        '            net.eval()\n'
        '            val_loss = 0.0\n'
        '            with torch.no_grad():\n'
        '                for x_batch, y_batch in val_loader:\n'
        '                    if x_batch.dim() == 3:\n'
        '                        x_batch = x_batch.unsqueeze(2).unsqueeze(3)\n'
        '                    x_batch = x_batch.to(device).float()\n'
        '                    y_batch = y_batch.to(device).long()\n'
        '                    outputs = net(x_batch, f=train_f)\n'
        '                    val_loss += loss_fn.probSpikes(outputs, y_batch).item()\n'
        '\n'
        '            val_loss /= len(val_loader)\n'
        '            epoch_loss /= batch_count\n'
        '            tau_val = net.get_tau().item() / MS\n'
        '            tau_history.append(tau_val)\n'
        '\n'
        '            delays = net.get_delays()\n'
        '            delay_history.append(delays)\n'
        '            avg_delay = (\n'
        '                float(np.mean([np.mean(d) for d in delays.values() if len(d) > 0]))\n'
        '                if delays else 0.0\n'
        '            )\n'
        '\n'
        '            log["epoch"].append(epoch)\n'
        '            log["train_loss"].append(epoch_loss)\n'
        '            log["val_loss"].append(val_loss)\n'
        '            log["tau"].append(tau_val)\n'
        '            log["delay_mean"].append(avg_delay)\n'
        '\n'
        '            if val_loss < best_val_loss:\n'
        '                best_val_loss = val_loss\n'
        '                best_model_state = {k: v.clone() for k, v in net.state_dict().items()}\n'
        '\n'
        '            scheduler.step()\n'
        '            pbar.set_postfix(\n'
        '                epoch=epoch + 1,\n'
        '                val_loss=f"{val_loss:.4f}",\n'
        '                best=f"{best_val_loss:.4f}",\n'
        '                tau=f"{tau_val:.1f}ms",\n'
        '                delay=f"{avg_delay:.1f}ms",\n'
        '            )\n'
        '\n'
        '    if best_model_state is not None:\n'
        '        net.load_state_dict(best_model_state)\n'
        '\n'
        '    log["tau_history"] = tau_history\n'
        '    log["delay_history_mean"] = [\n'
        '        float(np.mean([np.mean(d) for d in dh.values() if len(d) > 0])) if dh else 0.0\n'
        '        for dh in delay_history\n'
        '    ]\n'
        '    return net, log'
    )


def cell_test(class_name: str) -> dict:
    return code(
        'def test_with_hidden_perturbation(\n'
        f'    net: {class_name},\n'
        '    test_loader: DataLoader,\n'
        '    f: float = 0.0,\n'
        ') -> float:\n'
        '    """Evaluate accuracy with hidden-layer perturbation at level *f*."""\n'
        '    net.eval()\n'
        '    correct = 0\n'
        '    total = 0\n'
        '\n'
        '    with torch.no_grad():\n'
        '        for x_batch, y_batch in test_loader:\n'
        '            if x_batch.dim() == 3:\n'
        '                x_batch = x_batch.unsqueeze(2).unsqueeze(3)\n'
        '            x_batch = x_batch.to(device).float()\n'
        '            y_batch = y_batch.to(device)\n'
        '\n'
        '            outputs = net(x_batch, f=f)\n'
        '            predicted = snn.predict.getClass(outputs)\n'
        '\n'
        '            total += y_batch.size(0)\n'
        '            correct += (predicted.cpu() == y_batch.cpu()).sum().item()\n'
        '\n'
        '    return correct / total\n'
        '\n'
        '\n'
        'def eval_repeated(\n'
        f'    net: {class_name},\n'
        '    test_loader: DataLoader,\n'
        '    f: float,\n'
        '    num_repeats: int = 3,\n'
        '    base_seed: int = 0,\n'
        ') -> dict:\n'
        '    """Run multiple stochastic evaluations to get mean/std accuracy."""\n'
        '    accuracies: list[float] = []\n'
        '    for repeat in range(num_repeats):\n'
        '        torch.manual_seed(base_seed + repeat)\n'
        '        np.random.seed(base_seed + repeat)\n'
        '        accuracies.append(test_with_hidden_perturbation(net, test_loader, f=f))\n'
        '    return {\n'
        '        "mean": float(np.mean(accuracies)),\n'
        '        "std": float(np.std(accuracies)),\n'
        '        "values": [float(a) for a in accuracies],\n'
        '    }'
    )


def cell_plot_utils(*, with_delay: bool, task_label: str, variant_label: str) -> dict:
    plot_train = (
        'def plot_training_curves(log: dict, title_suffix: str = "") -> None:\n'
    )
    if with_delay:
        plot_train += (
            '    """Plot training/validation loss, tau and delay evolution."""\n'
            '    fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n'
            '\n'
            '    axes[0].plot(log["epoch"], log["train_loss"], "o-", label="Train", markersize=2)\n'
            '    axes[0].plot(log["epoch"], log["val_loss"], "s-", label="Val", markersize=2)\n'
            '    axes[0].set_xlabel("Epoch")\n'
            '    axes[0].set_ylabel("Loss")\n'
            '    axes[0].set_title(f"Loss {title_suffix}")\n'
            '    axes[0].legend()\n'
            '    axes[0].grid(True, alpha=0.3)\n'
            '\n'
            '    axes[1].plot(log["tau_history"])\n'
            '    axes[1].set_xlabel("Epoch")\n'
            '    axes[1].set_ylabel("Tau (ms)")\n'
            '    axes[1].set_title(f"Tau {title_suffix}")\n'
            '    axes[1].grid(True, alpha=0.3)\n'
            '\n'
            '    axes[2].plot(log["delay_history_mean"])\n'
            '    axes[2].set_xlabel("Epoch")\n'
            '    axes[2].set_ylabel("Mean delay (ms)")\n'
            '    axes[2].set_title(f"Delay {title_suffix}")\n'
            '    axes[2].grid(True, alpha=0.3)\n'
            '\n'
            '    plt.tight_layout()\n'
            '    plt.show()\n'
        )
    else:
        plot_train += (
            '    """Plot training/validation loss and tau evolution."""\n'
            '    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n'
            '\n'
            '    axes[0].plot(log["epoch"], log["train_loss"], "o-", label="Train", markersize=2)\n'
            '    axes[0].plot(log["epoch"], log["val_loss"], "s-", label="Val", markersize=2)\n'
            '    axes[0].set_xlabel("Epoch")\n'
            '    axes[0].set_ylabel("Loss")\n'
            '    axes[0].set_title(f"Loss {title_suffix}")\n'
            '    axes[0].legend()\n'
            '    axes[0].grid(True, alpha=0.3)\n'
            '\n'
            '    axes[1].plot(log["tau_history"])\n'
            '    axes[1].set_xlabel("Epoch")\n'
            '    axes[1].set_ylabel("Tau (ms)")\n'
            '    axes[1].set_title(f"Tau {title_suffix}")\n'
            '    axes[1].grid(True, alpha=0.3)\n'
            '\n'
            '    plt.tight_layout()\n'
            '    plt.show()\n'
        )

    color = "tab:orange" if with_delay else "tab:blue"
    plot_sweep = (
        '\n\n'
        'def plot_hidden_perturbation_curve(\n'
        '    sweep_results: dict,\n'
        '    save_path: str,\n'
        ') -> None:\n'
        '    """Plot accuracy vs hidden perturbation level f (train-at-f / eval-at-f)."""\n'
        '    f_vals = sorted(sweep_results.keys())\n'
        '    means = [sweep_results[f]["mean"] for f in f_vals]\n'
        '    stds = [sweep_results[f]["std"] for f in f_vals]\n'
        '\n'
        '    plt.figure(figsize=(8, 5))\n'
        '    plt.errorbar(\n'
        '        f_vals, means, yerr=stds, fmt="o-", capsize=5, capthick=2,\n'
        f'        color="{color}", label="{variant_label}",\n'
        '    )\n'
        '    plt.xlabel("Perturbation level f (train and eval)")\n'
        '    plt.ylabel("Test accuracy")\n'
        f'    plt.title("{task_label} (Ts=0.5) -- train-at-f / eval-at-f hidden perturbation")\n'
        '    plt.ylim(0, 1.05)\n'
        '    plt.grid(True, alpha=0.3)\n'
        '    plt.legend()\n'
        '\n'
        '    for f_val, mean in zip(f_vals, means):\n'
        '        plt.annotate(\n'
        '            f"{mean:.3f}", (f_val, mean),\n'
        '            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9,\n'
        '        )\n'
        '\n'
        '    plt.tight_layout()\n'
        '    plt.savefig(save_path, dpi=300, bbox_inches="tight")\n'
        '    plt.show()\n'
        '    print(f"Figure saved to {save_path}")'
    )
    return code(plot_train + plot_sweep)


def cell_run_loop(*, with_delay: bool, model_prefix: str) -> dict:
    delay_arg = "        max_delay=MAX_DELAY,\n" if with_delay else ""
    return code(
        'os.makedirs("data", exist_ok=True)\n'
        'os.makedirs("log", exist_ok=True)\n'
        '\n'
        '# Build DataLoaders once -- inputs are unperturbed; the hidden-layer\n'
        '# perturbation is parameterised by the f passed to the network forward.\n'
        'train_loader, val_loader, test_loader = build_dataloaders(\n'
        '    X_all, Y_all, batch_size=BATCH_SIZE,\n'
        ')\n'
        '\n'
        'sweep_results: dict = {}\n'
        'training_logs: dict = {}\n'
        '\n'
        'for f_val in F_VALUES:\n'
        '    print(f"\\n=== Train and eval at f = {f_val:.2f} ===")\n'
        '\n'
        '    net, log = train_model(\n'
        '        train_loader=train_loader,\n'
        '        val_loader=val_loader,\n'
        '        num_neurons=NUM_NEURONS,\n'
        '        num_classes=NUM_CLASSES,\n'
        '        train_f=f_val,\n'
        '        hidden_units=HIDDEN_UNITS,\n'
        f'{delay_arg}'
        '        epochs=EPOCHS,\n'
        '        lr=LEARNING_RATE,\n'
        '        seed=SEED,\n'
        '    )\n'
        '\n'
        f'    model_path = f"data/{model_prefix}_f{{int(round(f_val * 100)):03d}}_trained.pt"\n'
        '    torch.save(net.state_dict(), model_path)\n'
        '\n'
        '    eval_seed = SEED + int(round(f_val * 100))\n'
        '    stats = eval_repeated(\n'
        '        net, test_loader, f=f_val,\n'
        '        num_repeats=NUM_REPEATS, base_seed=eval_seed,\n'
        '    )\n'
        '    sweep_results[f_val] = stats\n'
        '    training_logs[f_val] = log\n'
        '\n'
        '    print(\n'
        '        f"  f={f_val:.2f}: accuracy = {stats[\'mean\']:.4f} +/- {stats[\'std\']:.4f}"\n'
        '        f"  (model saved to {model_path})"\n'
        '    )\n'
        '\n'
        '# Reference handle to the last-trained network (for downstream analysis cells).\n'
        'last_net = net\n'
        'last_log = log'
    )


def cell_plot_train() -> dict:
    return code(
        '# Plot the f = 0 training curves as a baseline reference.\n'
        'baseline_f = 0.0\n'
        'if baseline_f in training_logs:\n'
        '    plot_training_curves(training_logs[baseline_f], title_suffix=f"(train f={baseline_f:.2f})")\n'
        'else:\n'
        '    any_f = next(iter(training_logs))\n'
        '    plot_training_curves(training_logs[any_f], title_suffix=f"(train f={any_f:.2f})")'
    )


def cell_plot_sweep(plot_path: str) -> dict:
    return code(
        'plot_hidden_perturbation_curve(\n'
        '    sweep_results,\n'
        f'    save_path="{plot_path}",\n'
        ')'
    )


def cell_save(*, results_path: str, logs_path: str) -> dict:
    return code(
        'results_serialisable = {\n'
        '    str(f_val): {\n'
        '        "mean": float(stats["mean"]),\n'
        '        "std": float(stats["std"]),\n'
        '        "values": [float(v) for v in stats["values"]],\n'
        '    }\n'
        '    for f_val, stats in sweep_results.items()\n'
        '}\n'
        f'with open("{results_path}", "w") as fp:\n'
        '    json.dump(results_serialisable, fp, indent=2)\n'
        f'print(f"Perturbation results saved to {results_path}")\n'
        '\n'
        'training_logs_serialisable = {\n'
        '    str(f_val): {\n'
        '        k: ([float(v) for v in vals] if isinstance(vals, list) else vals)\n'
        '        for k, vals in log.items()\n'
        '    }\n'
        '    for f_val, log in training_logs.items()\n'
        '}\n'
        f'with open("{logs_path}", "w") as fp:\n'
        '    json.dump(training_logs_serialisable, fp, indent=2)\n'
        f'print(f"Training logs saved to {logs_path}")'
    )


def build_notebook(
    *,
    task: str,
    variant: str,
    title: str,
    overview: str,
    network_class: str,
    seed: int,
    max_delay_ms: int,
    plot_path: str,
    results_path: str,
    logs_path: str,
    model_prefix: str,
    task_label: str,
    variant_label: str,
) -> dict:
    with_delay = variant == "delay"

    cells = [
        md(f"# {title}\n\n## Overview\n\n{overview}"),
        md("## 1. Imports and Setup"),
        cell_imports(),
        md(
            '## 2. SLAYER and Training Parameters\n\n'
            'SLAYER neuron and simulation descriptors for the 2000-rate (Ts=0.5 ms)\n'
            'setup. Ts in `SIM_PARAMS` is in milliseconds. Per `sample_rate.md`:\n'
            '- Input spikes are scaled by `1/Ts` inside the network so SLAYER\'s\n'
            '  internal `1/Ts` spike convention is matched (Bug 1 fix).\n'
            + (
                '- `MAX_DELAY` is in **milliseconds**, not bins -- so the per-layer\n'
                f'  budget here is {max_delay_ms} ms (Bug 2 fix; matches the Ts=1 baseline).\n'
                if with_delay else ''
            )
        ),
        cell_params(with_delay=with_delay, max_delay_ms=max_delay_ms, seed=seed),
        md(
            f'## 3. Load {task.upper()} Dataset\n\n'
            f'Load the {task.upper()} spike-train dataset from the local HDF5 file.\n'
            'Each sample has shape `(num_neurons, T)` with binary 0/1 spike values;\n'
            'the network rescales them to SLAYER\'s `1/Ts` convention internally.'
        ),
        cell_load_data(task),
        md(
            '## 4. Hidden-Layer Spike Perturbation\n\n'
            'Given a hidden layer\'s spike output, randomly relocate a fraction `f`\n'
            'of each neuron\'s spikes while preserving spike count per neuron. The\n'
            'vectorised GPU implementation runs entirely on-device (no CPU round-trip),\n'
            'which matters here because perturbation runs on every training batch.'
        ),
        cell_perturb(),
        md(
            '## 5. Dataset and Data Splitting\n\n'
            'A `Dataset` wrapper and a helper that splits the data into train /\n'
            'validation / test loaders. The data is stored unperturbed; perturbation\n'
            'happens inside the network forward pass.'
        ),
        cell_dataset_split(),
        md(
            '## 6. Network Architecture\n\n'
            + (
                'A single-hidden-layer SLAYER SNN with **learnable tau** AND\n'
                '**learnable axonal delays** on both layers (the SGD-delay variant).\n'
                if with_delay else
                'A single-hidden-layer SLAYER SNN with a **learnable PSP filter** (tau),\n'
                'no delays (the SGD tau-only variant).\n'
            )
            + '\n'
            'Key implementation details:\n'
            '- `_prepare_input` scales binary inputs by `1/Ts` to match SLAYER\'s\n'
            '  internal spike convention (`sample_rate.md` Bug 1 fix).\n'
            '- `_apply_perturbation` wraps the GPU perturbation in a straight-through\n'
            '  estimator (`how_to_perturb_hidden_layer_during_training.md`) so that\n'
            '  gradients flow back through to the upstream layers even when training\n'
            '  at f > 0. Without this, fc1 / psp_filter'
            + (' / delay1' if with_delay else '')
            + ' would receive zero gradient.\n'
            '- `forward(x, f)` is the single entry point used for both training and\n'
            '  evaluation. Pass `f=train_f` during training and `f=eval_f` at test.'
        ),
        cell_network_delay(network_class) if with_delay else cell_network_tau(network_class),
        md(
            '## 7. Training Loop\n\n'
            'Train the network using SLAYER ProbSpikes loss with the Nadam optimiser.\n'
            '`train_f` is passed to every forward call so perturbation happens both\n'
            'in the training and validation passes (the model is being asked to learn\n'
            'a representation that is robust to that level of hidden-layer noise).\n'
            + (
                'Three parameter groups: regular weights at base LR, PSP filter at\n'
                'LR x 10, delays at LR x 5.\n'
                if with_delay else ''
            )
            + 'The best checkpoint (lowest validation loss) is restored at the end.'
        ),
        cell_train_delay(network_class) if with_delay else cell_train_tau(network_class),
        md(
            '## 8. Testing with Hidden-Layer Perturbation\n\n'
            'Evaluate a trained model with perturbation at level *f* (matching the\n'
            '`train_f` used to train it). `eval_repeated` runs the evaluation\n'
            '`NUM_REPEATS` times with different random seeds for error bars.'
        ),
        cell_test(network_class),
        md(
            '## 9. Visualisation Utilities\n\n'
            'Plot training curves for one f value and the accuracy-vs-f sweep across\n'
            'all trained models.'
        ),
        cell_plot_utils(with_delay=with_delay, task_label=task_label, variant_label=variant_label),
        md(
            '## 10. Run: Train AT f / Evaluate AT f for Each Perturbation Level\n\n'
            'For each `f` in `F_VALUES` train a fresh model with hidden-layer\n'
            'perturbation set to `f`, then evaluate that same model at the same `f`.\n'
            'Each trained checkpoint is written to `data/` so it can be re-loaded\n'
            'later without re-training.'
        ),
        cell_run_loop(with_delay=with_delay, model_prefix=model_prefix),
        md(
            '## 11. Plot Training Curves\n\n'
            'Visualise loss, tau'
            + (' and delay' if with_delay else '')
            + ' evolution for the f = 0 baseline run.'
        ),
        cell_plot_train(),
        md('## 12. Plot Accuracy vs Perturbation Level'),
        cell_plot_sweep(plot_path),
        md(
            '## 13. Save Results\n\n'
            'Persist the per-f accuracy statistics and the full per-f training logs\n'
            'as JSON for later cross-Ts / cross-variant comparison.'
        ),
        cell_save(results_path=results_path, logs_path=logs_path),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    specs = [
        {
            "path": ROOT / "isi" / "isi_tau.ipynb",
            "task": "isi",
            "variant": "tau",
            "title": (
                "Train-AT-f / Eval-AT-f Hidden-Layer Perturbation -- "
                "ISI (Ts=0.5, learnable tau)"
            ),
            "overview": (
                "For each perturbation level `f`, train a fresh SNN (with learnable\n"
                "tau, no delays) on the ISI synthetic task while applying hidden-layer\n"
                "spike perturbation at level `f` inside the forward pass. The same\n"
                "model is then evaluated at the same `f`.\n\n"
                "Compared to the train-once / eval-all protocol, this measures the\n"
                "*best* accuracy achievable when the network is given the chance to\n"
                "adapt to the perturbation, isolating whether the hidden representation\n"
                "*can* solve the task at level `f` from whether a clean-trained network\n"
                "*generalises* to it.\n\n"
                "Sample rate: Ts = 0.5 ms (2000 bins per 1000 ms sample). The two\n"
                "Ts-related bugs documented in `sample_rate.md` (input scaling and\n"
                "MAX_DELAY semantics) are applied throughout."
            ),
            "network_class": "ISINetwork",
            "seed": 48,
            "max_delay_ms": 15,
            "plot_path": "log/isi_tau_hidden_perturbation.png",
            "results_path": "log/isi_tau_hidden_perturbation_results.json",
            "logs_path": "log/isi_tau_training_logs.json",
            "model_prefix": "isi_tau",
            "task_label": "ISI -- SGD (tau)",
            "variant_label": "SGD (learnable tau)",
        },
        {
            "path": ROOT / "isi" / "isi_delay.ipynb",
            "task": "isi",
            "variant": "delay",
            "title": (
                "Train-AT-f / Eval-AT-f Hidden-Layer Perturbation -- "
                "ISI (Ts=0.5, learnable tau + delay)"
            ),
            "overview": (
                "For each perturbation level `f`, train a fresh SNN with learnable\n"
                "tau AND learnable axonal delays on both layers, applying hidden-layer\n"
                "spike perturbation at level `f` inside the forward pass. The same\n"
                "model is then evaluated at the same `f`.\n\n"
                "Sample rate: Ts = 0.5 ms (2000 bins per 1000 ms sample). MAX_DELAY\n"
                "is in **milliseconds**, not bins, and is set to 15 ms (matching the\n"
                "Ts=1 baseline) per `sample_rate.md` Bug 2."
            ),
            "network_class": "ISIDelayNetwork",
            "seed": 48,
            "max_delay_ms": 15,
            "plot_path": "log/isi_delay_hidden_perturbation.png",
            "results_path": "log/isi_delay_hidden_perturbation_results.json",
            "logs_path": "log/isi_delay_training_logs.json",
            "model_prefix": "isi_delay",
            "task_label": "ISI -- SGD-delay (tau + delay)",
            "variant_label": "SGD-delay (learnable tau + delay)",
        },
        {
            "path": ROOT / "ccisi" / "ccisi_tau.ipynb",
            "task": "ccisi",
            "variant": "tau",
            "title": (
                "Train-AT-f / Eval-AT-f Hidden-Layer Perturbation -- "
                "CCISI (Ts=0.5, learnable tau)"
            ),
            "overview": (
                "For each perturbation level `f`, train a fresh SNN (with learnable\n"
                "tau, no delays) on the CCISI synthetic task while applying hidden-layer\n"
                "spike perturbation at level `f` inside the forward pass. The same\n"
                "model is then evaluated at the same `f`.\n\n"
                "Sample rate: Ts = 0.5 ms. Input scaling and PSP filter length (100\n"
                "taps to cover 50 ms) are set per `sample_rate.md`."
            ),
            "network_class": "CCISINetwork",
            "seed": 42,
            "max_delay_ms": 10,
            "plot_path": "log/ccisi_tau_hidden_perturbation.png",
            "results_path": "log/ccisi_tau_hidden_perturbation_results.json",
            "logs_path": "log/ccisi_tau_training_logs.json",
            "model_prefix": "ccisi_tau",
            "task_label": "CCISI -- SGD (tau)",
            "variant_label": "SGD (learnable tau)",
        },
        {
            "path": ROOT / "ccisi" / "ccisi_delay.ipynb",
            "task": "ccisi",
            "variant": "delay",
            "title": (
                "Train-AT-f / Eval-AT-f Hidden-Layer Perturbation -- "
                "CCISI (Ts=0.5, learnable tau + delay)"
            ),
            "overview": (
                "For each perturbation level `f`, train a fresh SNN with learnable\n"
                "tau AND learnable axonal delays on both layers, applying hidden-layer\n"
                "spike perturbation at level `f` inside the forward pass. The same\n"
                "model is then evaluated at the same `f`.\n\n"
                "Sample rate: Ts = 0.5 ms. MAX_DELAY is in **milliseconds** and set\n"
                "to 10 ms (matching the Ts=1 baseline) per `sample_rate.md` Bug 2."
            ),
            "network_class": "CCISIDelayNetwork",
            "seed": 42,
            "max_delay_ms": 10,
            "plot_path": "log/ccisi_delay_hidden_perturbation.png",
            "results_path": "log/ccisi_delay_hidden_perturbation_results.json",
            "logs_path": "log/ccisi_delay_training_logs.json",
            "model_prefix": "ccisi_delay",
            "task_label": "CCISI -- SGD-delay (tau + delay)",
            "variant_label": "SGD-delay (learnable tau + delay)",
        },
    ]

    for spec in specs:
        nb = build_notebook(
            task=spec["task"],
            variant=spec["variant"],
            title=spec["title"],
            overview=spec["overview"],
            network_class=spec["network_class"],
            seed=spec["seed"],
            max_delay_ms=spec["max_delay_ms"],
            plot_path=spec["plot_path"],
            results_path=spec["results_path"],
            logs_path=spec["logs_path"],
            model_prefix=spec["model_prefix"],
            task_label=spec["task_label"],
            variant_label=spec["variant_label"],
        )
        path = spec["path"]
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(nb, fp, indent=1)
        print(f"Wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
