"""Diagnostic for the hidden-layer perturbation hook at Ts=0.5.

Goal: identify whether ``perturb_hidden_batch`` is doing something
sample-rate-dependent that masks the timing sensitivity of the
trained CCISI delay-model at Ts=0.5 ms.

The script:

1. Confirms ``slayer.spike`` emits values of 1/Ts.
2. Reproduces ``perturb_hidden_batch`` and an alternative numpy
   reference implementation; compares per-neuron spike counts and
   spike magnitudes before/after for a synthetic random spike batch
   at Ts=0.5.
3. Loads the trained CCISI delay Ts=0.5 model and:
   - prints the learned per-neuron delays (delay1 / delay2);
   - reports the fraction of hidden neurons that ever fire on the
     test set, plus the mean firing rate;
   - measures the L2 distance between fc2 pre-activation membrane
     potential with f=0 vs f=1 perturbation;
   - measures the L2 distance between the same network's POST delay2
     PSP signal with f=0 vs f=1.

Both signals are compared *relative to* a no-perturbation baseline.
If perturbation truly destroys timing as expected, the relative
change should be substantial; if it's tiny, the hidden layer is
already rate-coded and the perturbation is correct but uninformative.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import h5py

MS = 1e-3

# Make slayerSNN importable from the venv that the notebooks use.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import slayerSNN as snn  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reference perturbation implementations
# ---------------------------------------------------------------------------

@torch.no_grad()
def perturb_hidden_batch_gpu(
    hidden_spikes: torch.Tensor,
    f: float = 0.0,
) -> torch.Tensor:
    """The current (notebook) GPU implementation, copied verbatim."""
    if f <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5
    if not is_spike.any():
        return hidden_spikes

    spike_value = x[is_spike].max().item()

    n_spikes = is_spike.sum(dim=-1, keepdim=True)
    num_to_move = (n_spikes.float() * f).floor().long()

    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move
    keep_mask = is_spike & ~remove_mask

    available = ~keep_mask
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move

    new_spikes = (keep_mask | add_mask).to(hidden_spikes.dtype) * spike_value
    return new_spikes.view(B, C, H, W, T)


def perturb_hidden_batch_numpy(
    hidden_spikes_np: np.ndarray,
    f: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Numpy reference for the same operation, with explicit retry placement.

    Shape: (B, C, T) for clarity; the caller handles squeeze/unsqueeze.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if f <= 0:
        return hidden_spikes_np

    out = hidden_spikes_np.copy()
    B, C, T = out.shape
    for b in range(B):
        for c in range(C):
            spike_idx = np.where(out[b, c] > 0.5)[0]
            if len(spike_idx) == 0:
                continue
            spike_value = float(out[b, c, spike_idx[0]])
            n_move = int(len(spike_idx) * f)
            if n_move == 0:
                continue

            chosen = rng.choice(spike_idx, size=n_move, replace=False)
            out[b, c, chosen] = 0.0

            placed = 0
            attempts = 0
            while placed < n_move and attempts < 50 * n_move:
                t = int(rng.integers(0, T))
                attempts += 1
                if out[b, c, t] < 0.5:
                    out[b, c, t] = spike_value
                    placed += 1
    return out


# ---------------------------------------------------------------------------
# (1) Confirm SLAYER spike-magnitude convention
# ---------------------------------------------------------------------------

def confirm_slayer_spike_magnitude() -> None:
    print("\n[1] slayer.spike spike-magnitude convention")
    print("-" * 60)
    for Ts in (1.0, 0.5):
        T = int(1000 / Ts)
        sim = {"Ts": Ts, "tSample": 1000}
        lif = {
            "type": "SRMALPHA", "theta": 1.0,
            "tauSr": 1.0, "tauRho": 1.0, "tauRef": 1.0,
            "scaleRef": 2, "scaleRho": 1,
        }
        layer = snn.layer(lif, sim).to(DEVICE)
        mp = torch.zeros(1, 1, 1, 1, T, device=DEVICE)
        mp[..., ::int(50 / Ts)] = 5.0   # strong impulses every 50 ms
        spk = layer.spike(mp)
        nz = spk[spk > 0.0]
        print(f"  Ts={Ts}: spike value = {nz.mean().item():.4f} (expect {1 / Ts:.2f})")


# ---------------------------------------------------------------------------
# (2) Verify perturbation correctness for a synthetic batch at Ts=0.5
# ---------------------------------------------------------------------------

def verify_perturbation_semantics() -> None:
    print("\n[2] perturb_hidden_batch semantics (synthetic, Ts=0.5)")
    print("-" * 60)
    torch.manual_seed(0)

    B, C, T = 4, 8, 2000
    spike_value = 2.0           # = 1 / Ts for Ts = 0.5
    spikes = torch.zeros(B, C, 1, 1, T, device=DEVICE)

    # Sprinkle 30 spikes per (b, c)
    for b in range(B):
        for c in range(C):
            idx = torch.randperm(T)[:30]
            spikes[b, c, 0, 0, idx] = spike_value

    base_counts = (spikes > 0.5).sum(dim=-1).squeeze(-1).squeeze(-1)  # (B, C)

    for f in (0.0, 0.3, 0.7, 1.0):
        out = perturb_hidden_batch_gpu(spikes, f=f)

        # Spike value
        nz_vals = out[out > 0.0]
        val_ok = torch.allclose(
            nz_vals, torch.full_like(nz_vals, spike_value), atol=1e-6
        )

        # Spike counts
        new_counts = (out > 0.5).sum(dim=-1).squeeze(-1).squeeze(-1)
        count_ok = torch.equal(new_counts, base_counts)

        # Fraction of spikes that *moved* (overlap with original positions)
        same_position = ((spikes > 0.5) & (out > 0.5)).sum(dim=-1)
        overlap_frac = (
            same_position.float() / base_counts.unsqueeze(-1).unsqueeze(-1).float().clamp_min(1)
        ).mean().item()

        print(
            f"  f={f:>3.1f}: spike_value_ok={val_ok}  count_ok={count_ok}  "
            f"mean spike-position overlap={overlap_frac:.3f} "
            f"(expect ~{(1 - f) + f * 30 / T:.3f} for sparse)"
        )

    # Compare GPU vs numpy implementation statistics
    spikes_np = spikes.squeeze(-2).squeeze(-2).cpu().numpy()
    out_np = perturb_hidden_batch_numpy(spikes_np, f=1.0,
                                        rng=np.random.default_rng(1))
    out_gpu = perturb_hidden_batch_gpu(spikes, f=1.0)
    out_gpu_np = out_gpu.squeeze(-2).squeeze(-2).cpu().numpy()
    print(
        f"  numpy vs gpu @ f=1.0: same total spike count = "
        f"{((out_np > 0.5).sum() == (out_gpu_np > 0.5).sum())}, "
        f"both preserve per-neuron count = "
        f"{np.array_equal((out_np > 0.5).sum(-1), (spikes_np > 0.5).sum(-1))} / "
        f"{np.array_equal((out_gpu_np > 0.5).sum(-1), (spikes_np > 0.5).sum(-1))}"
    )


# ---------------------------------------------------------------------------
# (3) Load trained CCISI delay Ts=0.5 model and probe internal signals
# ---------------------------------------------------------------------------

SIM_TS05 = {"Ts": 0.5, "tSample": 1000}
LIF_PARAMS = {
    "type": "SRMALPHA", "theta": 1.0,
    "tauSr": 1.0, "tauRho": 1.0, "tauRef": 1.0,
    "scaleRef": 2, "scaleRho": 1,
}


class CCISIDelayNetwork(nn.Module):
    """Mirror of the notebook architecture, for loading the checkpoint."""

    def __init__(
        self,
        num_neurons: int = 20,
        num_classes: int = 2,
        hidden_units: int = 100,
        max_delay: int = 10,
    ):
        super().__init__()
        self.max_delay = max_delay
        slayer = snn.layer(LIF_PARAMS, SIM_TS05)
        self.slayer = slayer

        self.fc1 = nn.utils.weight_norm(
            slayer.dense(num_neurons, hidden_units), name="weight"
        )
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(hidden_units, num_classes), name="weight"
        )
        self.psp_filter = slayer.pspFilter(
            nFilter=1, filterLength=100, filterScale=1
        )
        self.delay1 = slayer.delay(num_neurons)
        self.delay2 = slayer.delay(hidden_units)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        return x.float().to(DEVICE) / self.slayer.simulation["Ts"]

    def hidden_spikes(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.delay1(x)
        x = self.psp_filter(x)
        return self.slayer.spike(self.fc1(x))

    def membrane_after_fc2(self, hidden_spikes: torch.Tensor) -> torch.Tensor:
        """Return the pre-spike membrane signal at the output layer."""
        x = self.delay2(hidden_spikes)
        x = self.slayer.psp(x)
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden_spikes(x)
        return self.slayer.spike(self.membrane_after_fc2(h))


def load_test_batch(
    dataset_path: str, batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(dataset_path, "r") as f:
        X = f["X"][:]
        Y = f["Y"][:].ravel()
    # Use the test-range fraction the notebooks use
    n = len(X)
    test_start = int(0.75 * n)
    test_end = int(0.9 * n)
    X = X[test_start:test_end][:batch_size]
    Y = Y[test_start:test_end][:batch_size]
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.long),
    )


def probe_trained_model(model_path: str, dataset_path: str) -> None:
    print("\n[3] Trained CCISI delay Ts=0.5 model probe")
    print("-" * 60)

    net = CCISIDelayNetwork().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    net.load_state_dict(state)
    net.eval()

    # ---- Learned delays ----
    d1 = net.delay1.delay.data.flatten().cpu().numpy()
    d2 = net.delay2.delay.data.flatten().cpu().numpy()
    print(
        f"  delay1 (input->hidden, {len(d1)} neurons): "
        f"min={d1.min():.2f} ms, max={d1.max():.2f} ms, "
        f"mean={d1.mean():.2f} ms, std={d1.std():.2f} ms"
    )
    print(
        f"  delay2 (hidden->out,  {len(d2)} neurons): "
        f"min={d2.min():.2f} ms, max={d2.max():.2f} ms, "
        f"mean={d2.mean():.2f} ms, std={d2.std():.2f} ms"
    )

    X, Y = load_test_batch(dataset_path, batch_size=64)
    X = X.to(DEVICE)

    with torch.no_grad():
        h_clean = net.hidden_spikes(X)

    # Hidden-layer firing-rate statistics
    spike_count = (h_clean > 0.5).sum(dim=-1).squeeze(-1).squeeze(-1)  # (B, C)
    active_neurons = (spike_count > 0).any(dim=0).sum().item()
    mean_count = spike_count.float().mean().item()
    print(
        f"  Hidden layer: {active_neurons}/{spike_count.shape[1]} neurons ever fire "
        f"on the test batch; mean spike count per (sample, neuron) = "
        f"{mean_count:.2f} (over {h_clean.shape[-1]} bins = "
        f"{1000.0:.0f} ms)"
    )

    # ---- L2 distance: pre-spike membrane after fc2 (output) ----
    print("\n  Pre-spike membrane (fc2 output) sensitivity to hidden perturbation")
    print("  " + "-" * 56)
    with torch.no_grad():
        m_clean = net.membrane_after_fc2(h_clean)

    base_norm = m_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    for f in (0.1, 0.3, 0.5, 0.7, 1.0):
        torch.manual_seed(42)
        with torch.no_grad():
            h_pert = perturb_hidden_batch_gpu(h_clean, f=f)
            m_pert = net.membrane_after_fc2(h_pert)
        delta = (m_pert - m_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        print(
            f"    f={f:>3.1f}: rel L2 d_membrane / membrane = "
            f"{delta / max(base_norm, 1e-9):.4f}"
        )

    # ---- Same probe but BEFORE delay2 and BEFORE psp ----
    print("\n  Hidden-spike-train L2 sensitivity (without any downstream filter)")
    print("  " + "-" * 56)
    h_norm = h_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    for f in (0.1, 0.3, 0.5, 0.7, 1.0):
        torch.manual_seed(42)
        with torch.no_grad():
            h_pert = perturb_hidden_batch_gpu(h_clean, f=f)
        delta_h = (h_pert - h_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        print(
            f"    f={f:>3.1f}: rel L2 d_hidden / hidden = "
            f"{delta_h / max(h_norm, 1e-9):.4f}"
        )

    # ---- Now: post-PSP (slayer.psp of perturbed hidden, AFTER delay2) ----
    # If post-PSP signal barely moves while raw hidden signal moves a lot,
    # the psp+delay2 stack is washing out the timing information.
    print("\n  Post-delay2/PSP signal sensitivity (input to fc2)")
    print("  " + "-" * 56)
    with torch.no_grad():
        psp_clean = net.slayer.psp(net.delay2(h_clean))
    psp_norm = psp_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    for f in (0.1, 0.3, 0.5, 0.7, 1.0):
        torch.manual_seed(42)
        with torch.no_grad():
            h_pert = perturb_hidden_batch_gpu(h_clean, f=f)
            psp_pert = net.slayer.psp(net.delay2(h_pert))
        delta_psp = (psp_pert - psp_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        print(
            f"    f={f:>3.1f}: rel L2 d_(psp_o_delay2) / (psp_o_delay2) = "
            f"{delta_psp / max(psp_norm, 1e-9):.4f}"
        )


SIM_TS1 = {"Ts": 1.0, "tSample": 1000}


class CCISIDelayNetworkTs1(nn.Module):
    """Mirror of the Ts=1 notebook architecture (filterLength=50, no input scaling)."""

    def __init__(
        self,
        num_neurons: int = 20,
        num_classes: int = 2,
        hidden_units: int = 100,
        max_delay: int = 10,
    ):
        super().__init__()
        self.max_delay = max_delay
        slayer = snn.layer(LIF_PARAMS, SIM_TS1)
        self.slayer = slayer
        self.fc1 = nn.utils.weight_norm(
            slayer.dense(num_neurons, hidden_units), name="weight"
        )
        self.fc2 = nn.utils.weight_norm(
            slayer.dense(hidden_units, num_classes), name="weight"
        )
        self.psp_filter = slayer.pspFilter(
            nFilter=1, filterLength=50, filterScale=1
        )
        self.delay1 = slayer.delay(num_neurons)
        self.delay2 = slayer.delay(hidden_units)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2).unsqueeze(3)
        # Ts=1 notebook does NOT divide by Ts (since 1/1 = 1 anyway).
        return x.float().to(DEVICE)

    def hidden_spikes(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.delay1(x)
        x = self.psp_filter(x)
        return self.slayer.spike(self.fc1(x))

    def membrane_after_fc2(self, hidden_spikes: torch.Tensor) -> torch.Tensor:
        x = self.delay2(hidden_spikes)
        x = self.slayer.psp(x)
        return self.fc2(x)


def probe_trained_model_ts1(model_path: str, dataset_path: str) -> None:
    print("\n[4] Trained CCISI delay Ts=1 model probe (apples-to-apples)")
    print("-" * 60)
    net = CCISIDelayNetworkTs1().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    net.load_state_dict(state)
    net.eval()

    d1 = net.delay1.delay.data.flatten().cpu().numpy()
    d2 = net.delay2.delay.data.flatten().cpu().numpy()
    print(
        f"  delay1: mean={d1.mean():.2f} ms, std={d1.std():.2f} ms"
    )
    print(
        f"  delay2: mean={d2.mean():.2f} ms, std={d2.std():.2f} ms"
    )

    X, Y = load_test_batch(dataset_path, batch_size=64)
    X = X.to(DEVICE)
    with torch.no_grad():
        h_clean = net.hidden_spikes(X)
        m_clean = net.membrane_after_fc2(h_clean)
        psp_clean = net.slayer.psp(net.delay2(h_clean))

    spike_count = (h_clean > 0.5).sum(dim=-1).squeeze(-1).squeeze(-1)
    active = (spike_count > 0).any(dim=0).sum().item()
    print(
        f"  Hidden: {active}/100 active; mean spikes/(sample,neuron) = "
        f"{spike_count.float().mean().item():.2f} over {h_clean.shape[-1]} bins"
    )

    h_norm = h_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    psp_norm = psp_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    m_norm = m_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    print(f"\n  f      d_h/h    d_psp/psp   d_mem/mem")
    for f in (0.1, 0.3, 0.5, 0.7, 1.0):
        torch.manual_seed(42)
        with torch.no_grad():
            h_pert = perturb_hidden_batch_gpu(h_clean, f=f)
            psp_pert = net.slayer.psp(net.delay2(h_pert))
            m_pert = net.membrane_after_fc2(h_pert)
        d_h = (h_pert - h_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        d_psp = (psp_pert - psp_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        d_m = (m_pert - m_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        print(
            f"  {f:>3.1f}   {d_h / h_norm:>6.3f}    "
            f"{d_psp / psp_norm:>6.3f}      {d_m / m_norm:>6.3f}"
        )


def main() -> None:
    confirm_slayer_spike_magnitude()
    verify_perturbation_semantics()

    project = Path(
        "d:/IC_2025/IRP/workspace/my_project/code/synthetic/diff_rate_test/2000_rate"
    )
    probe_trained_model(
        model_path=str(project / "ccisi/data/ccisi_delay_trained.pt"),
        dataset_path=str(project / "ccisi/ccisi_dataset.h5"),
    )

    project_ts1 = Path("d:/IC_2025/IRP/workspace/my_project/code/synthetic/ccisi")
    probe_trained_model_ts1(
        model_path=str(project_ts1 / "data/ccisi_delay_trained.pt"),
        dataset_path=str(project_ts1 / "ccisi_dataset.h5"),
    )

    print("\n[5] For comparison, the same probe on CCISI tau Ts=0.5")
    print("-" * 60)
    # Reload as a tau-only network (no delays) for fair comparison.
    class CCISITauNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            slayer = snn.layer(LIF_PARAMS, SIM_TS05)
            self.slayer = slayer
            self.fc1 = nn.utils.weight_norm(slayer.dense(20, 100), name="weight")
            self.fc2 = nn.utils.weight_norm(slayer.dense(100, 2), name="weight")
            self.psp_filter = slayer.pspFilter(
                nFilter=1, filterLength=100, filterScale=1
            )

        def _prepare_input(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(2).unsqueeze(3)
            return x.float().to(DEVICE) / self.slayer.simulation["Ts"]

        def hidden_spikes(self, x):
            x = self._prepare_input(x)
            return self.slayer.spike(self.fc1(self.psp_filter(x)))

        def membrane_after_fc2(self, h):
            return self.fc2(self.slayer.psp(h))

    tau_net = CCISITauNetwork().to(DEVICE)
    tau_net.load_state_dict(
        torch.load(str(project / "ccisi/data/ccisi_tau_trained.pt"),
                   map_location=DEVICE)
    )
    tau_net.eval()

    X, Y = load_test_batch(str(project / "ccisi/ccisi_dataset.h5"), batch_size=64)
    X = X.to(DEVICE)
    with torch.no_grad():
        h_clean = tau_net.hidden_spikes(X)
        m_clean = tau_net.membrane_after_fc2(h_clean)
        psp_clean = tau_net.slayer.psp(h_clean)

    base_norm = m_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    h_norm = h_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    psp_norm = psp_clean.flatten(start_dim=1).norm(dim=1).mean().item()
    spike_count = (h_clean > 0.5).sum(dim=-1).squeeze(-1).squeeze(-1)
    print(f"  Tau hidden: {(spike_count > 0).any(dim=0).sum().item()}/100 active; "
          f"mean spikes per (sample, neuron) = {spike_count.float().mean().item():.2f}")

    for f in (0.3, 0.7, 1.0):
        torch.manual_seed(42)
        with torch.no_grad():
            h_pert = perturb_hidden_batch_gpu(h_clean, f=f)
            m_pert = tau_net.membrane_after_fc2(h_pert)
            psp_pert = tau_net.slayer.psp(h_pert)
        d_h = (h_pert - h_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        d_psp = (psp_pert - psp_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        d_m = (m_pert - m_clean).flatten(start_dim=1).norm(dim=1).mean().item()
        print(
            f"  tau f={f:>3.1f}: Δh/h={d_h / h_norm:.3f}  "
            f"Δpsp/psp={d_psp / psp_norm:.3f}  "
            f"Δmem/mem={d_m / base_norm:.3f}"
        )


if __name__ == "__main__":
    main()
