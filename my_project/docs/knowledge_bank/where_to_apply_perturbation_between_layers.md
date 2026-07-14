# How to Construct SLAYER SNN Layers for Hidden-Layer Perturbation

## The Core Invariant

**The tensor injected into the perturbation function must be strictly binary (0/1).**

Hidden-layer perturbation functions (`partial_randomize_spike_train`, `jitter_spike_train`, etc.)
all rely on `np.where(spike_train == 1)` to locate spikes. If the tensor contains fractional
values, the comparison finds nothing and the perturbation silently becomes a no-op.

The only SLAYER operation that produces fractional spike-train tensors is `slayer.delay(...)`,
which applies linear interpolation between adjacent time bins. Therefore:

> **A hidden-layer's output — the tensor passed to the perturbation hook — must be the result
> of `slayer.spike(...)`. `slayer.delay` must never be the last operation before the hook.**

---

## Layer Construction Pattern

For a network with N hidden layers and delays, organise each layer as a **compute block**
followed by a **routing block**:

```
Layer k compute block:  psp(input) → fc_k → spike   ← OUTPUT IS BINARY HERE
Layer k routing block:  delay_k(binary_spikes)        ← feeds into layer k+1 compute
```

The perturbation hook sits **between** the compute and routing blocks of the target layer.

---

## Two-Hidden-Layer Example

### Network topology

```
Input spikes
    │
    ▼
┌─────────────────────────────┐
│ Hidden layer 1 (compute)    │  psp → fc1 → spike()
│  → binary hidden1 spikes    │  ← PERTURBATION HOOK HERE
└─────────────────────────────┘
    │
    ▼ (apply delay1 as routing)
┌─────────────────────────────┐
│ Hidden layer 2 (compute)    │  psp → fc2 → spike()
│  → binary hidden2 spikes    │  ← optional 2nd hook
└─────────────────────────────┘
    │
    ▼ (apply delay2 as routing)
┌─────────────────────────────┐
│ Output layer                │  psp → fc3 → spike()
└─────────────────────────────┘
```

### Correct PyTorch implementation

```python
class TwoHiddenSNN(nn.Module):
    def __init__(self, input_dim, hidden_units, num_classes, use_delay=True, max_delay=64):
        super().__init__()
        slayer = snn.layer(LIF_PARAMS, SIM_PARAMS)
        self.slayer = slayer
        self.use_delay = use_delay

        self.fc1 = slayer.dense(input_dim, hidden_units)
        self.fc2 = slayer.dense(hidden_units, hidden_units)
        self.fc3 = slayer.dense(hidden_units, num_classes)

        if use_delay:
            self.delay1 = slayer.delay(hidden_units)  # routes hidden1 → hidden2
            self.delay2 = slayer.delay(hidden_units)  # routes hidden2 → output

    def _hidden1(self, x: torch.Tensor) -> torch.Tensor:
        """Compute block for hidden layer 1. Returns BINARY spikes."""
        return self.slayer.spike(self.fc1(self.slayer.psp(x)))
        # delay1 is NOT applied here — it belongs in _hidden2's routing step

    def _hidden2(self, hidden1_spikes: torch.Tensor) -> torch.Tensor:
        """Routing (delay1) + compute block for hidden layer 2. Returns BINARY spikes."""
        x = self.delay1(hidden1_spikes) if self.use_delay else hidden1_spikes
        return self.slayer.spike(self.fc2(self.slayer.psp(x)))

    def _output(self, hidden2_spikes: torch.Tensor) -> torch.Tensor:
        """Routing (delay2) + output layer."""
        x = self.delay2(hidden2_spikes) if self.use_delay else hidden2_spikes
        return self.slayer.spike(self.fc3(self.slayer.psp(x)))

    def forward(self, x: torch.Tensor, perturb_fn=None) -> torch.Tensor:
        hidden1 = self._hidden1(x)

        # Perturbation hook: hidden1 is guaranteed binary here
        if perturb_fn is not None:
            hidden1 = perturb_fn(hidden1)

        hidden2 = self._hidden2(hidden1)
        return self._output(hidden2)
```

### Why this is correct

| Step | Operation | Output type |
|------|-----------|-------------|
| `psp(x)` | PSP filter (convolution) | continuous |
| `fc1(...)` | Linear projection | continuous |
| `slayer.spike(...)` | Hard threshold + surrogate grad | **binary 0/1** |
| **HOOK HERE** | `perturb_fn(hidden1)` | binary → binary |
| `delay1(hidden1)` | Fractional axonal delay | **fractional** |
| `psp(...)` | PSP filter on delayed signal | continuous |
| ... | continues to next layer | ... |

---

## Anti-Pattern to Avoid

```python
# WRONG — delay is inside the compute block, output is fractional
def _hidden1_broken(self, x):
    spikes = self.slayer.spike(self.fc1(self.slayer.psp(x)))
    return self.delay1(spikes)   # <-- fractional output

# Hook here receives fractional tensor; np.where(== 1) finds nothing
hidden1 = self._hidden1_broken(x)
hidden1 = perturb_fn(hidden1)   # silent no-op
```

This pattern was the root cause of the bug in `coin_delay.ipynb` and `jitter_train.ipynb`
(when `USE_DELAY=True`): the delay sat at the end of the hidden-layer method, so the
perturbation function received fractional values and moved zero spikes at every f > 0.

---

## Checklist Before Running a Perturbation Sweep

1. **Trace `_hidden1` (or equivalent).** Confirm the last operation is `slayer.spike(...)`.
2. **Print spike tensor stats.** After `_hidden1` and before `perturb_fn`, assert
   `torch.unique(hidden1).tolist() == [0.0, 1.0]` (or check `hidden1.max() == 1` and
   `hidden1.min() == 0`). If you see values in (0, 1), delay is leaking past spike.
3. **Check `num_to_move`.** With few hidden neurons or sparse spike trains,
   `int(spike_count * f)` may truncate to 0. Consider `max(1, int(...))` or `ceil` if
   f > 0 must guarantee at least one spike moves.
4. **Verify checkpoint matches architecture.** If the layer structure changed (e.g. delay
   moved), old `.pt` checkpoints encode the old computation graph — retrain from scratch.
