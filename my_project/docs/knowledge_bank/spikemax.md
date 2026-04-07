# SpikeMax Loss Function

## Reference

Shrestha, S. B., Zhu, L., & Sun, P. (2022). *Spikemax: Spike-based Loss Methods for Classification.* arXiv:2205.09845. IEEE IJCNN 2022.

---

## What Problem Does It Solve?

Standard loss functions (MSE, cross-entropy) are not directly compatible with SNN outputs because spikes are discrete binary events in time. Prior work was limited to **mean-squared error on spike counts**, which is arbitrary (requires manual target spike counts) and not grounded in classification theory.

SpikeMax introduces a principled, cross-entropy-equivalent loss for SNNs.

---

## How It Works

### Step 1 — Spike counts to probabilities

For each output neuron `i`, count spikes over the full time window:

```
count_i = Σ_t spike_i(t)
```

Normalise across all output neurons to get class probabilities:

```
p_i = count_i / (Σ_j count_j + ε)        # probability mode
```

or alternatively via softmax:

```
p_i = exp(count_i) / Σ_j exp(count_j)    # softmax mode
```

### Step 2 — Negative log-likelihood loss

```
L = -log(p_label)
```

This is identical to standard cross-entropy used in ANNs, but operating on spike-count-derived probabilities rather than logits.

---

## Comparison with `numSpikes`

| Aspect | `numSpikes` | `probSpikes` (SpikeMax) |
|---|---|---|
| **Target** | Manual spike counts (e.g. 60 correct, 10 wrong) | Class label index only |
| **Loss type** | MSE on spike count deviation | NLL on spike probabilities |
| **Extra config** | `tgtSpikeCount`, `tgtSpikeRegion` | None |
| **Gradient signal** | Penalises deviation from fixed target | Penalises low confidence in correct class |
| **Theoretical basis** | Heuristic | Maximum likelihood / cross-entropy |

---

## Implementation in This Project

### lava-dl reference (`lava.lib.dl.slayer.loss.SpikeMax`)

The newer lava-dl repo implements this in `loss.py`, using a `Rate.confidence()` helper from `classifier.py`:

```python
rate = spike.mean(dim=-1)                          # mean over time → rate per neuron
p = rate / (rate.sum(dim=1, keepdim=True) + eps)   # normalise → probabilities
loss = F.nll_loss(torch.log(p), label)             # NLL
```

### Patched into slayerSNN (`spikeLoss.probSpikes`)

The installed `slayerSNN` package (`slayerSNN-0.0.0`) had `probSpikes` defined as a broken stub (missing `self` parameter, body was `pass`). It was replaced with a working SpikeMax implementation:

**File:** `venv/Lib/site-packages/slayerSNN-0.0.0-py3.11.egg/slayerSNN/spikeLoss.py`

Key changes:
- Added `import torch.nn.functional as F` at the top
- Replaced the stub `probSpikes` with a proper method that:
  - Accepts `(self, spikeOut, desiredClass, mode='probability', eps=1e-6)`
  - Collapses spatial dims and sums over time to get spike counts `(N, C)`
  - Normalises to probabilities (or uses `log_softmax` in softmax mode)
  - Returns `F.nll_loss(log_p, desiredClass)`

### Usage in `isi_tau.ipynb`

```python
loss_fn = snn.spikeLoss.spikeLoss({
    "neuron": LIF_PARAMS,
    "simulation": SIM_PARAMS,
    "training": {"error": {"type": "ProbSpikes"}},
}).to(device)

# In training/validation loop — y_batch must be torch.long class indices
loss = loss_fn.probSpikes(outputs, y_batch)
```

No one-hot encoding or `tgtSpikeCount` configuration needed. The target is simply a 1-D `torch.long` tensor of class indices.

---

## Key Advantages for This Project

1. **No manual tuning** of target spike counts — removes an arbitrary hyperparameter
2. **Standard gradient behaviour** — same as cross-entropy in ANNs, well understood
3. **Energy efficient** — network learns to use the minimum spikes needed to separate classes, rather than being forced toward a fixed count target
