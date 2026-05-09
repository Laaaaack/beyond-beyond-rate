# Phase 1–4 Fixes

A running log of bugs that affect multiple notebooks/scripts and need to be
swept across the codebase.

---

## Issue 1 — Hidden-layer perturbation severs the autograd graph during training

**Status:** open. `isi_tau.ipynb` and `isi_delay.ipynb` already patched (STE).
All other notebooks listed below still have the bug.

### Symptom

When a perturbation function applied to hidden-layer spikes goes through
numpy (`.detach().cpu().numpy() → ... → torch.from_numpy(...)`), the returned
tensor is a fresh leaf with no edge in the autograd graph. If that tensor is
fed into the rest of the forward pass during **training**, `loss.backward()`
propagates gradients only through layers downstream of the perturbation site.
All upstream layers (e.g. `fc1`, `psp_filter`, `delay1`) receive **zero
gradient** for any perturbation level > 0 and stay frozen at their
initialization for the entire run.

This was previously safe because perturbation was applied only at evaluation
(under `torch.no_grad()`). The train-at-f / eval-at-f protocol moved
perturbation into the training forward pass, which is when the bug becomes
silent-but-fatal.

### How to spot it

Diagnostic in the post-training "Model Analysis" section: print mean/std of
the upstream layer's weights for every perturbation level. If the values are
**bit-identical across all f > 0 runs** (and equal to the post-init values
for an untrained model), the layer never received a gradient.

**Concrete example — `jitter_train.ipynb` (SHD-part, no delay):**

```
sigma=0:   fc1.weight_g mean=14.6847, std=14.5894   fc1.weight_v mean=-0.6226, std=6.1113
sigma=1:   fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=3:   fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=5:   fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=10:  fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=17:  fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=25:  fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
```

`fc1` is identical across every σ > 0 run — the initialization values,
frozen. Test accuracies confirm it: σ=0 reaches 38%, every σ > 0 collapses
to 14–18% (the network is solving the task using a random `fc1` plus only
`fc2`/`fc3` learning on top). Wall-clock time also balloons (16 min → 50
min) because of the per-batch CPU/numpy round-trip.

### The fix — straight-through estimator (STE)

```python
def _apply_perturbation(self, hidden_spikes: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return hidden_spikes
    perturbed = perturb_fn(hidden_spikes, p)   # the numpy round-trip
    return hidden_spikes + (perturbed - hidden_spikes).detach()
```

Forward value equals `perturbed`; backward gradient flows through
`hidden_spikes` as if no perturbation had been applied. Standard trick for
non-differentiable discrete ops. Accept that the gradient is biased — for
mild perturbations this is well-behaved.

Reference: `my_project/docs/knowledge_bank/how_to_perturb_hidden_layer_during_training.md`

### Sanity checks after fixing

1. After training at p > 0, log `fc1.weight.norm()` — must differ from the
   post-init norm.
2. Re-run the per-p model analysis; upstream weights must vary between p=0
   and p>0 runs.
3. Wall-clock per epoch should be roughly the same as p=0 (the numpy
   round-trip is unavoidable, but training shouldn't slow further once
   gradients are actually flowing).

### Affected files

Found via `grep -rln "detach().cpu().numpy()" my_project/code` plus inspection
of which calls are inside a training forward pass:

- [x] `my_project/code/synthetic/isi/isi_tau.ipynb` — fixed (STE in `_apply_perturbation`).
- [x] `my_project/code/synthetic/isi/isi_delay.ipynb` — fixed (STE in `_apply_perturbation`).
- [ ] `my_project/code/synthetic/ccisi/ccisi_tau.ipynb` — verify whether perturbation runs during training; apply STE if so.
- [ ] `my_project/code/synthetic/ccisi/ccisi_delay.ipynb` — verify; apply STE if needed.
- [ ] `my_project/code/synthetic/coincidence/coin_tau.ipynb` — verify; apply STE if needed.
- [ ] `my_project/code/synthetic/coincidence/coin_delay.ipynb` — verify; apply STE if needed.
- [ ] `my_project/code/perturbation/jitter/jitter_train.ipynb` — **confirmed buggy** (see weight-freeze evidence above). Re-run after fix; current σ > 0 results are not interpretable.
- [ ] `my_project/code/perturbation/shift/shift_train.ipynb` — same `*_train.ipynb` family; very likely buggy. Verify and patch.
- [ ] `my_project/code/perturbation/deletion/deletion_train.ipynb` — same family; verify and patch.

### Notes per family

- **ISI / CCISI / coincidence (synthetic, train-at-f / eval-at-f):** structurally
  identical to the ISI fix — STE on hidden spikes inside `forward(x, f)`.
- **Jitter / shift / deletion (perturbation/, SHD):** same pattern, but the
  perturbation is applied at the 1st of two hidden layers. Same STE fix
  inside `forward(x, sigma)` (or equivalent). Consider whether a soft /
  differentiable variant of the perturbation (e.g. Gaussian conv for jitter)
  would give a cleaner training signal — STE is the minimal fix; the soft
  variant is a more principled alternative if compute allows.

### Caveat: re-running cost

Every previously trained σ > 0 / f > 0 model needs to be retrained. Plan
sweep-by-sweep rather than all-at-once.
