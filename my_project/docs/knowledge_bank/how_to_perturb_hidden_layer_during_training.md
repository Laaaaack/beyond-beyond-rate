# How to Perturb Hidden-Layer Spikes During Training

## Context

In the "Beyond Beyond Rate" experiments (e.g. `isi_tau.ipynb`, `isi_delay.ipynb`)
we apply a non-differentiable perturbation to hidden-layer spike trains: a
fraction `f` of each neuron's spikes are removed and randomly relocated, while
the spike count per neuron is preserved. This destroys temporal structure but
keeps the rate code intact.

When perturbation is applied **only at evaluation** (inside `torch.no_grad()`),
the implementation is trivial — just replace the hidden tensor with the
perturbed one:

```python
hidden_spikes = perturb_hidden_batch(hidden_spikes, f)
```

When the protocol changes to **train-at-f / evaluate-at-f**, the same
perturbation must run during the training forward pass too. At that point the
naive replacement breaks gradient flow.

## Why the naive version breaks training

`perturb_hidden_batch` performs:

```python
spikes_np = hidden_spikes.detach().cpu().numpy()
# ... numpy ops ...
return torch.tensor(perturbed_np, ..., device=hidden_spikes.device)
```

The returned tensor is a **fresh leaf** — it has no edge in the autograd graph
back to the upstream computation that produced `hidden_spikes`.

Consequence: during `loss.backward()`, gradients propagate only through layers
**downstream** of the perturbation (e.g. `fc2`, and `delay2` in the delay
variant). Layers **upstream** (`fc1`, `psp_filter`, `delay1`) receive **zero
gradient** for any `f > 0`. They remain frozen at their initialization for the
entire run.

This silently produces a degenerate model: the upstream half never learns, and
the network is forced to solve the task using only the unperturbed second half
on top of an essentially random first half.

## The fix: straight-through estimator (STE)

```python
def _apply_perturbation(self, hidden_spikes: torch.Tensor, f: float) -> torch.Tensor:
    if f <= 0:
        return hidden_spikes
    perturbed = perturb_hidden_batch(hidden_spikes, f)
    return hidden_spikes + (perturbed - hidden_spikes).detach()
```

How it works:

- **Forward value**: `hidden + (perturbed - hidden).detach() == perturbed`.
  The network sees the perturbed spikes, exactly as intended.
- **Backward gradient**: the `.detach()` block contributes zero gradient, so
  `d(output)/d(hidden_spikes) = 1`. Gradients flow back through
  `hidden_spikes` into the upstream layers as if no perturbation had occurred.

This is the same trick used for any non-differentiable discrete operation
(quantization, hard sampling, the spike function itself in SNNs — SLAYER's
surrogate gradient is conceptually a sibling of this approach).

## When STE is appropriate

- The non-differentiable op is a **stochastic / discrete transformation** of a
  signal whose underlying continuous structure you still want to learn.
- You accept that the gradient is **biased** (it pretends the op is identity).
  For mild perturbations this is a standard, well-behaved approximation; for
  `f → 1` the bias grows but the gradient direction is still usable.

## When NOT to use STE

- If the perturbation is the *thing being optimized* (e.g. you want to learn
  the perturbation parameters), STE is wrong — use a differentiable
  reparameterization (Gumbel-softmax, REINFORCE, etc.) instead.
- If you only ever apply the perturbation at evaluation, don't bother — the
  direct assignment under `torch.no_grad()` is simpler and equivalent.

## Sanity checks

After switching to STE during training, verify:

1. **Gradient norms**: log `param.grad.norm()` for `fc1`, `psp_filter`, and
   (where applicable) `delay1`. They must be non-zero for `f > 0`.
2. **Parameter drift**: at the end of training, parameter values should differ
   from initialization by a margin comparable to the `f = 0` run.
3. **f = 0 vs f > 0 baseline**: a model trained at `f = 0` and one trained at
   small `f` should reach broadly similar clean-test accuracies; if the
   `f > 0` model is dramatically worse, the upstream layers are probably not
   training (suggests the STE wiring is broken).

## Performance: vectorise the perturbation, don't go through numpy

STE fixes correctness, but it does **not** fix throughput. The naive numpy
implementation of `perturb_hidden_batch` does, per training step:

1. `hidden.detach().cpu().numpy()` — a full GPU→CPU copy.
2. A Python-level loop over `batch × neurons` (e.g. 32 × 100 = 3200 inner
   iterations per batch for the ISI hidden layer), each running `np.where`,
   `np.random.choice`, and a retry loop.
3. `torch.from_numpy(...).to(device)` — CPU→GPU copy back.

Before the STE fix this only ran at evaluation, so the cost was invisible.
Once perturbation runs on every training batch, this is the dominant cost.
Empirically: ISI training at `f = 0` runs ~60 it/s; at `f = 0.2` with the
numpy path it drops to ~1.35 it/s — a ~45× slowdown that has nothing to do
with the model.

### Vectorised GPU implementation (preserves spike count exactly)

The relocation operation can be done with two `argsort`s on the GPU and no
host transfer:

```python
@torch.no_grad()
def perturb_hidden_batch(hidden_spikes: torch.Tensor, f: float = 0.0) -> torch.Tensor:
    if f <= 0:
        return hidden_spikes

    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5
    n_spikes = is_spike.sum(dim=-1, keepdim=True)            # (B, C, 1)
    num_to_move = (n_spikes.float() * f).floor().long()      # (B, C, 1)

    # 1. Pick which existing spikes to remove.
    #    Random key for each spike position; non-spikes get +inf so they sort last.
    key = torch.rand_like(x)
    key = torch.where(is_spike, key, torch.full_like(key, 2.0))
    rank = key.argsort(dim=-1).argsort(dim=-1)               # rank along T per (b,c)
    remove_mask = rank < num_to_move
    keep_mask = is_spike & ~remove_mask

    # 2. Place the same number of new spikes in currently-unoccupied bins.
    available = ~keep_mask
    key2 = torch.rand_like(x)
    key2 = torch.where(available, key2, torch.full_like(key2, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move                           # disjoint from keep_mask

    return (keep_mask | add_mask).to(hidden_spikes.dtype).view(B, C, H, W, T)
```

Why this is equivalent to the numpy version:

- `key` is uniform on `[0, 1)` at spike positions and `+∞` elsewhere. Its rank
  along `T` per `(b, c)` puts spike positions first in random order, so
  `rank < num_to_move` selects exactly `num_to_move` random spikes per
  neuron — the same distribution as `np.random.choice(spike_times,
  num_to_move, replace=False)`.
- `add_mask` samples `num_to_move` positions uniformly from `~keep_mask`,
  matching the placement step. Because the sampling pool excludes
  `keep_mask`, `add_mask` and `keep_mask` are disjoint by construction —
  spike count per neuron is preserved exactly.
- The retry loop in the numpy version handled collisions; here the
  exclusion is structural, so retries are unnecessary. Capacity is always
  sufficient as long as `n_spikes ≤ T` per neuron, which is always true
  for binary spike trains.

### Why this is fast

- Two GPU `argsort`s on a `(B, C, T)` tensor; for ISI sizes (`T = 1000`,
  `C = 100`, `B = 32`) this is microseconds per batch.
- No host transfer, no Python-level loops.
- `@torch.no_grad()` keeps the perturbation out of the autograd graph; the
  STE in `_apply_perturbation` still wires gradients through correctly via
  `hidden + (perturbed - hidden).detach()`.

### When you don't need the vectorised version

- Eval-only perturbation called rarely (a few hundred batches per run): the
  numpy version is fine and easier to read.
- Tiny networks where the CPU round-trip is negligible.

For training-time perturbation on any non-trivial model, vectorise.

## Reference implementations

- `my_project/code/synthetic/isi/isi_tau.ipynb` — `ISINetwork._apply_perturbation`
  (STE) and `perturb_hidden_batch` (vectorised GPU).
- `my_project/code/synthetic/isi/isi_delay.ipynb` — `ISIDelayNetwork._apply_perturbation`
  (STE) and `perturb_hidden_batch` (vectorised GPU).
