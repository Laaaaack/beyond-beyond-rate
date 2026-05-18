# Phase 1–4 Fixes

A running log of cross-cutting issues that affect multiple notebooks and need
to be swept across the codebase.

Branch-wide decision (2026-05-18): every experiment uses **train-clean /
eval-perturbed** at the 1st hidden layer. Train a single model on unperturbed
inputs; sweep the perturbation level only at evaluation time, inside
`torch.no_grad()`. The "train-at-f / eval-at-f" protocol that some notebooks
adopted is being retired here — it answered the wrong question for this
project (see the rationale below) and, as a side effect, retiring it also
makes the autograd-severance problem disappear.

---

## Issue 1 — Convert `jitter` / `shift` / `deletion` to train-clean / eval-perturbed

**Status:** open. Three Phase 3 notebooks still train a separate model per
perturbation level and apply the perturbation inside the *training* forward
pass. The rest of the codebase is already in the train-clean / eval-perturbed
shape (see audit at the bottom).

### Why this change

The `project_overview.md` table explicitly specifies:

| | Beyond Beyond Rate (ours) |
|---|---|
| Training | Standard (unperturbed inputs) — same trained models |
| Inference | Unperturbed inputs; hidden layer output intercepted and perturbed |

The core question is whether a network **trained naturally** uses spike
timing in its hidden layer. Train-at-f / eval-at-f at the hidden layer
measures something else — *can the network be trained to be robust to a
corrupted hidden representation* — and the answer to that question, if the
task is solvable at all under hidden perturbation, is "yes, by learning a
rate-coded hidden layer." A flat curve under train-at-f / eval-at-f is
therefore evidence *for* rate sufficiency, not against it. The
train-clean / eval-perturbed protocol keeps the trained network fixed and
asks how its natural representation reacts to losing timing — that is the
test the project actually wants.

A useful side effect: under eval-only perturbation the numpy round-trip
happens inside `torch.no_grad()`, so there is no autograd graph to sever.
The straight-through estimator (STE) workaround that the previous version of
this doc was building toward is no longer needed.

### Target shape (mirrors the already-clean notebooks)

The cleared notebooks (ccisi, coin, shd, ssc, inverse, isi_tau, isi_delay)
all share this layout. The Phase 3 notebooks should be refactored to match.

```python
class Net(nn.Module):
    def forward(self, x):                          # clean — used in training
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        return self._second_hidden_and_output(hidden1)

    def forward_with_hidden_perturbation(self, x, sigma=0.0):
        # called only from eval loops, inside `with torch.no_grad():`
        x = self._prepare_input(x)
        hidden1 = self._first_hidden(x)
        if sigma > 0:
            hidden1 = jitter_hidden_batch(hidden1, sigma)
        return self._second_hidden_and_output(hidden1)
```

Training loop calls `net(x_batch)`; sweep loop calls
`net.forward_with_hidden_perturbation(x_batch, sigma=σ)` inside `no_grad`.
One trained model per `(dataset_key, USE_DELAY)` cell of the grid, not one
per σ.

### Sanity checks after conversion

1. The σ=0 test-set accuracy of the single trained model should match (or
   beat) the σ=0 model's accuracy under the old protocol — the model is
   trained on the *easier* clean task.
2. The σ-sweep curve should look like the SHD/SSC/inverse curves: smooth
   monotone(-ish) degradation, no σ=1 cliff.
3. Wall-clock per run drops from `len(SIGMA_VALUES) × per-σ training cost`
   to a single training run plus a cheap sweep. Plan on roughly one order
   of magnitude less GPU time per notebook.

### Per-notebook conversion checklist

The three notebooks share the same code shape. Each conversion touches the
same five things:

1. **`Net.forward`** — drop the `sigma` / `p_d` parameter. Make it the clean
   pass that ends with `self._second_hidden_and_output(hidden1)`.
2. **Add `Net.forward_with_hidden_perturbation`** — takes `(x, sigma)` /
   `(x, p_d)`, applies `jitter_hidden_batch` / `shift_hidden_batch` /
   `delete_hidden_batch` only when the level is > 0, otherwise identical
   to `forward`. (Match the docstring / API style of `isi_tau.ipynb`'s
   `forward_with_hidden_perturbation`.)
3. **`train_model`** — remove the `sigma` / `p_d` argument. Call
   `net(x_batch)` in both the training pass and the in-training validation
   pass. Train *once* per `(dataset_key, USE_DELAY)` configuration.
4. **Sweep loop** — replace the `for sigma in SIGMA_VALUES: train; test`
   block with `train_once; for sigma in SIGMA_VALUES: test_with_*`. Each
   evaluation goes through `forward_with_hidden_perturbation` inside
   `torch.no_grad()`.
5. **Filenames / logs** — the trained model is now a single file
   (`data/{MODEL_PREFIX}_trained.pt`, no σ suffix). The sweep results JSON
   stays per-σ; the per-σ training-log JSONs go away (there is only one
   training log).

#### Affected files

- [ ] `my_project/code/perturbation/jitter/jitter_train.ipynb`
  - `JitterSHDNetwork.forward(x, sigma)` → split into clean `forward(x)` and
    `forward_with_hidden_perturbation(x, sigma)`.
  - `train_model(..., sigma)` → `train_model(...)`. Drop the outer
    `for sigma in SIGMA_VALUES` around training; train once.
  - `test_with_jitter` and `test_with_repeats` call
    `forward_with_hidden_perturbation`, not `net(x_batch, sigma=sigma)`.
  - Validation-during-training already runs under `no_grad` but currently
    calls `net(x_batch, sigma=sigma)` — change to `net(x_batch)`.
- [ ] `my_project/code/perturbation/shift/shift_train.ipynb` — same five
      edits, with `ShiftSHDNetwork` and `shift_hidden_batch`. Per-neuron
      shift is the only perturbation; structure is otherwise identical to
      jitter.
- [ ] `my_project/code/perturbation/deletion/deletion_train.ipynb` — same
      five edits, with `DeletionSHDNetwork`, `delete_hidden_batch`, and
      `p_d` instead of `sigma`. Note: the current `EPOCHS=20` in this
      notebook is a debug value carried over from earlier work and should
      be raised to match the jitter/shift schedules before the new
      converged sweep is taken at face value.

#### Re-training cost

Each notebook now trains exactly one model per `(DATASET_KEY, USE_DELAY)`
cell. Old `data/{MODEL_PREFIX}_sigma{σ}.pt` / `_pd{pd}.pt` files become
obsolete and should be deleted once the new single-model checkpoint is in
place. The old per-σ `_training_log.json` files likewise.

### Audit: where the codebase already sits

Verified 2026-05-18 by reading `forward` signatures and training-loop call
sites in every `*_train.ipynb` / `*_tau.ipynb` / `*_delay.ipynb` under
`my_project/code/`.

| Notebook | Forward shape | Training call site | Status |
|---|---|---|---|
| `synthetic/isi/isi_tau.ipynb`         | `forward(x)` + `forward_with_hidden_perturbation(x, f)` | `net(x_batch)`         | clean ✓ |
| `synthetic/isi/isi_delay.ipynb`       | same                                                    | `net(x_batch)`         | clean ✓ |
| `synthetic/ccisi/ccisi_tau.ipynb`     | same                                                    | `net(x_batch)`         | clean ✓ |
| `synthetic/ccisi/ccisi_delay.ipynb`   | same                                                    | `net(x_batch)`         | clean ✓ |
| `synthetic/coincidence/coin_tau.ipynb`| same                                                    | `net(x_batch)`         | clean ✓ |
| `synthetic/coincidence/coin_delay.ipynb`| same                                                  | `net(x_batch)`         | clean ✓ |
| `realistic/shd/shd_train.ipynb`       | same                                                    | `net(x_batch)`         | clean ✓ |
| `realistic/ssc/ssc_train.ipynb`       | same                                                    | `net(x_batch)`         | clean ✓ |
| `perturbation/inverse/inverse_train.ipynb` | `forward(x)` + `forward_with_hidden_perturbation(x, f)` + `forward_with_hidden_reversal(x, ...)` | `net(x_batch)` | clean ✓ |
| `perturbation/jitter/jitter_train.ipynb`   | `forward(x, sigma)` only                            | `net(x_batch, sigma=sigma)` | **convert** |
| `perturbation/shift/shift_train.ipynb`     | `forward(x, sigma)` only                            | `net(x_batch, sigma=sigma)` | **convert** |
| `perturbation/deletion/deletion_train.ipynb` | `forward(x, p_d)` only                            | `net(x_batch, p_d=p_d)`     | **convert** |

The current ISI sweep outputs in the repo confirm the train-clean /
eval-perturbed protocol works end-to-end on a hidden-perturbation task:

- `isi_tau` (clean acc 0.946):   f=0 0.946 → 0.924 → 0.922 → 0.916 → 0.900 → 0.893 → 0.887 → 0.877 → 0.874 → 0.880 → 0.868
- `isi_delay` (clean acc 0.937): f=0 0.937 → 0.920 → 0.864 → 0.814 → 0.740 → 0.688 → 0.648 → 0.609 → 0.581 → 0.582 → 0.578

The delay variant degrades much more steeply than tau-only — the kind of
signal we want to see across the rest of the experiments.

---

## Issue 2 — `delay-after-spike` anti-pattern (lower-priority hygiene)

**Status:** open, deferred. The structural anti-pattern is harmless in this
slayerSNN version but worth cleaning up.

`_first_hidden(x) = slayer.spike(...) ; if use_delay: x = delay1(x)` — the
delay sits *after* the spike inside the same method, so the method returns
"delayed spikes" rather than "raw spikes". An old version of the knowledge
bank predicted this would silently no-op the perturbation (because
`slayer.delay` was thought to do linear interpolation between bins, leaving
a fractional tensor that `np.where(... == 1)` cannot match). That prediction
does not hold for this slayer version: `slayer.delay` floors the delay
internally (`slayer.py:351-373` and `_delayFunction.apply →
slayerCuda.shift`), so the output is still strictly binary. Empirical
confirmation: SHD/SSC `_delay` sweeps degrade smoothly, identical in shape
to `_nodelay` sweeps just at higher baselines.

So there is no functional bug to fix. The anti-pattern is still worth
refactoring because:

1. It mixes routing (delay) with compute (psp+fc+spike) in the same method.
2. It is fragile across slayer versions — if a future release switches to
   fractional delays, the perturbation hook would silently break.
3. The synthetic delay notebooks (`isi_delay`, `coin_delay`, `ccisi_delay`)
   already use the cleaner pattern (delay at the start of `_second_layer`,
   after the hook), and they read more clearly.

### Affected files

`_first_hidden(x)` ends with delay in:

- `code/realistic/shd/shd_train.ipynb` (line ~453)
- `code/realistic/ssc/ssc_train.ipynb` (line ~484)
- `code/perturbation/inverse/inverse_train.ipynb` (line ~392)
- `code/perturbation/jitter/jitter_train.ipynb` (line ~481)
- `code/perturbation/shift/shift_train.ipynb` (line ~450)
- `code/perturbation/deletion/deletion_train.ipynb` (line ~601)

### Plan

- Fold the refactor into the Issue 1 conversions for jitter / shift /
  deletion: while editing `forward` anyway, move the delay to the start of
  `_second_hidden_and_output` (or before the spike inside `_first_hidden`).
- For shd / ssc / inverse, apply the refactor opportunistically — they do
  not need retraining for it because they are already train-clean and the
  current binary-in / binary-out behaviour is correct.
- Optional sanity assertion after `_first_hidden` in any eval loop:
  `assert torch.unique(hidden1).tolist() == [0.0, 1.0]`. If a future slayer
  release stops flooring, this catches the regression at runtime.
