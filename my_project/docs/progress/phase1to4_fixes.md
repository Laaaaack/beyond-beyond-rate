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
- [x] `my_project/code/synthetic/ccisi/ccisi_tau.ipynb` — **cleared** (test-time-only). Training uses `forward(x)` without `f`; perturbation lives in a separate `forward_with_hidden_perturbation` that is only called inside `torch.no_grad()` at eval. No fix needed.
- [x] `my_project/code/synthetic/ccisi/ccisi_delay.ipynb` — **cleared** (same pattern as ccisi_tau).
- [x] `my_project/code/synthetic/coincidence/coin_tau.ipynb` — **cleared** (same pattern; train forward has no `f`, perturbation only at eval under `no_grad`).
- [x] `my_project/code/synthetic/coincidence/coin_delay.ipynb` — **cleared** (same).
- [ ] `my_project/code/perturbation/jitter/jitter_train.ipynb` — **confirmed buggy** (see weight-freeze evidence above). Re-run after fix; current σ > 0 results are not interpretable.
- [ ] `my_project/code/perturbation/shift/shift_train.ipynb` — **confirmed buggy** by code inspection + result-curve signature (see Investigation findings below). `forward(x, sigma)` calls `shift_hidden_batch(hidden1, sigma)` (numpy round-trip) directly, no STE.
- [ ] `my_project/code/perturbation/deletion/deletion_train.ipynb` — **confirmed buggy** by code inspection + result-curve signature. `forward(x, p_d)` calls `delete_hidden_batch(hidden1, p_d)` directly, no STE.
- [x] `my_project/code/perturbation/inverse/inverse_train.ipynb` — **cleared** (Phase 4, test-time-only). `forward(x)` is the unperturbed training pass; `forward_with_hidden_perturbation` and `forward_with_hidden_reversal` are only called from eval loops under `torch.no_grad()`.
- [x] `my_project/code/realistic/shd/shd_train.ipynb` — **cleared** (test-time-only).
- [x] `my_project/code/realistic/ssc/ssc_train.ipynb` — **cleared** (test-time-only).

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

---

### Investigation findings (2026-05-09)

A full sweep of `my_project/code/` was carried out before committing to a fix,
to confirm that the autograd-severing diagnosis is correct and to scope which
notebooks actually need patching. The picture below is consistent: every
notebook with a flat / cliff-shaped sweep curve is structurally buggy in the
way described above; every notebook with a smoothly degrading curve perturbs
only at evaluation under `no_grad` and is structurally clean.

#### 1. The bug is real, and it is the *only* explanation for the flat curves

The three `*_train.ipynb` notebooks under `code/perturbation/` (jitter, shift,
deletion) all instantiate the same anti-pattern: a `forward(x, sigma)` /
`forward(x, p_d)` that calls a `*_hidden_batch(...)` helper which goes through
`hidden_spikes.detach().cpu().numpy() → ... → torch.from_numpy(...).to(dev)`,
then assigns the result back into the forward-pass tensor variable with no STE
wrapper. Concretely:

- `jitter_train.ipynb:514` — `hidden1 = jitter_hidden_batch(hidden1, sigma)`
- `shift_train.ipynb:483`  — `hidden1 = shift_hidden_batch(hidden1, sigma)`
- `deletion_train.ipynb:662` — `hidden1 = delete_hidden_batch(hidden1, p_d)`

In all three, `_first_hidden(x)` packages `psp → fc1 → spike → delay1`, so
losing the gradient at the perturbation site freezes `fc1`, `delay1`, and the
PSP filter at their initialization values.

#### 2. Diagnostic signature: cliff + plateau (not graceful decay)

Bug-affected sweep curves all share the same shape — a sharp cliff at the
first non-zero perturbation level, then a flat plateau across all higher
levels. Clean sweeps decay gradually. Side-by-side:

| Notebook | bug? | curve (acc at increasing perturbation) |
|---|---|---|
| `jitter_part_nodelay`     | yes | σ=0 0.380 → σ=1 0.156 → σ=3 0.145 → σ=5 0.146 → σ=10 0.158 → σ=17 0.182 → σ=25 0.174 |
| `shift_whole_delay`       | yes | σ=0 0.864 → σ=1 0.414 → σ=3 0.343 → σ=5 0.299 → σ=10 0.308 → σ=17 0.295 → σ=25 0.276 |
| `deletion_whole_delay`    | yes | pd=0.0 0.627 → pd=0.2 0.108 → pd=0.4 0.086 → pd=0.6 0.086 → pd=0.8 0.051 |
| `ccisi_tau` (test-time)   | no  | f=0.0 1.000 → f=0.2 0.986 → f=0.4 0.830 → f=0.6 0.672 → f=0.8 0.572 → f=1.0 0.539 |
| `shd_whole_delay` (test)  | no  | f=0.0 0.864 → f=0.2 0.678 → f=0.4 0.522 → f=0.6 0.382 → f=0.8 0.299 → f=1.0 0.287 |

The cliff between σ=0 and the first σ>0 in the buggy notebooks is huge
(0.38→0.16, 0.86→0.41, 0.63→0.11) and is not consistent with biological
perturbation strength; sigma=1 ms of jitter destroying half the accuracy
should not happen on its own. After the cliff, the plateau is essentially
flat — accuracy depends almost not at all on σ — which is the unmistakable
fingerprint of "the upstream half of the network has been frozen at random
init, and the downstream half is doing whatever it can with random features."

#### 3. Training-log evidence: identical loss plateau across σ>0

`jitter_part_nodelay_sigma{1,3,5,10,17,25}_training_log.json` all converge to
val_loss ≈ 351–356 and val_acc ≈ 0.15–0.18 — independent of σ. The σ=0 run
reaches val_acc 0.36 with val_loss 324. If the perturbation were actually
flowing through training, larger σ should produce systematically worse
training loss; instead all σ>0 runs land on the same plateau because they are
all training the same downstream-only sub-network. Same pattern in shift /
deletion training logs.

#### 4. Wall-clock evidence

In `jitter_part_nodelay`, σ=0 takes 16:32 and every σ>0 takes ~50 min — a
~3× slowdown that matches a per-batch CPU↔GPU + numpy round-trip and is
present at every σ>0 regardless of σ value. The slowdown is independent of
perturbation strength, which is what you would expect if the cost is the
trip itself, not the work done in numpy.

#### 5. Weight-freeze signature (already documented above for jitter)

`fc1.weight_g`/`weight_v` mean and std are bit-identical across σ ∈ {1, 3, 5,
10, 17, 25} — and differ from σ=0 — for `jitter_part_nodelay`. This is
the strongest single piece of evidence: parameters that share an identical
post-init fingerprint across six independent training runs cannot have
received any gradient signal in any of those runs.

#### 6. Notebooks that look superficially similar but are clean

The seven notebooks below all contain `detach().cpu().numpy()` calls inside
a `*_hidden_batch` function, but they are structurally clean because the
perturbation never enters the training forward pass:

- `synthetic/ccisi/ccisi_tau.ipynb`, `ccisi_delay.ipynb`
- `synthetic/coincidence/coin_tau.ipynb`, `coin_delay.ipynb`
- `realistic/shd/shd_train.ipynb`, `realistic/ssc/ssc_train.ipynb`
- `perturbation/inverse/inverse_train.ipynb`

Common pattern: a plain `forward(self, x)` (no perturbation argument) is
called as `outputs = net(x_batch)` during training; a separate
`forward_with_hidden_perturbation(self, x, f)` (or
`forward_with_hidden_reversal`) is called only from the eval loop, inside
`with torch.no_grad():`. Under `no_grad` there is no autograd graph to sever,
so the numpy round-trip is harmless. Their result curves degrade smoothly,
consistent with this diagnosis.

#### 7. Conclusion

The autograd-severing diagnosis is the correct and complete explanation for
the flat curves. There is no additional independent bug:

- All three structurally suspicious training notebooks (jitter, shift,
  deletion) exhibit the cliff+plateau accuracy signature, the identical
  loss-plateau-across-σ training-log signature, the per-batch slowdown
  signature, and (verified for jitter) the bit-identical-weights signature.
- All seven structurally clean notebooks (ccisi×2, coin×2, shd, ssc,
  inverse) exhibit smooth decay.
- Phase 1 ISI was already in the cliff regime before the STE patch, and
  moved to smooth decay after. Same mechanism.

The fix is the STE wrapper described above. Action: patch jitter_train,
shift_train, deletion_train; retrain all σ>0 / pd>0 sweeps. ccisi, coin,
realistic, and inverse notebooks are clean and do not need re-running.

---

## Issue 2 — Does the "delay-after-spike" anti-pattern cause a silent no-op?

`my_project/docs/knowledge_bank/where_to_apply_perturbation_between_layers.md`
warns that if `_first_hidden` ends with `if use_delay: x = self.delay1(x)`,
the perturbation hook receives a fractional tensor (because `slayer.delay` is
described there as doing linear interpolation between time bins) and
`np.where(spike_train == 1)` matches nothing, so the perturbation is a silent
no-op for delay runs.

**Status:** the *structural* anti-pattern is present in many notebooks (see
audit below), but in this version of `slayerSNN` it does **not** actually
produce the silent no-op described in the knowledge bank. Both code-level
inspection of slayer and the empirical sweep curves contradict the no-op
prediction. So this is a code-quality / robustness concern, not the cause of
the flat curves we are trying to fix.

### Why the silent-no-op prediction does not hold

1. **slayerSNN's delay layer floors the delay before shifting.** From
   `venv/Lib/site-packages/slayerSNN-.../slayerSNN/slayer.py:351-373` (the
   docstring of `slayer.delay`):

   > The delay parameter is stored as float values, however, **it is floored
   > during actual delay applicaiton internally**.

   The forward call is `_delayFunction.apply(input, delay, Ts)` →
   `slayerCuda.shift(input, delay.data, Ts)` (`slayer.py:923-942`). The
   shift is integer-step. Therefore `delay(binary_spikes)` is binary
   spikes shifted by `floor(delay)` time steps — still strictly binary,
   so `np.where(... == 1)` matches every spike. Perturbation is *not* a
   no-op.

2. **Empirical: the SHD/SSC `_delay` sweep curves degrade smoothly.** If
   the no-op were real, the delay sweeps should sit flat at the f=0
   accuracy. Instead:

   | Sweep | f=0 → f=1 |
   |---|---|
   | `shd_whole_delay`  | 0.864 → 0.678 → 0.522 → 0.382 → 0.299 → 0.287 |
   | `shd_part_delay`   | 0.697 → 0.499 → 0.342 → 0.224 → 0.171 → 0.149 |
   | `shd_norm_delay`   | 0.491 → 0.308 → 0.185 → 0.134 → 0.115 → 0.109 |
   | `ssc_whole_delay`  | (n/a, missing) |
   | `ssc_part_delay`   | 0.470 → 0.349 → 0.226 → 0.119 → 0.062 → 0.049 |
   | `ssc_norm_delay`   | 0.366 → 0.271 → 0.175 → 0.093 → 0.049 → 0.041 |

   These are textbook smooth-degradation curves, identical in shape to the
   `_nodelay` sweeps (just at higher absolute accuracy because the delay
   model is more capable). The hook is clearly receiving binary spikes.

3. **The two effects do not stack.** `jitter`/`shift`/`deletion` `_delay`
   variants show the same cliff+plateau as their `_nodelay` siblings, just
   from a higher baseline at σ=0 (e.g. `jitter_part_delay` 0.70 → 0.33 →
   0.29 → 0.23 → … vs `jitter_part_nodelay` 0.38 → 0.16 → 0.14 → 0.15 → …).
   If the no-op bug were active for delay variants, they would sit flat
   *at* the σ=0 accuracy, not collapse like the nodelay runs. They
   collapse because of Issue 1 (autograd severance), not Issue 2.

### Audit: where the structural anti-pattern actually exists

`_first_hidden(x) = slayer.spike(...) ; if use_delay: x = delay1(x)` — the
delay sits *after* the spike inside the same method, so the method's return
value is "delayed spikes" rather than "raw spikes". Found in:

- `code/realistic/shd/shd_train.ipynb` (line ~453)
- `code/realistic/ssc/ssc_train.ipynb` (line ~484)
- `code/perturbation/inverse/inverse_train.ipynb` (line ~392)
- `code/perturbation/jitter/jitter_train.ipynb` (line ~481)
- `code/perturbation/shift/shift_train.ipynb` (line ~450)
- `code/perturbation/deletion/deletion_train.ipynb` (line ~601)

`isi_delay.ipynb`, `coin_delay.ipynb`, and `ccisi_delay.ipynb` already follow
the cleaner pattern (delay sits at the start of `_second_layer`, after the
perturbation hook, or on the input side before the first spike — `_first_layer`
ends with `slayer.spike(...)` so the hook receives strictly binary spikes).
All three have an explicit comment calling out the binary-input requirement
of the perturbation function. Verified 2026-05-10 by reading
`_first_layer` / `_second_layer` in each of the three synthetic delay
notebooks; the previous audit entry for `isi_delay.ipynb` was stale.

### Recommendation

Treat Issue 2 as a separate, lower-priority hygiene fix. Do not block the
Issue 1 STE retraining on it. Specifically:

1. **Don't expect Issue 2 to change any current results.** The current
   slayer floors the delay; binary in → binary out. Nothing downstream
   actually mis-fires.
2. **Do refactor the layer methods anyway,** because:
   - The anti-pattern is fragile across slayer versions (the docstring
     could change, or a "fractional delay" variant could be enabled).
   - It mixes routing (delay) with compute (psp+fc+spike) in the same
     method; separating them makes the perturbation site obvious and
     makes the pre-hook tensor easier to assert on (`unique == [0, 1]`).
   - The two notebooks that already separate them (`coin_delay`,
     `ccisi_delay`) read more clearly and are easier to audit.
3. **Cheap sanity check to add after any retraining.** Inside the eval
   loop, after `_first_hidden`, assert
   `torch.unique(hidden1).tolist() == [0.0, 1.0]`. If a future slayer
   release stops flooring, this catches the regression immediately.

Issue 2 does not require re-running anything that wasn't already going to be
re-run for Issue 1. The Issue 1 STE retraining will fold the refactor in
naturally for jitter / shift / deletion. The realistic and inverse
notebooks can have the refactor applied opportunistically without retraining.

---

## Issue 3 — Coincidence dataset / SLAYER `tSample` mismatch

**Status:** resolved 2026-05-18. Generator patched, dataset regenerated, and
both coincidence notebooks re-run end-to-end on the 1000-step data. Surfaced
2026-05-10.

### Symptom

`coin_data_gen.py` produced samples with `N_TIMESTEPS = 4000` (20 windows of
200 ms each), but both training notebooks declare
`SIM_PARAMS = {"Ts": 1, "tSample": 1000}` and a dead-code constant `T = 1000`,
then load the data without slicing or padding. The notebooks' own startup cell
prints `Time steps: 4000`, contradicting their own `tSample=1000`.

| File | Declared time axis | Actual data (pre-fix) |
|---|---|---|
| `code/synthetic/coincidence/coin_data_gen.py:26` | `N_TIMESTEPS = 4000` | wrote `(60, 4000)` |
| `code/synthetic/coincidence/coin_tau.ipynb:79,93` | `tSample=1000`, `T=1000` (unused) | loaded `(60, 4000)` |
| `code/synthetic/coincidence/coin_delay.ipynb:102` | `tSample=1000`, `T=1000` (unused) | loaded `(60, 4000)` |

SLAYER does not error — `tSample` is used for internal time-axis bookkeeping
(loss masks, PSP filter length scaling, etc.), not as a hard bound on input
length — so the mismatch is silent. The previously trained checkpoints
(`data/coin_tau_lam*.pt`, `data/coin_delay_lam*.pt`) embed whatever convention
SLAYER ended up applying with this inconsistency in place and should be
considered stale once the dataset is regenerated.

### Fix applied — option 2 (match data to notebooks)

`coin_data_gen.py:26` was changed from `N_TIMESTEPS = 4000` to
`N_TIMESTEPS = 1000`. The generator's `n_windows = N_TIMESTEPS // WINDOW_SIZE`
expression now produces 5 coincidence windows per trial (was 20); no other
generator changes were needed because the rest of the pipeline is parametric
on `N_TIMESTEPS`. Both notebooks already declare `tSample=1000` / `T=1000`,
so they need no edits — once the dataset is regenerated they will load
1000-step samples that match.

The alternative (option 1: bump notebooks to `tSample=4000`) was rejected
because per-trial signal per window is bounded — preserving 20 windows mostly
buys redundancy, not new structure — and option 2 keeps wall-clock training
time roughly 4× lower per epoch.

### Remaining work

- [x] `my_project/code/synthetic/coincidence/coin_data_gen.py` — patched
  (`N_TIMESTEPS = 1000`).
- [x] Regenerated `coin_dataset.mat` by running
  `python coin_data_gen.py` from `my_project/code/synthetic/coincidence/`.
  Overwrote the per-lambda `coin_data_lam*.pt` files and the combined
  `coin_dataset.mat` with 1000-step data.
- [x] Re-ran `coin_tau.ipynb` end-to-end on the new 1000-step data;
  cached cell output now reports `Time steps: 1000` and all lambda models
  retrained from scratch.
- [x] Re-ran `coin_delay.ipynb` end-to-end (same).
- [x] Discarded the old (pre-fix) `data/coin_tau_lam*.pt` and
  `data/coin_delay_lam*.pt` checkpoints; current ones are the post-fix
  retrains.

### Cross-check: other phases

Re-verified 2026-05-10 by reading both the data generators and the
notebooks (a `tSample`-only grep was not sufficient — it missed the
length discrepancy described below for realistic).

- **Synthetic ISI** (`isi_tau.ipynb`, `isi_delay.ipynb`): `tSample=1000`,
  dataset shape `(N, 10, 1000)`. Clean match.
- **Synthetic CCISI** (`ccisi_tau.ipynb`, `ccisi_delay.ipynb`):
  `tSample=1000`, but the notebooks' cached output prints
  `Time steps: 10000` and the load line shows `X=(3598, 20, 10000)`. **Same
  family as coincidence — flag for follow-up; re-run the load cell against
  the current `ccisi_dataset.h5` to confirm whether the cached output is
  stale or the bug is live. If live, decide between option 1
  (bump `tSample` to 10000) and option 2 (regenerate at 1000).**
- **Realistic SHD/SSC** (`shd_train.ipynb`, `ssc_train.ipynb`): notebooks
  declare `tSample=200`, but the underlying generators
  (`shd_data_gen.py:sparse_to_dense(..., nb_steps=100, max_time=1.4)`,
  `ssc_data_gen_whole.py:nb_steps=100`, `ssc_data_gen_norm_part.py`
  inherits T from `ssc_whole.h5`) all produce `T=100` raw bins.
  **The 100 → 200 gap is actively handled** by the
  `load_shd_data(..., target_T=SIM_PARAMS["tSample"])` loader, which pads
  with trailing zeros (`if T < target_T: padded[:, :, :T] = X`).

  **This is the original Beyond Rate convention, not a my_project quirk.**
  Cross-checked 2026-05-10 against the upstream paper code:
  - `temporal_shd_project/code/realistic/shd/shd_data_gen.py:28` is
    byte-identical to my_project's version (`nb_steps=100, max_time=1.4`).
  - `temporal_shd_project/code/realistic/shd/shd_train.py:349` declares
    `sim_param = dict(Ts=1, tSample=200)`; lines 361–369 perform the same
    `T_target = 200; if T < T_target: padded[:, :, :T] = X` pad with the
    comment `# Pad time dimension to 200 as in notebook`.
  - The SpikeRate loss config (`shd_train.py:164`,
    `ssc_train.py:185`) sets `tgtSpikeRegion: {'start': 0, 'stop': 200}`,
    which only makes sense if the simulation actually runs for 200 bins.
    The trailing 100 zero-bins are settling time during which the readout
    accumulates spikes; the padding is load-bearing.

  Caveat for SSC: `temporal_shd_project/code/realistic/ssc/ssc_train.py`
  declares `tSample=200` but does **not** pad inside the script — it
  loads pre-split per-f h5 files directly via `SpikeDataset` and feeds
  them to SLAYER as-is. Either an unseen preprocessing step pads them
  to 200, or the original SSC runs went out with the same silent
  length mismatch we just fixed in coincidence. my_project's SSC
  notebook is **stricter than the original**: it routes SSC through
  the same `load_shd_data(..., target_T=200)` padder used for SHD,
  making the convention explicit. Treat this as a cleanup rather than
  a port issue.

- **Perturbation** (`inverse_train.ipynb`, `jitter_train.ipynb`,
  `shift_train.ipynb`, `deletion_train.ipynb`): all share the SHD pipeline
  — `tSample=200`, same `load_shd_data(..., target_T=200)` padder. Same
  100 → 200 padding as realistic. By-design, not a bug.

#### Secondary concern: physical-time interpretation in SHD/SSC

`nb_steps=100` over `max_time=1.4 s` means each SHD/SSC bin represents
~14 ms of recording, but the training notebooks declare `Ts=1` (1
simulation unit per bin). SLAYER itself is bin-agnostic, so this doesn't
cause runtime errors; however the `LIF_PARAMS` time constants
(`tauSr`, `tauRef`, etc.) and the learned `tau` reported in the
"Model Analysis" cells are in *bin units*, not milliseconds. Reading the
learned `tau` of e.g. 50 as "50 ms" would be wrong by ~14× for SHD/SSC.
Carried over from Beyond Rate's setup; flagged here for future
interpretation work, not part of the Issue 3 fix.

#### Mismatch summary

| Phase | Status |
|---|---|
| Coincidence | resolved 2026-05-18. Generator patched (option 2), dataset regenerated, both notebooks re-run end-to-end. |
| CCISI | suspected mismatch (cached output suggests `T=10000` vs `tSample=1000`). Verify with a fresh load cell run. |
| ISI | clean. |
| Realistic SHD/SSC | 100→200 zero-padding traces back to the original Beyond Rate paper (verified in `temporal_shd_project`); load-bearing for the SpikeRate readout window `[0, 200]`. Not a bug. Bin-vs-ms interpretation flagged separately. SSC: my_project port is stricter than the original (explicit `target_T=200` pad). |
| Perturbation (jitter/shift/deletion/inverse) | same as realistic SHD/SSC. Not a bug. |
