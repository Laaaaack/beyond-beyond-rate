# First Milestone — Does sparsity increase temporal processing?

**Status:** in progress — Step 1 (inducing sparsity): **calibration complete** on the
**SGD-delay** arm, full 15-model run ready to launch. A second **no-delay** arm has
been added (script written 2026-07-22, calibration probe not yet run). Steps 2–4 not
started for either arm. See the Progress log below.
**Owner:** _(you)_
**Full design & rationale:** [sparse_network.md](sparse_network.md) — read that first if
anything below is unclear; this file is the execution checklist for the *first,
smallest* experiment only.

---

## Progress log — current state (updated 2026-07-18)

We are still on **Step 1 (induce sparsity)**; the eval/analysis Steps 2–4 are
unchanged and not started. What follows is the record of getting the sparsity
mechanism to work, because the original §5 plan (plain L1 penalty, swept over
`lam`) turned out **not** to work with this optimiser.

**Where the code actually lives.** The training scripts are
[jitter_test_withDelay.py](../../exp_sparse_network/jitter/jitter_test_withDelay.py)
and [jitter_test_noDelay.py](../../exp_sparse_network/jitter/jitter_test_noDelay.py)
under `my_project/exp_sparse_network/jitter/` — *not* the
`exp_fixed_weight_perturbation/…` folder sketched in §4. (Naming history: the
with-delay script began as `jitter_train.py`; it became `jitter_train_withDelay.py`
when the no-delay arm was added, and both were renamed `jitter_test_*` on 2026-07-23
to mark that `jitter` is the *pilot* experiment — once its full run checks out the
protocol extends to `shift`, `deletion`, etc. Earlier log entries below may refer to
the old names.) Both write checkpoints → `.../jitter/data/` and logs
→ `.../jitter/log/`, tagged apart by `delay` vs `nodelay` in the run tag, so the two
arms share a folder without colliding; the summaries are
`sparse_whole_delay_train_summary.json` and `sparse_whole_nodelay_train_summary.json`.
`QUICK_TEST=True` runs a tiny fast grid for calibration; set it `False` for the real
15-model run.

**How the method evolved (three attempts):**

| Attempt | Result | Lesson |
|---|---|---|
| **1. Plain L1** `lam·Σspikes`, penalty on from epoch 0 | Every `lam` in 1e-4…1e-2 silences the hidden layer within **one epoch** → chance acc, 0 firing, unrecoverable. `lam=0` trains fine (86%). | Nadam/Adam rescales the persistent L1 gradient to full-size suppression steps regardless of `lam`; once a neuron is silent its surrogate gradient vanishes (dead-neuron) → total collapse. Calibrating `lam` *down* is impossible — there is no value between "no effect" and "instant death" (this is why AdamW exists). |
| **2. Hinge** `relu(rate−target)` **+ warm-up** (penalty off 100 ep) | No collapse (85–86% acc), but firing won't drop: at `target=3`, strengths 0.01→0.3 all leave firing at **9.5–12** (≈ dense anchor 11). | An additive penalty cannot escape the dense basin the warm-up settled into. |
| **3. Read attempt-2 trajectories** | After the penalty engages (ep 100), `train_rate` only ever **rises** (task pulls firing to ~13); even strength 0.3 dips to 8.2 then rebounds. The sparsest the net *ever* is = **4.67 at epoch 1**, before warm-up ends. | **The warm-up is counter-productive for sparsity** — it parks the net dense. Sparsity must be shaped *from the start of training*. |

**Calibration snapshot** (QUICK_TEST: SHD whole, SGD-delay, seed 42, 400 ep,
warm-up 100, target=3 — *not* final numbers):

| strength | spikes/neuron | clean_acc | silent |
|---|---|---|---|
| dense anchor (no penalty) | 10.98 | 86.2% | 35% |
| 0.01 | 11.59 | 86.4% | 33% |
| 0.03 | 12.12 | 84.6% | 27% |
| 0.10 | 10.90 | 84.6% | 31% |
| 0.30 | 9.50 | 85.2% | 37% |

No usable spread — the "sparsest" is 0.87× the anchor, far from the >2× the
milestone needs.

**✅ RESOLVED (2026-07-19) — drop the warm-up, sweep the strength.** Setting
`WARMUP_EPOCHS = 0` fixed it: with the hinge shaping firing from epoch 0 (no dense
basin to fight), calibration gives a clean, monotonic, no-collapse sparsity
gradient. Two-part quick-test at `target=3` (seed 42, 400 ep, warm-up 0), measured
spikes/neuron and clean_acc:

| strength | spikes/neuron | clean_acc | silent |
|---|---|---|---|
| 1e-3 | 11.57 | 83.9% | 27% |
| 1e-2 | 9.16 | 82.6% | 49% |
| 1e-1 | 5.79 | 79.8% | 63% |
| 3e-1 | 5.47 | 83.2% | 62% |
| 1.0 | 3.30 | 79.2% | 67% |

A **~3.5× firing spread** (11.6 → 3.3), every model well above chance, and **no
accuracy floor** even at strength 1.0 — the hinge is a robust collapse guard.

**Key correction to the plan.** The *target* is **not** the binding control: the
task loss pulls hidden firing up to a plateau *above* any target, and the hinge
only lowers where that plateau sits — partly by **silencing** neurons
(`silent_fraction` climbs 27% → 67% across the sweep). The penalty **strength** is
what actually sets the achieved sparsity. So the real run **sweeps strength at a
fixed target**, not the target sweep §5 sketched.

**Locked configuration for the 15-model run** (`QUICK_TEST = False`):
`SPARSITY_TARGETS = [3.0]` (fixed), `PENALTY_STRENGTHS = [1e-3, 1e-2, 3e-2, 1e-1,
1.0]` (the swept axis; 3e-1 dropped — it saturates on top of 1e-1),
`SEEDS = [42, 43, 44]`, `WARMUP_EPOCHS = 0`, `EPOCHS = 1250` → 5 × 3 = 15 models.

**Sweep knobs (final):** we sweep the **penalty strength** (`PENALTY_STRENGTHS`) at
a fixed **target** (`SPARSITY_TARGETS = [3.0]`), not `lam` and not the target.
Always analyse against the **measured** firing rate, never the target/strength.
Full rationale in the memory note `l1-spike-penalty-collapses-under-nadam`.

### Second arm added (2026-07-22): the no-delay network

The milestone originally fixed the network to SGD-delay (§3) and left the no-delay
variant to §9's "if it works" list. It is being run **in parallel** instead, via
[jitter_test_noDelay.py](../../exp_sparse_network/jitter/jitter_test_noDelay.py).

**Why now.** Learnable axonal delays are themselves a timing mechanism, so in the
SGD-delay net the jitter test measures sparsity's effect *on top of* whatever the
delays contribute — a confound sitting inside the headline result. Stripping the
delays removes it: any surviving sparsity → timing-dependence trend must come from
the SRM neurons' own membrane dynamics. Running both also turns the confound into a
finding — the gap between the two arms' temporal scores is how much of the temporal
processing the delays were doing.

**What differs from the with-delay script.** Only the network. `delay1`/`delay2`,
the adaptive delay clamping schedule (the epoch-250 / 150-update `thea` logic) and
the `delay_mean` log field are removed outright rather than switched off by a flag,
so the class is `JitterSHDNetworkNoDelay` and spikes go straight `fc1 → fc2 → fc3`.
Everything else is held identical for comparability: same dataset and fixed splits,
same hinge penalty applied from epoch 0, same `EPOCHS = 1250`, LR, scheduler,
batch size, early-stop patience, and the same `(target, strength, seed)` grid.
Run tags read `sparse_whole_nodelay_tgt3_str{strength}_seed{seed}`.

**First probe (2026-07-22): the delay grid does NOT transfer.** The strength → firing
map was calibrated on the *delay* net only; the reasoning that it would transfer (the
penalty acts on `fc1`, and `delay1` sits *after* the penalty site) turned out to be
wrong in two ways. A range-spanning probe (the inherited grid `[1e-3, 1e-2, 1e-1,
1.0]`, seed 42, 400 ep, warm-up 0) gave:

| strength | no-delay sp/neuron | delay sp/neuron | no-delay clean_acc | delay clean_acc | no-delay silent | no-delay trajectory |
|---|---|---|---|---|---|---|
| 1e-3 | 6.07 | 11.57 | 55.8% | 83.9% | 46% | stable, never silent |
| 1e-2 | 4.75 | 9.16 | 52.8% | 82.6% | 58% | stable, never silent |
| 1e-1 | 1.77 | 5.79 | 48.3% | 79.8% | 80% | **silent (rate=0) for ~26 ep, recovers to 1.72** |
| 1.0 | 0.97 | 3.30 | 48.0% | 79.2% | 81% | **silent (rate=0) for ~28 ep, recovers to 1.01** |

Two problems. (1) **The no-delay net fires ~half as much at every strength**, so the
same grid samples a lower, non-overlapping firing band and never reaches a dense
regime (even the weakest penalty, 1e-3, already sits at 6.1; there is no measured
`str=0` anchor). (2) **The strong half collapses**: at 1e-1 and 1.0 the layer goes
fully silent within 1–2 epochs and stays dead for ~26–28 epochs before partially
recovering to a degenerate ~1–1.7 sp/neuron (~80% silent) — the delay arm settled
smoothly at the same strengths and never went silent. The weaker no-delay net simply
fires less intrinsically, so the identical hinge is proportionally far more aggressive
and tips it over the edge. Seed 42 recovered; seeds 43/44 are not guaranteed to.

**Second probe (2026-07-23): recalibration at warm-up 0 — no clean gradient exists.**
Grid `[0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]`, seed 42, 400 ep, warm-up 0:

| strength | sp/neuron | clean_acc | silent | collapse test |
|---|---|---|---|---|
| 0 (anchor) | 6.43 | 54.2% | 47% | clean (rate min 4.28) |
| 1e-3 | 6.07 | 55.8% | 46% | clean (min 4.00) |
| 3e-3 | 6.38 | 55.4% | 44% | clean (min 3.97) |
| 1e-2 | 4.75 | 52.8% | 58% | clean (min 2.95) |
| 3e-2 | 3.64 | 49.2% | 69% | near-silent (0.01) at epoch 1, ~84 ep to recover, then stable 3.52 |
| 1e-1 | 1.77 | 48.3% | 80% | dead 26 ep, degenerate |

The recalibration explained the delay grid's failure but did **not** yield a usable
grid, because the no-delay net's strength → firing response is not a slope but
**flat-then-cliff**:

- **Dense end is inert.** `str=0`, `1e-3`, `3e-3` all sit at ~6.3 sp/neuron — the hinge
  does essentially nothing below `1e-2`. The natural (unpenalised) firing rate is
  ~6.3, and those three points are redundant.
- **Then a cliff, not a slope.** The only collapse-free band is `0 → 1e-2`, i.e.
  6.4 → 4.75 sp/neuron — a **1.35× spread**, well under the milestone's >2× target.
  Everything sparser is reached only by transiting a silence bottleneck: `3e-2` dips
  to 0.01 at init and takes ~84 ep to crawl back (fragile — likely to land dead on
  another seed), `1e-1` is dead for 26 ep and degenerate.

The delay arm had a smooth 11.6 → 3.3 slope; the no-delay arm has no equivalent
intermediate sparse regime at warm-up 0.

**Third probe (2026-07-23): warm-up 15 kills the collapse but over-corrects into the
attempt-3 dense-lock.** Added `WARMUP_EPOCHS = 15` (the task loss establishes the
~6-spike pattern before the hinge engages) and re-ran the strong end. Grid
`[1e-2, 3e-2, 1e-1, 3e-1, 1.0]`, seed 42, 400 ep, warm-up 15, against probe 2's
warm-up-0 numbers:

| strength | probe 2 (wu 0) sp/neuron | probe 2 collapse | probe 3 (wu 15) sp/neuron | probe 3 collapse | probe 3 clean_acc |
|---|---|---|---|---|---|
| 1e-2 | 4.75 | clean | 5.80 | clean | 57.6% |
| 3e-2 | 3.64 | dipped ~84 ep | 5.79 | clean | 53.4% |
| 1e-1 | 1.77 | dead 26 ep | 5.84 | clean | 57.6% |
| 3e-1 | — | — | 5.51 | clean | 54.1% |
| 1.0 | 0.97 | dead 28 ep | 2.80 | clean | 52.8% |

The warm-up did its safety job — **zero `rate=0` epochs anywhere**, the layer never
goes silent even at `str=1.0`. But it traded the collapse for the opposite failure:
penalties `1e-2 … 3e-1` (a 30× range) **all settle at ~5.5–5.8 sp/neuron**, barely
below the ~6.3 dense anchor, and only `str=1.0` escapes to 2.80. The trajectories show
why — once the penalty engages at epoch 15 (firing ~4.9), for `str ≤ 3e-1` firing
*drifts back up* to ~5.6 (the task loss out-pulls the hinge in the dense basin), while
only `str=1.0` is strong enough to steadily drag it down. This is exactly the attempt-3
dense-lock. Net: warm-up 0 gives sparsity-then-collapse; warm-up 15 gives
no-collapse-but-no-sparsity for the whole mid range.

**Fourth probe (2026-07-23): Path A works — grid found.** Kept warm-up 15 and moved the
sweep to the high-strength regime where (probe 3) the penalty actually bites. Grid
`[1e-2, 5e-1, 1.0, 3.0, 10.0]`, seed 42, 400 ep, warm-up 15:

| strength | sp/neuron | clean_acc | silent | collapse |
|---|---|---|---|---|
| 1e-2 | 5.80 | 57.6% | 44% | none |
| 5e-1 | 4.19 | 55.2% | 49% | none |
| 1.0 | 2.80 | 52.8% | 54% | none |
| 3.0 | 1.84 | 50.8% | 63% | none |
| 10.0 | 1.23 | 47.0% | 64% | none |

Clean on every axis: **no `rate=0` transit anywhere**; a **monotonic** firing spread
5.80 → 1.23 sp/neuron (~4.7× across the grid, ~5.1× including the ~6.3 `str=0` anchor,
well past the milestone's >2× bar); **all models well above chance** (57.6% → 47.0% vs
5%); and `silent_fraction` rising 44% → 64% but never reaching the degenerate 80%+ zone
the collapse route hit in probe 2. Points are well spaced in firing-rate space (the
analysis x-axis), slightly denser at the sparse end. `str=10` is still descending
(1.84 → 1.23), not fully saturated, but 1.23 sp/neuron at 47% acc is a good sparse
endpoint — no need to push into diminishing, lower-accuracy territory.

**Grid LOCKED (2026-07-23):** `PENALTY_STRENGTHS = [1e-2, 5e-1, 1.0, 3.0, 10.0]` at
target 3, warm-up 15. This is the probe-4 grid, measured end-to-end. Next: set
`QUICK_TEST = False` and run the full **5 strengths × 3 seeds (42/43/44) = 15 models**
at `EPOCHS = 1250`. Residual risk: the probe is seed-42-only; the warm-up guard makes
collapse unlikely on 43/44, and any dead model is dropped from the analysis per the
milestone rule, but the achieved firing rates on the new seeds are what get analysed
(never the strength). *Path B (shorter warm-up) was not needed — recorded above only
as the rejected alternative.*

> **Comparability note.** This leaves the no-delay arm on warm-up 15 while the delay
> arm is warm-up 0. That is acceptable: the milestone always analyses against
> *measured* firing rate and the chance-corrected `temporal_score`, and the protocol
> rule that matters (clean train, jitter eval-only) is untouched. The warm-up is only
> a training detail for reaching the firing range.

> Expect **lower clean accuracy** than the delay arm across the board (already visible:
> ~48–56% vs ~79–84% at 400 ep) — the no-delay net is the weaker model on SHD. That is
> not a failure of the sparsity mechanism. It is also exactly why `temporal_score` is
> chance-corrected and baseline-normalised (§4): the two arms must be compared on the
> normalised score, never on raw `acc(sigma)` drops.

---

## 1. What this milestone proves (in one paragraph)

We want to test one supervisor's prediction: **the sparser a hidden layer's
spiking activity, the more the network depends on precise spike *timing*
(temporal processing) rather than on firing *rate*.** We test it cheaply on a
single setting first. We train a handful of SHD networks that differ only in how
sparse their 1st hidden layer is (controlled by a regularisation penalty), and
then measure how much each one's accuracy collapses when we scramble the timing
of that layer's spikes at test time. If the prediction holds, the *sparser*
networks should collapse *more*. This milestone is done when we have one scatter
plot that either shows that trend or clearly doesn't.

## 2. Key idea in three lines

- **Sparsity** = how few spikes the 1st hidden layer fires. We set it with a
  **hinge** penalty toward a target firing rate during training (a plain L1
  penalty collapses the layer under Nadam — see the Progress log).
- **Temporal processing** = how much test accuracy drops when we jitter (randomly
  shift in time) that layer's spikes at evaluation only. Big drop = the network
  was relying on timing.
- We already have code for the jitter test. We only add the sparsity penalty.

> ⚠️ **Protocol rule (do not break):** sparsity is applied **during training**;
> jitter is applied **only at evaluation**. Never train with jitter on — that is
> a different experiment (perturbation-aware) and it erases the effect. See
> [sparse_network.md](sparse_network.md) §"Why the FIXED-WEIGHT protocol".

## 3. Scope of THIS milestone (deliberately narrow)

| Fixed choice | Value | Why |
|---|---|---|
| Dataset | SHD `whole` (input_dim 700) | one dataset only, for speed |
| Network | SGD-delay **and** SGD no-delay, as two separate arms | delay was expected to show the strongest effect; the no-delay arm removes the delays-as-timing-mechanism confound (Progress log, 2026-07-22) |
| Perturbation site | 1st hidden layer output | the project's primary site |
| Sparsity levels (`lam`) | 5 values, dense → sparse | enough to see a trend |
| Seeds per level | 3 | sparsity interacts with initialisation |
| **Total models to train** | **5 × 3 = 15 per arm** (30 across both) | |

Everything else (128–128 hidden units, 20 classes, `NumSpikes` loss, LR,
epochs, `SIGMA_VALUES`) is copied unchanged from the existing jitter scripts.

## 4. Files you will create

Make a new folder `my_project/exp_fixed_weight_perturbation/code/sparse_network/`
containing four scripts, each adapted from an existing one:

| New script | Copied from | Purpose |
|---|---|---|
| `sparse_train.py` | `.../perturbation/jitter/jitter_train.py` | clean training **+ sparsity penalty**; saves 15 checkpoints |
| `sparse_eval_jitter.py` | `.../perturbation/jitter/jitter_2ndLayer_evalOnly.py` | loads each checkpoint, sweeps **1st-layer** jitter at eval only |
| `measure_sparsity.py` | _(new, small)_ | reports each checkpoint's hidden firing rate + clean accuracy |
| `analyse.py` | _(new, small)_ | computes the temporal score and makes the plots |

> The eval template is the *2nd*-layer file, but this milestone perturbs the
> *1st* layer. The 1st-layer jitter logic already exists as
> `forward_with_hidden_perturbation(x, sigma)` inside
> `.../perturbation/jitter/jitter_train.py` — reuse that method rather than the
> 2nd-layer forward.

## 5. Step-by-step

### Step 0 — copy the scripts
Copy the two existing jitter scripts into the new folder under the new names
above. Confirm they run unchanged first (train 1 quick model, eval it), so you
know the baseline pipeline works before you modify anything.

### Step 1 — add the sparsity penalty to `sparse_train.py`

**1a. Let the network hand back its hidden spikes.** Change the clean `forward`:

```python
def forward(self, x, return_hidden=False):
    x = self._prepare_input(x)
    hidden1 = self._first_hidden(x)                 # binary spikes (differentiable)
    out = self._second_hidden_and_output(hidden1)
    return (out, hidden1) if return_hidden else out
```

**1b. Add the penalty in the training loop** (replace the plain loss line):

```python
outputs, hidden1 = net(x_batch, return_hidden=True)
task_loss = loss_fn.numSpikes(outputs, target)

# L1 sparsity penalty = mean spikes per hidden neuron per sample.
rate_reg = hidden1.sum(dim=-1).mean()
loss = task_loss + lam * rate_reg               # lam = 0 recovers normal training
```

**1c. Make `lam` a swept hyper-parameter**, exactly like `sigma` is swept in the
original script. Train one fresh model per `(lam, seed)` and save each as
`sparse_whole_delay_lam{lam}_seed{seed}.pt`.

Starting grid (these are *guesses* — calibrate in Step 1d):
```python
LAM_VALUES  = [0.0, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS       = [42, 43, 44]
```

**1d. Calibrate `lam` before committing to 15 runs.** Do this quickly:
1. Train `lam = 0` → record its hidden firing rate (the **dense anchor**) and
   clean accuracy.
2. Train the largest `lam` → if accuracy is already at chance (~5%), it is too
   strong; lower the top of the grid. If firing rate barely moved from the
   anchor, it is too weak; raise it.
3. Aim for a grid where the sparsest model still scores **clearly above chance**
   (say ≥ 40%) while firing much less than the dense anchor. Then run all 15.

> 💡 To validate the whole pipeline fast, temporarily set `EPOCHS = 400`. Only
> switch back to the full `EPOCHS = 1250` for the real 15-model run.

### Step 2 — measure sparsity + clean accuracy (`measure_sparsity.py`)
For every checkpoint, on the **test set**, record:
```python
# hidden1: (B, C, 1, 1, T) collected over the test set with sigma = 0
firing_rate     = hidden1.sum() / hidden1.numel()      # fraction of active bins  ← x-axis
spikes_per_neuron = hidden1.sum(dim=-1).mean()         # intuitive alt
silent_fraction = (hidden1.sum(dim=-1) == 0).float().mean()
clean_acc       = acc at sigma = 0
```
Save one row per checkpoint to a CSV/JSON: `lam, seed, firing_rate, silent_fraction, clean_acc`.

### Step 3 — jitter sweep per checkpoint (`sparse_eval_jitter.py`)
For every checkpoint, evaluate accuracy across the existing grid
`SIGMA_VALUES = [0, 1, 3, 5, 10, 17, 25]`, perturbing the **1st** hidden layer
(via `forward_with_hidden_perturbation`), `NUM_REPEATS = 3` for error bars, all
inside `torch.no_grad()`. Save `acc(sigma)` per checkpoint.

### Step 4 — score temporal processing + plot (`analyse.py`)
Turn each `acc(sigma)` curve into one number, **baseline-normalised and
chance-corrected** so that low-accuracy models are treated fairly:
```python
chance = 1 / 20                                   # = 0.05
retention = (acc[sigma_max] - chance) / (acc[0] - chance)   # sigma_max = 25
temporal_score = 1 - retention        # 0 = pure rate code, 1 = fully timing-dependent
```
Then make three plots:
1. **Headline:** `temporal_score` (y) vs `firing_rate` (x), one dot per
   (lam, seed). **H1 predicts a downward trend** (sparser = left = higher).
   Report Spearman correlation.
2. **Curves:** the raw `acc(sigma)` families, coloured by firing rate — should
   fan from flat (dense) to steep (sparse).
3. **Guard:** `clean_acc` vs `firing_rate` — proves the trend isn't just
   "sparser networks are worse."

## 6. How you know the milestone succeeded

- **Pipeline works:** 15 checkpoints trained; sparsity CSV and jitter curves
  produced without error.
- **Sparsity really varied:** the sparsest model's `firing_rate` is well below
  the `lam = 0` anchor (aim for a >2× spread), and all analysed models are above
  chance.
- **Result obtained (either direction is a valid finding):**
  - Headline plot shows a clear negative `firing_rate → temporal_score` trend
    (negative Spearman ρ) ⇒ supports H1, proceed to scale up.
  - Flat or positive trend ⇒ H1 not supported here; record it and check the
    diagnostics in [sparse_network.md](sparse_network.md) §"Pitfalls and confounds"
    before concluding.

## 7. Watch out for

1. **A plain L1 spike penalty collapses the layer under Nadam** — never use it;
   use the hinge `relu(rate − target)`. Any nonzero `lam` silenced the layer in
   one epoch (Progress log, attempt 1).
2. **A warm-up before the penalty prevents sparsity** — the net settles into a
   dense solution the additive penalty can't move. Apply the penalty from the
   start (Progress log, attempt 3).
3. **Dead network** — if a model sits at ~5% accuracy it silenced; a broken net
   is not "maximally temporal", so drop it from the analysis.
4. **Plot/analyse against measured `firing_rate`, never the target or strength** —
   the map from penalty settings to achieved sparsity is nonlinear and
   seed-dependent.
5. **Do not enable jitter during training** — training must be clean; jitter is
   eval-only.

## 8. Progress checklist

- [x] Training pipeline built
  ([jitter_test_withDelay.py](../../exp_sparse_network/jitter/jitter_test_withDelay.py)):
  `return_hidden` + sparsity penalty + clean-eval/summary in one script
- [x] Step 1 pitfall found & fixed: plain L1 collapses under Nadam → switched to
  hinge; also learned best-model tracking must be reset when the penalty engages
- [x] **Step 1 calibration DONE (2026-07-19)** — dropped the warm-up; sweeping the
  hinge **strength** (not the target) gives a clean ~3.5× firing spread, all above
  chance, no collapse (see the Progress log RESOLVED entry)
- [x] Step 1 done — strength grid `[1e-3,1e-2,3e-2,1e-1,1.0]` at target=3 spans
  11.6 → 3.3 spikes/neuron (>3× spread), all above chance
- [x] No-delay arm scripted (2026-07-22,
  [jitter_test_noDelay.py](../../exp_sparse_network/jitter/jitter_test_noDelay.py)):
  delays + clamping stripped, everything else held identical to the delay arm
- [ ] Step 1c (delay arm) — all 15 models trained (5 strengths × 3 seeds) at target=3
- [x] Step 1b (no-delay arm), probe 1 — inherited delay grid run; found the map does
  NOT transfer (fires ~half; 1e-1/1.0 collapse to silence then recover degenerate)
- [x] Step 1b (no-delay arm), probe 2 — recalibration grid `[0,1e-3,3e-3,1e-2,3e-2,1e-1]`
  run at warm-up 0; found flat-then-cliff response, no clean >2× gradient (dense end
  inert at ~6.3, collapse-free band only 6.4→4.75)
- [x] Step 1b (no-delay arm), probe 3 — warm-up 15 run; killed the collapse (no
  `rate=0` anywhere) but over-corrected into the attempt-3 dense-lock (strengths
  1e-2…3e-1 all ~5.5–5.8 sp/neuron; only 1.0 escapes to 2.8)
- [x] Step 1b (no-delay arm), probe 4 (Path A) — warm-up 15 + high-strength grid
  `[1e-2,5e-1,1.0,3.0,10.0]` gave a clean monotonic 5.8→1.2 spread, no collapse, all
  above chance. **Grid LOCKED**; `PENALTY_STRENGTHS` updated
- [~] Step 1b (no-delay arm) — `QUICK_TEST = False` set (2026-07-23); full run armed
  (5 strengths × seeds 42/43/44, `EPOCHS = 1250`, warm-up 15). **Launch pending**
- [ ] Step 2 — sparsity + clean accuracy table, per arm (largely produced inline by
  the training script's summary; may not need a separate `measure_sparsity.py`)
- [ ] Step 3 — jitter sweep (eval-only, **1st** layer) run for all checkpoints, per arm
- [ ] Step 4 — temporal scores + 3 plots produced, per arm
- [ ] Milestone verdict recorded (supports / does not support H1), plus the
  delay-vs-no-delay comparison of temporal scores

## 9. If it works — immediate next steps

Scale along one axis at a time (see [sparse_network.md](sparse_network.md)
§"Optional extensions"). The no-delay network was originally first on this list but
has been pulled forward and is now running as a parallel arm (Progress log,
2026-07-22), so what remains is: the 2nd-layer site, then the `part`/`norm` SHD
variants, and finally the synthetic ISI task as the cleanest confirmation venue.
