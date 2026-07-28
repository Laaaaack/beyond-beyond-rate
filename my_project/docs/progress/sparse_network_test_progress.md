# First Milestone — Does sparsity increase temporal processing?

**Status:** in progress — **Step 1 (inducing sparsity) COMPLETE for both arms**
(2026-07-27). no-delay: 6.5× monotone firing spread (1.3–7.9 sp/neuron, 49–59% acc);
delay (re-tuned): 5.5× monotone spread (2.1–11.7 sp/neuron, 78–89% acc). Both above
chance, no collapse. Next: **Step 3** — the eval-only 1st-layer jitter sweep — then
extend to shift/deletion. Steps 2–4 not started. See the Progress log below.
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

**Where the code actually lives (updated 2026-07-28 — reorg).** The sparsity training
is **shared** across the perturbation experiments (jitter, shift, deletion), so it now
sits at the top of `my_project/exp_sparse_network/`, with each perturbation's eval in
its own subfolder:

- **Training:** [sn_train_withDelay.py](../../exp_sparse_network/sn_train_withDelay.py)
  and [sn_train_noDelay.py](../../exp_sparse_network/sn_train_noDelay.py). They write
  checkpoints → `sn_data/` and per-model logs + summaries → `sn_log/`, tagged apart by
  `delay` vs `nodelay` in the run tag; the summaries are
  `sparse_whole_delay_train_summary.json` and `sparse_whole_nodelay_train_summary.json`.
- **Eval (jitter):** under `exp_sparse_network/jitter/` (`jitter_evalOnly_*` — eval code
  not finalised yet); it reads checkpoints from `../sn_data` and summaries from
  `../sn_log`.
- `QUICK_TEST=True` runs a tiny probe grid (outputs `_probe`-suffixed so a probe can
  never clobber a real run); `False` runs the real sweep.

(Naming/layout history: the with-delay script began as `jitter_train.py`, became
`jitter_train_withDelay.py` when the no-delay arm was added, then `jitter_test_*` on
2026-07-23, then **`sn_train_*` on 2026-07-28** when training moved up out of `jitter/`
into `exp_sparse_network/` and its outputs moved from `jitter/{data,log}` to
`sn_{data,log}`; the network classes were renamed `Jitter*SHDNetwork*` → `Sparse*`, and
the eval-only jitter code was moved out of the training scripts into `jitter/`. Earlier
log entries below may use the old names/paths.)

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
[sn_train_noDelay.py](../../exp_sparse_network/sn_train_noDelay.py).

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
so the class is `SparseSHDNetworkNoDelay` and spikes go straight `fc1 → fc2 → fc3`.
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

### Full-run results (2026-07-26): no-delay ready, delay needs a re-tune

Both arms completed the full 15-model × 1250-ep run. Measured spikes/neuron (seeds
42/43/44) and clean_acc:

**No-delay (warm-up 15) — clean, ready:**

| str | sp/neuron (s42,43,44) | mean | clean_acc |
|---|---|---|---|
| 0.01 | 6.50, 7.85, 6.80 | 7.05 | .59/.56/.56 |
| 0.5 | 4.31, 4.71, 3.99 | 4.34 | .56/.58/.58 |
| 1.0 | 2.91, 3.36, 3.17 | 3.15 | .55/.57/.53 |
| 3.0 | 2.04, 1.72, 1.82 | 1.86 | .55/.53/.52 |
| 10.0 | 1.42, 1.38, 1.21 | 1.34 | .50/.52/.49 |

Monotone every seed, **6.5× spread** (1.2 → 7.9), all well above chance, silent tops
at 61%. The full run reproduced probe 4. Nothing to fix.

**Delay (warm-up 0) — regressed at full epochs:**

| str | sp/neuron (s42,43,44) | mean | clean_acc |
|---|---|---|---|
| 0.001 | 12.63, 7.15, 12.76 | 10.85 | .87/.87/.89 |
| 0.01 | 11.66, 8.30, 9.41 | 9.79 | .89/.86/.86 |
| 0.03 | 8.15, 6.91, 8.53 | 7.86 | .88/.85/.87 |
| 0.1 | 8.69, 8.91, 8.28 | 8.63 | .87/.86/.86 |
| 1.0 | 4.80, 4.98, 4.79 | 4.85 | .86/.84/.84 |

Three problems: mean firing is **non-monotone** (0.03 → 0.1 rises, 7.86 → 8.63); the
four weak strengths **bunch at ~7–13 and are lost in seed noise** (str0.001 s43 = 7.15
is sparser than str0.03 s42 = 8.15 — seed spread exceeds the strength effect); and only
str1.0 clearly separates, so the arm has **two firing levels (~8–11 and ~4.8), not
five**. It clears >2× (2.66×) only technically, and its sparsest model (4.8 sp/neuron)
barely reaches the regime the no-delay arm covers down to 1.2.

**Root cause — the sparsity penalty re-densifies under longer training.** The delay
grid was calibrated at **400 ep** (seed 42: 11.6 → 3.3, monotone); the real run is
**1250 ep**, and over the extra 850 epochs the task loss pulls firing back up past
where the weak hinge held it (str0.1: 5.8 at 400 ep → 8.7 at 1250 ep). **Weak penalties
don't survive long training; only strong ones hold firing down.** This is the same
lesson the no-delay arm learned the hard way — its grid was pushed to `[1e-2…10]` and
held; the delay arm kept the weak `[1e-3…1.0]` grid and re-densified. See the memory
note [[sparsity-penalty-redensifies-at-full-epochs]].

**Why fix it before extending.** The sparsity-trained checkpoints are **reused across
jitter, shift, and deletion** (only the eval perturbation changes), so a weak delay
spread propagates into all three analyses. And `str1.0` delay still holds **84%
accuracy** — large headroom to push much sparser at no accuracy cost.

**Plan.** No-delay: proceed to Step 3 (eval-jitter sweep) against its 15 checkpoints.
Delay: re-tune stronger before extending — validate strong strengths with a short probe
first, then run the full re-tune.

**Delay re-tune probe (2026-07-26).** Grid `[1.0, 3.0, 10.0, 30.0]`, seed 42, 400 ep,
warm-up 0 (str1.0 included as a re-densification reference):

| str | sp/neuron | clean_acc | silent | trajectory |
|---|---|---|---|---|
| 1.0 | 3.30 | 79.2% | 67% | dead ~13 ep → stable 3.29 |
| 3.0 | 1.74 | 78.2% | 76% | dead ~18 ep → stable 1.71 |
| 10.0 | 0.93 | 72.6% | 80% | dead ~19 ep → stable 0.93 |
| 30.0 | 0.63 | 62.5% | 80% | dead ~20 ep → stable 0.69 |

Monotone, and every run **recovers to a stable plateau**. Three findings: (1) **the
delay arm collapses-and-recovers at warm-up 0 too — including str1.0** (correcting the
earlier "no collapse" claim, which never checked the trajectory). Every strong strength
goes dead for ~13–20 ep at init then recovers; unlike the no-delay net (which stayed
degenerate), the high-capacity delay net recovers, and str1.0's recovery is known-benign
(the full run gave tight 4.80/4.98/4.79, 84% acc across seeds). So transient collapse +
robust recovery is acceptable, and warm-up 0 stays consistent with the existing dense
anchors. (2) **Re-densification reference confirmed:** str1.0 → 3.30 @400 ep vs 4.85
@1250 ep = **1.47× factor** (an upper bound; strong hinges hold better), projecting str3
→ ~2.2 and str10 → ~1.3 at full epochs. (3) **Recovery degrades with strength:** str30
recovers only weakly (20-ep collapse, 0.63 sp/neuron, 80% silent, 62% acc) — fragile,
degenerate, seed-risky.

**Delay grid re-LOCKED (2026-07-26):** `PENALTY_STRENGTHS = [1e-2, 1.0, 3.0, 10.0]`,
warm-up 0. Projected full-run firing ~9.8 → 4.85 → ~2.2 → ~1.3: a clean ~7× monotone
spread whose sparse end matches the no-delay arm (1.34), all above chance, no degenerate
points. **str30 dropped** (fragile/degenerate per finding 3). This is 4 strengths vs the
no-delay arm's 5; str0.3 is the natural 5th (fills the 9.8→4.85 gap) but is unprobed, so
4 probe-validated points are preferred over gambling a 5th. `str1e-2` and `str1.0`
checkpoints already exist from the first full run (same warm-up-0 recipe) and can be
reused; only `str3`/`str10` are new. The pre-re-tune 1250-ep summary is backed up at
`sn_log/sparse_whole_delay_train_summary_redensified_1250ep.json`.

**Delay re-tune full run (2026-07-27): SUCCESS.** Fresh 12-model run of
`[1e-2, 1.0, 3.0, 10.0]`, 1250 ep, warm-up 0:

| str | sp/neuron (s42,43,44) | mean | clean_acc | silent |
|---|---|---|---|---|
| 0.01 | 11.66, 8.30, 9.41 | 9.79 | .89/.86/.86 | 25–39% |
| 1.0 | 4.80, 4.98, 4.79 | 4.85 | .86/.84/.84 | 43–47% |
| 3.0 | 3.01, 2.78, 3.21 | 3.00 | .82/.80/.85 | 42–55% |
| 10.0 | 2.12, 2.21, 2.48 | 2.27 | .82/.78/.82 | 35–51% |

**Monotone in the mean and every individual seed, 5.51× spread (11.7 → 2.1), all
78–89% acc** — the clean gradient the original run lacked (2.66×, bunched, non-monotone).
str3/str10 recovered tightly on all three seeds (the seed-fragility worry was unfounded
for str ≤ 10). The fresh run also self-healed the probe-clobbered str1.0-seed42
checkpoint.

**Projection correction:** str10 landed at 2.27, not the ~1.3 I projected from str1.0's
1.47× factor — the strong penalties re-densified *more* than str1.0 (str10: 0.93 @400 ep
→ 2.27 @1250 ep, 2.44×), so the reduced-epoch under-estimate is *worse* for stronger
penalties, not better. Harmless (we analyse measured firing), but it left the delay
sparse end at 2.1 rather than ~1.3. See [[sparsity-penalty-redensifies-at-full-epochs]].

**Both arms are now ready (Step 1 complete).** no-delay: 6.5× spread, 1.3–7.9 sp/neuron,
49–59% acc. delay: 5.5× spread, 2.1–11.7 sp/neuron, 78–89% acc. Both monotone, all above
chance. The arms overlap in firing (2.1–7.9), so the cross-arm comparison is supported;
each arm's within-arm trend is what H1 is tested on. Optional (not required): add
`str30` to the delay arm to reach ~1.5 sp/neuron and 5-point parity — at full epochs it
would likely land healthy (~1.5, not the degenerate 0.63 the 400-ep probe showed), given
how much the strong end re-densifies. Proceeding without it.

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
  ([sn_train_withDelay.py](../../exp_sparse_network/sn_train_withDelay.py)):
  `return_hidden` + sparsity penalty + clean-eval/summary in one script
- [x] Step 1 pitfall found & fixed: plain L1 collapses under Nadam → switched to
  hinge; also learned best-model tracking must be reset when the penalty engages
- [x] **Step 1 calibration DONE (2026-07-19)** — dropped the warm-up; sweeping the
  hinge **strength** (not the target) gives a clean ~3.5× firing spread, all above
  chance, no collapse (see the Progress log RESOLVED entry)
- [x] Step 1 done — strength grid `[1e-3,1e-2,3e-2,1e-1,1.0]` at target=3 spans
  11.6 → 3.3 spikes/neuron (>3× spread), all above chance
- [x] No-delay arm scripted (2026-07-22,
  [sn_train_noDelay.py](../../exp_sparse_network/sn_train_noDelay.py)):
  delays + clamping stripped, everything else held identical to the delay arm
- [x] Step 1c (delay arm) — DONE. First 15-model run re-densified (bunched, ~2.66×);
  re-tuned to strong grid `[1e-2,1.0,3.0,10.0]` → clean 5.51× monotone spread (2026-07-27)
- [x] Delay re-tune probe (2026-07-26) — `[1.0,3,10,30]` @400 ep confirmed monotone +
  stable recovery; str30 dropped (fragile/degenerate). **Grid re-LOCKED to
  `[1e-2,1.0,3.0,10.0]`** (warm-up 0), projected ~9.8→~1.3 spread
- [x] Delay arm full re-run (2026-07-27) — fresh 12-model run `[1e-2,1.0,3.0,10.0]` at
  1250 ep gave a clean **5.51× monotone spread** (11.7 → 2.1 sp/neuron), all seeds
  monotone, 78–89% acc. **Delay arm ready.**
- [x] Probe-clobber fix (2026-07-26) — both jitter scripts now append `RUN_SUFFIX`
  (`_probe` when `QUICK_TEST`) to checkpoints/logs/summary, so probes can't overwrite
  real-run artifacts. Carry this into the shift/deletion scripts
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
- [x] Step 1b (no-delay arm) — full run complete (2026-07-26): clean 6.5× monotone
  spread (7.05 → 1.34 sp/neuron), all above chance, no collapse. **Ready for Step 3**
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
