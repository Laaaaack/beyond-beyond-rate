# First Milestone v2 — Execution log: closing the perturbation-immune channels

**Status (2026-07-29):** **three mechanism iterations probed, none yet accepted.**
v2 (band) and v2.1 (exact-count) both failed the manipulation check and are
superseded. v2.2 (hard top-k truncation) is the current mechanism: its no-delay
probe is **done** and moved the manipulation much further than anything before it,
but at a large accuracy cost and with a residual leak. **The with-delay v2.2 probe
has not been run.** No full sweep has been run in any version.
**Owner:** _(you)_

**This is document 4 of 5. Read in order:**

| # | Document | What it is |
|---|---|---|
| 1 | [sparse_network.md](sparse_network.md) | the question and the conceptual landscape — **start there** |
| 2 | [sparse_network_test_progress.md](sparse_network_test_progress.md) | v1 execution log — getting a sparsity gradient to exist at all |
| 3 | [sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, and the diagnosis |
| 4 | **this file** | v2 execution log — attempts to make H1's premise actually hold |
| 5 | [sparse_network_test_progress_v3.md](sparse_network_test_progress_v3.md) | v3 design — why "sparsity" is two variables, and the experiment that separates them |

> ⚠️ **SUPERSEDED (2026-07-29) — do not run §5's remaining steps.** v3 retires the
> channel-closing programme. The short version: closing the immune channel makes the
> covariate zero, but *measuring* it costs ~40 min for all 27 checkpoints and it turns
> out to barely move across v1's whole gradient (count decode .70–.78), so it can be
> controlled rather than eliminated. Meanwhile v2.2's price — accuracy .552 → .237, raw
> firing inflated from ~6.3 to 12.5–21.9 — means it is no longer testing
> "sparser-activity networks" at all. The real blocking flaw was elsewhere: spikes per
> *active* neuron and *silent fraction* are confounded at ρ = −0.943 in v1's no-delay
> arm. See document 5 §6 for the full argument, and §2 for the measurements behind it.
> **Everything below stands as the record of what was learned; only the plan is
> retired.** The v2.2 checkpoints stay useful as an extreme point on the
> spikes-per-active-neuron axis.

**Where this picks up.** v1 (documents 2–3) produced a clean sparsity gradient and a
clear headline result — but the diagnosis showed it could not test H1's *mechanism*,
because the perturbation-immune count and identity channels stayed wide open, and
sparsity-by-silencing actually *strengthened* one of them. v2 is the attempt to close
those channels by construction, so that a second negative result would refute H1 on
its own terms. The conceptual argument is in [sparse_network.md](sparse_network.md)
§5–6; the empirical story is below.

---

## 1. Why there is a v2

v1 answered the question as posed and got a clear negative: in all 8 perturbation ×
arm combinations, sparser networks were **more** perturbation-robust, not less
(Spearman ρ = +0.67…+0.96). But the diagnosis showed v1 could not have tested H1 in
the first place. H1's mechanism assumes that when a layer is sparse each neuron fires
**0–2 spikes**, so *count* carries no resolution and the network is forced onto a
*latency* code. v1 never reached that regime — even its sparsest models fired **3.1
(no-delay) / 4.1 (delay) spikes per _active_ neuron**; the headline
`spikes_per_neuron` of 1.4 / 2.1 was diluted by silent neurons. Worse, v1's one-sided
hinge had **zero gradient at or below its target**, so a neuron could dodge the
penalty entirely by falling silent on a sample. Sparsity therefore arrived partly as
**stimulus selectivity** (silent fraction .25–.33 → .48–.54), which *strengthens* a
population-identity code that every perturbation in the study leaves perfectly
intact.

**v2's goal:** close that escape hatch, making H1's premise true by construction, so
that a second negative result would refute H1 *on its own terms* rather than refuting
a regime H1 never claimed.

**The single acceptance test, unchanged across all iterations:** a linear decode of
the label from the per-neuron **count** vector (and from the binarised **identity**
vector), fit on train and scored on test. Both are exactly invariant to every
rate-preserving perturbation, so the count decode is the ceiling on what a
perturbation-immune readout could achieve. It must fall to near chance (.05).
**v1 baseline to beat: count .687–.765, identity .339–.513.**

---

## 2. Mechanism iterations — what was tried and what each probe showed

All probes: seed 42, 400 epochs, `QUICK_TEST = True`, artifacts `_probe`-suffixed.
Measured cost ≈ **35–38 min/model (no-delay), ~45 min/model (delay)**.

### v2 — two-sided band `[lo, hi]`, per (sample, neuron) ❌ superseded

```python
penalty = (relu(count - band_hi) + under_weight * relu(band_lo - count)).mean()
```

Replaced v1's hinge-on-batch-mean. Two intended fixes: a **lower arm** so silence
costs something, and **per-sample charging** so bursty selective neurons stop getting
a free pass (verified numerically — on a neuron firing 10 spikes in 1 sample of 128,
v1 charges 0.0000, v2 charges 1.0547).

**Probe (2026-07-28):** bands `[1,2] / [3,5] / [8,12]`, strength 3, `UNDER_WEIGHT` 1.

| arm | band | clean_acc | sp/ACTIVE | in band? | silent | decode-COUNT | decode-IDENT |
|---|---|---|---|---|---|---|---|
| no-delay | [1,2] | .5418 | 3.08 | ✗ | 39.7% | .707 | .554 |
| no-delay | [3,5] | .5631 | 4.29 | ✓ | 11.6% | .725 | .443 |
| no-delay | [8,12] | .5778 | 9.78 | ✓ | 1.6% | .754 | **.218** |
| delay | [1,2] | .7963 | 4.96 | ✗ | 32.7% | .693 | .453 |
| delay | [3,5] | .8210 | 5.20 | ✗ | 16.4% | .753 | .511 |
| delay | [8,12] | .8617 | 9.92 | ✓ | 1.6% | .765 | **.239** |

**What worked:** no collapse anywhere (peak `train_silent` .55–.59 at epoch 1, then
recovery — the lower arm genuinely replaced the 15-epoch warm-up guard the no-delay
arm needed under v1), and accuracy held throughout. The tight band is learnable.

**What failed:** the band bound only at the dense end, and `silent_fraction` tracked
`band_lo` monotonically (1 → 33–40%, 3 → 12–16%, 8 → 1.6%) — a tighter band drives
neurons below threshold, where the surrogate gradient vanishes and the lower arm can
no longer reach them. Fatally, **the count channel never closed**: .69–.77 at every
band, statistically identical to v1.

**Why no calibration could rescue it.** Clipping measured counts to progressively
tighter sets showed the requirement is effectively binary:

| count representation | at-target | std | decode |
|---|---|---|---|
| as measured | 59.8% | 1.50 | .661 |
| clip to {0,1,2} | 59.8% | 0.63 | .605 |
| clip to {0,1} | 78.9% | 0.41 | .528 |
| **pin to exactly 1** | **100%** | **0.00** | **.050 = chance** |

Going 60% → 79% at-target moved the decode only .66 → .53. **A band of any nonzero
width cannot close the count channel** — the design premise ("pin the count in a
narrow band and count carries ~1 bit") is true *per neuron*, but the readout uses all
128 jointly, so the population still carries ~128 bits.

### v2.1 — L1 exact-count `|count − k|`, with k annealed 8 → target ❌ superseded

Swept the exact count instead of a band; `PENALTY_STRENGTH` 3 → 10; `UNDER_WEIGHT`
removed (`|count − k|` is symmetric); `k` annealed over 150 epochs to keep neurons out
of the dead zone.

**Probe (2026-07-28/29):** k ∈ {1, 3}.

| arm | k | clean_acc | sp/neuron | count_std | at_target | silent | decode-COUNT | decode-IDENT |
|---|---|---|---|---|---|---|---|---|
| no-delay | 1 | .5524 | 1.19 | 1.50 | 59.8% | 21.1% | .664 | .527 |
| no-delay | 3 | .5037 | 3.13 | 1.74 | 39.8% | 2.4% | .659 | .304 |
| delay | 1 | .7715 | 1.54 | 2.64 | 39.4% | 31.6% | .719 | .612 |
| delay | 3 | .7836 | 3.26 | 2.46 | 30.9% | 5.5% | .720 | .462 |

**What worked — and v2.2 still depends on both:** the constraint **binds** (sp/neuron
landed at 1.19–3.26 against targets 1 and 3, versus 4.96 for v2's band), and **k = 1
is learnable**, at .552 / .772 clean accuracy.

**What failed:** `count_std` plateaued at 1.5–2.6 (trajectories converged — more
epochs would not have helped), and the count decode barely moved: .659–.720. The
escape route this time was a **heavy tail** — 3.6–9.2% of pairs above k+2. L1 pins the
median, not the distribution.

**The pattern, by now unmistakable:** each soft penalty leaves some slack, and the
network fills it. It has strong incentive to: the count channel is worth more to it
(.66) than its own accuracy (.55).

### v2.2 — hard top-k truncation + one-sided floor penalty ⏳ current, partly probed

Stopped *penalising* count variation and made it **impossible**:

```python
spike_index = flat.cumsum(dim=-1)          # n-th spike -> cumulative sum n
keep = (spike_index <= k).to(flat.dtype)   # boolean -> constant, no gradient
hidden1 = flat * keep                      # only the first k spikes leave the layer
penalty = relu(k - count).mean()           # floor only; truncation caps the top
```

Truncation is part of the **layer definition**, not a regulariser. The anneal was
switched off (`ANNEAL_EPOCHS = 0`) — the dead-zone hazard it existed for is a
consequence of pushing firing *down*, which v2.2 never does. Raw firing above k is
unconstrained and logged separately as `raw_spikes_per_neuron`.

Scripts verified 47/47 (truncation keeps the *earliest* spikes, stays binary,
gradients flow through survivors; floor penalty one-sided; `under_target_fraction`
detects a synthetic leak exactly).

**Probe (2026-07-29), NO-DELAY ARM ONLY:** k ∈ {1, 3}, strength 10.

| k | clean_acc | sp/neuron | raw_rate | under_k | count_std | at_target | silent |
|---|---|---|---|---|---|---|---|
| 1 | **.2371** | 0.83 | **12.5** | **16.8%** | 0.374 | 83.2% | 16.8% |
| 3 | **.2792** | 2.76 | **21.9** | **10.9%** | 0.740 | 89.1% | 5.1% |

**The manipulation moved further than any previous version, and the channel ranking
inverted for the first time:**

| version | decode-COUNT | decode-IDENTITY | decode-TIMING |
|---|---|---|---|
| v1 (hinge) | .687–.749 | .453–.513 | — |
| v2 (band) | .707–.754 | .218–.554 | — |
| v2.1 (L1) | .659–.664 | .443–.554 | — |
| **v2.2 k=1** | **.238** | **.237** | **.275** |
| **v2.2 k=3** | **.279** | **.145** | **.345** |

Count decode fell ~3× from v1, and **timing now decodes better than count** — what
v2 set out to achieve. (At k=1 counts are binary, so count and identity decode the
same vector; their agreement is a consistency check, not two findings.)

**Three problems, all recorded rather than fixed:**

1. **The leak reopened during training.** `under_k` hit **exactly 0.000** — epochs
   1–20 (k=1), epoch 6 (k=3) — then climbed back to .165 / .104. The floor penalty at
   strength 10 *can* close the channel; the task loss buys the silence back over the
   next ~80 epochs. A dynamic failure, not a static calibration miss.
2. **Accuracy roughly halved**, .552 → .237 (k=1). Decomposition on the checkpoints:

   | weights | eval untruncated | eval at k=1 |
   |---|---|---|
   | v2.1 (trained untruncated) | **.552** | .156 |
   | v2.2 (trained truncated) | .061 | **.239** |

   Truncating a network not trained for it costs .552 → .156, so most of that .552 was
   riding on the count channel. Training under truncation recovers .156 → .239. The
   v2.2 network is **near chance (.061) without truncation** — it has specialised to
   the regime, not had a filter bolted on. So .237 is not a broken model.
3. **Raw firing blew up** to 12.5 / 21.9 against a natural ~6.3 — the floor penalty
   raised global drive to fix under-firing pairs, overshooting pairs that were already
   fine. This compresses the surviving spike times:

   | model | mean spike time | std across samples | % in first 10 ms |
   |---|---|---|---|
   | v2.2 k=1 | 9.82 | 9.86 | **63.3%** |
   | v2.2 k=3 | 8.85 | 5.17 | **71.7%** |
   | v2.1 k=1 (untruncated) | 15.44 | 7.64 | 19.9% |

   Nearly two-thirds of surviving spikes land in the first 10 ms of a 200 ms window —
   exactly the crowding the `raw_rate` diagnostic was added to catch.

**Under-training caveat:** both runs were still improving at epoch 400 (k=1 peaked
.268 at epoch 271; k=3 peaked .314 at epoch 393, still climbing). The .237 / .279
figures are lower bounds on what 1250 epochs would give.

---

## 3. The open question v2.2 does not settle

The .237 admits two readings, and the no-delay probe cannot separate them:

- **Substantive:** timing on SHD really does carry only ~.24 once counts are removed
  — a real result about H1's premise.
- **Artifact:** the raw-rate blow-up crowded spikes into an early window and
  artificially capped what timing could carry.

One hint against a purely substantive reading: a linear decoder on mean spike times
scores **.275**, *above* the network's own **.237** — the network is not extracting
everything present. That is suggestive, not decisive.

**The with-delay arm is the more informative one and has not been run.** It carries
~.77 clean accuracy versus this arm's ~.55, so the same absolute drop would leave far
more signal to interpret.

---

## 4. Current configuration (v2.2, as it stands)

**Scripts:** [sn_train_withDelay_v2.py](../../exp_sparse_network/sn_train_withDelay_v2.py)
· [sn_train_noDelay_v2.py](../../exp_sparse_network/sn_train_noDelay_v2.py)
(both now hold v2.2; the v2 and v2.1 mechanisms are gone from the code and survive
only in this log and in their `_probe` artifacts.)

```python
TARGET_COUNTS    = [1.0, 2.0, 3.0, 5.0, 8.0]   # the swept axis = truncation level
PENALTY_STRENGTH = 10.0                        # floor penalty relu(k - count)
ANNEAL_EPOCHS    = 0                           # anneal retained but off
WARMUP_EPOCHS    = 0                           # both arms
SEEDS            = [42, 43, 44]
EPOCHS           = 1250
MAX_UNDER_TARGET = 0.02                        # acceptance threshold
```

**Unchanged from v1 throughout, for comparability:** architecture (128–128, SRMALPHA),
dataset and the fixed 60/15/15 splits, `NumSpikes` loss, Nadam, LR 0.1, MultiStepLR
milestone 300, batch 128, `EPOCHS = 1250`, early-stop patience 300, seeds, and every
eval-side perturbation grid.

**Artifacts by version** — all three coexist without collision:

| Version tag | Checkpoints | Summary |
|---|---|---|
| `v2` (band) | `sn_data/sparse_whole_{arm}_v2_band{lo}-{hi}_str3_seed42_probe.pt` | `sn_log/sparse_whole_{arm}_v2_train_summary_probe.json` |
| `v2_1` (L1) | `..._v2_1_k{k}_str10_seed42_probe.pt` | `..._v2_1_train_summary_probe.json` |
| `v2_2` (truncation) | `..._v2_2_k{k}_str10_seed42_probe.pt` | `..._v2_2_train_summary_probe.json` |

> ⚠️ **Protocol rule (do not break):** sparsity is applied **during training** on
> clean data; every perturbation is applied **only at evaluation**.

> ⚠️ **v2.2 checkpoints require truncation at eval.** Truncation defines the layer.
> The 8 eval scripts carry their own copy of the network class and **do not have it**
> — evaluating a v2.2 checkpoint with them today measures a *different network*. They
> currently point at v1 summaries, so there is no live hazard, but this must be fixed
> before any v2.2 evaluation.

---

## 5. Remaining steps

### Step 1 — probe (partly done)
- ✅ v2 band, both arms · ✅ v2.1 exact-count, both arms · ✅ v2.2 truncation, no-delay
- ⬜ **v2.2 truncation, with-delay** (~1.5 h for 2 models)

### Step 2 — full sweep
Not started in any version. 15 models per arm at 1250 ep: **~27 h no-delay, ~35 h
delay, ~62 h total** (scaled from measured probe times; v1's 1250-ep logs suggest up
to ~94 h, so budget in that range).

### Step 3 — eval sweeps
Four perturbations × two arms. **Three edits needed in each of the 8 scripts:**
1. `TRAIN_SUMMARY_FILE` → the v2.2 summary.
2. Add `"spikes_per_active_neuron"` (and `"raw_spikes_per_neuron"`) to
   `SUMMARY_PASSTHROUGH_FIELDS`.
3. **Add `truncate_to_k_spikes` to each network class**, with `truncate_k` read from
   the summary's `target_count` per checkpoint — see the warning in §4.

Plus two grid fixes carried from v1's caveats: extend deletion to
`p_d ∈ {0.9, 0.95}` (v1's equal-damage numbers were censored at the 0.8 edge), and
refine `shd` at low f (`{0.02, 0.05, 0.1, 0.15, 0.2}` — all 15 v1 no-delay models lost
everything by f = 0.2). Cost ~1 h for all eight.

### Step 4 — analysis
`temporal_score = 1 − (acc(max) − 0.05) / (acc(0) − 0.05)`, as in v1.
**4a. Manipulation check first** — count/identity decode per checkpoint against the
v1 baseline; if it has not fallen, the headline is uninterpretable.
**4b.** Headline: `temporal_score` vs measured firing, per arm, Spearman ρ.
**4c.** Deletion control + partial correlation + equal-damage comparison (v1's
decisive analysis — a raw correlation that vanishes here is general robustness).
**4d.** v1-vs-v2 at matched firing rate.
**4e.** Three plots: headline, curve families, guard.

---

## 6. How to read the outcome

| Outcome | Meaning |
|---|---|
| Manipulation passes **and** ρ < 0 | **H1 supported** under its own premise; v1's positive ρ was an artifact of the identity escape hatch. |
| Manipulation passes **and** ρ > 0 | **H1 refuted on its own terms** — stronger than v1, because the immune channels were closed by construction. |
| Manipulation passes **and** ρ ≈ 0 | Timing dependence invariant to sparsity once the immune channels are closed. |
| `clean_acc` at/near chance under the manipulation | H1 **untestable in this architecture** — report as such; the synthetic ISI task is the venue where the latency regime can be imposed by design. |
| Manipulation fails | Uninterpretable — the state v2 and v2.1 both ended in. |

**Either direction is a valid finding.** The milestone is done when the headline plot
exists and the manipulation check is documented alongside it.

---

## 7. Watch out for (accumulated lessons)

1. **Soft penalties lose the arms race.** Three designs, three escape routes: silence
   (v1), within-band variation (v2), heavy tail (v2.1). The count channel is worth
   more to the network than its own accuracy, so it exploits any slack left.
2. **The count requirement is effectively binary.** 79% at-target still decodes at
   .53; only ~100% collapses to chance. There is no gradual approach.
3. **Population capacity, not per-neuron resolution, is what matters.** ~1 bit per
   neuron × 128 neurons is ample for 20 classes.
4. **Reduced-epoch numbers under-estimate the full run** — and v2.2's probes were
   still improving at epoch 400. Never lock a grid from a probe.
5. **`spikes_per_neuron` alone lies** — diluted by silent neurons. Always report
   `spikes_per_active_neuron`, and for v2.2 `raw_spikes_per_neuron` too.
6. **Dead neurons are unrecoverable** — pushing firing *down* past threshold kills the
   surrogate gradient. Pushing *up* is the safe direction.
7. **A closed channel can reopen during training.** v2.2 reached `under_k` = 0 and
   then lost it. Check the trajectory, not just the final number.
8. **Never analyse against the band / strength / k** — only measured statistics.
9. **A chance-level model is broken, not "maximally temporal."** Drop it and say so.
10. **Keep the two arms' hyper-parameters matched.**

---

## 8. Progress checklist

- [x] v1 result recorded and diagnosed
  ([sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md))
- [x] **v2 (band)** designed, verified 21/21, probed both arms — **failed**: count
  decode .69–.77, unchanged from v1; band bound only at the dense end
- [x] Counterfactual clip test — proved no band width can close the count channel
- [x] **v2.1 (L1 exact-count)** implemented, verified 40/40, probed both arms —
  **failed**: count decode .66–.72; L1 pins the median, leaves a heavy tail
- [x] Established that the constraint can bind and that **k = 1 is learnable**
  (.552 / .772) — the foundation v2.2 rests on
- [x] **v2.2 (top-k truncation)** implemented, verified 47/47
- [x] v2.2 probe — **no-delay arm** (2026-07-29): count decode .238/.279, timing now
  the dominant channel; accuracy .237/.279; `under_k` 16.8%/10.9%; raw rate blown up
- [ ] **v2.2 probe — with-delay arm** (not run; the more informative arm)
- [ ] Resolve the residual leak (`under_k` → 0 and *stays* there) and the raw-rate
  blow-up, or record that they cannot be resolved
- [ ] Step 2 — full sweep, 15 models × 2 arms at 1250 ep
- [ ] Step 3 — patch the 8 eval scripts (**incl. truncation**), extend the deletion
  grid, refine the low-f `shd` grid; run all four sweeps × both arms
- [ ] Step 4a — manipulation check on the full-run checkpoints
- [ ] Step 4b–4e — headline ρ, deletion control, v1-vs-v2 at matched firing, plots
- [ ] v2 verdict recorded against §6, plus the delay-vs-no-delay comparison

---

## 9. Cost record

| Stage | Measured / estimated |
|---|---|
| Probe, per model @ 400 ep | **~35–38 min** (no-delay), **~45 min** (delay) |
| v2 probe (3 bands × 2 arms) | ~4 h — spent |
| v2.1 probe (2 k × 2 arms) | ~2.7 h — spent |
| v2.2 probe (2 k × 1 arm) | ~1.3 h — spent; ~1.5 h outstanding for the delay arm |
| Full sweep (15 models × 2 arms @ 1250 ep) | ~62 h scaled from probes; up to ~94 h by v1's logs |
| Eval sweeps (4 perturbations × 2 arms) | ~1 h |

**~8 h of probing has so far prevented ~62–94 h being spent on three mechanisms that
would each have failed the manipulation check.** That is the case for never skipping
Step 1.
