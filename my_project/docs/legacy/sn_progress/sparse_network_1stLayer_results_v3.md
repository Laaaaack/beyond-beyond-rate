# v3 1st-layer results — four perturbations, two arms, 48 models

**Status (2026-08-01): COMPLETE.** All four eval-only perturbations swept over both
v3 factorial grids — 24 checkpoints per arm (4 `k` × 2 floor × 3 seeds), 8 sweeps in
total. This document reports what those sweeps say. The design, the calibration and
the Phase 0/Phase 1 execution log live in
[sparse_network_test_progress_v3.md](sparse_network_test_progress_v3.md); this is the
v3 analogue of document 3, which reported v1's results.

**The headline: H1′ is refuted on the two spike-placement probes, in both arms
independently.** Holding selectivity `s` fixed by construction and controlling for
general robustness, the coefficient on `log(a)` is **positive** — the direction
opposite to H1′'s prediction — and clears α in both arms, on relocation *and* on
jitter measured separately. Two further findings matter as much:

- **Shift is not a timing probe in the delay arm.** Its damage is almost entirely
  predicted by the deletion control (β\* = +0.801, p < .0001, R² = .65) and it carries
  no signal on either sparsity axis. Reporting it beside relocation and jitter as a
  third timing measure would be wrong.
- **The no-delay arm's deletion control is a proxy for silence** — ρ(`s`, control)
  = **−0.938**, and `s` alone explains R² = .87 of it. Nothing in that arm that is
  attributed to `s` can be separated from general robustness.

**Accuracy figures:**
[result_visualization/1stLayer/result_visualization.ipynb](../../exp_sparse_network/result_visualization/1stLayer/result_visualization.ipynb).

---

## 1. What was measured

Eight sweeps, one per (arm × perturbation), all eval-only against frozen checkpoints:

| perturbation | what it destroys | rate-preserving? | grid | script |
|---|---|---|---|---|
| **relocation** | spike placement anywhere in the support; per-neuron count held **exactly** | yes | `f` = 0 … 1 | `shd/shd_evalOnly_*_v3.py` |
| **jitter** | spike placement, locally (`N(0, σ)` per spike) | yes | `σ` = 0 … 25 ms | `jitter/jitter_evalOnly_*_v3.py` |
| **shift** | cross-neuron alignment (rigid per-neuron translation) | yes | `σ` = 0 … 25 ms | `shift/shift_evalOnly_*_v3.py` |
| **deletion** | the spikes themselves — the **general-robustness control** | **no** | `p_d` = 0 … 0.8 | `deletion/deletion_evalOnly_*_v3.py` |

Relocation and jitter draw destinations from the layer's measured temporal support
`[0, 88)`; shift and deletion correctly do not (document 5 §8 rule 2).

Each sweep runs 3 repeats per grid point, and each `(k, floor)` cell averages its 3
seeds. Results files carry both: `per_setup` (seed-averaged, the headline) and
`per_checkpoint` (the raw per-seed rows, which every number below is computed from).

**The score.** For each checkpoint, `temporal_score = 1 − retention`, where
`retention` is the chance-corrected fraction of clean accuracy surviving the
strongest setting of that perturbation. Higher = more damage. It is computed per
checkpoint against that checkpoint's *own* clean accuracy, then averaged — clean
accuracy varies across seeds and the normalisation exists to remove that.

**Cross-check.** The relocation sweep's `f = 1` endpoint reproduces
`phase1_measure.py`'s independently-measured `usage` to a mean |Δ| of **.006**
(delay) and **.012** (no-delay); the deletion `p_d = 0.8` endpoint reproduces its
`control_deletion` to **.008** / **.014**. Two separate implementations, same answer.

---

## 2. The grid as trained, and whether it is interpretable

| | delay | no-delay |
|---|---|---|
| n | 24 | 24 |
| `a` (spikes/active neuron) | 2.43–5.29 (**2.18×**) | 1.79–4.06 (**2.26×**) |
| `s` (silent fraction) | 5.2% – 66.3% | 5.6% – 76.3% |
| clean accuracy | .707 – .859 | .464 – .552 |
| **ρ(`a`, `s`) — the design check** | **−0.087** (p = .69) | **−0.342** (p = .10) |
| ρ(`s`, deletion control) | **−0.046** (p = .83) | **−0.938** (p = 1.3e−11) |

Both arms pass the design acceptance check (|ρ| ≤ 0.5) against v1's **−0.455** and
**−0.943**. The factorial did what it was built to do, in both arms.

**But the two arms are not equally interpretable, and the last row is why.** The
factorial governs the two *independent* variables; it says nothing about the
covariate. In the delay arm `s` and the deletion control are independent (−0.046), so
a coefficient on `s` means what it says. In the no-delay arm they are nearly the same
variable (−0.938) — regressing `s` and the control together there gives VIF ≈ 3.2 and
neither coefficient is separately identified. Fitting the control *alone* on the two
axes makes the point:

```
no-delay:   deletion_score ~ log(a) + s      R2 = .868, n = 24, df = 21
              log_a  +0.0218  p = .175   beta* = +0.113
              s      -0.1550  p < 1e-6   beta* = -0.909      <-- the control IS s

delay:      deletion_score ~ log(a) + s      R2 = .015, n = 24, df = 21
              log_a  +0.0208  p = .578   beta* = +0.122
              s      +0.0051  p = .901   beta* = +0.027      <-- the control is neither
```

This is Phase 0's finding about the no-delay arm reappearing under the factorial, and
it is the reason the delay arm remains the primary venue for interpretation even
though the no-delay arm carries the larger coefficients.

---

## 3. Clean accuracy

Notebook §2. Nothing here is at chance, and nothing is collapsing:

| | `k`=1 | `k`=2 | `k`=4 | `k`=8 |
|---|---|---|---|---|
| delay, floor OFF | .775 | .788 | .806 | .832 |
| delay, floor ON | .757 | .794 | .824 | .844 |
| no-delay, floor OFF | .478 | .499 | .489 | .534 |
| no-delay, floor ON | .519 | .532 | .538 | .542 |

Accuracy rises gently with `k` (a looser ceiling costs less) and the floor is
approximately free — slightly *positive* in the no-delay arm (+.041 at `k`=1). Both
arms sit inside their v1 ranges (.780–.883 and .490–.590). §5's "clean accuracy at
chance in the floor-on column" failure row is not in play.

---

## 4. The accuracy curves

Notebook §1: rows = arms, columns = perturbations, 8 curves per panel, band = across-seed sd.

What the panels show before any scoring:

- **The delay arm's curves are tightly bunched.** At the strongest setting of each
  perturbation the floor-OFF and floor-ON columns end within ±.02 of each other at
  matched `k`, with inconsistent sign. Its clean accuracies already differ by .087
  across cells, so much of the visible spread is that starting offset rather than
  differential damage — which is exactly why the normalised score exists.
- **The no-delay arm separates, and the separation grows with `k`.** Floor-OFF (high
  `s`, dashed) ends *above* floor-ON at the strongest setting, by a margin that widens
  along the row axis — at `k` = 1 the two columns finish level (−.010 to +.035 across
  the four perturbations), by `k` = 8 the gap is +.032 to +.077. Since floor-ON
  *starts* from the higher clean accuracy, that end-state gap is a real difference in
  damage rather than an inherited offset. It is also — see §2 — inseparable from the
  general-robustness effect in this arm.
- **Deletion at `p_d` = 0.8 lands everything at .23–.32**, from starting points
  spanning .478–.844. Nothing survives losing 80% of its hidden spikes.

---

## 5. The regressions

Per arm, per perturbation: `score ~ β_a·log(a) + β_s·s + β_d·(deletion control)`.
H1′ predicts **β_a < 0** (fewer spikes per active neuron ⇒ more timing reliance).
H2 predicts β_s carries the effect and β_a is null.

### Delay arm (n = 24, df = 20) — the interpretable one

```
relocation   R2 = .408      jitter       R2 = .371      shift        R2 = .651
  log_a   +0.0864 p=.031      log_a   +0.0923 p=.032      log_a   -0.0128 p=.578
          beta* = +0.404              beta* = +0.411              beta* = -0.075
  s       -0.0743 p=.078      s       -0.0881 p=.055      s       -0.0289 p=.251
          beta* = -0.320              beta* = -0.361              beta* = -0.157
  control +0.4075 p=.076      control +0.2703 p=.266      control +0.7990 p<.0001
          beta* = +0.324              beta* = +0.205              beta* = +0.801
```

### No-delay arm (n = 24, df = 20) — β_a clean, β_s not identified

```
relocation   R2 = .869      jitter       R2 = .881      shift        R2 = .847
  log_a   +0.1943 p<.0001     log_a   +0.1765 p=.0001     log_a   +0.0813 p=.012
          beta* = +0.471              beta* = +0.411              beta* = +0.257
  s       -0.0607 p=.453      s       -0.0933 p=.248      s       -0.1041 p=.128
  control +1.1723 p=.023      control +1.1667 p=.022      control +0.7756 p=.062
```

### Reading them

1. **β_a is positive and significant on both placement probes, in both arms.** Four
   independent tests (2 perturbations × 2 arms): p = .031, .032, <.0001, .0001. H1′
   predicts the opposite sign. This is document 5 §5's **row 2** — "H1 refuted on its
   own terms, the strongest available negative, since `s` was held fixed by
   construction and the prediction still ran backwards".
2. **Relocation and jitter agree closely** — β\* = +0.404 vs +0.411 (delay), +0.471 vs
   +0.411 (no-delay), and their scores correlate at ρ = **+0.933** / **+0.991**. They
   are two different ways of destroying spike placement and they return the same
   answer, which is the closest thing to a replication available inside one grid.
3. **Shift behaves differently, and in the delay arm it is not a timing probe at
   all.** β_a is null there (−0.013, p = .58) and the deletion control takes β\* =
   +0.801 — shift damage is general-robustness damage. In the no-delay arm shift does
   carry a β_a (+0.081, p = .012) but at a third the size of the placement probes.
4. **β_s runs in H2's direction everywhere but never clears α**, and in the no-delay
   arm it is not separable from the control anyway. See §7 on how fragile it is.

---

## 6. The four perturbations do not measure one thing

Spearman between the four scores, across the 24 checkpoints of each arm:

| pair | delay | no-delay |
|---|---|---|
| relocation ↔ jitter | **+0.933** | +0.991 |
| shift ↔ deletion | **+0.833** | +0.891 |
| relocation ↔ shift | +0.453 | +0.901 |
| relocation ↔ deletion | +0.307 (n.s.) | +0.860 |
| jitter ↔ shift | +0.311 (n.s.) | +0.917 |
| jitter ↔ deletion | +0.232 (n.s.) | +0.880 |

**The delay arm splits cleanly into two factors**: {relocation, jitter} and
{shift, deletion}, with weak, non-significant cross-correlations between the groups.
The two probes that scramble *where spikes sit* cluster together; shift clusters with
the pure rate insult.

**In the no-delay arm all four collapse onto one factor** (+0.86 … +0.99). A single
axis of general fragility dominates everything there — the same picture v1 showed,
and the reason Phase 0 judged that arm unable to separate a rate insult from a timing
insult.

Why shift sides with deletion in the delay arm is **not resolved here**. A rigid
translation preserves per-neuron count exactly and is not clipped at σ = 25 (the
support ends at bin 87 of 200), so a simple "shift secretly deletes spikes" account
does not fit. The honest statement is the measurement: in this arm shift adds nothing
beyond the general-robustness control, so it should not be quoted as timing evidence.

---

## 7. What is fragile, and what is not

The same coefficient measured twice — once from `phase1_measure.py`'s endpoint, once
from this relocation sweep's `f = 1` endpoint, on the same checkpoints, differing only
in the random draws of the perturbation:

| delay arm term | via `phase1_measure` | via the `f=1` sweep |
|---|---|---|
| β_a on `log(a)` | +0.0850, **p = .033** | +0.0864, **p = .031** |
| β_s on `s` | −0.0868, **p = .045** | −0.0743, **p = .078** |
| β_d on the control | +0.4332, p = .056 | +0.4075, p = .076 |

**β_a is stable across the re-measurement; β_s is not.** The H1′ refutation survives
the measurement noise at n = 24. The H2 signal in the delay arm sits on the α = .05
boundary and moves across it depending on which of two equally valid measurements of
the same quantity is used — so it should be reported as *directional and unresolved*,
not as a confirmed effect. Three repeats per grid point is the binding constraint on
that term; raising `NUM_REPEATS` would tighten it more cheaply than more models would.

---

## 8. v1 versus v3, measured identically

ρ(temporal_score, `spikes_per_neuron`) — the composite rate variable v1 analysed —
computed the same way on both generations:

| perturbation | v1 delay (n=12) | **v3 delay (n=24)** | v1 no-delay (n=15) | **v3 no-delay (n=24)** |
|---|---|---|---|---|
| relocation | +0.853 | **+0.566** | +0.939 | **+0.906** |
| jitter | +0.811 | **+0.574** | +0.921 | **+0.903** |
| shift | +0.881 | **+0.325** | +0.957 | **+0.870** |
| deletion (control) | +0.713 | **+0.174** | +0.936 | **+0.899** |

In v1, *every* perturbation correlated with the rate variable at +0.71 … +0.96,
including the one that is not a timing probe at all. **In the v3 delay arm that
structure is gone** — the control falls to +0.174 and shift to +0.325, while the two
placement probes retain a real but much smaller correlation. That is the confound
being removed, visible without any regression.

**In the v3 no-delay arm it is not gone** (+0.87 … +0.91 across the board), because
breaking ρ(`a`, `s`) there did not break ρ(`s`, control). Decorrelating the design's
own axes is necessary and not sufficient.

---

## 9. Verdict

Against document 5 §5:

- **Row 2 — "β_a > 0, significant: H1 refuted on its own terms"** — reached on
  relocation and jitter, in both arms, four independent tests. `s` was held fixed by
  construction and the prediction still ran backwards.
- **Row 3 (H2) is not reached.** β_s runs in H2's direction throughout but never
  clears α; in the delay arm it straddles the boundary under re-measurement, and in
  the no-delay arm it is confounded with general robustness at ρ = −0.938.
- No cell is unlearnable, and the design check passes in both arms, so neither of
  §5's failure rows applies.

**Limitations, stated rather than hidden.**

1. **The `a` axis is narrow** — 2.18× / 2.26× against v1's observational 4.1× / 3.8×,
   because the ceiling stops binding at `k` = 8. β_a is significant across that span,
   but the span is the main thing a follow-up should widen (a no-ceiling or `k` = 16
   row, 4–6 models per arm).
2. **The no-delay arm's readout leaves +.300 of accuracy unextracted** on every
   checkpoint (delay: +.026), so its scores are partly a statement about `fc2`/`fc3`.
   It carries the larger coefficients; the delay arm carries the attributable ones.
3. **β_s is unresolved**, per §7.
4. **Three repeats per grid point** is what makes §7's term unstable; it is the
   cheapest thing to increase.
5. **Shift should not be quoted as a timing result in the delay arm** (§6).

---

## 10. Reproducing this

```
# training (already done) -- 24 models per arm
python sn_train_withDelay_v3.py          # SEEDS = [42, 43, 44]
python sn_train_noDelay_v3.py

# the four sweeps, per arm -- writes {folder}/log/sparse_whole_{arm}_v3_*_eval.json
python shd/shd_evalOnly_{withDelay,noDelay}_v3.py
python jitter/jitter_evalOnly_{withDelay,noDelay}_v3.py
python shift/shift_evalOnly_{withDelay,noDelay}_v3.py
python deletion/deletion_evalOnly_{withDelay,noDelay}_v3.py

# the endpoint measures + regressions used in section 5
python v3_analysis/phase1_measure.py
python v3_analysis/phase1_regress.py

# accuracy figures
result_visualization/1stLayer/result_visualization.ipynb
```

Every sweep is eval-only: checkpoints are loaded frozen, no gradients are taken, and
the perturbation is applied to the 1st hidden layer at evaluation time only.
