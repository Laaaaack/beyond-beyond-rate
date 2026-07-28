# First Milestone — Results: 1st-layer perturbation sweeps

**Status:** Steps 3–4 **COMPLETE** (2026-07-28). Four eval-only perturbations ×
two arms × all checkpoints, 3 repeats each. **Verdict: H1 is not supported.** In all
8 perturbation × arm combinations the sparsity → temporal-processing trend is
statistically significant and runs **opposite** to the prediction.
**Owner:** _(you)_
**Design & rationale:** [sparse_network.md](sparse_network.md) ·
**Training/Step-1 history:** [sparse_network_test_progress.md](sparse_network_test_progress.md)

---

## 1. TL;DR

- **H1 predicted:** sparser 1st hidden layer → more dependent on spike *timing* →
  bigger accuracy collapse under timing perturbation (negative Spearman ρ between
  measured firing rate and `temporal_score`).
- **Measured:** ρ is **positive and significant in all 8 combinations**
  (+0.67 … +0.96). Sparser networks are *more* robust to every perturbation tried.
- **The deletion control reframes it.** Deletion is a *rate* perturbation, not a
  timing one, and it shows the **same** trend (ρ = +0.936 / +0.713). So a large part
  of the effect is **general perturbation robustness**, not anything about timing.
- **After controlling for general robustness** (partial correlation on the deletion
  score), the picture splits by arm:
  - **No-delay arm:** a genuine, significant *reduction* in timing dependence with
    sparsity survives every control. H1 is **actively contradicted** here.
  - **Delay arm:** under complete timing destruction (`shd` f=1) the timing
    dependence is **flat** across sparsity (partial ρ = +0.364, p = 0.27; the
    severity-matched contrast is flat too). Sparsity does not change *how much*
    timing matters — only how *precisely* it matters.
- **Mechanism:** the hinge lowers firing mostly by **thinning** active neurons (~75 %
  of the reduction) and partly by making them **stimulus-selective** (~25 %) — silent
  on more *samples*, not dead. The selectivity strengthens a population-identity code
  that every perturbation here leaves intact, and the 0–2-spike regime H1 needs was
  never reached (3.1–4.1 spikes per *active* neuron even at the sparsest). This is
  pitfall 3 in [sparse_network.md](sparse_network.md#pitfalls-and-confounds).

This is a clean negative result with a diagnosed mechanism, not a pipeline failure.

---

## 2. What was run

All four sweeps are eval-only on the frozen Step-1 checkpoints (15 no-delay, 12
delay), `QUICK_TEST = False`, `NUM_REPEATS = 3`, 1st hidden layer, inside
`torch.no_grad()`. The protocol rule held: sparsity during training, perturbation at
evaluation only.

| Perturbation | Script | Sweep | Preserves rate? | What it destroys |
|---|---|---|---|---|
| **jitter** | [jitter/](../../exp_sparse_network/jitter/) | σ ∈ {0,1,3,5,10,17,25} ms | yes | *local* spike timing (each spike moves independently) |
| **shift** | [shift/](../../exp_sparse_network/shift/) | σ ∈ {0,1,3,5,10,17,25} ms | yes | cross-neuron alignment & onset (whole train moves together; within-train pattern intact) |
| **shd** (relocation) | [shd/](../../exp_sparse_network/shd/) | f ∈ {0,0.2,…,1.0} | **exactly** | *all* timing at f=1 — the pure count-only limit |
| **deletion** | [deletion/](../../exp_sparse_network/deletion/) | p_d ∈ {0,0.2,…,0.8} | **no** | spikes themselves — the **rate / general-robustness control** |

`shd` at f = 1 is the most theoretically clean timing probe in the set: every spike
is relocated to a random bin, per-neuron count is exactly preserved, and no σ has to
be chosen. `acc(f=1)` therefore *is* the accuracy the network achieves from the
count + identity code alone.

Score (as specified in Step 4, chance-corrected and baseline-normalised):
`temporal_score = 1 − (acc(max) − 0.05) / (acc(0) − 0.05)`.

### Validity checks (all pass)

- Level-0 std is **exactly 0** for every checkpoint in every sweep → no stray
  randomness leaks into evaluation.
- Eval clean accuracy matches the training summaries to **≤ 0.004**.
- The clean baseline `acc(0)` agrees across all four experiments to **0.00000** —
  the same checkpoints and the same fixed test split were used throughout.
- The perturbation code is byte-identical between the two arms (only class names and
  run tags differ), so arm differences are not implementation differences.
- The jitter operator preserves spike count to ≥ 99.8 % at σ ≥ 3 at every density, so
  there is no hidden rate-loss artifact inflating the dense arm's fragility.

---

## 3. Headline result — every combination contradicts H1

Spearman ρ between **measured** spikes/neuron and `temporal_score`. **H1 predicts
ρ < 0.**

| Perturbation | Arm | n | ρ | p | Verdict |
|---|---|---|---|---|---|
| jitter | no-delay | 15 | **+0.950** | <1e-4 | opposite to H1 |
| jitter | delay | 12 | **+0.832** | 0.0008 | opposite to H1 |
| shift | no-delay | 15 | **+0.957** | <1e-4 | opposite to H1 |
| shift | delay | 12 | **+0.881** | 0.0002 | opposite to H1 |
| shd (f=1) | no-delay | 15 | **+0.950** | <1e-4 | opposite to H1 |
| shd (f=1) | delay | 12 | **+0.671** | 0.0168 | opposite to H1 |
| **deletion** | no-delay | 15 | **+0.936** | <1e-4 | *(rate control)* |
| **deletion** | delay | 12 | **+0.713** | 0.0092 | *(rate control)* |

The trend is not an artifact of the scoring choice: for jitter it holds with
identical sign for relative drop, absolute drop, area-over-curve, and σ₅₀
(half-degradation point: 11.3 ms for the densest no-delay model vs ≥ 25 ms for the
sparsest). `acc(max perturbation)` rises with sparsity in **absolute** terms, so no
normalisation can flip it.

### Group means (3 seeds per strength)

**No-delay arm**

| str | sp/neu | silent | jitter σ=25 | shift σ=25 | shd f=1 | del p=.8 | | score: jit / shf / shd / del |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 7.05 | .33 | .2938 | .1741 | .2311 | .1885 | | .532 / .762 / .653 / .734 |
| 0.5 | 4.34 | .38 | .3225 | .2017 | .2448 | .2032 | | .475 / .708 / .625 / .705 |
| 1.0 | 3.15 | .47 | .3362 | .2236 | .2846 | .2052 | | .424 / .651 / .529 / .688 |
| 3.0 | 1.86 | .52 | .4079 | .2956 | .3866 | .2547 | | .257 / .490 / .301 / .575 |
| 10.0 | 1.34 | .57 | .4090 | .3139 | .3776 | .2672 | | .203 / .414 / .272 / .517 |

**Delay arm**

| str | sp/neu | silent | jitter σ=25 | shift σ=25 | shd f=1 | del p=.8 | | score: jit / shf / shd / del |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 9.79 | .32 | .3339 | .2492 | .2611 | .2407 | | .654 / .757 / .743 / .767 |
| 1.0 | 4.85 | .45 | .4268 | .2557 | .2469 | .2354 | | .528 / .742 / .753 / .768 |
| 3.0 | 3.00 | .49 | .4322 | .2746 | .2702 | .2537 | | .507 / .710 / .716 / .737 |
| 10.0 | 2.27 | .45 | .4478 | .3219 | .3157 | .3007 | | .474 / .640 / .648 / .669 |

Clean accuracy falls only mildly across each sweep (.571→.500 no-delay, .869→.806
delay) and no model is near chance, so the guard condition is satisfied — this is not
"sparser networks are simply broken."

---

## 4. The deletion control changes the interpretation

Deletion destroys *rate*, not timing. If sparsity specifically moved information into
a timing channel, deletion should show a **flat or reversed** trend against the timing
perturbations. It does not — it shows the same strong positive ρ.

Note the control is **conservative in the right direction**: at p_d = 0.8 the sparsest
no-delay model is left with 0.27 spikes/neuron (near silence) while the densest keeps
1.41. Deletion at matched p_d is therefore *harsher* on sparse networks in absolute
terms — and they are still more robust to it.

So the raw headline correlations conflate two things: **(a)** sparse networks are
more robust to everything, and **(b)** whatever is specific to timing. Separating them
is the real analysis.

### 4a. Partial correlation — ρ(sp/neuron, timing score | deletion score)

| Arm | Timing pert. | raw ρ | **partial ρ** | partial p | Timing effect beyond robustness? |
|---|---|---|---|---|---|
| no-delay | jitter | +0.950 | **+0.576** | 0.031 | yes — still opposite to H1 |
| no-delay | shift | +0.957 | **+0.586** | 0.028 | yes — still opposite to H1 |
| no-delay | shd | +0.950 | **+0.620** | 0.018 | yes — still opposite to H1 |
| delay | jitter | +0.832 | **+0.741** | 0.009 | yes — still opposite to H1 |
| delay | shift | +0.881 | **+0.741** | 0.009 | yes — still opposite to H1 |
| delay | **shd** | +0.671 | **+0.364** | **0.271** | **no — explained by robustness** |

### 4b. Severity-matched contrast

`shd(f=1) − deletion(p=.8)` isolates "all timing destroyed" against "most spikes
destroyed":

- **No-delay:** ρ(sp/neuron, shd − deletion) = **+0.825, p = 0.0002.** The contrast
  widens from −0.05 (dense) to −0.27 (sparse) — timing destruction costs sparse
  networks progressively *less* relative to spike destruction.
- **Delay:** ρ = **−0.161, p = 0.62** — flat. Every delay model sits at
  shd − deletion ≈ −0.02.

### 4c. Equal-damage comparison

*What fraction of spikes must deletion destroy to hurt as much as destroying all
timing?* Higher = more timing-dependent.

| Arm | densest → sparsest | ρ | p |
|---|---|---|---|
| no-delay | **0.73 → 0.51** | +0.868 | <1e-4 |
| delay | 0.74 → 0.80 (flat) | −0.147 | 0.65 |

A dense no-delay network loses as much from scrambling all timing as from deleting
~73 % of its spikes; the sparsest loses only as much as deleting ~51 %. The delay arm
is flat at ~0.75–0.80.

> ⚠️ **Delay-arm censoring caveat.** The deletion grid stops at p_d = 0.8 and two
> delay models interpolate exactly to 0.800, so their true equivalent probability may
> be higher. The delay arm's "flat" reading is compressed against the grid edge and
> should be confirmed with p_d extended to 0.9/0.95 before being leaned on hard.

---

## 5. Refined conclusions

**H1 is not supported in either arm.** Beyond that, the two arms say different things:

- **No-delay arm — H1 is actively contradicted.** Sparsity *reduces* timing
  dependence, and the reduction survives the general-robustness control on all three
  timing perturbations (partial ρ ≈ +0.58…+0.62) and both severity-matched contrasts.
  The count-only accuracy `acc(f=1)` *rises* with sparsity (.231 → .378), i.e. sparse
  no-delay networks carry more of their performance in the jitter-immune channel.

- **Delay arm — timing dependence is roughly invariant to sparsity.** Under complete
  timing destruction the trend vanishes once robustness is controlled. But bounded
  perturbations (jitter, shift at σ=25) *do* retain a significant positive partial ρ.
  The natural reading: sparsity does not change **how much** the delay network relies
  on timing, only **how precisely** — sparse delay networks use a coarser temporal
  code that tolerates ±25 ms displacement better, while still collapsing just as far
  when timing is destroyed outright. That is a *timescale* effect, not a
  *presence-of-timing* effect, and it is not what H1 claims.

---

## 6. Why it inverted — the mechanism

[sparse_network.md:90-92](sparse_network.md#L90-L92) grounds H1 in the assumption that
in a sparse layer "each neuron fires 0–2 spikes, so count carries almost no
resolution." **That assumption is false for these networks** — `spikes_per_neuron` is
diluted by silent neurons and hides the real per-neuron statistics:

| checkpoint (seed 42) | sp/neuron | silent | **sp/ACTIVE neuron** | decode-from-identity | **decode-from-count** | net acc(f=1) |
|---|---|---|---|---|---|---|
| nodelay str0.01 | 6.50 | .33 | 9.74 | .453 | .749 | .244 |
| nodelay str10 | 1.42 | .54 | **3.12** | **.513** | .687 | .369 |
| delay str0.01 | 11.66 | .25 | 15.47 | .339 | .759 | .313 |
| delay str10 | 2.12 | .48 | **4.10** | **.502** | .736 | .314 |

*(decode = multinomial logistic regression fit on the train split, scored on the test
split, from the per-neuron spike-count vector or its binarisation — both exactly
invariant to every rate-preserving perturbation here.)*

Three things follow:

1. **Count resolution never collapses.** Even the sparsest models fire 3–4 spikes per
   *active* neuron. The regime H1's mechanism requires (0–2 spikes, no count
   resolution) was never reached.
2. **Sparsity is mostly thinning, but the selectivity it adds is what matters.**
   Decomposing `sp/neuron = (1 − silent) × sp/ACTIVE` across each arm's full range:

   | Arm | total | = selectivity | × thinning | selectivity share |
   |---|---|---|---|---|
   | no-delay | 4.58× | 1.46× | 3.12× | 24.8 % |
   | delay | 5.50× | 1.44× | 3.77× | 21.6 % |

   So ~75 % of the rate reduction comes from active neurons firing less, and ~25 %
   from neurons falling silent on more *samples* (they are stimulus-selective, not
   dead). That secondary effect is nonetheless decisive: it *strengthens* the
   population-identity channel — decode from identity alone **rises** with sparsity
   (.453→.513, .339→.502) — and every perturbation in this study leaves neuron
   identity untouched.
3. **The jitter-immune channel stays rich throughout.** A linear decode from counts
   alone scores .69–.78 at every sparsity level, far above any network's own
   `acc(f=1)`. Information was pushed **into** the perturbation-immune channels, not
   out of them.

This is pitfall 3 of the design doc, realised. The mechanism section of
[sparse_network.md](sparse_network.md) needs correcting, not the experiment re-running.

---

## 7. Secondary findings

**(a) Delays do carry temporal processing — as expected.** At matched firing rate
(2.7–5.1 spikes/neuron, n = 6 per arm) the delay arm is more perturbation-sensitive
on every measure, and the gap is largest for the cleanest timing probe:

| Perturbation | no-delay | delay | gap |
|---|---|---|---|
| jitter | 0.450 | 0.518 | +0.068 |
| shift | 0.679 | 0.726 | +0.047 |
| **shd (f=1)** | **0.577** | **0.735** | **+0.158** |
| deletion *(control)* | 0.697 | 0.753 | +0.056 |

The shd gap (+0.158) is nearly three times the deletion gap (+0.056), so this is not
just "the delay net is more fragile overall" — learnable axonal delays genuinely add
timing dependence. This was the reason the second arm was added, and it is the one
prediction in the design that held.

**(b) Shift is more damaging than jitter.** At σ = 25, shift scores .41–.76 vs
jitter's .20–.65 in the no-delay arm. Moving a neuron's whole train coherently
destroys cross-neuron alignment and onset information, which these networks use more
than within-train fine structure.

**(c) The no-delay relocation curve is a cliff, not a slope.** All 15 no-delay models
lose essentially all their perturbable accuracy by **f = 0.2** and are flat
thereafter (e.g. str0.01 seed43: .560 → .234 → .238 → .255 → .242 → .235), whereas the
delay arm degrades smoothly (.869 → .818 → .702 → .526 → .356 → .261). The no-delay
`shd` score is therefore effectively determined by a single grid point and is
under-resolved; a finer grid (f ∈ {0.02, 0.05, 0.1, 0.15, 0.2}) would be needed to
characterise that arm's relocation curve properly.

---

## 8. Caveats

1. **Perturbation magnitude is not matched across densities.** Moving or deleting a
   fixed *fraction* of 900 spikes perturbs the downstream membrane more than the same
   fraction of 170. The deletion control addresses this (and is conservative), but it
   does not fully equate perturbation energy. This is the main residual threat to the
   no-delay conclusion.
2. **Delay-arm equal-damage numbers are censored** at the p_d = 0.8 grid edge (§4c).
3. **The no-delay `shd` grid is under-resolved** at low f (§7c).
4. **Readout caveat (inherited).** The probe only reveals timing the *readout* uses; a
   layer could hold temporal structure `fc3` ignores. Stated, not fixed.
5. **The sparsity mechanism is one specific intervention.** These conclusions concern
   sparsity induced by a hinge rate penalty, which silences neurons. A sparsity
   mechanism that thinned all neurons uniformly without silencing any might behave
   differently — and would be the fair test of H1's actual mechanism.
6. **Arms differ in warm-up** (no-delay 15, delay 0) and in clean accuracy
   (~.50–.59 vs ~.78–.89). Cross-arm claims use the chance-corrected, baseline-
   normalised score at matched firing rate for this reason.

---

## 9. Recommended next steps

1. **Record the verdict** in the milestone checklist — Steps 3–4 are done; the
   milestone succeeded on its own terms ("either direction is a valid finding").
2. **Correct the mechanism section** of [sparse_network.md](sparse_network.md): the
   "0–2 spikes, no count resolution" premise does not hold, and pitfall 3 is what
   actually happened.
3. **Log spikes-per-*active*-neuron** in the training summaries. The silent-diluted
   mean is what hid this for the whole of Step 1.
4. **Extend the deletion grid** to p_d ∈ {0.9, 0.95} to de-censor §4c.
5. **Refine the no-delay `shd` grid** at low f (§7c).
6. **The decisive follow-up:** induce sparsity *without* silencing neurons (e.g. a
   per-neuron hinge that penalises only above-target neurons, or an explicit
   anti-silencing term). If H1's mechanism is right, that variant should push each
   neuron toward the 0–2 spike latency-code regime the current intervention never
   reached — and is the only way to test the supervisor's prediction as stated.
7. Produce the three Step-4 plots (headline / curve families / guard) from these
   result files for the writeup.

---

## 10. Reproduction

| Artefact | Path |
|---|---|
| Checkpoints (27) | `exp_sparse_network/sn_data/` |
| Training summaries | `exp_sparse_network/sn_log/sparse_whole_{delay,nodelay}_train_summary.json` |
| Jitter results | `exp_sparse_network/jitter/log/sparse_whole_{delay,nodelay}_jitter_eval.json` |
| Shift results | `exp_sparse_network/shift/log/sparse_whole_{delay,nodelay}_shift_eval.json` |
| Relocation results | `exp_sparse_network/shd/log/sparse_whole_{delay,nodelay}_shd_eval.json` |
| Deletion results | `exp_sparse_network/deletion/log/sparse_whole_{delay,nodelay}_deletion_eval.json` |

Each result file is self-contained: run tag → `penalty_strength`, `seed`, measured
sparsity metrics, and the full sweep with per-repeat values. Analyse against the
**measured** firing rate, never the penalty strength.
