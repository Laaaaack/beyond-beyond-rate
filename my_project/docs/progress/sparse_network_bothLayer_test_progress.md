# Both-layer v3 — the same factorial, on the whole hidden network (design, calibration)

**Status (2026-07-30): CALIBRATION STEP 0 COMPLETE. Both training scripts implemented
and verified end to end. Corner probe NOT yet run; neither grid launched; the primary
arm is undecided.**

The 1st-layer v3 factorial finished on 2026-07-29 with a null, and the 2nd-layer
factorial was built on 2026-07-30 to ask whether that null is a fact about temporal
coding or a fact about layer 1. Both constrain **one** layer. Neither asks what happens
when the *network* is made sparse — and the measurement below says that is a third
question, not a rephrasing of either.

**The calibration finding that justifies the whole exercise:** under v1's
**layer-1-only** penalty, the two layers' silent fractions move in **opposite**
directions in the no-delay arm — ρ(`s1`, `s2`) = **−0.829**. As layer 1 is driven toward
silence, layer 2 becomes *less* silent. A layer-1 penalty does not make the network
selective; it trades layer 1's silence for layer 2's activity. Pooled over both layers,
`a` and `s` are then confounded at **−0.879**, comfortably past the |ρ| ≤ 0.5 acceptance
threshold. **Owner:** _(you)_

**This is document 7 of 7. It is a sibling of documents 5 and 6, not a successor —
read document 5 first.**

| # | Document | What it is |
|---|---|---|
| 1 | [sparse_network.md](sparse_network.md) | the question and the conceptual landscape — **start there** |
| 2 | [sparse_network_test_progress.md](sparse_network_test_progress.md) | v1 execution log |
| 3 | [sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](sparse_network_test_progress_v2.md) | v2 execution log |
| 5 | [sparse_network_test_progress_v3.md](sparse_network_test_progress_v3.md) | **v3 at layer 1** — design, Phase 0, Phase 1 result. Everything here assumes it |
| 6 | [sparse_network_2ndLayer_test_progress_v3.md](sparse_network_2ndLayer_test_progress_v3.md) | v3 at layer 2 — calibration, design deltas |
| 7 | **this file** | v3 at **both** layers — calibration, design deltas, status |

Everything in document 5 that is not about *which layer* carries over unchanged: the
two-variable decomposition of sparsity (§3b), the H1′/H2 restatement (§3c), the
capacity-matched decoder (§3a), the mechanism of the two penalties (§4b), and the
reading table for the outcome (§5). This document records only what had to be
**re-measured, re-decided, or is newly at risk** because the target is now two layers
rather than one.

---

## 1. Why constrain both layers at once

Document 6 §1 argued that layer 1 may not be where the network's timing computation
lives. This document makes a different and more basic point: **the two single-layer
grids never produce a sparse network.** They produce a network with one sparse layer and
one unconstrained layer that is free to compensate — and the measurement in §2 shows it
does exactly that.

Three consequences, in order of how much they matter:

1. **"Sparser-activity networks" is the supervisor's phrase, and neither single-layer
   grid instantiates it.** v1's densest and sparsest no-delay models differ by **6.5×**
   in layer-1 firing (1.21–7.85 spikes/neuron) but only **2.8×** network-wide
   (2.37–6.69 pooled), because layer 2 partly absorbs the cut. The manipulation is
   **57% weaker** than the headline layer-1 numbers suggest.
2. **The compensation is not neutral — it runs against the manipulation's intent.**
   ρ(`s1`, `s2`) = −0.829 (no-delay) means the layer-1 penalty *reduces* layer 2's
   selectivity while raising layer 1's. Since document 5's Phase 1 verdict was that the
   residual signal is **selectivity** (β\* = −0.44, the largest term), a design in which
   the two layers' selectivity moves in opposite directions is measuring a partly
   cancelled quantity.
3. **It gives the layer-1 and layer-2 grids their control.** With all three grids run,
   the same `k` × floor design exists at layer 1 alone, layer 2 alone, and both. If β_a
   is null in all three, the null is about the architecture; if it appears only when the
   whole network is constrained, that is a positive result no single-layer grid could
   have produced.

This is a replication in **venue**, not a new hypothesis. H1′ and H2 are unchanged.

---

## 2. Calibration step 0 — what the network-wide axes actually look like

**Source:** [layer2_baseline.py](../../exp_sparse_network/v3_analysis/layer2_baseline.py)
→ `v3_analysis/log/layer2_baseline_{arm}.json`, which already records **both** layers per
checkpoint. The cross-layer and pooled correlations below were computed from that file
over all 27 frozen v1 checkpoints; no new training and no new eval pass was needed.

Pooling is over `(sample, neuron)` pairs across both layers. Both layers have 128 units,
so `a_net = (spikes1 + spikes2) / (active1 + active2)` and `s_net` is the plain pair
fraction — exact, not an average of two ratios.

### 2a. The headline table

| across v1's gradient | no-delay (n=15) | delay (n=12) |
|---|---|---|
| **ρ(`a1`, `s1`)** — layer 1's own confound | **−0.943** (p=1.4e−7) | **−0.455** (p=.14) |
| **ρ(`a2`, `s2`)** — layer 2's own confound | **+0.829** (p=1.4e−4) | **+0.427** (p=.17) |
| **ρ(`s1`, `s2`)** — cross-layer selectivity | **−0.829** (p=1.4e−4) | **+0.203** (p=.53) |
| **ρ(`a1`, `a2`)** — cross-layer activity | **+0.961** (p=1.3e−8) | **+0.839** (p=6.4e−4) |
| **ρ(`a_net`, `s_net`)** — the pooled confound | **−0.879** (p=1.6e−5) | **+0.042** (p=.90) |
| ρ(`a1`, `s2`) | +0.796 | +0.434 |
| ρ(`a2`, `s1`) | −0.954 | −0.259 |
| `a1` range | 3.06 – 11.49 | 3.82 – 15.47 |
| `a2` range | 7.49 – 12.46 | 20.71 – 30.44 |
| **`a_net` range** | **5.47 – 11.87** (2.17×) | **14.47 – 22.60** (1.56×) |
| `s1` range | 31.7% – 60.5% | 24.6% – 55.1% |
| `s2` range | 46.9% – 61.1% | 4.4% – 39.6% |
| **`s_net` range** | **43.6% – 56.7%** | **19.7% – 47.4%** |
| **`spikes/neuron` network-wide** | **2.37 – 6.69** (2.8×) | **8.17 – 17.20** (2.1×) |

### 2b. Five things this settles

**1. The layer-1 penalty makes the network less selective at layer 2, not more.**
ρ(`s1`, `s2`) = **−0.829** in the no-delay arm. This is the single most important number
in this document and it has no analogue in documents 5 or 6, because neither measured
two layers at once. In the delay arm it is **+0.203** — not opposed, but essentially
unrelated, which is the same failure in a weaker form: v1's penalty simply had no
consistent effect on layer 2's selectivity.

**2. The network-wide manipulation is much weaker than the layer-1 one.** Measured over
the same checkpoints:

| | layer 1 span | network-wide span | weaker by |
|---|---|---|---|
| `spikes/neuron`, no-delay | 6.50× | **2.83×** | **57%** |
| `spikes/neuron`, delay | 5.51× | **2.11×** | **62%** |
| `a`, no-delay | 3.76× | **2.17×** | **42%** |
| `a`, delay | 4.05× | **1.56×** | **61%** |

Every effect size quoted per layer in documents 3 and 5 is an effect size for a network
that was sparsified in one place and compensated in another.

**3. The pooled confound is arm-specific, and only the no-delay arm needs breaking.**
At **−0.879** the no-delay arm fails the |ρ| ≤ 0.5 threshold outright. At **+0.042** the
delay arm already passes. So the two arms buy different things here, and the scripts say
so: in the delay arm what this design adds is the **controlled** floor-on vs floor-off
contrast at matched `a` — a manipulated comparison where v1's gradient was only a
correlation — not decorrelation. That is the same reason the 1st-layer factorial was
worth running in that arm at ρ = −0.455.

**4. ρ(`a1`, `a2`) is high and this design will not lower it.** +0.961 / +0.839
observationally, and both knobs here act on both layers, so `a1` and `a2` are moved
together **by construction**. This grid therefore cannot attribute an effect to one
layer's activity rather than the other's. That is a property of the question, not a
defect — attribution is what the two single-layer grids are for — but it must be stated
wherever a β from this grid is reported. Both scripts print ρ(`a1`, `a2`) and
ρ(`s1`, `s2`) as **diagnostics explicitly excluded from the pass/fail check**.

**5. The two layers' natural rates differ enough to force a per-layer `k`.** `a2`/`a1`
at the dense end is ~1.1× in the no-delay arm and ~2.0× in the delay arm. See §3a.

---

## 3. What changed in the design, and what did not

Both scripts are forks of their 1st-layer siblings, cross-checked against the 2nd-layer
ones. Data, splits, optimiser, schedule, early stopping, the two penalty *forms* and the
acceptance criteria are all unchanged. The parameter set is untouched, so these
checkpoints load into the existing eval pipeline exactly as v1's do (verified with
`strict=True` against both arms — see §4).

**Implemented in:**
[sn_bothLayer_train_noDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_noDelay_v3.py)
and
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py).

**The mechanical change.** `forward(return_hidden=True)` now returns
`(out, hidden1, potential1, hidden2, potential2)`. The forward pass is split into
`_first_hidden` / `_second_hidden` / `_output`, each returning its layer's potential
alongside its spikes. Each penalty is charged **once per layer and summed**, at the same
coefficient its single-layer sibling used:

```python
loss = task_loss
for layer in (1, 2):
    loss += CEILING_STRENGTH * relu(count[layer] - k[layer]).mean()
    loss += floor_strength   * relu(THETA_MARGIN - peak_u[layer]).mean()
```

Charging each layer at the **full** coefficient rather than at half is deliberate: it
means a cell of this grid applies to layer 1 exactly the pressure the 1st-layer grid
applied, and to layer 2 exactly what the 2nd-layer grid applied, so the three grids are
comparable cell by cell. The cost is that the *total* penalty gradient is roughly
doubled — see §3b.

**Artifacts are tagged `v3L12`.** All three grids share dataset, arm, `k`, floor and
seed, so without distinct tags (`v3`, `v3L2`, `v3L12`) they would produce identical
filenames and silently overwrite one another. Run tags are
`sparse_whole_{arm}_v3L12_k{k}_floor{f}_seed{seed}`, and each summary row carries
`target_layers: [1, 2]`.

**Logging is per layer *and* pooled.** The per-epoch log carries `a`, `s`, `over_k` and
`sub_threshold` separately for each layer, plus the pooled `a_net` / `s_net` /
`rate_net`. The progress bar shows `a1 s1 a2 s2` together, because the failure this run
is most exposed to is one layer collapsing while the other looks healthy.

**The summary's canonical keys point at the network.** `spikes_per_active_neuron`,
`silent_fraction` and `spikes_per_neuron` are emitted **unsuffixed carrying the pooled
values**, alongside the explicit `_l1` / `_l2` / `_net` keys. This is so that analysis
code written against the single-layer summaries reads the quantity this experiment
manipulates rather than one layer of it by accident. It is deliberate, documented in
both scripts, and should not be repointed at a single layer.

### 3a. The one substantive design change: the row axis is scaled per layer

`CEILING_K` lists **layer 1's** levels; layer 2's are `CEILING_K_LAYER2_RATIO` times
them, so one row knob delivers an equal *relative* cut at both layers rather than an
equal absolute budget.

| arm | natural `a` (L1 / L2) | ratio | `k1` | `k2` | models |
|---|---|---|---|---|---|
| no-delay | 3.06–11.49 / 7.49–12.46 | **1.0** | `[1, 2, 4, 8]` | `[1, 2, 4, 8]` | 16 |
| delay | 3.82–15.47 / **20.71–30.44** | **2.0** | `[1, 2, 4, 8]` | **`[2, 4, 8, 16]`** | 16 |

The no-delay ranges largely overlap, so there an equal absolute budget already *is* an
equal relative one and the ratio is a no-op — but it is not decorative, because the
delay arm's layer 2 runs about twice as hard and a shared `k` would be a mild cut for
layer 1 and a savage one for layer 2 in the same cell.

This also settles, more cheaply, the problem the 2nd-layer delay script had to solve
with an extra grid row. That script kept a shared budget and added `k = 16` to anchor
the top of layer 2's axis near its natural rate — 20 models, ~46 h. Here the ratio does
the same job inside 4 rows: the top row is `k1 = 8, k2 = 16`, just under each layer's
own natural range. **16 models per arm, not 20.**

*This is the one decision in this document that is a judgement call rather than a
measurement.* The alternative — one shared absolute `k`, the most literal reading of
"the same mechanism on both layers" — was considered and rejected on the delay arm's 2×
rate gap. It is a one-line revert (`CEILING_K_LAYER2_RATIO = 1.0`), at the cost of
needing the 5th row back in that arm.

### 3b. Inherited but NOT yet validated — the reason the probe exists

| constant | value (no-delay / delay) | status |
|---|---|---|
| `NATURAL_SPIKES_PER_NEURON` | {L1 7.85, L2 5.53} / {L1 11.66, L2 23.93} | **measured per layer, per arm** |
| `CEILING_K_LAYER2_RATIO` | 1.0 / 2.0 | **set from the measured natural `a`** |
| observational ρ(`a`, `s`) baselines | −0.943 / +0.829 / −0.879 and −0.455 / +0.427 / +0.042 | **measured (§2a)** |
| `CEILING_STRENGTH` | 10.0 | **inherited — unvalidated with two penalties** |
| `FLOOR_STRENGTH` | 1.0 | **inherited — unvalidated with two penalties** |
| `WARMUP_EPOCHS` | 20 | **inherited — unvalidated with two penalties** |

Both scripts ship with `QUICK_TEST = True` for this reason. Every inherited constant was
calibrated against a *single* layer's activity, and this configuration is new in a way
neither sibling was:

- **`CEILING_STRENGTH` has arguments in both directions, which is why it must be
  measured rather than reasoned about.** `fc1` now receives ceiling pressure twice —
  directly from the layer-1 term and through the surrogate from the layer-2 term
  (measured at §4: |grad| 0.81 and 2.05 from the layer-2 term alone in the two arms) —
  which argues for a *smaller* value. But at layer 1 even strength 10 left 41% of pairs
  above `k` at `k = 1`, and the delay arm's layer 2 starts twice as dense, which argues
  for a *larger* one. The two `over_k` columns say which is happening, per layer. **If
  one layer stalls high and the other does not, that asymmetry is the thing to fix, not
  the overall level.**
- **`WARMUP_EPOCHS = 20` guards an absorbing failure, and this is the most exposed of
  the three scripts.** The ceiling is minimised at `count = 0`, so in the floor-off
  column only the task loss opposes silence — now at *both* layers at once. And the two
  failures compound: emptying layer 1 additionally zeroes the layer-2 floor's gradient
  path (§3c), so the instrument that would rescue layer 2 dies at the same moment. The
  `k = 1`, floor = 0 corner is the cell that would show it, and checking it is
  **criterion 1** of the probe rather than criterion 4.

### 3c. The floor's layer-2 reachability edge now cuts both ways

`∂potential2 / ∂W2 = psp(hidden1)` (no-delay) or `psp(delay1(hidden1))` (delay) is
surrogate-free, **but exactly zero for any sample whose entire layer 1 is silent** —
such a pair is reachable only back through `fc1`'s surrogate. Document 6 §2c measured
this at **0.00% of samples** across all 27 v1 checkpoints and flagged it as a documented
edge with a known trigger.

This design moves it in **both** directions, and both are worth stating:

- **In the floor-ON column it is now actively guarded.** The layer-1 floor forbids
  layer-1 silence, which is the precondition for the edge case. Neither single-layer
  script had that protection — the 2nd-layer script left layer 1 entirely unconstrained,
  so its floor's own reachability depended on a layer nothing was defending.
- **In the floor-OFF column it is worse than in either sibling.** The ceiling now pushes
  both layers toward silence with nothing opposing either, so emptying layer 1 is more
  reachable here than anywhere previously. Same corner as §3b: `k` at its lowest, floor
  0.

---

## 4. Implementation verification — what was actually checked

Run against both arms on GPU, ~1 min, no training. Twenty checks per arm, all passing.

| # | check | no-delay | delay |
|---|---|---|---|
| 1 | `forward(return_hidden=True)` returns **5** tensors, `hidden1 ≠ hidden2` | ✔ (L1 3217 / L2 7362 spikes on random input) | ✔ (9122 / 18721) |
| 2 | v1 checkpoint loads with `strict=True` — parameter set untouched | ✔ 8 tensors | ✔ 10 tensors |
| 3 | ceiling at L1 → nonzero grad on `fc1` | \|grad\| 1.595 | 3.385 |
| 4 | floor at L1 → nonzero grad on `fc1` | 1.104 | 0.593 |
| 5 | ceiling at L2 → nonzero grad on `fc2` | 1.351 | 7.455 |
| 6 | floor at L2 → nonzero grad on `fc2` | 0.811 | 0.427 |
| 7 | **L2 ceiling also reaches `fc1`** — the "`fc1` is squeezed twice" claim of §3b, now measured rather than asserted | **0.808** | **2.046** |
| 8 | **§3c edge case reproduces**: forcing layer 1 silent makes `∂(floor2)/∂W2` **exactly 0** | ✔ 0.000000 | ✔ 0.000000 |
| 9 | `pool_layer_stats` matches a direct pooled computation to 1e−9, on all three axes | ✔ | ✔ |
| 10 | layer 2's recorded ceiling is the **scaled** one (`k1 = 1` → `k2`) | 1.0 | **2.0** |
| 11 | summary emits every documented key; canonical unsuffixed keys alias the **network** pool | ✔ | ✔ |
| 12 | per-epoch log carries both axes at both layers plus the pool (+ `delay_mean`) | ✔ | ✔ |
| 13 | both report functions run; probe resolves to the 4 corners at 400 epochs | ✔ | ✔ |

Rows 7 and 8 are the two that could not have been settled by reading the code. Row 7
quantifies the argument that `CEILING_STRENGTH` may now be too *strong* for layer 1; row
8 confirms the failure mode is real and reachable, so §3b's criterion 1 is guarding
something that exists.

---

## 5. What has to happen next

### Step 1 — corner probe, both arms (~4.8 h total)

`QUICK_TEST = True` is already set in both scripts. Runs `k1 ∈ {1, 8}` × floor ∈ {0, 1},
one seed, 400 epochs — layer 2 seeing `{1, 8}` (no-delay) or `{2, 16}` (delay). Probe
artifacts carry a `_probe` suffix and cannot clobber a real run.

Acceptance criteria, **reordered** from documents 5 and 6 — the survival check is now
first, for the reason in §3b:

1. **Both layers survive the `k = 1`, floor = 0 corner.** `val_acc` at chance (5%) with
   `silent → 100%` at *either* layer means `WARMUP_EPOCHS` is too short. Raise it before
   touching anything else;
2. `silent_fraction` roughly constant as `k` varies *within* a column, at each layer;
3. `silent_fraction` differing by **≥ 20 points** *across* columns at matched `k`, **at
   both layers** — a floor that bites at one layer only leaves the column knob doing
   half its job, and makes the pooled `s_net` axis a statement about one layer;
4. `spikes_per_neuron` not inflated beyond that layer's own natural rate — 7.85 / 5.53
   (no-delay), 11.66 / 23.93 (delay);
5. clean accuracy comfortably above chance in every corner.

Both scripts print criteria 3 and 5 automatically (`report_manipulation_check` reports
the separation per `k` **per layer**, flagging `TOO SMALL` under 20 points).

**Read the probe for the manipulation only.** Every v1 model was still improving at
epoch 1250, so 400 epochs is a different regime, not a scaled-down one. Document 5 also
established that penalties calibrated at reduced epochs **re-densify** by the full run —
achieved `a` will sit *above* the probe's numbers.

### Step 2 — the grids

| arm | models | est. cost | what it buys |
|---|---|---|---|
| no-delay | 4 `k` × 2 floor × 2 seeds = 16 | ~29 h | breaks a pooled confound of **−0.879** |
| delay | 4 × 2 × 2 = 16 | ~37 h | the pooled baseline (+0.042) already passes — buys the **controlled contrast**, and a direct comparison against the completed 1st-layer delay grid |

Cost is scaled from the siblings' measured per-model times (1.8 h / 2.3 h); the extra
penalty terms are negligible against the forward/backward pass.

**Which arm is primary is an open question here, exactly as it is in document 6 §4.**
Document 5's delay-arm decision rested on two Phase 0 measurements of the *dependent*
variable — the readout-efficiency gap (+.300 vs +.026) and the deletion control being
indistinguishable from the timing probe in the no-delay arm (+0.936 vs +0.939) — both
measured **at layer 1 only**. Deciding on them here would be inheriting a constant
across a layer boundary, which is the mistake this calibration exists to avoid.

There is, however, one consideration that is specific to this document and points at the
**no-delay** arm: it is the only arm where the pooled confound actually fails (−0.879),
so it is where this design has something to break rather than merely something to
control. Set against that, the delay arm has a **completed** 1st-layer grid at 1250
epochs to compare against, which is the comparison in §1 point 3. Decide after step 3.

### Step 3 — retarget the measurement layer (not yet started)

The largest piece of outstanding work, shared with document 6, and **not** optional: the
grids are uninterpretable without it. Every measurement script is currently hardwired to
layer 1.

| what | files | change needed |
|---|---|---|
| perturbation injection site | 12 eval scripts under `{jitter,shift,shd,deletion}/` | run at **both** `hidden1` and `hidden2` |
| perturbation window | same 12, `SUPPORT_BINS = 88` | per layer **and** per arm: L1 `[0, 88)` both arms; L2 `[0, 90)` no-delay, **`[0, 160)`** delay |
| availability decode | `v3_analysis/hidden_channel_decode.py` | probe both layers; window per layer and arm |
| usage + deletion control | `v3_analysis/phase1_measure.py` | same |

Three traps, all still live:

- **The window correction is not uniform across perturbations.** Relocation and jitter
  choose a destination bin and must respect the support; **shift and deletion must be
  left alone.** Clipping a rigid translation would pile spikes at the edge and merge
  them, destroying per-neuron count — a rate insult that does not currently exist.
- **`delay1` moves from downstream to upstream depending on the probe site.** A layer-1
  probe correctly ignores it; a layer-2 probe **must apply it**. Getting this wrong
  silently measures the wrong tensor rather than raising an error.
  [layer2_baseline.py](../../exp_sparse_network/v3_analysis/layer2_baseline.py) already
  does it correctly and is the reference.
- **A network-wide dependent variable is a design choice, not a given.** The independent
  variables here are pooled; the dependent ones need not be. Perturbing both layers at
  once is a *different* and harsher insult than perturbing either — report per-layer
  usage and availability first, and treat any pooled dependent variable as an addition
  rather than a replacement.

---

## 6. Watch out for

Inherits document 1 §7, document 4 §7, document 5 §8 and document 6 §5 in full. New to
the both-layer experiment:

1. **This grid cannot attribute an effect to a layer.** ρ(`a1`, `a2`) is +0.96 / +0.84
   observationally and is moved together by construction here. A β from this grid is a
   **network-level** coefficient. Attribution is what the two single-layer grids are
   for, and any claim of the form "layer N's sparsity causes X" needs one of them.
2. **The pooled axes are not the layer-1 axes, and the gap is large.** v1's no-delay
   gradient is 6.5× in layer-1 firing but 2.8× network-wide. Quoting a layer-1 range as
   the manipulation's strength overstates it by **57%** (42% on the `a` axis, and 61–62%
   in the delay arm).
3. **ρ(`s1`, `s2`) is the number to watch, and its v1 sign is a warning.** At −0.829 the
   layer-1 penalty *reduced* layer 2's selectivity. If this design does not pull it
   clearly positive, the column knob is not producing a network-wide selectivity
   manipulation and the pooled `s_net` axis is a statement about one layer. Both scripts
   print it against its v1 baseline.
4. **The `v3L12` tag is load-bearing, now against two prior grids.** All three share
   dataset, arm, `k`, floor and seed. Without the tag this run overwrites the completed
   1st-layer delay grid (~37 h to retrain) or the 2nd-layer one.
5. **The canonical unsuffixed summary keys mean the network here, not layer 1.** A
   reader or script that assumes the single-layer convention will silently compare a
   pooled quantity against a per-layer one. Every per-layer value is available under an
   explicit `_l1` / `_l2` suffix — use those when comparing across grids.
6. **The floor-off column at `k = 1` is the most dangerous cell in the project so far.**
   Two one-sided-toward-zero ceilings, nothing opposing either, and a documented
   gradient path that dies precisely when the first layer empties (§3c). It is probe
   criterion 1 for that reason.
7. **`CEILING_K_LAYER2_RATIO` is an arm constant.** 1.0 and 2.0 are measured from each
   arm's own natural `a`. Carrying either across would produce an unequal cut in the
   other arm — the general rule of document 6 §5.1, in its newest instance.
8. **The arm decision is not inherited** (§5 step 2), and unlike documents 5 and 6 there
   is now a consideration pointing the *other* way: the no-delay arm is the only one
   whose pooled confound actually fails.

---

## 7. Checklist

- [x] Both training scripts retargeted to **both** hidden layers — penalties, logged
      statistics, clean evaluation and summary all act on `hidden1`/`potential1` **and**
      `hidden2`/`potential2`, plus the pooled network-wide axes
- [x] Artifacts tagged `v3L12`; summary rows carry `target_layers`; probe runs carry
      `_probe` — no path can collide with the 1st-layer (`v3`) or 2nd-layer (`v3L2`)
      grids
- [x] Parameter set verified unchanged: both scripts load v1 checkpoints with
      `strict=True` (8 / 10 tensors), so the existing eval pipeline stays compatible
- [x] **Calibration step 0** — cross-layer and pooled axes computed over all 27 v1
      checkpoints from `layer2_baseline_{arm}.json`
- [x] The motivating finding measured: **ρ(`s1`, `s2`) = −0.829** (no-delay) — a
      layer-1-only penalty makes the two layers' selectivity move in *opposite*
      directions; **+0.203** (delay), i.e. unrelated
- [x] Pooled confound measured: **ρ(`a_net`, `s_net`) = −0.879** (no-delay, fails the
      threshold) and **+0.042** (delay, already passes) — the two arms buy different
      things and both scripts say which
- [x] Cross-layer activity confound measured (**+0.961 / +0.839**) and explicitly
      excluded from the acceptance check as something this design does not break
- [x] Network-wide manipulation strength measured: `spikes/neuron` **2.83× / 2.11×**
      against layer 1's 6.50× / 5.51×, and `a_net` **2.17× / 1.56×** against `a1`'s
      3.76× / 4.05× — the single-layer figures overstate the manipulation by 42–62%
- [x] Row axis scaled per layer (`CEILING_K_LAYER2_RATIO` = 1.0 / **2.0**) from the
      measured natural `a`, keeping both arms at **16 models** where the 2nd-layer delay
      grid needed 20
- [x] Implementation verified end to end (§4): 5-tensor forward, all four penalty→weight
      gradient paths nonzero, the L2-ceiling→`fc1` path measured at 0.81 / 2.05, the
      §3c edge case reproducing exactly 0, and pooling exact to 1e−9
- [ ] **Corner probe, both arms** (~4.8 h) — validate `CEILING_STRENGTH`,
      `FLOOR_STRENGTH` and `WARMUP_EPOCHS` *with both penalties running*.
      `QUICK_TEST = True` is set. **Criterion 1 is survival of the `k = 1`, floor = 0
      corner at both layers**
- [ ] Re-measure the dependent variable per layer (readout gap, deletion control vs
      timing probe) and **decide the primary arm** on it
- [ ] Retarget the 12 eval scripts + 2 analysis scripts to both layers at each layer's
      own per-arm window, leaving shift and deletion's windows alone
- [ ] Grid: no-delay 16 models (~29 h) and/or delay 16 models (~37 h) at 1250 epochs
- [ ] Design acceptance check on the real grid at all three levels, against the
      observational baselines **−0.943 / +0.829 / −0.879** (no-delay) and
      **−0.455 / +0.427 / +0.042** (delay)
- [ ] Confirm ρ(`s1`, `s2`) turned positive — the column knob is a network-wide
      selectivity manipulation or it is not
- [ ] Regressions and the controlled floor-on vs floor-off contrast at matched `a`,
      then read against document 5 §5
- [ ] **Compare β_a across all three grids** (layer 1, layer 2, both) — a null in all
      three is a fact about the architecture; an effect only under the network-wide
      constraint is a result no single-layer grid could have produced
