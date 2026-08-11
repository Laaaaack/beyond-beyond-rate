# Both-layer v3 — the same factorial, on the full hidden network (design, calibration, grid result)

**Status (2026-08-09): COMPLETE, INCLUDING THE REGRESSIONS.** We trained 48 models
(24 for each arm: 4 `k` × 2 floor × **3** seeds, 1250 epochs) on the remote server and
retrieved them on 2026-08-07. We ran all 8 perturbation sweeps at **three injection
sites** (`l1`, `l2`, `both`) on 2026-08-08. We ran the availability and usage
measurement and the Phase 1 regressions on 2026-08-09. **The design acceptance check
passes in both arms, and the finding that made this work necessary is confirmed.**
ρ(`s1`, `s2`) came back at **+0.977 / +0.944**, against **−0.829 / +0.203** in v1. The
pooled confound fell from **−0.879** to **−0.199** in the no-delay arm (§6a to §6c).
**H1′ also got no support at the network level.** β_a is null at each injection site in
both arms, β_s carries the effect, and the controlled floor contrast is positive and
significant in both arms (§6h). The **delay arm is primary**. Its readout gap, measured
again, is +0.076, against +0.285 for the no-delay arm (§6i). **The three-way comparison
of β_a is complete, and it is the main result (§6j). β_a is significant in 3 of 4
single-layer cells and in 0 of 2 both-layer cells. The both-layer intervals exclude the
single-layer estimates. Therefore the anti-H1′ signal exists only when *one* layer is
constrained. This is the layer-2 compensation of §1 point 1 in the form of a
coefficient.**

> **Execution convention — what we did.** The local machine (RTX 2050, 4 GB) ran the
> **probes only** (§5 steps 1 and 1b). The full 1250-epoch grids ran on the remote
> server with `QUICK_TEST = False`. Both arms ran **3 seeds (42/43/44), and not the 2
> seeds that this document first planned**. That is 24 models for each arm, and not 16.
> All other properties of the grids are as this document specifies:
> `CEILING_STRENGTH = 10`, `FLOOR_STRENGTH = 1`, `WARMUP_EPOCHS = 20` in both arms, and
> `SETTLE_EPOCHS = 150` in the delay arm only.

> **The one item to remember from this page.** The selection of the best model on the
> *task* validation loss can save a checkpoint from **before the sparsity constraint
> binds**. Each sparsity statistic from such a checkpoint is then wrong in the direction
> of "the penalty does not work". This caused one incorrect diagnosis in this experiment
> (§5 step 1b), and it affects 2 of the 16 cells in the **completed** 1st-layer delay
> grid, with no visible sign. It then tried to happen again in **3 of the 24 cells** of
> the delay grid of this document. `SETTLE_EPOCHS = 150` caught all three (§6d). Before
> you trust any cell, compare `argmin(val_loss)` against the run length in its training
> log.

The 1st-layer v3 factorial finished on 2026-07-29 with a null result. We built the
2nd-layer factorial on 2026-07-30 to ask whether that null result is a fact about
temporal coding or a fact about layer 1. Both experiments constrain **one** layer.
Neither of them asks what happens when the *network* becomes sparse. The measurement
below shows that this is a third question, and not a different form of the first two.

**The calibration finding that makes the full experiment necessary:** under the
**layer-1-only** penalty of v1, the silent fractions of the two layers move in
**opposite** directions in the no-delay arm — ρ(`s1`, `s2`) = **−0.829**. Layer 2
becomes *less* silent when layer 1 moves toward silence. A layer-1 penalty does not make
the network selective. It exchanges the silence of layer 1 for the activity of layer 2.
Pooled over both layers, `a` and `s` are then confounded at **−0.879**. This is far past
the acceptance threshold of |ρ| ≤ 0.5. **Owner:** _(you)_

**This is document 7 of 7. It is a sibling of documents 5 and 6, and not a successor.
Read document 5 first.**

| # | Document | What it is |
|---|---|---|
| 1 | [sparse_network.md](sparse_network.md) | the question and the conceptual landscape — **start there** |
| 2 | [sparse_network_test_progress.md](../legacy/sn_progress/sparse_network_test_progress.md) | v1 execution log |
| 3 | [sparse_network_1stLayer_results.md](../legacy/sn_progress/sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](../legacy/sn_progress/sparse_network_test_progress_v2.md) | v2 execution log |
| 5 | [sparse_network_test_progress_v3.md](../legacy/sn_progress/sparse_network_test_progress_v3.md) | **v3 at layer 1** — design, Phase 0, Phase 1 result. Everything here uses it |
| 6 | [sparse_network_2ndLayer_test_progress_v3.md](../legacy/sn_progress/sparse_network_2ndLayer_test_progress_v3.md) | v3 at layer 2 — calibration, design changes |
| 7 | **this file** | v3 at **both** layers — calibration, design changes, status |

Everything in document 5 that is not about *which layer* stays the same here. This
includes the two-variable decomposition of sparsity (§3b), the H1′ and H2 restatement
(§3c), the capacity-matched decoder (§3a), the mechanism of the two penalties (§4b), and
the table for the reading of the outcome (§5). This document records what we had to
**measure again, decide again, or watch as a new risk**, because the target is now two
layers and not one (§1 to §5). It then records **what the two grids produced** (§6).

---

## 1. Why we constrain both layers at the same time

Document 6 §1 says that layer 1 is possibly not where the timing computation of the
network happens. This document makes a different and more basic point: **the two
single-layer grids never make a sparse network.** They make a network with one sparse
layer and one unconstrained layer, and the unconstrained layer is free to compensate.
The measurement in §2 shows that it does exactly that.

There are three results, in the order of their importance:

1. **"Sparser-activity networks" is the phrase of the supervisor, and neither
   single-layer grid makes such a network.** The densest and sparsest no-delay models of
   v1 differ by **6.5×** in the firing of layer 1 (1.21–7.85 spikes/neuron). They differ
   by only **2.8×** across the network (2.37–6.69 pooled), because layer 2 absorbs one
   part of the decrease. The manipulation is **57% weaker** than the primary layer-1
   numbers show.
2. **The compensation is not neutral. It moves against the intent of the
   manipulation.** ρ(`s1`, `s2`) = −0.829 (no-delay) means that the layer-1 penalty
   *decreases* the selectivity of layer 2 while it increases the selectivity of layer 1.
   The Phase 1 verdict of document 5 was that the residual signal is **selectivity**
   (β\* = −0.44, the largest term). Therefore a design in which the selectivity of the
   two layers moves in opposite directions measures a quantity that is partly cancelled.
3. **It gives the layer-1 grid and the layer-2 grid their control.** With all three
   grids complete, the same `k` × floor design exists at layer 1 alone, at layer 2
   alone, and at both layers. If β_a is null in all three grids, the null result is
   about the architecture. If β_a occurs only when the full network is constrained, that
   is a positive result that no single-layer grid can give.

This is a replication in a new **location**, and not a new hypothesis. H1′ and H2 do not
change.

---

## 2. Calibration step 0 — what the network-wide axes look like

**Source:** [layer2_baseline.py](../../exp_sparse_network/v3_analysis/layer2_baseline.py)
writes `v3_analysis/log/layer2_baseline_{arm}.json`, which already records **both**
layers for each checkpoint. We computed the cross-layer and pooled correlations below
from that file, over all 27 frozen v1 checkpoints. We needed no new training and no new
evaluation pass.

We pool over the `(sample, neuron)` pairs across both layers. Both layers have 128
units. Therefore `a_net = (spikes1 + spikes2) / (active1 + active2)`, and `s_net` is the
plain fraction of the pairs. This pooling is exact. It is not an average of two ratios.

### 2a. The primary table

| across the gradient of v1 | no-delay (n=15) | delay (n=12) |
|---|---|---|
| **ρ(`a1`, `s1`)** — the confound inside layer 1 | **−0.943** (p=1.4e−7) | **−0.455** (p=.14) |
| **ρ(`a2`, `s2`)** — the confound inside layer 2 | **+0.829** (p=1.4e−4) | **+0.427** (p=.17) |
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
| **`spikes/neuron` across the network** | **2.37 – 6.69** (2.8×) | **8.17 – 17.20** (2.1×) |

### 2b. Five items that this settles

**1. The layer-1 penalty makes the network less selective at layer 2, and not more.**
ρ(`s1`, `s2`) = **−0.829** in the no-delay arm. This is the most important number in
this document, and it has no equivalent in documents 5 or 6, because neither of them
measured two layers at the same time. In the delay arm the value is **+0.203**. That is
not opposite, but it shows almost no relation. This is the same failure in a weaker
form: the penalty of v1 had no consistent effect on the selectivity of layer 2.

**2. The manipulation across the network is much weaker than the manipulation at layer
1.** We measured this over the same checkpoints:

| | layer 1 span | span across the network | weaker by |
|---|---|---|---|
| `spikes/neuron`, no-delay | 6.50× | **2.83×** | **57%** |
| `spikes/neuron`, delay | 5.51× | **2.11×** | **62%** |
| `a`, no-delay | 3.76× | **2.17×** | **42%** |
| `a`, delay | 4.05× | **1.56×** | **61%** |

Each effect size for one layer in documents 3 and 5 is an effect size for a network that
became sparse in one place and compensated in another place.

**3. The pooled confound is different in each arm, and only the no-delay arm needs a
correction.** At **−0.879**, the no-delay arm fails the threshold of |ρ| ≤ 0.5. At
**+0.042**, the delay arm already passes. Therefore the two arms get different results
from this design, and the scripts say so. In the delay arm, this design adds the
**controlled** contrast between floor-on and floor-off at equal `a`. That is a
manipulated comparison, where the gradient of v1 gave only a correlation. It does not
add decorrelation. This is the same reason why the 1st-layer factorial was necessary in
that arm at ρ = −0.455.

**4. ρ(`a1`, `a2`) is high, and this design will not decrease it.** The observed values
are +0.961 / +0.839, and both controls here act on both layers. Therefore `a1` and `a2`
move together **by construction**. This grid thus cannot attribute an effect to the
activity of one layer and not the other. That is a property of the question, and not a
defect. The two single-layer grids give the attribution. But you must state this fact
wherever you report a β from this grid. Both scripts print ρ(`a1`, `a2`) and
ρ(`s1`, `s2`) as **diagnostics that are explicitly outside the pass/fail check**.

**5. The natural rates of the two layers differ enough to need a different `k` for each
layer.** The ratio `a2`/`a1` at the dense end is about 1.1× in the no-delay arm and
about 2.0× in the delay arm. Refer to §3a.

---

## 3. What changed in the design, and what did not

Both scripts are copies of their 1st-layer siblings, and we compared them against the
2nd-layer ones. The data, the splits, the optimiser, the schedule, the early stopping,
the two penalty *forms* and the acceptance criteria do not change. The parameter set
does not change. Therefore these checkpoints load into the existing evaluation pipeline
exactly as the v1 checkpoints do. We checked this with `strict=True` against both arms —
refer to §4.

**Implemented in:**
[sn_bothLayer_train_noDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_noDelay_v3.py)
and
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py).

**The mechanical change.** `forward(return_hidden=True)` now returns
`(out, hidden1, potential1, hidden2, potential2)`. The forward pass divides into
`_first_hidden`, `_second_hidden` and `_output`. Each part returns the potential of its
layer with its spikes. We charge each penalty **once for each layer and then add the
results**, at the same coefficient that its single-layer sibling used:

```python
loss = task_loss
for layer in (1, 2):
    loss += CEILING_STRENGTH * relu(count[layer] - k[layer]).mean()
    loss += floor_strength   * relu(THETA_MARGIN - peak_u[layer]).mean()
```

We charge each layer at the **full** coefficient and not at one half. This is
intentional. A cell of this grid then applies to layer 1 exactly the pressure of the
1st-layer grid, and to layer 2 exactly the pressure of the 2nd-layer grid. The three
grids are thus comparable cell by cell. The cost is that the *total* penalty gradient is
about two times larger — refer to §3b.

**All artifacts have the tag `v3L12`.** All three grids share the dataset, the arm, `k`,
the floor and the seed. Without different tags (`v3`, `v3L2`, `v3L12`) they make
identical file names, and one grid overwrites another grid with no message. The run tags
are `sparse_whole_{arm}_v3L12_k{k}_floor{f}_seed{seed}`, and each summary row holds
`target_layers: [1, 2]`.

**The log holds the values for each layer *and* the pooled values.** The log for each
epoch holds `a`, `s`, `over_k` and `sub_threshold` separately for each layer. It also
holds the pooled `a_net`, `s_net` and `rate_net`. The progress bar shows `a1 s1 a2 s2`
together, because the most probable failure of this run is that one layer collapses
while the other layer looks healthy.

**The canonical keys of the summary hold the network values.**
`spikes_per_active_neuron`, `silent_fraction` and `spikes_per_neuron` have no suffix and
hold the pooled values. The explicit `_l1`, `_l2` and `_net` keys are also present. This
makes analysis code that was written against the single-layer summaries read the
quantity that this experiment manipulates, and not one layer of it by accident. This is
intentional, both scripts document it, and you must not point these keys at a single
layer.

### 3a. The one important design change: the row axis has a different scale for each layer

`CEILING_K` gives the levels of **layer 1**. The levels of layer 2 are
`CEILING_K_LAYER2_RATIO` times those levels. One row control then gives an equal
*relative* decrease at both layers, and not an equal absolute budget.

| arm | natural `a` (L1 / L2) | ratio | `k1` | `k2` | models |
|---|---|---|---|---|---|
| no-delay | 3.06–11.49 / 7.49–12.46 | **1.0** | `[1, 2, 4, 8]` | `[1, 2, 4, 8]` | 16 |
| delay | 3.82–15.47 / **20.71–30.44** | **2.0** | `[1, 2, 4, 8]` | **`[2, 4, 8, 16]`** | 16 |

The no-delay ranges overlap for the most part. There, an equal absolute budget already
*is* an equal relative budget, and the ratio has no effect. But the ratio is still
necessary, because layer 2 of the delay arm fires about two times more. A shared `k`
would give a small decrease to layer 1 and a very large decrease to layer 2 in the same
cell.

This also solves, at a lower cost, the problem that the 2nd-layer delay script solved
with an extra grid row. That script kept a shared budget and added `k = 16` to fix the
top of the layer-2 axis near its natural rate. That needed 20 models and about 46 h.
Here the ratio does the same work inside 4 rows. The top row is `k1 = 8, k2 = 16`, just
below the natural range of each layer. **This needs 16 models for each arm, and not
20.**

*This is the one decision in this document that is a judgement and not a measurement.*
The alternative is one shared absolute `k`, which is the most literal form of "the same
mechanism on both layers". We considered it and rejected it, because of the 2× rate
difference in the delay arm. You can reverse the decision with one line
(`CEILING_K_LAYER2_RATIO = 1.0`), but you then need the 5th row again in that arm.

### 3b. Inherited, but NOT yet validated — the reason for the probe

| constant | value (no-delay / delay) | status |
|---|---|---|
| `NATURAL_SPIKES_PER_NEURON` | {L1 7.85, L2 5.53} / {L1 11.66, L2 23.93} | **measured for each layer and each arm** |
| `CEILING_K_LAYER2_RATIO` | 1.0 / 2.0 | **set from the measured natural `a`** |
| the observed ρ(`a`, `s`) baselines | −0.943 / +0.829 / −0.879 and −0.455 / +0.427 / +0.042 | **measured (§2a)** |
| `CEILING_STRENGTH` | 10.0 | **inherited — not validated with two penalties** |
| `FLOOR_STRENGTH` | 1.0 | **inherited — not validated with two penalties** |
| `WARMUP_EPOCHS` | 20 | **inherited — not validated with two penalties** |

For this reason, both scripts have `QUICK_TEST = True` when they ship. We calibrated
each inherited constant against the activity of a *single* layer, and this configuration
is new in a way that neither sibling was:

- **`CEILING_STRENGTH` has arguments in both directions. Therefore you must measure it
  and not reason about it.** `fc1` now receives ceiling pressure two times: directly from
  the layer-1 term, and through the surrogate from the layer-2 term. §4 measures the
  second path at |grad| 0.81 and 2.05 in the two arms. This gives an argument for a
  *smaller* value. But at layer 1, even a strength of 10 left 41% of the pairs above `k`
  at `k = 1`, and layer 2 of the delay arm starts two times denser. This gives an
  argument for a *larger* value. The two `over_k` columns show which of the two happens,
  for each layer. **If one layer stays high and the other does not, correct that
  asymmetry, and not the overall level.**
- **`WARMUP_EPOCHS = 20` guards an absorbing failure, and this script is the most
  exposed of the three.** The ceiling has its minimum at `count = 0`. Therefore, in the
  floor-off column, only the task loss opposes silence — now at *both* layers at the same
  time. The two failures also accumulate: an empty layer 1 also makes the gradient path
  of the layer-2 floor zero (§3c). Thus the instrument that must rescue layer 2 dies at
  the same moment. The corner at `k = 1`, floor = 0 is the cell that shows this.
  Therefore its check is **criterion 1** of the probe, and not criterion 4.

### 3c. The reachability edge of the floor at layer 2 now works in both directions

`∂potential2 / ∂W2 = psp(hidden1)` (no-delay) or `psp(delay1(hidden1))` (delay). It
needs no surrogate, **but it is exactly zero for any sample whose full layer 1 is
silent**. Such a pair is reachable only back through the surrogate of `fc1`. Document 6
§2c measured this condition at **0.00% of the samples** across all 27 v1 checkpoints, and
recorded it as a known edge with a known trigger.

This design moves the edge in **both** directions, and both directions are important:

- **In the floor-ON column, the edge now has an active guard.** The layer-1 floor
  prevents silence at layer 1, which is the precondition for the edge case. Neither
  single-layer script had that protection. The 2nd-layer script left layer 1 completely
  unconstrained. Therefore the reachability of its floor depended on a layer that nothing
  defended.
- **In the floor-OFF column, the edge is worse than in either sibling.** The ceiling now
  pushes both layers toward silence, and nothing opposes it. Therefore an empty layer 1
  is more probable here than anywhere before. This is the same corner as in §3b: `k` at
  its lowest value, and the floor at 0.

---

## 4. Verification of the implementation — what we checked

We ran the checks against both arms on the GPU. They needed about 1 min and no training.
There are twenty checks for each arm, and all of them pass.

| # | check | no-delay | delay |
|---|---|---|---|
| 1 | `forward(return_hidden=True)` returns **5** tensors, `hidden1 ≠ hidden2` | ✔ (L1 3217 / L2 7362 spikes on random input) | ✔ (9122 / 18721) |
| 2 | a v1 checkpoint loads with `strict=True` — the parameter set does not change | ✔ 8 tensors | ✔ 10 tensors |
| 3 | the ceiling at L1 gives a nonzero gradient on `fc1` | \|grad\| 1.595 | 3.385 |
| 4 | the floor at L1 gives a nonzero gradient on `fc1` | 1.104 | 0.593 |
| 5 | the ceiling at L2 gives a nonzero gradient on `fc2` | 1.351 | 7.455 |
| 6 | the floor at L2 gives a nonzero gradient on `fc2` | 0.811 | 0.427 |
| 7 | **the L2 ceiling also reaches `fc1`** — the "`fc1` receives two compressions" claim of §3b, now measured and not only stated | **0.808** | **2.046** |
| 8 | **the edge case of §3c repeats**: a forced silent layer 1 makes `∂(floor2)/∂W2` **exactly 0** | ✔ 0.000000 | ✔ 0.000000 |
| 9 | `pool_layer_stats` agrees with a direct pooled computation to 1e−9, on all three axes | ✔ | ✔ |
| 10 | the recorded ceiling of layer 2 is the **scaled** one (`k1 = 1` → `k2`) | 1.0 | **2.0** |
| 11 | the summary emits each documented key; the canonical keys with no suffix hold the **network** pool | ✔ | ✔ |
| 12 | the log for each epoch holds both axes at both layers and the pool (and `delay_mean`) | ✔ | ✔ |
| 13 | both report functions run; the probe resolves to the 4 corners at 400 epochs | ✔ | ✔ |

Rows 7 and 8 are the two checks that a reading of the code cannot settle. Row 7
quantifies the argument that `CEILING_STRENGTH` is possibly now too *strong* for layer
1. Row 8 confirms that the failure mode is real and reachable. Therefore criterion 1 of
§3b guards something that exists.

---

## 5. Execution log — probes, calibration, and the start of the grids

We wrote this section as the work happened. Therefore it reads as a plan in some places.
Steps 1, 1b and 2 are complete. Step 3 is half complete. **§6 gives what the grids
produced.** §6g and the open boxes of §8 collect the work that is still outstanding.

### Step 1 — corner probe, both arms — **DONE, 2026-07-31 (5.9 h, 2×2×2 = 8 models)**

We ran `k1 ∈ {1, 8}` × floor ∈ {0, 1}, seed 42, 400 epochs. Layer 2 thus saw `{1, 8}`
(no-delay) or `{2, 16}` (delay). The runs were sequential on one 4 GB GPU, and took
2 h 56 m for each arm. The probe artifacts have the `_probe` suffix, and we touched
nothing real.

We changed the **order** of the acceptance criteria from documents 5 and 6. The survival
check is now first, for the reason in §3b:

1. **Both layers survive the corner at `k = 1`, floor = 0.** If `val_acc` is at chance
   (5%) and `silent → 100%` at *either* layer, `WARMUP_EPOCHS` is too short. Increase it
   before you change anything else;
2. `silent_fraction` stays about constant when `k` changes *inside* a column, at each
   layer;
3. `silent_fraction` differs by **20 points or more** *between* the columns at equal
   `k`, **at both layers**. A floor that binds at one layer only lets the column control
   do one half of its work. It also makes the pooled `s_net` axis a statement about one
   layer;
4. `spikes_per_neuron` is not higher than the natural rate of that layer — 7.85 / 5.53
   (no-delay), 11.66 / 23.93 (delay);
5. the clean accuracy is well above chance in each corner.

Both scripts print criteria 3 and 5 automatically. `report_manipulation_check` gives the
separation for each `k` **and each layer**, and it marks a value below 20 points as
`TOO SMALL`.

**Read the probe for the manipulation only.** Each v1 model was still better at epoch
1250. Therefore 400 epochs is a different regime, and not a smaller form of the same
regime. Document 5 also showed that penalties which we calibrate at fewer epochs
**become dense again** by the end of a full run. The `a` value that we get sits *above*
the numbers of the probe.

#### Probe result — no-delay arm

| `k1`/`k2` | floor | acc | `a1` | `s1` | sp1 | over-`k1` | `a2` | `s2` | sp2 | over-`k2` | `a_net` | `s_net` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 / 1 | 0 | .441 | 2.48 | **86.89%** | 0.33 | 6.2% | 2.59 | **70.73%** | 0.76 | 18.0% | 2.56 | 78.81% |
| 1 / 1 | **1** | .538 | 1.84 | **23.78%** | 1.40 | 33.1% | 2.45 | **13.04%** | 2.13 | 56.3% | 2.17 | 18.41% |
| 8 / 8 | 0 | .502 | 3.57 | **59.12%** | 1.46 | 3.3% | 5.05 | **55.44%** | 2.25 | 6.9% | 4.34 | 57.28% |
| 8 / 8 | **1** | .555 | 3.91 | **6.91%** | 3.64 | 6.1% | 5.09 | **2.34%** | 4.97 | 11.3% | 4.52 | 4.63% |

| # | criterion | result |
|---|---|---|
| 1 | both layers survive `k=1`, floor = 0 | **PASS** — acc .441 against a chance of .05; `s1` 86.9%, `s2` 70.7%, and neither layer at 100%. `WARMUP_EPOCHS = 20` transfers to the two-penalty configuration |
| 2 | `s` is about constant when `k` changes inside a column | **partial** — floor-off drifts 86.89→59.12 at L1 (**27.8 pts**) and 70.73→55.44 at L2 (15.3 pts); floor-on drifts 16.9 / 10.7 pts. This is the same violation as in the 1st-layer probe (23.6 pts), and it is a little worse |
| 3 | the floor separates `s` by 20 pts or more at **both** layers | **PASS, with a very large margin** — **+63.1 / +57.7** at `k=1`, and **+52.2 / +53.1** at `k=8`. Each cell is OK |
| 4 | the firing is not higher than the natural rate of each layer | **PASS** — sp1 ≤ 3.64 against 7.85, and sp2 ≤ 4.97 against 5.53. No `!` flags |
| 5 | the clean accuracy is well above chance | **PASS** — .441–.555 against a chance of .05, and against .49–.59 for no-delay v1 |

**Layer 1 behaves almost identically to its single-layer sibling. The "`fc1` receives two
compressions" risk did not happen.** Against the 1st-layer-only probe at the same
settings (document 5):

| cell | `a1` both / 1st-only | `s1` both / 1st-only | over-`k` both / 1st-only |
|---|---|---|---|
| `k=1`, floor 0 | 2.48 / 2.69 | 86.9% / 80.9% | 6.2% / 9.4% |
| `k=1`, floor 1 | 1.84 / 1.91 | 23.8% / 21.6% | 33.1% / 34.2% |
| `k=8`, floor 0 | 3.57 / 3.57 | 59.1% / 57.3% | 3.3% / 2.8% |
| `k=8`, floor 1 | 3.91 / 3.81 | 6.9% / 6.7% | 6.1% / 5.2% |

The largest difference is +6.0 points of `s1` at the floor-off corner at `k=1`. Each
other value is within about 2 points or 0.1 spikes. Therefore, in this arm, the layer-2
penalty constrains layer 2 and **does not** disturb the work of the layer-1 penalty.
This is the best possible result for a design that must be comparable to its siblings
cell by cell.

**Verdict: the no-delay arm is ready for its 16-model grid, with no change.**

#### Probe result — delay arm

| `k1`/`k2` | floor | acc | `a1` | `s1` | sp1 | over-`k1` | `a2` | `s2` | sp2 | over-`k2` | `a_net` | `s_net` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 / 2 | 0 | .717 | 4.20 | **75.83%** | 1.02 | 14.4% | 4.41 | **50.82%** | 2.17 | 31.0% | 4.34 | 63.32% |
| 1 / 2 | **1** | .758 | 5.14 | **18.11%** | 4.21 | **62.2%** | 5.61 | **7.33%** | 5.20 | **65.8%** | 5.39 | 12.72% |
| 8 / 16 | 0 | .765 | 4.73 | **53.93%** | 2.18 | 6.6% | 9.10 | **42.83%** | 5.21 | 5.4% | 7.15 | 48.38% |
| 8 / 16 | **1** | .786 | 7.50 | **5.53%** | 7.08 | 30.6% | 10.50 | **1.70%** | 10.32 | 16.6% | 9.03 | 3.61% |

Criteria 1, 3, 4 and 5 all pass, and some of them pass with a large margin. The network
survives at `k=1` floor 0 (acc **.717**). The separation is **+57.7 / +43.5** and
**+48.4 / +41.1** points. There is no inflation (sp1 ≤ 7.08 against 11.66; sp2 ≤ 10.32
against 23.93). The accuracy is **.717–.786**, which is *better* than the 1st-layer arm
gave at strength 10 (.685–.771). Criterion 2 has a partial violation, as everywhere else
(21.9 pts of floor-off drift at L1).

**But the ceiling almost does not bind in the floor-on column, and this is the finding.**
At `k1 = 1` with the floor on, **62.2%** of the layer-1 pairs and **65.8%** of the
layer-2 pairs still fire above their ceiling. The result is a compressed row axis:

| span of `a` across `k1` = 1 → 8 | no-delay | delay |
|---|---|---|
| layer 1, floor off / on | 1.44× / **2.13×** | 1.13× / **1.46×** |
| layer 2, floor off / on | 1.95× / 2.08× | 2.06× / 1.87× |
| **network, floor off / on** | **1.70× / 2.08×** | **1.65× / 1.67×** |

The floor-on axis of layer 1 moves only **1.46×** in the delay arm. It moves 2.13× in
the no-delay arm, and the observed span of this arm in v1 was 4.05×. Document 5 met
exactly this problem at layer 1 ("a factorial whose H1 axis moves 1.4× … will estimate
β_a badly"). It corrected the problem when it moved `CEILING_STRENGTH` from 3 to 10.

**The mechanism is visible, and it is new.** Compare layer 1 against the 1st-layer-only
delay calibration at strength 10, `k = 1`:

| | `a1` | `s1` | sp1 | over-`k` |
|---|---|---|---|---|
| 1st-layer only, floor 0 | 4.00 | 72.1% | 1.12 | 17% |
| **both-layer, floor 0** | **4.20** | 75.8% | 1.02 | 14.4% |
| 1st-layer only, floor 1 | **2.66** | 22.6% | 2.06 | 41% |
| **both-layer, floor 1** | **5.14** | 18.1% | 4.21 | **62.2%** |

The floor-off column agrees closely with its sibling. The floor-**on** column does not.
Layer 1 becomes **1.9× denser** (`a1` 2.66 → 5.14) than when only layer 1 was
constrained. The probable mechanism is as follows. `NumSpikes` holds the *output* firing
rate constant. With a ceiling also on layer 2, layer 1 must drive layer 2 more to keep
the output at its target. The layer-1 ceiling at strength 10 is soft enough to permit
this. **A constraint on layer 2 partly cancels the layer-1 ceiling.** This interaction
has no equivalent in either single-layer script, and it is the specific problem that
this arm must have calibrated again.

**Verdict: the delay arm needs one ceiling calibration run before the grid** — refer to
step 1b. Everything else in that arm passed.

#### What the probe does *not* settle: the design acceptance check

Both scripts printed the check. At n = 4 **you must not read it**: PASS at −0.400 in the
no-delay arm, and FAIL at −0.400 to −1.000 in the delay arm. With only two `k` levels,
the floor column controls the correlation, and a Spearman on 4 points is degenerate. The
value of −1.000 at layer 1 means only that 4 points were monotone. The true check needs
the grid with 4 levels and 2 seeds. The probe-stage figure of the 1st-layer experiment
was −0.238, against a final grid value of −0.053.

There is one diagnostic that is already informative. **ρ(`s1`, `s2`) came back at +1.000
in both arms, against −0.829 (no-delay) and +0.203 (delay) in v1.** The column control
moves the selectivity of both layers together. That is exactly the failure of the
single-layer design that §1 point 2 describes. The magnitude is degenerate at n = 4, but
the change of sign is the central claim of the design, and it has the correct direction.

### Step 1b — ceiling calibration for the delay arm — **RUN, AND IT DIAGNOSED A DIFFERENT PROBLEM**

We ran `k1 = 1`, both floor columns, 400 ep, seed 42, at `CEILING_STRENGTH = 20`, under
`_ceilcal_str20` tags. The strength-10 probe thus stayed on disk (about 1.5 h).

| floor | strength | acc | `a1` | `s1` | over-`k1` | `a2` | `s2` | over-`k2` |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | .717 | 4.20 | 75.8% | 14.4% | 4.41 | 50.8% | 31.0% |
| 0 | **20** | **.564** | 2.42 | 85.6% | 7.2% | 2.78 | 51.5% | 20.2% |
| 1 | 10 | .758 | 5.14 | 18.1% | 62.2% | 5.61 | 7.3% | 65.8% |
| 1 | **20** | **.698** | 3.16 | 31.4% | 38.8% | 3.70 | 11.4% | 50.7% |

A strength of 20 does make the ceiling tighter. But it costs **.153** of accuracy in the
floor-off column (.717 → .564). That is worse than the .13 that the step from 3 to 10
cost at layer 1, and it puts that corner far below the .78–.88 of v1. This table alone
would say "20 is too expensive. Keep 10, and report a soft row axis".

**But the ceiling strength is not the true problem, and this table is not the important
evidence.**

#### The finding: in the floor-ON column of the delay arm, the saved checkpoint is older than the constraint

The selection of the best model uses the **task** validation loss. This was an
intentional decision in document 5: "the saved checkpoint is chosen on the task rather
than on how well the constraint is satisfied". In the floor-on cells of the delay arm,
the task validation loss is at its best about 20 epochs after the penalties start, and it
never improves again. Therefore the early stopping operates at about epoch 340, and **the
program saves and measures the checkpoint from epoch 37 to 45**. That network had the
constraint for about twenty epochs.

The constraint state at the saved epoch and at the last epoch that we trained (train
set):

| cell | saved / last epoch | `a1` | over-`k1` | `a2` | over-`k2` |
|---|---|---|---|---|---|
| `k1` floor 1, str 10 | **41 / 341** | 4.94 → **2.27** | 62.5% → **36.9%** | 5.48 → 3.31 | 66.4% → **48.8%** |
| `k8` floor 1, str 10 | **37 / 337** | 7.29 → **4.07** | 28.9% → **7.2%** | 10.23 → 8.15 | 14.4% → **4.5%** |
| `k1` floor 1, str 20 | **45 / 345** | 3.08 → 1.56 | 39.5% → 21.9% | 3.60 → 2.44 | 51.8% → 31.8% |
| `k1` floor 0, str 10 | 394 / 399 | 4.04 → 4.04 | 14.8% → 14.6% | — | — |
| `k8` floor 0, str 10 | 385 / 399 | 4.60 → 4.59 | 6.2% → 6.3% | — | — |

**Therefore the "62.2% ceiling leak" above is an artifact of the model selection.** The
same run at strength 10, trained to the end, sits at **36.9%**. That is inside the 30–40%
band of the 1st-layer grid. The row axis is also *wider* at the last epoch (`a1` 2.27 →
4.07 across `k`, 1.79×) than at the saved checkpoints (4.94 → 7.29, 1.48×).
`CEILING_STRENGTH = 10` was never the problem. An increase to 20 buys a tighter
constraint and pays .153 of accuracy for a correction that the run did not need.

Only the **floor-on** cells have this problem. The floor-off cells select at epoch 385 to
397, and their constraint state is identical at the best epoch and the last epoch.

#### The no-delay arm does not have the problem — its grid can continue

| cell | saved / last | `a1` at saved → last | over-`k1` saved → last |
|---|---|---|---|
| `k1` floor 0 | 326 / 399 | 2.54 → 2.40 | 6.8% → 6.6% |
| `k1` floor 1 | 398 / 399 | 1.84 → 1.84 | 33.8% → 34.2% |
| `k8` floor 0 | 396 / 399 | 3.52 → 3.48 | 3.2% → 3.0% |
| `k8` floor 1 | 398 / 399 | 3.80 → 3.79 | 5.2% → 5.1% |

Each cell selects late, and the constraint state does not change between the saved epoch
and the final epoch. No action is necessary.

#### The problem also touched the completed 1st-layer v3 delay grid, in 2 of 16 cells

We audited `best_epoch / n_epochs` across that grid (the Phase 1 result of document 5).
14 of the 16 cells select at a fraction of 0.71 to 1.00, and are thus healthy. The two
exceptions are both at `k = 1` with the floor on: `seed42` at 433/734 (0.59) and
**`seed43` at 48/349 (0.14)**. This is the same fault. The primary conclusion of that
grid comes from 16 models, and 14 of them are clean. The conclusion is thus probably
reliable. But the two `k=1` floor-on cells are less constrained than the design intended,
and you must note this against any new reading of β_a there.

#### The candidate correction, and why it costs almost nothing

Follow the best validation loss only from `warmup_epochs + settle`. The `settle` value
must be long enough for the constraint to bind (about 150 epochs). This keeps the
selection on the **task** loss, which is the design intent of document 5. But it refuses
checkpoints from before the manipulation exists.

The correction has almost no effect on healthy runs. If we apply it to the completed
1st-layer delay grid at 1250 epochs, it changes the chosen checkpoint in **1 of 16
cells** (the 48/349 one). The other fifteen cells already select at epoch 433 or later.
Therefore it repairs the bad cells and does not disturb the comparison with the grids
that are already complete.

#### Resolution — adopted 2026-07-31

`SETTLE_EPOCHS = 150` is now in
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py).
The program holds the tracking of the best model *and* the early stopping until
`warmup_epochs + settle_epochs`, that is, until epoch 170. `CEILING_STRENGTH` stays at
**10**. We keep the strength-20 run on disk as evidence, but we do not adopt it.

There is one useful side effect. At 400 epochs the early stopping can no longer operate
(170 + 300 > 400). Therefore the probe trains each cell to its full length. In the
1250-epoch grid the early stopping can operate from epoch 470.

**The two arms now have different selection policies, and this is intentional.** The
no-delay script does *not* have the window, for two reasons. First, we measured that the
fault is absent there: each probe cell selects at epoch 326 to 398 of 400 with an
unchanged constraint state. Second, its 16-model grid was already running when we found
the fault. To add the window would make the script different from the artifacts that it
produces. The no-delay script has a comment that records the asymmetry, the evidence, and
an instruction to add the window if that arm ever runs again. There the window is free,
because a settle window of 150 would not have changed any cell that the arm produced.

#### Delay re-probe with the settle window — **DONE, 2026-07-31 (2 h 23 m)**

We ran the 4 corners again at `CEILING_STRENGTH = 10` with `SETTLE_EPOCHS = 150`. The
pre-settle artifacts stay under `_probe_presettle` tags, because the re-probe uses the
same `_probe` names.

| `k1`/`k2` | floor | acc | `a1` | `s1` | over-`k1` | `a2` | `s2` | over-`k2` | `a_net` | `s_net` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 / 2 | 0 | .717 | 4.20 | 75.8% | 14.4% | 4.41 | 50.8% | 31.0% | 4.34 | 63.3% |
| 1 / 2 | **1** | **.762** | 2.33 | 24.9% | **36.0%** | 3.37 | 9.9% | **49.1%** | 2.90 | 17.4% |
| 8 / 16 | 0 | .765 | 4.73 | 53.9% | 6.6% | 9.10 | 42.8% | 5.4% | 7.15 | 48.4% |
| 8 / 16 | **1** | **.806** | 4.24 | 7.4% | **8.3%** | 8.32 | 0.9% | **4.9%** | 6.35 | 4.2% |

**The window does exactly its function, and nothing more.** The selection now lands at
epoch 341 to 394 in each cell, and no cell stops early. All four cells train the full 400
epochs. The two floor-**off** cells are identical to the pre-settle run, byte for byte.
They are the control: they already selected late, thus the change could not touch them,
and it did not.

The floor-on cell at `k1 = 1` is the proof. Its global `argmin(val_loss)` is still at
epoch 41, because the task loss truly never recovers. But the rule now refuses that
checkpoint and takes epoch **341**:

| `k1=1`, floor 1 | pre-settle (saved ep 41) | post-settle (saved ep 341) |
|---|---|---|
| clean acc | .758 | **.762** |
| `a1` | 5.14 | **2.33** |
| over-`k1` | **62.2%** | **36.0%** |
| over-`k2` | 65.8% | **49.1%** |

**The row axis, which is what the fault was costing:**

| span of `a`, `k1` 1 → 8, floor ON | pre-settle | post-settle |
|---|---|---|
| layer 1 | 1.46× | **1.82×** |
| layer 2 | 1.87× | **2.47×** |
| **network** | 1.67× | **2.19×** |

The network span of 2.19× is now above the final 2.18× of the completed 1st-layer delay
grid, and it is near the 2.08× of the no-delay arm. The floor-off spans do not change, by
construction.

**The correction cost nothing.** The clean accuracy *increased* in both affected cells
(.758 → .762 and .786 → .806). The range is now **.717–.806**, against .78–.88 for v1.
The top of the probe is thus inside the range of v1, and before the correction it was
below that range. There is no inflation at either layer (sp1 ≤ 3.93 against 11.66;
sp2 ≤ 8.25 against 23.93). The floor still separates `s` by **+51.0/+41.0** and
**+46.6/+41.9** points.

**A residual limitation, which we state and do not hide.** The ceiling of layer 2 still
leaks **49.1%** at the tightest row (`k2 = 2`). That is worse than the 36.0% of layer 1,
and worse than the 41% of the 1st-layer delay grid. Therefore the row axis of layer 2 is
the softer of the two. Document 5 accepted the same property in its own words: "a soft,
manipulated axis with real range, not a pinned one". The span that this property
compromised has now recovered. Thus this is a limitation to report with β_a, and not a
blocker. At `k2 = 16` the leak is 4.9%.

**Verdict: the delay arm is ready for its grid**, at `CEILING_STRENGTH = 10` with
`SETTLE_EPOCHS = 150`.

### Step 2 — the grids — **DONE, retrieved 2026-08-07 (48 models, remote server)**

| arm | models as planned | **models as run** | what it gives |
|---|---|---|---|
| no-delay | 4 `k` × 2 floor × 2 seeds = 16 | **4 × 2 × 3 = 24** | it breaks a pooled confound of **−0.879** |
| delay | 4 × 2 × 2 = 16 | **4 × 2 × 3 = 24** | the pooled baseline (+0.042) already passes. It gives the **controlled contrast**, and a direct comparison against the completed 1st-layer delay grid |

Both arms ran **3 seeds (42/43/44)**, and not the 2 seeds that this document planned.
This matches the seed count of the 1st-layer and 2nd-layer grids. That is the only
difference from the specification above, and it gives more statistical power (n = 24 for
each arm for the acceptance correlations, against the n = 16 that this document costed).
All runs used 1250 epochs and all the constants of §3b, with `SETTLE_EPOCHS = 150` in the
delay arm only.

Artifacts: `sn_log/sparse_whole_{arm}_v3L12_train_summary.json` (24 rows each) and 48
training logs with one row for each epoch, under the same prefix. **§6 gives the
results.**

**The choice of the primary arm is still open, exactly as in document 6 §4.** The grids
did not settle it, because we have not made the measurement that it depends on (the
second half of step 3).
The delay-arm decision of document 5 came from two Phase 0 measurements of the
*dependent* variable: the readout-efficiency gap (+.300 against +.026), and the deletion
control, which was not different from the timing probe in the no-delay arm (+0.936
against +0.939). We measured both **at layer 1 only**. To decide from them here would
carry a constant across a layer boundary, and this calibration exists to prevent that
error.

But one consideration is specific to this document, and it selects the **no-delay** arm.
That arm is the only one whose pooled confound truly fails (−0.879). Therefore it is the
arm where this design has something to break, and not only something to control. Against
that, the delay arm has a **completed** 1st-layer grid at 1250 epochs for a comparison.
That comparison is §1 point 3. Decide after step 3.

### Step 3 — move the measurement to the correct layer — **HALF DONE**

This is the largest piece of outstanding work. It is shared with document 6, and it is
**not** optional: the grids have no interpretation without it. The perturbation half is
complete. The availability and usage half has not started.

| what | files | change needed | status |
|---|---|---|---|
| perturbation injection site | 8 new `*_bothLayer_evalOnly_*_v3.py` scripts under `{jitter,shift,shd,deletion}/` | run at `hidden1`, `hidden2` **and both** | **DONE** — `SITES = ("l1", "l2", "both")`, each checkpoint swept at all three sites |
| perturbation window | the same 8 scripts, `SUPPORT_BINS_L1` / `SUPPORT_BINS_L2` | for each layer **and** each arm: L1 `[0, 88)` in both arms; L2 `[0, 90)` no-delay, **`[0, 160)`** delay | **DONE** — relocation and jitter hold the per-layer dictionary; shift and deletion hold no window, as required |
| availability decode | `v3_analysis/hidden_channel_decode.py` | probe both layers; use a window for each layer and each arm | **DONE** — the `GRID` control selects the layer set; the views are `l1`/`l2`/`net` |
| usage and deletion control | `v3_analysis/phase1_measure.py` | the same | **DONE** — `SITES = ("l1", "l2", "both")`, relocation windows for each layer, deletion not clipped |

There are three traps. The evaluation scripts now **discharge** the first two. The third
one is a live design question, and the sweeps answered it empirically (§6e):

- **The window correction is not the same for all perturbations.** Relocation and jitter
  select a destination bin, and they must respect the support. **Shift and deletion must
  stay as they are.** If you clip a rigid translation, the spikes collect at the edge and
  merge. This destroys the count of each neuron and adds a rate insult that does not
  exist now.
- **`delay1` moves from downstream to upstream with the probe site.** A layer-1 probe
  correctly ignores it. A layer-2 probe **must apply it**. An error here measures the
  wrong tensor and raises no error message.
  [layer2_baseline.py](../../exp_sparse_network/v3_analysis/layer2_baseline.py) does this
  correctly, and it is the reference.
- **A dependent variable for the full network is a design decision, and not a given.**
  The independent variables here are pooled. The dependent variables do not have to be
  pooled. A perturbation at both layers at the same time is a *different* and harder
  insult than a perturbation at one layer. Therefore report the usage and the
  availability for each layer first. Treat any pooled dependent variable as an addition,
  and not as a replacement. **The sweeps settled this empirically, and they support the
  careful reading** — refer to §6e.

---

## 6. The grid — what came back

We trained 48 models, 24 for each arm (4 `k` × 2 floor × 3 seeds, 1250 epochs) on the
remote server, and retrieved them on 2026-08-07. We ran all eight perturbation sweeps at
all three sites on 2026-08-08. Everything below comes from the **test** set, from the
evaluation files `{shd,jitter,shift,deletion}/log/sparse_whole_{arm}_v3L12_*_eval.json`.
The `per_setup` rows are the mean over the 3 seeds of each cell. The `per_checkpoint`
rows are the 24 individual models that give the correlations of each arm.

**Everything in this section is descriptive.** We fitted no β, because the regression
scripts did not yet point at this grid (§5 step 3). Read this section as "the
manipulation worked, and this is what it produced". Do not read it as an answer to H1′.

### 6a. The design acceptance check — it passes, at all three levels, in both arms

n = 24 for each arm, Pearson, against the observed baselines of §2a:

| | no-delay: v1 → **grid** | delay: v1 → **grid** |
|---|---|---|
| **ρ(`a_net`, `s_net`)** — the acceptance number | −0.879 → **−0.199** (p=.35) | +0.042 → **+0.040** (p=.85) |
| ρ(`a1`, `s1`) — layer 1 alone | −0.943 → **−0.297** (p=.16) | −0.455 → **−0.001** (p=1.0) |
| ρ(`a2`, `s2`) — layer 2 alone | +0.829 → **−0.122** (p=.57) | +0.427 → **+0.106** (p=.62) |
| **ρ(`s1`, `s2`)** — the finding that made this work necessary | −0.829 → **+0.977** (p=2e−16) | +0.203 → **+0.944** (p=4e−12) |
| ρ(`a1`, `a2`) — excluded by the design | +0.961 → **+0.973** | +0.839 → **+0.893** |

**The pooled confound of the no-delay arm is broken.** It moved from −0.879 to −0.199,
which is well inside |ρ| ≤ 0.5. That was the one function of this design in that arm
(§2b point 3). The confound of the delay arm already passed, and it still passes. In
that arm the design gives the controlled contrast, as specified.

**Both layers also pass individually.** This is more important than it looks. It shows
that the pooled decorrelation is not an artifact of an average of two layers that are
each still confounded. The v1 confound of layer 2 had the *opposite* sign to the confound
of layer 1 (+0.829 against −0.943), and the same two controls moved both to almost zero.

**ρ(`a1`, `a2`) behaved exactly as §2b point 4 said.** The values are +0.973 / +0.893,
which are a little higher than the observed values. We state this again, because it
limits each claim that this grid can support: a β from this grid is a **network-level**
coefficient, and you cannot attribute it to a layer.

### 6b. The finding that made this work necessary is confirmed, and it is not marginal

ρ(`s1`, `s2`) = **+0.977** (no-delay) and **+0.944** (delay). Under the layer-1-only
penalty of v1, the selectivity of the two layers moved in *opposite* directions in the
no-delay arm (−0.829), and had no relation in the delay arm (+0.203). Here the two
layers move together in both arms, at n = 24 and not at the degenerate n = 4 of the
probe.

This discharges §1 point 2. The column control **is** a selectivity manipulation for the
full network. Therefore the pooled `s_net` axis is a statement about the network, and not
about one layer of it. It is also the one result that no single-layer grid can give.

### 6c. The manipulation

**Row axis (the `a` span across `k1` = 1 → 8), against the prediction of the post-settle
probe:**

| span of `a` | no-delay floor OFF / ON | delay floor OFF / ON |
|---|---|---|
| layer 1 | 1.79× / **2.15×** | **1.48×** / 1.82× |
| layer 2 | 2.22× / 2.04× | 2.37× / 2.58× |
| **network** | **1.95× / 2.07×** | **1.87× / 2.25×** |

The floor-on layer-1 axis of the delay arm landed at **1.82×**. The re-probe predicted
this value to two decimal places. Its network axis landed at 2.25×, against the 2.19× of
the probe. The probe was a good instrument.

**One row is not monotone.** The **floor-OFF layer 1** of the delay arm runs 3.54 → 3.01
→ 3.31 → 4.45 across `k1`. That is, it falls at `k1 = 2` and then increases again. Its
1.48× is thus a max/min span, and not an end-to-end span. The other **eleven of twelve**
curves (3 levels × 2 floor columns × 2 arms) are monotone. This is the weakest cell of
the full design, and you must state it wherever you discuss a layer-1 activity effect in
the delay arm. But note that the pooled axis is monotone and moves 1.87× there, and the
pooled axis is what this grid manipulates.

**Floor separation in `s`, at both layers, in each row** (criterion 3, threshold 20 pts):

| | layer 1 | layer 2 | network |
|---|---|---|---|
| no-delay | +43.9 … +57.6 | +43.8 … +48.0 | +45.9 … +51.7 |
| delay | +32.7 … +45.9 | +27.6 … +30.4 | +30.2 … +38.1 |

**PASS in all 16 row × layer checks** (4 `k` × 2 arms × 2 layers). The smallest margin
(+27.6) is still 38% above the threshold.

**Criterion 2 has a partial violation, as in each grid of this project.** `s` is not flat
in `k` inside a column. The floor-OFF column drifts **+29.4 pts** at layer 1 in the
no-delay arm (79.9% → 50.5%) and **+30.3 pts** in the delay arm (69.0% → 38.7%). Layer 2
drifts much less (8.0 / 11.0 pts), and the floor-ON columns drift less again (6–17 pts).
Therefore the row control moves `s` and also `a`, mostly at layer 1 and mostly with the
floor off. Documents 5 and 6 record the same violation at the same magnitude. It is the
reason why the design decorrelates `a` from `s`, and does not make the two controls
orthogonal by construction.

**Criterion 4 — no rate inflation anywhere.** The maxima of `spikes_per_neuron` against
the natural rate of each layer: no-delay 3.59 / 4.99 against 7.85 / 5.53; delay
4.21 / 8.52 against 11.66 / 23.93.

**Criterion 5 — clean accuracy.** No-delay **.480 – .572** (v1: .49–.59; probe:
.441–.555). Delay **.739 – .840** (v1: .78–.88; probe: .717–.806). Both arms are above
their probes, and inside or just below the range of v1. The floor **costs nothing and
helps**: floor-ON is better than floor-OFF in all 8 rows (+.034 … +.053 no-delay,
+.002 … +.045 delay).

**Ceiling leak, the residual limitation.** At the tightest row: no-delay over-`k1` 7.4% /
33.3% and over-`k2` 23.2% / 59.2% (floor off / on); delay 16.1% / 37.1% and 39.5% /
50.6%. The ceiling of layer 2 leaks more than the ceiling of layer 1 in **all 8 no-delay
cells and in 7 of 8 delay cells**. The exception is the loosest row of the delay arm,
`k8` floor-on, at 5.7% against 9.2%. Document 6 recorded this property at layer 2, and §5
step 1b confirmed it again on the probe. Therefore the row axis of layer 2 is the softer
of the two, and that fact belongs with each coefficient that you report from it.

### 6d. The model-selection fault tried to happen again — and the settle window caught it

We audited `argmin(val_loss)` against the run length for all 48 cells, exactly as the
warning at the top of this page instructs.

**Delay arm — 3 of 24 cells show the fault.** Their *global* minimum of the validation
loss is at epoch 42 to 44, about 20 epochs after the penalties start, and it never
recovers:

| cell | global `argmin(val_loss)` | **selected under `SETTLE_EPOCHS = 150`** | over-`k1` at global → selected | `a1` at global → selected |
|---|---|---|---|---|
| `k1` floor 1, seed 44 | **44** / 1249 | **1197** | 61.1% → **37.7%** | 5.05 → **2.31** |
| `k2` floor 1, seed 43 | **42** / 1016 | **716** | 50.3% → **23.7%** | 5.27 → **2.55** |
| `k2` floor 1, seed 44 | **43** / 867 | **567** | 49.7% → **23.1%** | 5.17 → **2.50** |

Without the window, these three cells would have entered the grid at **2.07–2.19×** the
intended activity and **1.6–2.2×** the intended ceiling leak. This is the same corruption
that affects 2 of 16 cells in the completed 1st-layer delay grid, with no visible sign.
All three cells are floor-ON, as predicted. In the floor-on column the task validation
loss is at its best just after the penalties start, and it never recovers.

**With the window, each of the 48 cells is clean.** The selected epoch as a fraction of
the run length is **0.65 – 1.00** in the delay arm and **0.67 – 1.00** in the no-delay
arm. The constraint state (`over_k` at both layers, and `a1`) does not change between the
selected epoch and the final epoch in any cell. The differences are in the third decimal.

**The no-delay arm confirms that it did not need the window**, as the probe measured. Its
earliest global `argmin` is epoch 617 of 917. Therefore a settle window of 150 would have
changed nothing. The intentional asymmetry in the selection policy (§5 step 1b) now has
validation on the real grids, and not only on 4-corner probes.

### 6e. The perturbation sweeps, and what the three sites say

We swept each checkpoint at `l1`, `l2` and `both`, with four perturbations, in two arms.
The third trap of §5 step 3 was "a dependent variable for the full network is a design
decision, and not a given". The data settle it, and they **support the careful reading**:
the three sites do not reduce to one another.

This table gives the mean retention, corrected for chance, at the end of each sweep:
`(acc_end − .05)/(acc_clean − .05)`, averaged over the 8 setups.

| arm | perturbation | `l1` | `l2` | `both` | `both` − `l1` |
|---|---|---|---|---|---|
| no-delay | relocation | .683 | .985 | .705 | **+.022** |
| no-delay | jitter | .693 | .963 | .711 | **+.018** |
| no-delay | shift | .515 | .706 | .411 | −.104 |
| no-delay | deletion | .410 | .518 | .187 | **−.223** |
| delay | relocation | .434 | .830 | .445 | **+.011** |
| delay | jitter | .498 | .886 | .470 | −.028 |
| delay | shift | .317 | .788 | .268 | −.049 |
| delay | deletion | .296 | .566 | .142 | **−.154** |

Three readings:

1. **Layer 2 is a much cheaper place to receive a perturbation.** This is true for each
   perturbation in both arms, by a large margin. These sweeps cannot show the cause.
   Layer 2 can carry less of the timing code. Alternatively, damage there has one layer
   less to move through. This is a question for the availability and usage measurement,
   which we have not made.
2. **The `both` site is not a uniform increase over `l1`.** Relocation and jitter
   **saturate**. When we add the layer-2 insult to the layer-1 insult, nothing measurable
   changes, and the small positive signs are inside the spread across the setups. Both of
   these perturbations move spikes *inside* a window and hold the count of each neuron
   exactly. Therefore, when the placement code of layer 1 is gone, a second pass finds
   almost nothing more to remove.
3. **Deletion is the perturbation that accumulates** (−.223 / −.154), and shift
   accumulates a little. Deletion is also the only perturbation that does **not** keep
   the count. This is the order that an accumulation argument predicts. It is a positive
   reason to report the three sites separately, and not to collapse them into one pooled
   dependent variable.

### 6f. Where to read the figures

[result_visualization/bothLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/bothLayer/results_visualization.ipynb)
holds the clean accuracy, one section for each injection site (`l1`, `l2`, `both`), the
three sites side by side, the full manipulation check at both layers and pooled, and the
tables. It holds the non-sparse baseline for `l1` and `l2`. **No baseline exists for the
`both` site**, because the `exp_fixed_weight_perturbation` sweeps always perturbed one
layer at a time. Therefore the notebook omits the reference line there. It does not use a
single-layer file, because that file would put a weaker insult on the same axes.

### 6g. The dependent variable, measured at last (2026-08-09)

We pointed the three analysis scripts at `v3L12` and ran them on both arms, with 24
models each. Each script now has a `GRID` control that selects the constrained layer set.
Therefore **one** script serves the layer-1 grid, the layer-2 grid and the both-layer
grid. This is what makes the cross-grid comparison of β in §1 point 3 legitimate, and not
a comparison of three copies that became different.

| script | what changed | artifacts |
|---|---|---|
| `hidden_channel_decode.py` | it probes both layers. Each layer uses **its own** window for each arm (L1 `[0,90)`; L2 `[0,90)` no-delay, **`[0,160)`** delay). It applies `delay1` structurally on the layer-2 path. The views are `l1`/`l2`/**`net`** | `hidden_channel_decode_v3L12_{arm}.json` |
| `phase1_measure.py` | `SITES = ("l1","l2","both")`. Relocation stays inside the window of each layer (`88`/`90`/`160`), and **deletion is not clipped**. The forward pass divides, thus `delay1` is downstream of an L1 injection and upstream of an L2 injection | `phase1_measure_v3L12_{arm}.json` |
| `phase1_regress.py` | the axes come from the `net` view. There is one usage model for each site, one availability model for each view, and the report that selects the arm | `phase1_regress_v3L12.json`, 2 figures |

**Three verifications, because a reading of the code cannot settle any of this.** If we
skip `delay1`, the rate of layer 2 moves by **19%** and its silent fraction moves by **6
points**. With `delay1` applied, the probe reproduces the per-layer and pooled axes of the
training summary to 3 or 4 decimals in each cell that we checked. Therefore a measurement
discharges the second trap. On the `v3` checkpoints, `phase1_measure` reproduces the
archived layer-1 numbers to **1e−9**, and `phase1_regress` reproduces its published
coefficients exactly. The new report that selects the arm independently recovers the
readout gap of document 5 (**+0.284** no-delay against **+0.042** delay, against the
+.300 / −.013 of document 5 at layer 1).

> **We found a file name collision and closed it.** `phase1_measure.py` wrote
> `phase1_measure_{arm}.json` with **no version tag**. Therefore a run on `v3L12` would
> have overwritten the completed layer-1 measurements with no message. This is the danger
> of §7 point 4, in a script and not in a checkpoint path. The outputs now have tags. We
> moved the existing files with `git mv` to `phase1_measure_v3_{arm}.json`, and we moved
> `phase1_regress.json` and its two figures to the `_v3` names.

#### The measurement

These are the means over 24 models for each arm. The `a` and `s` values are the
**network** axes.

| | no-delay | delay |
|---|---|---|
| **availability** (timing fraction) `l1` / `l2` / `net` | .270 / .234 / **.224** | .233 / .079 / **.097** |
| **usage** `l1` / `l2` / `both` | .315 / .016 / **.296** | .565 / .171 / **.551** |
| **deletion control** `l1` / `l2` / `both` | .594 / .487 / **.819** | .704 / .435 / **.861** |
| **readout gap** (`decode_full` − clean acc) at `net` | **+0.285** | **+0.076** |

**The site structure of §6e occurs again in the endpoint measurement.** The usage
saturates — `both` ≈ `l1` (.296 against .315; .551 against .565) — and the deletion
control **accumulates** (.594 → .819; .704 → .861). The full sweeps and the single
endpoint measurement agree. This is the check that the two scripts use the same injection
sites and the same windows.

**Layer 2 is the cheaper place to receive a perturbation, and now also the poorer place
to read timing from.** Its availability is .079 in the delay arm, against .233 for layer
1. Its usage is .171, against .565. The two measurements agree here, but they did not
have to agree.

### 6h. β — H1′ gets no support, in either arm, at any site

The model is `usage ~ log(a_net) + s_net + control`, with n = 24. The full output is in
`v3_analysis/log/phase1_regress_v3L12.out`.

| | no-delay | delay |
|---|---|---|
| ρ(`a_net`, `s_net`) Spearman | −0.422 (p=.040) **PASS** | −0.064 (p=.77) **PASS** |
| **β_a** at `l1` / `l2` / `both` | +.025 / −.001 / **−.001** | −.010 / **+.137*** / **+.021** |
| **β_s** at `both` | **−0.168** (p=.049) | −0.089 (p=.12) |
| β_a for availability at `net` | **−0.035** (p=.030) | +0.003 (p=.68) |
| matched floor-on against floor-off contrast | **+0.100** (p=.0001, 12 pairs) | **+0.068** (p=.011, 8 pairs) |

\* The one significant β_a in the table is the **layer-2 usage in the delay arm**. It is
**+0.137**, that is, *against* H1′ and not for it.

**The verdict is: H2 is CONFIRMED in the no-delay arm, and the delay arm is inconclusive
but has the same direction.** β_a is null at each site in both arms, except at the one
site where it has the wrong sign. β_s carries the effect. The controlled floor contrast is
positive and significant in **both** arms: when we remove the silence, the use of timing
*increases*. That is the prediction of H2, and it is the first truly manipulated form of
that comparison at the network level in this project.

**This verdict differs from the verdict of the layer-1 grid, and that difference is the
point of §1 point 3.** That grid gave **H1′ REFUTED**, with β_a significantly *positive*
at **+0.200 (p = .0001)** no-delay and **+0.085 (p = .033)** delay. Here the same
coefficient is **−0.001 (p = .99)** and **+0.021 (p = .32)**. Therefore a constraint on
the full network did not change a null result into an effect. It changed a significant
**anti**-H1′ coefficient into a null result.

**We checked the obvious deflationary explanation, and it is not correct.** That
explanation is "the network-wide axis is only a weaker manipulation". It would account
for the result. But the axes that the two factorials *achieved* have the same size. The
`a` axis spans **2.26× / 2.18×** in the layer-1 grid, against **2.11× / 2.72×** here. The
`s` axis spans 5.6–76.3% / 5.2–66.3%, against 4.1–69.9% / 3.4–57.3% here. (The 3.76× /
4.05× figures in §2b point 2 are the *observed* layer-1 spans of v1. They are not the
correct comparison for a manipulated axis.) Therefore the coefficient moved, but the axis
of that coefficient did not.

What stays is the reading that §1 point 1 predicted. The positive β_a of layer 1 was, at
least in part, the **compensation by layer 2**. Under a layer-1-only penalty, layer 2 is
free to absorb the decrease. The β_a of the layer-1 grid followed that compensation, and
not the sparsity of the network. A constraint on both layers removes the compensator, and
the coefficient becomes zero. §6j runs the `v3L2` leg that separates these accounts, and
it supports the compensation reading.

**One item that this grid adds, and neither single-layer grid can add.** The residual
signal is not only the selectivity. The deletion control is the **largest standardised
term** in the primary model of the delay arm (β\* = +0.537). At the `both` site the
control reaches .82 to .86, where the usage reaches only .30 to .55. An insult to the
full network that destroys the rate does much more damage than one that destroys only the
timing. This is true at each level of sparsity that we tested.

**Read each β above as a network-level coefficient** (§7 point 1). ρ(`a1`, `a2`) is
+0.97 / +0.89 here, and both controls move both layers by construction.

### 6i. The primary arm — the evidence, measured again

Document 5 selected the delay arm on two Phase 0 findings that it measured **at layer 1
only**. §5 step 2 refused to inherit them. We measured both again on the checkpoints of
this grid:

| criterion | no-delay | delay | it selects |
|---|---|---|---|
| the readout-efficiency gap at `net` (a smaller value is better) | **+0.285** | **+0.076** | **delay** |
| \|β_a(usage) − β_a(control)\| at `both` (a larger value is better) | −0.012 | +0.015 | delay, but only by a small margin |

**The readout gap of the no-delay arm is the decisive number, and a constraint on both
layers did not improve it.** A linear decoder that reads the hidden network is better
than the readout of the network by **.285**, on **100% of the checkpoints**. This is
almost the +.300 that document 5 measured at layer 1. Perturbation experiments in that
arm measure the inefficiency of the readout as much as the structure of the code. No
factorial design can correct this.

Against that is the consideration of §7 point 8, which points the other way. The no-delay
arm is the only arm whose pooled confound truly failed. Therefore it is the only arm where
this design had something to break. We report both arms. The **delay arm is primary**,
which matches documents 5 and 6. The no-delay arm is secondary, and we do not discard it.

### 6j. β_a across all three grids — the main result, and §1 did not expect it

We also ran the retargeted scripts at `GRID = "v3L2"` (54 models: 24 no-delay and 30
delay). Therefore the same dependent variable, measured in the same way, now exists for
all three grids. The model is `usage ~ log(a) + s + control`. Each grid uses its own
constrained layer and its own achieved axes:

| arm | grid | constrained | n | **β_a** | 95% CI | p | β\* |
|---|---|---|---|---|---|---|---|
| **delay** | `v3` | layer 1 | 24 | **+0.085** | [+0.008, +0.162] | **.033** | +0.394 |
| **delay** | `v3L2` | layer 2 | 30 | **+0.087** | [+0.062, +0.113] | **<.0001** | **+0.638** |
| **delay** | `v3L12` | **both** | 24 | **+0.021** | [−0.021, +0.063] | .32 | +0.137 |
| no-delay | `v3` | layer 1 | 24 | **+0.200** | [+0.118, +0.283] | **.0001** | +0.495 |
| no-delay | `v3L2` | layer 2 | 24 | −0.007 | [−0.031, +0.016] | .53 | −0.081 |
| no-delay | `v3L12` | **both** | 24 | −0.0005 | [−0.078, +0.076] | .99 | −0.002 |

**The pattern is exact: β_a is significantly different from zero in 3 of the 4
single-layer cells, and in 0 of the 2 both-layer cells.** This is not a problem of
statistical power. The both-layer intervals **exclude** the single-layer point estimates
in **3 of 4** comparisons (both delay-arm grids, and layer 1 in the no-delay arm). Only
no-delay layer 2 falls inside, and that cell is a null result itself. The single-layer
cell with the most power is `v3L2` delay. It has n = 30, the widest achieved `a` axis of
all the grids at **3.50×**, and the largest standardised β_a at **+0.638**. Therefore the
single-layer effect is not marginal, and the both-layer null result does not come from
low power.

**§1 point 3 gave two possible outcomes, and the data chose a third.** It said that a
null result in all three grids is a fact about the architecture, and that an effect *only*
under the network-wide constraint is a result that no single-layer grid can give. What
happened is the mirror image of the second outcome: **the effect occurs only when exactly
one layer is constrained, and it disappears when the network is constrained.** That is
still a result that no single-layer grid can give. But it is a result *about the
single-layer designs*, and not about sparsity.

**Read this with §1 point 1. It is the compensation that the calibration predicted, now
in the form of a coefficient.** ρ(`s1`, `s2`) = −0.829 under a layer-1-only penalty
(§2a): constrain one layer, and the other layer moves the opposite way. Therefore a
single-layer β_a comes from a network where we made one layer sparse and the other layer
compensated, and the coefficient follows that compensation. If you remove the compensator
with a constraint on both layers, the coefficient becomes zero and β_s survives (−0.089 /
−0.168, with the matched floor contrast significant in both arms). **Selectivity is the
one variable that survives each location.**

Two cautions belong with this result, and §7 already holds both. First, each significant
β_a in the table is **positive**, that is, against H1′, which predicted β_a < 0.
Therefore nothing here rescues H1′, and the both-layer grid removes the anti-H1′ signal
instead of a reversal of it. Second, the three grids differ in what a coefficient can be
attributed to. `v3` and `v3L2` can name a layer. `v3L12` cannot (§7 point 1).

---

## 7. Points to watch

This section inherits document 1 §7, document 4 §7, document 5 §8 and document 6 §5 in
full. These points are new to the both-layer experiment:

1. **This grid cannot attribute an effect to a layer.** ρ(`a1`, `a2`) is +0.96 / +0.84
   in the observed data, and this design moves the two values together by construction.
   A β from this grid is a **network-level** coefficient. The two single-layer grids give
   the attribution to a layer. Any claim of the form "the sparsity of layer N causes X"
   needs one of them.
2. **The pooled axes are not the layer-1 axes, and the difference is large.** The
   no-delay gradient of v1 is 6.5× in the firing of layer 1, but 2.8× across the network.
   If you quote a layer-1 range as the strength of the manipulation, you overstate it by
   **57%** (42% on the `a` axis, and 61–62% in the delay arm).
3. ~~**ρ(`s1`, `s2`) is the number to watch, and its sign in v1 is a warning.**~~
   **RESOLVED by the grid (§6b): +0.977 / +0.944**, against −0.829 / +0.203 in v1. The
   column control is a selectivity manipulation for the full network. Therefore the
   pooled `s_net` axis is a statement about the network. This point stays here, because
   it is the central claim of the design, and each new run must check it again. Both
   scripts still print it against its v1 baseline.
4. **The `v3L12` tag is necessary, now against two earlier grids.** All three grids share
   the dataset, the arm, `k`, the floor and the seed. Without the tag, this run
   overwrites the completed 1st-layer delay grid (about 37 h to train again) or the
   2nd-layer grid.
5. **The canonical summary keys with no suffix mean the network here, and not layer 1.**
   A reader or a script that uses the single-layer convention compares a pooled quantity
   against a per-layer quantity, and nothing shows the error. Each per-layer value has an
   explicit `_l1` or `_l2` suffix. Use those keys when you compare the grids.
6. **The floor-off column at `k = 1` is the most dangerous cell in the project.** It has
   two ceilings that push toward zero, nothing opposes either of them, and there is a
   known gradient path that dies exactly when the first layer becomes empty (§3c). This
   is why it is criterion 1 of the probe.
7. **`CEILING_K_LAYER2_RATIO` is a constant of the arm.** We measured 1.0 and 2.0 from
   the natural `a` of each arm. If you carry either value to the other arm, you make an
   unequal decrease there. This is the general rule of document 6 §5.1, in its newest
   form.
8. **The choice of the arm is not inherited** (§5 step 2). Unlike documents 5 and 6,
   there is now a consideration that points the *other* way: the no-delay arm is the only
   arm whose pooled confound truly fails, and §6a shows that it is the only arm where
   this design measurably changed the confound structure.
9. **New, from the grid: the floor-OFF layer-1 activity axis of the delay arm is not
   monotone** (§6c). It falls at `k1 = 2`, and its 1.48× is a max/min span. Of the twelve
   `a`-against-`k` curves of this grid, it is the only bad one. Check this cell before
   you quote any layer-1 activity effect in that arm.
10. **New, from the grid: the three injection sites are not interchangeable** (§6e).
    Relocation and jitter saturate — `both` ≈ `l1` — and deletion accumulates strongly. A
    single dependent variable for the network would hide this. Therefore the sweeps and
    the notebook keep all three sites.
11. **New, from the retarget: the `v3L12` tag is also necessary in the *analysis*
    scripts, and not only in the checkpoint paths.** `phase1_measure.py` wrote an
    **untagged** `phase1_measure_{arm}.json`, and `phase1_regress.py` wrote an untagged
    `phase1_regress.json`. If you point either script at a second grid, it overwrites the
    completed layer-1 measurements in place, and the file name shows nothing. Both
    scripts now write tagged files, and the archived layer-1 files carry `_v3`. **Before
    you run any analysis script against a new grid, check that its output path contains
    the grid tag.** Point 4 applies to everything that this project writes, and not only
    to `sn_data/`.
12. **New, from the retarget: `GRID` must have the same value in all three analysis
    scripts.** They communicate through file names. Therefore a decode run at `v3L12`
    that joins a measurement run at `v3` mixes the axes of one grid with the dependent
    variable of another grid. The tags normally prevent this. But each grid now has
    measurements, thus all the file names exist, and an old `GRID` value in one script
    alone no longer raises an error. Therefore `phase1_regress.py` compares the recorded
    `grid` and `sites` of each input file against its own values, and it refuses the
    join. Refer to `check_measurement_provenance`. **This now has a guard, and not only
    documentation.**

---

## 8. Checklist

- [x] Both training scripts point at **both** hidden layers. The penalties, the logged
      statistics, the clean evaluation and the summary all act on `hidden1`/`potential1`
      **and** `hidden2`/`potential2`, and on the pooled network-wide axes
- [x] The artifacts carry the tag `v3L12`. The summary rows carry `target_layers`. The
      probe runs carry `_probe`. No path can collide with the 1st-layer (`v3`) or
      2nd-layer (`v3L2`) grids
- [x] The parameter set is unchanged, and we checked it: both scripts load v1
      checkpoints with `strict=True` (8 / 10 tensors), thus the existing evaluation
      pipeline stays compatible
- [x] **Calibration step 0** — we computed the cross-layer and pooled axes over all 27
      v1 checkpoints from `layer2_baseline_{arm}.json`
- [x] We measured the finding that makes the work necessary: **ρ(`s1`, `s2`) = −0.829**
      (no-delay). A layer-1-only penalty makes the selectivity of the two layers move in
      *opposite* directions. The value is **+0.203** (delay), which shows no relation
- [x] We measured the pooled confound: **ρ(`a_net`, `s_net`) = −0.879** (no-delay, which
      fails the threshold) and **+0.042** (delay, which already passes). The two arms get
      different results, and both scripts say which
- [x] We measured the cross-layer activity confound (**+0.961 / +0.839**), and we
      excluded it from the acceptance check, because this design does not break it
- [x] We measured the strength of the manipulation across the network: `spikes/neuron`
      **2.83× / 2.11×**, against 6.50× / 5.51× at layer 1; and `a_net` **2.17× / 1.56×**,
      against 3.76× / 4.05× for `a1`. The single-layer figures overstate the manipulation
      by 42–62%
- [x] The row axis has a different scale for each layer (`CEILING_K_LAYER2_RATIO` = 1.0 /
      **2.0**), taken from the measured natural `a`. Both arms thus need **16 models**,
      where the 2nd-layer delay grid needed 20
- [x] We verified the implementation end to end (§4): the 5-tensor forward pass, all four
      gradient paths from a penalty to the weights nonzero, the L2-ceiling→`fc1` path
      measured at 0.81 / 2.05, the §3c edge case exactly 0, and the pooling exact to 1e−9
- [x] **Corner probe, both arms** — 8 models, 400 ep, 5.9 h, 2026-07-31
- [x] Probe criterion 1 — **`WARMUP_EPOCHS = 20` survives the two-penalty
      configuration**. The corner at `k = 1`, floor = 0 holds at acc .441 (no-delay) and
      .717 (delay), and neither layer is fully silent. The absorbing failure that this
      guards cannot happen
- [x] Probe criterion 3 — **`FLOOR_STRENGTH = 1` separates `s` at BOTH layers in both
      arms**: +63.1/+57.7 and +52.2/+53.1 (no-delay), +57.7/+43.5 and +48.4/+41.1
      (delay). Each cell is OK, with a large margin
- [x] Probe criterion 4 — no rate inflation at either layer in either arm
- [x] Probe criterion 5 — accuracy .441–.555 (no-delay, v1: .49–.59) and .717–.786
      (delay, v1: .78–.88, and better than the .685–.771 of the 1st-layer arm at strength
      10)
- [x] **`CEILING_STRENGTH = 10` is validated for the no-delay arm.** Layer 1 reproduces
      its single-layer sibling to within about 2 points of `s` and 0.1 spikes of `a` in
      each cell. Therefore the layer-2 penalty constrains layer 2 and does not disturb
      layer 1
- [x] ρ(`s1`, `s2`) came back at **+1.000** in both arms, against −0.829 / +0.203 in v1.
      The column control is a selectivity manipulation for the full network. (The
      magnitude is degenerate at n = 4. The sign is the important part)
- [x] We ran the ceiling calibration of the delay arm at `CEILING_STRENGTH = 20` (about
      1.5 h). **We tested it and rejected it.** It makes the ceiling tighter, but it
      costs **.153** of clean accuracy in the floor-off column (.717 → .564), far below
      the .78–.88 of v1. We keep it on disk under `_ceilcal_str20` tags
- [x] **We found and corrected the model-selection fault.** In the floor-ON column of the
      delay arm, the task validation loss is at its best about 20 epochs after the
      penalties start, and it never recovers. Therefore the saved checkpoint came from
      epoch 37 to 45 of about 340, and it was almost unconstrained. The apparent "62.2%
      ceiling leak" was this artifact. Trained to the end, the same strength-10 run sits
      at **36.9%**, inside the band of the 1st-layer grid, with a *wider* row axis (1.79×
      against 1.48×). We adopted `SETTLE_EPOCHS = 150` in the delay script, and
      `CEILING_STRENGTH = 10` stays
- [x] We confirmed that the fault does **not** affect the no-delay arm (each cell selects
      at epoch 326–398 of 400, with the constraint state unchanged from the saved epoch to
      the last epoch). Therefore its running grid is sound
- [x] We audited the **completed 1st-layer v3 delay grid** for the same fault. 14 of 16
      cells are healthy (they select at a fraction of 0.71–1.00). The two `k=1` floor-on
      cells are not healthy (seed42 433/734, **seed43 48/349**). The conclusion of
      document 5 comes mostly from clean cells, but those two cells are less constrained
      than the design intended
- [x] We tested both scripts on the GPU after the `SETTLE_EPOCHS` change. All checks
      pass. The settle guard is inside `train_model`, and the tracking starts at epoch
      170
- [x] The pre-settle delay probe artifacts stay under `_probe_presettle` tags (4
      checkpoints, 4 training logs, 1 summary). They are the evidence for the
      model-selection finding, and the re-probe writes to the same `_probe` names
- [x] **We STARTED the delay-arm re-probe** on 2026-07-31, with the settle window at
      `CEILING_STRENGTH = 10` (about 3 h; longer than the first probe, because no cell
      can stop early at 400 epochs now)
- [x] **The delay re-probe PASSES.** The selection moves to epoch 341–394 in each cell,
      and no cell stops early. The two floor-off cells reproduce the pre-settle run
      exactly, and they are the control. At `k1 = 1` floor-on: over-`k1` **62.2% →
      36.0%**, `a1` 5.14 → 2.33, and the network row axis **1.67× → 2.19×**, now above the
      2.18× of the completed 1st-layer delay grid. The accuracy *increased* (.717–.806
      against .717–.786), thus the correction cost nothing. The floor separation is
      +51.0/+41.0 and +46.6/+41.9. There is no inflation
- [x] **We RAN the NO-DELAY grid** on the remote server and retrieved it on 2026-08-07.
      `QUICK_TEST = False`; `CEILING_STRENGTH = 10`, `FLOOR_STRENGTH = 1`,
      `WARMUP_EPOCHS = 20`, and no settle window (we measured that this arm does not need
      it). **24 models** (3 seeds, and not the 2 planned), 1250 epochs
- [x] **We RAN the DELAY grid** on the remote server and retrieved it on 2026-08-07. It
      used the same constants and `SETTLE_EPOCHS = 150`. `CEILING_K` = [1,2,4,8], with
      layer 2 at ×2 → [2,4,8,16]. **24 models**, 1250 epochs
- [x] We pointed the **8 `*_bothLayer_evalOnly_*_v3.py`** scripts at
      `SITES = ("l1", "l2", "both")`, each at the window of its own layer and arm — L1
      `[0, 88)` in both arms, L2 `[0, 90)` / `[0, 160)` — and we **left shift and
      deletion unclipped**. We ran all eight sweeps on 2026-08-08
- [x] **The design acceptance check PASSES on the real grid, at all three levels, in both
      arms** (§6a, n = 24): ρ(`a_net`, `s_net`) **−0.879 → −0.199** (no-delay) and
      **+0.042 → +0.040** (delay); for each layer −0.943 → −0.297 / +0.829 → −0.122 and
      −0.455 → −0.001 / +0.427 → +0.106. All values are inside |ρ| ≤ 0.5
- [x] **ρ(`s1`, `s2`) became positive, and it is not marginal: +0.977 / +0.944**, against
      −0.829 / +0.203 in v1 (§6b). The column control is a selectivity manipulation for
      the full network. This is the central claim of the design, confirmed at n = 24 and
      not at the degenerate n = 4 of the probe
- [x] We checked the probe criteria again on the real grid (§6c). The floor separates `s`
      by 20 pts or more at **both layers in all 16 row × layer checks** (the smallest is
      +27.6). There is no rate inflation at either layer. The clean accuracy is
      .480–.572 (no-delay) and .739–.840 (delay), above the probes and inside the range of
      v1. Floor-ON is better than floor-OFF in all 8 rows
- [x] We measured the row axis: across the network **1.95× / 2.07×** (no-delay floor off
      / on) and **1.87× / 2.25×** (delay). The delay figure agrees with the 2.19×
      prediction of the post-settle probe. **One curve is not monotone**: the floor-OFF
      layer 1 of the delay arm (3.54 → 3.01 → 3.31 → 4.45). We record it as a limitation,
      and we do not hide it
- [x] **We audited the model selection on all 48 cells (§6d).** 3 of the 24 delay cells
      had their global `argmin(val_loss)` at epoch 42–44, which is the same fault.
      `SETTLE_EPOCHS = 150` refused all three. It moved the selection to epoch
      567/716/1197, and over-`k1` from about 50–61% down to 23–38%. Each of the 48 cells
      now selects at 0.65–1.00 of the run length, with the constraint state unchanged to
      the final epoch. The earliest selection in the no-delay arm is 617/917, which
      confirms that this arm never needed the window
- [x] **Perturbation sweeps at all three sites, both arms** (§6e). The finding is that
      relocation and jitter **saturate** (`both` ≈ `l1`), and that deletion
      **accumulates** (−.223 / −.154 of retention). This is the empirical answer to the
      third trap of §5 step 3, and the reason to keep the three sites separate
- [x] **We visualised the results** —
      [bothLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/bothLayer/results_visualization.ipynb),
      with one section for each injection site, the three sites side by side, and the
      manipulation check at both layers and pooled
- [x] **We pointed `hidden_channel_decode.py` and `phase1_measure.py` at both layers**,
      each at the window of its own layer and arm (§6g). Both scripts now have a `GRID`
      control that selects the constrained layer set. Therefore **one** script serves all
      three grids, and the cross-grid comparison of β is a true comparison, and not a
      comparison of three copies that became different. The decode views are
      `l1`/`l2`/`net`, and the injection sites are `l1`/`l2`/`both`. **Deletion stays
      unclipped**, and relocation stays inside the window of each layer
- [x] **A measurement verified the `delay1` routing, and not an assertion.** If we skip
      `delay1`, the rate of layer 2 moves 19% and its silent fraction moves 6 points. With
      `delay1` applied, the probe reproduces the per-layer and pooled axes of the training
      summary to 3 or 4 decimals. The second trap of §5 step 3 is thus discharged
- [x] **We tested the retarget against the completed layer-1 analysis.** On the `v3`
      checkpoints, `phase1_measure` reproduces the archived numbers to 1e−9, and
      `phase1_regress` reproduces its published coefficients exactly. Therefore the
      retarget moved nothing that it must not move
- [x] **We found and closed a silent file name collision.** `phase1_measure.py` wrote an
      **untagged** `phase1_measure_{arm}.json`, and it would have overwritten the layer-1
      measurements on the first `v3L12` run (§7 point 4, in a script and not in a
      checkpoint path). The outputs now have tags, and we moved the existing files with
      `git mv` to `_v3`
- [x] **We measured the dependent variable, both arms, 24 models each** (§6g):
      availability `net` **.224 / .097**, usage `both` **.296 / .551**, and deletion
      control `both` **.819 / .861**. The site structure of §6e occurs again in the
      endpoint measurement: the usage **saturates** (`both` ≈ `l1`), and the control
      **accumulates**
- [x] **We fitted the regressions** (§6h) — `usage ~ log(a_net) + s_net + control` at all
      three sites, and availability at all three views, with n = 24 for each arm. **β_a is
      null at each site in both arms**, except for the layer-2 usage in the delay arm,
      which is **+0.137, against H1′**. β_s carries the effect. The controlled floor-on
      against floor-off contrast at equal `a` is **+0.100 (p=.0001)** and **+0.068
      (p=.011)**. That is the direction that H2 predicts, and it is manipulated and not
      observed
- [x] **We decided the primary arm: DELAY** (§6i). We measured the two Phase 0 criteria
      again **on this grid**, and we did not inherit them across a layer boundary (§5 step
      2). The readout gap of the no-delay arm is **+0.285 on 100% of the checkpoints**,
      against +0.076 for the delay arm. This is almost the +.300 of document 5, and a
      constraint on both layers did not change it. We keep the no-delay arm as secondary
- [x] **We compared β_a across all three grids** (§6j). We ran the `v3L2` leg with the
      same retargeted scripts (54 models). Therefore all three grids now share one
      dependent variable, measured in one way. **β_a is significant in 3 of 4
      single-layer cells and in 0 of 2 both-layer cells**, and the both-layer intervals
      **exclude** the single-layer point estimates in 3 of 4 comparisons. Thus the
      both-layer null result is not a problem of power. §1 point 3 gave "a null result
      everywhere" or "an effect only when the network is constrained". The answer is the
      **mirror image of the second**: the effect exists only when a single layer is
      constrained, and it disappears when the network is constrained. This is the
      compensation of §1 point 1 in the form of a coefficient. Each significant β_a is
      **positive**, that is, against H1′. Selectivity is the variable that survives each
      location
- [ ] Optional, if the `both`-site sweeps must ever be read against an unconstrained
      reference: make the **missing non-sparse baseline for the `both` site** in
      `exp_fixed_weight_perturbation`. The sweeps that exist always perturb one layer
