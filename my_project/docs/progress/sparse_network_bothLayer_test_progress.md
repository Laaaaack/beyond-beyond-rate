# Both-layer v3 — the same factorial, on the whole hidden network (design, calibration)

**Status (2026-07-31): ALL LOCAL CALIBRATION COMPLETE. **BOTH ARMS ARE READY FOR THEIR
16-MODEL GRIDS**, to be run on the remote server. 14 probe-class models trained locally
(~11 h): 4-corner probes in both arms, a rejected `CEILING_STRENGTH = 20` calibration,
and a delay-arm re-probe that validated the `SETTLE_EPOCHS = 150` fix.
`CEILING_STRENGTH = 10` stands in both arms. The primary arm is still undecided — that
decision needs the dependent variable re-measured at both layers (§5 step 3), which is
independent of running the grids.**

> **Execution convention.** Local machine (RTX 2050, 4 GB) runs **probes only**. The
> full 1250-epoch grids run on the remote server. Both scripts therefore ship with
> `QUICK_TEST = True`; flip it to `False` on the remote host. A local grid run was
> started on 2026-07-31 and deliberately aborted during its first cell — no artifacts
> were written, since checkpoints and logs are only saved once a model completes.

> **The one thing to carry away from this page if you read nothing else.** Best-model
> selection on the *task* validation loss can save a checkpoint from **before the
> sparsity constraint has bound**, and when it does, every sparsity statistic measured
> off that checkpoint is wrong in the direction of "the penalty isn't working". It cost
> this experiment one incorrect diagnosis (§ step 1b) and it silently affects 2 of the
> 16 cells in the **completed** 1st-layer delay grid. Before trusting any cell, check
> `argmin(val_loss)` against the run length in its training log.

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

### Step 1 — corner probe, both arms — **DONE, 2026-07-31 (5.9 h, 2×2×2 = 8 models)**

Ran `k1 ∈ {1, 8}` × floor ∈ {0, 1}, seed 42, 400 epochs — layer 2 seeing `{1, 8}`
(no-delay) or `{2, 16}` (delay). Sequential on one 4 GB GPU, 2 h 56 m per arm. Probe
artifacts carry the `_probe` suffix; nothing real was touched.

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

#### Probe result — no-delay arm

| `k1`/`k2` | floor | acc | `a1` | `s1` | sp1 | over-`k1` | `a2` | `s2` | sp2 | over-`k2` | `a_net` | `s_net` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 / 1 | 0 | .441 | 2.48 | **86.89%** | 0.33 | 6.2% | 2.59 | **70.73%** | 0.76 | 18.0% | 2.56 | 78.81% |
| 1 / 1 | **1** | .538 | 1.84 | **23.78%** | 1.40 | 33.1% | 2.45 | **13.04%** | 2.13 | 56.3% | 2.17 | 18.41% |
| 8 / 8 | 0 | .502 | 3.57 | **59.12%** | 1.46 | 3.3% | 5.05 | **55.44%** | 2.25 | 6.9% | 4.34 | 57.28% |
| 8 / 8 | **1** | .555 | 3.91 | **6.91%** | 3.64 | 6.1% | 5.09 | **2.34%** | 4.97 | 11.3% | 4.52 | 4.63% |

| # | criterion | verdict |
|---|---|---|
| 1 | both layers survive `k=1`, floor = 0 | **PASS** — acc .441 against chance .05; `s1` 86.9%, `s2` 70.7%, neither at 100%. `WARMUP_EPOCHS = 20` transfers to the two-penalty configuration |
| 2 | `s` roughly constant as `k` varies within a column | **partial** — floor-off drifts 86.89→59.12 at L1 (**27.8 pts**) and 70.73→55.44 at L2 (15.3 pts); floor-on 16.9 / 10.7 pts. Same violation the 1st-layer probe had (23.6 pts), marginally worse |
| 3 | floor separates `s` by ≥ 20 pts at **both** layers | **PASS, with enormous room** — **+63.1 / +57.7** at `k=1`, **+52.2 / +53.1** at `k=8`. Every cell OK |
| 4 | firing not inflated past each layer's natural rate | **PASS** — sp1 ≤ 3.64 against 7.85, sp2 ≤ 4.97 against 5.53. No `!` flags |
| 5 | clean accuracy comfortably above chance | **PASS** — .441–.555 against chance .05, and against v1's no-delay .49–.59 |

**Layer 1 behaves almost identically to its single-layer sibling — the "`fc1` is squeezed
twice" worry did not materialise.** Against the 1st-layer-only probe at the same
settings (document 5):

| cell | `a1` both / 1st-only | `s1` both / 1st-only | over-`k` both / 1st-only |
|---|---|---|---|
| `k=1`, floor 0 | 2.48 / 2.69 | 86.9% / 80.9% | 6.2% / 9.4% |
| `k=1`, floor 1 | 1.84 / 1.91 | 23.8% / 21.6% | 33.1% / 34.2% |
| `k=8`, floor 0 | 3.57 / 3.57 | 59.1% / 57.3% | 3.3% / 2.8% |
| `k=8`, floor 1 | 3.91 / 3.81 | 6.9% / 6.7% | 6.1% / 5.2% |

The largest discrepancy is +6.0 points of `s1` at the floor-off `k=1` corner; everything
else is within ~2 points or 0.1 spikes. So in this arm, adding the layer-2 penalty
constrains layer 2 **without** disturbing what the layer-1 penalty was already doing —
which is the cleanest possible outcome for a design meant to be comparable to its
siblings cell by cell.

**Verdict: the no-delay arm is ready for its 16-model grid, unchanged.**

#### Probe result — delay arm

| `k1`/`k2` | floor | acc | `a1` | `s1` | sp1 | over-`k1` | `a2` | `s2` | sp2 | over-`k2` | `a_net` | `s_net` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 / 2 | 0 | .717 | 4.20 | **75.83%** | 1.02 | 14.4% | 4.41 | **50.82%** | 2.17 | 31.0% | 4.34 | 63.32% |
| 1 / 2 | **1** | .758 | 5.14 | **18.11%** | 4.21 | **62.2%** | 5.61 | **7.33%** | 5.20 | **65.8%** | 5.39 | 12.72% |
| 8 / 16 | 0 | .765 | 4.73 | **53.93%** | 2.18 | 6.6% | 9.10 | **42.83%** | 5.21 | 5.4% | 7.15 | 48.38% |
| 8 / 16 | **1** | .786 | 7.50 | **5.53%** | 7.08 | 30.6% | 10.50 | **1.70%** | 10.32 | 16.6% | 9.03 | 3.61% |

Criteria 1, 3, 4 and 5 all pass, several of them comfortably: survival at `k=1` floor 0
(acc **.717**), separation **+57.7 / +43.5** and **+48.4 / +41.1** points, no inflation
(sp1 ≤ 7.08 against 11.66; sp2 ≤ 10.32 against 23.93), and accuracy **.717–.786** —
*better* than the 1st-layer arm managed at strength 10 (.685–.771). Criterion 2 is
partially violated as everywhere else (21.9 pts of floor-off drift at L1).

**But the ceiling barely binds in the floor-on column, and this is the finding.** At
`k1 = 1` with the floor on, **62.2%** of layer-1 pairs and **65.8%** of layer-2 pairs
still fire above their ceiling. The consequence is a compressed row axis:

| span of `a` across `k1` = 1 → 8 | no-delay | delay |
|---|---|---|
| layer 1, floor off / on | 1.44× / **2.13×** | 1.13× / **1.46×** |
| layer 2, floor off / on | 1.95× / 2.08× | 2.06× / 1.87× |
| **network, floor off / on** | **1.70× / 2.08×** | **1.65× / 1.67×** |

Layer 1's floor-on axis moves only **1.46×** in the delay arm, against 2.13× in the
no-delay arm and against this arm's v1 observational span of 4.05×. Document 5 met
exactly this at layer 1 ("a factorial whose H1 axis moves 1.4× … will estimate β_a
badly") and fixed it by taking `CEILING_STRENGTH` from 3 to 10.

**The mechanism is visible and it is new.** Compare layer 1 against the 1st-layer-only
delay calibration at strength 10, `k = 1`:

| | `a1` | `s1` | sp1 | over-`k` |
|---|---|---|---|---|
| 1st-layer only, floor 0 | 4.00 | 72.1% | 1.12 | 17% |
| **both-layer, floor 0** | **4.20** | 75.8% | 1.02 | 14.4% |
| 1st-layer only, floor 1 | **2.66** | 22.6% | 2.06 | 41% |
| **both-layer, floor 1** | **5.14** | 18.1% | 4.21 | **62.2%** |

The floor-off column matches its sibling closely. The floor-**on** column does not:
layer 1 ends up **1.9× denser** (`a1` 2.66 → 5.14) than when only layer 1 was
constrained. The likely mechanism is that `NumSpikes` pins the *output* firing rate, so
with layer 2 also held under a ceiling, layer 1 must drive it harder to keep the output
on target — and the layer-1 ceiling is soft enough at strength 10 to permit it.
**Constraining layer 2 partly undoes the layer-1 ceiling.** That interaction has no
analogue in either single-layer script and is the specific thing this arm needs
recalibrated.

**Verdict: the delay arm needs one ceiling calibration run before the grid** — see
step 1b. Everything else about it passed.

#### What the probe does *not* settle: the design acceptance check

Both scripts printed it, and at n = 4 **it should not be read**: PASS at −0.400 in the
no-delay arm and FAIL at −0.400 … −1.000 in the delay arm. With only two `k` levels the
correlation is dominated by the floor column and a Spearman on 4 points is degenerate
(layer 1's −1.000 simply means 4 points happened to be monotone). The real check needs
the 4-level, 2-seed grid — the 1st-layer experiment's own probe-stage figure was −0.238
against a final grid value of −0.053.

The one diagnostic that *is* informative already: **ρ(`s1`, `s2`) came back +1.000 in
both arms, against v1's −0.829 (no-delay) and +0.203 (delay).** The column knob moves
both layers' selectivity together, which is precisely the failure of the single-layer
design that §1 point 2 was written about. Degenerate at n = 4 in magnitude, but the sign
flip is the design's central claim and it is in the right direction.

### Step 1b — delay-arm ceiling calibration — **RUN, AND IT DIAGNOSED SOMETHING ELSE**

Ran `k1 = 1`, both floor columns, 400 ep, seed 42, at `CEILING_STRENGTH = 20`, under
`_ceilcal_str20` tags so the strength-10 probe was preserved (~1.5 h).

| floor | strength | acc | `a1` | `s1` | over-`k1` | `a2` | `s2` | over-`k2` |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | .717 | 4.20 | 75.8% | 14.4% | 4.41 | 50.8% | 31.0% |
| 0 | **20** | **.564** | 2.42 | 85.6% | 7.2% | 2.78 | 51.5% | 20.2% |
| 1 | 10 | .758 | 5.14 | 18.1% | 62.2% | 5.61 | 7.3% | 65.8% |
| 1 | **20** | **.698** | 3.16 | 31.4% | 38.8% | 3.70 | 11.4% | 50.7% |

Strength 20 does tighten the ceiling, but it costs **.153** of accuracy in the floor-off
column (.717 → .564) — worse than the .13 that the 3 → 10 step cost at layer 1, and it
puts that corner far below v1's .78–.88. On its own this table would say "20 is too
expensive, keep 10 and report a soft row axis".

**But the real problem is not the ceiling strength, and this table is not the evidence
that matters.**

#### The finding: in the delay arm's floor-ON column, the saved checkpoint predates the constraint

Best-model selection uses the **task** validation loss (document 5's deliberate choice:
"the saved checkpoint is chosen on the task rather than on how well the constraint is
satisfied"). In the delay arm's floor-on cells the task val loss is best ~20 epochs after
the penalties engage and never improves again, so early stopping fires at ~340 and **the
checkpoint that gets saved and measured is the one from epoch 37–45** — one that has had
the constraint applied for roughly twenty epochs.

Constraint state at the saved epoch versus at the last epoch trained (train set):

| cell | saved / last epoch | `a1` | over-`k1` | `a2` | over-`k2` |
|---|---|---|---|---|---|
| `k1` floor 1, str 10 | **41 / 341** | 4.94 → **2.27** | 62.5% → **36.9%** | 5.48 → 3.31 | 66.4% → **48.8%** |
| `k8` floor 1, str 10 | **37 / 337** | 7.29 → **4.07** | 28.9% → **7.2%** | 10.23 → 8.15 | 14.4% → **4.5%** |
| `k1` floor 1, str 20 | **45 / 345** | 3.08 → 1.56 | 39.5% → 21.9% | 3.60 → 2.44 | 51.8% → 31.8% |
| `k1` floor 0, str 10 | 394 / 399 | 4.04 → 4.04 | 14.8% → 14.6% | — | — |
| `k8` floor 0, str 10 | 385 / 399 | 4.60 → 4.59 | 6.2% → 6.3% | — | — |

**So the "62.2% ceiling leak" reported above is a model-selection artifact.** The same
run at strength 10, trained out, sits at **36.9%** — inside the 30–40% band the 1st-layer
grid ran at. The row axis is also *wider* at the last epoch (`a1` 2.27 → 4.07 across
`k`, 1.79×) than at the saved checkpoints (4.94 → 7.29, 1.48×). `CEILING_STRENGTH = 10`
was never the problem, and raising it to 20 buys a tighter constraint by paying .153 of
accuracy for a fix the run did not need.

Only the **floor-on** cells are affected; the floor-off cells select at epoch 385–397 and
their constraint state is identical at best and last.

#### The no-delay arm is not affected — its grid is safe to keep running

| cell | saved / last | `a1` at saved → last | over-`k1` saved → last |
|---|---|---|---|
| `k1` floor 0 | 326 / 399 | 2.54 → 2.40 | 6.8% → 6.6% |
| `k1` floor 1 | 398 / 399 | 1.84 → 1.84 | 33.8% → 34.2% |
| `k8` floor 0 | 396 / 399 | 3.52 → 3.48 | 3.2% → 3.0% |
| `k8` floor 1 | 398 / 399 | 3.80 → 3.79 | 5.2% → 5.1% |

Every cell selects late and the constraint state is unchanged between the saved and final
epoch. No action needed.

#### It also touched the completed 1st-layer v3 delay grid, in 2 of 16 cells

Auditing `best_epoch / n_epochs` across that grid (document 5's Phase 1 result): 14 of 16
cells select at fraction 0.71–1.00, i.e. healthy. The exceptions are both `k = 1`,
floor-on: `seed42` at 433/734 (0.59) and **`seed43` at 48/349 (0.14)** — the same
pathology. That grid's headline conclusion rests on 16 models of which 14 are clean, so
it is probably robust, but the two `k=1` floor-on cells are less constrained than the
design intended and this is worth noting against any re-reading of β_a there.

#### The candidate fix, and why it is nearly free

Track the best validation loss only from `warmup_epochs + settle`, with `settle` long
enough for the constraint to bind (~150 epochs). This keeps selection on the **task**
loss — document 5's design intent — while refusing checkpoints from before the
manipulation exists.

It is close to a no-op on healthy runs: applied to the completed 1st-layer delay grid at
1250 epochs it would change the chosen checkpoint in **1 of 16 cells** (the 48/349 one),
because the other fifteen already select at epoch 433 or later. So it repairs the
pathological cells without disturbing comparability with the grids already completed.

#### Resolution — adopted 2026-07-31

`SETTLE_EPOCHS = 150` is now implemented in
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py):
best-model tracking *and* early stopping are held off until
`warmup_epochs + settle_epochs` = epoch 170. `CEILING_STRENGTH` stays at **10**; the
strength-20 run is kept on disk as evidence but is not adopted.

One useful side effect: at 400 epochs early stopping can now no longer fire at all
(170 + 300 > 400), so the probe trains every cell to full length. In the 1250-epoch grid
it can fire from epoch 470.

**The two arms now differ in selection policy, deliberately.** The no-delay script does
*not* carry the window, for two reasons: the pathology is measurably absent there (every
probe cell selects at epoch 326–398 of 400 with an unchanged constraint state), and its
16-model grid was already running when the fault was found, so adding it would desync the
script from the artifacts it is producing. The no-delay script carries a comment
recording the asymmetry, the evidence, and an instruction to add the window if that arm
is ever re-run — where it would be free, since a settle window of 150 would not have
changed any cell it has produced.

#### Delay re-probe with the settle window — **DONE, 2026-07-31 (2 h 23 m)**

Re-ran the 4 corners at `CEILING_STRENGTH = 10` with `SETTLE_EPOCHS = 150`. The
pre-settle artifacts are preserved under `_probe_presettle` tags (the re-probe reuses the
same `_probe` names).

| `k1`/`k2` | floor | acc | `a1` | `s1` | over-`k1` | `a2` | `s2` | over-`k2` | `a_net` | `s_net` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 / 2 | 0 | .717 | 4.20 | 75.8% | 14.4% | 4.41 | 50.8% | 31.0% | 4.34 | 63.3% |
| 1 / 2 | **1** | **.762** | 2.33 | 24.9% | **36.0%** | 3.37 | 9.9% | **49.1%** | 2.90 | 17.4% |
| 8 / 16 | 0 | .765 | 4.73 | 53.9% | 6.6% | 9.10 | 42.8% | 5.4% | 7.15 | 48.4% |
| 8 / 16 | **1** | **.806** | 4.24 | 7.4% | **8.3%** | 8.32 | 0.9% | **4.9%** | 6.35 | 4.2% |

**The window does exactly what it was meant to and nothing else.** Selection now lands at
epoch 341–394 in every cell, and no cell early-stops (all four train the full 400
epochs). The two floor-**off** cells are byte-identical to the pre-settle run, which is
the control: they already selected late, so the change could not and did not touch them.

The `k1 = 1`, floor-on cell is the proof. Its global `argmin(val_loss)` is still at epoch
41 — the task loss genuinely never recovers — but the rule now refuses that checkpoint
and takes epoch **341** instead:

| `k1=1`, floor 1 | pre-settle (saved ep 41) | post-settle (saved ep 341) |
|---|---|---|
| clean acc | .758 | **.762** |
| `a1` | 5.14 | **2.33** |
| over-`k1` | **62.2%** | **36.0%** |
| over-`k2` | 65.8% | **49.1%** |

**Row axis, which is what the fault was costing:**

| span of `a`, `k1` 1 → 8, floor ON | pre-settle | post-settle |
|---|---|---|
| layer 1 | 1.46× | **1.82×** |
| layer 2 | 1.87× | **2.47×** |
| **network** | 1.67× | **2.19×** |

2.19× network-wide now exceeds the completed 1st-layer delay grid's final 2.18× and sits
alongside the no-delay arm's 2.08×. Floor-off spans are unchanged by construction.

**It cost nothing.** Clean accuracy *rose* in both affected cells (.758 → .762,
.786 → .806), so the range is now **.717–.806** against v1's .78–.88 — the top of the
probe is inside v1's range, where before it was below. No inflation at either layer
(sp1 ≤ 3.93 against 11.66; sp2 ≤ 8.25 against 23.93), and the floor still separates `s`
by **+51.0/+41.0** and **+46.6/+41.9** points.

**Residual limitation, stated rather than hidden.** Layer 2's ceiling still leaks
**49.1%** at the tightest row (`k2 = 2`). That is worse than layer 1's 36.0% and worse
than the 41% the 1st-layer delay grid ran at, so layer 2's row axis is the softer of the
two. Document 5 accepted the same property in its own words — "a soft, manipulated axis
with real range, not a pinned one" — and the span it was compromising has now recovered,
so this is a limitation to report alongside β_a rather than a blocker. At `k2 = 16` the
leak is 4.9%.

**Verdict: the delay arm is ready for its grid**, at `CEILING_STRENGTH = 10` with
`SETTLE_EPOCHS = 150`.

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
- [x] **Corner probe, both arms** — 8 models, 400 ep, 5.9 h, 2026-07-31
- [x] Probe criterion 1 — **`WARMUP_EPOCHS = 20` survives the two-penalty
      configuration**: the `k = 1`, floor = 0 corner holds at acc .441 (no-delay) and
      .717 (delay), neither layer fully silent. The absorbing failure this guarded is
      ruled out
- [x] Probe criterion 3 — **`FLOOR_STRENGTH = 1` separates `s` at BOTH layers in both
      arms**: +63.1/+57.7 and +52.2/+53.1 (no-delay), +57.7/+43.5 and +48.4/+41.1
      (delay). Every cell OK, with large margin
- [x] Probe criterion 4 — no rate inflation at either layer in either arm
- [x] Probe criterion 5 — accuracy .441–.555 (no-delay, v1: .49–.59) and .717–.786
      (delay, v1: .78–.88, and better than the 1st-layer arm's .685–.771 at strength 10)
- [x] **`CEILING_STRENGTH = 10` validated for the no-delay arm** — layer 1 reproduces
      its single-layer sibling to within ~2 points of `s` and 0.1 spikes of `a` in every
      cell, so the layer-2 penalty constrains layer 2 without disturbing layer 1
- [x] ρ(`s1`, `s2`) came back **+1.000** in both arms against v1's −0.829 / +0.203 —
      the column knob is a network-wide selectivity manipulation (degenerate in
      magnitude at n = 4; the sign is the point)
- [x] Delay-arm ceiling calibration run at `CEILING_STRENGTH = 20` (~1.5 h) — **tested
      and rejected**: it tightens the ceiling but costs **.153** of clean accuracy in
      the floor-off column (.717 → .564), far below v1's .78–.88. Kept on disk under
      `_ceilcal_str20` tags
- [x] **Model-selection fault found and fixed.** In the delay arm's floor-ON column the
      task val loss peaks ~20 epochs after the penalties engage and never recovers, so
      the saved checkpoint came from epoch 37–45 of ~340 — barely constrained. The
      apparent "62.2% ceiling leak" was this artifact: trained out, the same strength-10
      run sits at **36.9%**, inside the band the 1st-layer grid ran at, with a *wider*
      row axis (1.79× vs 1.48×). `SETTLE_EPOCHS = 150` adopted in the delay script;
      `CEILING_STRENGTH = 10` stands
- [x] Confirmed the fault does **not** affect the no-delay arm (all cells select at
      epoch 326–398 of 400, constraint state unchanged saved → last), so its running
      grid is sound
- [x] Audited the **completed 1st-layer v3 delay grid** for the same fault: 14 of 16
      cells healthy (select at fraction 0.71–1.00); the two `k=1` floor-on cells are not
      (seed42 433/734, **seed43 48/349**). Document 5's conclusion rests mostly on clean
      cells but those two are less constrained than intended
- [x] Smoke-tested both scripts on GPU after the `SETTLE_EPOCHS` change — all checks
      pass; the settle guard is wired into `train_model` and tracking starts at epoch 170
- [x] Pre-settle delay probe artifacts preserved under `_probe_presettle` tags (4
      checkpoints, 4 training logs, 1 summary) — they are the evidence for the
      model-selection finding and the re-probe writes to the same `_probe` names
- [x] **Delay-arm re-probe LAUNCHED** 2026-07-31 with the settle window at
      `CEILING_STRENGTH = 10` (~3 h; longer than the first probe because no cell can
      early-stop at 400 epochs any more)
- [x] **Delay re-probe PASSES.** Selection moves to epoch 341–394 in every cell and no
      cell early-stops; the two floor-off cells reproduce the pre-settle run exactly
      (the control). At `k1 = 1` floor-on: over-`k1` **62.2% → 36.0%**, `a1` 5.14 → 2.33,
      and the network row axis **1.67× → 2.19×**, now above the completed 1st-layer delay
      grid's 2.18×. Accuracy *rose* (.717–.806 against .717–.786), so the fix cost
      nothing. Floor separation +51.0/+41.0 and +46.6/+41.9; no inflation
- [ ] **NO-DELAY 16-model grid — READY FOR THE REMOTE SERVER.** `QUICK_TEST = False`;
      `CEILING_STRENGTH = 10`, `FLOOR_STRENGTH = 1`, `WARMUP_EPOCHS = 20`, no settle
      window (measured unnecessary in this arm). ~2 h/model locally → scale to the
      remote GPU
- [ ] **DELAY 16-model grid — READY FOR THE REMOTE SERVER.** `QUICK_TEST = False`;
      same constants plus `SETTLE_EPOCHS = 150`. `CEILING_K` = [1,2,4,8] with layer 2 at
      ×2 → [2,4,8,16]
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
