# 2nd-layer v3 — the same dissociation factorial, one layer deeper

**Status (2026-08-09): BOTH GRIDS TRAINED AND SWEPT. 54 models at 1250 epochs (24
no-delay, 30 delay, 3 seeds each), all four eval-only perturbations run over both arms,
and the design acceptance check passes on the real grid: ρ(`a`, `s`) = +0.310 (n = 24)
and −0.052 (n = 30), against observational baselines of +0.829 and +0.427. Accuracy
figures are in
[result_visualization/2ndLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/2ndLayer/results_visualization.ipynb).**

**UPDATE 2026-08-09: the analysis is now run (§4 step 4).** Availability, usage and the
Phase 1 regression were measured at layer 2 with the retargeted scripts, as part of
document 7's three-way comparison. **β_a = −0.007 (p=.53) no-delay and +0.087 (p<.0001)
delay** — the latter the largest standardised **β_a** of any grid here (β\* = +0.638)
and running **against** H1′, which predicted β_a < 0. **Primary arm: delay**, on a
readout gap of +0.011 against the no-delay arm's +0.218. Read alongside the both-layer
grid, β_a is significant in 3 of 4 single-layer cells and 0 of 2 both-layer cells — the
compensation §1 argues about, appearing in a coefficient
([document 7 §6j](sparse_network_bothLayer_test_progress.md)). What remains is the
layer-2 **results document**.

The 1st-layer v3 factorial finished on 2026-07-29 with a null: `a` and `s` were
decorrelated by construction (ρ = −0.053), and H1′ still gained no support — β_a's point
estimate ran *against* it, and the residual signal was selectivity plus general
robustness. That result is about **layer 1**. This document is the same experiment moved
to **layer 2**, which is the cheapest way to find out whether the null is a fact about
temporal coding or a fact about the first layer of this particular network.

**The calibration finding that justifies the whole exercise:** the two layers are not
two copies of the same thing. Measured across all 27 v1 checkpoints, ρ(`a`, `s`) at
layer 2 is **+0.829** (no-delay) and **+0.427** (delay) — the *opposite sign* from
layer 1's −0.943 / −0.455. Layer 2 carries its own confound, running the other way, and
in the no-delay arm it is comfortably past the |ρ| ≤ 0.5 acceptance threshold. The
factorial is as necessary here as it was at layer 1, and it is a **different** confound
being broken. **Owner:** _(you)_

**This is document 6 of 7. It is a sibling of documents 5 and 7, not a successor —
read document 5 first.**

| # | Document | What it is |
|---|---|---|
| 1 | [sparse_network.md](sparse_network.md) | the question and the conceptual landscape — **start there** |
| 2 | [sparse_network_test_progress.md](sparse_network_test_progress.md) | v1 execution log |
| 3 | [sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](sparse_network_test_progress_v2.md) | v2 execution log |
| 5 | [sparse_network_test_progress_v3.md](sparse_network_test_progress_v3.md) | **v3 at layer 1** — design, Phase 0, Phase 1 result. Everything here assumes it |
| 6 | **this file** | v3 at layer 2 — calibration, design deltas, execution, sweeps |
| 7 | [sparse_network_bothLayer_test_progress.md](sparse_network_bothLayer_test_progress.md) | v3 at **both** layers — a third question, not a rephrasing of this one |

Layer 1's *results* are reported separately in
[sparse_network_1stLayer_results_v3.md](sparse_network_1stLayer_results_v3.md) — the
regressions, the H1′ refutation on relocation and jitter, and the two caveats that
qualify it. That document is what this experiment's layer-2 numbers have to be read
against, and its layer-2 counterpart does not exist yet (§4 step 4).

Everything in document 5 that is not about *which layer* carries over unchanged: the
two-variable decomposition of sparsity (§3b), the H1′/H2 restatement (§3c), the
capacity-matched decoder (§3a), the mechanism of the two penalties (§4b), and the
reading table for the outcome (§5). This document records only what had to be
**re-measured, re-decided, or is newly at risk** because the target moved — plus this
experiment's own execution log (§4), which now includes what the grids and the sweeps
came back with.

---

## 1. Why run it at layer 2

Document 5's Phase 1 verdict was "H1′ unsupported, residual signal is `s` plus general
robustness, underpowered to settle H2". Two readings of that are live, and they are not
distinguishable from layer-1 data alone:

1. **The temporal-coding reading.** Timing dependence is simply not driven by temporal
   sparsity in this architecture, at any depth.
2. **The layer-1 reading.** Layer 1 sits directly on the input; its job is largely to
   re-represent a 700-channel spike train, and the network's timing computation may not
   live there. Document 5 §2d already found something of this shape from the other
   direction — the no-delay readout leaves +.300 of accuracy unextracted on *every*
   checkpoint, meaning layer 1's code and the network's use of it are only loosely
   coupled.

In the delay arm the second reading has particular force: **`delay1` sits immediately
after layer 1**, so the learnable-delay machinery that arm exists to study does its work
*between* layer 1 and layer 2. Layer 2 is the first place its output is visible. Testing
H1′ only at layer 1 tests it upstream of the mechanism.

This is a replication in venue, not a new hypothesis. H1′ and H2 are unchanged.

---

## 2. Calibration step 0 — what layer 2 actually looks like

**Script:** [layer2_baseline.py](../../exp_sparse_network/v3_analysis/layer2_baseline.py)
→ `v3_analysis/log/layer2_baseline_{arm}.json`. Eval-only, all 27 frozen v1
checkpoints, test split, ~3 min. Trains nothing.

This step exists because **none of the four calibrated constants could be inherited** —
every one was measured against layer 1's activity, and document 5's own history is that
these constants do not transfer for free (`CEILING_STRENGTH` had to go 3 → 10 when the
row axis came back at 1.4×; `WARMUP_EPOCHS = 0` was unsurvivable).

**Validation of the probe itself:** it reproduces layer 1's known figures exactly —
median spike at bin 17–20, p90 at 37–41, p99 at 52–57, last occupied bin 82–87, i.e.
document 5's `[0, 88)` bound and ρ(`a`, `s`) of −0.943 / −0.455. The layer-2 numbers
below come off the same code path.

### 2a. The headline table

| | layer 1 (no-delay) | **layer 2 (no-delay)** | layer 1 (delay) | **layer 2 (delay)** |
|---|---|---|---|---|
| `spikes_per_neuron` | 1.21 – 7.85 | **3.52 – 5.53** | 2.12 – 11.66 | **14.12 – 23.93** |
| `a` (per active neuron) | 3.06 – 11.49 | **7.49 – 12.46** | 3.82 – 15.47 | **20.71 – 30.44** |
| `s` (silent fraction) | 31.7% – 60.5% | **46.9% – 61.2%** | 24.6% – 55.1% | **4.4% – 39.6%** |
| **ρ(`a`, `s`)** | **−0.943** (p=1.4e−7) | **+0.829** (p=1.4e−4) | **−0.455** (p=.14) | **+0.427** (p=.17) |
| temporal support | `[0, 88)` | **`[0, 90)`** | `[0, 88)` | **`[0, 160)`** |
| median / p90 / p99 spike bin | 17–20 / 37–41 / 52–57 | **22–23 / 44–46 / 61–62** | 18–20 / 38–41 / 55–57 | **65–79 / 97–112 / 116–130** |
| median peak `u` of **silent** pairs | 0.19 – 2.08 | **0.00 – 0.36** | 0.00 – 2.13 | **0.65 – 1.26** |

*(All figures above are from this one measurement pass, so the layer-1 columns are the
full-27 values. Document 5 §2e quotes a 3-checkpoint spot check for the silent-`u` row —
same conclusion, slightly narrower ranges.)*

### 2b. Four things this settles

**1. ρ(`a`, `s`) flips sign, and the factorial is still required.** At layer 1, v1's
penalty drove `a` down while pushing `s` up — the confound document 5 exists to break. At
layer 2, which v1 never penalised directly, the two axes move **together**: as the
layer-1 penalty strengthens, layer 2 becomes both less active per neuron *and* less
silent. In the no-delay arm that is **+0.829**, well past the |ρ| ≤ 0.5 threshold. In the
delay arm **+0.427** sits just inside it, but on n = 12 at p = .17 that is a power
statement, not a clean bill of health.

The consequence for interpretation is worth stating plainly: this is the confound induced
in layer 2 by penalising *layer 1*. What a **layer-2** penalty would produce
unconstrained is not known and cannot be read off this table — v1 never ran that
experiment. The number's job is to be the observational baseline the factorial must beat,
exactly as −0.943 was at layer 1.

**2. Layer 2 in the delay arm fires about twice as hard as layer 1** (`a` of 20–30
against 4–15), where in the no-delay arm the two layers are comparable (7–12 against
3–11). This is the one measurement that **changed a design decision** — see §3a.

**3. The temporal support nearly doubles in the delay arm**, `[0, 88)` → `[0, 160)`,
because `delay1` (up to 64 bins) shifts layer 1's spikes later before `fc2` ever sees
them. The median layer-2 spike lands at bin 65–79, where layer 1's lands at 18–20. Any
relocation or jitter perturbation measured against this layer **must** use the wider
window; re-using `[0, 88)` would discard more than a third of the layer's own support and
mix a large rate insult into a timing probe — the exact artifact document 5 §2a was
written to remove, reintroduced by inheriting a constant across a layer boundary. The
no-delay arm moves only 88 → 90.

**4. The floor's mechanism argument holds, so the instrument transfers.** Silent layer-2
pairs sit at a median peak potential of **0.00–0.36** (no-delay) and **0.65–1.26** (delay)
against `theta = 10` — as deeply silent as layer 1's. The dead zone that defeated every
count-based floor in v1/v2/v2.1/v2.2 is present here too, so the membrane-potential floor
is still the right instrument and a count-based one would still fail.

### 2c. A failure mode with no layer-1 analogue — measured, and absent

The floor works at layer 1 because `potential1 = fc1(psp(x))` is a convolution of the
*input*, so `∂u/∂W1` never vanishes however far below threshold a neuron sits. At
layer 2:

```
potential2 = fc2(psp(delay1(spike(potential1))))
∂potential2 / ∂W2 = psp(delay1(hidden1))
```

Still surrogate-free — so the mechanism survives the move. **But it is exactly zero for
any sample whose entire layer 1 is silent**, and such a pair is reachable only back
through `fc1`, i.e. through the very surrogate the floor exists to avoid.

Measured across all 27 checkpoints, both arms: **0.00% of samples** have an entirely
silent layer 1. The risk is real in principle and absent in this network.

It is also *demonstrably* reachable rather than hypothetical — the implementation smoke
test forces layer 1 to zero and confirms `∂(floor)/∂W2` becomes exactly 0, while on a
trained checkpoint it is 0.49–0.75. So this is a documented edge with a known trigger,
and it is the first thing to check if the floor ever stops biting. It becomes live only
if the ceiling is ever driven hard enough to empty layer 1 — which the floor-off column
at small `k` is the corner most likely to do.

---

## 3. What changed in the design, and what did not

Both scripts are forks of their 1st-layer siblings. Data, splits, optimiser, schedule,
early stopping, logging, the two penalty *forms* and the acceptance criteria are all
unchanged. The parameter set is untouched, so these checkpoints load into the existing
eval pipeline exactly as v1's do (verified with `strict=True` against both arms).

**Implemented in:**
[sn_2ndLayer_train_noDelay_v3.py](../../exp_sparse_network/sn_2ndLayer_train_noDelay_v3.py)
and
[sn_2ndLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_2ndLayer_train_withDelay_v3.py).

**The mechanical change.** `forward(return_hidden=True)` now returns
`(out, hidden2, potential2)`; both penalties and all five logged statistics act on layer
2. The forward pass is split into `_first_hidden` / `_second_hidden` / `_output` so the
constrained layer is a named stage rather than an inline expression.

**Artifacts are tagged `v3L2`.** This matters more than usual: the two grids share
dataset, arm, `k`, floor and seed, so without a distinct tag the layer-2 run would
silently overwrite the completed layer-1 delay grid. Run tags are
`sparse_whole_{arm}_v3L2_k{k}_floor{f}_seed{seed}`, and each summary row carries
`target_layer: 2`.

### 3a. The one substantive design change: the delay arm's row axis

| | measured layer-2 natural `a` | `CEILING_K` | why |
|---|---|---|---|
| no-delay | 7.49 – 12.46 | `[1, 2, 4, 8]` — unchanged | `k = 8` already sits at the top of the natural range |
| **delay** | **20.71 – 30.44** | **`[1, 2, 4, 8, 16]`** | with `k = 8` the *loosest* row is still a ~3× cut, so the top of the `a` axis would sit unanchored far below where the layer naturally runs |

Document 5 diagnosed exactly this failure at layer 1 and could not fix it: *"adding a
no-ceiling row (or `k = 16`) would anchor the top of the `a` axis and widen the span from
~2× toward v1's 3.8× … Not applied, because the delay arm is already running and changing
the row levels mid-flight would cost the cross-arm comparison."* Nothing is running here
yet, so it is applied. It is also independently supported: on the *unconstrained* v1
`str0.01_seed42` checkpoints, 76.2% of layer-2 pairs already exceed `k = 1` in the delay
arm against 42.9% in the no-delay arm — an indicative figure from the 16-sample
implementation smoke test rather than a test-split measurement, but the 1.8× gap matches
the natural-`a` gap in the table above.

**Cost: 20 models rather than 16, ~46 h rather than ~37 h.** This is the one decision
here that is a judgement call rather than a measurement, and it is a one-line revert
(`CEILING_K` in the delay script) if the extra ~9 h is not wanted.

### 3b. The constants, and how each was settled

| constant | value | status |
|---|---|---|
| `NATURAL_SPIKES_PER_NEURON` | 5.53 / 23.93 | **measured for layer 2, per arm** (§2) |
| temporal support | `[0, 90)` / `[0, 160)` | **measured for layer 2, per arm** (§2) |
| `CEILING_K` | `[1,2,4,8]` / `[1,2,4,8,16]` | **set from the measured natural `a`** (§3a) |
| `CEILING_STRENGTH` | 10.0 | inherited → **validated at layer 2, unchanged** (§4) |
| `FLOOR_STRENGTH` | 1.0 | inherited → **validated at layer 2, unchanged** (§4) |
| `WARMUP_EPOCHS` | 20 | inherited → **validated at layer 2, unchanged** (§4) |

Both scripts now ship with `QUICK_TEST = False` and `SEEDS = [42, 43, 44]`, because both
grids have been run (§4 step 2). A re-run of either script as committed retrains the
whole grid — 24 or 30 models — and overwrites its checkpoints. Set `QUICK_TEST = True`
first if what is wanted is a probe.

The two paragraphs below record why two of the three were suspect *before* the probe.
Both concerns were tested and neither materialised; they are kept because they name the
failure signatures to watch for if the grid is ever re-tuned.

- **`CEILING_STRENGTH` is more likely too weak than too strong here.** Layer 2 starts
  about twice as dense in the delay arm, so the ceiling has further to push at every `k`
  — and at layer 1 even strength 10 left 41% of pairs above `k` at `k = 1`. The
  `over_k` column is what says whether it binds; raise this before widening `k` further.
- **`WARMUP_EPOCHS = 20` guards an absorbing failure, and layer 2 is arguably *more*
  exposed.** The ceiling is minimised at `count = 0`, so in the floor-off column only the
  task loss opposes silence. Silencing layer 2 cuts the output layer off from the input
  entirely, and the recovery path runs back through the surrogate at **both** `fc2` and
  `fc1` rather than at `fc1` alone. The `k = 1`, floor = 0 corner is the cell that would
  show it: `val_acc` pinned at chance (5%) with `silent → 100%`.

---

## 4. Execution — what has happened, and what has not

Steps 1–3 are done and each carries its result inline. Step 4 is open and is the only
thing standing between this grid and an answer.

### Step 1 — corner probe, both arms (~4.8 h total)

`QUICK_TEST = True` is already set in both scripts. Runs `k ∈ {1, 8}` (no-delay) or
`{1, 16}` (delay) × floor ∈ {0, 1}, one seed, 400 epochs. Probe artifacts carry a
`_probe` suffix and cannot clobber a real run.

Acceptance criteria, unchanged from document 5 except that criterion 3 uses this layer's
own natural rate:

1. `silent_fraction` roughly constant as `k` varies *within* a column;
2. `silent_fraction` differing by **≥ 20 points** *across* columns at matched `k` — if
   not, raise `FLOOR_STRENGTH`;
3. `spikes_per_neuron` not inflated beyond **5.53** (no-delay) / **23.93** (delay);
4. clean accuracy comfortably above chance in every corner.

Plus one check that is specific to this experiment: **the floor-off column at `k = 1`
must survive**, per §3b.

**Read the probe for the manipulation only.** Every v1 model was still improving at
epoch 1250, so 400 epochs is a different regime, not a scaled-down one. Document 5 also
established that penalties calibrated at reduced epochs **re-densify** by the full run —
achieved `a` will sit *above* the probe's numbers.

#### Probe RESULT — both arms, 8 models @ 400 ep, ~4.7 h — **COMPLETE, 2026-07-30**

Run locally, sequentially on one GPU. No-delay 16:52–19:08, delay 19:08–21:33, both
exit 0.

**No-delay** (`F = 1`, `CEILING_STRENGTH = 10`, seed 42; natural sp/neuron **5.53**):

| `k` | floor | clean acc | `a` | `s` | sp/neuron | over-`k` |
|---|---|---|---|---|---|---|
| 1 | 0 | .541 | 3.84 | **71.41%** | 1.10 | 21.3% |
| 1 | **1** | .597 | 2.72 | **15.06%** | 2.31 | 57.4% |
| 8 | 0 | .574 | 5.90 | **63.13%** | 2.18 | 8.0% |
| 8 | **1** | .573 | 4.86 | **3.49%** | 4.69 | 10.7% |

**Delay** (`F = 1`, `CEILING_STRENGTH = 10`, seed 42; natural sp/neuron **23.93**):

| `k` | floor | clean acc | `a` | `s` | sp/neuron | over-`k` |
|---|---|---|---|---|---|---|
| 1 | 0 | .830 | 5.96 | **73.11%** | 1.60 | 22.3% |
| 1 | **1** | .876 | 3.34 | **16.53%** | 2.79 | 54.8% |
| 16 | 0 | .842 | 10.35 | **55.20%** | 4.64 | 6.9% |
| 16 | **1** | .888 | 8.80 | **1.36%** | 8.68 | 6.0% |

**Against the five criteria — all pass, in both arms:**

| # | criterion | no-delay | delay |
|---|---|---|---|
| 1 | `s` constant across `k` within a column | **PASS** — 8.3 / 11.6 pts drift | **PASS** — 17.9 / 15.2 pts |
| 2 | `s` separated ≥ 20 pts at matched `k` | **PASS** — **+56.4**, **+59.6** | **PASS** — **+56.6**, **+53.8** |
| 3 | firing not inflated past natural | **PASS** — max 4.69 vs 5.53 | **PASS** — max 8.68 vs 23.93 |
| 4 | accuracy above chance | **PASS** — .541–.597 (v1: .49–.59) | **PASS** — .830–.888 (v1: .78–.88) |
| 5 | `k=1`, floor=0 survives (layer-2 specific) | **PASS** — 71.4% silent, acc .541 | **PASS** — 73.1% silent, acc .830 |

**ρ(`a`, `s`) = +0.000 (p = 1) in both arms**, against the layer-2 observational
baselines of +0.829 and +0.427. On n = 4, one seed, 400 epochs this is indicative only —
a 2×2 factorial with monotone knobs produces exactly 0 almost by construction — and the
script recomputes it on the real grid. It is worth recording that the 3-epoch pre-flight
printed **+0.600**, so the number moved a long way once training actually happened.

**Four things the probe settles:**

1. **All three inherited constants transfer unchanged.** `WARMUP_EPOCHS = 20`,
   `FLOOR_STRENGTH = 1`, `CEILING_STRENGTH = 10` — validated *at layer 2*, in both arms,
   independently of layer 1. No change needed to any of them.
2. **Criterion 5, the layer-2-specific risk, did not materialise.** The absorbing
   all-silent state was the single largest threat to this design — silencing layer 2 cuts
   the output layer off entirely, with recovery running back through the surrogate at
   *both* `fc2` and `fc1`. The floor-off column at `k = 1` settled at 71–73% silent with
   usable accuracy in both arms. §2c's reachability edge stayed shut.
3. **Constraining layer 2 costs far less accuracy than constraining layer 1 did.** Every
   cell sits at or above the *top* of its arm's v1 range, where the layer-1 probe at the
   same settings ran .473–.536 (no-delay) and .685–.771 (delay). The delay arm's .888 at
   `k=16` floor-on is above every v1 delay model. This was not predicted and is the
   probe's most surprising result.
4. **The row axis is wider here than it ever was at layer 1.** The delay arm's floor-on
   column spans `a` = 3.34 → 8.80, **2.63×**, on two `k` levels at 400 epochs — against
   the layer-1 delay grid's **2.18×** across all 16 models at 1250. Since β_a's power was
   document 5's binding limitation, this is the most consequential number in the table.

**The `k = 16` row earns its cost.** Fitting `a ∝ k^0.35` to the floor-on cells puts
`k = 8` at `a ≈ 6.9`, i.e. a 2.06× span; the measured `k = 16` gives 2.63×. That is ~34%
more log-range on the axis the whole experiment is underpowered on, for 4 extra models.
Recommended to keep, and it is a one-line revert if not.

**One residual limitation, stated rather than hidden.** The ceiling leaks more at layer 2
than at layer 1: 57.4% (no-delay) and 54.8% (delay) of pairs still exceed `k = 1` in the
floor-on cells, against layer 1's 34.2%, and `a` settles ~2.7× above target. This is the
predicted consequence of layer 2 starting denser. It did **not** compress the row axis —
the spans above are wider than layer 1's — so it is a soft-axis caveat, not a blocker.
Raising `CEILING_STRENGTH` was considered and rejected: the span is already better than
layer 1's, accuracy headroom is not worth spending on marginal tightening, and v2's whole
history is that buying constraint tightness with accuracy destroys what is being
measured. Revisit only if β_a comes back underpowered on the real grid.

### Step 2 — the grids

| arm | models | est. cost |
|---|---|---|
| no-delay | 4 `k` × 2 floor × **3** seeds = **24** | ~42 h |
| delay | 5 `k` × 2 floor × **3** seeds = **30** | ~57 h |

**Three seeds, not two.** The plan above said 2 seeds; both scripts were run with
`SEEDS = [42, 43, 44]` to match layer 1's three, because the whole point is a
cross-layer comparison of β_a and a grid with fewer seeds would have had a different
standard error for no reason other than saving GPU time. Costs are scaled from the
probe's measured throughput (34.0 and 36.3 min/model at 400 ep on the local RTX 2050,
i.e. ~1.77 and ~1.89 h/model at 1250) and are upper bounds.

#### Grid RESULT — both arms, 54 models @ 1250 ep — **COMPLETE, 2026-08-07**

Trained under commit `85c9b34` (2026-08-07). Every one of the 54 runs went the full 1250
epochs; early stopping (patience 300) never fired. Per-model wall times are not
recoverable — the local training logs all carry a single sync timestamp — so the cost
column above stays an estimate.

Cells are seed means (3 seeds); `acc` carries its across-seed sd. `spn` is
`spikes_per_neuron`, `ovk` the fraction of pairs still above the ceiling.

**No-delay** (natural sp/neuron **5.53**, non-sparse baseline acc **.584**):

| cell | clean acc | `a` | `s` | spn | ovk |
|---|---|---|---|---|---|
| k1 floor0 | .602 ± .011 | 3.75 | **65.6%** | 1.29 | 25.8% |
| k1 **floor1** | .622 ± .006 | 2.72 | **14.3%** | 2.33 | 58.6% |
| k2 floor0 | .601 ± .008 | 3.83 | **63.1%** | 1.41 | 20.7% |
| k2 **floor1** | .630 ± .015 | 3.00 | **10.8%** | 2.68 | 43.0% |
| k4 floor0 | .603 ± .008 | 4.50 | **59.1%** | 1.84 | 15.4% |
| k4 **floor1** | .626 ± .016 | 3.65 | **6.4%** | 3.41 | 25.6% |
| k8 floor0 | .593 ± .010 | 5.50 | **54.7%** | 2.49 | 8.8% |
| k8 **floor1** | .607 ± .009 | 4.98 | **3.0%** | 4.83 | 11.4% |

**Delay** (natural sp/neuron **23.93**, non-sparse baseline acc **.890**):

| cell | clean acc | `a` | `s` | spn | ovk |
|---|---|---|---|---|---|
| k1 floor0 | .874 ± .007 | 4.42 | **45.7%** | 2.39 | 40.4% |
| k1 **floor1** | .895 ± .003 | 3.22 | **13.9%** | 2.78 | 59.5% |
| k2 floor0 | .878 ± .011 | 4.60 | **41.9%** | 2.66 | 35.2% |
| k2 **floor1** | .897 ± .009 | 3.48 | **10.3%** | 3.12 | 47.0% |
| k4 floor0 | .881 ± .006 | 5.48 | **39.9%** | 3.29 | 29.5% |
| k4 **floor1** | .901 ± .003 | 4.15 | **5.9%** | 3.91 | 32.7% |
| k8 floor0 | .890 ± .011 | 6.86 | **34.9%** | 4.47 | 19.0% |
| k8 **floor1** | .900 ± .010 | 5.76 | **2.1%** | 5.64 | 18.1% |
| k16 floor0 | .884 ± .006 | 10.41 | **27.4%** | 7.54 | 9.4% |
| k16 **floor1** | .892 ± .003 | 9.42 | **0.7%** | 9.36 | 7.4% |

**Against the five criteria — all pass, in both arms, on the real grid:**

| # | criterion | no-delay | delay |
|---|---|---|---|
| 1 | `s` constant across `k` within a column | **PASS** — 10.9 / 11.3 pts drift | **PASS** — 18.3 / 13.2 pts |
| 2 | `s` separated ≥ 20 pts at matched `k` | **PASS** — **+51.3…+52.7** | **PASS** — **+26.8…+34.1** |
| 3 | firing not inflated past natural | **PASS** — max 4.83 vs 5.53 | **PASS** — max 9.36 vs 23.93 |
| 4 | accuracy above chance | **PASS** — .593–.630 | **PASS** — .874–.901 |
| 5 | `k=1`, floor=0 survives (layer-2 specific) | **PASS** — 65.6% silent, acc .602 | **PASS** — 45.7% silent, acc .874 |

**Design acceptance — PASS in both arms.** ρ(`a`, `s`) = **+0.310** (n = 24, p = .14) in
the no-delay arm and **−0.052** (n = 30, p = .78) in the delay arm, per checkpoint,
against the observational baselines of **+0.829** and **+0.427**. The no-delay confound
is the one that mattered — it was the arm past the threshold — and the factorial cut it
from +0.83 to +0.31. The delay arm is decorrelated to three decimal places. Per setup
(seed-averaged) the figures are +0.312 and −0.065, i.e. the same picture.

**Five things the grid settles:**

1. **The row axis is the widest this experiment has had.** `a` spans **2.72 → 5.50**
   (2.02×) in the no-delay arm and **3.22 → 10.41** (**3.23×**) in the delay arm, against
   the layer-1 delay grid's 2.18× on all 16 of its models. Within the floor-on column
   alone the delay span is 2.92×, comfortably past the probe's projected 2.63×. Since
   β_a's power was document 5's binding limitation, this is the most consequential number
   here — and it is what the `k = 16` row was bought for.
2. **`a` did not re-densify; `s` did.** Document 5's warning was that penalties
   calibrated at 400 epochs come back weaker at 1250. On `a` that did not happen —
   matched cells moved by ~0.1 (no-delay k1 floor-on 2.72 → 2.72, k8 floor-on 4.86 →
   4.98). On `s` it happened hard, and only in the floor-**off** column: delay k1 floor0
   fell 73.1% → 45.7% and k16 floor0 55.2% → 27.4%. The floor-on column barely moved
   (16.5% → 13.9%). Longer training de-silences the *unconstrained* column, so the
   probe's +53.8…+59.6-point separation over-states the real grid's +26.8…+34.1. Still a
   pass, with less margin than advertised.
3. **Constraining layer 2 costs no accuracy — it buys some.** Every no-delay cell
   (.593–.630) sits **above** the non-sparse baseline's .584 and above the top of v1's
   .49–.59 range. In the delay arm, .874–.901 straddles the baseline's .890, with 5 of 10
   cells above it. The probe called this its most surprising result; the full grid
   confirms it at 1250 epochs and 3 seeds.
4. **The ceiling still leaks, as predicted, and still does not compress the axis.**
   `over_k` runs to 58.6% (no-delay) and 59.5% (delay) in the floor-on `k = 1` cells,
   against layer 1's 34.2%. The leak is real and `a` settles ~2.7× above target — but the
   axis it produced is wider than layer 1's, so the §4 decision not to raise
   `CEILING_STRENGTH` stands.
5. **Document 7's checkpoint-selection hazard does not touch this grid — audited, not
   assumed.** Best-model selection here runs on the *task* validation loss with no
   `SETTLE_EPOCHS` guard, which is exactly the failure that silently affected 2 of the 16
   cells in the completed 1st-layer delay grid. Across all 54 training logs,
   `argmin(val_loss)` lands at epoch **1104–1250** (median 1219 / 1226), i.e. long after
   both penalties have bound. No cell is at risk. Re-run this audit if either script is
   ever run at a shorter epoch budget.

Which arm is primary is **an open question here, not an inherited one.** Document 5's
delay-arm decision rested on two Phase 0 measurements of the *dependent* variable — the
readout-efficiency gap (+.300 vs +.026) and the deletion control being indistinguishable
from the timing probe in the no-delay arm (+0.936 vs +0.939). Both were measured **at
layer 1**. Neither has been measured at layer 2, and the readout gap in particular is a
different quantity there: at layer 2 the readout is `fc3` alone, not `fc2`+`fc3`. Deciding
the arm before re-measuring would be inheriting a constant across a layer boundary, which
is the mistake this whole calibration exists to avoid.

### Step 3 — retarget the measurement layer — **perturbations DONE, analysis NOT**

Every measurement script was hardwired to layer 1. Half of them have been moved.

| what | files | status |
|---|---|---|
| perturbation injection site | 8 new `*_2ndLayer_evalOnly_{noDelay,withDelay}_v3.py` under `{shd,jitter,shift,deletion}/` | **DONE** — inject at `hidden2` |
| perturbation window | same 8 | **DONE** — `SUPPORT_BINS = 90` / **`160`**, per arm |
| availability decode | `v3_analysis/hidden_channel_decode.py`, `SUPPORT_BINS = 90` | **not started** — still layer 1 |
| usage + deletion control | `v3_analysis/phase1_measure.py`, `SUPPORT_BINS = 88` | **not started** — still layer 1 |

The eight new eval scripts are forks, not edits, so the layer-1 scripts and their results
are untouched. Both traps carried over from document 5 §4 Phase 0 step 3 were handled,
and both are worth re-checking in any further fork:

- **The window correction is not uniform across perturbations.** Relocation and jitter
  confine destinations to `[0, SUPPORT_BINS)`; **shift and deletion were deliberately
  left alone** and clip to the full `[0, T-1]`. Clipping a rigid translation would pile
  spikes at the edge and merge them, destroying per-neuron count — a rate insult that
  does not currently exist. The shift scripts carry that reasoning in their module
  docstring so the "missing" constant is not read as an oversight.
- **`delay1` moves from downstream to upstream of the probe site.** A layer-1 probe
  correctly ignores `delay1`; a layer-2 probe **must apply it**. Getting this wrong
  silently measures the wrong tensor rather than raising an error.
  `layer2_baseline.py` already does it correctly and is the reference.

#### Sweep RESULT — 8 sweeps, both arms, all 54 checkpoints — **COMPLETE, 2026-08-07**

Four eval-only perturbations × two arms, 3 repeats per grid point, seed-averaged into
`{shd,jitter,shift,deletion}/log/sparse_whole_{arm}_v3L2_{stem}_eval.json`. Each file
carries `per_setup` (seed means and sds) and `per_checkpoint` (per-seed rows, tagged
`target_layer: 2`). Accuracy figures, the manipulation check and the numbers as a table
are in
[result_visualization/2ndLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/2ndLayer/results_visualization.ipynb).
What the sweeps show, descriptively, is directly below; the analysis that
settles anything is step 4.

#### What the sweeps show, descriptively

**This subsection is not the Phase 1 analysis and must not be quoted as it.** It is the
across-cell correlation of a single summary number against the two design axes, with no
deletion control in the model, no standardised coefficients, and no inference beyond a
Pearson r on 8 or 10 cells. It is here because the grid is decorrelated (ρ(`a`, `s`) =
+0.31 / −0.05), so unlike v1 these correlations at least *point* at one axis rather than
at a fused one. The real answer needs §4 step 4.

`retention` is the chance-corrected fraction of accuracy surviving the **strongest** grid
point of each sweep, computed per seed against that seed's own clean accuracy; the
analysis's usage score is `1 − retention`. Lower retention = more damage.

| arm | cell range | relocation | jitter | shift | deletion |
|---|---|---|---|---|---|
| no-delay | best → worst | .937 – .978 | .915 – .964 | .563 – .734 | .455 – .551 |
| delay | best → worst | .791 – .948 | .843 – .970 | .733 – .935 | .641 – .714 |

Correlating each cell's retention against `log(a)` and against `s`:

| arm | | relocation | jitter | shift | deletion |
|---|---|---|---|---|---|
| **no-delay** (n = 8) | r(log `a`) | −0.25 (p=.56) | +0.02 (p=.97) | **−0.94** (p<.001) | −0.72 (p=.04) |
| | r(`s`) | +0.62 (p=.10) | **+0.84** (p=.009) | −0.50 (p=.21) | **−0.84** (p=.009) |
| **delay** (n = 10) | r(log `a`) | **−0.96** (p<.0001) | **−0.97** (p<.0001) | **−0.98** (p<.0001) | −0.66 (p=.04) |
| | r(`s`) | −0.02 (p=.97) | +0.03 (p=.93) | −0.09 (p=.81) | −0.63 (p=.05) |

**Four readings, all provisional:**

1. **The sign is layer 1's sign.** Retention *falls* as `a` rises, i.e. usage rises with
   `a`, i.e. β_a > 0 — the direction **opposite** to H1′, which wants sparser codes to
   rely more on timing. Layer 1 refuted H1′ on relocation and jitter with the same sign.
   If the regression holds this up, the answer to "is the null a fact about layer 1?" is
   no.
2. **The delay arm is where the signal is.** All three timing probes track `log(a)` at
   −0.96…−0.98 and carry essentially nothing on `s`. The no-delay arm is the opposite
   picture: relocation and jitter barely bite at all (retention .92–.98 across the whole
   grid — there is almost no dynamic range to explain), and what varies loads on `s`.
3. **The deletion control does not explain the delay arm the way it explained layer 1.**
   In the delay arm the timing probes track `a` *more* tightly (−0.96…−0.98) than the
   pure rate insult does (−0.66). At layer 1, Phase 0 found the control and the timing
   probe indistinguishable in the no-delay arm (+0.936 vs +0.939). That is the single
   most encouraging number here — and it is exactly the comparison that needs
   `phase1_measure.py` at layer 2 to be made properly, since the control belongs *in* the
   model, not beside it.
4. **The floor-on / floor-off contrast at matched `k` is confounded and is not read
   here.** The floor also lowers `a` (e.g. delay `k=1`: 4.42 → 3.22), so the column
   contrast mixes both axes. The controlled version — matched `a`, floor on vs off — is a
   checklist item, not something to eyeball off this table.

### Step 4 — the analysis — **RUN 2026-08-09**

All four items below were run as part of document 7 §6j, which needed the layer-2 leg to
complete its three-way β_a comparison. The three analysis scripts now carry a `GRID` knob
selecting the constrained layer set, so `v3`, `v3L2` and `v3L12` are measured by **one**
script rather than three forks — which is what makes the comparison a comparison. See
[document 7 §6g](sparse_network_bothLayer_test_progress.md) for the retarget itself and
the verifications behind it.

Artifacts: `v3_analysis/log/{hidden_channel_decode_v3L2,phase1_measure_v3L2}_{arm}.json`,
`phase1_regress_v3L2.json`, `v3L2_pipeline.out`, and two figures under `v3_analysis/fig/`.

**1–3. Availability, usage and the regression.** 54 models (24 no-delay, 30 delay),
relocation confined to layer 2's per-arm window (`[0, 90)` / `[0, 160)`), deletion
unclipped, `delay1` applied because layer 2 is downstream of it.

| | no-delay (n=24) | delay (n=30) |
|---|---|---|
| ρ(`a`, `s`) — acceptance | **+0.054** (p=.80) PASS | **−0.084** (p=.66) PASS |
| achieved `a` span | 2.11× | **3.50×** — the widest of any grid |
| **β_a** (usage ~ log a + s + control) | −0.007 (p=.53) | **+0.087 (p<.0001)**, β\* **+0.638** |
| β_s | −0.110 (p<.0001) | −0.060 (p=.050) |
| matched floor-on vs floor-off | +0.026 (p=.012, 8 pairs) | −0.020 (p=.13, 12 pairs) |
| verdict | **H2 CONFIRMED** | **H1′ REFUTED** — β_a > 0 |

The delay arm's β_a is the **largest standardised β_a of any grid in this project**
(+0.638) on the widest axis (3.50×), so the layer-2 effect is neither marginal nor
underpowered — and it runs *against* H1′, which predicted β_a < 0.

**The arm decision, re-measured at layer 2.** The readout gap is **+0.011** in the delay
arm (positive on 83% of checkpoints) against **+0.218** in the no-delay arm (100%) — the
same asymmetry document 5 found at layer 1 (+.300 / −.013) and document 7 at the network
level (+0.285 / +0.076). **Delay is primary here too**, on evidence measured at layer 2
rather than inherited across a layer boundary.

**4. The comparison this experiment exists for — and it is bigger than layer 1 vs
layer 2.** With the both-layer grid also in hand, β_a is significant in **3 of the 4
single-layer cells and 0 of the 2 both-layer cells**, and the both-layer intervals
*exclude* the single-layer point estimates in 3 of 4 comparisons. The anti-H1′
coefficient exists only while **one** layer is constrained and vanishes when the network
is — which is the layer-2 compensation this document's §1 argued about, appearing in a
coefficient. Full table and reading in
[document 7 §6j](sparse_network_bothLayer_test_progress.md).


---

## 5. Watch out for

Inherits document 1 §7, document 4 §7 and document 5 §8 in full. New to the 2nd-layer
experiment:

1. **Never inherit a constant across a layer boundary.** Every one of the four that was
   *measured* turned out to differ, two of them by ~2×, and one (`ρ(a, s)`) changed
   *sign*. The three that were inherited survived the probe unchanged — but that was
   established by testing them at layer 2, not by assuming they carried.
2. **The `v3L2` tag is load-bearing.** The two grids differ only by target layer;
   without it the layer-2 run overwrites the completed layer-1 delay grid, whose
   re-training costs ~37 h.
3. **`delay1` is upstream of the constrained layer.** It shapes the activity the
   penalties charge, and it is why the delay arm's layer-2 support runs to bin 159. Any
   probe that ignores it is measuring a different network.
4. **The floor has a reachability precondition at layer 2** that it does not have at
   layer 1: a sample whose entire layer 1 is silent is invisible to it. Currently 0.00%
   of samples, but the floor-off column at small `k` is the corner that could create
   them (§2c).
5. **ρ(`a`, `s`) at layer 2 is positive.** A reader who carries document 5's −0.943
   across will misread the sign of the confound being broken. The design acceptance check
   is on |ρ|, so the threshold is unaffected — but the *interpretation* of a failure
   differs.
6. **The arm decision is not inherited** (§4 step 2). It rested on layer-1 measurements
   of the dependent variable that have no layer-2 counterpart yet, and §4 step 3's
   descriptive correlations are not a substitute for them.
7. **The probe over-states the floor separation.** Longer training de-silences the
   floor-**off** column while leaving the floor-on column alone, so the real grid
   separates by +27…+34 points in the delay arm where the 400-epoch probe promised +54…
   +57 (§4 step 2, finding 2). Calibrate `FLOOR_STRENGTH` against a *full-length* run
   before believing any margin measured at reduced epochs — the same lesson document 5
   learned for the ceiling, on the other axis.
8. **Best-model selection has no `SETTLE_EPOCHS` guard in either 2nd-layer script.** It
   happens not to bite — all 54 runs selected at epoch 1104–1250 — but that is a measured
   fact about this grid at 1250 epochs, not a property of the scripts. Any shorter run
   re-opens document 7's hazard: a checkpoint saved before the penalty bound makes every
   sparsity statistic wrong in the direction of "the penalty isn't working". Audit
   `argmin(val_loss)` against run length before trusting a cell.
9. **§4 step 3's correlations are descriptive.** Pearson r on 8–10 cells, no deletion control in the model, one
   summary statistic per sweep. It agrees with layer 1's sign, which is suggestive and
   nothing more. The claim "H1′ fails at layer 2 too" is not licensed until §4 step 4 has
   run.

---

## 6. Checklist

- [x] Both training scripts retargeted to hidden layer 2 — penalties, logged statistics,
      clean evaluation and summary all act on `hidden2`/`potential2`
- [x] Artifacts tagged `v3L2`; summary rows carry `target_layer`; probe runs carry
      `_probe` — no path can collide with the completed 1st-layer grids
- [x] Parameter set verified unchanged: both scripts load v1 checkpoints with
      `strict=True`, so the existing eval pipeline stays compatible
- [x] Implementation verified end to end — `forward` returns layer 2 and not layer 1,
      both penalties produce nonzero `fc2` gradients on a trained checkpoint (0.49–8.41),
      and the §2c edge case reproduces exactly 0 when layer 1 is forced silent
- [x] **Calibration step 0** — layer-2 baseline measured on all 27 v1 checkpoints, both
      arms ([layer2_baseline.py](../../exp_sparse_network/v3_analysis/layer2_baseline.py))
- [x] Probe validated against layer 1's known figures (support `[0, 88)`,
      ρ = −0.943 / −0.455 reproduced)
- [x] `NATURAL_SPIKES_PER_NEURON` measured per arm: **5.53** / **23.93**
- [x] Temporal support measured per arm: **`[0, 90)`** / **`[0, 160)`**
- [x] Floor mechanism re-validated at layer 2 — silent pairs at peak `u` of 0.00–1.26
      against `theta = 10`, so the potential floor is still the right instrument
- [x] Gradient-reachability risk identified and measured — **0.00%** of samples have an
      entirely silent layer 1
- [x] `CEILING_K` extended to `[1, 2, 4, 8, 16]` in the delay arm on the measured
      natural `a` of 20.71–30.44 (+4 models, ~+9 h; one-line revert)
- [x] **Corner probe, both arms** (8 models, 400 ep, ~4.7 h) — **all five acceptance
      criteria pass in both arms**; floor separation +53.8…+59.6 points, no rate
      inflation, accuracy at or above the top of each arm's v1 range
- [x] Probe — all three inherited constants **validated at layer 2 and unchanged**:
      `WARMUP_EPOCHS = 20`, `FLOOR_STRENGTH = 1`, `CEILING_STRENGTH = 10`
- [x] Probe — criterion 5 (the layer-2-specific absorbing-state risk) **cleared**:
      `k=1`/floor=0 settled at 71–73% silent, not 100%, with acc .541 / .830
- [x] Probe — `k = 16` row justified on measurement: floor-on span 2.63× against ~2.06×
      that `k = 8` alone would have given, and against layer 1's 2.18× on its full grid
- [x] Grid costs re-estimated from measured throughput: **~28 h** / **~38 h** (at 2 seeds;
      the grids were run at 3, so ~42 h / ~57 h)
- [x] **Both grids trained at 1250 epochs, 3 seeds** — no-delay 24 models, delay 30
      models, `QUICK_TEST = False`, commit `85c9b34` (2026-08-07). No run early-stopped
- [x] **Design acceptance check on the real grid — PASS in both arms**: ρ(`a`, `s`) =
      **+0.310** (n = 24) and **−0.052** (n = 30), against the observational baselines
      **+0.829** / **+0.427**
- [x] All five probe criteria re-checked on the full grid — pass in both arms; floor
      separation **+51…+53** (no-delay) and **+27…+34** (delay) points
- [x] Row axis measured on the full grid: `a` = 2.72–5.50 (**2.02×**, no-delay) and
      3.22–10.41 (**3.23×**, delay) — wider than layer 1's 2.18×, so the `k = 16` row paid
- [x] Checkpoint-selection audit against document 7's hazard — `argmin(val_loss)` at
      epoch **1104–1250** across all 54 runs, so no cell was saved before its penalty bound
- [x] **8 perturbation eval scripts retargeted to layer 2** at the per-arm window
      (`90` / `160`), shift and deletion deliberately left unclipped
- [x] **All four perturbations swept over both grids** (2026-08-07) — 8 result files,
      `per_setup` + `per_checkpoint`, every row tagged `target_layer: 2`
- [x] Accuracy figures, manipulation check and value table plotted
      ([results_visualization.ipynb](../../exp_sparse_network/result_visualization/2ndLayer/results_visualization.ipynb))
- [x] **Retargeted `hidden_channel_decode.py` (availability) and `phase1_measure.py`
      (usage + deletion control) to layer 2** at the per-arm window (`90` / `160`),
      deletion left unclipped, `delay1` applied because layer 2 sits downstream of it.
      Both now carry a `GRID` knob, so one script serves all three grids — see
      [document 7 §6g](sparse_network_bothLayer_test_progress.md) for the retarget and
      its verifications
- [x] **Dependent variable re-measured at layer 2 and the primary arm decided: DELAY** —
      readout gap **+0.011** (delay, positive on 83%) against **+0.218** (no-delay, 100%),
      the same asymmetry document 5 found at layer 1
- [x] **Phase 1 regression run at layer 2** (§4 step 4): acceptance passes in both arms
      (+0.054 / −0.084); **β_a = −0.007 (p=.53) no-delay** and **+0.087 (p<.0001),
      β\* = +0.638 delay** — the largest standardised β_a of any grid in this project,
      on the widest achieved `a` axis (3.50×), and it runs **against** H1′.
      Matched floor contrast +0.026 (p=.012) / −0.020 (p=.13)
- [x] **β_a compared across layer 1, layer 2 and both** — significant in **3 of 4
      single-layer cells, 0 of 2 both-layer cells**, with the both-layer intervals
      excluding the single-layer estimates in 3 of 4 comparisons. The anti-H1′ signal
      exists only while one layer is constrained; §1's compensation argument, in a
      coefficient. Table in
      [document 7 §6j](sparse_network_bothLayer_test_progress.md)
- [ ] Write the layer-2 results document, the counterpart of
      [sparse_network_1stLayer_results_v3.md](sparse_network_1stLayer_results_v3.md)
