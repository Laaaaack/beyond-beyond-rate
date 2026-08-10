# First Milestone v3 — separating the two things "sparsity" does (design, Phase 0, Phase 1)

**Status (2026-08-01): PHASE 0 COMPLETE. PHASE 1 COMPLETE IN BOTH ARMS —
48 models trained (4 `k` × 2 floor × 3 seeds × 2 arms), measured and analysed.**

§2 was five spot checks on 3 checkpoints per arm, made to decide what v3 should be.
§4's Phase 0 has now run all of them properly over all 27 checkpoints, plus the
correctness fix to the perturbation window. **The dissociation §2c predicted holds at
full n, and it is now the result:** availability is flat (ρ = −0.211, p = .45 /
−0.035, p = .91) while usage climbs steeply (+0.939 / +0.881).

Phase 0 also **changed which arm Phase 1 runs in first — the delay arm**, on two
findings about the *dependent* variable that a factorial cannot fix. Both training
scripts are written and three settings are calibrated on measured runs rather than
assumed: `WARMUP_EPOCHS = 20` (0 is unsurvivable), `FLOOR_STRENGTH = 1` (separates `s`
by ~50 points at matched `k`, at no accuracy cost), and `CEILING_STRENGTH = 10` (3 left
the H1 axis moving only 1.4×). Both arms **pass the design acceptance check** —
ρ(`a`, `s`) = **−0.087** (delay) and **−0.342** (no-delay), against v1's −0.455 and
−0.943. `a` and `s` are separable for the first time in this project, in both arms.

**The Phase 1 answer, now that n = 24 per arm: H1′ is refuted on its own terms, in
both arms independently.** β_a is *positive* and significant — the direction opposite
to H1′'s prediction — at β\* = **+0.394** (p = .033) in the delay arm and β\* =
**+0.495** (p = .0001) in the no-delay arm, with `s` held fixed *by construction* and
general robustness controlled. Selectivity carries a real effect in H2's direction
(β\* = −0.370, p = .045, delay), and the matched floor-on/floor-off contrast is
significant in the no-delay arm (+0.119, p = .0002). The earlier "underpowered"
verdict was exactly that: the n = 16 delay estimate (β\* = +0.30, p = .20) was the same
effect, unresolved. See §4's Phase 1 RESULT. **Owner:** _(you)_

**This is document 5 of 5. Read in order:**

| # | Document | What it is |
|---|---|---|
| 1 | [sparse_network.md](../../progress/sparse_network.md) | the question and the conceptual landscape — **start there** |
| 2 | [sparse_network_test_progress.md](sparse_network_test_progress.md) | v1 execution log — getting a sparsity gradient to exist at all |
| 3 | [sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, and the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](sparse_network_test_progress_v2.md) | v2 execution log — attempts to close the perturbation-immune channels |
| 5 | **this file** | v3 — why the question needs a different experiment, what it is, and the executed Phase 0 **and Phase 1** (§4) |

**The one-line thesis.** v1 and v2 both treat "sparsity" as one number and "temporal
processing" as one number. Neither is. Sparsity as induced by rate regularisation
moves **two** variables that push in **opposite** directions, and in v1's no-delay arm
they are correlated at **ρ = −0.94**, so that experiment structurally cannot tell them
apart. v3's job is to break that confound by design, and to measure the dependent
variable in a way that does not inherit the perturbation's own confounds.

---

## 1. Where v1 and v2 left the question

**v1 answered the question as posed, and the answer was no.** Across all 8
perturbation × arm combinations, sparser networks were *more* robust to hidden
perturbation, not less (ρ = +0.67…+0.96). The diagnosis in document 3 §6 was that
regularisation-induced sparsity arrives partly by neurons falling silent on some
stimuli, which builds a stimulus-selective **identity** code — and every
rate-preserving perturbation leaves neuron identity perfectly intact.

**v2 tried to remove that escape route and could not afford to.** Three mechanisms
(two-sided band, L1 exact-count, hard top-k truncation), ~8 h of probing, no full
sweep in any version, and 62–94 h still outstanding. v2.2 did move the manipulation
further than anything before it, but at .552 → .237 clean accuracy, with raw firing
inflated from ~6.3 to 12.5–21.9 spikes/neuron and 63% of surviving spikes crowded
into the first 10 ms.

Both versions share a hidden assumption worth naming: that the way to rescue the
experiment is to make the *network* simpler, by forcing the perturbation-immune
channels shut. §2 and §3 argue the opposite — the analysis was the thing that needed
to get sharper, and the channel that mattered can be **controlled** far more cheaply
than it can be **closed**.

---

## 2. Five measurements made before designing v3

All were run today against the existing v1 checkpoints; nothing was retrained. The
code is committed at
[v3_analysis/hidden_channel_decode.py](../../exp_sparse_network/v3_analysis/hidden_channel_decode.py)
and
[v3_analysis/relocation_window_check.py](../../exp_sparse_network/v3_analysis/relocation_window_check.py).
Decode figures below are seed-42 checkpoints only (3 per arm) — Phase 0 extends them
to all 27.

### 2a. The relocation endpoint is distorted by the padding window — but the conclusion survives

`shd_whole.mat` holds **100** time bins; `load_shd_data` zero-pads them to the
simulator's **200**. Measured on trained checkpoints, 1st-hidden-layer spikes
therefore never occur beyond **bin 86**, and 90% of them fall inside the first
**~39** bins. *(Phase 0 re-measured this over all 27 checkpoints with
[temporal_support.py](../../exp_sparse_network/v3_analysis/temporal_support.py): the
last occupied bin is **87**, not 86, so the bound adopted everywhere is
`[0, 88)`. Median spike falls at bin 17–20, p90 at 37–41, p99 at 52–57.)* But
`perturb_hidden_batch` relocates spikes uniformly across all 200,
so at `f = 1` about **57%** of relocated spikes land where no hidden spike ever
naturally occurs, and the population's instantaneous spike density drops by roughly
**5×**. That is a rate-like insult riding on top of the intended timing-only one — in
a study whose entire premise is that the two must be kept separate.

Re-running the `f = 1` endpoint with destinations confined to the measured support:

| Arm | `acc(f=1)` densest → sparsest, full window | …restricted to support | ρ full window | ρ restricted |
|---|---|---|---|---|
| no-delay (n=15) | .246 → .379 | .294 → .409 | **+0.918** | **+0.932** |
| delay (n=12) | .305 → .318 | .330 → .418 | **+0.769** | **+0.867** |

The artifact is real — it suppressed `acc(f=1)` by .05–.14 throughout — but the
headline correlation does not flip. It **strengthens**. So this is a correctness fix
to carry forward, not a rescue, and one candidate explanation for v1's result is now
eliminated. *(ρ differs slightly from document 3's +0.950 / +0.671 because these are
different relocation repeat seeds; same conclusion.)*

**Phase 0 confirmed this at the corrected bound `[0, 88)` and on the full sweep.**
Re-running the whole `f` grid through the patched eval scripts lifts `acc(f=1)` by
+.010…+.115 (no-delay) and −.002…+.157 (delay), and moves ρ(spikes/neuron,
temporal_score) from **+0.950 → +0.939** (no-delay) and **+0.671 → +0.853** (delay).
The artifact was **specific to relocation**: jitter, being local, shifted by at most
.026 and its ρ barely moved (+0.950 → +0.921, +0.832 → +0.811). Deletion and shift
need no correction at all — see §4's Phase 0 record.

### 2b. The two sparsity axes are collinear — this is the blocking flaw

`spikes_per_neuron = (1 − silent_fraction) × spikes_per_active_neuron`. The two
factors are different codes doing different things, and across v1's checkpoints:

| Arm | sp/neuron | sp/ACTIVE | silent | **ρ(sp/ACTIVE, silent)** |
|---|---|---|---|---|
| no-delay (n=15) | 1.21–7.85 | 3.06–11.49 | .32–.61 | **−0.943** |
| delay (n=12) | 2.12–11.66 | 3.82–15.47 | .25–.55 | **−0.455** |

In the no-delay arm — the one that produced v1's *significant* anti-H1 result, the
one that survived every control — the two are almost perfectly confounded. **No
analysis of that arm can attribute the effect to one rather than the other.** The
delay arm has genuine separable variance, but only n=12 to spend on it.

This is the single most important number in this document. It says v1's design, not
v1's execution, is what prevents an answer.

*Phase 0 reproduced both figures exactly on an independent code path:*
**ρ(a, s) = −0.943** (no-delay, n=15) and **−0.455** (delay, n=12). The blocking flaw
is confirmed, and §4's Phase 0 record shows it biting: the no-delay arm's usage effect
does not survive being partialled on `s`, and the delay arm's does.

### 2c. Timing information in the hidden layer is *flat* across sparsity

A capacity-matched decoding analysis (§3a defines it; the script documents it) fit on
the train split and scored on the test split:

| checkpoint | sp/neu | silent | COUNT | IDENT | FULL | SHUF | **TIMING** | **frac** |
|---|---|---|---|---|---|---|---|---|
| nodelay str0.01 | 6.50 | 33% | .760 | .461 | .851 | .646 | **+.205** | **.26** |
| nodelay str1 | 2.91 | 46% | .766 | .437 | .838 | .646 | **+.192** | **.24** |
| nodelay str10 | 1.42 | 54% | .704 | .529 | .820 | .615 | **+.206** | **.27** |
| delay str0.01 | 11.66 | 25% | .780 | .339 | .870 | .649 | **+.220** | **.27** |
| delay str3 | 3.01 | 51% | .747 | .411 | .854 | .645 | **+.208** | **.26** |
| delay str10 | 2.12 | 48% | .739 | .500 | .863 | .663 | **+.200** | **.25** |

Three readings:

1. **The hidden layer does carry timing information** — about a quarter of its
   decodable class information requires knowing *when* spikes occurred.
2. **That quantity does not move with sparsity.** Across a 4.6× (no-delay) and 5.5×
   (delay) change in firing rate, the timing fraction sits at .24–.27. Flat.
3. **The v1 diagnosis reproduces independently.** Identity decode rises with sparsity
   (.461→.529, .339→.500) while count decode falls slightly (.760→.704, .780→.739) —
   exactly document 3 §6, now on a separate implementation.

Point 2 is a *third* answer, distinct from both H1 and v1's finding, and it only
appears once availability is measured separately from usage.

**Phase 0 extended this from 6 checkpoints to all 27 and point 2 held.** Timing
fraction spans **.201–.242** (no-delay, mean .229) and **.188–.264** (delay, mean
.239) across a 3.8× and 4.1× change in `a` — ρ = **−0.211** (p = .45) and **−0.035**
(p = .91), and **+0.015** / **+0.212** once partialled on `s`. Nothing there is
distinguishable from flat. Point 3 held too: identity decode rises with sparsity in
both arms (ρ with spikes/neuron = **−0.721**, **−0.846**) while count decode falls
(**+0.913**, **+0.775**).

### 2d. The two arms' readouts differ enormously in efficiency

Comparing each network's own clean accuracy against a linear decoder on its 1st
hidden layer:

| checkpoint | network clean acc | FULL decode | gap |
|---|---|---|---|
| nodelay str0.01 | .594 | .851 | **+.257** |
| delay str0.01 | .883 | .870 | −.013 |

The delay network's readout extracts essentially everything a linear decoder can find
in layer 1. **The no-delay network's readout leaves a quarter of the task on the
table.** So the no-delay arm's `temporal_score` is substantially a statement about
`fc2`/`fc3`, not about layer 1's code — which is document 1 §7's "readout caveat"
turning out to be large rather than nominal, and it explains why the two arms behaved
so differently in v1.

*Phase 0, all 27 checkpoints:* the gap is **+.300 on average** in the no-delay arm
(range +.262…+.329, every checkpoint) and **+.026** in the delay arm (range
−.011…+.084). The caveat is not just large, it is **uniform** — it applies to every
no-delay model, not to a subset.

### 2e. Silent neurons are deeply silent, not marginally sub-threshold

Median peak membrane potential `max_t u(t)`, against `theta = 10`:

| checkpoint | silent (sample, neuron) pairs | active pairs |
|---|---|---|
| nodelay str0.01 | 2.05 | 49.2 |
| nodelay str10 | 0.68 | 23.0 |
| delay str0.01 | **0.00** | 119.7 |
| delay str10 | **0.00** | 24.0 |

Silent pairs receive essentially **no drive at all**, not slightly too little. This
explains why v2.2's floor penalty `relu(k − count)` needed strength 10, inflated raw
firing to 12.5–21.9, and still leaked: it reaches those pairs only through SLAYER's
surrogate gradient, which at `u ≈ 0` against `theta = 10` carries no signal. It also
tells us the fix, and sizes it — see §4b.

---

## 3. The reframe

### 3a. "Does more temporal processing" is two questions

| | Question | Measured by | v1 status |
|---|---|---|---|
| **Availability** | Does the hidden layer *contain* class information that requires timing? | capacity-matched decoding of the layer | never measured |
| **Usage** | Does the network's *own readout* depend on that information? | eval-only relocation at `f = 1` | this is v1's `temporal_score` |

§2c and §2a together say these **disagree**: availability is flat across sparsity
while usage falls. That is not a contradiction, it is the finding — *sparse networks
hold just as much timing information in layer 1, and use less of it.* Neither
measurement alone could have said that. v3 reports both, always.

**The capacity-matched decoder.** Four decoders on the same hidden activity, fit on
train, scored on test:

| view | features | dim | immune to relocation? |
|---|---|---|---|
| `COUNT` | per-neuron spike count | 128 | yes |
| `IDENT` | per-neuron count, binarised | 128 | yes |
| `FULL` | per-neuron counts in 10 time bins | 1280 | no |
| `SHUF` | `FULL` after redrawing each neuron's spikes from *that neuron's own* marginal temporal profile, count preserved exactly | 1280 | yes |

**`FULL − SHUF` is the timing information.** `SHUF` is the null that holds feature
dimension, decoder, sample count, per-neuron count and per-neuron average temporal
profile all identical, destroying only the sample-specific placement of spikes.
Contrasting `FULL` against `COUNT` instead would confound timing with dimensionality
— which is exactly why the two contrasts differ so much in §2c (.09 vs .21).

This measure needs no perturbation, so it inherits none of the perturbation's
problems: no magnitude mismatch across densities (document 3 caveat 1, "the main
residual threat"), no contamination by the readout's general robustness, and no
requirement that the count channel be closed.

### 3b. "Sparsity" is two variables, and they oppose each other

| Variable | What it is | What it does to the code | H1's claim |
|---|---|---|---|
| **temporal sparsity** `a` | spikes per *active* neuron per sample | fewer spikes per neuron ⇒ count resolution per neuron falls ⇒ latency coding | **this is the one H1 is about** |
| **selectivity** `s` | fraction of (sample, neuron) pairs silent | neurons silent on some stimuli ⇒ labelled-line **identity** code, which relocation leaves intact | H1 says nothing about it |

v1's penalty moved both together (§2b), and the diagnosis in document 3 §6 measured
the split: ~75% of the rate reduction was `a`, ~25% was `s` — but the 25% is what
built the identity channel, and it is confounded with the 75% at ρ = −0.94.

**So H1 has never been tested.** v1 tested `a` and `s` moving together and found the
composite runs against H1. H1 is a claim about `a` at fixed `s`.

### 3c. H1, restated so it can be tested

> **H1′:** holding selectivity `s` fixed, reducing spikes per active neuron `a`
> increases the network's reliance on hidden spike timing.

with the companion, which is the account v1's data actually support:

> **H2 (identity-code account):** the anti-H1 trend in v1 is driven by `s`, not `a`:
> silencing builds a perturbation-immune identity code, and that is what makes sparse
> networks look robust.

These make **opposite, separable** predictions in a design where `a` and `s` vary
independently. Producing that design is the whole of Phase 1.

---

## 4. The design

### Phase 0 — the measurement layer (no training) — **DONE, 2026-07-29**

Everything in §2 was a spot check on 3 checkpoints per arm. Phase 0 ran it properly.
No training; every number below is eval-only against the frozen v1 checkpoints.

**What was run.**

| # | Step | Artifact |
|---|---|---|
| 0 | Temporal support measured over all 27, to put the window bound on evidence rather than on a 6-checkpoint spot check | [temporal_support.py](../../exp_sparse_network/v3_analysis/temporal_support.py) → `log/temporal_support.json` |
| 1 | Capacity-matched decode, all 27, both arms | [hidden_channel_decode.py](../../exp_sparse_network/v3_analysis/hidden_channel_decode.py) → `log/hidden_channel_decode_{arm}.json` |
| 2 | Relocation `f=1` endpoint, full window vs support window, all 27 | [relocation_window_check.py](../../exp_sparse_network/v3_analysis/relocation_window_check.py) → `log/relocation_window_check_{arm}.json` |
| 3 | Perturbation windows patched; relocation, jitter and deletion sweeps re-run, both arms | the 8 eval scripts → `*_eval_v3window.json` |
| 4 | Headline join, correlations, partials and the figure | [phase0_headline.py](../../exp_sparse_network/v3_analysis/phase0_headline.py) → `log/phase0_headline*.{json,csv}`, `fig/phase0_headline.png` |

**Step 0 — the support bound is 88, not 87.** The last bin holding any 1st-hidden
spike is **87** across all 27 checkpoints (the spot check saw 86 on 6 of them), so
every window is now `[0, 88)`. Median spike at bin 17–20, p90 at 37–41, p99 at 52–57 —
i.e. the layer really does finish in the first ~44% of the simulated window.

**Step 3 — which scripts actually needed the fix, and why.** The literal instruction
was "fix `perturb_hidden_batch` in all 8 eval scripts". Only two of the four
perturbations have a destination bin to confine, so the correction is not uniform, and
forcing it on the other two would have *introduced* the artifact it removes:

| perturbation | corrected? | why |
|---|---|---|
| **relocation** (`shd/`) | **yes** — destinations drawn from `[0, 88)` | scatters spikes across the whole window; ~57% landed in the padded tail |
| **jitter** | **yes** — clip to `[0, 88)` rather than `[0, 200)` | local, so few spikes reach the tail; count is preserved either way by the collision retry |
| **shift** | **no**, deliberately | a rigid per-neuron translation preserves spike density exactly and never dilutes it. Clipping to the support would pile spikes at the edge and **merge** them, destroying per-neuron count — a rate insult that does not currently exist |
| **deletion** | **no** — nothing to correct | removes spikes, never chooses a destination bin |

Each of the 8 scripts carries this reasoning in its own docstring. The corrected runs
write `*_v3window.json` alongside v1's files rather than over them, and flipping
`SUPPORT_BINS = None` reproduces v1 exactly. The deletion sweep was re-run anyway and
came back **byte-identical** to v1's — which is worth having, since it also
establishes that this eval pipeline is deterministic.

**The results.** `a` = spikes per active neuron, `s` = silent fraction, availability =
timing fraction of decodable information, usage = `temporal_score` at `f = 1` on the
corrected window, control = the same score under 80% spike deletion.

| | no-delay (n=15) | delay (n=12) |
|---|---|---|
| **ρ(`a`, `s`) — the confound** | **−0.943** (p = 1.4e−7) | **−0.455** (p = .14) |
| ρ(availability, `a`) | −0.211 (p = **.45**) | −0.035 (p = **.91**) |
| ρ(usage, `a`) | **+0.939** (p = 2.1e−7) | **+0.881** (p = 1.5e−4) |
| ρ(usage, `s`) | −0.925 (p = 8.0e−7) | −0.476 (p = .12) |
| ρ(deletion control, spikes/neuron) | **+0.936** (p = 3.0e−7) | **+0.713** (p = .0092) |
| ρ(count decode, spikes/neuron) | +0.913 | +0.775 |
| ρ(identity decode, spikes/neuron) | −0.721 | −0.846 |

and the partials that Phase 0's own success criterion turns on:

| partial | no-delay | delay |
|---|---|---|
| availability vs `log a`, given `s` | +0.015 (p = .96) | +0.212 (p = .53) |
| usage vs `log a`, given `s` | +0.530 (p = .051) | +0.849 (p = .00096) |
| usage vs `s`, given `log a` | −0.345 (p = .23) | −0.178 (p = .60) |
| **usage vs `log a`, given `s` *and* deletion** | **+0.238 (p = .43)** | **+0.716 (p = .020)** |
| usage vs `s`, given `log a` and deletion | −0.381 (p = .20) | −0.187 (p = .61) |

**Reading it.**

1. **The dissociation is the result, and it is now on 27 models rather than 6.** The
   layer holds a constant ~23% of its decodable class information in spike timing
   right across the sparsity gradient, while the readout's measured dependence on that
   timing collapses by a factor of ~3. Sparse networks hold just as much timing
   information and use less of it. Neither measurement alone says this, which is why
   §3a insists on reporting both.
2. **Much of the usage trend is not about timing at all.** The deletion control —
   which destroys rate, not timing — correlates with sparsity at +0.936 / +0.713,
   nearly as strongly as the timing probes do. Sparse networks here are simply more
   robust to having their hidden layer interfered with, whatever the interference.
3. **The §2b confound bites exactly where §2b said it would.** In the no-delay arm,
   partialling usage on `s` *and* the deletion control leaves +0.238 (p = .43) — the
   effect does not survive its own controls. In the delay arm, where `a` and `s` are
   far less collinear, it survives at +0.716 (p = .020).

**Verdict against the criterion this section set itself.** Phase 0 said it might make
Phase 1 unnecessary "if the timing fraction is flat at n=27 **and** the usage trend
survives partialling on both `s` and the deletion control". Flatness: **yes, both
arms.** Survival: **no in the no-delay arm, yes in the delay arm.** So Phase 1 is
still needed — and Phase 0 has changed which arm it should run in. See the note at the
head of Phase 1.

**Cost: ~2.5 h GPU** as run (the estimate was ~1.5 h; the four full sweeps rather than
the `f=1` endpoint alone account for the difference).

### Phase 1 — the dissociation experiment (the decisive one)

A 2-D factorial that makes `a` and `s` vary independently, which no previous version
has done.

> **Arm decision, taken 2026-07-29 after Phase 0: the DELAY arm.** The grid below
> still reads `ARM = "nodelay"`; the implementation does not follow it. Three of the
> four reasons originally given for no-delay stand, but they are all reasons about the
> *independent* variables, and Phase 0 found two problems with the no-delay arm's
> **dependent** variable — which a factorial cannot fix, because it manipulates `a` and
> `s`, not the readout:
>
> 1. **The readout leaves +.300 of accuracy unextracted on every no-delay checkpoint**
>    (delay: +.026, §2d). Its `temporal_score` is as much a statement about `fc2`/`fc3`
>    as about layer 1's code, at every point on the gradient.
> 2. **In the no-delay arm the deletion control (+0.936) and the timing probe (+0.939)
>    are indistinguishable.** A pure rate insult and a pure timing insult measure the
>    same thing there. In the delay arm they separate (+0.713 vs +0.881), and the usage
>    effect survives partialling on both `s` and the deletion control (+0.716, p = .020)
>    where the no-delay one does not (+0.238, p = .43).
>
> The weak no-delay partial on its own would *not* justify this: at ρ(`a`, `s`) = −0.943
> and n = 15 with two controls, that null is as much a power problem as a finding, and
> breaking the confound is exactly what the factorial does. Points 1 and 2 are what
> decide it. A clean β on a dependent variable that conflates layer-1 timing with
> readout competence and with general robustness is worse than no β. Cost: ~2.3 h/model
> against ~1.8 h, so ~37 h rather than ~29 h.
>
> The no-delay script is implemented and kept as the **secondary** arm — worth running
> to replicate a delay-arm result, since with no learnable delays any surviving timing
> effect is attributable to membrane dynamics alone, and it is where the factorial adds
> the most *manipulation* value.

**Where the code goes.** The empty placeholders
[sn_train_noDelay_v3.py](../../exp_sparse_network/sn_train_noDelay_v3.py) and
[sn_train_withDelay_v3.py](../../exp_sparse_network/sn_train_withDelay_v3.py) already
exist, matching the v1/v2 one-script-per-arm convention. Fork them from the v2.2
scripts: keep the data, splits, optimiser, schedule and logging, drop
`truncate_to_k_spikes` and the anneal, and replace the penalty block with the two terms
below. Artifacts tag `_v3_`, so nothing can collide with v1, v2, v2.1 or v2.2.

**Mechanism — two terms with separate jobs.**

*Ceiling, sets `a`.* Per-(sample, neuron) penalty `relu(count − k)`, charged per pair
rather than on the batch mean. This is v2's upper arm, which the v2 probe showed
binds cleanly; charging per pair is what stops a neuron dodging the penalty by going
silent on some samples, the exact hole in v1's batch-mean hinge.

*Floor, sets `s`.* Penalise the **membrane potential**, not the spike count:

```python
potential = self.fc1(self.slayer.psp(x))        # pre-refractory, pre-spike
floor_penalty = torch.relu(THETA_MARGIN - potential.max(dim=-1).values).mean()
```

Two properties make this the right fix, both established in §2e. `max_t u ≥ theta`
guarantees at least one spike, so this is sufficient to prevent silence. And
`∂u/∂w` is an ordinary convolution gradient — **no surrogate is involved**, so the
gradient does not vanish however far below threshold the neuron sits. That is
precisely the dead zone (document 1 pitfall 6) that silenced v1's no-delay arm for 26
epochs, left v2/v2.1 at 21–40% silent, and defeated v2.2's count-based floor. Targeting
one suprathreshold crossing rather than driving the count should also avoid v2.2's raw-rate
blow-up, since it asks for the minimum sufficient condition rather than more spikes.

Set `THETA_MARGIN = 11` (`1.1 × theta`) so a satisfied pair is not sitting on the
boundary. The floor's **strength is the column knob**: 0 in one column, a fixed
nonzero value in the other.

**Grid.**

```python
CEILING_K       = [1, 2, 4, 8]      # sets a; the H1 axis
FLOOR_STRENGTH  = [0.0, F]          # sets s; the H2 axis
SEEDS           = [42, 43, 44]      # was [42, 43]; the 3rd seed is what settled beta_a
ARM             = "delay"           # 24 models/arm -- delay FIRST, see above
EPOCHS          = 1250
```

As executed, **both** arms ran this grid: 24 models each, 48 in total. The delay arm was
launched first at seeds 42/43 and extended to seed 44 once its n = 16 result came back
underpowered; the no-delay arm ran all three seeds in one pass.

Implemented as
[sn_train_withDelay_v3.py](../../exp_sparse_network/sn_train_withDelay_v3.py)
(primary) and
[sn_train_noDelay_v3.py](../../exp_sparse_network/sn_train_noDelay_v3.py)
(secondary). Both carry a `CEILING_STRENGTH` fixed across the grid so that `k` is the
only thing varying along the row axis, and both print the manipulation check and the
design acceptance check at the end of a run.

**Log both axes from the start.** The training summary must carry
`spikes_per_active_neuron` and `silent_fraction` alongside `spikes_per_neuron`, and
the per-epoch log must carry their trajectories. v1's summaries record only
`spikes_per_neuron` and `silent_fraction` — which is why the confound in §2b went
unnoticed for the whole of v1 and v2, and it is document 3's own recommendation 3,
still unimplemented.

*Why no-delay first,* despite the delay arm's better accuracy: it is where the
confound is total (ρ = −0.943 vs −0.455), so it is where the factorial adds the most;
it produced v1's significant anti-H1 result, so it is the claim most in need of
decomposition; it is cheaper (~1.8 h vs ~2.3 h per model); and with no learnable
delays, any timing effect is attributable to membrane dynamics alone. Extend to the
delay arm only if the no-delay result warrants it. Add seed 44 only for whichever
`k` turns out decisive.

> **Superseded by the arm decision above.** Reason 2 no longer holds — Phase 0
> decomposed v1's no-delay result and what remains after controls is +0.238 (p = .43).
> Reasons 1, 3 and 4 stand and are why the no-delay arm is kept as the secondary run.
> The order is now delay first, no-delay to replicate.
>
> **Superseded again, 2026-08-01.** Both arms ran the full grid at all three seeds, so
> the ordering was a scheduling decision rather than a scoping one, and the selective
> "seed 44 only for the decisive `k`" plan was dropped — every cell got the third seed.
> That was the right call: it is what moved β_a from p = .204 to p = .033, and picking
> which `k` was "decisive" in advance would have required knowing the answer first.

**A warm-up is required, and 0 is not survivable.** Found before the probe was run,
by a 60-epoch check on the full training set at `k = 1`, floor = 0 — the corner where
nothing opposes the ceiling. v1 and v2.2 both used `WARMUP_EPOCHS = 0`, but their
penalties pushed firing toward a *nonzero* target. v3's ceiling `relu(count − k)` is
one-sided and is minimised at `count = 0`, so in the floor-off column only the task
loss resists silence — and total silence is an **absorbing** state, because at
`u ≈ 0` against `theta = 10` the only route back is the surrogate gradient, which
carries no signal there (§2e again, now biting during training rather than at eval):

| warm-up | outcome at `k = 1`, floor = 0 |
|---|---|
| **0** | **100% silent by epoch 5**, val acc pinned at chance (5.1%) for all 60 epochs — dead layer, unrecoverable |
| **20** | 40–48% silent, `spikes/neuron` 8.6 → 5.5 and still falling, val acc 74.5% |

20 epochs is enough for the representation to establish (val acc was already 65.7% at
epoch 19) before the ceiling engages, after which it binds gradually rather than
catastrophically. `WARMUP_EPOCHS = 20` in both scripts. Note this is a *training-time*
instance of the same dead zone that document 1's pitfall 6 describes, and it is the
one thing about this design that could not have been predicted from the v1/v2 logs —
their penalties never pushed toward zero.

**Probe first — 4 corner models at 400 epochs (~2.5 h).** `k ∈ {1, 8}` ×
floor ∈ {0, F}. The probe validates the *manipulation*, which is a property of the
constraint and readable early:

- within a column, `silent_fraction` roughly constant as `k` varies;
- across columns at matched `k`, `silent_fraction` differs by **≥ 20 points**;
- `raw_spikes_per_neuron` not inflated beyond the natural ~6.3 (v2.2's failure);
- clean accuracy comfortably above chance in every corner.

Do **not** read achieved firing rates or accuracy levels off the probe. §2 of document
4 already warned that reduced-epoch numbers under-estimate the full run, and the v1
training logs confirm it is worse than assumed: **every v1 model was still improving at
epoch 1250** (best val loss at epochs 1168–1243; 99% of peak accuracy first reached at
epochs 978–1169). 400 epochs is not a scaled-down run, it is a different regime.
Calibrate nothing from it. Pick `F` from the probe only by whether the floor bites.

#### Probe result (delay arm, `F = 1`, `CEILING_STRENGTH = 3`, seed 42, 400 ep, ~2.5 h)

| `k` | floor | clean acc | `a` | `s` | sp/neuron | over-`k` |
|---|---|---|---|---|---|---|
| 1 | 0 | .818 | 6.17 | **62.71%** | 2.30 | 27.3% |
| 1 | **1** | .824 | 4.90 | **10.96%** | 4.36 | 67.1% |
| 8 | 0 | .824 | 7.63 | **45.91%** | 4.13 | 16.9% |
| 8 | **1** | .864 | 6.98 | **4.27%** | 6.68 | 25.8% |

Against the four acceptance criteria:

| # | criterion | verdict |
|---|---|---|
| 1 | `s` roughly constant as `k` varies within a column | **partial** — floor-on drifts 10.96→4.27 (6.7 pts), floor-off 62.71→45.91 (**16.8 pts**). Both are small against the column separation, but the floor-off drift is not negligible |
| 2 | `s` differs by ≥ 20 pts across columns at matched `k` | **PASS, with room** — **+51.7** at `k=1`, **+41.6** at `k=8` |
| 3 | firing not inflated past the arm's natural rate | **PASS** — 2.30–6.68 against this arm's natural **11.66**. *(The design doc's "~6.3" is the **no-delay** figure; v1's delay models run 2.12–11.66. The scripts now carry the correct per-arm constant.)* |
| 4 | clean accuracy comfortably above chance | **PASS** — .818–.864 against chance .05, and squarely inside v1's delay range of .78–.88 |

**`F = 1` is settled.** The membrane-potential floor does what four count-based floors
could not: it takes `s` from 63% to 11% at `k = 1` without inflating firing and without
costing any accuracy (.818 → .824). §4b's mechanism argument is confirmed empirically.

**But the probe found a different problem, on the other axis.** The ceiling barely
moves `a`:

| | `a` at `k=1` | `a` at `k=8` | span |
|---|---|---|---|
| floor off | 6.17 | 7.63 | 1.24× |
| floor on | 4.90 | 6.98 | 1.42× |

v1's *observational* range in this arm was 3.82–15.47, a **4.1×** span. A factorial
whose H1 axis moves 1.4× while its H2 axis moves 15× in `s` will estimate β_a badly —
the regression would be well powered for exactly the hypothesis v3 is not primarily
testing. The cause is visible in `over-k`: at `k = 1` with the floor on, **67% of pairs
still fire above the ceiling**, so `CEILING_STRENGTH = 3` reaches equilibrium far above
`k` and simply stops pushing.

Why the ceiling leaks into `s` rather than into `a` in the floor-off column is worth
stating plainly, because it is a limit of the per-pair charge rather than a bug: per-pair
charging removes the *over-fire-elsewhere* escape route, but going **silent** still costs
the ceiling nothing. So with the floor off, ceiling pressure is absorbed by `s` (62.7% at
`k=1`); only in the floor-on column, where silence is blocked, is the ceiling forced onto
`a`. That is the design working as intended — it is why the floor-on column is the one
that carries the H1 contrast — but it means the ceiling has to be strong enough to bind
there.

#### Ceiling calibration (delay arm, `k = 1`, both floor columns, 400 ep, ~1.3 h)

| strength | floor | `a` | `s` | sp/neuron | over-`k` | clean acc |
|---|---|---|---|---|---|---|
| 3 | 0 | 6.17 | 62.7% | 2.30 | 27% | .818 |
| 3 | 1 | 4.90 | 11.0% | 4.36 | 67% | .824 |
| **10** | 0 | **4.00** | 72.1% | 1.12 | 17% | .685 |
| **10** | 1 | **2.66** | 22.6% | 2.06 | 41% | .771 |

**`CEILING_STRENGTH = 10` adopted.** It roughly doubles the row-axis span, holds the
floor column's separation at **~49 points** (from ~52), keeps firing far below this
arm's natural 11.66, does **not** re-enter the absorbing all-silent state, and costs
accuracy that remains usable (.685–.771, against v1's .78–.88 and chance .05). It is
also the safer side of the 400-epoch caveat: penalties calibrated at reduced epochs
**re-densify** by the full 1250, so achieved `a` will sit *above* these numbers in the
real run.

**Residual limitation, stated rather than hidden.** Even at strength 10 the ceiling
does not bind hard: 41% of pairs remain above `k` at `k = 1`, and `a` settles 2.7×
above the target. The row axis is a **soft, manipulated axis with real range**, not a
pinned one — which is a weaker instrument than the floor, and the honest reason to
expect β_a to carry wider error bars than β_s. Pushing further (strength 30) was not
tried; the 3 → 10 step already cost .05–.13 of accuracy, and v2's whole history is that
buying constraint tightness with accuracy destroys the thing being measured.

**Status: the delay-arm grid is complete** — 24 models at 1250 epochs (seeds 42 and 43
remote from 2026-07-29, seed 44 added and finished 2026-08-01).

#### No-delay arm calibration

Run locally while the delay grid runs remotely. **None of the three settings can simply
be inherited**: they were calibrated on the delay arm, and this arm differs on every
axis that matters — natural firing 7.85 vs 11.66, v1 clean accuracy .49–.59 vs .78–.88
(so far less headroom to spend on constraint), and it is the arm whose v1 models sat
completely silent for their first 26 epochs.

**Warm-up: 20 confirmed, and 50 buys nothing.** This needed re-checking because
`WARMUP_EPOCHS = 20` was validated at `CEILING_STRENGTH = 3` *on the delay arm* — two
changes at once against a setting measured under neither, guarding an absorbing failure.
At `k = 1`, floor off, strength 10, 120 epochs:

| warm-up | `s` | `a` | sp/neuron | clean acc | verdict |
|---|---|---|---|---|---|
| 20 | 82.1% | 2.96 | 0.53 | .438 | survived |
| 50 | 80.3% | 3.51 | 0.69 | .425 | survived |

Same endpoint either way, so **`WARMUP_EPOCHS = 20` stays**, which also keeps the two
arms comparable. Worth noting the landing is much sharper here than on the delay arm:
silence goes 45% → 61% → 79% within five epochs of the penalty engaging, then holds.

The engaged state is also more extreme than the delay arm's at identical settings —
`s` ≈ 82% against 72%, `spikes/neuron` 0.53 against 1.12, clean accuracy .438 against
.685. That is the expected direction (this arm starts with less to give), but it puts
the floor-off column close to an empty layer, and it is the number to watch in the
corner probe.

**Corner probe (no-delay, `F = 1`, `CEILING_STRENGTH = 10`, seed 42, 400 ep, ~2.3 h).**

| `k` | floor | clean acc | `a` | `s` | sp/neuron | over-`k` |
|---|---|---|---|---|---|---|
| 1 | 0 | .473 | 2.69 | **80.90%** | 0.51 | 9.4% |
| 1 | **1** | .500 | 1.91 | **21.55%** | 1.50 | 34.2% |
| 8 | 0 | .501 | 3.57 | **57.31%** | 1.52 | 2.8% |
| 8 | **1** | .536 | 3.81 | **6.72%** | 3.55 | 5.2% |

**All three settings transfer, and the manipulation is *cleaner* here than on the
delay arm** — the opposite of what the accuracy headroom argument would have predicted:

| | no-delay | delay |
|---|---|---|
| floor separation at matched `k` | **+59.4 / +50.6** | +51.7 / +41.6 |
| ceiling leak (over-`k`) | **2.8 – 34.2%** | 16.9 – 67.1% |
| firing vs this arm's natural | 0.51–3.55 vs 7.85 | 2.30–6.68 vs 11.66 |
| accuracy vs this arm's v1 range | .473–.536 vs .49–.59 | .818–.864 vs .78–.88 |
| cost of going strength 3 → 10 | — | .818 → .685 at `k=1` |

The ceiling binds much harder here (2.8% leak at `k=8` against the delay arm's 16.9%),
and strength 10 costs this arm essentially nothing relative to v1 — where on the delay
arm the same step cost .13 of accuracy. So the concern that drove the arm decision —
this arm having less to spend on constraint — did **not** materialise as a
manipulation problem. It remains a problem for the *dependent* variable, which is what
the arm decision actually rested on, and that is unchanged.

Two caveats, both honest limits rather than blockers:

- **Criterion 1 is violated more here.** `s` drifts 80.90 → 57.31 (**23.6 points**)
  across `k` in the floor-off column, against the delay arm's 16.8. Mechanically
  sensible — a looser ceiling means less pressure to go silent — and small against the
  ~55-point column separation, but it does couple the two axes within a column.
- **The row axis is narrow at 400 epochs**: `a` spans 2.69 → 3.57 (1.33×) floor-off and
  1.91 → 3.81 (1.99×) floor-on, against v1's 3.8× observational span in this arm. It
  should *widen* by 1250 epochs rather than narrow, because the weakly-constrained
  cells re-densify while the strongly-constrained ones hold — at `k = 8` the ceiling has
  already stopped charging (2.8% over-`k`), so nothing pins those cells, whereas `k = 1`
  is still actively charging at 34%. This is the one place where the "400 epochs
  calibrates nothing" rule works in the design's favour.

**Interior of the row axis measured too (`k` = 2, 4, both columns, ~2.3 h).** The corner
probe tested only the ends, and the 29 h grid rests on the interior, so the four
remaining cells were run at seed 42 to complete the 8-cell map:

| `k` | floor | clean acc | `a` | `s` | sp/neuron | over-`k` |
|---|---|---|---|---|---|---|
| 1 | 0 | .473 | 2.69 | 80.90% | 0.51 | 9.4% |
| 1 | 1 | .500 | 1.91 | 21.55% | 1.50 | 34.2% |
| 2 | 0 | .447 | 2.46 | 77.33% | 0.56 | 7.0% |
| 2 | 1 | .510 | 2.11 | 13.77% | 1.82 | 19.2% |
| 4 | 0 | .464 | 3.02 | 67.46% | 0.98 | 6.4% |
| 4 | 1 | .518 | 2.74 | 9.28% | 2.49 | 10.8% |
| 8 | 0 | .501 | 3.57 | 57.31% | 1.52 | 2.8% |
| 8 | 1 | .536 | 3.81 | 6.72% | 3.55 | 5.2% |

**ρ(`a`, `s`) = −0.238 (p = .57) across all 8 cells — the design acceptance check
passes, and it is now a measurement rather than an extrapolation from two corners.**
Against v1's **−0.943** in this same arm, that is v3's central claim demonstrated: the
factorial breaks the confound that made v1 uninterpretable. Caveat: one seed, 400
epochs, so it is a strong indication and not the final check — the script recomputes it
across the real grid. *(It did: **−0.342** on the 24 real models. The 400-epoch map
under-stated the residual coupling, as §4b's "calibrate nothing from a probe" rule
would predict, but the check still passes with room.)*

Three further things the interior settles:

1. **`s` is cleanly monotone in `k` in both columns** (80.90 → 57.31 and 21.55 → 6.72),
   and the floor separation holds at **+59.4, +63.6, +58.2, +50.6** points at `k` = 1,
   2, 4, 8. The column knob works at every row level, which is what criterion 2 asks.
2. **`a` is monotone in `k` only in the floor-ON column** (1.91 → 2.11 → 2.74 → 3.81).
   Floor-off is non-monotone (2.69 → **2.46** → 3.02 → 3.57). This confirms by
   measurement what was argued for the delay arm: with silence free, ceiling pressure is
   absorbed by `s`, so `a` in the floor-off column is only loosely coupled to `k`. The
   practical consequence is that **the coverage is L-shaped** — the floor-on column
   supplies the `a` variance, the floor-off column supplies the `s` variance — rather
   than a fully crossed square. The regression is still identified, but β_a rests
   mainly on eight models rather than sixteen.
3. **Accuracy is stable across the whole map**, .447–.536 against chance .05 and v1's
   .49–.59. No cell is unlearnable, so §5's "clean accuracy at chance in the floor-on
   column" row is not in play here.

**Optional design improvement, not applied.** The top of the row axis is capped by the
ceiling ceasing to bind: at `k = 8` only 2.8% of pairs exceed it, yet `a` sits at 3.57
against this arm's *unconstrained* ~11.9. Adding a **no-ceiling row** (or `k = 16`)
would anchor the top of the `a` axis near the natural rate and widen the span from ~2×
toward v1's 3.8×, for 2–4 extra models. Not applied, because the delay arm is already
running the 4-level grid and changing the row levels mid-flight would cost the
cross-arm comparison. Worth considering if β_a comes back underpowered.

**Status: the no-delay arm is complete** — 24 models at 1250 epochs, finished
2026-08-01. All three constants were validated *in this arm* before launch —
`WARMUP_EPOCHS = 20`, `FLOOR_STRENGTH = 1`, `CEILING_STRENGTH = 10` — and no change
was needed to any of them.

**Design acceptance check — this one is on the *design*, not the network.** Across the
16 trained models, compute ρ(`a`, `s`). **If |ρ| > 0.5 the factorial has failed to
break the confound** and the regression below is not interpretable — report that and
stop, exactly as v2 would have had to report a failed manipulation check. This is the
v3 analogue of v2's acceptance test, and unlike v2's it is achievable: it asks for two
variables to be decorrelated, not for one to be driven to zero.

**Analysis.** Per model, both dependent variables from Phase 0's measurement layer.
Then, per arm:

```
usage_score      ~ β_a·log(a) + β_s·s + β_d·(deletion control score)
timing_fraction  ~ β_a·log(a) + β_s·s
```

- **H1′ predicts β_a < 0** on `log(a)` for usage (fewer spikes per active neuron ⇒
  more timing reliance), *after* `s` is held fixed.
- **H2 predicts β_s carries the effect** and β_a is null.

Because `s` is *manipulated* here rather than merely observed, the floor-on vs
floor-off contrast at matched `a` is a genuine controlled comparison — the first in
this line of work, and the thing that lets v3 make a causal claim where v1 could only
correlate.

#### Phase 1 RESULT — both arms, 24 models each @ 1250 ep — **COMPLETE, 2026-08-01**

Grids trained (delay remotely, no-delay locally); measured and analysed locally. Scripts:
[phase1_measure.py](../../exp_sparse_network/v3_analysis/phase1_measure.py) (usage +
deletion control) and
[phase1_regress.py](../../exp_sparse_network/v3_analysis/phase1_regress.py)
(regressions, contrast, figure), with availability from
[hidden_channel_decode.py](../../exp_sparse_network/v3_analysis/hidden_channel_decode.py)
retargeted at the v3 summaries via its `VERSION_TAG`. Both arms now run through the same
three scripts; the arm lists in all three were widened from `("delay",)` to both.

**1. The design acceptance check PASSES in both arms.**

| | ρ(`a`, `s`) | VIFs in the usage model |
|---|---|---|
| v1 no-delay (observational) | **−0.943** | — |
| v1 delay (observational) | **−0.455** | — |
| **v3 delay factorial (n=24)** | **−0.087** (p = .69) | **1.01–1.03** |
| **v3 no-delay factorial (n=24)** | **−0.342** (p = .10) | **1.03–3.21** |

| | delay | no-delay |
|---|---|---|
| `a` span | 2.43–5.29 (2.18×) | 1.79–4.06 (2.26×) |
| `s` span | 5.2%–66.3% | 5.6%–76.3% |
| clean acc (v1's range) | .707–.859 (.780–.883) | .465–.550 (.490–.590) |
| firing vs this arm's natural | 1.10–4.55 vs 11.66 | 0.50–3.83 vs 7.85 |

**For the first time in this project, `a` and `s` are separable** — and in the no-delay
arm the confound went from **−0.943 to −0.342**, which is the single number v3 was built
to produce. That is what makes everything below a test rather than a correlation.

One caveat, and it is this arm's own: in the no-delay arm the deletion control is itself
strongly tied to selectivity, ρ(`s`, control) = **−0.818** (delay: +0.030), giving VIF
3.2 on both terms. So the no-delay arm can identify β_a cleanly (VIF 1.03, ρ(`a`,
control) = +0.05) but **cannot cleanly separate β_s from general robustness**. That is
Phase 0's §2d/§4 finding about this arm reappearing under the factorial — the factorial
fixes the *independent* variables, and this was never one of them.

**2. Availability: flat in the delay arm, and NOT flat in the no-delay arm.**

```
delay       timing_fraction ~ log(a) + s    R² = .106 (adj .021),  n = 24, df = 21
  log_a   +0.0007  [−0.0360, +0.0373]  p = .970   β* = +0.008
  s       −0.0301  [−0.0699, +0.0097]  p = .131   β* = −0.325

no-delay    timing_fraction ~ log(a) + s    R² = .560 (adj .518),  n = 24, df = 21
  log_a   −0.0302  [−0.0529, −0.0074]  p = .012   β* = −0.403
  s       −0.0452  [−0.0652, −0.0252]  p = .0001  β* = −0.688
```

The delay arm replicates the flatness for a third time, now with both axes set by
construction. **The no-delay arm does not** — and this is new to v3, because Phase 0
could only look at the raw correlation, which here is ρ(availability, `a`) = −0.289
(p = .17), i.e. invisible until `s` is partialled out. Two things to hold onto:

- The **sign is H1's**: β_a < 0 means *fewer* spikes per active neuron ⇒ *more* of the
  layer's decodable information requires timing. So in this arm the layer does shift
  toward a timing code as `a` falls — while its readout moves the opposite way (§3
  below). The dissociation is not just "one moves, one doesn't"; here **the two move in
  opposite directions**, which is a stronger form of the same finding.
- The **effect is small in absolute terms.** Timing fraction spans .230–.299 (mean .264)
  across the whole no-delay grid, and .168–.254 (mean .220) in the delay arm. A
  significant β on a range that narrow is worth reporting and not worth leaning on.

**3. The usage regression — the actual test of H1′ vs H2.**

```
delay       usage ~ log(a) + s + deletion control   R² = .415 (adj .328),  n = 24, df = 20
  log_a             +0.0850  [+0.0076, +0.1624]  p = .033   β* = +0.394   VIF 1.01
  s                 −0.0868  [−0.1714, −0.0023]  p = .045   β* = −0.370   VIF 1.02
  control_deletion  +0.4332  [−0.0119, +0.8784]  p = .056   β* = +0.351   VIF 1.03

no-delay    usage ~ log(a) + s + deletion control   R² = .814 (adj .786),  n = 24, df = 20
  log_a             +0.2003  [+0.1179, +0.2828]  p = .0001  β* = +0.495   VIF 1.03
  s                 −0.1147  [−0.2429, +0.0134]  p = .077   β* = −0.323   VIF 3.21
  control_deletion  +0.8997  [+0.1194, +1.6800]  p = .026   β* = +0.413   VIF 3.16
```

with the control omitted, for comparison — delay: β_a = +0.0915 (p = .031), β_s =
−0.0757 (p = .093), R² = .295; no-delay: β_a = +0.1932 (p = .0002), β_s = −0.2368
(p = 1.3e−5), R² = .760.

**4. The controlled floor-on vs floor-off contrast at matched `a`** (nearest-neighbour
pairs, |Δ log `a`| ≤ 0.2):

| arm | pairs | mean Δusage | p |
|---|---|---|---|
| delay | 10 | +0.015 (se .017) | .41 |
| **no-delay** | **11** | **+0.119 (se .021)** | **.0002** |

H2 predicts Δ > 0 — removing silence should *raise* timing reliance. The no-delay arm
delivers that at 11 of 11 pairs positive; the delay arm's mean runs the same way but is
indistinguishable from zero.

**Verdict against §5.** The result has moved off row 4 and onto **row 2 — "β_a > 0,
significant: H1 refuted on its own terms"** — in **both arms independently**. The
honest reading:

- **H1′ is refuted, not merely unsupported.** β_a is positive and clears α in both arms
  (p = .033 delay, p = .0001 no-delay), with `s` held fixed *by construction* and
  general robustness controlled. H1′ predicts β_a < 0; both 95% CIs exclude zero on the
  wrong side. This is §5's strongest available negative, and the two arms reach it by
  different routes — no learnable delays in one, so the effect there cannot be about
  delay lines at all.
- **What changed from n = 16 was power, exactly as predicted.** The delay arm's n = 16
  estimate was β\* = +0.298, p = .204; at n = 24 the same effect is β\* = +0.394,
  p = .033. Direction, magnitude and rank all held; only the interval shrank. The
  "underpowered" verdict recorded on 2026-07-29 was correct and has now been resolved by
  the third seed rather than overturned.
- **H2 gets partial, arm-dependent support.** In the delay arm — the one that can
  identify it — β_s is significant and in H2's direction (β\* = −0.370, p = .045), but
  the matched contrast there is null. In the no-delay arm the matched contrast is
  decisive (+0.119, p = .0002) but β_s is entangled with the deletion control at
  ρ = −0.818. So: selectivity does something, in H2's direction, in both arms; neither
  arm delivers it cleanly *and* by both routes at once.
- **Much of what looks like "timing reliance" is still not about timing.** The deletion
  control is a significant positive term in both models (β\* = +0.351, +0.413), and in
  the no-delay arm ρ(usage, control) = **+0.822**. Any reading of `usage` that does not
  hold this fixed is partly a statement about general robustness — which is why the
  control is in the model rather than in a footnote.
- **The no-delay readout caveat still stands and is not fixed by any of this.** Phase 0
  measured a +.300 gap between that arm's own accuracy and a linear decoder on its layer
  1, on every checkpoint. Its `usage` is as much about `fc2`/`fc3` as about layer 1's
  code, so the delay arm remains the primary venue for the *interpretation* even though
  the no-delay arm carries the larger coefficients.

**What is genuinely left.** Not power, and not the mechanism. The remaining design limit
is the **width of the `a` axis**: 2.18× (delay) and 2.26× (no-delay) against v1's
observational 4.1× / 3.8×, because the ceiling stops binding at `k = 8` (2.8–11% of pairs
still above it). A no-ceiling or `k = 16` row would anchor the top of the axis near the
natural rate, for 4–6 extra models per arm. With β_a now significant in both arms this
would sharpen an existing result rather than decide an open one.

### Phase 2 — confirmation venue (only if Phase 1 is ambiguous)

The synthetic tasks already in the repo under
`exp_fixed_weight_perturbation/code/synthetic/`. **CCISI** is the best choice: class
identity lives in a cross-channel lag δ, both neurons in a pair share a firing rate,
so the *input's* rate channel carries **zero** class information by construction, and
the ground-truth temporal signal is known exactly. ISI is a useful second (rate
carries partial information there, so it brackets CCISI).

> **Correction to document 1 §9.** That section says a synthetic task means "no
> manipulation is needed at all: the count channel carries nothing". That is true of
> the **input** and false of the **hidden layer**. A network can perfectly well read
> input timing at layer 1 and re-encode the answer as hidden spike *counts* — which is
> the rate-coding outcome this whole project exists to detect. What synthetic data
> actually buys is a known ground truth, a computable rate-only accuracy ceiling, and
> no incentive to preserve an input rate code. Worth having; not a free pass.

---

## 5. How to read the outcome

Phase 1, with the design acceptance check passed:

| Result | Reading |
|---|---|
| β_a < 0, significant | **H1 supported under its own premise.** v1's positive ρ was the selectivity confound; temporal sparsity really does push the code toward timing. |
| β_a > 0, significant | **H1 refuted on its own terms** — the strongest available negative, since `s` was held fixed by construction and the prediction still ran backwards. |
| β_a ≈ 0, β_s < 0 significant | **H2 confirmed.** "Sparsity" was never the operative variable; *selectivity* was. The supervisor's question has no single answer, because the answer depends on which channel the sparsity mechanism leaves open — a more useful finding than either yes or no. |
| β_a ≈ 0 and β_s ≈ 0 | Timing dependence is invariant to both. Consistent with §2c's flat availability; report the dissociation as the result. |
| Clean accuracy at chance in the floor-on column | The constraint is unlearnable in this architecture. Report it; move to Phase 2. |
| \|ρ(`a`, `s`)\| > 0.5 across the grid | Design failed; the factorial did not decorrelate. Not interpretable as a test of H1′. |

Rows 3 and 4 are the ones §2c makes most likely, and both are perfectly good results.
Note that row 3 in particular **reconciles every observation in this project** — v1's
positive ρ, the identity-decode rise, and the flat availability — under one mechanism.

> **Outcome, 2026-08-01: row 2, in both arms** (β_a > 0, p = .033 delay / .0001
> no-delay). Row 3's mechanism is *partly* in play alongside it — β_s is significant and
> in H2's direction in the delay arm, and the matched contrast is decisive in the
> no-delay arm — but β_a is not null, so this is not row 3. The two rows are not
> mutually exclusive in practice: selectivity does build the identity code §2c and v1
> both point at, *and* temporal sparsity moves usage in the direction opposite to H1′.
> See §4's Phase 1 RESULT.

---

## 6. Why v2's channel-closing programme is retired

v2 was not wasted: it established that the count requirement is effectively binary,
that soft penalties lose an arms race, and that `k = 1` is learnable. But it should
not continue, for four reasons.

1. **It solves a harder problem than the question needs.** Closing the immune channel
   makes the covariate zero. *Controlling* it only requires measuring it — which costs
   ~40 min for all 27 checkpoints and is already implemented.
2. **The covariate barely moves anyway.** Count decode across v1's entire gradient is
   .70–.78 (no-delay) and .74–.78 (delay) — near-constant (§2c). What *does* move is
   the identity decode, and that is exactly what Phase 1 manipulates directly.
3. **The cost is the experiment's own validity.** v2.2 halves accuracy (.552 → .237),
   inflates raw firing to 12.5–21.9 against a natural ~6.3, and crowds 63% of surviving
   spikes into the first 10 ms. A network whose underlying firing rate has *doubled*
   is no longer an instance of "sparser-activity networks", which is the thing the
   supervisor asked about.
4. **It changes the question.** With the count channel closed at every `k`, sweeping
   `k` asks "given a pure latency code, does its rate matter?" That is worth knowing,
   but it is not the milestone question, and H1's mechanism — sparsity *causing* the
   count channel to close — is unobservable once the closing is done by hand.

Keep the v2.2 checkpoints. They are a useful extreme point on the `a` axis for
Phase 1's regression, provided the truncation is applied at eval (document 4 §4).

---

## 7. Cost

| Stage | Estimate | Note |
|---|---|---|
| Phase 0 — decode all 27 | ~25 min | **done** |
| Phase 0 — temporal support, all 27 | ~5 min | **done** — added; the window bound needed evidence |
| Phase 0 — relocation `f=1` window check, both arms | ~20 min | **done** |
| Phase 0 — patch 8 eval scripts + re-run relocation, jitter, deletion | ~1.7 h | **done** — correctness fix from §2a |
| Phase 0 — headline join, correlations, figure | ~1 min | **done** |
| Phase 1 — corner probe + calibration, delay arm | ~3.8 h | **done** — `F`, then `CEILING_STRENGTH` |
| Phase 1 — warm-up check + 8-cell map, no-delay arm | ~5 h | **done** — all three constants re-validated in-arm |
| Phase 1 — delay grid, 24 models @ 1250 ep | **~55 h** | **done** — 2.3 h/model, seeds 42/43 then 44 |
| Phase 1 — no-delay grid, 24 models @ 1250 ep | **~43 h** | **done** — 1.8 h/model |
| Phase 1 — measurement layer, both arms, 48 checkpoints | ~35 min | **done** — decode + relocation/deletion endpoints + regressions |
| Phase 2 — synthetic CCISI/ISI | a few hours | not needed; see §9 |

**~2.5 h bought the complete Phase 0 answer** (estimated ~3 h). Phase 1 cost ~107 h of
training and turned a correlation into a controlled comparison in two independent arms.
Compare with v2's outstanding 62–94 h, which would have produced neither.

The third seed was the cheapest thing in the whole programme relative to what it bought:
~35 h of training moved the delay arm's β_a from p = .204 to p = .033 and its β_s from
p = .074 to p = .045, i.e. it converted the headline result from "inconclusive but
directional" to a refutation. The measurement layer that reads all 48 models costs
**~35 min**, so re-analysing is free and only training is not.

---

## 8. Watch out for

Inherits everything in document 1 §7 and document 4 §7. New to v3:

1. **Never analyse `spikes_per_neuron` alone.** It is the product of two variables that
   push opposite ways (§3b). Always report `a` and `s` separately, and always show
   `ρ(a, s)` so a reader can see whether the design supports the claim being made.
2. **Perturbations must respect the layer's temporal support.** Hidden activity ends
   at bin 87 of 200; relocating or jittering into the padded region mixes a rate insult
   into a timing probe (§2a). Both are fixed, at `[0, 88)`. But the rule is *respect the
   support*, not *clip to it*: shift and deletion are correctly left alone, and clipping
   shift would have destroyed per-neuron count. Ask what a perturbation does to spike
   *density* before deciding it needs the window (§4, Phase 0 step 3).
3. **Availability and usage are different measurements and can disagree** — here they
   do (§2c vs §2a), and in the no-delay factorial they **move in opposite directions**
   (Phase 1 RESULT §2 vs §3): as `a` falls the layer holds *more* timing information and
   the readout uses *less* of it. Reporting one without the other is how v1 became hard
   to read.
4. **Compare `FULL` against the capacity-matched shuffle, never against `COUNT`.**
   The dimensionality difference alone is worth ~.12 of decode accuracy (§3a).
5. **400-epoch probes cannot calibrate anything.** Every v1 model was still improving
   at epoch 1250 (§4b). Use probes for manipulation checks only.
6. **The no-delay readout is inefficient by ~.26 accuracy** (§2d). Any statement about
   what "the network uses" in that arm is a statement about `fc2`/`fc3` as much as about
   layer 1. State it wherever the arms are compared.
7. **The floor penalty acts on the membrane potential, not the spike count.** If it is
   ever reimplemented against counts, it will hit the dead zone again — silent pairs sit
   at `u ≈ 0` against `theta = 10` (§2e).
8. **A one-sided penalty toward zero needs a warm-up; a penalty toward a target does
   not.** v1 and v2.2 ran `WARMUP_EPOCHS = 0` safely because their penalties pushed
   firing toward a nonzero value. v3's ceiling is minimised at `count = 0`, and with
   `warmup = 0` the layer is 100% silent by epoch 5 and never recovers — the dead zone
   as an *absorbing training state* rather than an eval-time artifact. Do not copy a
   warm-up setting across from a version whose penalty had a different sign.
9. **Decorrelating `a` from `s` does not decorrelate `s` from the deletion control.**
   The factorial only governs the two *independent* variables. In the no-delay arm
   ρ(`s`, control) = **−0.818** (VIF 3.2), so β_s there is not separable from general
   robustness even though ρ(`a`, `s`) = −0.342 passes. Check the covariate's
   correlations too, not just the design's — a passing design acceptance check is not a
   licence to read every coefficient in the model.
10. **The v3 training scripts write their summary whole.** A run launched with a reduced
    `SEEDS` list used to *replace* the arm's summary and drop every row it did not
    retrain — which is how the delay arm's seed-42/43 rows were lost when seed 44 was
    added. Both scripts now seed the summary from the existing file, and
    [rebuild_train_summary.py](../../exp_sparse_network/v3_analysis/rebuild_train_summary.py)
    recovers missing rows from the checkpoints. Note the checkpoints are the durable
    artifact and the summary is not — re-measuring a checkpoint is not bit-exact
    (cuDNN path; `clean_acc` moves by ≤ .002), so recover rather than re-measure
    wholesale.

---

## 9. Checklist

- [x] v1 and v2 read, and the blocking flaw identified: `a` and `s` confounded at
      ρ = −0.943 (no-delay), −0.455 (delay)
- [x] Relocation window artifact found, quantified, and shown **not** to flip v1's
      conclusion (ρ +0.918 → +0.932 no-delay, +0.769 → +0.867 delay)
- [x] Capacity-matched decoder designed, implemented, and spot-checked on 6
      checkpoints — timing fraction **flat** at .24–.27 across a 4.6–5.5× rate change
- [x] Readout-efficiency gap measured (+.257 no-delay, −.013 delay)
- [x] Membrane-potential scale measured, sizing the Phase 1 floor penalty
- [x] Analysis scripts committed under
      [v3_analysis/](../../exp_sparse_network/v3_analysis/)
- [x] **Phase 0** — decode all 27 checkpoints, both arms — availability **flat**
      (ρ = −0.211 p=.45 / −0.035 p=.91), usage steep (+0.939 / +0.881)
- [x] Phase 0 — temporal support measured on all 27; bound is `[0, 88)`, not `[0, 87)`
- [x] Phase 0 — patch the 8 eval scripts' relocation/jitter windows (shift and
      deletion correctly exempt); re-run relocation, jitter and the deletion control
      — deletion reproduced byte-identically
- [x] Phase 0 — headline pair of plots; **Phase 1 is still needed** — the usage trend
      survives its controls in the delay arm (+0.716 p=.020) but not in the no-delay
      arm (+0.238 p=.43)
- [x] Phase 1 — **arm decided: DELAY**, on the two Phase 0 findings the factorial
      cannot fix (readout gap +.300 on every no-delay checkpoint; deletion control
      +0.936 vs timing probe +0.939 there). No-delay kept as the secondary arm.
- [x] **Phase 1** — ceiling + potential-floor training script implemented for both
      arms ([withDelay](../../exp_sparse_network/sn_train_withDelay_v3.py) primary,
      [noDelay](../../exp_sparse_network/sn_train_noDelay_v3.py) secondary)
- [x] Phase 1 — `WARMUP_EPOCHS` resolved at **20**: 0 is unsurvivable for a
      one-sided-toward-zero ceiling (100% silent by epoch 5, chance accuracy, no
      recovery), unlike v1's and v2.2's penalties toward a nonzero target
- [x] Phase 1 — 4-model corner probe run (delay arm, 400 ep, ~2.5 h): **`F = 1`
      settled** — floor separates `s` by +51.7 / +41.6 points at matched `k`, no rate
      inflation, accuracy .818–.864
- [x] Phase 1 — row axis calibrated: **`CEILING_STRENGTH = 10`** roughly doubles the
      `a` span, holds the floor separation at ~49 points, no collapse, accuracy
      .685–.771. Row axis remains *soft* (41% of pairs still above `k` at `k=1`)
- [x] Phase 1 — **delay-arm grid at 1250 epochs LAUNCHED** on a remote server,
      2026-07-29 (16 models at seeds 42/43; extended to seed 44 and completed
      2026-08-01, 24 models, ~55 h total)
- [x] Phase 1 — **no-delay arm calibrated** (locally, while the delay grid runs): all
      three constants validated unchanged in this arm; manipulation is *cleaner* here
      (floor separation +50…+64 points at every `k`, ceiling leak 2.8–34%, accuracy
      .447–.536 against v1's .49–.59)
- [x] Phase 1 — design acceptance check **pre-verified on the no-delay 8-cell map**:
      **ρ(`a`, `s`) = −0.238** against v1's −0.943 (1 seed, 400 ep — indicative, and
      recomputed on the real grid by the script)
- [x] Phase 1 — **no-delay 24-model grid at 1250 epochs complete** (~43 h),
      2026-08-01
- [x] Phase 1 — **both grids complete and analysed at n = 24 per arm** (48 models);
      measurement layer widened to both arms in `hidden_channel_decode.py`,
      `phase1_measure.py` and `phase1_regress.py`
- [x] Phase 1 — design acceptance check on the real grids: ρ(`a`, `s`) = **−0.087**
      (delay, p = .69) and **−0.342** (no-delay, p = .10) — the factorial broke the
      confound in both arms, including the one v1 confounded at −0.943
- [x] Phase 1 — regressions and the controlled floor-on vs floor-off contrast at
      matched `a`, both arms
- [x] Phase 1 — delay-arm summary rows for seeds 42/43 **recovered** after the seed-44
      run overwrote them; cause fixed in both training scripts (see §8 item 10)
- [x] v3 verdict recorded against §5 — **row 2 in both arms: H1′ refuted on its own
      terms** (β_a > 0, p = .033 delay / .0001 no-delay). H2 gets partial,
      arm-dependent support: β_s significant in the delay arm (p = .045), matched
      contrast decisive in the no-delay arm (+0.119, p = .0002)
- [x] Phase 1 — the power limit recorded on 2026-07-29 is **resolved** by the third
      seed (n 16 → 24 per arm): the delay arm's β_a went p = .204 → **.033** and β_s
      p = .074 → **.045**, same direction and magnitude throughout
- [ ] *Optional, sharpens rather than decides:* widen the `a` axis with a no-ceiling
      or `k = 16` row (4–6 models/arm). Currently 2.18× / 2.26× against v1's
      4.1× / 3.8×, because the ceiling stops binding at `k = 8`
- [ ] Phase 2 (CCISI/ISI) — **not needed.** §5 landed on row 2, not an ambiguous row,
      and it did so independently in two arms. Keep as a confirmation venue only if a
      reviewer wants the known-ground-truth replication
