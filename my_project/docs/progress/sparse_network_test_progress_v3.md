# First Milestone v3 — separating the two things "sparsity" does (design + Phase 0)

**Status (2026-07-29): PHASE 0 COMPLETE. Phase 1 implemented and calibrated; the
16-model grid is ready and has NOT been launched.**
§2 was five spot checks on 3 checkpoints per arm, made to decide what v3 should be.
§4's Phase 0 has now run all of them properly over all 27 checkpoints, plus the
correctness fix to the perturbation window. **The dissociation §2c predicted holds at
full n, and it is now the result:** availability is flat (ρ = −0.211, p = .45 /
−0.035, p = .91) while usage climbs steeply (+0.939 / +0.881).

Phase 0 also **changed which arm Phase 1 runs in — it is now the delay arm**, on two
findings about the *dependent* variable that a factorial cannot fix. Both training
scripts are written and three settings are now calibrated on measured runs rather than
assumed: `WARMUP_EPOCHS = 20` (0 is unsurvivable), `FLOOR_STRENGTH = 1` (separates `s`
by ~50 points at matched `k`, at no accuracy cost), and `CEILING_STRENGTH = 10` (3 left
the H1 axis moving only 1.4×). The one open weakness is that the row axis stays soft
even at 10. See §4. **Owner:** _(you)_

**This is document 5 of 5. Read in order:**

| # | Document | What it is |
|---|---|---|
| 1 | [sparse_network.md](sparse_network.md) | the question and the conceptual landscape — **start there** |
| 2 | [sparse_network_test_progress.md](sparse_network_test_progress.md) | v1 execution log — getting a sparsity gradient to exist at all |
| 3 | [sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, and the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](sparse_network_test_progress_v2.md) | v2 execution log — attempts to close the perturbation-immune channels |
| 5 | **this file** | v3 — why the question needs a different experiment, what it is, and the executed Phase 0 (§4) |

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
SEEDS           = [42, 43]
ARM             = "delay"           # 16 models -- CHANGED after Phase 0, see above
EPOCHS          = 1250
```

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
> The order is now delay first, no-delay to replicate. Seed 44 for the decisive `k` is
> unchanged.

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

**Status: the grid is ready to launch and has not been launched.** ~37 h for 16 models
at 1250 epochs. That is the next decision, and it is a large enough commitment to be
worth taking deliberately rather than as a continuation of the probe.

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
| \|ρ(`a`, `s`)\| > 0.5 across the 16 models | Design failed; the factorial did not decorrelate. Not interpretable as a test of H1′. |

Rows 3 and 4 are the ones §2c makes most likely, and both are perfectly good results.
Note that row 3 in particular **reconciles every observation in this project** — v1's
positive ρ, the identity-decode rise, and the flat availability — under one mechanism.

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
| Phase 1 — corner probe, 4 models @ 400 ep | ~2.5 h | manipulation check only |
| Phase 1 — full grid, 16 models @ 1250 ep (no-delay) | **~29 h** | 1.8 h/model, scaled from v2's measured probe times |
| Phase 1 — delay arm, if warranted | ~37 h | optional |
| Phase 2 — synthetic CCISI/ISI | a few hours | much smaller networks |

**~2.5 h bought the complete Phase 0 answer** (estimated ~3 h). Phase 1 is the ~30 h
that turns a correlation into a controlled comparison. Compare with v2's outstanding
62–94 h, which would have produced neither.

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
   do (§2c vs §2a). Reporting one without the other is how v1 became hard to read.
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
- [ ] Phase 1 — **full 16-model grid at 1250 epochs (~37 h) — READY, NOT LAUNCHED.**
      Set `QUICK_TEST = False` in
      [sn_train_withDelay_v3.py](../../exp_sparse_network/sn_train_withDelay_v3.py)
- [ ] Phase 1 — design acceptance check: |ρ(`a`, `s`)| < 0.5
- [ ] Phase 1 — regressions, controlled floor-on vs floor-off contrast at matched `a`
- [ ] v3 verdict recorded against §5
- [ ] Phase 2 only if §5 lands on an ambiguous row
