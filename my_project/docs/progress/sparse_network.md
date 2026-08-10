# Sparsity → Temporal Processing: the question, and what it takes to answer it

> *"Induce different levels of sparsity in the hidden layers (through
> regularisation), then observe whether sparser-activity networks do more
> temporal processing than denser ones."* — supervisor's suggestion

**Hypothesis (H1):** as hidden-layer activity gets sparser, the network relies
*more* on precise hidden spike timing (more temporal processing) and less on a
rate code.

This document is the **conceptual reference**: what the question means, how each
side of it can be measured, what makes it hard to test honestly, and what would
count as an answer. It deliberately contains no scripts, grids, file paths or
hyper-parameters — those live in the execution logs below, which also record what
actually happened when we tried.

**Read in this order:**

| # | Document | What it is |
|---|---|---|
| 1 | **this file** | the question and the conceptual landscape |
| 2 | [sparse_network_test_progress.md](../legacy/sn_progress/sparse_network_test_progress.md) | v1 execution log — getting a sparsity gradient to exist at all |
| 3 | [sparse_network_1stLayer_results.md](../legacy/sn_progress/sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, and the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](../legacy/sn_progress/sparse_network_test_progress_v2.md) | v2 execution log — attempts to make H1's premise actually hold |
| 5 | [sparse_network_test_progress_v3.md](../legacy/sn_progress/sparse_network_test_progress_v3.md) | v3 design — why "sparsity" is two variables, and the experiment that separates them |

Much of §5–§7 below was learned the hard way in documents 2–4. It is written here
as conceptual guidance so the next person does not rediscover it.

> **Corrections from v3 (document 5), which supersede parts of this file:** §5's
> "sparsity" is not one independent variable but two — spikes per *active* neuron and
> the *silent fraction* — which push in opposite directions and are confounded at
> ρ = −0.94 in v1's no-delay arm. §6a's demand that the immune channels be *closed*
> is stronger than the question needs; they can be *measured and controlled* instead,
> which v2 spent ~8 h discovering the hard way. §9's claim that a synthetic task needs
> no manipulation holds for the *input* only — a network can still re-encode input
> timing as hidden spike counts. See document 5 §2–§3.

---

## 1. The shape of the experiment

```
independent variable  :  hidden-layer sparsity      (set during training)
dependent variable    :  temporal processing        (measured at evaluation)
prediction (H1)       :  sparser ⇒ more temporal ⇒ negative correlation
                         between hidden firing rate and timing-dependence
```

The project already has the measuring device: the **fixed-weight hidden-perturbation
sweep** — train on clean data, then perturb one hidden layer's output at evaluation
only, and watch accuracy fall (see `knowledge_bank/phase1_investigation.md`). A steep
fall means the representation depends on hidden spike timing; a flat curve means the
readout is content with a rate code.

So the experiment is: train a family of networks differing only in hidden sparsity,
run the same perturbation sweep on each, and correlate.

---

## 2. Why the perturbation must be evaluation-only

This is the single most important protocol decision. Two protocols answer different
questions:

| Protocol | What it measures | Curve at a hidden site |
|---|---|---|
| **Fixed weight** (train clean → evaluate perturbed) | Does the trained representation *rely on* hidden timing? | Informative — flat vs steep |
| **Perturbation aware** (train and evaluate at the same perturbation) | Is the task *solvable* under the perturbation? | Collapses to flat |

Under perturbation-aware training the upstream layer simply learns to route
everything through the perturbation-immune channel, so the curve is flat *regardless*
of sparsity — it would erase the effect being looked for.

> ⚠️ **Protocol rule:** sparsity is induced **during training**, on clean data. The
> perturbation is applied **only at evaluation**. They are two separate
> interventions and must never be combined into one training run.

---

## 3. Measuring each side

**Sparsity (the independent variable).** Reported per trained model, on the test set:

- **Mean firing rate** — fraction of active (neuron, time-bin) slots; the headline
  x-axis.
- **Spikes per neuron per sample** — the intuitive form of the same thing.
- **Spikes per *active* neuron** — the same quantity conditioned on the neuron firing
  at all.
- **Silent fraction** — the proportion of (sample, neuron) pairs that fire nothing.

The last two are not optional extras. A layer at "1.4 spikes per neuron" can be a
layer where every neuron fires 1–2 spikes, or one where half the neurons are silent
and the rest fire 3–4. Those are completely different codes, and the headline
statistic cannot tell them apart. **Always report the conditional statistic beside
the mean.**

**Temporal processing (the dependent variable).** From the accuracy curve over
perturbation strength, collapse to one number that controls for baseline accuracy,
since sparse networks may start lower:

```
retention      = (acc at max perturbation − chance) / (acc at 0 − chance)
temporal_score = 1 − retention          0 = pure rate code, 1 = fully timing-dependent
```

Chance-correcting **before** taking the ratio matters: without it, a network that has
collapsed to chance masquerades as "maximally temporal". A more robust variant uses
the normalised area over the whole curve rather than just its endpoints.

**Always analyse against *measured* sparsity, never against the regularisation
setting.** The map from any penalty knob to achieved sparsity is nonlinear,
seed-dependent, and — as the execution logs show repeatedly — frequently not what
was intended.

---

## 4. What a perturbation actually isolates: the three channels

A hidden layer's spike tensor can carry class information in three distinguishable
places:

| Channel | What it is | Survives a rate-preserving perturbation? |
|---|---|---|
| **Count** | how many spikes each neuron fires | **yes, exactly** |
| **Identity** | *which* neurons fire at all | **yes, exactly** |
| **Timing** | *when* the spikes occur | no — this is what is destroyed |

Identity is formally a special case of count (fired vs didn't), but it is worth
naming separately because it is what stimulus selectivity creates, and because it can
be rich even when counts are nearly uniform.

Rate-preserving perturbations — per-spike jitter, whole-train shift, random
relocation — move spikes in time while leaving each neuron's spike *count* and
*identity* untouched. That is exactly what makes them a clean timing probe. But it
also means:

> **A perturbation sweep measures how much the readout depends on the timing
> channel *given that the count and identity channels survive intact*.** If those
> channels carry the answer, the curve will be flat no matter how sparse the layer
> is — not because timing is unused, but because it was never needed.

---

## 5. The mechanism that motivates H1 — and the flaw in it

The intuitive argument for H1 runs:

1. A **dense** layer gives each neuron many spikes, so per-neuron count has
   resolution; the network can store the class in a rate code, which a timing
   perturbation cannot touch ⇒ flat curve.
2. A **sparse** layer gives each neuron 0–2 spikes, so count carries almost no
   resolution; the only informative axis left is *when* the lone spike occurs — a
   latency code, which the perturbation destroys ⇒ steep curve.
3. Therefore sparsity pushes information out of the immune count channel into the
   fragile timing channel.

**Step 2 does not follow, and this is the central conceptual correction of the whole
investigation.** Count resolution is a **population** property, not a per-neuron one.
One bit per neuron across a layer of N neurons is N bits of count information — for a
20-class problem, orders of magnitude more than needed. Collapsing each neuron to
"fires once or twice" does *not* collapse the population's ability to encode the class
in counts.

Two consequences follow, and both were observed:

- **Making a layer sparse does not, by itself, close the count channel.** Measured
  directly, count remained highly decodable at every sparsity level reached.
- **Sparsity can *open* the identity channel.** If sparsity arrives by neurons
  falling silent on some stimuli and not others, the result is stimulus selectivity —
  a labelled-line code that a timing perturbation leaves perfectly intact, and which
  gets *richer* as sparsity increases. The intervention can therefore push the network
  in the opposite direction to the one H1 assumes.

So H1 as stated conflates "sparse" with "count-uninformative". They are not the same
thing, and the first does not imply the second.

---

## 6. What it takes to test H1 honestly

Two conditions, both of which have to be *verified rather than assumed*.

### 6a. The immune channels must actually be closed — and it must be measured

Since sparsity does not close the count channel by itself, something else has to. And
whatever is tried, the closure must be **measured, before any headline correlation is
read**:

> **Manipulation check.** Fit a simple linear decoder to predict the label from (i)
> the per-neuron spike-count vector and (ii) its binarisation. Both are exactly
> invariant to every rate-preserving perturbation, so the count decode is a *ceiling*
> on what a perturbation-immune readout could achieve. It must fall to near chance.

If it does not, the headline result is uninterpretable — a flat curve could mean
"timing is unused" or "the network didn't need timing because counts were still
available", and nothing in the curve distinguishes them. A useful sanity reference:
compare the count decode against the network's *own* accuracy. If counts decode
*better* than the network scores, the network is demonstrably not forced onto timing.

Two hard-won points about achieving closure:

- **The requirement is close to binary.** Narrowing the count distribution partway
  buys very little — the decode falls only slowly until the counts collapse to
  essentially a single value, at which point it drops to chance. There is no gradual
  approach, so "tune the penalty a bit harder" is not a strategy.
- **Soft penalties lose an arms race.** The count channel can be worth more to a
  network than its own accuracy, so it will exploit whatever slack a penalty leaves —
  silence, within-band variation, a heavy tail. A channel is reliably removed by
  **constraining it architecturally**, not by making it expensive.

### 6b. Timing-specific fragility must be separated from general robustness

A network that is more robust to a timing perturbation may simply be more robust to
*everything*. Comparing timing perturbations against each other cannot detect this.

> **Control.** Include a perturbation that destroys **rate** rather than timing (e.g.
> random spike deletion). If the sparsity trend appears there too, it is general
> robustness. Partial the control out of the timing correlation, or compare the two at
> matched damage, before claiming anything about timing.

This control changed the interpretation of the v1 results substantially. It should be
considered mandatory, not optional.

---

## 7. Confounds and pitfalls

1. **The count/identity escape hatch (§5).** The dominant one. Verify closure; do not
   assume it.
2. **General robustness (§6b).** Always run the rate-destroying control.
3. **Statistical dilution.** `spikes per neuron` averaged over silent neurons hides
   the actual per-neuron code. Report the conditional statistic too.
4. **Accuracy confound.** Sparse networks may have lower clean accuracy. Never compare
   raw accuracy drops; use the chance-corrected, baseline-normalised score, and show a
   guard plot of clean accuracy against sparsity to demonstrate the trend is not
   merely "sparser networks are worse".
5. **Broken ≠ temporal.** A network sitting at chance has no representation to probe.
   Drop it from the analysis and say so.
6. **Dead neurons.** Driving spiking *down* can push a neuron so far below threshold
   that its surrogate gradient vanishes and no penalty can revive it. Pushing activity
   *up* is the safe direction. Whatever mechanism is used, watch the silence
   trajectory from the first epoch, not just its final value — a constraint that is
   satisfied early can be given back later as the task loss reasserts itself.
7. **Perturbation magnitude is not matched across sparsity levels.** Displacing a
   fixed fraction of many spikes disturbs a downstream neuron more than the same
   fraction of few. This is a standing limitation of the design; the rate control
   partly addresses it but does not eliminate it.
8. **Readout caveat (inherited).** The probe reveals only the timing that the
   *readout* uses. A layer could hold temporal structure the output layer ignores.
   State it; do not try to fix it here.
9. **Nominal ≠ achieved.** Analyse measured statistics, never the knob that was set.
10. **Seeds.** Sparsity regularisation interacts strongly with initialisation. Use
    several seeds and treat each (setting, seed) as one data point.

---

## 8. What would count as an answer

The milestone is complete when a headline plot exists — timing-dependence against
measured firing rate — **with the manipulation check and the robustness control
documented alongside it**. Any of these is a legitimate result:

| Outcome | Reading |
|---|---|
| Negative correlation, immune channels verified closed | **H1 supported.** Sparsity does shift the code toward timing. |
| Positive correlation, immune channels verified closed | **H1 refuted on its own terms** — the strongest possible negative, since the network had no non-timing option and still did not become more timing-dependent. |
| No correlation, immune channels verified closed | Timing-dependence is invariant to sparsity; sparsity may change the *precision* of a timing code without changing whether one is used. |
| Correlation of any sign, immune channels **not** closed | Uninterpretable as a test of H1. It may still be a valid finding about the intervention (this is what v1 delivered), but the mechanism claim cannot be evaluated. |
| Accuracy collapses to chance once the immune channels are closed | H1 is **untestable in this setting** — the task cannot be done on timing alone here. A real and reportable finding. |

Note that the fourth and fifth rows are not failures of the experiment; they are
findings about the intervention and the task respectively, and both are worth
reporting.

---

## 9. Where to go if the current setting cannot answer it

The difficulty throughout is that on a natural dataset the count channel is *useful*,
so the network fights to keep it, and closing it by force may leave the task
unsolvable. A cleaner venue removes the fight instead of winning it:

**A synthetic task whose class information is carried by inter-spike intervals** and
whose stimuli are constructed so that spike *counts* are uninformative **by design**.
Then no manipulation is needed at all: the count channel carries nothing because the
data put nothing in it, the network has no incentive to preserve it, and the
perturbation sweep measures exactly what it is supposed to. The temporal signal is
also exactly known, which makes it a far stronger confirmation venue than a natural
dataset.

Other axes worth varying once the core question is settled: the perturbation site
(2nd hidden layer as well as 1st), dataset variants, and the presence or absence of
learnable axonal delays — the latter being itself a timing mechanism, so running both
turns a confound into a measurement of how much of the temporal processing the delays
were doing.
