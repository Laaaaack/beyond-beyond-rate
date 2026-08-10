# Both-layer sparse network (v3) — a self-contained summary

**Status: complete (2026-08-09).** 48 networks trained, all eight perturbation sweeps
run at three injection sites, the dependent variables measured, the regressions fitted,
and the cross-grid comparison done.

This page is written for someone who has **not** followed the project. It explains what
the experiment is, why it exists, exactly how it was set up, what came out, and what the
numbers mean. Every term is defined where it first appears. The blow-by-blow execution
log — every calibration run, every rejected setting, every date — lives in
[sparse_network_bothLayer_test_progress.md](sparse_network_bothLayer_test_progress.md);
this document is the readable digest of it.

**Sibling documents, if you want the surrounding context:**
[the original question](sparse_network.md) ·
[v3 at layer 1 only](../legacy/sn_progress/sparse_network_test_progress_v3.md) ·
[v3 at layer 2 only](../legacy/sn_progress/sparse_network_2ndLayer_test_progress_v3.md)

---

## 0. The short version

A supervisor asked a simple-sounding question: *if you make a spiking network's hidden
activity sparser, does it rely more on the precise timing of its spikes?* Two earlier
rounds of this project showed the question hides a trap — "making a network sparser"
actually turns **two** different dials at once, and they push in opposite directions. A
third round (v3) built an experiment that turns those two dials **independently**, first
on the network's 1st hidden layer, then on its 2nd.

**This experiment is the third and last member of that family: it turns both dials on
both hidden layers at once.** It exists because constraining one layer never produces a
sparse *network* — measurements showed the unconstrained layer quietly compensates for
the constrained one, so the single-layer experiments were measuring a network that had
been squeezed in one place and had puffed up in another.

**What came out:**

1. **The design worked.** The two dials were genuinely separated network-wide (the
   confound between them fell from −0.879 to −0.199 in one arm), and the two layers'
   behaviour finally moved together instead of in opposition (+0.977 / +0.944, against
   −0.829 / +0.203 before).
2. **The hypothesis under test (H1′) gained no support.** The coefficient that would
   have to be negative for it is statistically indistinguishable from zero at every
   injection site in both arms — and the single coefficient that *is* significant runs
   the wrong way for H1′.
3. **The rival account (H2) is supported.** What actually predicts a network's reliance
   on spike timing is not how few spikes its neurons fire, but how often neurons go
   completely silent. A directly manipulated comparison confirms it in both arms.
4. **The payoff, and it was not one of the two outcomes anticipated in advance.**
   Comparing all three grids: the effect that looked like "sparsity matters" is
   significant in **3 of 4** cells where exactly one layer was constrained and in **0 of
   2** where the whole network was. The apparent effect was the *other layer
   compensating*, not sparsity — and it disappears once there is no other layer left free
   to compensate.

---

## 1. Where the question comes from

### 1.1 The original question

> *"Induce different levels of sparsity in the hidden layers (through regularisation),
> then observe whether sparser-activity networks do more temporal processing than denser
> ones."*

Formally, **H1: as hidden activity gets sparser, the network relies more on precise
hidden spike timing and less on a rate code.**

The intuition is appealing. A busy neuron firing 10 spikes can carry information in
*how many* it fires (a rate code). A neuron firing 0 or 1 spikes cannot — the only thing
left to vary is *when* that single spike happens (a timing code). So squeezing the
spike budget should push information out of the rate channel and into the timing
channel.

The measuring device is a **fixed-weight perturbation sweep**: train the network on
clean data, freeze it, then at evaluation time damage its hidden spikes in a way that
scrambles *when* spikes occur while keeping *how many* each neuron fires exactly
unchanged. If accuracy collapses, the network was leaning on timing. If accuracy barely
moves, it was leaning on something the damage did not touch.

> **Protocol rule.** Sparsity is induced **during training**, on clean data. The
> perturbation is applied **only at evaluation**, to an already-frozen network. Mixing
> the two (training under perturbation) makes the network route around the damage and
> erases the effect being looked for.

### 1.2 Why the first answer could not be trusted

Round 1 ("v1") ran exactly this and got a clean answer in the **opposite** direction:
sparser networks were consistently *more* robust to timing damage, not less
(correlations +0.67 to +0.96 across every perturbation and arm).

The diagnosis was that a hidden layer can carry class information in three distinguishable
places, and a timing perturbation only destroys one of them:

| Channel | What it is | Survives a timing-only perturbation? |
|---|---|---|
| **Count** | how many spikes each neuron fires | **yes, exactly** |
| **Identity** | *which* neurons fire at all | **yes, exactly** |
| **Timing** | *when* the spikes occur | **no — this is what gets destroyed** |

The key realisation: making a layer sparse does **not** close the count channel. Count
resolution is a *population* property. One bit per neuron across 128 neurons is 128 bits
— far more than a 20-class problem needs. Worse, sparsity often arrives by neurons
falling *silent* on some stimuli and not others, which builds a stimulus-selective
**identity** code — a "labelled line" that a timing perturbation leaves perfectly intact
and which gets *richer* as sparsity increases.

So v1's result was real but uninterpretable as a test of H1: the intervention was
plausibly pushing the network *toward* a perturbation-immune code rather than away from
one.

### 1.3 "Sparsity" is two dials, not one

This is the central correction, and everything downstream depends on it. Note the
identity:

```
spikes per neuron  =  (1 − silent fraction)  ×  spikes per ACTIVE neuron
```

The two factors on the right are different things doing different jobs:

| Symbol | Name | Plain description | What it does to the code |
|---|---|---|---|
| **`a`** | temporal sparsity | how many spikes a neuron fires *on the trials where it fires at all* | fewer spikes per neuron ⇒ less count resolution per neuron ⇒ pushes toward latency/timing coding. **This is the dial H1 is about.** |
| **`s`** | selectivity | the fraction of (trial, neuron) pairs where the neuron fires *nothing* | neurons silent on some stimuli ⇒ a labelled-line **identity** code, which timing perturbations leave intact. **H1 says nothing about this.** |

In v1 a single rate penalty moved both at once, and in the no-delay arm they were
correlated at **ρ = −0.94**. At that level of collinearity no analysis can tell which of
the two produced the effect. **v1's design, not its execution, was what prevented an
answer.**

So the hypothesis was restated so that it can actually be tested, alongside the rival
account that v1's data actually support:

> **H1′** — holding selectivity `s` fixed, reducing spikes per active neuron `a`
> **increases** the network's reliance on hidden spike timing. *(Predicts a negative
> coefficient on `a`.)*
>
> **H2 (identity-code account)** — the anti-H1 trend in v1 is driven by `s`, not `a`:
> silencing builds a perturbation-immune identity code, and that is what makes sparse
> networks look robust. *(Predicts a negative coefficient on `s`, and that removing
> silence should raise timing reliance.)*

These make **opposite, separable** predictions in a design where `a` and `s` vary
independently. Producing such a design is what "v3" means.

### 1.4 What v3 found at one layer

- **Layer 1 only (24 models per arm):** the two dials were successfully separated, and
  H1′ was **refuted on its own terms** — the coefficient on `a` came out significantly
  **positive** (the direction opposite to H1′'s prediction) in both arms.
- **Layer 2 only (24 / 30 models):** same design one layer deeper. Null in one arm, and
  significantly **positive** again in the other.

Both of those constrain **one** layer. Neither asks what happens when the *network* is
made sparse — and that turns out to be a genuinely different question.

---

## 2. What this experiment adds: the both-layer question

### 2.1 The measurement that justifies the whole exercise

Before building anything, the two layers of all 27 frozen v1 checkpoints were measured
together (no retraining — the numbers were already on disk). Under v1's **layer-1-only**
penalty:

| across v1's sparsity gradient | no-delay arm (n=15) | delay arm (n=12) |
|---|---|---|
| **ρ(`s1`, `s2`)** — do the two layers' silence levels move together? | **−0.829** | **+0.203** |
| ρ(`a_net`, `s_net`) — the two dials, pooled over both layers | **−0.879** | +0.042 |
| ρ(`a1`, `a2`) — do the two layers' activity levels move together? | +0.961 | +0.839 |
| `spikes/neuron` span, layer 1 alone | **6.50×** | 5.51× |
| `spikes/neuron` span, **network-wide** | **2.83×** | 2.11× |

**Read the first row.** In the no-delay arm, as layer 1 is driven toward silence, layer 2
becomes *less* silent. A layer-1 penalty does not make the network selective — **it
trades layer 1's silence for layer 2's activity.** In the delay arm the relationship is
merely absent rather than opposed, which is the same failure in a weaker form.

### 2.2 Three consequences

1. **Neither single-layer grid ever produced a sparse network.** v1's densest and
   sparsest no-delay models differ by 6.5× in *layer-1* firing but only 2.8×
   *network-wide*, because layer 2 absorbs part of the cut. Every effect size quoted per
   layer overstates the actual manipulation by **42–62%**.
2. **The compensation runs against the manipulation's intent.** Because v3's residual
   signal is *selectivity*, a design in which the two layers' selectivity moves in
   opposite directions is measuring a partly cancelled quantity.
3. **It gives the two single-layer grids their control.** With all three run, the same
   design exists at layer 1 alone, layer 2 alone, and both. If the key coefficient is
   null everywhere, the null is about the architecture; if it appears only when the whole
   network is constrained, that is a positive result no single-layer grid could have
   produced. *(As §6.6 shows, the data chose a third option neither of these anticipated.)*

**This is a replication in venue, not a new hypothesis.** H1′ and H2 are unchanged.

---

## 3. Vocabulary

Everything you need to read §6, in one table.

| Term | Meaning |
|---|---|
| **arm** | one of two network variants: **delay** (has learnable axonal delays — small trainable time-shifts on each neuron's output) and **no-delay** (does not). Delays are themselves a timing mechanism, so running both turns a confound into a measurement. |
| **`a`** | spikes per **active** neuron per trial. "How busy are the neurons that fire at all." |
| **`s`** | silent fraction — share of (trial, neuron) pairs firing nothing. "How often does a neuron stay quiet." |
| **`a_net` / `s_net`** | the same two quantities pooled over both hidden layers (exact pooling over all (trial, neuron) pairs, not an average of two ratios). |
| **ceiling** | the training penalty that sets `a` — charges a cost for every spike above a budget `k`. |
| **floor** | the training penalty that sets `s` — forbids a neuron from staying silent. Its strength is either 0 (off) or 1 (on). |
| **`k`** | the ceiling's spike budget per (trial, neuron) pair. Lower `k` = harsher squeeze. |
| **relocation / jitter / shift** | three **timing** perturbations, applied at evaluation only. Each moves spikes in time while preserving each neuron's spike count *exactly*. |
| **deletion** | the **control** perturbation: randomly removes spikes. It destroys rate, not timing. Included because a network that resists timing damage may simply resist *all* damage. |
| **retention** | chance-corrected fraction of accuracy surviving a perturbation: `(acc_perturbed − .05) / (acc_clean − .05)`. |
| **usage** | `1 − retention` under the strongest timing perturbation. **"How much the network's own readout leans on spike timing."** Higher = leans harder. |
| **control** | the same score under 80% spike deletion. **"How fragile the network is to damage in general."** |
| **availability** | **"How much timing information the hidden layer contains"**, whether or not the network uses it. Measured by decoding, with no perturbation involved (see §4.5). |
| **injection site** | where the evaluation-time damage is applied: `l1` (1st hidden layer), `l2` (2nd), or `both`. |
| **β_a, β_s** | regression coefficients on `log(a)` and `s` when predicting usage. H1′ predicts β_a < 0; H2 predicts β_s < 0. |
| **β\*** | the standardised version of a coefficient — comparable across variables on different scales. |

---

## 4. Setup

### 4.1 Task and network

| | |
|---|---|
| **Dataset** | SHD (Spiking Heidelberg Digits) — spoken digits 0–9 in English and German, converted to spike trains by a simulated cochlea. **20 classes**, so chance accuracy is **5%**. |
| **Input** | 700 input channels × 200 time bins of 1 ms. (The raw data holds 100 bins, zero-padded to the simulator's 200 — this matters, see §4.5.) |
| **Splits** | train 0–60%, validation 60–75%, test 75–90% of the file. All reported numbers are **test set**. |
| **Network** | 700 → **128** → **128** → 20. Two hidden layers of spiking neurons (SLAYER `SRMALPHA`, threshold θ = 10), one output neuron per class. |
| **Delays** | in the delay arm, `delay1` sits **between** the two hidden layers and `delay2` between the 2nd hidden layer and the output; each is learnable, clamped to 0–64 bins. |
| **Loss** | `NumSpikes` — the correct class's output neuron should fire 40 spikes, the others 4. This pins the *output* firing rate, which keeps the sparsity intervention confined to the hidden layers. |
| **Optimiser** | Nadam, lr 0.1, step decay ×0.1 at epoch 300, batch 128. |
| **Training length** | **1250 epochs**, early-stopping patience 300. (Long: v1 models were still improving at epoch 1250, so short runs are a *different regime*, not a scaled-down one.) |

### 4.2 The two penalties — the two dials

Both are added to the task loss and, uniquely to this experiment, **charged once per
layer and summed**:

```python
loss = task_loss
for layer in (1, 2):
    loss += 10.0 * relu(count[layer] - k[layer]).mean()      # ceiling  -> sets a
    loss += floor_strength * relu(11 - peak_potential[layer]).mean()   # floor -> sets s
```

**Ceiling (sets `a`).** `relu(count − k)` charged per **(trial, neuron) pair**, not on
the batch average. Per-pair is the whole point: a batch-average penalty lets a neuron
satisfy the budget by going *silent* on some trials and firing freely on others — which
is exactly how `a` and `s` became confounded in v1.

**Floor (sets `s`).** It penalises the **membrane potential**, not the spike count:
`relu(11 − max_t u(t))`. Two reasons, both measured rather than assumed. Reaching
threshold at least once guarantees at least one spike, so this is sufficient to prevent
silence. And the gradient of the potential with respect to the weights is an ordinary
convolution gradient — **no surrogate gradient is involved** — so it still carries signal
however far below threshold a neuron sits. Silent neurons here were measured to sit at a
peak potential of ~0–2 against a threshold of 10, i.e. *deeply* silent, where a
count-based penalty (which must go through the surrogate) carries no signal at all.

**Charging each layer at the full coefficient** (rather than halving) is deliberate: it
means one cell of this grid applies to layer 1 exactly the pressure the layer-1 grid
applied, and to layer 2 exactly what the layer-2 grid applied — which is what makes the
three grids comparable cell by cell.

Two guards were needed and both were calibrated on measured runs:

- **`WARMUP_EPOCHS = 20`** — the ceiling is minimised at *zero spikes*, so in the
  floor-off column nothing but the task loss opposes total silence, and total silence is
  an **absorbing state** (once every neuron sits far below threshold, only the surrogate
  gradient can revive it, and it carries no signal there). With warm-up 0 the layer is
  100% silent by epoch 5 and never recovers.
- **`SETTLE_EPOCHS = 150`, delay arm only** — see §5.3.

### 4.3 The grid

| knob | levels | what it varies |
|---|---|---|
| `k` (ceiling budget) | **1, 2, 4, 8** at layer 1 | the `a` axis — the H1′ axis |
| floor strength | **0, 1** | the `s` axis — the H2 axis |
| seed | **42, 43, 44** | replication |
| arm | delay, no-delay | run as two separate grids |

**24 models per arm, 48 in total**, all at 1250 epochs, trained on a remote server.

**One substantive design change from the single-layer grids: the ceiling budget is
scaled per layer.** `CEILING_K` lists layer 1's levels; layer 2 gets
`CEILING_K_LAYER2_RATIO` times them, so one knob delivers an equal *relative* cut at both
layers rather than an equal absolute budget:

| arm | natural `a` (L1 / L2) | ratio | `k1` | `k2` |
|---|---|---|---|---|
| no-delay | 3.06–11.49 / 7.49–12.46 | **1.0** (a no-op — the ranges overlap) | 1, 2, 4, 8 | 1, 2, 4, 8 |
| delay | 3.82–15.47 / **20.71–30.44** | **2.0** | 1, 2, 4, 8 | **2, 4, 8, 16** |

The delay arm's layer 2 naturally runs about twice as hard as its layer 1, so a shared
budget would be a mild cut for one layer and a savage one for the other in the same cell.

**Artifacts are tagged `v3L12`.** All three grids share dataset, arm, `k`, floor and
seed, so without distinct tags (`v3`, `v3L2`, `v3L12`) they would produce identical
filenames and silently overwrite one another.

### 4.4 The dependent variables

Three quantities are measured per trained model, on the test set, with the network frozen.

**1. Usage — does the network's own readout depend on spike timing?**
Apply the strongest timing perturbation (relocation at `f = 1`: every hidden spike is
moved to a random bin, per-neuron count preserved exactly), measure the accuracy drop,
chance-correct it. `usage = 1 − retention`.

**2. Control — is it fragile to damage in general?**
The same score under 80% random spike **deletion**, which destroys rate rather than
timing. This is *not* optional: in v1's no-delay arm the control tracked sparsity at
+0.936 against the timing probe's +0.939 — i.e. a pure rate insult and a pure timing
insult were measuring the same thing. It goes in the regression, not in a footnote.

**3. Availability — does the hidden layer *contain* timing information at all?**
No perturbation involved. Four linear decoders (logistic regression) are fit on the
train split and scored on test, on the same hidden activity:

| view | features | destroys what? |
|---|---|---|
| `COUNT` | per-neuron spike count (128 features) | — |
| `IDENT` | the same, binarised | — |
| `FULL` | per-neuron counts in 10 time bins (1280 features) | nothing |
| `SHUF` | `FULL` after redrawing each neuron's spikes from *that neuron's own* average temporal profile, count preserved exactly | only the trial-specific placement of spikes |

`availability = (FULL − SHUF) / (FULL − chance)` — the share of decodable class
information that genuinely requires knowing *when* spikes occurred. `SHUF` is the right
null because it holds feature count, decoder, sample count, per-neuron spike count and
per-neuron average timing all identical.

Availability and usage are **two different questions** and they have repeatedly
disagreed: the layer can hold just as much timing information while the readout uses less
of it. Both are always reported.

**A subtlety that took real work: the perturbation window.** The data occupies 100 of
the 200 simulated bins, and layer-1 hidden spikes were measured to stop at bin 87. If a
relocation scatters spikes uniformly over all 200 bins, ~57% land where no hidden spike
ever naturally occurs — thinning the population's spike density by ~5× and smuggling a
*rate* insult into what must be a timing-only probe. So relocation and jitter draw
destinations from each layer's own measured support (layer 1 `[0,88)` in both arms; layer
2 `[0,90)` no-delay and **`[0,160)`** delay, because `delay1` shifts layer 1's spikes by
up to 64 bins before layer 2 sees them). **Shift and deletion are deliberately left
uncorrected** — clipping a rigid translation would pile spikes at the edge and merge
them, destroying per-neuron count, i.e. *introducing* the artifact the correction removes.

**Three injection sites.** Every checkpoint was swept at `l1`, `l2` and `both`. Applying
the damage at both layers at once is a *different and harsher* insult than either alone —
at the `both` site, layer 2 is damaged after already being computed from a damaged layer
1 — so the three sites are reported separately rather than collapsed into one
"network-wide" number. §6.3 shows that this caution was justified empirically.

---

## 5. Checks made before any result was believed

### 5.1 Implementation verification

Twenty automated checks per arm, on GPU, no training. All passed. The two that could not
have been settled by reading the code:

- **"The first layer is now squeezed twice"** — the layer-2 ceiling also pushes gradient
  back into `fc1`, measured at |grad| 0.81 (no-delay) and 2.05 (delay). Real, quantified,
  and it argued for possibly *lowering* the ceiling strength.
- **A known dead gradient path reproduces exactly.** The layer-2 floor's gradient is
  surrogate-free but **exactly zero** for any trial whose entire layer 1 is silent. Forced
  that condition; the gradient came back 0.000000. The failure mode is real and reachable,
  which is why survival at the harshest corner is the *first* thing the probe checks.

### 5.2 The corner probe

8 models (harshest and gentlest `k` × floor off/on, one seed, 400 epochs, 5.9 h) were
trained before committing ~40 h of server time, and checked against five criteria:

| # | criterion | verdict |
|---|---|---|
| 1 | both layers survive the harshest corner (`k=1`, floor off) | **PASS** — accuracy .441 (no-delay) / .717 (delay) against chance .05; neither layer fully silent |
| 2 | `s` roughly constant as `k` varies within a floor column | **partial** — it drifts, as in every grid in this project |
| 3 | the floor separates `s` by ≥ 20 points at **both** layers | **PASS with huge margin** — +52 to +63 points |
| 4 | firing not inflated past each layer's natural rate | **PASS** |
| 5 | clean accuracy comfortably above chance | **PASS** |

The no-delay arm passed unchanged. Layer 1 there behaved almost identically to its
single-layer sibling (within ~2 points of `s` and 0.1 spikes of `a` in every cell), which
is the cleanest possible outcome for a design meant to be comparable cell by cell.

The delay arm did **not** pass first time, and chasing that produced the single most
transferable finding of the whole experiment.

### 5.3 The model-selection fault — the lesson worth carrying away

The delay arm's probe appeared to show the ceiling barely binding: **62% of layer-1
pairs still fired above their budget** in the floor-on column. The obvious fix (raise the
ceiling strength from 10 to 20) was tried and cost **.153 of accuracy** — expensive.

**But the ceiling was never the problem.** Best-model selection uses the **task**
validation loss (a deliberate choice: the saved checkpoint is chosen on the task, not on
how well the constraint is satisfied). In the delay arm's floor-on cells the task
validation loss is best about 20 epochs *after* the penalties engage and never improves
again — so early stopping fired at ~epoch 340, and **the checkpoint that got saved and
measured was the one from epoch 37–45**: a network that had had the constraint applied
for roughly twenty epochs. Trained out, the same run sits at **36.9%** over budget, not
62%, with a *wider* manipulation axis.

> **Every sparsity statistic measured off such a checkpoint is wrong in the direction of
> "the penalty isn't working".** It cost this experiment one incorrect diagnosis, and it
> silently affects 2 of the 16 cells in the already-completed layer-1 delay grid.

**The fix:** `SETTLE_EPOCHS = 150` — track the best validation loss only from
`warmup + settle` onward. This keeps selection on the task loss while refusing
checkpoints from before the manipulation exists. It is nearly a no-op on healthy runs
(it would have changed 1 of 16 cells in the completed layer-1 grid), and on the real
1250-epoch grids it caught **3 of 24 delay cells** whose global best validation loss sat
at epoch 42–44, moving their selection to epochs 567 / 716 / 1197 and their over-budget
fraction from ~50–61% down to 23–38%.

The no-delay arm was measured *not* to need it (its earliest selection is epoch 617 of
917), so the two arms deliberately differ in selection policy, with the asymmetry and its
evidence recorded in the script.

**After the fix, all 48 cells are clean**: every one selects at 0.65–1.00 of its run
length, with the constraint state unchanged between the selected and the final epoch.

---

## 6. Results

### 6.1 Did the manipulation work?

This gates everything else. The design acceptance criterion is |ρ(`a`, `s`)| ≤ 0.5 —
i.e. the two dials must be genuinely separable. Pearson, n = 24 per arm:

| | no-delay: v1 → **this grid** | delay: v1 → **this grid** |
|---|---|---|
| **ρ(`a_net`, `s_net`)** — the acceptance number | −0.879 → **−0.199** (p=.35) ✔ | +0.042 → **+0.040** (p=.85) ✔ |
| ρ(`a1`, `s1`) — layer 1 alone | −0.943 → **−0.297** ✔ | −0.455 → **−0.001** ✔ |
| ρ(`a2`, `s2`) — layer 2 alone | +0.829 → **−0.122** ✔ | +0.427 → **+0.106** ✔ |
| **ρ(`s1`, `s2`)** — the motivating finding | −0.829 → **+0.977** | +0.203 → **+0.944** |
| ρ(`a1`, `a2`) — excluded by design | +0.961 → +0.973 | +0.839 → +0.893 |

*(The regression script reports the same check as a Spearman rank correlation: −0.422,
p = .040 no-delay and −0.064, p = .77 delay. Both forms pass.)*

Three readings:

1. **The no-delay arm's pooled confound is broken** — from −0.879 to −0.199. That was
   the single thing this design existed to do in that arm.
2. **Both layers pass individually too**, which matters more than it looks: the pooled
   decorrelation is not an artifact of averaging two layers that are each still
   confounded. Layer 2's v1 confound ran the *opposite* way to layer 1's, and the same
   two knobs pulled both to near zero.
3. **The motivating finding is confirmed and is not marginal.** ρ(`s1`, `s2`) went from
   −0.829 (layers opposing each other) to **+0.977** (layers moving together). The column
   knob **is** a network-wide selectivity manipulation, so the pooled `s_net` axis is a
   statement about the network and not about one layer of it. **This is the one result no
   single-layer grid could have produced.**

The last row is what this design deliberately does *not* fix: both knobs act on both
layers, so the two layers' activity levels move together **by construction**. That is a
property of the question, not a defect — but it means **every coefficient from this grid
is a network-level coefficient and cannot be attributed to a layer.**

### 6.2 How hard the knobs pushed, and what it cost

| | no-delay | delay |
|---|---|---|
| **`a` axis span** (floor off / on, network-wide) | 1.95× / 2.07× | 1.87× / 2.25× |
| **`s` range across the grid** | 4.1% – 69.9% | 3.4% – 57.3% |
| **floor separation in `s`** (at matched `k`, per layer) | +43.8 … +57.6 pts | +27.6 … +45.9 pts |
| **clean accuracy** (cell means) | .480 – .572 | .739 – .840 |
| v1's accuracy range, for reference | .49 – .59 | .78 – .88 |
| rate inflation? | none — max 3.59 / 4.99 vs natural 7.85 / 5.53 | none — 4.21 / 8.52 vs 11.66 / 23.93 |

- **The floor separation passes in all 16 row × layer checks**, the smallest margin
  (+27.6 points) still 38% above the 20-point threshold.
- **The floor costs nothing and helps.** Floor-on beats floor-off on clean accuracy in
  all 8 rows of both arms.
- **The probe was a good instrument.** It predicted the delay arm's floor-on layer-1 axis
  to two decimal places (1.82×) and its network axis to within 0.06×.

Three honest limitations, all recorded rather than smoothed over:

- **`s` is not perfectly flat as `k` varies.** The floor-off column drifts by ~30 points
  at layer 1 in both arms. So the row knob moves `s` somewhat as well as `a` — which is
  precisely why the design *decorrelates* the two axes rather than claiming they are
  orthogonal by construction.
- **Layer 2's ceiling leaks more than layer 1's** in 15 of 16 cells (at the tightest row,
  up to 59% of pairs still over budget). Layer 2's row axis is the softer of the two.
- **One of twelve activity curves is not monotone** — the delay arm's floor-off layer 1
  dips at `k = 2` before recovering. The pooled axis, which is what this grid actually
  manipulates, is monotone.

### 6.3 What the perturbation sweeps show

Mean chance-corrected **retention** at the strongest setting of each sweep (higher =
less damage), averaged over the 8 cells per arm:

| arm | perturbation | at `l1` | at `l2` | at `both` | `both` − `l1` |
|---|---|---|---|---|---|
| no-delay | relocation | .683 | .985 | .705 | **+.022** |
| no-delay | jitter | .693 | .963 | .711 | **+.018** |
| no-delay | shift | .515 | .706 | .411 | −.104 |
| no-delay | **deletion (control)** | .410 | .518 | .187 | **−.223** |
| delay | relocation | .434 | .830 | .445 | **+.011** |
| delay | jitter | .498 | .886 | .470 | −.028 |
| delay | shift | .317 | .788 | .268 | −.049 |
| delay | **deletion (control)** | .296 | .566 | .142 | **−.154** |

1. **Layer 2 is much the cheaper place to be damaged** — on every perturbation in both
   arms, by a wide margin. (Whether that means layer 2 carries less of the timing code, or
   merely that damage there has one fewer layer to propagate through, these sweeps cannot
   separate.)
2. **`both` is not a uniform escalation over `l1`.** Relocation and jitter **saturate** —
   adding the layer-2 insult on top of the layer-1 one changes nothing measurable. Both of
   those redistribute spikes *inside* a window and preserve per-neuron count exactly, so
   once layer 1's placement code is gone there is little left for a second pass to take.
3. **Deletion is the one that compounds** (−.223 / −.154), and it is the only
   perturbation that does not preserve count. That is exactly the ordering a compounding
   argument predicts — and it is the empirical reason the three sites are kept separate
   rather than collapsed.

### 6.4 The endpoint measures

Means over 24 models per arm; `a`/`s` are the network axes.

| | no-delay | delay |
|---|---|---|
| **availability** (timing fraction) at `l1` / `l2` / `net` | .270 / .234 / **.224** | .233 / .079 / **.097** |
| **usage** at `l1` / `l2` / `both` | .315 / .016 / **.296** | .565 / .171 / **.551** |
| **control** (deletion) at `l1` / `l2` / `both` | .594 / .487 / **.819** | .704 / .435 / **.861** |
| **readout gap** (decoder accuracy − network's own accuracy) at `net` | **+0.285** | **+0.076** |

- The site structure from §6.3 reproduces here: **usage saturates** (`both` ≈ `l1`) while
  the **control compounds** (.594 → .819; .704 → .861). Two independently written scripts
  agree, which is the check that they share their injection sites and windows.
- **Layer 2 is both the cheaper place to be damaged and the poorer place to read timing
  from.** In the delay arm its availability is .079 against layer 1's .233, and its usage
  .171 against .565. The two measures need not have agreed; they do.
- The **readout gap** says how much class information a simple linear decoder can pull
  out of the hidden network that the network's *own* output layers fail to use. At +0.285
  in the no-delay arm, that arm's perturbation score is as much a statement about its
  readout as about its hidden code. See §6.7.

### 6.5 The regressions — the actual test

Model: `usage ~ log(a_net) + s_net + control`, n = 24 per arm. H1′ needs **β_a < 0**;
H2 needs **β_s < 0**.

| | no-delay | delay |
|---|---|---|
| design acceptance check | PASS | PASS |
| **β_a** at `l1` / `l2` / `both` | +.025 / −.001 / **−.0005** (p=.99) | −.010 / **+.137\*** / **+.021** (p=.32) |
| **β_s** at `both` | **−0.168** (p=.049), β\* = −0.66 | −0.089 (p=.12), β\* = −0.35 |
| β_a for **availability** at `net` | −0.035 (p=.030) | +0.003 (p=.68) |
| largest standardised term at `both` | β_s | **control** (β\* = +0.537) |
| **matched floor-on vs floor-off contrast** | **+0.100** (p=.0001, 12 pairs) | **+0.068** (p=.011, 8 pairs) |

\* the one significant β_a in the whole table is layer-2 usage in the delay arm, and it
runs **+0.137** — i.e. *against* H1′, not for it.

**The last row is the strongest single piece of evidence in this experiment.** It is a
*manipulated* comparison rather than a correlation: take pairs of models matched on
activity (|Δ log `a`| ≤ 0.2) but differing in whether the floor was on, and ask whether
removing silence raises timing reliance. H2 predicts it should. **It does, significantly,
in both arms** — and this is the first time this project has had that comparison at the
network level.

**Verdict: H2 confirmed in the no-delay arm; inconclusive but directional in the delay
arm.** β_a is null at every site in both arms except the one that runs the wrong way for
H1′; β_s carries the effect; and the controlled contrast is positive and significant in
both arms.

### 6.6 The payoff — β_a across all three grids

Because all three grids were re-measured with the *same* retargeted scripts (one script
with a `GRID` knob, not three forks that drifted apart), the same coefficient measured
the same way now exists for all three:

| arm | grid | constrained layers | n | **β_a** | 95% CI | p | β\* |
|---|---|---|---|---|---|---|---|
| **delay** | `v3` | layer 1 | 24 | **+0.085** | [+0.008, +0.162] | **.033** | +0.394 |
| **delay** | `v3L2` | layer 2 | 30 | **+0.087** | [+0.062, +0.113] | **<.0001** | **+0.638** |
| **delay** | `v3L12` | **both** | 24 | +0.021 | [−0.021, +0.063] | .32 | +0.137 |
| no-delay | `v3` | layer 1 | 24 | **+0.200** | [+0.118, +0.283] | **.0001** | +0.495 |
| no-delay | `v3L2` | layer 2 | 24 | −0.007 | [−0.031, +0.016] | .53 | −0.081 |
| no-delay | `v3L12` | **both** | 24 | −0.0005 | [−0.078, +0.076] | .99 | −0.002 |

**β_a is significantly non-zero in 3 of the 4 single-layer cells and in 0 of the 2
both-layer cells.**

**And it is not a power story.** The both-layer confidence intervals **exclude** the
single-layer point estimates in 3 of 4 comparisons. The best-powered single-layer cell is
`v3L2` delay — n = 30, the widest achieved activity axis of any grid (3.50×), the largest
standardised effect (+0.638) — so the single-layer effect is not marginal and the
both-layer null is not underpowered.

**The obvious deflationary explanation was checked and does not hold.** "The network-wide
axis is just a weaker manipulation" would explain the pattern — but the two factorials'
*achieved* axes are the same size (`a` spans 2.26× / 2.18× in the layer-1 grid against
2.11× / 2.72× here; `s` spans are comparable too). **The coefficient moved while the axis
it is a coefficient on did not.**

§2.2 offered two possible outcomes — a null everywhere (a fact about the architecture),
or an effect only under the network-wide constraint (a result no single-layer grid could
produce). **The data chose the mirror image of the second: the effect exists only while
exactly one layer is constrained, and vanishes when the network is.** That is still a
result no single-layer grid could have produced — but it is a result *about the
single-layer designs*, not about sparsity.

### 6.7 Which arm is primary

The layer-1 experiment chose the delay arm on two measurements taken **at layer 1 only**.
Rather than inherit them across a layer boundary — the exact mistake this calibration
exists to avoid — both were re-measured on this grid's own checkpoints:

| criterion | no-delay | delay | favours |
|---|---|---|---|
| readout-efficiency gap at `net` (smaller is better) | **+0.285** | **+0.076** | **delay** |
| \|β_a(usage) − β_a(control)\| at `both` (larger is better — the timing probe must be distinguishable from a pure rate insult) | −0.012 | +0.015 | delay, marginally |

**The readout gap is the decisive number and constraining both layers did not improve
it**: in the no-delay arm a simple linear decoder reading the hidden network beats the
network's own output layers by .285, on **100% of checkpoints**. Perturbation experiments
in that arm are measuring the readout's inefficiency as much as the hidden code's
structure, and no factorial can fix that.

**Primary arm: delay. The no-delay arm is kept as secondary, not discarded** — it is the
only arm whose pooled confound actually failed, so it is the only arm where this design
had something to break, and with no learnable delays any surviving timing effect there is
attributable to membrane dynamics alone.

---

## 7. Analysis — what the numbers mean

**1. H1′ gained no support at the network level, in either arm, at any injection site.**
Holding selectivity fixed by construction and controlling for general robustness, the
coefficient on activity is indistinguishable from zero everywhere except one cell, where
it runs the *wrong way* for H1′. Squeezing the whole network's spike budget does not make
it lean harder on spike timing.

**2. Selectivity is the variable that survives every venue.** β_s is negative in both
arms (significant in the no-delay one), and the manipulated floor-on/floor-off contrast
at matched activity is positive and significant in **both** arms. Read plainly: *what
makes a "sparse" network look robust to timing damage is not that its neurons fire few
spikes — it is that its neurons go completely silent on many stimuli, which builds an
identity code that timing damage cannot touch.* The supervisor's original question turns
out not to have a single answer, because the answer depends on **which channel the
sparsity mechanism leaves open**.

**3. The single-layer effect was the other layer compensating.** This is the strongest
new claim, and it follows a chain of measurements rather than an argument. Under a
layer-1-only penalty the two layers' selectivity moves in *opposite* directions
(ρ = −0.829). So a single-layer β_a is fitted on a network in which one layer was
sparsified and the other compensated — and it tracks that compensation. Remove the
compensator by constraining both layers, and the coefficient goes to zero while β_s
survives. The calibration predicted this before the grid was run; §6.6 is the same effect
appearing in a coefficient.

**4. A great deal of what looks like "timing reliance" is not about timing.** At the
`both` site the deletion control — pure rate damage, no timing component — is the
**largest standardised term** in the delay arm's primary model (β\* = +0.537), and it
reaches .82–.86 where usage reaches only .30–.55. A network-wide insult that destroys
rate does far more damage than one that destroys only timing, at every sparsity level
tested. Any reading of "timing reliance" that does not hold general robustness fixed is
partly a statement about something else entirely.

**5. The two hidden layers are not two copies of the same thing.** Layer 2 is cheaper to
damage, holds less decodable timing information, is used less by the readout, and its
ceiling leaks more. Its natural firing rate is twice layer 1's in the delay arm and
*lower* than layer 1's in the no-delay arm. No constant calibrated at one layer should
ever be carried to the other — and none was.

**6. Availability and usage are still two different things.** In the no-delay arm the
amount of timing information *present* in the hidden network **rises** slightly as
activity falls (β_a = −0.035, p = .030 — H1's own direction: fewer spikes per active
neuron ⇒ more of the decodable information requires timing), while the readout's *use* of
it does not move at all (β_a = −0.0005, p = .99). The layer shifts toward a timing code;
the readout does not follow. Reporting either measure alone would give a different
impression of the same network.

---

## 8. What this experiment cannot say

1. **It cannot attribute anything to a specific layer.** Both knobs move both layers, so
   the two layers' activity is correlated at +0.97 / +0.89 **by construction**. Every
   coefficient here is a network-level coefficient. Layer attribution is what the two
   single-layer grids are for.
2. **The activity axis is soft, not pinned.** The ceiling still leaks — up to 59% of
   pairs over budget at the tightest row at layer 2. The manipulation is a real
   manipulated range (~2×), not a hard constraint, and layer 2's is the softer of the two.
3. **One activity curve is non-monotone** (delay arm, floor off, layer 1), so any
   layer-1 activity claim in that arm needs that cell checked first.
4. **The no-delay arm's readout leaves ~.285 of accuracy unextracted on every
   checkpoint.** Its perturbation scores are partly a statement about its output layers
   rather than about its hidden code. This is why the delay arm is primary.
5. **`s` is not perfectly constant along the `a` axis** (up to ~30 points of drift in the
   floor-off column). The design decorrelates the two axes; it does not make them
   orthogonal by construction.
6. **Perturbation magnitude is not matched across sparsity levels.** Displacing a fixed
   fraction of many spikes disturbs a downstream neuron more than the same fraction of
   few. The deletion control partly addresses this; it does not eliminate it.
7. **There is no non-sparse reference curve for the `both` injection site.** The existing
   unconstrained-baseline sweeps were only ever generated one layer at a time, so the
   `both`-site figures are plotted without a reference line rather than against a
   strictly gentler insult.

---

## 9. Where everything lives

**Training scripts**
[sn_bothLayer_train_noDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_noDelay_v3.py) ·
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py)

**Checkpoints and training logs** —
`sn_data/sparse_whole_{arm}_v3L12_k{k}_floor{f}_seed{seed}.pt`,
`sn_log/sparse_whole_{arm}_v3L12_train_summary.json` (24 rows each) plus 48 per-epoch logs.

> **Note on the summary keys.** In this grid the canonical unsuffixed keys
> (`spikes_per_active_neuron`, `silent_fraction`, `spikes_per_neuron`) carry the
> **pooled network** values, not layer 1's. Every per-layer value is available under an
> explicit `_l1` / `_l2` suffix — use those when comparing across grids.

**Perturbation sweeps** — eight `*_bothLayer_evalOnly_*_v3.py` scripts under
[shd/](../../exp_sparse_network/shd/) (relocation),
[jitter/](../../exp_sparse_network/jitter/),
[shift/](../../exp_sparse_network/shift/) and
[deletion/](../../exp_sparse_network/deletion/), writing
`{folder}/log/sparse_whole_{arm}_v3L12_*_eval.json`.

**Analysis** — all three scripts carry a `GRID` knob (`v3` / `v3L2` / `v3L12`) so one
script serves all three grids, which is what makes the cross-grid comparison legitimate:
[hidden_channel_decode.py](../../exp_sparse_network/v3_analysis/hidden_channel_decode.py) (availability) ·
[phase1_measure.py](../../exp_sparse_network/v3_analysis/phase1_measure.py) (usage + control) ·
[phase1_regress.py](../../exp_sparse_network/v3_analysis/phase1_regress.py) (regressions, contrast, figures)

**Figures** —
[result_visualization/bothLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/bothLayer/results_visualization.ipynb)
(accuracy curves, one section per injection site, the three sites side by side, and the
manipulation check at both layers and pooled), plus
`v3_analysis/fig/phase1_regress_v3L12_{arm}.png`.

### Reproducing it

```
# training (already done) -- 24 models per arm, 1250 epochs
python sn_bothLayer_train_noDelay_v3.py
python sn_bothLayer_train_withDelay_v3.py

# the four sweeps, per arm, at sites l1 / l2 / both
python shd/shd_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python jitter/jitter_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python shift/shift_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python deletion/deletion_bothLayer_evalOnly_{noDelay,withDelay}_v3.py

# the endpoint measures and the regressions -- set GRID = "v3L12" in all three
python v3_analysis/hidden_channel_decode.py
python v3_analysis/phase1_measure.py
python v3_analysis/phase1_regress.py
```

> **Two filename hazards, both now guarded.** All three grids share dataset, arm, `k`,
> floor and seed, so the `v3L12` tag is load-bearing in the **analysis outputs** as well
> as the checkpoint paths — `phase1_measure.py` originally wrote an untagged file and
> would have silently overwritten the completed layer-1 measurements. And because every
> grid has now been measured, all the filenames exist, so a stale `GRID` in one script
> alone would no longer fail loudly: `phase1_regress.py` therefore checks each input
> file's recorded grid against its own and refuses the join.

Every sweep and every measurement is **evaluation-only**: checkpoints are loaded frozen,
no gradients are taken, and the perturbation is applied at evaluation time only.
