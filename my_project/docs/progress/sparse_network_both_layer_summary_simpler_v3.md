# Does a sparser spiking network use spike timing more? (accuracy-only test)

**Both-layer spike-budget grid (`v3L12`, floor-on column). Status: complete (2026-08-09).**
24 networks, four perturbation sweeps at three injection sites, one dependent measure:
**test accuracy**.

> **Provenance.** The numbers below are the floor-on cells of the `v3L12` both-layer
> factorial, re-analysed under a single hypothesis and a single accuracy-based measure.
> The full two-hypothesis treatment, with the chance-corrected and decoding measures, is
> in [sparse_network_bothLayer_summary_v3.md](sparse_network_bothLayer_summary_v3.md);
> the execution log is [here](sparse_network_bothLayer_test_progress.md).

---

## Abstract

A spiking network trained under a spike-budget penalty fires fewer spikes per neuron.
Does it compensate by making more use of *when* those spikes occur? This experiment
trains a two-hidden-layer SNN on SHD under four spike budgets applied to **both** hidden
layers, with a silence floor switched **on in every cell** so that the budget compresses
active neurons rather than switching neurons off. Every model is then frozen and swept
with three timing perturbations and one rate control, at three injection sites. The only
score used is **test accuracy** — clean, perturbed, and the difference between them.

Three results:

1. **The manipulation worked.** The budget moved spikes-per-active-neuron across a
   **2.32× / 2.11×** range (delay / no-delay) while the floor held the silent fraction
   between 3.4% and 18.1% in all 24 models. All eight per-layer activity curves are
   monotone in the budget.
2. **A tighter budget costs clean accuracy.** Accuracy rises with the budget in both
   arms — .783 → .840 (delay) and .525 → .568 (no-delay) from `k=1` to `k=8`.
3. **H1 received no support; the data point the other way.** H1 predicts that sparser
   networks lose *more* accuracy under a timing perturbation. In all 18 timing
   regressions (3 perturbations × 3 sites × 2 arms) the coefficient carries the
   **opposite** sign, significantly so in 17. Sparser networks lose *less* accuracy — and
   the pure-rate deletion control moves the same way in 5 of its 6 cells, so much of even
   that reversed effect is general robustness rather than anything about timing.

---

## 1. Background and hypothesis

### 1.1 The question

> *Induce different levels of sparsity in the hidden layers through regularisation, then
> observe whether sparser-activity networks do more temporal processing than denser ones.*

The intuition is straightforward. A neuron that fires 10 spikes can carry information in
*how many* spikes it fires — a rate code. A neuron restricted to 0 or 1 spikes cannot.
The only degree of freedom left to it is *when* the single spike occurs — a timing code.
A smaller spike budget should therefore push information out of the rate channel and into
the timing channel.

### 1.2 The measurement instrument

A **fixed-weight perturbation sweep**. Train on clean data, freeze the network, then
damage the hidden spikes at evaluation and re-measure test accuracy. The timing
perturbations change *when* spikes occur while holding each neuron's spike count exactly
fixed. If accuracy falls, the readout was using timing. If accuracy holds, the readout
was using something the damage did not touch.

> **Protocol rule.** Sparsity is induced **during training**, on clean data. Perturbation
> is applied **only at evaluation**, to an already-frozen network. Training under the
> perturbation lets the network learn a route around the damage, which erases the very
> effect being measured.

### 1.3 Why the silence floor is on in every cell

A spike-budget penalty has two ways to satisfy itself. It can make active neurons fire
fewer spikes — the effect the question is about — or it can switch neurons off entirely.
The second route is not a compression of the rate code; it is the construction of a
different code. Neurons that go quiet for some stimuli and not for others build a
stimulus-selective **identity** code, and a timing perturbation leaves an identity code
completely intact. A network that reached its low spike count by falling silent would
therefore look "robust to timing damage" for a reason that has nothing to do with timing.

The floor closes that route. It is a second penalty, active in every cell of the grid
from the same epoch as the budget, that prevents a neuron from going silent. What remains
for the budget to move is the activity of the neurons that do fire — which is the variable
the question is about.

The design check for this is §4.1: the floor must hold the silent fraction low and roughly
flat while the budget moves activity.

### 1.4 Hypothesis

> **H1 (activity hypothesis)** — with neuron silence prevented, a decrease in spikes per
> active neuron **increases** the network's use of hidden spike timing, and therefore
> **increases** the accuracy a timing perturbation destroys.

There is one hypothesis, and it makes a directional prediction about a difference of two
accuracies. Everything in §4 tests that prediction.

### 1.5 Why the constraint acts on both layers

A penalty applied to one hidden layer constrains a *layer*, not a *network*. The
unconstrained layer remains free to compensate — to become busier as its neighbour is
compressed — so the network-wide manipulation actually achieved may be far smaller than
the per-layer manipulation apparently applied. Charging the budget at both layers removes
the free compensator, at the cost that no coefficient here can be attributed to one layer
(§6.1).

---

## 2. Notation

| Term | Meaning |
|---|---|
| **arm** | one of two network variants. The **delay** variant has learnable axonal delays — small trainable time-shifts on each neuron's output. The **no-delay** variant does not. Delays are themselves a timing mechanism, so running both variants converts a confound into a measurement. |
| **`a`** | spikes per **active** neuron per trial. "How busy are the neurons that fire." **This is the manipulated variable.** |
| **`s`** | silent fraction — the fraction of (trial, neuron) pairs firing nothing. Held low by the floor; reported as a design check, not as a factor. |
| **`a_net` / `s_net`** | the same two quantities pooled over both hidden layers. Pooling is exact over all (trial, neuron) pairs; it is not an average of two ratios. |
| **ceiling** | the training penalty that sets `a`. Charges a cost for every spike above a budget `k`. |
| **floor** | the training penalty that prevents silence. **On in every cell.** |
| **`k`** | the ceiling's spike budget per (trial, neuron) pair. Lower `k` = harder compression. |
| **relocation / jitter / shift** | three **timing** perturbations, evaluation-only. Each moves spikes in time and holds every neuron's spike count *exactly* fixed. |
| **deletion** | the **control** perturbation. Removes spikes at random: destroys rate, not timing. Required, because a network that resists timing damage may simply resist *all* damage. |
| **clean accuracy** | test accuracy of the frozen network with no perturbation. |
| **drop** | `clean accuracy − perturbed accuracy`, in accuracy points. **The dependent measure.** H1 predicts a *larger* drop at *lower* `a`. |
| **injection site** | where the damage is applied at evaluation: `l1` (1st hidden layer), `l2` (2nd), or `both`. |
| **β** | the OLS coefficient on `log(a_net)` when predicting a drop, n = 12 per arm. **H1 predicts β < 0.** |
| **β\*** | the standardised form of that coefficient. |

---

## 3. Methods

### 3.1 Task and network

| | |
|---|---|
| **Dataset** | SHD (Spiking Heidelberg Digits) — spoken digits 0–9 in English and German, converted to spike trains by a simulated cochlea. **20 classes**, so chance accuracy is **5%**. |
| **Input** | 700 input channels × 200 time bins of 1 ms. The raw data occupies 100 bins; zero-padding gives the simulator's 200. This matters for the perturbation window (§3.5). |
| **Splits** | train 0–60%, validation 60–75%, test 75–90% of the file. Every number in this document is from the **test set**. |
| **Network** | 700 → **128** → **128** → 20. Two hidden layers of spiking neurons (SLAYER `SRMALPHA`, threshold θ = 10) and one output neuron per class. |
| **Delays** | in the delay arm, `delay1` sits **between** the two hidden layers and `delay2` between the 2nd hidden layer and the output. Both are learnable and clamped to 0–64 bins. |
| **Loss** | `NumSpikes` — the correct class's output neuron must fire 40 spikes, the others 4. This holds the *output* firing rate constant, so the intervention stays confined to the hidden layers. |
| **Optimiser** | Nadam, lr 0.1, step decay ×0.1 at epoch 300, batch 128. |
| **Training length** | **1250 epochs**, early-stopping patience 300. This is deliberately long: models are still improving well past the point where a short run would stop, so short runs are a *different regime*, not a scaled-down version of this one. |

### 3.2 The two penalties

Both are added to the task loss and both are charged **once per layer and summed**:

```python
loss = task_loss
for layer in (1, 2):
    loss += 10.0 * relu(count[layer] - k[layer]).mean()          # ceiling -> sets a
    loss += 1.0 * relu(11 - peak_potential[layer]).mean()        # floor   -> prevents silence
```

**Ceiling (sets `a`).** `relu(count − k)` is charged **per (trial, neuron) pair**, not on
the batch average. The per-pair form is essential: under a batch-averaged penalty a neuron
can satisfy its budget by going *silent* on some trials and firing freely on others —
which is exactly the escape route the floor exists to close.

**Floor (prevents silence).** The floor charges the **membrane potential**, not the spike
count: `relu(11 − max_t u(t))`. Two reasons, both verified by measurement. First, a neuron
that reaches threshold even once must emit at least one spike, so this penalty is
sufficient to prevent silence, and because `relu` zeroes out above 11 it acts only on
neurons about to fall silent — it applies no pressure to fire *more*, and so does not
fight the ceiling. Second, the gradient of the potential with respect to the weights is an
ordinary convolution gradient: **no surrogate gradient is involved**, so it still carries
signal when a neuron sits far below threshold. A count-based floor must route its gradient
through the surrogate, which carries no signal at that depth.

**Each layer is charged at the full coefficient, not half**, so the pressure applied to
each layer matches what a single-layer grid would apply to it.

### 3.3 Checkpoint selection and training guards

- **`WARMUP_EPOCHS = 20`.** Both penalties switch on at epoch 20, after the task loss has
  established a working solution.
- **`SETTLE_EPOCHS = 150`, delay arm only.** The saved checkpoint is the best **task**
  validation loss — selection is on the task, not on the quality of the constraint. In
  several delay cells, however, the global best task validation loss falls at epochs
  42–44, i.e. roughly twenty epochs after the penalties switch on, and never improves
  again; selecting there would report sparsity statistics from a network that had barely
  carried the constraint, biased in the direction of "the penalty does not bind". The
  settle rule restricts selection to epochs after `warmup + settle`. It binds in **3 of
  the 12 delay cells** (`k=1` seed 44, `k=2` seeds 43 and 44), moving their selection to
  epochs 1197, 716 and 567 and their over-budget fraction from about 50–61% down to
  23–38%.
- **The no-delay arm needs no settle rule** — its earliest selection is epoch 617 of 917 —
  so the two arms carry different selection policies by design.
- **After selection, all 24 cells are clean.** Each selects at 0.65–1.00 of its run length,
  and the constraint state does not change between the selected epoch and the final epoch.

### 3.4 The grid

| control | levels | what it moves |
|---|---|---|
| `k` (ceiling budget) | **1, 2, 4, 8** at layer 1 | the `a` axis — the axis of H1 |
| floor strength | **1 (on) in every cell** | not a factor; a fixed feature of the design |
| seed | **42, 43, 44** | replication |
| arm | delay, no-delay | run as two separate grids |

**12 models per arm, 24 in total**, all at 1250 epochs on a remote server.

**The budget is scaled per layer.** `CEILING_K` gives layer 1's levels; layer 2 receives
`CEILING_K_LAYER2_RATIO` times those levels, so one control produces an equal *relative*
compression at both layers rather than an equal absolute budget:

| arm | natural `a` (L1 / L2) | ratio | `k1` | `k2` |
|---|---|---|---|---|
| no-delay | 3.06–11.49 / 7.49–12.46 | **1.0** (inactive — the ranges overlap) | 1, 2, 4, 8 | 1, 2, 4, 8 |
| delay | 3.82–15.47 / **20.71–30.44** | **2.0** | 1, 2, 4, 8 | **2, 4, 8, 16** |

Layer 2 of the delay arm naturally fires about twice as fast as its layer 1; under a
shared budget the same cell would apply a mild compression to one layer and a severe one
to the other.

### 3.5 Dependent measure, perturbations, windows and sites

**The measure is test accuracy**, and nothing else. Each frozen checkpoint is scored
clean, then re-scored under each perturbation at each of its sweep levels. The quantity
tested against H1 is the **drop**, `clean − perturbed`, in accuracy points, at the
strongest level of each sweep. Raw perturbed accuracy is reported alongside it, because
the two answer different questions when clean accuracy itself varies across the grid
(§4.3, §5.2).

**Four sweeps.** Relocation (every hidden spike moves to a random bin, `f` = 0 … 1),
jitter (Gaussian time-shift per spike, σ = 0 … 25 bins) and shift (rigid translation of
the whole train, σ = 0 … 25 bins) are the **timing** perturbations — all three preserve
every neuron's spike count exactly. Deletion (each spike dropped with probability
`p_d` = 0 … 0.8) is the **rate control**: it destroys count and not placement. The control
is not optional. A pure rate insult and a pure timing insult can track the same underlying
quantity, in which case an accuracy drop measured without it is partly a measure of
general fragility.

**The perturbation window required calibration.** The data occupies 100 of the 200
simulated bins, and layer-1 hidden spikes were measured to stop at bin 87. Relocating
spikes uniformly across all 200 bins would send about 57% of them into a region where no
hidden spike naturally occurs, cutting population spike density about 5× — adding a *rate*
insult to a probe that must touch only timing. Relocation and jitter therefore draw
destinations from each layer's measured support: `[0,88)` for layer 1 in both arms,
`[0,90)` for layer 2 in the no-delay arm, and **`[0,160)`** for layer 2 in the delay arm,
because `delay1` displaces layer-1 spikes by up to 64 bins before layer 2 receives them.

**Shift and deletion are left uncorrected, deliberately.** Clipping a rigid translation
piles spikes up at the edge where they merge, destroying per-neuron counts — the
correction would introduce the artifact it is meant to remove.

**Three injection sites.** Every checkpoint was swept at `l1`, `l2` and `both`. Damaging
both layers at once is a *different and harder* insult than damaging one: at the `both`
site, layer 2 is damaged after having already been computed from a damaged layer 1. The
three sites are reported separately and never collapsed into a single number.

### 3.6 Design validation

**Implementation checks.** Twenty automated checks per arm, on GPU, without training; all
passed. Two could not be settled by reading the code:

- **Layer 1 receives two compressions.** Layer 2's ceiling also propagates gradient back
  into `fc1`; |grad| measured at 0.81 (no-delay) and 2.05 (delay). The effect is real and
  now quantified, and it argues for a *lower* rather than higher ceiling strength.
- **A dead gradient path exists and is reachable.** Layer 2's floor needs no surrogate,
  but its gradient is **exactly zero** on a trial where the whole of layer 1 is silent.
  Forcing that condition returned 0.000000. This is why validation probes survival at the
  hardest corner first.

**The floor earns its place.** Trained without it at the same budgets, the same
architecture reaches a pooled silent fraction of **24.9–57.3%** (delay) and **49.2–69.9%**
(no-delay), against **3.4–16.4%** and **4.1–18.1%** with the floor on. Without the floor,
most of what the budget achieves is neurons switching off, not neurons firing less.

**Pilot.** Models at the hardest and softest `k`, one seed, 400 epochs, were trained before
committing ~40 h of server time, and confirmed that both layers survive the hardest corner
(`k=1`), that firing does not exceed each layer's natural rate, and that clean accuracy
stays well above chance.

---

## 4. Results

### 4.1 Manipulation check

Cell means over 3 seeds. `a` is spikes per active neuron, `s` the silent fraction,
`over-k` the fraction of (trial, neuron) pairs still above budget.

**Delay arm**

| `k` | `a_net` | `s_net` | `a` L1 / L2 | `s` L1 / L2 | over-`k` L1 / L2 | clean acc |
|---|---|---|---|---|---|---|
| 1 | 2.93 | .162 | 2.46 / 3.33 | .231 / .093 | 37.1% / 50.6% | .783 (sd .023) |
| 2 | 3.32 | .112 | 2.58 / 3.98 | .170 / .054 | 23.4% / 31.5% | .782 (sd .009) |
| 4 | 4.44 | .064 | 3.16 / 5.61 | .106 / .021 | 13.9% / 15.5% | .807 (sd .007) |
| 8 | 6.59 | .035 | 4.47 / 8.61 | .060 / .010 | 9.2% / 5.7% | .840 (sd .004) |

**No-delay arm**

| `k` | `a_net` | `s_net` | `a` L1 / L2 | `s` L1 / L2 | over-`k` L1 / L2 | clean acc |
|---|---|---|---|---|---|---|
| 1 | 2.16 | .174 | 1.79 / 2.49 | .223 / .125 | 33.3% / 59.2% | .525 (sd .017) |
| 2 | 2.46 | .116 | 2.04 / 2.86 | .148 / .083 | 18.6% / 43.2% | .535 (sd .003) |
| 4 | 3.17 | .072 | 2.69 / 3.62 | .096 / .047 | 10.4% / 25.8% | .572 (sd .013) |
| 8 | 4.49 | .044 | 3.84 / 5.10 | .066 / .021 | 5.3% / 11.5% | .568 (sd .011) |

Four readings:

1. **The activity axis is real and monotone.** `a_net` spans **2.32×** (delay, 2.91 → 6.77
   across the 12 models) and **2.11×** (no-delay, 2.15 → 4.53). All eight per-layer curves
   — 2 arms × 2 layers — increase monotonically with the budget.
2. **The floor held.** No model in either arm exceeds **18.1%** silence, and no *layer* of
   any cell exceeds 23.6%. Compare 57.3% / 69.9% without it (§3.6).
3. **The remaining silence is not flat.** `s_net` still falls from about .17 at `k=1` to
   about .04 at `k=8`, correlating with `log(a_net)` at ρ = −0.945 (delay) and −0.935
   (no-delay). The floor suppressed the escape route; it did not eliminate it. This is a
   real limitation on the interpretation and is carried into §6.3.
4. **The ceiling leaks, more at layer 2 than layer 1.** At the tightest row, 51% (delay)
   and 59% (no-delay) of layer-2 pairs remain above budget. The manipulation is a genuine
   ~2× range in achieved activity, not a hard cap at `k`.

### 4.2 Clean accuracy

Tighter budgets cost accuracy, in both arms.

| arm | `k=1` | `k=2` | `k=4` | `k=8` | slope on `log(a_net)` |
|---|---|---|---|---|---|
| delay | .783 | .782 | .807 | **.840** | **+0.075** [+0.050, +0.100], p < .001, β\* = +0.90 |
| no-delay | .525 | .535 | **.572** | .568 | **+0.063** [+0.027, +0.099], p = .003, β\* = +0.78 |

The delay arm is monotone apart from a tie between `k=1` and `k=2`; the no-delay arm peaks
at `k=4`. Across the 24 models the cost of the tightest budget is about **6 accuracy
points** in the delay arm and **4** in the no-delay arm. This is context for everything
below: the sparse networks are not equivalent networks that merely fire less, they are
*worse* networks.

### 4.3 Perturbation sweeps

Accuracy at the strongest level of each sweep — relocation `f = 1`, jitter σ = 25, shift
σ = 25, deletion `p_d` = 0.8 — with the drop from that cell's own clean accuracy in
brackets. Cell means over 3 seeds. Chance is .05.

**Delay arm** (clean: .783 / .782 / .807 / .840)

| perturbation | site | `k=1` | `k=2` | `k=4` | `k=8` |
|---|---|---|---|---|---|
| relocation | `l1` | .338 (−.445) | .336 (−.446) | .354 (−.453) | .370 (−.470) |
| relocation | `l2` | .715 (−.068) | .690 (−.093) | .683 (−.124) | .668 (−.172) |
| relocation | `both` | .367 (−.416) | .334 (−.449) | .358 (−.449) | .380 (−.460) |
| jitter | `both` | .361 (−.422) | .356 (−.426) | .385 (−.422) | .397 (−.443) |
| shift | `both` | .221 (−.562) | .218 (−.565) | .200 (−.608) | .219 (−.621) |
| **deletion (control)** | `both` | .143 (−.640) | .145 (−.638) | .138 (−.669) | .161 (−.679) |

**No-delay arm** (clean: .525 / .535 / .572 / .568)

| perturbation | site | `k=1` | `k=2` | `k=4` | `k=8` |
|---|---|---|---|---|---|
| relocation | `l1` | .352 (−.173) | .360 (−.175) | .375 (−.197) | .359 (−.210) |
| relocation | `l2` | .524 (−.001) | .524 (−.011) | .559 (−.012) | .541 (−.028) |
| relocation | `both` | .372 (−.153) | .377 (−.158) | .385 (−.187) | .378 (−.191) |
| jitter | `both` | .371 (−.154) | .371 (−.163) | .380 (−.192) | .377 (−.191) |
| shift | `both` | .242 (−.283) | .225 (−.310) | .231 (−.341) | .208 (−.361) |
| **deletion (control)** | `both` | .139 (−.386) | .139 (−.396) | .135 (−.437) | .130 (−.438) |

Three structural facts, before the hypothesis test:

1. **Layer 2 is a much cheaper place to take damage.** Relocating every spike in layer 2
   costs the no-delay arm between .001 and .028 accuracy. The same insult at layer 1 costs
   it .173–.210. These sweeps cannot say why: layer 2 may carry less of the timing code,
   or damage there may simply have one fewer layer to propagate through.
2. **The `both` site is not a uniform escalation of `l1` for the count-preserving
   perturbations.** Relocation and jitter **saturate** — adding the layer-2 insult to the
   layer-1 insult changes accuracy by a point or two. Both move spikes within a window
   while preserving every count, so once layer 1's placement code is gone, a second pass
   finds little left to remove. Deletion, the one perturbation that does not preserve
   counts, is the one that accumulates.
3. **The drop grows with the budget, in every row of both tables.** That is the sign
   opposite to H1's prediction, and §4.4 tests it.

### 4.4 The H1 test

OLS of `drop ~ log(a_net)` across the 12 models of each arm. **H1 predicts β < 0** — fewer
spikes per active neuron, more timing use, a larger drop.

| arm | site | relocation | jitter | shift | **deletion (control)** |
|---|---|---|---|---|---|
| delay | `l1` | **+.032** p=.002 | **+.040** p<.001 | **+.049** p=.010 | −.010 p=.61 |
| delay | `l2` | **+.123** p<.001 | **+.081** p=.001 | **+.180** p<.001 | +.015 p=.31 |
| delay | `both` | **+.042** p=.040 | +.023 p=.15 | **+.080** p<.001 | **+.055** p<.001 |
| no-delay | `l1` | **+.055** p=.003 | **+.058** p=.004 | **+.098** p<.001 | **+.064** p=.004 |
| no-delay | `l2` | **+.033** p=.010 | **+.034** p=.003 | **+.103** p<.001 | **+.043** p=.006 |
| no-delay | `both` | **+.056** p=.004 | **+.054** p=.006 | **+.102** p<.001 | **+.075** p=.001 |

**Every one of the 18 timing coefficients is positive, and 17 of the 18 are significant.**
H1 requires all 18 to be negative. Standardised, they run from +0.44 to +0.98. The single
non-significant cell (jitter at `both`, delay arm, β\* = +0.44) is not a case *for* H1; it
is simply a weaker instance of the same reversal.

**The control changes how much of even that reversal is about timing.** Deletion destroys
rate and not placement, and its coefficient is positive and significant in 4 of 6 site ×
arm cells — most of the no-delay arm and the `both` site of the delay arm. Re-running each
timing regression with the deletion drop as a covariate:

| arm | site | relocation | jitter | shift |
|---|---|---|---|---|
| delay | `l1` | **+.033** p=.002 | **+.041** p<.001 | **+.054** p=.001 |
| delay | `l2` | **+.113** p<.001 | **+.066** p=.001 | **+.177** p<.001 |
| delay | `both` | +.046 p=.27 | +.024 p=.48 | +.028 p=.31 |
| no-delay | `l1` | +.036 p=.13 | +.022 p=.30 | **+.067** p=.003 |
| no-delay | `l2` | +.025 p=.15 | +.027 p=.084 | **+.095** p<.001 |
| no-delay | `both` | +.020 p=.43 | +.006 p=.82 | **+.063** p=.009 |

Nine of the eighteen survive the control. **All nine still carry the wrong sign for H1.**
Controlling for general fragility removes about half of the reversal and reverses none of
it.

**Verdict: H1 is not supported at any injection site in either arm, under any of the three
timing perturbations, with or without the rate control.** The measured effect is
consistently in the opposite direction: within this grid, the *denser* networks lose more
accuracy to a timing perturbation than the sparser ones.

### 4.5 Which arm is primary

Both arms give the same answer, so no arbitration is needed for the conclusion. Where the
two differ is dynamic range: the delay arm reaches .803 mean clean accuracy against the
no-delay arm's .550, so it has roughly 1.6× more accuracy between clean and chance for a
perturbation to remove. The delay arm is reported as primary on that basis, and the
no-delay arm is retained as the check that any effect does not depend on the learnable
delays — a timing mechanism the network could otherwise use in place of spike placement.

---

## 5. Discussion

**1. H1 received no support, and the data are not merely null.** With silence suppressed
by the floor and general fragility controlled by the deletion sweep, the coefficient on
activity is *positive* at every site in both arms. Compressing a network's spike budget
did not make it rely more on spike timing by this measure; if anything it made the
network's accuracy less dependent on where its spikes sit.

**2. Most of the reversal is general robustness, not a timing story in reverse.** The
deletion control — which destroys count and not placement — moves in the same direction as
the timing perturbations in 4 of 6 site × arm cells, and removes about half of the
apparent effect when entered as a covariate. Sparser networks are more robust to *damage*,
not specifically to *timing damage*. Any reading of a perturbation sweep that omits a rate
control will mistake the first for the second.

**3. Read the drop and the raw accuracy together.** At the `l1` and `both` sites, raw
post-perturbation accuracy barely moves along the budget axis at all: in the no-delay arm,
relocation at `both` leaves .372 → .378 across a 2.11× activity range (ρ = +0.14, p = .67).
What changes across that axis is the *clean* accuracy, which the budget costs (§4.2).
Mechanically, then, a large part of the drop trend is the clean-accuracy trend: sparse
networks "lose less" partly because they had less to lose. This is exactly the situation a
chance-corrected measure exists to handle, and it is the main price of an accuracy-only
analysis. The conclusion survives it — a measure that removed the clean-accuracy
difference would move the coefficients toward zero, not across it — but the coefficient
magnitudes in §4.4 should not be read as pure timing quantities.

**4. The two hidden layers are not two copies of the same object.** Layer 2 is far cheaper
to damage, leaks more through its ceiling (up to 59% of pairs above budget against layer
1's 33%), and in the delay arm naturally fires about twice as fast. A constant calibrated
at one layer must never be carried to the other.

**5. What a null H1 does and does not say.** It says that, in this architecture, on this
task, over a 2× range of achieved activity with silence held below 18%, reducing the spike
budget does not increase the readout's measured dependence on spike placement. It does not
say the hidden layers contain no timing information, nor that a wider activity range, a
harder cap, or a task with more temporal structure would give the same answer. It does say
that the intuition in §1.1 — fewer spikes per neuron, therefore more timing — does not hold
automatically, and that a sparsity intervention needs its own control before its
perturbation robustness can be read as a statement about timing.

---

## 6. Limitations

1. **No attribution to a specific layer.** The budget moves both layers together, so the
   two layers' activity is correlated at +0.89 to +0.97 **by construction**. Every
   coefficient here is a network coefficient.
2. **The activity axis is soft, not pinned.** The ceiling leaks — up to 59% of pairs stay
   above budget at the tightest row at layer 2. This is a real manipulated range of about
   2×, not a hard constraint, and layer 2's axis is the softer of the two.
3. **Silence is suppressed, not held constant.** `s_net` still falls from about .17 to .04
   along the budget axis (ρ ≈ −0.94 with `log a_net`). Under H1's own logic the tightest
   cells therefore retain a small amount of the identity-code advantage that the floor was
   meant to remove — which biases the test *toward* H1's prediction, not against it. The
   observed coefficients run the other way regardless.
4. **Clean accuracy is not constant along the axis.** A tight budget costs 4–6 accuracy
   points, so the drop measure mixes "how much timing the readout used" with "how much
   accuracy there was to lose" (§5, point 3).
5. **Perturbation magnitude is not equal across sparsity levels.** Moving a given fraction
   of many spikes disturbs a downstream neuron more than moving the same fraction of few
   spikes. The deletion control reduces this problem without removing it.
6. **No non-sparse reference curve exists for the `both` injection site.** The available
   unconstrained-baseline sweeps all perturbed one layer at a time, so the `both`-site
   figures carry no reference line.
7. **n = 12 per arm.** Four budget levels × three seeds. The effects reported are large and
   consistent in sign across 18 regressions, but the grid is small, and single
   non-significant cells should not be interpreted individually.

---

## 7. Data, code, and reproduction

**Training scripts**
[sn_bothLayer_train_noDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_noDelay_v3.py) ·
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py)

**Checkpoints and training logs** —
`sn_data/sparse_whole_{arm}_v3L12_k{k}_floor1_seed{seed}.pt`,
`sn_log/sparse_whole_{arm}_v3L12_train_summary.json`, plus the per-epoch logs.

> **Key convention in the summary files.** The unsuffixed canonical keys
> (`spikes_per_active_neuron`, `silent_fraction`, `spikes_per_neuron`) hold the **pooled
> network** values, not layer-1 values. Per-layer values carry an explicit `_l1` or `_l2`
> suffix.

**Perturbation sweeps** — the `*_bothLayer_evalOnly_*_v3.py` scripts under
[shd/](../../exp_sparse_network/shd/) (relocation),
[jitter/](../../exp_sparse_network/jitter/),
[shift/](../../exp_sparse_network/shift/) and
[deletion/](../../exp_sparse_network/deletion/), writing
`{folder}/log/sparse_whole_{arm}_v3L12_*_eval.json`. Every accuracy in §4.3 and every
regression in §4.4 comes from the `per_checkpoint` block of those four files, restricted
to the `floor1` cells.

**Figures** —
[result_visualization/bothLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/bothLayer/results_visualization.ipynb)
holds the accuracy curves, one section per injection site plus the three sites side by
side.

### Reproducing the experiment

```
# training -- 12 floor-on models per arm, 1250 epochs
python sn_bothLayer_train_noDelay_v3.py
python sn_bothLayer_train_withDelay_v3.py

# the four sweeps, per arm, at sites l1 / l2 / both
python shd/shd_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python jitter/jitter_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python shift/shift_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python deletion/deletion_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
```

Every sweep is **evaluation-only**: the scripts load frozen checkpoints, take no
gradients, and apply perturbation only at evaluation.
