# Sparsity and the use of spike timing in a two-hidden-layer spiking network

**The both-layer factorial (`v3L12`). Status: complete (2026-08-09).** 48 networks, eight
perturbation sweeps at three injection sites, three dependent measures, and a
coefficient comparison against the two single-layer factorials.

**Companion grids:** [layer 1 only (`v3`)](../legacy/sn_progress/sparse_network_test_progress_v3.md) ·
[layer 2 only (`v3L2`)](../legacy/sn_progress/sparse_network_2ndLayer_test_progress_v3.md) ·
[execution log for this grid](sparse_network_bothLayer_test_progress.md)

---

## Abstract

A spiking network trained with a sparsity penalty fires fewer hidden spikes. Does it
compensate by making more use of *when* those spikes occur? The question is not directly
testable, because "sparsity" is not one variable. It factorises into the activity of the
neurons that fire and the fraction of neurons that fire nothing at all, and those two
components act on different coding channels. This experiment manipulates them
independently and simultaneously on **both** hidden layers, in a 4 (spike budget) × 2
(silence floor) × 3 (seed) factorial, run separately for a network with learnable axonal
delays and one without: 24 models per arm, 48 in total, 1250 epochs each.

Four results:

1. **The manipulation separated the two variables across the full network.** Their
   correlation is −0.199 (no-delay) and +0.040 (delay), against an acceptance criterion
   of |ρ| ≤ 0.5, and both hidden layers pass individually as well as pooled. The
   selectivity of the two layers moves together at +0.977 / +0.944, so the pooled axis
   is a statement about the network rather than about one layer of it.
2. **The activity hypothesis (H1) received no support.** H1 requires a negative
   coefficient on activity. That coefficient is statistically indistinguishable from
   zero at every injection site in both arms. The single coefficient in the table that
   *is* significant carries the wrong sign for H1.
3. **The identity-code account (H2) is supported.** What predicts a network's measured
   use of spike timing is not how many spikes its neurons fire but how often they fall
   completely silent. A matched, directly manipulated contrast confirms this in both
   arms (+0.100, p = .0001; +0.068, p = .011).
4. **The activity effect is a property of single-layer constraints, not of sparsity.**
   The coefficient on activity is significant in **3 of 4** cells where exactly one
   layer is constrained and in **0 of 2** cells where the whole network is constrained —
   and not because the network-wide manipulation is weaker, since the achieved axes are
   the same size.

---

## 1. Background and hypotheses

### 1.1 The question

> *Induce different levels of sparsity in the hidden layers through regularisation, then
> observe whether sparser-activity networks do more temporal processing than denser
> ones.*

The intuition is straightforward. A neuron that fires 10 spikes can carry information in
*how many* spikes it fires — a rate code. A neuron restricted to 0 or 1 spikes cannot.
The only degree of freedom left to it is *when* the single spike occurs — a timing code.
A smaller spike budget should therefore push information out of the rate channel and
into the timing channel.

The measurement instrument is a **fixed-weight perturbation sweep**. Train on clean data,
freeze the network, then damage the hidden spikes at evaluation. The damage changes when
spikes occur while holding each neuron's spike count exactly fixed. If accuracy falls,
the readout was using timing. If accuracy holds, the readout was using something the
damage did not touch.

> **Protocol rule.** Sparsity is induced **during training**, on clean data.
> Perturbation is applied **only at evaluation**, to an already-frozen network. Training
> under the perturbation lets the network learn a route around the damage, which erases
> the very effect being measured.

### 1.2 Three channels in a hidden layer

A hidden layer can carry class information in three places, and a timing perturbation
destroys only one of them:

| Channel | What it is | Survives a timing-only perturbation? |
|---|---|---|
| **Count** | how many spikes each neuron fires | **yes, exactly** |
| **Identity** | *which* neurons fire | **yes, exactly** |
| **Timing** | *when* the spikes occur | **no — this is the channel the perturbation destroys** |

Two consequences follow, and together they are the reason this experiment has the design
it has.

First, **a sparse layer does not close the count channel.** Count resolution is a
property of the population, not of the individual neuron. One bit per neuron across 128
neurons is 128 bits; a 20-class problem needs far less. Compressing each neuron's spike
count therefore need not force information out of the rate channel at all.

Second, and more consequentially, **sparsity is often achieved by silence.** Neurons go
quiet for some stimuli and not for others, which builds a stimulus-selective *identity*
code. Such a labelled-line code is left fully intact by a timing perturbation, and it
grows *richer* as sparsity increases. A sparsity intervention can therefore move a
network *toward* a code the perturbation cannot touch, and a naive experiment would
score that as "robust to timing damage" and read it as evidence about timing.

### 1.3 Two variables, not one

The whole design follows from one identity:

```
spikes per neuron  =  (1 − silent fraction)  ×  spikes per ACTIVE neuron
```

The two factors on the right are distinct quantities acting on distinct channels:

| Symbol | Name | Plain description | Effect on the code |
|---|---|---|---|
| **`a`** | temporal sparsity | how many spikes a neuron fires *on the trials where it fires at all* | fewer spikes per neuron ⇒ less per-neuron count resolution ⇒ pressure toward a latency or timing code. **This is the variable the question is about.** |
| **`s`** | selectivity | the fraction of (trial, neuron) pairs firing nothing | neurons silent for some stimuli ⇒ a labelled-line **identity** code, which timing perturbations leave complete. **The question says nothing about this variable.** |

Any single rate penalty moves both at once. When `a` and `s` are collinear, no analysis
can attribute an effect to either. A design that moves them independently is therefore
not a refinement of the experiment — it is the precondition for the experiment meaning
anything.

### 1.4 Hypotheses

> **H1 (activity hypothesis)** — with selectivity `s` held constant, a decrease in
> spikes per active neuron `a` **increases** the network's use of hidden spike timing.
> *Predicts a negative coefficient on `a`.*
>
> **H2 (identity-code hypothesis)** — the variable governing measured timing use is `s`,
> not `a`. Silence builds an identity code the perturbation cannot touch, which is why
> sparse networks appear robust to timing damage. *Predicts a negative coefficient on
> `s`, and that removing silence increases measured timing use.*

The two make **opposite and separable** predictions in a design where `a` and `s` move
independently. The analysis scripts label the first hypothesis `H1'`.

### 1.5 Why the constraint must act on both layers

A penalty applied to one hidden layer constrains a *layer*, not a *network*. The
unconstrained layer remains free to compensate — to become busier or less selective as
its neighbour is compressed. A coefficient estimated on such a grid is estimated on a
network that was compressed in one place and possibly expanded in another, and the
network-wide manipulation it actually achieved may be much smaller than the per-layer
manipulation it appears to have applied.

Constraining both layers removes the free compensator. It also completes a family: with
the same factorial run at layer 1 alone, at layer 2 alone, and at both layers, the key
coefficient can be compared across three grids that differ only in *where* the
constraint acts. A null everywhere would be a fact about the architecture; an effect
only under the network-wide constraint would be a result no single-layer grid can
deliver. §4.6 reports which of these the data gave.

This grid is a replication in a new location, not a new hypothesis. H1 and H2 are
unchanged.

---

## 2. Notation

Everything needed to read §4.

| Term | Meaning |
|---|---|
| **arm** | one of two network variants. The **delay** variant has learnable axonal delays — small trainable time-shifts on each neuron's output. The **no-delay** variant does not. Delays are themselves a timing mechanism, so running both variants converts a confound into a measurement. |
| **`a`** | spikes per **active** neuron per trial. "How busy are the neurons that fire." |
| **`s`** | silent fraction — the fraction of (trial, neuron) pairs firing nothing. "How often does a neuron stay quiet." |
| **`a_net` / `s_net`** | the same two quantities pooled over both hidden layers. Pooling is exact over all (trial, neuron) pairs; it is not an average of two ratios. |
| **ceiling** | the training penalty that sets `a`. Charges a cost for every spike above a budget `k`. |
| **floor** | the training penalty that sets `s`. Prevents a neuron from going silent. Strength 0 (off) or 1 (on). |
| **`k`** | the ceiling's spike budget per (trial, neuron) pair. Lower `k` = harder compression. |
| **relocation / jitter / shift** | three **timing** perturbations, evaluation-only. Each moves spikes in time and holds every neuron's spike count *exactly* fixed. |
| **deletion** | the **control** perturbation. Removes spikes at random: destroys rate, not timing. Required, because a network that resists timing damage may simply resist *all* damage. |
| **retention** | the chance-corrected fraction of accuracy surviving a perturbation: `(acc_perturbed − .05) / (acc_clean − .05)`. |
| **usage** | `1 − retention` under the strongest timing perturbation. **"How much the network's readout uses spike timing."** Higher = more use. |
| **control** | the same score under 80% spike deletion. **"How fragile is the network to damage in general."** |
| **availability** | **"How much timing information the hidden layer contains"**, whether the network uses it or not. Measured by decoding, without perturbation (§3.5). |
| **injection site** | where the damage is applied at evaluation: `l1` (1st hidden layer), `l2` (2nd), or `both`. |
| **β_a, β_s** | regression coefficients on `log(a)` and `s` when predicting usage. H1 predicts β_a < 0; H2 predicts β_s < 0. |
| **β\*** | the standardised form of a coefficient, comparable across variables with different scales. |
| **ρ** | Pearson correlation across the models of a grid (n = 24 per arm here), unless stated as Spearman. |

---

## 3. Methods

### 3.1 Task and network

| | |
|---|---|
| **Dataset** | SHD (Spiking Heidelberg Digits) — spoken digits 0–9 in English and German, converted to spike trains by a simulated cochlea. **20 classes**, so chance accuracy is **5%**. |
| **Input** | 700 input channels × 200 time bins of 1 ms. The raw data occupies 100 bins; zero-padding gives the simulator's 200. This matters for the perturbation window (§3.6). |
| **Splits** | train 0–60%, validation 60–75%, test 75–90% of the file. Every number in this document is from the **test set**. |
| **Network** | 700 → **128** → **128** → 20. Two hidden layers of spiking neurons (SLAYER `SRMALPHA`, threshold θ = 10) and one output neuron per class. |
| **Delays** | in the delay arm, `delay1` sits **between** the two hidden layers and `delay2` between the 2nd hidden layer and the output. Both are learnable and clamped to 0–64 bins. |
| **Loss** | `NumSpikes` — the correct class's output neuron must fire 40 spikes, the others 4. This holds the *output* firing rate constant, so the sparsity intervention stays confined to the hidden layers. |
| **Optimiser** | Nadam, lr 0.1, step decay ×0.1 at epoch 300, batch 128. |
| **Training length** | **1250 epochs**, early-stopping patience 300. This is deliberately long: models are still improving well past the point where a short run would stop, so short runs are a *different regime*, not a scaled-down version of this one. |

### 3.2 The two penalties

Both penalties are added to the task loss. In this grid each penalty is **charged once
per layer and summed**:

```python
loss = task_loss
for layer in (1, 2):
    loss += 10.0 * relu(count[layer] - k[layer]).mean()      # ceiling  -> sets a
    loss += floor_strength * relu(11 - peak_potential[layer]).mean()   # floor -> sets s
```

**Ceiling (sets `a`).** `relu(count − k)` is charged **per (trial, neuron) pair**, not on
the batch average. The per-pair form is essential: under a batch-averaged penalty a
neuron can satisfy its budget by going *silent* on some trials and firing freely on
others — which manufactures exactly the `a`–`s` confound the design exists to prevent.

**Floor (sets `s`).** The floor charges the **membrane potential**, not the spike count:
`relu(11 − max_t u(t))`. Two reasons, both verified by measurement. First, a neuron that
reaches threshold even once must emit at least one spike, so this penalty is sufficient
to prevent silence, and because `relu` zeroes out above 11 it acts only on neurons about
to fall silent — it applies no pressure to fire *more*. Second, the gradient of the
potential with respect to the weights is an ordinary convolution gradient: **no surrogate
gradient is involved.** The gradient therefore still carries signal when a neuron sits
far below threshold. Silent neurons here were measured at peak potentials of roughly 0–2
against a threshold of 10 — *deeply* silent. A count-based floor must route its gradient
through the surrogate, which carries no signal at that depth and so cannot revive an
already-dead neuron.

**Each layer is charged at the full coefficient, not half.** This is deliberate: one cell
of this grid then applies to layer 1 exactly the pressure the layer-1 grid applies, and
to layer 2 exactly the pressure the layer-2 grid applies, which is what makes the three
grids comparable cell by cell.

### 3.3 Checkpoint selection and training guards

- **`WARMUP_EPOCHS = 20`.** The ceiling is minimised at *zero spikes*, so in the
  floor-off column only the task loss opposes total silence. Total silence is an
  **absorbing state**: once every neuron sits far below threshold, only the surrogate
  gradient could reactivate the layer, and the surrogate carries no signal there. Without
  warm-up the layer is 100% silent by epoch 5 and never recovers.
- **`SETTLE_EPOCHS = 150`, delay arm only.** The saved checkpoint is the best **task**
  validation loss — selection is on the task, not on the quality of the constraint. In
  the floor-on delay cells, however, the global best task validation loss falls at
  epochs 42–44, i.e. roughly twenty epochs after the penalties switch on, and never
  improves again; selecting there would report sparsity statistics from a network that
  had barely carried the constraint, biased in the direction of "the penalty does not
  bind". The settle rule restricts selection to epochs after `warmup + settle`, keeping
  the criterion on the task loss while refusing checkpoints from before the manipulation
  exists. It binds in **3 of 24 delay cells**, moving their selection to epochs 567, 716
  and 1197 and their over-budget fraction from about 50–61% down to 23–38%.
- **The no-delay arm needs no settle rule** — its earliest selection is epoch 617 of 917
  — so the two arms carry different selection policies by design, and the script records
  the asymmetry with its evidence.
- **After selection, all 48 cells are clean.** Each selects at 0.65–1.00 of its run
  length, and the constraint state does not change between the selected epoch and the
  final epoch.

### 3.4 The grid

| control | levels | what it moves |
|---|---|---|
| `k` (ceiling budget) | **1, 2, 4, 8** at layer 1 | the `a` axis — the axis of H1 |
| floor strength | **0, 1** | the `s` axis — the axis of H2 |
| seed | **42, 43, 44** | replication |
| arm | delay, no-delay | run as two separate grids |

**24 models per arm, 48 in total**, all at 1250 epochs on a remote server.

**The ceiling budget is scaled per layer.** `CEILING_K` gives layer 1's levels; layer 2
receives `CEILING_K_LAYER2_RATIO` times those levels, so one control produces an equal
*relative* compression at both layers rather than an equal absolute budget:

| arm | natural `a` (L1 / L2) | ratio | `k1` | `k2` |
|---|---|---|---|---|
| no-delay | 3.06–11.49 / 7.49–12.46 | **1.0** (inactive — the ranges overlap) | 1, 2, 4, 8 | 1, 2, 4, 8 |
| delay | 3.82–15.47 / **20.71–30.44** | **2.0** | 1, 2, 4, 8 | **2, 4, 8, 16** |

Layer 2 of the delay arm naturally fires about twice as fast as its layer 1; under a
shared budget the same cell would apply a mild compression to one layer and a severe one
to the other.

**All artifacts carry the tag `v3L12`.** The three grids share dataset, arm, `k`, floor
and seed, so distinct tags (`v3`, `v3L2`, `v3L12`) are what keep their filenames
distinct.

### 3.5 Dependent measures

Three quantities per trained model, on the test set, with the network frozen.

**1. Usage — does the readout use spike timing?**
Apply the strongest timing perturbation: relocation at `f = 1`, where every hidden spike
moves to a random bin and every neuron's count is preserved exactly. Measure the accuracy
drop, chance-correct it, and take `usage = 1 − retention`.

**2. Control — is the network fragile to damage in general?**
The same score under 80% random spike **deletion**, which destroys rate and not timing.
This control is not optional and does not belong in a footnote: a pure rate insult and a
pure timing insult can track the same underlying quantity, in which case "timing use"
measured without it is partly a measure of general fragility. It therefore enters the
regression as a covariate. §4.4 and §5.4 show the control earning that place in this
grid's own data.

**3. Availability — does the hidden layer *contain* timing information?**
No perturbation. Four linear decoders (logistic regression) are fitted on the train split
and scored on the test split, all on the same hidden activity:

| view | features | what it destroys |
|---|---|---|
| `COUNT` | each neuron's spike count (128 features) | — |
| `IDENT` | the same features, binarised | — |
| `FULL` | each neuron's counts in 10 time bins (1280 features) | nothing |
| `SHUF` | `FULL`, but each neuron's spikes are redrawn from *its own* average temporal profile, count held exactly fixed | only the within-trial position of the spikes |

`availability = (FULL − SHUF) / (FULL − chance)` — the fraction of decodable class
information that genuinely requires knowing *when* the spikes occurred. `SHUF` is the
correct null: it holds feature count, decoder, sample count, per-neuron spike count and
per-neuron average timing all identical.

**Availability and usage answer different questions and frequently disagree.** A layer can
hold a constant amount of timing information while the readout uses less of it. Both are
always reported.

### 3.6 Perturbation battery, windows, and injection sites

Four sweeps: relocation, jitter and shift (timing, count-preserving) plus deletion
(rate, the control).

**The perturbation window required calibration.** The data occupies 100 of the 200
simulated bins, and layer-1 hidden spikes were measured to stop at bin 87. Relocating
spikes uniformly across all 200 bins would send about 57% of them into a region where no
hidden spike naturally occurs, cutting population spike density about 5× — adding a
*rate* insult to a probe that must touch only timing. Relocation and jitter therefore
draw destinations from each layer's measured support: `[0,88)` for layer 1 in both arms,
`[0,90)` for layer 2 in the no-delay arm, and **`[0,160)`** for layer 2 in the delay arm,
because `delay1` displaces layer-1 spikes by up to 64 bins before layer 2 receives them.

**Shift and deletion are left uncorrected, deliberately.** Clipping a rigid translation
piles spikes up at the edge where they merge, destroying per-neuron counts — the
correction would introduce the artifact it is meant to remove.

**Three injection sites.** Every checkpoint was swept at `l1`, `l2` and `both`. Damaging
both layers at once is a *different and harder* insult than damaging one: at the `both`
site, layer 2 is damaged after having already been computed from a damaged layer 1. The
three sites are therefore reported separately and never collapsed into a single
network-level number. §4.3 shows this separation was necessary.

### 3.7 Design validation

**Implementation checks.** Twenty automated checks per arm, on GPU, without training; all
passed. Two could not be settled by reading the code:

- **Layer 1 receives two compressions.** Layer 2's ceiling also propagates gradient back
  into `fc1`; |grad| measured at 0.81 (no-delay) and 2.05 (delay). The effect is real and
  now quantified, and it argues for a *lower* rather than higher ceiling strength.
- **A dead gradient path exists and is reachable.** Layer 2's floor needs no surrogate,
  but its gradient is **exactly zero** on a trial where the whole of layer 1 is silent.
  Forcing that condition returned 0.000000. This is why validation probes survival at the
  hardest corner first.

**Pilot.** Eight models — hardest and softest `k`, floor off and on, one seed, 400 epochs
(5.9 h) — were trained before committing ~40 h of server time, and assessed against five
criteria:

| # | criterion | result |
|---|---|---|
| 1 | both layers survive the hardest corner (`k=1`, floor off) | **PASS** — accuracy .441 (no-delay) / .717 (delay) against chance .05; neither layer fully silent |
| 2 | `s` approximately constant as `k` varies within a floor column | **partial** — it drifts (carried into §6) |
| 3 | the floor separates `s` by ≥20 points at **both** layers | **PASS by a wide margin** — +52 to +63 points |
| 4 | firing does not exceed each layer's natural rate | **PASS** |
| 5 | clean accuracy well above chance | **PASS** |

The pilot also proved predictive of the full grid, anticipating the achieved floor-on
layer-1 axis of the delay arm at 1.82× and the network axis to within 0.06×.

---

## 4. Results

### 4.1 Manipulation check

Everything else is conditional on this. The design acceptance criterion is
|ρ(`a`, `s`)| ≤ 0.5 — the two variables must be genuinely separable. Pearson, n = 24 per
arm:

| | no-delay | delay |
|---|---|---|
| **ρ(`a_net`, `s_net`)** — the acceptance number | **−0.199** (p=.35) ✔ | **+0.040** (p=.85) ✔ |
| ρ(`a1`, `s1`) — layer 1 alone | **−0.297** ✔ | **−0.001** ✔ |
| ρ(`a2`, `s2`) — layer 2 alone | **−0.122** ✔ | **+0.106** ✔ |
| **ρ(`s1`, `s2`)** — do the layers' silence levels move together? | **+0.977** | **+0.944** |
| ρ(`a1`, `a2`) — not targeted by the design | +0.973 | +0.893 |

*(The regression script reports the same check as a Spearman rank correlation: −0.422,
p = .040 for no-delay; −0.064, p = .77 for delay. Both forms pass.)*

Three readings:

1. **The pooled confound is broken in both arms.** This was the design's primary
   function.
2. **Both layers also pass individually.** This matters more than it looks: the pooled
   decorrelation is therefore not an artifact of averaging two layers that each remain
   confounded internally.
3. **The two layers' selectivity moves together at +0.977 / +0.944.** The floor column is
   thus a selectivity manipulation *for the network*, which makes the pooled `s_net` axis
   a network-level statement rather than a layer-level one. No single-layer grid can
   produce this.

The last row is what this design intentionally does **not** decorrelate. Both controls act
on both layers, so the two layers' activity levels move together **by construction**.
That is a property of the question, not a defect — but it has a consequence: **every
coefficient from this grid is a network-level coefficient and cannot be attributed to a
layer.**

### 4.2 Achieved axes and cost

| | no-delay | delay |
|---|---|---|
| **`a` axis span** (floor off / on, network-wide) | 1.95× / 2.07× | 1.87× / 2.25× |
| **`s` range across the grid** | 4.1% – 69.9% | 3.4% – 57.3% |
| **floor separation in `s`** (at equal `k`, per layer) | +43.8 … +57.6 pts | +27.6 … +45.9 pts |
| **clean accuracy** (cell means) | .480 – .572 | .739 – .840 |
| rate inflation? | none — max 3.59 / 4.99 against a natural 7.85 / 5.53 | none — 4.21 / 8.52 against 11.66 / 23.93 |

- **Floor separation passes in all 16 row × layer checks.** The narrowest margin
  (+27.6 points) still clears the 20-point threshold by 38%.
- **The floor is free, and in fact beneficial.** Floor-on gives better clean accuracy
  than floor-off in all 8 rows of both arms.

Three properties of the achieved manipulation are recorded rather than hidden:

- **`s` is not perfectly flat along the `k` axis.** The floor-off column drifts by about
  30 points at layer 1 in both arms, so the row control moves `s` as well as `a`. This is
  precisely why the design *decorrelates* the two axes; it does not claim they are
  orthogonal by construction.
- **Layer 2's ceiling leaks more than layer 1's** in 15 of 16 cells; at the tightest row
  up to 59% of pairs remain above budget. Layer 2's row axis is the softer of the two.
- **One of twelve activity curves is non-monotone** — the delay arm, floor off, layer 1,
  which dips at `k = 2` and rises again. The pooled axis is monotone, and the pooled axis
  is what this grid manipulates.

### 4.3 Perturbation sweeps by injection site

Mean chance-corrected **retention** at the strongest setting of each sweep, averaged over
the 8 cells of each arm. Higher = less damage.

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

1. **Layer 2 is a much cheaper place to take damage** — for every perturbation, in both
   arms, by a wide margin. These sweeps cannot identify the cause: layer 2 may carry less
   of the timing code, or damage there may simply have one fewer layer to propagate
   through.
2. **The `both` site is not a uniform escalation of `l1`.** Relocation and jitter
   **saturate**: adding the layer-2 insult to the layer-1 insult changes nothing
   measurable. Both perturbations move spikes *within* a window while preserving every
   count, so once layer 1's placement code is gone, a second pass finds almost nothing
   left to remove.
3. **Deletion is the perturbation that accumulates** (−.223 / −.154), and it is also the
   only one that does not preserve counts — exactly the ordering an accumulation argument
   predicts, and the empirical reason to keep the three sites separate.

### 4.4 Endpoint measures

Means over the 24 models of each arm; `a` and `s` are the network axes.

| | no-delay | delay |
|---|---|---|
| **availability** (timing fraction) at `l1` / `l2` / `net` | .270 / .234 / **.224** | .233 / .079 / **.097** |
| **usage** at `l1` / `l2` / `both` | .315 / .016 / **.296** | .565 / .171 / **.551** |
| **control** (deletion) at `l1` / `l2` / `both` | .594 / .487 / **.819** | .704 / .435 / **.861** |
| **readout gap** (decoder accuracy − network accuracy) at `net` | **+0.285** | **+0.076** |

- The site structure of §4.3 reappears here: **usage saturates** (`both` ≈ `l1`) while
  **control accumulates** (.594 → .819; .704 → .861). Two independently written scripts
  agree, which is the check that they use the same injection sites and windows.
- **Layer 2 is both the cheaper place to take damage and the poorer place to read timing
  from.** In the delay arm its availability is .079 against layer 1's .233, and its usage
  .171 against .565. The two measures did not have to agree, but they do.
- The **readout gap** is how much class information a simple linear decoder can extract
  from the hidden network that the network's own output layers do not use. At +0.285 in
  the no-delay arm, that arm's perturbation scores are as much a statement about its
  readout as about its hidden code (§4.7).

### 4.5 Regressions

The model is `usage ~ log(a_net) + s_net + control`, n = 24 per arm. H1 requires
**β_a < 0**; H2 requires **β_s < 0**.

| | no-delay | delay |
|---|---|---|
| design acceptance check | PASS | PASS |
| **β_a** at `l1` / `l2` / `both` | +.025 / −.001 / **−.0005** (p=.99) | −.010 / **+.137\*** / **+.021** (p=.32) |
| **β_s** at `both` | **−0.168** (p=.049), β\* = −0.66 | −0.089 (p=.12), β\* = −0.35 |
| β_a for **availability** at `net` | −0.035 (p=.030) | +0.003 (p=.68) |
| largest standardised term at `both` | β_s | **control** (β\* = +0.537) |
| **matched floor-on vs floor-off contrast** | **+0.100** (p=.0001, 12 pairs) | **+0.068** (p=.011, 8 pairs) |

\* The only significant β_a in the table is layer-2 usage in the delay arm, at
**+0.137** — *against* H1, not for it.

**The last row is the strongest single piece of evidence here**, because it is a
*manipulated* comparison rather than a correlation. Take pairs of models matched on
activity (|Δ log `a`| ≤ 0.2) that differ in floor condition, and ask whether removing the
silence increases the use of timing. H2 predicts it does. **It does, significantly, in
both arms** — and at the network level.

**Verdict: H2 is confirmed in the no-delay arm; the delay arm is inconclusive but points
the same way.** β_a is null at every site in both arms, its one significant instance
carries the wrong sign for H1, β_s carries the effect, and the controlled contrast is
positive and significant in both arms.

### 4.6 β_a across the three grids

All three grids were measured with the *same* retargeted scripts — one script with a
`GRID` switch, not three copies that drifted apart — so the same coefficient, computed
the same way, now exists for all three:

| arm | grid | constrained layers | n | **β_a** | 95% CI | p | β\* |
|---|---|---|---|---|---|---|---|
| **delay** | `v3` | layer 1 | 24 | **+0.085** | [+0.008, +0.162] | **.033** | +0.394 |
| **delay** | `v3L2` | layer 2 | 30 | **+0.087** | [+0.062, +0.113] | **<.0001** | **+0.638** |
| **delay** | `v3L12` | **both** | 24 | +0.021 | [−0.021, +0.063] | .32 | +0.137 |
| no-delay | `v3` | layer 1 | 24 | **+0.200** | [+0.118, +0.283] | **.0001** | +0.495 |
| no-delay | `v3L2` | layer 2 | 24 | −0.007 | [−0.031, +0.016] | .53 | −0.081 |
| no-delay | `v3L12` | **both** | 24 | −0.0005 | [−0.078, +0.076] | .99 | −0.002 |

**β_a differs significantly from zero in 3 of the 4 single-layer cells and in 0 of the 2
both-layer cells.**

**This is not a power problem.** The both-layer confidence intervals **exclude** the
single-layer point estimates in 3 of 4 comparisons. The best-powered single-layer cell is
`v3L2` delay, with n = 30, the widest activity axis of any grid (3.50×) and the largest
standardised effect (+0.638). The single-layer effect is not marginal, and the both-layer
null does not come from low power.

**The obvious deflationary explanation was checked and does not hold.** That explanation
— "the network-wide axis is simply a weaker manipulation" — would account for the
pattern, but the axes the two factorials *achieved* are the same size: the `a` axis spans
2.26× / 2.18× in the layer-1 grid against 2.11× / 2.72× here, with comparable `s` spans.
**The coefficient moved; the axis underneath it did not.**

§1.5 anticipated two outcomes: a null everywhere, which would be a fact about the
architecture, or an effect only under the network-wide constraint, which no single-layer
grid could deliver. **The data gave the mirror image of the second.** The effect appears
only when exactly one layer is constrained and vanishes when the network is constrained.
That is still a result no single-layer grid could produce — but it is a result *about
single-layer designs*, not about sparsity.

### 4.7 Primary arm

Two criteria, both re-measured on this grid's own checkpoints rather than carried across
a layer boundary:

| criterion | no-delay | delay | selects |
|---|---|---|---|
| readout-efficiency gap at `net` (smaller is better) | **+0.285** | **+0.076** | **delay** |
| β_a(usage) − β_a(control) at `both` (larger is better — the timing probe must behave differently from a pure rate insult) | −0.012 | +0.015 | delay, narrowly |

**The readout gap is decisive, and constraining both layers did not improve it.** In the
no-delay arm a simple linear decoder reading the hidden network beats the network's own
output layers by .285, on **100% of checkpoints**. Perturbation experiments in that arm
measure readout inefficiency as much as hidden-code structure, and no factorial design
can correct that.

**The delay arm is primary; the no-delay arm is retained as secondary, not discarded.** It
is the arm whose pooled confound had the most to break, and it has no learnable delays —
so any timing effect surviving there must arise from membrane dynamics alone.

---

## 5. Discussion

**1. H1 received no support at the network level, in either arm, at any injection site.**
With selectivity held fixed by construction and general robustness controlled for, the
coefficient on activity is indistinguishable from zero everywhere except in a single
cell, where it carries the *wrong* sign. Reducing the spike budget of a whole network
does not make that network use spike timing more.

**2. Selectivity is the variable that survives every location.** β_s is negative in both
arms and significant in the no-delay arm, and the manipulated floor-on/floor-off contrast
at matched activity is positive and significant in **both**. Plainly: *a "sparse" network
looks robust to timing damage not because its neurons fire few spikes, but because its
neurons fall completely silent for many stimuli, and that silence builds an identity code
timing damage cannot touch.* The original question therefore has no single answer — the
answer depends on **which channel the sparsity mechanism leaves open**.

**3. The single-layer activity effect is best read as compensation by the unconstrained
layer.** Under a network-wide constraint the two layers' selectivity moves together
(ρ = +0.977 / +0.944, §4.1), which is what a genuine network-level manipulation looks
like; under a single-layer constraint the other layer is free to move independently, and
a coefficient estimated there is estimated on a network compressed in one place and free
to expand in another. Removing the compensator drives β_a to zero and leaves β_s standing
(§4.6). This is an inference from the grid contrast rather than a directly measured
mediation — measuring ρ(`s1`, `s2`) *within* the two single-layer grids would test it
head-on, and that measurement is not yet in hand.

**4. Much of what looks like "timing use" is not about timing.** At the `both` site the
deletion control is pure rate damage with no timing component, and it is the **largest
standardised term** in the primary model of the delay arm (β\* = +0.537), reaching .82–.86
where usage reaches only .30–.55. A network-wide insult that destroys rate does far more
damage than one that destroys only timing, at every sparsity level tested. Any reading of
"timing use" that does not hold general robustness constant is partly a statement about a
different quantity.

**5. The two hidden layers are not two copies of the same object.** Layer 2 is cheaper to
damage, holds less decodable timing information, is used less by the readout, and leaks
more through its ceiling. Its natural firing rate is twice layer 1's in the delay arm and
*lower* than layer 1's in the no-delay arm. A constant calibrated at one layer must never
be carried to the other.

**6. Availability and usage remain distinct quantities.** In the no-delay arm the amount
of timing information *present* in the hidden network rises slightly as activity falls
(β_a = −0.035, p = .030) — the direction H1 predicts, with fewer spikes per active neuron
yielding more decodable information that requires timing. But the readout's *use* of that
information does not move (β_a = −0.0005, p = .99). The layer shifts toward a timing code
and the readout does not follow. Reporting only one of the two measures would give
opposite impressions of the same network.

---

## 6. Limitations

1. **No attribution to a specific layer.** Both controls move both layers, so the two
   layers' activity is correlated at +0.97 / +0.89 **by construction**. Every coefficient
   here is a network coefficient; layer attribution comes from the two single-layer grids.
2. **The activity axis is soft, not pinned.** The ceiling leaks — up to 59% of pairs stay
   above budget at the tightest row at layer 2. This is a real manipulated range (about
   2×), not a hard constraint, and layer 2's axis is the softer of the two.
3. **One activity curve is non-monotone** (delay arm, floor off, layer 1), so that cell
   must be checked first before any claim about layer-1 activity in that arm.
4. **The no-delay readout leaves about .285 of accuracy unused on every checkpoint**, so
   its perturbation scores are partly a statement about its output layers rather than its
   hidden code. This is why the delay arm is primary.
5. **`s` is not perfectly constant along the `a` axis** — it drifts by up to about 30
   points in the floor-off column. The design decorrelates the two axes; it does not make
   them orthogonal by construction.
6. **Perturbation magnitude is not equal across sparsity levels.** Moving a given fraction
   of many spikes disturbs a downstream neuron more than moving the same fraction of few
   spikes. The deletion control reduces this problem without removing it.
7. **No non-sparse reference curve exists for the `both` injection site.** The available
   unconstrained-baseline sweeps all perturbed one layer at a time, so the `both`-site
   figures carry no reference line and are not compared against a weaker insult.
8. **The layer-1 grid predates the settle rule of §3.3.** In 2 of its 16 cells the
   selected checkpoint comes from before the constraint had bound, biasing those cells'
   sparsity statistics toward "the penalty does not bind". Nothing in the reported results
   flags this, so it is a caveat on the `v3` row of §4.6 rather than on this grid.

---

## 7. Data, code, and reproduction

**Training scripts**
[sn_bothLayer_train_noDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_noDelay_v3.py) ·
[sn_bothLayer_train_withDelay_v3.py](../../exp_sparse_network/sn_bothLayer_train_withDelay_v3.py)

**Checkpoints and training logs** —
`sn_data/sparse_whole_{arm}_v3L12_k{k}_floor{f}_seed{seed}.pt`,
`sn_log/sparse_whole_{arm}_v3L12_train_summary.json` (24 rows each), plus 48 per-epoch
logs.

> **Key convention in the summary files.** In this grid the unsuffixed canonical keys
> (`spikes_per_active_neuron`, `silent_fraction`, `spikes_per_neuron`) hold the **pooled
> network** values, not layer-1 values. Per-layer values carry an explicit `_l1` or `_l2`
> suffix. Use the suffixed keys when comparing grids.

**Perturbation sweeps** — eight `*_bothLayer_evalOnly_*_v3.py` scripts under
[shd/](../../exp_sparse_network/shd/) (relocation),
[jitter/](../../exp_sparse_network/jitter/),
[shift/](../../exp_sparse_network/shift/) and
[deletion/](../../exp_sparse_network/deletion/), writing
`{folder}/log/sparse_whole_{arm}_v3L12_*_eval.json`.

**Analysis** — all three scripts take a `GRID` switch (`v3` / `v3L2` / `v3L12`), so one
script serves all three grids and the cross-grid comparison of §4.6 is legitimate:
[hidden_channel_decode.py](../../exp_sparse_network/v3_analysis/hidden_channel_decode.py) (availability) ·
[phase1_measure.py](../../exp_sparse_network/v3_analysis/phase1_measure.py) (usage and control) ·
[phase1_regress.py](../../exp_sparse_network/v3_analysis/phase1_regress.py) (regressions, contrast, figures)

**Figures** —
[result_visualization/bothLayer/results_visualization.ipynb](../../exp_sparse_network/result_visualization/bothLayer/results_visualization.ipynb)
holds the accuracy curves (one section per injection site, plus the three sites side by
side) and the manipulation check at both layers and pooled. Regression figures are at
`v3_analysis/fig/phase1_regress_v3L12_{arm}.png`.

### Reproducing the experiment

```
# training -- 24 models per arm, 1250 epochs
python sn_bothLayer_train_noDelay_v3.py
python sn_bothLayer_train_withDelay_v3.py

# the four sweeps, per arm, at sites l1 / l2 / both
python shd/shd_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python jitter/jitter_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python shift/shift_bothLayer_evalOnly_{noDelay,withDelay}_v3.py
python deletion/deletion_bothLayer_evalOnly_{noDelay,withDelay}_v3.py

# endpoint measures and regressions -- set GRID = "v3L12" in all three
python v3_analysis/hidden_channel_decode.py
python v3_analysis/phase1_measure.py
python v3_analysis/phase1_regress.py
```

> **Filename discipline.** The three grids share dataset, arm, `k`, floor and seed, so the
> `v3L12` tag is what keeps checkpoints and analysis outputs distinct; every grid now has
> measurements on disk, so a stale `GRID` value would silently target the wrong files
> rather than error. `phase1_regress.py` therefore checks the recorded grid tag of every
> input file against its own and refuses to join files that disagree.

Every sweep and every measurement is **evaluation-only**: the scripts load frozen
checkpoints, take no gradients, and apply perturbation only at evaluation.
