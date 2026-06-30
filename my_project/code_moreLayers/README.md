# Beyond Beyond Rate — Deeper Networks & Layer-wise Hidden Perturbation

## Motivation: from 2 points to a depth trajectory

The 2-hidden-layer experiments (`../code/`) established the core method: perturb the
**output spike trains of a hidden layer** at test/train time and measure how accuracy
degrades, to ask whether the hidden layers **maintain spike-timing codes** or **collapse
temporal information into rate codes**.

With only two hidden layers there are only two perturbation sites (1st and 2nd hidden
layer). Two points are too few to characterise a *trend* across depth. This sub-project
deepens the network to **4 hidden layers** and applies the same hidden-layer
perturbations **layer by layer** at each of the four sites.

This turns a 2-point comparison into a **depth trajectory**:

- We can measure how temporal sensitivity changes as a function of depth.
- We can locate *where* (if anywhere) along the network the temporal code is converted
  into a rate code.
- A trend over four sites is a far more reliable basis for the claim "the SNN does / does
  not genuinely use temporal information internally" than a single 1st-vs-2nd comparison.

**Only depth changes.** The neuron model, hidden width, delay scheme, datasets, loss,
optimizer, schedule, and the train-all / eval-all protocol are all held identical to the
2-layer work, so any change in behaviour is attributable to depth rather than to an
architectural confound, and the new results stay comparable with the existing 2-layer
results.

## Model: 4 hidden layers (homogeneous extension)

The same SLAYER / `SRMALPHA` building block as the 2-layer model — just more of it. Hidden
width stays **128** in every hidden layer (capacity per layer is held constant, not total
parameter count).

```
input
  → fc1 → spike → h1   ← perturbation site 1
  → (delay1) → fc2 → spike → h2   ← perturbation site 2
  → (delay2) → fc3 → spike → h3   ← perturbation site 3
  → (delay3) → fc4 → spike → h4   ← perturbation site 4
  → (delay4) → fc5 (readout) → spike → output
```

- **5 dense layers** (`fc1`…`fc5`), **4 hidden spiking layers** (`h1`…`h4`), **1 spiking
  readout** (`fc5`).
- **4 perturbation sites**: the output spike train of each hidden layer `h1`…`h4`. The
  readout output is the decision, not a perturbation site.
- **Two variants (unchanged):**
  - **SGD** — no delays.
  - **SGD-delay** — learnable per-transition delays `delay1`…`delay4`, one before each
    `fc` that follows a hidden layer (same placement rule as the 2-layer model, where a
    delay sits at the input of every non-first dense layer).

### Unified, depth-parameterised implementation

The 2-layer code hard-codes the perturbation site by shipping two near-identical scripts
per experiment (`*_train.py` for the 1st layer, `*_2ndLayer_train.py` for the 2nd). With
four sites that would mean four copies per experiment. Instead, the 4-layer model holds
its hidden layers in a `nn.ModuleList` and takes a **`perturb_layer ∈ {1, 2, 3, 4}`**
selector, so the site becomes a single config value driving one model definition and one
training script per experiment.

## Perturbation protocol: train-all / eval-all

For a chosen perturbation family and site `s`, the **train-all / eval-all** protocol
(identical in spirit to the 2-layer `train-at-f / eval-at-f`) is:

1. For each perturbation level, **train a fresh 4-hidden-layer model from scratch** with
   the perturbation **active at site `s` on every batch**.
2. The perturbation is wired through a **straight-through estimator (STE)** so gradient
   still flows to the layers before the site (the layers at and below `s` are trained
   *through* the perturbation; without the STE those layers would freeze at random init
   for any non-zero level).
3. **Evaluate at the same level**, repeated `NUM_REPEATS` times, reporting mean ± std.

This is repeated independently for every site `s ∈ {1, 2, 3, 4}` to build the
per-layer trajectory.

## Perturbation families (inherited from the 2-layer suite)

Each family is applied at the selected hidden site. "Preserves" indicates what is held
fixed, which determines whether the intervention is a pure *timing* attack or also changes
*rate*.

| Family | Control parameter | Mechanism at the hidden site | Preserves |
|---|---|---|---|
| **Spike relocation** (core) | `f ∈ [0, 1]` | A fraction `f` of each neuron's spikes are removed and re-placed at randomly chosen previously-empty time bins | Per-neuron spike count (rate) |
| **Per-spike jitter** | `σ` | Every spike is displaced by an independent Gaussian offset | Spike count; coarse timing |
| **Per-neuron jitter (shift)** | `σ` | One Gaussian offset is drawn per neuron and applied to *all* of that neuron's spikes | Spike count + intra-neuron ISI; disrupts cross-channel timing |
| **Deletion** | `p_d` | Each spike is independently dropped with probability `p_d` | Nothing — reduces spike count (rate) |
| **Time reversal (inverse)** | `f` + on/off | The hidden spike train is reversed in time | Spike count; reverses temporal order |

Rate-preserving families (relocation, shift, inverse) are the cleanest probes of *internal
temporal coding*: if the trained network is using a rate code at a given site, perturbing
that site should leave accuracy roughly unchanged.

## Tasks / datasets

- **Synthetic:** ISI, CCISI, coincidence.
- **Realistic:** SHD and SSC, each with the `whole` / `part` / `norm` variants.
- **Network variants:** SGD and SGD-delay for every task.

## Hyperparameters (reference: realistic SHD config)

| Group | Setting |
|---|---|
| Neuron | `SRMALPHA` (theta=10, tauSr=1, tauRho=0.1, tauRef=2, scaleRef=2, scaleRho=0.1) |
| Simulation | Ts=1, tSample=200 |
| Hidden width | 128 (× 4 hidden layers) |
| Delays (SGD-delay) | learnable, max 64, adaptive clamping |
| Loss | `NumSpikes` over [0, 200], target count 40 (true) / 4 (false) |
| Optimizer | Nadam, lr=0.1, MultiStepLR(milestones=[300], gamma=0.1) |
| Training | epochs=1250, batch=128, early-stop patience=300 |
| Reproducibility | seed=42; evaluation repeats=3 |
| Core level sweep | `f ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` (jitter/shift use a `σ` sweep, deletion a `p_d` sweep) |
| I/O dims | dataset-specific (e.g. SHD: input 700 `whole` / 224 `part`,`norm`, 20 classes; SSC: 35 classes) |

## Experiment scope

For each perturbation family, the full sweep is:

```
4 sites  ×  |levels|  ×  datasets  ×  {SGD, SGD-delay}   fresh models
```

Run independently for the core relocation family and each of the four biologically
inspired families (jitter, shift, deletion, inverse). The rate-control sanity test
(`../code/diff_rate_test/`) can be replicated at depth if needed.

## Planned directory structure

```
code_moreLayers/
├── README.md                     ← this file
├── synthetic/
│   ├── isi/
│   ├── ccisi/
│   └── coincidence/
├── realistic/
│   ├── shd/
│   └── ssc/
└── perturbation/
    ├── jitter/
    ├── shift/
    ├── deletion/
    └── inverse/
```

Each leaf holds one depth-parameterised training script driven by a `perturb_layer`
selector, mirroring the layout of `../code/` but collapsing the per-site script
duplication into a single configurable model.

## Expected outcomes (depth framing)

**If the hidden layers preserve temporal codes at all depths:**
- Accuracy degrades with the perturbation level at every site `s`.
- SGD-delay stays at least as sensitive as SGD across depth (it learns richer temporal
  structure), and rate-preserving families (relocation, shift, inverse) still hurt at the
  deeper sites.

**If the hidden layers collapse to a rate code with depth:**
- Rate-preserving perturbations lose their effect at deeper sites — the accuracy-vs-level
  curve flattens as `s` increases — while count-changing deletion still bites everywhere.
- The SGD vs SGD-delay gap shrinks at deeper sites.
- The depth at which the curve flattens **locates the temporal → rate transition**.

## References

- Yu, Z., Sun, P., Akarca, D., & Goodman, D. F. M. — *Beyond Rate Coding: Surrogate
  Gradients Enable Spike Timing Learning in Spiking Neural Networks*.
- 2-hidden-layer project overview: `../docs/project_overview.md`.
- 2-layer implementation this extends: `../code/`.
