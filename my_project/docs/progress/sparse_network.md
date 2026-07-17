# Sparsity → Temporal Processing in Hidden Layers

## The question

> *"Induce different levels of sparsity in the hidden layers (through
> regularisation), then observe whether sparser-activity networks do more
> temporal processing than denser ones."* — supervisor's suggestion

**Hypothesis (H1):** as hidden-layer activity gets sparser, the network relies
*more* on precise hidden spike timing (more temporal processing) and less on a
rate code.

## Why this fits the existing project with no new probe

"Beyond Beyond Rate" already has a device that measures *how much a trained
network relies on hidden spike timing*: the **fixed-weight hidden-perturbation
sweep** (train on clean data, then perturb one hidden layer's output at
evaluation only, and watch accuracy fall). See
`knowledge_bank/phase1_investigation.md`:

- Steep accuracy drop under hidden jitter ⇒ the representation depends on
  hidden *timing* ⇒ lots of temporal processing.
- Flat curve ⇒ the readout is happy with a rate code ⇒ little temporal
  processing.

So the whole experiment is:

```
independent variable  :  hidden-layer sparsity   (set by a regulariser during training)
dependent variable    :  temporal processing     (= steepness of the fixed-weight jitter curve)
prediction (H1)        :  sparser  ⇒  steeper curve  ⇒  negative correlation
                          between hidden firing rate and jitter sensitivity
```

We reuse the SGD / SGD-delay `JitterSHDNetwork` architecture unchanged and only
add a sparsity term to the training loss.

## Why the FIXED-WEIGHT protocol, not perturbation-aware

This is the single most important design decision. The two protocols answer
different questions (`phase1_investigation.md`):

| Protocol | What it measures | Curve at hidden site |
|---|---|---|
| **Fixed weight** (train clean → eval-perturbed) | Does the trained representation *rely on* hidden timing? | Informative — flat vs steep |
| **Perturbation aware** (train-at-σ → eval-at-σ) | Is the task *solvable* under the perturbation? | Collapses to flat (upstream layer pre-translates to rate) |

Perturbation-aware training lets `fc1` learn to route everything through the
jitter-immune rate channel, so its curve is flat *regardless* of sparsity — it
would wash out the effect we are trying to see. **Sparsity is introduced during
training; the timing perturbation is applied only at evaluation.** They are two
separate interventions and must not be mixed into one training run.

Base scripts to copy from (the clean-train + eval-only pair):
- Train clean: `exp_fixed_weight_perturbation/code/perturbation/jitter/jitter_train.py`
- Eval sweep (1st layer): the `forward_with_hidden_perturbation(x, sigma)` path in that same file
- Eval sweep (2nd layer): `exp_fixed_weight_perturbation/code/perturbation/jitter/jitter_2ndLayer_evalOnly.py`

## Operational definitions

**Sparsity metric (report per trained model, measured on the test set).**
Let `h1` be the 1st hidden layer spike tensor, shape `(B, C, 1, 1, T)`.
- Mean firing rate: `r = h1.sum() / (B * C * T)` — fraction of active
  (neuron, time-bin) slots. This is the headline x-axis.
- Mean spikes per neuron per sample: `h1.sum(dim=-1).mean()` — more intuitive.
- Silent-neuron fraction and the per-neuron spike-count histogram (diagnostic).

**Temporal-processing metric (dependent variable).** From the fixed-weight
jitter curve `acc(σ)` over `σ ∈ SIGMA_VALUES`, quantify the drop in a way that
**controls for baseline accuracy** (sparse nets may start lower):

- Primary: **relative retained accuracy** `acc(σ_max) / acc(0)` (lower = more
  temporal), or equivalently relative drop `1 − acc(σ_max)/acc(0)`.
- Robust: **normalised area over the curve**,
  `1 − mean_σ[acc(σ)] / acc(0)`, which uses the whole sweep, not just the ends.
- Always chance-correct first: replace `acc` with `(acc − 1/NUM_CLASSES)`
  before ratioing, so a net that has collapsed to chance doesn't masquerade as
  "maximally temporal."

## Why the mechanism predicts H1 (and what jitter actually isolates)

Per-spike Gaussian jitter (`jitter_hidden_batch`) moves each spike in time but
**preserves, per neuron, both the spike count and the neuron's identity**. It
therefore destroys only *timing* codes (latency, synchrony, within-train
temporal patterns) and leaves *count/identity* (rate + population) codes intact.

- **Dense hidden layer:** each neuron fires many spikes, so per-neuron *count*
  has resolution — the network can store class information in a rate code, which
  jitter cannot touch ⇒ flat curve.
- **Sparse hidden layer:** each neuron fires 0–2 spikes, so count carries almost
  no resolution. The only informative axis left for an active neuron is *when*
  its lone spike occurs — a latency code, which jitter destroys ⇒ steep curve.

So sparsity **pushes information out of the jitter-immune count channel into the
jitter-fragile latency channel**. That is exactly H1, sharpened.

## Variables

- **Independent:** regularisation strength `λ` (produces a range of *measured*
  sparsities — analyse against measured sparsity `r`, not against `λ`, because
  `λ`→sparsity is nonlinear and seed-dependent).
- **Dependent:** temporal-processing metric above.
- **Control / held fixed:** architecture (128–128), epochs, LR, seed set,
  dataset variant, delay setting, output loss (`NumSpikes`, which *pins the
  output firing rate* so the sparsity intervention stays localised to the
  hidden layer), and the eval `SIGMA_VALUES` grid.
- **Split by (not pooled over):** delay vs no-delay (SGD-delay is expected to
  show the stronger effect), and hidden-layer site (1st vs 2nd).

## Procedure

### Step 1 — add a hidden-layer spike-rate regulariser

Expose the hidden spikes and add an L1 penalty on their count. In the
fixed-weight `JitterSHDNetwork` (clean-train version):

```python
def forward(self, x, return_hidden=False):
    x = self._prepare_input(x)
    hidden1 = self._first_hidden(x)          # binary spikes, surrogate-grad differentiable
    out = self._second_hidden_and_output(hidden1)
    return (out, hidden1) if return_hidden else out
```

In the training loop:

```python
outputs, hidden1 = net(x_batch, return_hidden=True)
task_loss = loss_fn.numSpikes(outputs, target)

# L1 sparsity penalty: mean spikes per hidden neuron per sample.
rate_reg = hidden1.sum(dim=-1).mean()
loss = task_loss + lam * rate_reg
```

Gradient flows to `fc1` through SLAYER's surrogate spike gradient (the same
path the task loss already uses), so the penalty genuinely drives the layer
toward fewer spikes.

- **Default:** plain L1 (`lam * rate_reg`), sweep `lam` on a coarse log grid,
  e.g. `[0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]`, then *measure* the sparsity each
  produces. Exact values need calibrating to the `NumSpikes` loss scale — start
  by finding the `lam` where clean accuracy first degrades, then bracket below it.
- **Alternative (tighter control of the x-axis):** target-rate hinge
  `rate_reg = torch.relu(counts − target_count)` and sweep `target_count`
  downward; gives more evenly spaced sparsity points but adds a hyperparameter.
  Start with L1; switch only if the achieved sparsities cluster.

### Step 2 — train the sparsity family (clean, NO perturbation)

For each `lam` (and ≥3 seeds), train one model with `sigma = 0` everywhere —
this is ordinary clean training plus the reg term. Save each checkpoint tagged
by `lam` and seed. Keep everything else identical to the existing clean run.

### Step 3 — measure achieved sparsity + clean accuracy

For each checkpoint, on the test set record: mean firing rate `r`, spikes/
neuron/sample, silent-neuron fraction, count histogram, and clean accuracy
`acc(0)`. **Report `acc(0)` alongside sparsity** — it is the main confound.

### Step 4 — run the fixed-weight jitter sweep per model

Feed each checkpoint through the eval-only sweep over `SIGMA_VALUES`
(`NUM_REPEATS` for error bars), perturbing the **same layer you regularised**.
This produces `acc(σ)` per model. No training here — pure `torch.no_grad()`
evaluation, reusing the existing eval script.

### Step 5 — quantify the temporal metric

Collapse each `acc(σ)` curve to the temporal-processing scalar from
"Operational definitions" (chance-corrected relative retention / normalised
area-over-curve).

### Step 6 — analysis

- **Headline plot:** temporal-processing metric (y) vs measured hidden firing
  rate `r` (x), one point per (lam, seed), separate panels/markers for
  delay vs no-delay. H1 predicts a clear downward trend (sparser → more
  temporal). Report Spearman ρ and a regression with seed as a grouping factor.
- **Supporting plot:** the raw `acc(σ)` families, coloured by sparsity — should
  fan from flat (dense) to steep (sparse).
- **Guard plot:** clean accuracy `acc(0)` vs sparsity, to show the trend is not
  merely "sparser nets are worse."

## Pitfalls and confounds

1. **Dead network.** Too-large `lam` drives the hidden layer to (near) silence
   and accuracy to chance (`1/20 = 5%`). Only analyse models whose clean
   accuracy stays meaningfully above chance; a chance-level net is not
   "maximally temporal," it is broken — chance-correct the metric (Step 5).
2. **Accuracy confound.** Sparse nets may have lower `acc(0)`. Never compare raw
   drops; use the baseline-normalised metric and show the guard plot.
3. **Jitter preserves count *and* population identity.** A pure population/rate
   code survives jitter even when sparse. That is fine — it is *why* the curve
   is a clean timing probe — but it means the effect lives in the *latency*
   channel; if H1 fails, check whether sparse nets moved to a population-identity
   code instead (diagnostic: does f=1 relocation, which also scrambles nothing
   about identity, drop accuracy more than jitter?).
4. **Readout caveat (inherited).** The probe only reveals timing the *readout*
   uses; a layer could hold temporal structure that `fc3` ignores. Same caveat
   as the rest of the project — state it, don't try to fix it here.
5. **`λ` is not the x-axis.** Always plot against *measured* sparsity, not `λ`.
6. **Seeds.** Sparsity regularisation interacts strongly with initialisation;
   ≥3 seeds per `lam`, and treat each (lam, seed) as one data point.

## Minimal first milestone (before scaling up)

Prove the effect exists cheaply, then scale:

- One dataset (`whole`), **one** setting (SGD-delay), 1st hidden layer only.
- 4–5 `lam` values spanning dense→sparse, 3 seeds each ⇒ 12–15 clean trainings.
- Step 4 eval + Step 6 plot.

If the downward trend appears, expand to: no-delay, 2nd-layer site, `part`/`norm`
variants, and the synthetic ISI task (cleanest, because its temporal signal is
exactly understood — a strong confirmation venue).

## Optional extensions

- **Pure-rate ceiling:** evaluate each model under f=1 relocation (or σ→large)
  as the "all timing destroyed, count kept" floor; the gap between `acc(0)` and
  this floor is another temporal-reliance estimator, independent of the σ grid.
- **Regularise/perturb the 2nd layer** to check the effect is not specific to
  layer 1.
- **SpikeMax (`probSpikes`) instead of `NumSpikes`** as a robustness check on
  the loss choice (it lets the net use fewer output spikes; keep for later —
  `NumSpikes` keeps the output rate pinned, which is desirable here).
```
