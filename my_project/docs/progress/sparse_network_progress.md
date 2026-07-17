# First Milestone — Does sparsity increase temporal processing?

**Status:** not started
**Owner:** _(you)_
**Full design & rationale:** [sparse_network.md](sparse_network.md) — read that first if
anything below is unclear; this file is the execution checklist for the *first,
smallest* experiment only.

---

## 1. What this milestone proves (in one paragraph)

We want to test one supervisor's prediction: **the sparser a hidden layer's
spiking activity, the more the network depends on precise spike *timing*
(temporal processing) rather than on firing *rate*.** We test it cheaply on a
single setting first. We train a handful of SHD networks that differ only in how
sparse their 1st hidden layer is (controlled by a regularisation penalty), and
then measure how much each one's accuracy collapses when we scramble the timing
of that layer's spikes at test time. If the prediction holds, the *sparser*
networks should collapse *more*. This milestone is done when we have one scatter
plot that either shows that trend or clearly doesn't.

## 2. Key idea in three lines

- **Sparsity** = how few spikes the 1st hidden layer fires. We set it with an
  L1 penalty on hidden spikes during training.
- **Temporal processing** = how much test accuracy drops when we jitter (randomly
  shift in time) that layer's spikes at evaluation only. Big drop = the network
  was relying on timing.
- We already have code for the jitter test. We only add the sparsity penalty.

> ⚠️ **Protocol rule (do not break):** sparsity is applied **during training**;
> jitter is applied **only at evaluation**. Never train with jitter on — that is
> a different experiment (perturbation-aware) and it erases the effect. See
> [sparse_network.md](sparse_network.md) §"Why the FIXED-WEIGHT protocol".

## 3. Scope of THIS milestone (deliberately narrow)

| Fixed choice | Value | Why |
|---|---|---|
| Dataset | SHD `whole` (input_dim 700) | one dataset only, for speed |
| Network | SGD-delay (`use_delay = True`) | expected to show the strongest effect |
| Perturbation site | 1st hidden layer output | the project's primary site |
| Sparsity levels (`lam`) | 5 values, dense → sparse | enough to see a trend |
| Seeds per level | 3 | sparsity interacts with initialisation |
| **Total models to train** | **5 × 3 = 15** | |

Everything else (128–128 hidden units, 20 classes, `NumSpikes` loss, LR,
epochs, `SIGMA_VALUES`) is copied unchanged from the existing jitter scripts.

## 4. Files you will create

Make a new folder `my_project/exp_fixed_weight_perturbation/code/sparse_network/`
containing four scripts, each adapted from an existing one:

| New script | Copied from | Purpose |
|---|---|---|
| `sparse_train.py` | `.../perturbation/jitter/jitter_train.py` | clean training **+ sparsity penalty**; saves 15 checkpoints |
| `sparse_eval_jitter.py` | `.../perturbation/jitter/jitter_2ndLayer_evalOnly.py` | loads each checkpoint, sweeps **1st-layer** jitter at eval only |
| `measure_sparsity.py` | _(new, small)_ | reports each checkpoint's hidden firing rate + clean accuracy |
| `analyse.py` | _(new, small)_ | computes the temporal score and makes the plots |

> The eval template is the *2nd*-layer file, but this milestone perturbs the
> *1st* layer. The 1st-layer jitter logic already exists as
> `forward_with_hidden_perturbation(x, sigma)` inside
> `.../perturbation/jitter/jitter_train.py` — reuse that method rather than the
> 2nd-layer forward.

## 5. Step-by-step

### Step 0 — copy the scripts
Copy the two existing jitter scripts into the new folder under the new names
above. Confirm they run unchanged first (train 1 quick model, eval it), so you
know the baseline pipeline works before you modify anything.

### Step 1 — add the sparsity penalty to `sparse_train.py`

**1a. Let the network hand back its hidden spikes.** Change the clean `forward`:

```python
def forward(self, x, return_hidden=False):
    x = self._prepare_input(x)
    hidden1 = self._first_hidden(x)                 # binary spikes (differentiable)
    out = self._second_hidden_and_output(hidden1)
    return (out, hidden1) if return_hidden else out
```

**1b. Add the penalty in the training loop** (replace the plain loss line):

```python
outputs, hidden1 = net(x_batch, return_hidden=True)
task_loss = loss_fn.numSpikes(outputs, target)

# L1 sparsity penalty = mean spikes per hidden neuron per sample.
rate_reg = hidden1.sum(dim=-1).mean()
loss = task_loss + lam * rate_reg               # lam = 0 recovers normal training
```

**1c. Make `lam` a swept hyper-parameter**, exactly like `sigma` is swept in the
original script. Train one fresh model per `(lam, seed)` and save each as
`sparse_whole_delay_lam{lam}_seed{seed}.pt`.

Starting grid (these are *guesses* — calibrate in Step 1d):
```python
LAM_VALUES  = [0.0, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS       = [42, 43, 44]
```

**1d. Calibrate `lam` before committing to 15 runs.** Do this quickly:
1. Train `lam = 0` → record its hidden firing rate (the **dense anchor**) and
   clean accuracy.
2. Train the largest `lam` → if accuracy is already at chance (~5%), it is too
   strong; lower the top of the grid. If firing rate barely moved from the
   anchor, it is too weak; raise it.
3. Aim for a grid where the sparsest model still scores **clearly above chance**
   (say ≥ 40%) while firing much less than the dense anchor. Then run all 15.

> 💡 To validate the whole pipeline fast, temporarily set `EPOCHS = 400`. Only
> switch back to the full `EPOCHS = 1250` for the real 15-model run.

### Step 2 — measure sparsity + clean accuracy (`measure_sparsity.py`)
For every checkpoint, on the **test set**, record:
```python
# hidden1: (B, C, 1, 1, T) collected over the test set with sigma = 0
firing_rate     = hidden1.sum() / hidden1.numel()      # fraction of active bins  ← x-axis
spikes_per_neuron = hidden1.sum(dim=-1).mean()         # intuitive alt
silent_fraction = (hidden1.sum(dim=-1) == 0).float().mean()
clean_acc       = acc at sigma = 0
```
Save one row per checkpoint to a CSV/JSON: `lam, seed, firing_rate, silent_fraction, clean_acc`.

### Step 3 — jitter sweep per checkpoint (`sparse_eval_jitter.py`)
For every checkpoint, evaluate accuracy across the existing grid
`SIGMA_VALUES = [0, 1, 3, 5, 10, 17, 25]`, perturbing the **1st** hidden layer
(via `forward_with_hidden_perturbation`), `NUM_REPEATS = 3` for error bars, all
inside `torch.no_grad()`. Save `acc(sigma)` per checkpoint.

### Step 4 — score temporal processing + plot (`analyse.py`)
Turn each `acc(sigma)` curve into one number, **baseline-normalised and
chance-corrected** so that low-accuracy models are treated fairly:
```python
chance = 1 / 20                                   # = 0.05
retention = (acc[sigma_max] - chance) / (acc[0] - chance)   # sigma_max = 25
temporal_score = 1 - retention        # 0 = pure rate code, 1 = fully timing-dependent
```
Then make three plots:
1. **Headline:** `temporal_score` (y) vs `firing_rate` (x), one dot per
   (lam, seed). **H1 predicts a downward trend** (sparser = left = higher).
   Report Spearman correlation.
2. **Curves:** the raw `acc(sigma)` families, coloured by firing rate — should
   fan from flat (dense) to steep (sparse).
3. **Guard:** `clean_acc` vs `firing_rate` — proves the trend isn't just
   "sparser networks are worse."

## 6. How you know the milestone succeeded

- **Pipeline works:** 15 checkpoints trained; sparsity CSV and jitter curves
  produced without error.
- **Sparsity really varied:** the sparsest model's `firing_rate` is well below
  the `lam = 0` anchor (aim for a >2× spread), and all analysed models are above
  chance.
- **Result obtained (either direction is a valid finding):**
  - Headline plot shows a clear negative `firing_rate → temporal_score` trend
    (negative Spearman ρ) ⇒ supports H1, proceed to scale up.
  - Flat or positive trend ⇒ H1 not supported here; record it and check the
    diagnostics in [sparse_network.md](sparse_network.md) §"Pitfalls and confounds"
    before concluding.

## 7. Watch out for (top 3)

1. **Dead network** — if a model sits at ~5% accuracy, `lam` was too high; drop
   it from the analysis (a broken net is not "maximally temporal").
2. **Plot against measured `firing_rate`, never against `lam`** — the map from
   `lam` to sparsity is nonlinear and seed-dependent.
3. **Do not enable jitter during training** — training must be clean; jitter is
   eval-only.

## 8. Progress checklist

- [ ] Step 0 — folder created, copied scripts run unchanged
- [ ] Step 1a/1b — `return_hidden` + penalty added
- [ ] Step 1d — `lam` grid calibrated (dense anchor + above-chance sparse end)
- [ ] Step 1c — all 15 models trained and checkpointed
- [ ] Step 2 — sparsity + clean accuracy table produced
- [ ] Step 3 — jitter sweep run for all 15 checkpoints
- [ ] Step 4 — temporal scores + 3 plots produced
- [ ] Milestone verdict recorded (supports / does not support H1)

## 9. If it works — immediate next steps

Scale along one axis at a time (see [sparse_network.md](sparse_network.md)
§"Optional extensions"): add the no-delay network, then the 2nd-layer site, then
the `part`/`norm` SHD variants, and finally the synthetic ISI task as the
cleanest confirmation venue.
