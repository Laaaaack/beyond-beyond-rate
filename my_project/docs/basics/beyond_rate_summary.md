# Summary: Beyond Rate Coding — Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks

**Authors:** Ziqiao Yu, Pengfei Sun, Danyal Akarca, Dan F. M. Goodman (Imperial College London)

## Core Question

Can Surrogate Gradient Descent (Surrogate GD) trained SNNs learn and depend on spike timing codes *beyond* rate statistics? The paper systematically tests whether SNNs can extract information encoded in the timing (not just the rate) of spikes.

## Models

Two architectures are compared throughout:

- **Learnable Tau (SGD):** Feedforward SNN with learnable synaptic weights and shared membrane time constant tau.
- **Tau + Delay (SGD-delay):** Same as above, augmented with learnable axonal delays.

Both use the SLAYER framework (SRMALPHA neuron model) with Spikemax loss. A **MLP baseline** (trained on spike count vectors only) serves as a proxy for rate-only performance.

### SNN Structure by Experiment

| Experiment | Hidden Layers | Hidden Size | Delay Config | Notes |
|---|---|---|---|---|
| ISI | 1 | 100 | max 15 time steps, both layers | — |
| CCISI | 1 | 100 | max 10 time steps, both layers | — |
| Coincidence | 1 | 3 | max 15 time steps, hidden only | Hidden size matches 3-class task |
| SHD / SSC | 2 | 128 each | 0–64 time steps, after each hidden layer | Delays dynamically expandable |
| Perturbation / Reversal | 2 | 128 each | (same as SHD/SSC) | Reuses SHD/SSC models |

---

## Experiment 1: Synthetic ISI Task

**Goal:** Test if SNNs can learn from inter-spike interval (ISI) structure alone.

**SNN Setup:** 1 hidden layer (Input → 100 hidden → Output), SRMALPHA neurons, learnable tau (init 50 ms). Delay variant adds learnable delays (max 15 time steps) on both layers.

**Setup:**
- 10 input neurons, each generating spike pairs with a class-specific ISI delta and firing rate r.
- Total spike count is kept constant; a perturbation factor f in [0,1] replaces a fraction of spikes with random ones (preserving count).
- f=0: all ISI info intact; f=1: fully randomized (rate-only).

**Results:**
- At f=0, both models achieve ~100% accuracy.
- Performance degrades gracefully as f increases; remains near-perfect until f=0.2, substantially degrades at f >= 0.4.
- At f=1, accuracy converges to ~60% (rate-only baseline).
- Delay module provides **no additional benefit** for this single-neuron ISI task.

**Code:** `temporal_shd_project/code/synthetic/isi/`
- `isi_data_gen.py` — dataset generation
- `isi_delay.py` — SNN with delay modules
- `isi_tau.py` — SNN with learnable tau only

---

## Experiment 2: Synthetic CCISI Task (Cross-Channel ISI)

**Goal:** Test if delays help learn cross-neuron causal temporal dependencies (polychrony).

**SNN Setup:** 1 hidden layer (Input → 100 hidden → Output), SRMALPHA neurons, learnable tau (init 50 ms). Delay variant adds learnable delays (max 10 time steps) on both layers.

**Setup:**
- 20 input neurons arranged in pairs (a, b). Neuron a fires first, neuron b fires at t + delta.
- Same perturbation scheme as ISI (factor f).
- Maximum ISI varied from 50 ms to 500 ms.

**Results:**
- At short intervals (50 ms), both models perform similarly.
- At longer intervals (200-500 ms), **Tau + Delay significantly outperforms Learnable Tau**.
- The delay module provides a complementary mechanism to membrane time constants for capturing long-range temporal dependencies.

**Code:** `temporal_shd_project/code/synthetic/ccisi/`
- `ccisi_data_gen.py` — dataset generation
- `ccisi_delay.py` — SNN with delay + tau
- `ccisi_tau.py` — SNN with learnable tau only

---

## Experiment 3: Synthetic Coincidence Task

**Goal:** Test if SNNs can learn from synchrony/coincidence statistics (not precise spike patterns).

**SNN Setup:** 1 hidden layer (Input → **3** hidden → 3 output), SRMALPHA neurons, learnable tau (init 50 ms, clamped 10–100 ms). Delay variant adds learnable delays (max 15 time steps) on hidden layer only. Hidden size matches the 3-class task.

**Setup:**
- 60 input neurons divided into 3 groups of 20.
- Class identity defined by which two groups fire synchronously (ON together / OFF together).
- Spike counts are balanced across groups — firing rate gives zero class information.
- Synchrony overlap factor lambda in [0,1] controls difficulty (lambda=0: easy; lambda=1: indistinguishable).

**Results:**
- At lambda=0, near-perfect accuracy.
- Performance degrades to chance (33.3%) as lambda -> 1.
- **No difference** between models with and without delays (expected, since time differences play no part).

**Code:** `temporal_shd_project/code/synthetic/coincidence/`
- `coin_data_gen.py` — dataset generation
- `coin_delay.py` — SNN with delay modules
- `coin_tau.py` — SNN with learnable tau only

---

## Experiment 4: Realistic Speech Datasets (SHD & SSC)

**Goal:** Test spike timing learning on real-world auditory tasks (Spiking Heidelberg Digits / Spiking Speech Commands).

**SNN Setup:** 2 hidden layers (Input → 128 → 128 → num_classes output), SRMALPHA neurons (theta=10, tauRef=2). Delay variant adds learnable delays (range 0–64 time steps, dynamically expandable) after each hidden layer. SHD output = 20 classes; SSC output = number of speech command classes.

### Dataset Construction

Three variants with progressively removed rate information:

1. **Whole:** Original dataset (700 neurons for SHD). Contains both rate and timing info.
2. **Part:** Neurons with very low minimum spike counts are removed (threshold theta=2). Samples causing low counts are also filtered. Classes are balanced by downsampling.
3. **Norm:** Spike trains are subsampled to a fixed per-neuron spike count (the minimum across all samples). **All rate information is eliminated**; only spike timing remains.

### Results (Spike Timing Perturbation f)
- In every case where f < 1, the SNN outperforms the MLP (rate-only) baseline.
- As f -> 1, SNN performance converges to MLP level.
- **SGD-delay consistently outperforms SGD:**
  - SHD-norm at f=0: SGD=23%, SGD-delay=48%
  - SSC-norm at f=0: SGD=15%, SGD-delay=33%
- SGD-delay degrades sharply around f=0.8, suggesting strong reliance on high-fidelity spike timing.

**Code:** `temporal_shd_project/code/realistic/shd/`
- `shd_data_gen.py` — converts SHD to dense spike trains
- `shd_train.py` — training script (supports whole/part/norm, delay/no-delay)
- `utils.py` — dataset download utility
- `draft/` — notebooks for shd_norm, shd_part, shd_whole, shdelay experiments

**Code:** `temporal_shd_project/code/realistic/ssc/`
- `data_gen_ssc_norm_part.py`, `data_gen_ssc_whole.py` — SSC dataset generation
- `data_norm_pre.py`, `data_part_pre.py`, `data_whole_pre.py` — preprocessing scripts
- `ssc_train.py` — main SSC training script (supports whole/part/norm, delay/no-delay)
- `draft/` — training scripts for SG/SLAYER models

---

## Experiment 5: Biologically Inspired Perturbations (Jitter, Deletion)

**Goal:** Determine what types of timing information the networks actually use.

**SNN Setup:** Reuses the SHD/SSC models (2 hidden layers, 128 neurons each). No new architectures; perturbation is applied to the data only.

### Perturbation Types
- **Per-spike jitter:** Independent Gaussian noise added to each spike time (sigma: 0-25 ms).
- **Per-neuron jitter:** Same Gaussian shift applied to ALL spikes from a given neuron (preserves intra-neuron ISI, disrupts cross-neuron timing).
- **Per-spike deletion:** Each spike independently deleted with probability p_d.

### Results (on SHD whole/part/norm)
- **Per-spike jitter:** SGD-delay maintains accuracy above MLP baseline for all noise levels; SGD falls to/below MLP.
- **Per-neuron jitter:** Both models drop to MLP level. SGD-delay is *more* disrupted than SGD, suggesting **delay-based networks rely heavily on cross-channel timing**.
- **Spike deletion:** Both models are fairly robust; no major difference between architectures.

**Code:** `temporal_shd_project/code/pertubation/`
- `utils.py` — perturbation utilities: `jitter_per_neuron()`, `jitter_per_spike()`, `deletion_per_spike()`
- `draft/jitter/` — jitter experiment notebooks
- `draft/deletion/` — deletion experiment notebooks (norm/part/whole variants)
- `draft/shift/` — shift experiment notebooks

---

## Experiment 6: Time Reversal

**Goal:** Test whether networks use temporal order / causal structure (like humans are sensitive to reversed speech).

**SNN Setup:** Reuses the SHD/SSC models (2 hidden layers, 128 neurons each). Time reversal is applied to the data only.

**Setup:**
- Train on original data, test on time-reversed spike trains.
- Time reversal preserves spike counts, neuron identities, single-neuron ISIs, and coincidence patterns — but disrupts temporal order and cross-channel causal structure (CCISI).
- Tested across perturbation levels f.

### Results
- **SGD (no delay):** Much less affected by time reversal — performance remains relatively stable.
- **SGD-delay:** Strongly degraded under time reversal, indicating it captures causal temporal dependencies.
- **Whole variant:** Under reversal at f=0, performance drops *below* rate-only MLP baseline.
- **Norm variant:** Performance drops but remains *above* MLP baseline under reversal.

**Code:** `temporal_shd_project/code/pertubation/`
- `inv_data_gen.py` — generates time-reversed (inverted) SHD datasets (whole_inv, part_inv, norm_inv)
- `draft/inverse/` — 12 notebooks for inverse experiments across dataset variants

---

## Key Takeaways

1. **Surrogate GD can learn diverse spike timing codes** — ISI, coincidence, and cross-channel ISI — not just rate-based features.
2. **Learnable axonal delays significantly enhance temporal learning**, especially for cross-channel and long-range causal dependencies, at minimal parameter cost (<0.5% increase).
3. **Delay-based networks rely more on cross-channel timing** (shown by per-neuron jitter sensitivity) and **causal/temporal order** (shown by time reversal sensitivity).
4. **SHD and SSC contain substantial temporal information**, but also allow surprisingly good rate-only classification on the original datasets — motivating the Norm variants that remove all rate information.
