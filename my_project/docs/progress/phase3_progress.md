# Phase 3: Biologically Inspired Hidden Perturbations — Jitter & Deletion

## Overview

Apply the same biologically inspired perturbation types from Beyond Rate (per-spike jitter, per-neuron jitter, spike deletion) to the **1st hidden layer output** instead of the input. This probes what type of temporal information the internal representation carries.

## Perturbation Types (applied at hidden layer)

1. **Per-spike jitter:** Independent Gaussian noise (sigma: 0–25 ms) added to each spike time in the hidden layer output.
2. **Per-neuron jitter:** Same Gaussian shift applied to ALL spikes from a given hidden neuron (preserves intra-neuron ISI, disrupts cross-neuron timing within the hidden layer).
3. **Per-spike deletion:** Each spike in the hidden layer output independently deleted with probability p_d.

## Experiments

### 3A: Hidden jitter (per-spike)

| | Detail |
|---|---|
| **Architecture** | 2 hidden layers (128 each), SRMALPHA |
| **Perturbation target** | 1st hidden layer output |
| **Datasets** | SHD whole, part, norm |
| **Sweep** | sigma = 0, 5, 10, 15, 20, 25 ms |
| **Original code base** | `temporal_shd_project/code/pertubation/` |

### 3B: Hidden jitter (per-neuron)

| | Detail |
|---|---|
| **Architecture** | Same as 3A |
| **Perturbation target** | 1st hidden layer output |
| **Datasets** | SHD whole, part, norm |
| **Sweep** | sigma = 0, 5, 10, 15, 20, 25 ms |

**Key question:** Does per-neuron jitter at the hidden layer disrupt SGD-delay more than SGD? If so, cross-channel timing is maintained internally — not just at the input.

### 3C: Hidden deletion

| | Detail |
|---|---|
| **Architecture** | Same as 3A |
| **Perturbation target** | 1st hidden layer output |
| **Datasets** | SHD whole, part, norm |
| **Sweep** | p_d = 0, 0.2, 0.4, 0.6, 0.8 |

**Key question:** Is the internal representation more or less robust to deletion than the input representation?

## Progress

- [ ] Adapt perturbation utilities (jitter_per_spike, jitter_per_neuron, deletion_per_spike) to work on hidden layer spike trains
- [ ] Run per-spike jitter sweep × (whole, part, norm) × (SGD, SGD-delay)
- [ ] Run per-neuron jitter sweep × (whole, part, norm) × (SGD, SGD-delay)
- [ ] Run deletion sweep × (whole, part, norm) × (SGD, SGD-delay)
- [ ] Compare SGD vs SGD-delay sensitivity patterns against Beyond Rate input-level results
