# Phase 2: Realistic Speech Tasks — Hidden-Layer Perturbation

## Overview

Apply hidden-layer spike-timing perturbation to the SHD and SSC experiments (whole, part, norm variants), testing whether temporal representations survive the first hidden layer in realistic auditory tasks.

## Perturbation Method

At test time, after the 1st hidden layer (128 neurons) produces its spike train:
- A fraction f (in [0, 1]) of spikes are replaced with random spikes, preserving spike count per neuron.
- f=0: hidden output untouched; f=1: all temporal structure destroyed (rate-only internal signal).

## Experiments

### 2A: SHD (hidden perturbation)

| | Detail |
|---|---|
| **Architecture** | Input → 128 hidden → 128 hidden → 20 output (SRMALPHA, theta=10, tauRef=2) |
| **Delay variant** | Learnable delays (0–64 time steps) after each hidden layer |
| **Perturbation target** | 1st hidden layer (128 neurons) output |
| **Dataset variants** | whole, part, norm |
| **Original code base** | `temporal_shd_project/code/realistic/shd/` |

**Key questions:**
- On SHD-norm (zero rate information in input), does the 1st hidden layer output still carry temporal information?
- Does SGD-delay maintain richer internal temporal representations than SGD?
- How does the hidden accuracy-vs-f curve compare to the input perturbation curve?

### 2B: SSC (hidden perturbation)

| | Detail |
|---|---|
| **Architecture** | Input → 128 hidden → 128 hidden → num_classes output (SRMALPHA) |
| **Delay variant** | Learnable delays (0–64 time steps) after each hidden layer |
| **Perturbation target** | 1st hidden layer (128 neurons) output |
| **Dataset variants** | whole, part, norm |
| **Original code base** | `temporal_shd_project/code/realistic/ssc/` |

**Key questions:**
- Same as SHD, but on a harder multi-class task — does the pattern hold?
- A steeper accuracy drop under hidden perturbation would suggest the internal representation is *more* temporally structured than the input.

## Progress

- [ ] Implement hidden-layer perturbation hook for 2-hidden-layer SLAYER models
- [ ] Run SHD hidden perturbation sweep (f = 0 to 1) × (whole, part, norm) × (SGD, SGD-delay)
- [ ] Run SSC hidden perturbation sweep (f = 0 to 1) × (whole, part, norm) × (SGD, SGD-delay)
- [ ] Compare accuracy-vs-f curves against original input perturbation results from Beyond Rate
