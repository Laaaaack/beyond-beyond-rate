# Phase 1: Synthetic Tasks — Hidden-Layer Perturbation

## Overview

Reproduce the three synthetic experiments from Beyond Rate (ISI, CCISI, Coincidence), but apply spike-timing perturbation to the **output of the 1st hidden layer** instead of the input.

## Perturbation Method

At test time, after the first hidden layer produces its spike train:
- A fraction f (in [0, 1]) of spikes are replaced with random spikes, preserving spike count per neuron.
- f=0: hidden output untouched; f=1: all temporal structure destroyed (rate-only internal signal).

## Experiments

### 1A: ISI (hidden perturbation)

| | Detail |
|---|---|
| **Architecture** | Input → 100 hidden → Output (SRMALPHA, learnable tau) |
| **Delay variant** | Learnable delays (max 15 time steps) on both layers |
| **Perturbation target** | 100-neuron hidden layer output |
| **Original code base** | `temporal_shd_project/code/synthetic/isi/` |

**Key questions:**
- Does the hidden layer preserve ISI structure, or has it already converted to rate?
- At f=1 (hidden), does accuracy match the original f=1 (input) baseline (~60%)?

### 1B: CCISI (hidden perturbation)

| | Detail |
|---|---|
| **Architecture** | Input → 100 hidden → Output (SRMALPHA, learnable tau) |
| **Delay variant** | Learnable delays (max 10 time steps) on both layers |
| **Perturbation target** | 100-neuron hidden layer output |
| **Original code base** | `temporal_shd_project/code/synthetic/ccisi/` |

**Key questions:**
- For CCISI at long intervals (200–500 ms), does the delay model maintain cross-channel timing internally?
- Does the delay advantage seen at the input level persist when perturbation is applied at the hidden layer?

### 1C: Coincidence (hidden perturbation)

| | Detail |
|---|---|
| **Architecture** | Input → 3 hidden → 3 output (SRMALPHA, learnable tau, clamped 10–100 ms) |
| **Delay variant** | Learnable delays (max 15 time steps) on hidden layer only |
| **Perturbation target** | 3-neuron hidden layer output |
| **Original code base** | `temporal_shd_project/code/synthetic/coincidence/` |

**Key questions:**
- Does the hidden layer preserve synchrony patterns?
- With only 3 hidden neurons, is there enough capacity to maintain a temporal code?

## Progress

- [ ] Implement hidden-layer perturbation hook for 1-hidden-layer SLAYER models
- [ ] Run ISI hidden perturbation sweep (f = 0, 0.2, 0.4, 0.6, 0.8, 1.0) for both SGD and SGD-delay
- [ ] Run CCISI hidden perturbation sweep across interval lengths (50, 200, 500 ms)
- [ ] Run Coincidence hidden perturbation sweep across lambda values
- [ ] Compare accuracy-vs-f curves against original input perturbation results
