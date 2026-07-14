# Phase 4: Time Reversal at Hidden Layer

## Overview

Reverse the temporal order of the 1st hidden layer output spike trains at test time. This tests whether downstream layers rely on causal/temporal order in the internal representation — the strongest evidence that temporal structure is preserved (not just received) by the network.

## Perturbation Method

At test time, after the 1st hidden layer produces its spike train:
- The spike train for each hidden neuron is time-reversed within its active window.
- This preserves spike counts, neuron identities, single-neuron ISIs, and coincidence patterns — but disrupts temporal order and cross-channel causal structure.
- Combined with the spike-timing perturbation factor f to test across temporal fidelity levels.

## Experiments

### 4A: SHD time reversal at hidden layer

| | Detail |
|---|---|
| **Architecture** | 2 hidden layers (128 each), SRMALPHA |
| **Perturbation target** | 1st hidden layer output (time-reversed) |
| **Datasets** | SHD whole, part, norm |
| **Sweep** | Reversal × f = 0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| **Original code base** | `temporal_shd_project/code/pertubation/` |

**Key questions:**
- Does SGD-delay show stronger degradation under internal time reversal than SGD, mirroring the input-level result?
- If so, causal temporal structure is preserved in the hidden representation, not just at the input.
- Under reversal at f=0, does hidden-layer reversal cause a larger or smaller drop than input-level reversal?

## Progress

- [ ] Implement time reversal function for hidden layer spike trains
- [ ] Run SHD time reversal sweep × (whole, part, norm) × (SGD, SGD-delay)
- [ ] Compare reversal sensitivity (SGD vs SGD-delay) against Beyond Rate input-level reversal results
- [ ] Analyse whether hidden reversal + f interaction follows the same pattern as input reversal + f
