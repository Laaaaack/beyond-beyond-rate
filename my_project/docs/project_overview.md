# Beyond Beyond Rate — Probing Internal Temporal Representations in SNNs

## Motivation

The *Beyond Rate* paper (Yu et al.) demonstrated that surrogate-gradient-trained SNNs can learn from spike timing cues, not just firing rates. Their key experimental tool was **input perturbation**: progressively replacing temporal structure in the input spike trains with random spikes (controlled by factor f), then measuring how accuracy degrades. This revealed what temporal information the *input* carries and the *network* exploits.

However, input perturbation only tells us that the network *receives* temporal information — it does not tell us **how the network internally represents it**. A trained SNN might extract timing cues at the input layer and immediately convert them into a rate code internally, or it might preserve and transform temporal structure throughout its hidden layers.

## Core Question

**Do the hidden layers of surrogate-gradient-trained SNNs maintain spike-timing-based representations, or do they collapse temporal information into rate codes?**

We answer this by shifting the perturbation site: instead of corrupting the input, we perturb the **output spike trains of the first hidden layer** at test time and observe how downstream layers and final accuracy are affected.

## Approach

### What stays the same (from Beyond Rate)

- **All original experiments are reproduced:** synthetic ISI, CCISI, coincidence, realistic SHD/SSC (whole/part/norm), biologically inspired perturbations (jitter, deletion), and time reversal.
- **Same model architectures:** SGD (learnable tau) and SGD-delay (tau + learnable delays), using the SLAYER framework with SRMALPHA neurons.
- **Same datasets and data generation pipelines.**
- **Same perturbation factor f in [0, 1]** controlling the degree of temporal disruption.

### What changes

| | Beyond Rate (original) | Beyond Beyond Rate (ours) |
|---|---|---|
| **Perturbation site** | Input spike trains | Output spike trains of the 1st hidden layer |
| **What is tested** | Whether the network exploits input temporal cues | Whether the network maintains internal temporal representations |
| **Training** | Standard (unperturbed inputs) | Standard (unperturbed inputs) — same trained models |
| **Inference** | Perturbed inputs fed to the network | Unperturbed inputs; hidden layer output intercepted and perturbed before passing to the next layer |

## Experiment Phases

Detailed plans and progress for each phase are tracked separately:

- [Phase 1: Synthetic Tasks](progress/phase1_progress.md) — ISI, CCISI, coincidence with hidden-layer perturbation
- [Phase 2: Realistic Speech Tasks](progress/phase2_progress.md) — SHD/SSC (whole/part/norm) with hidden-layer perturbation
- [Phase 3: Biologically Inspired Hidden Perturbations](progress/phase3_progress.md) — jitter, deletion at hidden layer
- [Phase 4: Time Reversal at Hidden Layer](progress/phase4_progress.md) — temporal order reversal at hidden layer

## Expected Outcomes

**If hidden layers preserve temporal codes:**
- Accuracy should degrade with increasing hidden perturbation f, similar to input perturbation.
- SGD-delay should be more sensitive to hidden perturbation than SGD (since it learns richer temporal representations).
- Jitter and reversal results at the hidden layer should mirror input-level patterns.

**If hidden layers collapse to rate codes:**
- Hidden perturbation (which preserves spike counts) should have minimal effect on accuracy.
- No difference between SGD and SGD-delay under hidden perturbation.
- The network would be using temporal input features but encoding them as rates internally.

## Project Structure

```
my_project/
├── docs/
│   ├── project_overview.md          ← this file
│   ├── basics/
│   │   └── beyond_rate_summary.md   ← summary of the original paper
│   └── progress/
│       ├── phase1_progress.md       ← synthetic tasks
│       ├── phase2_progress.md       ← realistic speech tasks
│       ├── phase3_progress.md       ← jitter / deletion at hidden layer
│       └── phase4_progress.md       ← time reversal at hidden layer
├── code/
│   ├── synthetic/                   ← ISI, CCISI, coincidence with hidden perturbation
│   ├── realistic/                   ← SHD/SSC with hidden perturbation
│   └── perturbation/                ← jitter, deletion, reversal at hidden layer
└── results/
```

## References

- Yu, Z., Sun, P., Akarca, D., & Goodman, D. F. M. — *Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks*
- Original codebase: `temporal_shd_project/code/`
