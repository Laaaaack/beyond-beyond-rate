# Beyond Beyond Rate — Probing Internal Temporal Representations in SNNs

## Motivation

The *Beyond Rate* paper (Yu et al.) demonstrated that surrogate-gradient-trained SNNs can learn from spike timing cues, not just firing rates. Their key experimental tool was **input perturbation**: progressively replacing temporal structure in the input spike trains with random spikes (controlled by factor f), then measuring how accuracy degrades. This revealed what temporal information the *input* carries and the *network* exploits.

However, input perturbation only tells us that the network *receives* temporal information — it does not tell us **how the network internally represents it**. A trained SNN might extract timing cues at the input layer and immediately convert them into a rate code internally, or it might preserve and transform temporal structure throughout its hidden layers.

## Core Question

**Do the hidden layers of surrogate-gradient-trained SNNs maintain spike-timing-based representations, or do they collapse temporal information into rate codes?**

We answer this by shifting the perturbation site: instead of corrupting the input, we perturb the **output spike trains of a hidden layer** (the 1st and, separately, the 2nd) at test time and observe how downstream layers and final accuracy are affected.

## Approach

### What stays the same (from Beyond Rate)

- **All original experiments are reproduced:** synthetic ISI, CCISI, coincidence, realistic SHD/SSC (whole/part/norm), biologically inspired perturbations (jitter, shift, deletion), and time reversal.
- **Same model architectures:** SGD (learnable tau) and SGD-delay (tau + learnable delays), using the SLAYER framework with SRMALPHA neurons.
- **Same datasets and data generation pipelines.**
- **Same perturbation factor f in [0, 1]** controlling the degree of temporal disruption.

### What changes

| | Beyond Rate (original) | Beyond Beyond Rate (ours) |
|---|---|---|
| **Perturbation site** | Input spike trains | Output spike trains of the 1st or 2nd hidden layer |
| **What is tested** | Whether the network exploits input temporal cues | Whether the network maintains internal temporal representations |
| **Inference** | Perturbed inputs fed to the network | Unperturbed inputs; a hidden layer's output intercepted and perturbed before the next layer |

## Two perturbation protocols

The project studies the same question under **two training protocols**, kept as parallel tracks because they answer *different* questions (see [knowledge_bank/phase1_investigation.md](knowledge_bank/phase1_investigation.md) for the full rationale). Each lives in its own top-level folder.

| Protocol | Also called | Training | Evaluation | Folder |
|---|---|---|---|---|
| **Fixed weight perturbation** | train-once / eval-all; train-clean / eval-perturbed | One model trained on clean (unperturbed) data | Sweep the perturbation level f only at test time, inside `torch.no_grad()` | `exp_fixed_weight_perturbation/` |
| **Perturbation aware training** | train-all / eval-all; train-at-f / eval-at-f | A separate model trained at each perturbation level f | Each model evaluated at the level it was trained on | `exp_perturbation_awared_training/` |

- **Fixed weight perturbation** freezes the naturally-trained representation and stresses it — it measures whether the trained network's hidden code *relies on* spike timing.
- **Perturbation aware training** lets the network adapt to the perturbation during training — it measures whether the task is *solvable at all* under that perturbation (e.g. by learning a rate-coded hidden layer).

A perturbation site can be the **input** (reproducing the original paper) or a **hidden layer** (our contribution). The original-paper reproductions under both protocols live in `exp_beyond_rate/`, and serve as the pipeline-validation baseline.

## Experiment families

The same four families run across both protocols and both hidden-layer sites (1st and 2nd), for SGD and SGD-delay:

- **Synthetic** — ISI, CCISI, coincidence.
- **Realistic** — SHD / SSC (whole / part / norm variants).
- **Biologically inspired perturbations** — per-spike jitter, per-neuron shift, spike deletion.
- **Time reversal** — temporal-order reversal (the `inverse` experiments).

Two additional axes are explored as robustness checks:

- **Deeper networks** — `code_moreLayers/` repeats experiments with more than two hidden layers.
- **Sample rate** — `diff_rate_test/` (e.g. `2000_rate/`, `4000_rate/`) checks whether the simulation time step / sampling rate changes the conclusions. See [knowledge_bank/sample_rate.md](knowledge_bank/sample_rate.md).

## Expected Outcomes

**If hidden layers preserve temporal codes:**
- Under **fixed weight perturbation**, accuracy should degrade with increasing hidden perturbation f, similar to input perturbation.
- SGD-delay should be more sensitive to hidden perturbation than SGD (since it learns richer temporal representations).
- Jitter and reversal results at the hidden layer should mirror input-level patterns.

**If hidden layers collapse to rate codes:**
- Hidden perturbation (which preserves spike counts) should have minimal effect on accuracy under fixed weight perturbation.
- No difference between SGD and SGD-delay under hidden perturbation.
- The network would be using temporal input features but encoding them as rates internally.

Note that **perturbation aware training is expected to produce flatter curves** at the hidden site regardless of the answer above, because the upstream layer can pre-translate information into the rate channel the perturbation preserves. This is not a contradiction — it is exactly the difference the two protocols are designed to expose.

## Project Structure

```
my_project/
├── docs/
│   ├── project_overview.md              ← this file
│   ├── basics/                          ← SNN, dataset, and Beyond Rate background
│   ├── knowledge_bank/                  ← technical deep-dives (protocols, SLAYER, losses, sample rate)
│   └── legacy/
│       └── progress/                    ← original phase-based plans (historical, pre-reorg)
├── exp_beyond_rate/                     ← reproductions of the original paper (INPUT perturbation)
│   ├── fixed_weight_perturbation_test/
│   └── perturbation_awared_training_test/
├── exp_fixed_weight_perturbation/       ← Beyond Beyond Rate, train-once / eval-all
│   ├── code/                            ← default architecture (1 hidden synthetic, 2 hidden realistic/perturbation)
│   │   ├── synthetic/                   ← isi, ccisi, coincidence (+ diff_rate_test)
│   │   ├── realistic/                   ← shd, ssc
│   │   └── perturbation/                ← jitter, shift, deletion, inverse
│   ├── code_moreLayers/                 ← >2 hidden layers
│   ├── result_visualization/            ← paper figures + summary (1stLayer / 2ndLayer)
│   └── legacy/
└── exp_perturbation_awared_training/    ← Beyond Beyond Rate, train-all / eval-all
    ├── code/                            ← (migration in progress; currently diff_rate_test subset)
    ├── code_moreLayers/
    └── result_visualization/
```

## References

- Yu, Z., Sun, P., Akarca, D., & Goodman, D. F. M. — *Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks*
- Original codebase: `temporal_shd_project/code/` (the upstream authors' repository, referenced throughout `basics/` and `legacy/progress/`)
