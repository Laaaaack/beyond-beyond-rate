# Phase 0: Reproducing Original Beyond Rate Results

## Motivation

Phases 1–4 (hidden-layer perturbation experiments) are largely complete, but some results appear inconsistent with expectations. Before drawing conclusions, we need to verify our pipeline reproduces the **original Beyond Rate (input-perturbation) results**. If our reproductions match the paper, any anomalies in Phases 1–4 reflect genuine findings about hidden-layer representations rather than bugs in training, perturbation, or evaluation code.

## Scope

A representative subset is selected across the three task families — enough to validate the pipeline without redoing every experiment in the paper.

| Family | Experiment | Rationale |
|---|---|---|
| Synthetic | **ISI** | Simplest synthetic task; direct test of basic timing readout. |
| Realistic | **SHD** | Standard speech benchmark; covers the realistic-data pipeline. |
| Biological perturbation | **Per-spike jitter** | Probes graded temporal degradation rather than full randomisation. |

CCISI, coincidence, SSC, deletion, and time reversal are deferred unless the three above reveal pipeline issues.

## What stays original (vs. Phases 1–4)

- **Perturbation site:** input spike trains (not hidden layer).
- **Models:** SGD (learnable tau) and SGD-delay (tau + learnable delays), SLAYER + SRMALPHA.
- **Perturbation factor:** f in [0, 1] sweep matching the paper.
- **Datasets and generators:** unchanged from Beyond Rate.

## Experiments

### 0A: ISI (input perturbation)

| | Detail |
|---|---|
| **Architecture** | Input -> 100 hidden -> Output |
| **Delay variant** | SGD and SGD-delay (max 15 time steps) |
| **Perturbation target** | Input spike trains |
| **Sweep** | f = 0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| **Reference code** | `temporal_shd_project/code/synthetic/isi/` |

**Reference targets (paper):**
- SGD-delay: high accuracy at f=0, degrading toward chance at f=1.
- f=1 floor near chance (~50% for binary ISI task).

### 0B: SHD (input perturbation)

| | Detail |
|---|---|
| **Architecture** | Per Beyond Rate SHD config |
| **Delay variant** | SGD and SGD-delay |
| **Perturbation target** | Input spike trains |
| **Sweep** | f = 0, 0.2, 0.4, 0.6, 0.8, 1.0 |
| **Variants** | whole / part / norm (start with whole) |
| **Reference code** | `temporal_shd_project/code/realistic/shd/` |

**Reference targets (paper):**
- Test accuracy at f=0 around the published SHD numbers for SGD vs SGD-delay.
- Monotonic accuracy decay with f; SGD-delay degrades faster than SGD if it relies more on input timing.

### 0C: Per-spike jitter (input perturbation)

| | Detail |
|---|---|
| **Task** | SHD (re-uses 0B trained models) |
| **Perturbation** | Each input spike shifted by Gaussian noise of std sigma_j ms |
| **Sweep** | sigma_j across the range used in the paper |
| **Reference code** | `temporal_shd_project/code/perturbation/jitter/` |

**Reference targets (paper):**
- Smooth accuracy decay with sigma_j.
- Larger drop for SGD-delay than SGD at moderate jitter.

## Success Criteria

A reproduction is considered successful if, for each experiment:
1. The accuracy-vs-f (or vs sigma_j) curve qualitatively matches the published shape.
2. Endpoint accuracies at f=0 and f=1 (or low/high jitter) fall within a reasonable margin of the paper's values.
3. The relative ordering of SGD vs SGD-delay matches the paper.

If all three pass, the pipeline is trusted and Phases 1–4 anomalies are treated as substantive results. If any fail, debug training/perturbation/eval before re-interpreting Phases 1–4.

## Progress

- [ ] 0A: ISI input-perturbation sweep, SGD and SGD-delay
- [ ] 0B: SHD input-perturbation sweep, SGD and SGD-delay
- [ ] 0C: SHD per-spike jitter sweep, SGD and SGD-delay
- [ ] Side-by-side comparison plots against paper figures
- [ ] Decision: pipeline trusted vs. debug required
