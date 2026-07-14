# Why the Two Protocols Answer Different Questions (Fixed-Weight vs Perturbation-Aware)

This project runs every perturbation experiment under two training protocols,
kept as parallel tracks because they measure genuinely different things:

- **Perturbation aware training** (a.k.a. train-all / eval-all, or
  train-at-f / eval-at-f): a separate model is trained from scratch at each
  perturbation level *f* and evaluated at that level. Folder:
  `exp_perturbation_awared_training/`.
- **Fixed weight perturbation** (a.k.a. train-once / eval-all, or
  train-clean / eval-perturbed): one model is trained on clean data and the
  perturbation level is swept only at evaluation, inside `torch.no_grad()`.
  Folder: `exp_fixed_weight_perturbation/`.

The two protocols coincide at the **input** site (which is why the original
paper's train-at-f curves are clean and interpretable) but diverge sharply at a
**hidden** site. This note explains why, so the results from each track are read
against the right question. Neither protocol is "wrong" — they probe different
properties, and the shape of a curve only means something once you know which
protocol produced it.

## The observation

The original *Beyond Rate* paper applies its perturbation directly to the
**input spike trains**, using perturbation aware training (a separate model
trained from scratch at each level *f*). Under this protocol, the ISI accuracy
curve degrades cleanly as *f* increases:

| f   | input perturbation (original) |
|-----|--------------------------------|
| 0.0 | 0.972 |
| 0.2 | 0.894 |
| 0.4 | 0.828 |
| 0.6 | 0.776 |
| 0.8 | 0.654 |
| 1.0 | 0.607 |

When we copied that same train-at-f / eval-at-f protocol over to **hidden-layer
perturbation** for ISI (with the autograd bug already fixed via STE so the
network really does receive gradients), the curve was essentially flat:

| f   | hidden perturbation (train-at-f / eval-at-f, STE) |
|-----|---------------------------------------------------|
| 0.0 | 0.972 |
| 0.2 | 0.957 |
| 0.4 | 0.958 |
| 0.6 | 0.959 |
| 0.8 | 0.946 |
| 1.0 | 0.956 |

Same task, same architecture, same protocol — only the *site* of the
perturbation moved one layer deeper. Why does the curve collapse to flat?

## The answer: where the perturbation sits relative to learnable layers

The two experiments look symmetric but aren't. The asymmetry is whether the
network has a learnable layer **upstream** of the perturbation site that can
re-encode information into a code the perturbation does not destroy.

### Input perturbation: no upstream layer to compensate

```
[input X]
   │  ← timing destroyed here
   ▼
[layer 1]       ← downstream of the destruction
[hidden]
[layer 2]
[output]
```

When the perturbation is applied to the input, every layer of the network —
including the first — receives the already-corrupted spike train. There is
**no upstream learnable component** that can pre-process the data and recover
information that has already been destroyed. Whatever the network can do, it
must do using the perturbed input as its starting point.

For the ISI task, the discriminative signal at the input lives in inter-spike
intervals. Randomizing input timing erases that signal. What survives at f=1
is per-neuron rate, which carries some residual class information (the ISI
dataset's `firing_rates` and `isis` features are correlated with the class
label, but not perfectly). So train-at-f / eval-at-f at the input answers a
clean, well-posed question:

> "How much class information remains in the input after perturbation level
> f, and how well can the network exploit what's left?"

The 0.97 → 0.61 curve is the answer: the network gets ~0.61 from rate alone
at f=1, and gradually more as timing is restored at lower f.

### Hidden perturbation: the first layer is upstream and can route around it

```
[input X]              ← intact at every f
   │
   ▼
[layer 1]              ← FULL FREEDOM here; can rewrite the code
   │
   ▼ (hidden spikes)
   │  ← timing destroyed here
   ▼
[layer 2]              ← downstream of the destruction
[output]
```

When the perturbation is applied between the first and second hidden layer,
the input is **not touched at any f**. The first layer sees the full input
spike train with all temporal structure intact, on every batch, at every f.
Its job is to produce a hidden-layer spike train; it can choose any
input-feature → output-code mapping that minimizes the loss.

Crucially, the perturbation destroys per-neuron *timing* of the hidden
spikes but *preserves* per-neuron spike count (rate). So if the first layer
learns to encode the input feature it cares about (ISI) into the hidden
layer's rate channel, the perturbation is essentially a no-op from the
loss's point of view. Concretely: a "5 ms ISI" input sample causes hidden
neuron A to fire many times and hidden neuron B to fire few times; a
"30 ms ISI" sample reverses that. The class-discriminative signal lives in
the *number* of hidden spikes per neuron, not in *when* they occur, and that
signal is preserved by the perturbation by construction.

When we train under f=1 hidden perturbation, the loss landscape rewards
exactly this kind of representation, and the network finds it. So the
train-at-f curve at the hidden layer answers a different question — one about
solvability rather than reliance:

> "Can the first layer find an input-feature → hidden-rate mapping that lets
> the readout solve the task without using hidden timing?"

For ISI specifically the answer is yes, trivially, because ISI is a
per-neuron statistic that converts cleanly to a per-neuron count after a
learnable filter + spike. The flat 0.95 curve is the network confirming
this. It tells us almost nothing about whether the *unperturbed-trained*
network would actually use hidden timing if free to do so.

## What each protocol measures — and why we keep both

The protocols answer fundamentally different questions. Rather than pick one,
the project runs both and reads each curve against its own question.

| Protocol × site | What it asks |
|----------|--------------|
| Perturbation aware, **input** (train-at-f / eval-at-f) | "How much class signal survives at the input under perturbation level f?" |
| Perturbation aware, **hidden** (train-at-f / eval-at-f) | "Can the first layer find a hidden representation that's robust to perturbation level f?" |
| Fixed weight, **hidden** (train-at-0 / eval-at-f) | "Does the trained network's hidden representation rely on spike timing?" |

The headline Beyond Beyond Rate question — "do hidden layers maintain
spike-timing-based representations?" — is the **third row**, and that is what the
`exp_fixed_weight_perturbation/` track is built to answer. The **first row** is
the original Beyond Rate question; the input is the only information source it
can probe, so perturbation aware training there is exactly right. The **second
row** is a distinct, still-meaningful question — *is the task solvable under a
corrupted hidden representation?* — and it is what the
`exp_perturbation_awared_training/` track measures at the hidden site.

Fixed weight perturbation is the right lens for the third-row question because:

1. **It freezes the representation we want to probe.** The network commits
   to whatever hidden code it finds when training is unconstrained. We then
   stress that fixed representation.
2. **It removes the first layer's incentive to pre-translate.** Without
   perturbation pressure during training, the first layer has no reason to
   route everything through rate; it ends up using whichever code minimizes
   the f=0 loss, which may include timing.
3. **It makes the curve interpretable.** A drop in accuracy under hidden
   eval-at-f means the trained network's representation depends on hidden
   timing for that fraction of its discriminative power. A flat curve means
   it doesn't.

Perturbation aware training at the hidden site is not discarded — its flat
curve is a *positive result* for that protocol's question: it is direct
evidence that a rate-coded hidden representation is sufficient to solve the
task when the network is allowed to adapt to the perturbation.

## A subtle caveat that applies to both protocols

A perturbation probe only reveals codes that the **readout** is using. If
the network maintains rich hidden temporal structure but the readout layer
ignores it (e.g., layer 2 weights pool hidden spikes uniformly over time),
hidden perturbation will show no drop even though timing is present.

This is a general limitation of any perturbation-based representation probe
— the original Beyond Rate paper has the same caveat at the input level —
and it applies equally to all three protocols above. The test-time-only
protocol still gets us closer to the actual research question than
train-at-f does for the hidden site; it just doesn't get us all the way.

## Summary

- **Perturbation aware training** measures information content in the
  surviving channel. At the **input** (a hard bottleneck the network cannot
  route around) this coincides with "how much class signal survives," which is
  the original Beyond Rate question. At a **hidden** site it instead measures
  "can an upstream layer pre-translate the signal into the surviving (rate)
  channel," so the curve flattens once the network is allowed to adapt — a
  meaningful answer to *that* question, not a bug.
- **Fixed weight perturbation** freezes a naturally-trained model and sweeps
  the perturbation only at evaluation. This is the track that answers the
  Beyond Beyond Rate question — *does the trained hidden representation rely on
  spike timing?* — because it removes the upstream layer's incentive to
  pre-translate.
- Both tracks therefore coexist by design: `exp_fixed_weight_perturbation/`
  for the representation-reliance question and `exp_perturbation_awared_training/`
  for the solvability question, each reproduced at the input under
  `exp_beyond_rate/` as a pipeline check. The two tracks agree at the input
  site and diverge at hidden sites exactly as the analysis above predicts.
