# Phase 1 Investigation — Why Train-at-f Works for Input Perturbation but Not for Hidden Perturbation

## The puzzle

The original *Beyond Rate* paper applies its perturbation directly to the
**input spike trains**, with a separate model trained from scratch at each
perturbation level *f*. Under this train-at-f / eval-at-f protocol, the ISI
accuracy curve degrades cleanly as *f* increases:

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
train-at-f curve at the hidden layer answers a different — and much less
interesting — question:

> "Can the first layer find an input-feature → hidden-rate mapping that lets
> the readout solve the task without using hidden timing?"

For ISI specifically the answer is yes, trivially, because ISI is a
per-neuron statistic that converts cleanly to a per-neuron count after a
learnable filter + spike. The flat 0.95 curve is the network confirming
this. It tells us almost nothing about whether the *unperturbed-trained*
network would actually use hidden timing if free to do so.

## Why this isn't just a minor methodology nit

The two protocols answer fundamentally different questions, and only one of
them lines up with our research goal.

| Protocol | What it asks |
|----------|--------------|
| Input train-at-f / eval-at-f | "How much class signal survives at the input under perturbation level f?" |
| Hidden train-at-f / eval-at-f | "Can the first layer find a hidden representation that's robust to perturbation level f?" |
| Hidden train-at-0 / eval-at-f (test-time only) | "Does the trained network's hidden representation rely on spike timing?" |

The Phase 1–4 research question — "do hidden layers maintain spike-timing-
based representations?" — is the third row. The first row was the right
question for the original Beyond Rate paper because the input is the only
information source it could probe. The second row is what *isi_tau.ipynb*
and *isi_delay.ipynb* were doing before the protocol fix; it sounded like
the natural transposition of the original method, but it isn't.

Test-time-only is the right protocol for hidden perturbation because:

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

- Train-at-f / eval-at-f is the right protocol when the perturbation site
  is at a hard bottleneck the network cannot route around (the **input**).
  It measures the information content remaining in the surviving channel.
- Train-at-f / eval-at-f is the wrong protocol when there is a learnable
  layer upstream of the perturbation site (the **hidden layer**). The
  upstream layer pre-translates information into the surviving channel and
  the curve flattens regardless of whether the unperturbed-trained network
  actually uses the destroyed channel.
- For Phase 1 onward, hidden perturbation experiments must train once at
  f=0 and sweep f only at evaluation. `isi_tau.ipynb` and `isi_delay.ipynb`
  have been refactored to follow this; `ccisi_*`, `coin_*`, `shd_train`,
  `ssc_train`, and `inverse_train` already follow it.
