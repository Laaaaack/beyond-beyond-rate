# SNN Basics: What You Need to Know to Read "Beyond Rate Coding"

## 1. What Is a Spiking Neural Network (SNN)?

In a standard Artificial Neural Network (ANN), neurons pass continuous values (e.g. 0.73) to each other. In a **Spiking Neural Network**, neurons communicate via discrete, all-or-nothing events called **spikes** (like biological neurons firing action potentials). A spike has no magnitude — it either happens or it doesn't, at a specific point in time.

This means **when** a neuron fires matters, not just **how much** it fires.

## 2. The Leaky Integrate-and-Fire (LIF) Neuron

The most common SNN neuron model. It works like a leaky bucket:

1. **Integrate:** Incoming spikes add charge to the neuron's **membrane potential** V(t).
2. **Leak:** The potential decays over time toward a resting value (like water leaking out).
3. **Fire:** When V(t) crosses a **threshold** V_th, the neuron emits a spike and resets.

The key equation (simplified):

```
tau * dV/dt = -V(t) + I(t)
```

- **tau (membrane time constant):** Controls how fast the potential decays. Large tau = long memory (slow leak); small tau = short memory (fast leak). This is a critical parameter in the paper.
- **I(t):** Input current from incoming spikes, weighted by synaptic weights.

After firing, V resets to 0 (or a reset value), and there may be a brief **refractory period** where the neuron cannot fire again.

## 3. Spike Coding Schemes

This is the central theme of the paper. There are fundamentally different ways information can be represented in spike trains:

### Rate Coding
Count how many spikes occur in a time window. More spikes = stronger signal. This discards all timing information — only the firing rate matters. Simple but wasteful.

### Temporal Coding (what the paper investigates)

- **Inter-Spike Interval (ISI):** Information is in the *time gap between consecutive spikes* of the same neuron. E.g., a 10 ms gap means class A, a 20 ms gap means class B.

- **Coincidence / Synchrony:** Information is in *which neurons fire at the same time*. E.g., if neurons 1 and 2 fire together, it's class A; if neurons 1 and 3 fire together, it's class B. The total spike count per neuron can be identical across classes.

- **Cross-Channel ISI (CCISI) / Polychrony:** Information is in the *time gap between spikes of different neurons*, with a causal direction. E.g., neuron A fires, then neuron B fires 15 ms later. This is a spatio-temporal spike pattern — it requires both spatial (which neuron) and temporal (when) information.

The paper's argument: temporal codes are vastly more information-rich than rate codes, but it's unclear whether training algorithms actually learn to use them.

## 4. The Training Problem: Why SNNs Are Hard to Train

In standard ANNs, we use **backpropagation** — computing gradients of the loss with respect to weights. This requires the activation function to be differentiable.

The problem: a spike is a **Heaviside step function** H(x) — it jumps from 0 to 1 at threshold. Its derivative is zero everywhere except at the threshold (where it's undefined). So standard backpropagation produces zero gradients almost everywhere, and training gets stuck.

## 5. Surrogate Gradient Descent

The solution used in this paper. The idea:

- **Forward pass:** Use the real Heaviside spike function (sharp threshold).
- **Backward pass:** Replace the Heaviside derivative with a smooth **surrogate derivative** that has useful gradients.

The paper uses:

```
dH/dx ≈ 1 / (alpha * |x| + 1)^2,   alpha = 100
```

This is a smooth, bell-shaped curve centered at the threshold. It gives larger gradients when the membrane potential is close to threshold (where small weight changes can shift spike timing), and smaller gradients far from threshold.

**Why this enables timing learning:** If an input spike arrives slightly earlier or later, the postsynaptic neuron crosses threshold at a different time. The surrogate derivative produces a correspondingly shifted gradient, so the network can learn to adjust weights to favor specific spike timings.

## 6. Axonal Delays

In biology, a spike takes time to travel along an axon from one neuron to another. This travel time is the **axonal delay**.

In the paper's models, delays are **learnable parameters**. A delay d on a connection means: "the postsynaptic neuron receives this spike d time steps after it was emitted."

Why delays matter:
- Without delays, a neuron can only integrate spikes arriving within its membrane time constant window (tau). To detect a pattern spanning 200 ms, tau must be ~200 ms, but then the neuron loses sensitivity to fine timing differences.
- With delays, the network can **shift spikes in time** to align them, then integrate with a short tau. This decouples the temporal range from temporal precision.

The paper shows delays add < 0.5% extra parameters but can double accuracy on timing-dependent tasks.

## 7. SLAYER Framework

**SLAYER** (Spike LAYer Error Reassignment in time) is a training framework for SNNs. Key features:
- Implements surrogate gradient backpropagation through time.
- Supports learnable axonal delays.
- Uses temporal credit assignment — it can figure out which spikes at which times were responsible for errors.

The paper uses SLAYER for all its SNN experiments.

## 8. Spikemax Loss

The loss function used for classification. At the output layer, each class has a dedicated neuron. **Spikemax** looks at the total spike count of each output neuron over the simulation window and treats the neuron with the most spikes as the predicted class. The loss encourages the correct class neuron to spike more than others.

This is analogous to softmax in standard ANNs, but operating on spike counts.

## 9. Key Datasets

### SHD (Spiking Heidelberg Digits)
- 20 classes of spoken digits (0-9 in English and German).
- Audio converted to spike trains via a simulated cochlea model (700 input neurons).
- Widely used neuromorphic benchmark.

### SSC (Spiking Speech Commands)
- 35 classes of spoken words.
- Same cochlea-based spike encoding as SHD.
- More challenging (more classes, more variability).

Both datasets contain **both** rate and timing information, which is why the paper constructs Norm variants to remove rate cues.

## 10. Perturbation Testing

The paper uses several perturbation methods to probe what information networks actually use:

| Perturbation | What it disrupts | What it preserves |
|---|---|---|
| **Spike timing perturbation (f)** | Replaces fraction f of spikes with random ones | Total spike count |
| **Per-spike jitter** | Independent Gaussian noise on each spike time | Spike count, approximate rate |
| **Per-neuron jitter** | Same shift for all spikes of one neuron | Intra-neuron ISIs, spike count |
| **Spike deletion** | Removes spikes with probability p | Timing of remaining spikes |
| **Time reversal** | Temporal order, causal structure, CCISI | Spike count, ISIs, coincidence patterns |

The logic: if a network's accuracy drops sharply under perturbation X but not Y, it tells you the network relies on the features disrupted by X.

## Glossary

| Term | Meaning |
|---|---|
| **Spike train** | A sequence of spike times for one neuron |
| **Raster plot** | Visualization with neurons on y-axis, time on x-axis, dots for spikes |
| **Firing rate** | Number of spikes per unit time |
| **Membrane potential** | The "charge" accumulated in a neuron before it fires |
| **Threshold** | The membrane potential value that triggers a spike |
| **Refractory period** | Brief time after a spike when the neuron cannot fire again |
| **Surrogate gradient** | Smooth approximation of the spike function's derivative, used only during backpropagation |
| **Polychrony** | Reproducible time-locked spiking patterns across multiple neurons |
| **Neuromorphic** | Hardware/algorithms inspired by biological neural systems |
| **Feedforward** | Network where information flows in one direction (input -> hidden -> output), no recurrence |
