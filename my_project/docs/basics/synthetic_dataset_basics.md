## ISI Dataset

The plot (fig. 1(a)) shows the **joint distribution of two features** across all samples, with each dot representing one neuron observation:

- **X-axis:** Firing rate of that neuron (Hz)
- **Y-axis:** ISI (inter-spike interval) — the time gap between the two spikes that neuron fires (ms)
- **Red vs Blue dots:** The two classes

### What the dataset looks like

Each sample has 10 neurons. Every neuron fires **exactly 2 spikes** (a spike pair), so:
- Its **rate** = how often it fires (x-axis)
- Its **ISI** = the gap between those 2 spikes (y-axis)

Class identity is encoded in the *combination* of (rate, ISI) — not in rate alone.

### The two lines explain the core tension

| Line | Meaning |
|---|---|
| **Vertical dashed line** ("Rate-only Threshold") | The best a rate-only classifier can do — it draws a vertical cut at ~6 Hz. But notice: at every firing rate, **both red and blue dots coexist**, so rate alone can't fully separate them. |
| **Diagonal solid line** ("Full Decision Boundary") | The optimal boundary when you use **both** rate and ISI. At this angle, the two classes are perfectly separable. |

### The punchline

If your model only counts spikes (rate coding), it gets stuck at the vertical threshold — many errors remain. Only by reading the **timing** between spikes (ISI) can it find the diagonal boundary and achieve ~100% accuracy.

This is exactly what the paper tests: does Surrogate GD let the SNN learn that diagonal boundary, or does it collapse to the vertical one?

---

## CCISI Dataset

CCISI stands for **Cross-Channel ISI** — the timing relationship being tested here is *between* two different neurons, not within a single neuron like in the ISI task.

### The core idea

Neurons are grouped into fixed pairs **(a, b)**. In each pair:
- Neuron **a** fires a spike at some time **t**
- Neuron **b** is constrained to fire at exactly **t + δ**

The class-specific information is encoded in **δ** — the lag between the two neurons. This is a **directional, causal** dependency: a always precedes b by exactly δ.

### Why this can't be decoded from rate or single-neuron ISI

- Both neurons in a pair share the same firing rate → **rate gives no class information**
- Each individual neuron has its own ISI pattern, but those ISIs don't contain δ → **single-neuron ISI gives no class information**
- The only way to extract δ is to **compare spike times across the two channels** — i.e., measure the lag between neuron a and neuron b

### Structure of a sample

With N neurons total, there are N/2 pairs indexed (2i, 2i+1) for i = 0, …, N/2 − 1. For a firing rate r (Hz) over a window of duration T (ms), each pair fires r · 2T/1000 spike pairs. The non-overlap constraint — no two pairs share the interval [t − δ, t + δ] — ensures spike pairs don't collide and confuse the signal.

### Why delays help here (but not in the ISI task)

In the ISI task, a single neuron's membrane integrates both spikes itself — the network can detect the gap using its time constant τ. But in CCISI, the two spikes land on *different* neurons. The post-synaptic neuron receiving b's spike has no memory of when a fired, unless an axonal **delay** on the a→post connection is tuned to match δ.

- For **short δ** (e.g., 50 ms): τ alone can bridge the gap, so both models perform similarly.
- For **long δ** (e.g., 200–500 ms): τ decays before the signal arrives. A learnable delay can shift neuron a's input forward in time to coincide with neuron b's spike, making the causal structure detectable. This is why **SGD-delay significantly outperforms SGD** at large δ.

---

## Coincidence Dataset

### Core idea

60 neurons are split into **3 groups of 20**. Time is divided into windows of 200 timesteps. In each window, every group is either **ON** (firing at high rate μ_on) or **OFF** (firing at low rate μ_off).

**Class identity** is defined by *which two groups are synchronous* — i.e., move ON/OFF together — while the third group is always inverted:

| Class | Groups in sync | Inverted group |
|---|---|---|
| A | 1 & 2 | 3 |
| B | 1 & 3 | 2 |
| C | 2 & 3 | 1 |

### Why rate is useless here

The ON/OFF toggling is **random per window**, so across windows each group fires ON roughly as often as OFF. Total spike counts are balanced across all groups and classes → **firing rate carries zero class information**. The only signal is the *correlation pattern* between groups.

### The λ difficulty knob

λ ∈ [0, 1] interpolates the ON and OFF firing rates toward each other:

- **λ = 0:** μ_on = 12, μ_off = 2 — groups are easy to tell apart, synchrony is obvious
- **λ → 1:** μ_on = μ_off = μ_avg = 5 — ON and OFF are indistinguishable, synchrony becomes invisible

The formula is: μ_on = (1−λ)·12 + λ·5, μ_off = (1−λ)·2 + λ·5

### What a network must do to solve this

It can't use rate. It can't use single-neuron ISI. It must detect **co-activation patterns across groups** — which groups tend to be active at the same time. This is pure **synchrony/coincidence detection**.

### Why delays don't help here

Unlike CCISI, there's no directional lag δ between neurons to learn. Coincidence is *simultaneous* co-activation, not a delayed causal chain. So learnable delays provide no benefit — both SGD and SGD-delay perform identically on this task.
