# `LIF_PARAMS` — SLAYER SRM Neuron Descriptor

`LIF_PARAMS` is the neuron configuration dict passed to `snn.layer(LIF_PARAMS, SIM_PARAMS)`
in every training script. Despite the name "LIF", the model is not a plain
leaky-integrate-and-fire neuron: it is SLAYER's **Spike Response Model (SRM)** with
alpha-function kernels (`type: "SRMALPHA"`). This note explains each field, grounded in
the installed SLAYER source (`slayerSNN/slayer.py`).

## The value used in this project

```python
LIF_PARAMS = {
    "type": "SRMALPHA",
    "theta": 10,      # spike threshold — 2-layer models; the 4-layer models use 2 (see below)
    "tauSr": 1,       # PSP (spike-response) time constant
    "tauRho": 0.1,    # surrogate-gradient width (scaled by theta)
    "tauRef": 2,      # refractory time constant
    "scaleRef": 2,    # refractory magnitude (× theta)
    "scaleRho": 0.1,  # surrogate-gradient height
}
```

Time constants are in **simulation steps** (`SIM_PARAMS = {"Ts": 1, "tSample": 200}`, so
one step ≈ 1 ms and each sample is 200 steps long).

## How the neuron works

For each neuron SLAYER computes a membrane potential and emits a spike when it crosses the
threshold:

1. **Input filtering (PSP).** Incoming spikes are convolved with the **spike-response
   kernel** `ε(t)` (an alpha function), then weighted by the layer weights `W`. In code:
   `spike(fc(psp(x)))` → membrane potential `u = W · (ε * x)`.
2. **Thresholding.** A spike is emitted wherever `u` crosses `theta`.
3. **Refractoriness.** Each output spike is convolved with the **refractory kernel** `ν(t)`
   (a *negative* alpha function of magnitude `scaleRef · theta`) and subtracted from `u`,
   resetting the neuron and suppressing immediate re-firing.
4. **Backward pass.** The spike function is non-differentiable, so its gradient is replaced
   by a **surrogate gradient** — a smooth bump centred on the threshold (see below).

### Alpha kernel

Both `ε` and `ν` are alpha functions of the general form

```
k(t) = mult · (t / τ) · exp(1 − t / τ),   for t ≥ 0
```

which rises to a peak of `mult` at `t = τ` and then decays. The kernel is sampled at `Ts`
and truncated once it falls below a small epsilon.

| Kernel | `τ` | `mult` | Peak |
|---|---|---|---|
| Spike response `ε` (PSP) | `tauSr` | `1` | `1` at `t = tauSr` |
| Refractory `ν` | `tauRef` | `−scaleRef · theta` | `−scaleRef · theta` at `t = tauRef` |

### Surrogate gradient

The backward pass of the spike operation uses (SLAYER `_spikeFunction`):

```
∂spike/∂u  ≈  (scaleRho / (tauRho · theta)) · exp( −|u − theta| / (tauRho · theta) )
```

A Laplace-shaped bump centred at `theta`. Its **width** is `tauRho · theta` (note the
scaling by `theta`, applied internally) and its **peak height** is
`scaleRho / (tauRho · theta)`. Gradient only flows to neurons whose membrane potential sits
within roughly this width of threshold.

## Parameter-by-parameter

| Field | Meaning | This project | SLAYER default | Effect of the project value |
|---|---|---|---|---|
| `type` | Neuron/kernel model | `SRMALPHA` | `SRMALPHA` | SRM with alpha PSP and refractory kernels. |
| `theta` | Spike threshold; also scales `ν` and the surrogate-gradient width | `10` (2-layer) → `2` (4-layer) | `10` | Membrane potential must reach threshold to spike; lowered to 2 for the deep stacks (see below). |
| `tauSr` | PSP time constant `τ` of `ε` | `1` | `10.0` | **Sharp, brief EPSP** (peaks at 1 step) — each input spike has short-lived influence, emphasising precise timing. |
| `tauRef` | Refractory time constant `τ` of `ν` | `2` | `1.0` | Reset dip lasts ~2 steps. |
| `scaleRef` | Refractory magnitude, relative to `theta` | `2` | `2` | After a spike, `u` is pushed down by up to `2·theta = 20` — a strong reset (~2× threshold). |
| `tauRho` | Surrogate-gradient width (× `theta`) | `0.1` | `1` | Effective width `tauRho·theta = 1`: a **narrow** gradient window, nonzero only within ~1 unit of threshold. |
| `scaleRho` | Surrogate-gradient height | `0.1` | `1` | Peak gradient `scaleRho/(tauRho·theta) = 0.1` — a small per-spike gradient magnitude. |

## Why `theta = 10` trains at 2 layers but not at 4

The surrogate gradient is a **narrow** bump of width `tauRho·theta = 1` centred on
`theta = 10`. A neuron only fires — and only passes gradient — when its membrane potential
`u` comes within ~1 of 10. Every spiking layer attenuates the drive, so with depth the
later layers' potentials fall further below threshold and **both firing and gradient decay
geometrically**.

Measured at initialisation on SHD `norm` (fresh model, `theta = 10`, `fc[0]` gradient norm):

| Depth | Hidden firing at init | `fc[0]` gradient |
|---|---|---|
| 2 hidden layers | weak but non-zero | ~8e-9 |
| 3 hidden layers | nearly silent | ~5e-13 |
| 4 hidden layers | silent from `h2` on | ~6e-17 |

At **2 layers** the gradient is tiny but *non-zero*: only `h1 → h2 → readout` lies between
input and loss, so the little activity that survives still yields a few output spikes and a
usable learning signal. With Nadam, `lr = 0.1`, and hundreds of epochs the network
bootstraps itself out of the near-silent regime — the flat start is temporary.

At **4 layers** spikes die after `h1` (h2–h4 emit nothing), so the readout is silent for
*every* input: the loss is constant and the gradient reaching `fc[0]` is ~1e-17 —
effectively zero. There is no signal to climb out of, so no number of epochs helps. The two
extra layers of geometric attenuation turn a marginal-but-workable 2-layer gradient into a
dead 4-layer one. (This is the "flat training curve" symptom.)

### Fix — lower `theta`

Dropping the threshold both raises firing (neurons reach threshold more easily) and, because
the refractory reset (`scaleRef·theta`) and the gradient window (`tauRho·theta`) both scale
*with* `theta`, keeps the neuron self-consistent — so `theta` alone suffices. A finer sweep
at 4 layers shows `theta = 2` is the depth-stable choice: firing stays roughly constant
across all four layers and the gradient is healthy.

| theta | h1 | h2 | h3 | h4 | `fc[0]` gradient |
|---|---|---|---|---|---|
| 10 | 0 | 0 | 0 | 0 | 6e-17 (dead) |
| 3 | 0.89 | 0.35 | 0.12 | 0.04 | 1e-2 |
| **2** | **1.60** | **1.42** | **1.33** | **1.23** | **0.43 (healthy)** |
| 1.5 | 2.21 | 2.45 | 2.62 | 2.68 | 1.26 (firing grows → risks saturation) |

`theta ≥ 2.5` decays back toward silence with depth; `theta ≤ 1.5` over-drives (activity
grows with depth). The four `code_moreLayers` 4-layer scripts therefore set **`theta = 2`**,
and **only** `theta` is changed — `tauSr` stays at 1 (raising it would boost firing but smear
spike timing, undermining the temporal-coding question), and the refractory and
surrogate-gradient parameters are left untouched. The shallower 2-layer scripts in `code/`
keep the original `theta = 10`.

## Source

- `slayerSNN/slayer.py` — `_calculateAlphaKernel` (kernels), `calculateRefKernel`
  (`mult = −scaleRef·theta`), and `_spikeFunction.forward/backward`
  (`pdfTimeConstant = tauRho · theta`, `spikePdf = scaleRho / pdfTimeConstant · exp(...)`).
- Shrestha & Orchard, *SLAYER: Spike Layer Error Reassignment in Time*, NeurIPS 2018.
