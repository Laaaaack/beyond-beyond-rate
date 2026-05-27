# Sample-Rate (Ts) Behavior in SLAYER

## TL;DR

In SLAYER, `Ts` is **not** a free discretization knob. Changing it (e.g. from
1 ms to 0.5 ms) cascades into many places — spike magnitudes, kernel sampling,
delay-parameter semantics, gradient scaling — and even after careful
re-normalization, trained models can land in qualitatively different solutions
(rate-coded vs timing-coded hidden representations).

**Practical recommendation:** treat Ts as a fixed part of the experimental
setup, not a hyperparameter. The headline results in this project use Ts=1 ms
to match the original Beyond Rate paper. Use 0.5 ms only as a robustness check
and document any Ts-dependent variation that survives the fixes below.

---

## What SLAYER actually does with Ts

Verified empirically (see end of file for the snippets). At Ts in `SIM_PARAMS`:

1. **Emitted spikes are valued `1/Ts`**, not 0/1. `slayer.spike(...)` produces
   tensors whose nonzero entries equal `1/Ts` (so 1.0 at Ts=1, 2.0 at Ts=0.5).
   This is SLAYER's way of representing a Dirac delta in discretized form so
   that PSP integrals are Ts-invariant.

2. **PSP integration is multiplied by Ts.** `_pspFilter.forward` and
   `_pspFunction.forward` both apply `output = conv(input, filter) * Ts`. This
   is a Riemann-sum integration: combined with `1/Ts`-valued spikes, the PSP
   integral per spike is Ts-invariant.

3. **Alpha / SRM / refractory kernels are sampled at Ts intervals.** The
   continuous-time kernel shape is the same; the discrete kernel has more taps
   at smaller Ts but the same total integral. `tauSr` is in the same units as
   `Ts` (so `tauSr=1` with Ts in ms means a 1 ms decay).

4. **`slayer.delay()` parameters are in milliseconds, not time bins.** The
   parameter `delay.delay` is stored as a float in the same units as Ts; the
   internal shift is `floor(delay / Ts)` bins. SLAYER's own docstring confirms
   this ("initialized uniformly between 0 ms and 1 ms"). Naming a constant
   `MAX_DELAY = 20  # time steps` and expecting it to mean "20 bins = 10 ms at
   Ts=0.5" is wrong — it means 20 ms.

5. **The delay gradient is scaled by `1/Ts`.** From the backward of
   `_delayFunction`: `diffFilter = [-1, 1] / Ts`. At Ts=0.5, delays receive
   ~2× larger gradients than at Ts=1 for the same loss surface.

6. **Spike-loss terms include a `* Ts` factor.** `spikeTime` returns
   `0.5 * sum(error**2) * Ts`; `numSpikes` scales spike counts by `Ts`. So the
   loss magnitude itself is Ts-dependent unless the LR or loss is renormalized.

---

## Bugs we found and fixed

### Bug 1: Binary input not scaled to `1/Ts`

**Symptom.** At Ts=1, clean accuracy and perturbation curves looked sensible.
At Ts=0.5, ISI tau gave a flat perturbation curve (~0.92 at every f), suggesting
the hidden layer had collapsed to a rate code. CCISI showed asymmetric
behavior (tau got more timing-sensitive, but inconsistently).

**Root cause.** Our HDF5 datasets store binary 0/1 spike trains. SLAYER's
internal convention is `1/Ts`-valued spikes (point 1 above). At Ts=1 the two
happen to coincide; at Ts=0.5 the input PSP comes out half-magnitude relative
to SLAYER's semantics, which the network has to compensate for with larger
fc1 weights — pushing it into a different optimization regime.

**Fix.** Scale the input by `1/Ts` inside `_prepare_input` in all four
`2000_rate/` notebooks:

```python
return x.float().to(device) / self.slayer.simulation['Ts']
```

Verified: a single binary input spike now produces a PSP peak of 1.0 at both
Ts=1 and Ts=0.5.

### Bug 2: `MAX_DELAY` doubled when porting to Ts=0.5

**Symptom.** The CCISI delay (0.5 ms) model had unusually low val_loss
(~0.046) and was **less** sensitive to hidden perturbation than the
tau-only model — the opposite of the expected pattern (delay models should
rely more on timing, not less).

**Root cause.** The 0.5 ms notebooks set `MAX_DELAY` to 2× the 1 ms value
("30 time steps = 15 ms at Ts=0.5" / "20 time steps = 10 ms at Ts=0.5"),
intending to preserve the delay budget in real time. But the delay parameter
is in ms (point 4 above), so 30 means 30 ms, not 15 ms. The 0.5 ms models had
**2× the delay budget** of the 1 ms baseline.

**Fix.** Set `MAX_DELAY` to the same numerical value as the 1 ms baseline
(15 for ISI, 10 for CCISI). Updated comments to say "milliseconds" instead of
"time steps".

**Caveat:** Fixing this changed the trained model but did **not** restore the
expected "delay model is more timing-sensitive than tau model" relationship at
Ts=0.5. The CCISI delay (0.5 ms) drop went from 0.20 → 0.10 (i.e. became
**more** rate-coded), not less. So the bug was real and worth fixing for
apples-to-apples comparison, but it isn't the dominant cause of the Ts=0.5
anomaly.

---

## What still doesn't line up after both fixes

CCISI hidden perturbation results (each cell is mean accuracy):

| f   | tau (1 ms) | delay (1 ms) | tau (0.5 ms) | delay (0.5 ms) |
|-----|-----------:|-------------:|-------------:|---------------:|
| 0.0 | 0.996      | 0.994        | 0.987        | 0.989          |
| 0.5 | 0.962      | ~0.93        | 0.828        | 0.962          |
| 1.0 | 0.898      | 0.745        | 0.635        | **0.885**      |
| drop | 0.10      | 0.25         | 0.35         | **0.10**       |

Expected pattern (matches ISI and CCISI 1 ms): `delay drop > tau drop`. At
Ts=0.5 it reverses for CCISI — delay model is **more** robust to perturbation
than tau model. Best guesses for why:

- **Loss-landscape shift.** Points 5 and 6 above mean the surface optimizers
  see at Ts=0.5 is genuinely different from Ts=1. SNNs are known to find
  qualitatively different codes (rate vs timing) depending on the landscape.
- **2× more time bins per sample.** Same wall-clock simulation (1000 ms) but
  the forward/backward pass operates on 2× as many bins. Surrogate gradients
  accumulate differently across them.
- **Dataset stochasticity.** Each `*_data_gen.py` uses `ProcessPoolExecutor`
  without seeding the workers, so the 1 ms and 0.5 ms HDF5 files have
  different spike placements even for the same `(firing_rate, ISI)` pairs.

None of these is a "bug" we can patch; they're inherent to the framework /
experimental design. They explain why even apparently safe changes to Ts can
shift which solution the network converges to.

---

## Checklist for any future Ts change

If someone introduces a Ts other than 1 ms (or changes it again), audit these
sites at minimum:

- [ ] Input data values: do they need to be scaled by `1/Ts` to match SLAYER's
      spike convention?
- [ ] `MAX_DELAY` and any other delay-related constants: are they in ms (most
      likely) or in bins? Did the port preserve the **real-time** budget?
- [ ] PSP filter length (`pspFilter(filterLength=...)`): should scale with
      `1/Ts` to keep the same real-time coverage. (50 taps at Ts=1 = 100 taps
      at Ts=0.5, both covering 50 ms.)
- [ ] Refractory / surrogate-gradient time constants in `LIF_PARAMS` (`tauSr`,
      `tauRho`, `tauRef`): values are in the same unit as Ts.
- [ ] Perturbation / spike-detection thresholds: do they use `> 0.5` (works
      at any Ts since `1/Ts > 0.5` for sane Ts) or hard-code 1?
- [ ] Loss-function scaling: SLAYER's spike losses include a `* Ts` factor;
      LRs may need rescaling to keep effective step sizes comparable.
- [ ] Dataset regeneration: spike-train placement RNG state should be seeded
      in workers, or use the same generated dataset across Ts variants by
      down/upsampling deterministically.

---

## Empirical verification snippets

Quick standalone tests used to ground the claims above. Run from a directory
where `slayerSNN` is importable.

```python
import torch, slayerSNN as snn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LIF = {'type': 'SRMALPHA', 'theta': 1, 'tauSr': 1,
       'tauRho': 1, 'tauRef': 1, 'scaleRef': 2, 'scaleRho': 1}

# 1. Spike value emitted by slayer.spike is 1/Ts
for Ts, T in [(1.0, 1000), (0.5, 2000)]:
    s = snn.layer(LIF, {'Ts': Ts, 'tSample': 1000}).to(device)
    mp = torch.zeros(1, 1, 1, 1, T, device=device); mp[..., ::20] = 5.0
    spk = s.spike(mp)
    print(f'Ts={Ts}: spike value = {spk[spk>0.5].mean().item()}')   # 1.0 vs 2.0

# 2. Single binary input spike: input PSP is halved at Ts=0.5
for Ts, T in [(1.0, 1000), (0.5, 2000)]:
    s = snn.layer(LIF, {'Ts': Ts, 'tSample': 1000}).to(device)
    x = torch.zeros(1, 1, 1, 1, T, device=device); x[..., int(round(10/Ts))] = 1.0
    print(f'Ts={Ts}: psp peak = {s.psp(x).max().item()}')           # 1.0 vs 0.5

# 3. Delay parameter is in ms (= same units as Ts)
for Ts, T, d_val in [(1.0, 1000, 5), (0.5, 2000, 5), (0.5, 2000, 20)]:
    s = snn.layer(LIF, {'Ts': Ts, 'tSample': 1000}).to(device)
    d = s.delay(1).to(device); d.delay.data.fill_(float(d_val))
    x = torch.zeros(1, 1, 1, 1, T, device=device); x[..., 100] = 1.0
    out = (d(x)[0,0,0,0] > 0.5).nonzero().flatten().cpu().numpy()
    print(f'Ts={Ts} delay={d_val}: shift = {(out[0]-100)*Ts} ms')   # always = d_val ms
```
