# SLAYER `SIM_PARAMS`: meaning of `Ts` and `tSample`

`SIM_PARAMS = {"Ts": ..., "tSample": ...}` is the SLAYER simulation
descriptor. Both fields are expressed in the same time unit (typically
milliseconds in this project).

## `Ts` — simulation time step

The width of one time bin. With `Ts = 1` (ms), every step in the
discrete-time SLAYER simulation corresponds to 1 ms of real time, so
quantities such as ISI (in ms) and `isi_steps` (in bins) are numerically
identical.

## `tSample` — total simulated duration per input

The total length of the time window that SLAYER expects to simulate for a
single input sample. The number of internal time steps is
`Ns = round(tSample / Ts)`. Conceptually, this is "how long is one
example" from SLAYER's point of view.

`tSample` should equal the time-axis length `T` of the input spike
tensors:

```
tSample == X.shape[-1]  (when Ts == 1)
```

It is **not** a free hyper-parameter — it must be aligned with the data.

### Where `tSample` is actually used

- Sizes the time axis of internal SLAYER buffers (PSP filter, refractory
  kernels, spike-time targets).
- Enters the loss configuration. For example, `ProbSpikes` / `SpikeMax`
  normalises spike counts by the simulation duration; if `tSample`
  disagrees with the input tensor's time axis, the target spike-count
  scaling is mis-calibrated.
- Anchors the absolute firing-rate semantics: a sample with `K` spikes
  over `tSample` ms has rate `K / tSample` (in 1/ms), so a wrong
  `tSample` silently rescales rate-based quantities.

### What `tSample` does *not* do

- It does **not** crop or pad the input tensor. SLAYER's convolutional
  ops along the time axis follow the actual length of the tensor passed
  in. So a mismatch between `tSample` and `X.shape[-1]` does **not**
  raise an error — the network still runs, but with miscalibrated
  loss / rate semantics. Training can still succeed (argmax over
  spike counts is monotone), which is why such bugs are easy to miss.

## Concrete settings used in this project

| Experiment | `Ts` | `tSample` | Data `T` | Notes |
|---|---|---|---|---|
| SHD (realistic) | 1 ms | 200 | 100 (zero-padded → 200) | Mismatch: dataset has 100 bins, padded to match `tSample`. Effective resolution ≈ 14 ms/bin. |
| SSC (realistic) | 1 ms | 200 | 200 | Matches by design; follows original Beyond Rate pipeline. |
| ISI (synthetic) | 1 ms | 1000 | 1000 (after fix) | Originally generated at `T=10000` while `tSample=1000` — silently inconsistent. Fixed by regenerating the dataset at `TIME_STEPS=1000`. |
| CCISI (synthetic) | 1 ms | 1000 | 1000 (after fix) | Same fix as ISI: regenerate at `TIME_STEPS=1000` so loss/rate semantics match the simulated 1 s window. |

## Rule of thumb

When in doubt, treat `tSample` as a contract: it must agree with both
the input tensor's time axis **and** the physical sample duration the
loss assumes. Pick one source of truth (the data) and propagate it
everywhere.
