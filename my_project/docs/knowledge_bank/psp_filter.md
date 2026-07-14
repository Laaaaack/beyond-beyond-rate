# PSP Filtering in slayerSNN (slayerPytorch)

## Overview

slayerSNN provides two mechanisms for post-synaptic potential (PSP) filtering
of spike trains. Both convolve a temporal kernel with the spike input to produce
a continuous current signal, but they differ in whether the kernel is fixed or
learnable.

| Mechanism | Class | Kernel | Learnable | Gradient to kernel |
|---|---|---|---|---|
| `slayer.psp(spike)` | `_pspFunction` | Fixed (`srmKernel`) | No | `None` (explicitly) |
| `slayer.pspFilter(...)` | `_pspFilter` | Stored as `weight` | Yes | Standard autograd |

## 1. Fixed PSP: `slayer.psp(spike)`

Uses the `srmKernel`, an alpha-function kernel computed at layer construction
from the neuron descriptor's `tauSr` parameter:

```
eps(t) = t / tauSr * exp(1 - t / tauSr)
```

The kernel is sampled at every `Ts` ms from `t = 0` until the value drops below
`EPSILON = 0.01` past the peak. It is stored as a 1-D buffer (not a parameter),
registered via `register_buffer('srmKernel', ...)`.

**Key properties:**

- The kernel is applied via `slayerCuda.conv`, a custom CUDA kernel that performs
  true **convolution** (not cross-correlation). The kernel is stored in
  time-forward order: `srmKernel = [eps(0), eps(Ts), eps(2*Ts), ...]`.
- The peak of the alpha function is at `t = tauSr`, i.e., at index `tauSr / Ts`.
  For example, with `tauSr = 10` and `Ts = 1`, the peak is at index 10.
- **Not differentiable with respect to the kernel.** The `_pspFunction.backward`
  method explicitly returns `gradFilter = None`. Only the gradient with respect
  to the input spike tensor is propagated (via `slayerCuda.corr`).
- The output is the causal response: a spike at time `t` produces the alpha
  waveform starting at `t`.

## 2. Learnable PSP Filter: `slayer.pspFilter(nFilter, filterLength, filterScale)`

Returns a `_pspFilter` instance, which is a subclass of `nn.Conv3d`:

```python
class _pspFilter(nn.Conv3d):
    def __init__(self, nFilter, filterLength, Ts, filterScale=1):
        super().__init__(
            in_channels=1,
            out_channels=nFilter,
            kernel_size=(1, 1, filterLength),
            bias=False,
        )
        self.Ts = Ts
        self.pad = ConstantPad3d(padding=(filterLength-1, 0, 0, 0, 0, 0), value=0)

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        inPadded = self.pad(input.reshape((N, 1, 1, -1, Ns)))
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape((N, -1, H, W, Ns))
```

**Key properties:**

- The weight tensor has shape `(nFilter, 1, 1, 1, filterLength)`.
- **Left-padding** of `filterLength - 1` zeros ensures the convolution is
  **causal**: the output at time `t` depends only on inputs at times `<= t`.
- `F.conv3d` performs **cross-correlation**, not convolution. This means the
  stored weight tensor is the **time-reversed** version of the desired impulse
  response:

  ```
  stored_weight = [h(T_max), h(T_max - Ts), ..., h(Ts), h(0)]
  impulse_response = flip(stored_weight) = [h(0), h(Ts), ..., h(T_max)]
  ```

- The output is scaled by `Ts` (the simulation time step), matching the
  `slayerCuda.conv` convention used by the fixed PSP.
- **Fully differentiable.** Since it uses standard `F.conv3d`, PyTorch autograd
  computes gradients with respect to `self.weight` automatically.
- Default initialization uses PyTorch's `nn.Conv3d` init (Kaiming uniform).
  To mimic the fixed PSP, the user must manually seed the weights (see below).

## 3. Storage Convention (Correlation vs. Convolution)

This is the most important subtlety when working with `pspFilter`.

**The fixed PSP** (`slayerCuda.conv`) performs true convolution:

```
output[t] = sum_k kernel[k] * input[t - k]
```

So the kernel is stored in natural time-forward order.

**The learnable PSP** (`F.conv3d`) performs cross-correlation:

```
output[t] = sum_k weight[k] * input[t + k - (filterLength - 1)]
```

(with left-padding of `filterLength - 1` to maintain causality).

This means a spike at time `t_spike` produces a response where `weight[-1]`
appears at `t_spike`, `weight[-2]` at `t_spike + 1`, and so on. In other words,
the impulse response is the **reverse** of the stored weight vector.

**To seed `pspFilter` with a known impulse response `h`:**

```python
stored = np.flip(h)  # reverse for cross-correlation
psp_filter.weight.data = torch.FloatTensor(stored).reshape(weight_shape)
```

**To read back the impulse response from stored weights:**

```python
stored = psp_filter.weight.data.squeeze().cpu().numpy()
impulse_response = np.flip(stored)  # un-reverse
```

### Verified example

With `weight = [1, 2, 3, 4, 5]` (stored) and a spike at `t = 2`:

```
Input:  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Output: [0, 0, 5, 4, 3, 2, 1, 0, 0, 0]  (x Ts)
```

The impulse response is `[5, 4, 3, 2, 1]` = `flip(stored)`, starting at the
spike time.

## 4. Seeding with an Alpha Function

The `_initialize_alpha_filter` method in `isi_tau.ipynb` seeds the learnable
PSP with an alpha-function shape:

```python
tau_init = 50 * MS          # 50 ms in seconds
Ts_sec = Ts * MS            # Ts in seconds (e.g. 1e-3 for 1 ms)
filt_len = filterLength     # number of taps

alpha_kernel = [t / tau_init * exp(1 - t / tau_init)
                for t in arange(0, filt_len * Ts_sec, Ts_sec)]
alpha_kernel /= max(abs(alpha_kernel))        # normalize peak to 1
stored = np.flip(alpha_kernel)                 # reverse for correlation
```

The alpha function peaks at `t = tau_init`. With `tau_init = 50 ms` and
`Ts = 1 ms`, the peak is at index 50. This means `filterLength` must be
**significantly larger than `tau_init / Ts`** (e.g., 150 taps for a 50 ms tau)
to avoid truncating the kernel at or before its peak.

## 5. Estimating Tau from the Learned Filter

### Argmax method (`get_tau` in `isi_tau.ipynb`)

```python
weights = np.flip(stored_weights)      # recover impulse response
peak_idx = np.argmax(np.abs(weights))
estimated_tau = peak_idx * Ts          # in ms (with Ts in ms)
```

The alpha function has its peak at `t = tau`, so `peak_idx * Ts` directly
gives tau.

**Note:** The original `get_tau()` in `isi_tau.ipynb` uses a factor of `3`:
`estimated_tau = 3 * peak_idx * Ts * MS`. This 3x multiplier has no physical
basis for the alpha function and over-reports tau by a factor of 3. The correct
formula is simply `peak_idx * Ts` (in the same time units as Ts).

**Limitation:** Argmax is discrete. With `Ts = 1 ms`, tau can only change in
steps of 1 ms. Small, continuous changes to the filter shape do not affect the
argmax until a neighbouring tap exceeds the current peak.

### Centre-of-mass method (continuous alternative)

```python
weights = np.flip(stored_weights)
abs_w = np.abs(weights)
com_idx = (abs_w * arange(len(weights))).sum() / abs_w.sum()
estimated_tau = com_idx * Ts    # in ms
```

This varies smoothly as the filter weights update, making it more suitable for
tracking gradual learning. However, it does not directly correspond to the
peak of the impulse response and is sensitive to the tail shape.

## 6. Gradient Flow Through the Network

In a SLAYER SNN with a learnable PSP filter, the forward path is:

```
input spikes --> pspFilter (F.conv3d, learnable) --> fc1 (dense) --> spike()
            --> psp (fixed, slayerCuda.conv) --> fc2 (dense) --> spike() --> output
```

Gradient flows backward through:

1. **`spike()` backward:** surrogate gradient
   `dS/du = scaleRho / tauRho * exp(-|u - theta| / tauRho)`.
   With `tauRho = 1, theta = 1`, this is a narrow exponential window around
   threshold. Membrane potentials far from threshold receive near-zero gradient.
2. **`fc1` backward:** standard linear layer gradient.
3. **`pspFilter` backward:** standard `F.conv3d` gradient. Autograd computes
   `d(loss)/d(weight)` normally.

The surrogate gradient at step 1 is the bottleneck. If `tauRho` is too small,
the gradient window is too narrow and very little signal propagates to the
filter. Increasing `tauRho` or `scaleRho` widens the surrogate window.

## 7. Common Pitfalls

1. **Filter too short for tau_init.** If `filterLength <= tau_init / Ts`, the
   alpha peak is truncated at the last index and cannot move rightward (toward
   longer tau). The argmax appears frozen at the boundary.

2. **Forgetting to reverse.** `F.conv3d` does correlation. The stored weight
   is the time-reverse of the impulse response. Seeding without `np.flip`
   produces a time-reversed (anti-causal) response.

3. **The 3x multiplier bug.** `get_tau()` in the original code uses
   `3 * peak_idx * Ts * MS`. The factor 3 is not justified by the alpha
   function and should be removed.

4. **Confusing `psp` and `pspFilter`.** `psp()` uses a fixed, non-learnable
   CUDA kernel. `pspFilter()` uses a learnable `Conv3d`. They produce identical
   output when the `pspFilter` is seeded with `flip(srmKernel)`, but only
   `pspFilter` propagates gradients to the kernel weights.
