# Phase 1–4 Refactor: Train-at-f / Eval-at-f Protocol

This branch (`version_2`) ports every training notebook in `my_project/code/`
to the **train-at-f / eval-at-f** protocol: for each perturbation level *f*
(or σ, p_d) we train one model from scratch with the perturbation active at
the 1st hidden layer during training, then evaluate it at the same level —
mirroring how the original *Beyond Rate* paper runs its input-perturbation
sweeps, just moved to the hidden layer.

The doc below is organised around the refactor itself. Section 7 keeps the
underlying bug analysis so the *why* behind the STE wrapper and the
delay-routing refactor stays visible.

---

## 1. Goal

For every notebook in `my_project/code/`:

- Training forward pass takes the perturbation level as an argument
  (`forward(x, f)` / `forward(x, sigma)` / `forward(x, p_d)`).
- For each perturbation level in the sweep:
  - Instantiate a **fresh** model (new seed-derived init).
  - Train end-to-end with perturbation applied at the 1st hidden layer
    output on **every batch**.
  - Save the checkpoint with the level baked into the filename.
  - Evaluate on the test set at the **same** level with `num_repeats` for
    error bars.
- Aggregate into a sweep result identical in shape to the previous
  test-time-only sweep, so the downstream `result_visualization.ipynb`
  notebooks continue to work.

Two blockers must be resolved before this protocol gives interpretable
curves:

1. **Issue 1** (Section 7.1): perturbations that go through numpy or
   `@torch.no_grad()` sever the autograd graph. Without an STE wrapper,
   `fc1` / `psp_filter` / `delay1` receive zero gradient and the curve
   collapses to a cliff+plateau.
2. **Issue 2** (Section 7.2): in several notebooks `_first_hidden` ends
   with `if use_delay: x = self.delay1(x)`. Cosmetic in the current
   slayer (floors the delay, output stays binary), but it routes the
   delay learning *downstream* of the perturbation hook, which makes
   the perturbation site harder to audit. We refactor it out while we
   are touching these cells anyway.

**Performance requirement (binding for all experiments):** every
`*_hidden_batch` helper that is called inside `forward` during training
**must** be a GPU-vectorised kernel that keeps the tensor on its input
device — no `.cpu().numpy() → ... → torch.from_numpy(...)` round-trip.
The numpy reference implementation may stay in the notebook as
documentation but must not be invoked from the training/eval path.
Rationale: the round-trip is what produced the ~3× wall-clock penalty
at `p > 0` (Section 7.1.3); with per-batch perturbation now active on
every training step, this cost compounds across all 6 levels × all
variants and is the dominant bottleneck if left in. Section 3.7 below
specifies the kernel template.

---

## 2. Current state per notebook

| Notebook | `forward(x, f)`? | Hook sees binary? | STE? | Action |
|---|---|---|---|---|
| `synthetic/isi/isi_tau.ipynb` | **yes** | yes (`_first_layer` ends with `spike`) | **yes** | **done (2026-05-19) — refactored & re-run on `version_2`** |
| `synthetic/isi/isi_delay.ipynb` | **yes** | yes (delay1 before fc1; delay2 in `_second_layer`) | **yes** | **done (2026-05-19) — refactored & re-run on `version_2`** |
| `synthetic/ccisi/ccisi_tau.ipynb` | **yes** | yes (`_first_layer` ends with `spike`) | **yes** | **refactor done (2026-05-19) on `version_2` — STE + GPU kernel landed, pending re-run** |
| `synthetic/ccisi/ccisi_delay.ipynb` | **yes** | yes (delay1 before fc1; delay2 in `_second_layer`) | **yes** | **refactor done (2026-05-19) on `version_2` — STE + GPU kernel landed, pending re-run** |
| `synthetic/coincidence/coin_tau.ipynb` | no | yes | n/a | add train-at-f path + STE |
| `synthetic/coincidence/coin_delay.ipynb` | no | yes (delay1 lives in `_second_layer`, after hook) | n/a | add train-at-f path + STE |
| `realistic/shd/shd_train.ipynb` | **yes** | yes (delay1 moved to start of `_second_hidden_and_output`, Option B) | **yes** | **refactor done (2026-05-19) on `version_2` — STE + GPU kernel landed, pending re-run** |
| `realistic/ssc/ssc_train.ipynb` | no | **no** (delay after spike) | n/a | add train-at-f path + STE + delay refactor + GPU kernel |
| `perturbation/jitter/jitter_train.ipynb` | **yes** | yes (delay1 moved to start of `_second_hidden_and_output`, Option B) | **yes** | **refactor done (2026-05-19) on `version_2` — STE + GPU kernel landed, pending retrain for σ>0 (cached `data/jitter_*_sigma{1,3,...}.pt` are stale)** |
| `perturbation/shift/shift_train.ipynb` | **yes (buggy)** | no | **missing** | add STE + delay refactor + GPU kernel; retrain σ>0 |
| `perturbation/deletion/deletion_train.ipynb` | **yes (buggy)** | no | **missing** | add STE + delay refactor + GPU kernel; retrain pd>0 |
| `perturbation/inverse/inverse_train.ipynb` | no | no | n/a | add train-at-f path + STE + delay refactor + GPU kernel; mirror for reversal |

`isi_tau` currently uses a GPU-vectorised `perturb_hidden_batch` decorated
with `@torch.no_grad()`. That decorator severs the graph just like the
numpy round-trip does — for eval-only it is fine, but the moment the call
moves into the training forward pass it needs the STE wrapper too. Same
applies wherever `perturb_hidden_batch` or any other `*_hidden_batch`
helper is used.

---

## 3. The common refactor pattern

The same six edits apply to every notebook (with minor naming differences:
`f` for synthetic / SHD / SSC / inverse, `sigma` for jitter / shift, `p_d`
for deletion). Apply them in this order inside each notebook so the
diagnostics catch problems early.

### 3.1 Pull the perturbation level into a config block

At the top configuration cell, replace the existing `F_VALUES` /
`SIGMA_VALUES` / `P_D_VALUES` list with the **training** sweep:

```python
PERT_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # name varies per notebook
NUM_REPEATS = 3
```

For SHD / SSC and the perturbation/* notebooks that already have a
multi-sigma loop, the list already exists — just rename for consistency
and confirm σ=0 is in it (control baseline).

### 3.2 Add an `_apply_perturbation` helper on the model

Single source of truth for the STE wrapper. One method per model class:

```python
def _apply_perturbation(self, hidden: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return hidden
    perturbed = perturb_hidden_batch(hidden, p)          # or jitter_hidden_batch, etc.
    return hidden + (perturbed - hidden).detach()
```

Forward value is `perturbed`; backward gradient is the identity through
`hidden`. This is the only change that actually unblocks training.

### 3.3 Make `forward` take the perturbation level

Collapse `forward` and `forward_with_hidden_perturbation` into a single
method that always goes through the hook:

```python
def forward(self, x: torch.Tensor, p: float = 0.0) -> torch.Tensor:
    x = self._prepare_input(x)
    hidden = self._first_layer(x)            # or _first_hidden(x)
    hidden = self._apply_perturbation(hidden, p)
    return self._second_layer(hidden)        # or _second_hidden_and_output(hidden)
```

Remove the old eval-only `forward_with_hidden_perturbation`. `p=0.0`
preserves the previous unperturbed behaviour exactly.

For **inverse_train**, keep `forward_with_hidden_reversal` as a separate
method (reversal is a different operation, not parameterised by *f*), but
route it through the same STE wrapper and add a `f` argument too so the
combined "reverse + relocate-f-fraction" case used in the current eval
loop still works at train time.

### 3.4 Move the delay out of `_first_hidden` where it follows `spike`

This is the Issue 2 fix. Anti-pattern in `jitter_train`, `shift_train`,
`deletion_train`, `inverse_train`, `shd_train`, `ssc_train`:

```python
def _first_hidden(self, x):
    x = self.slayer.spike(self.fc1(self.slayer.psp(x)))
    if self.use_delay:
        x = self.delay1(x)        # <- runs AFTER spike, before the hook
    return x
```

Pick one of two clean routings (match what the synthetic `*_delay`
notebooks already do):

- **Option A — delay on the input side** (mirrors `ccisi_delay`,
  `isi_delay`): move `self.delay1` to the beginning of `_first_hidden`,
  acting on `x` before `psp + fc1 + spike`. The "delay1" then represents
  per-input-neuron axonal delay; the hook receives binary spikes
  straight out of `slayer.spike`.
- **Option B — delay at the start of `_second_layer`** (mirrors
  `coin_delay`): drop `delay1` from `_first_hidden` entirely and apply
  it as the first step of `_second_hidden_and_output(hidden)`, before
  `psp + fc2 + spike`.

Both routings are equivalent in expressive power; **Option B is the
cleaner port for the 2-hidden-layer SHD-family notebooks** because the
existing `delay1` was already attached to the hidden-layer output side.
For the 1-hidden-layer ISI / coin / ccisi delay variants, Option A is
already in place — no change needed beyond Section 3.2 / 3.3.

After the move, add a sanity assertion (only inside the eval loop to
keep training fast):

```python
assert torch.unique(hidden).tolist() in ([0.0], [1.0], [0.0, 1.0]), (
    "Hidden hook expects binary spikes; got non-binary values."
)
```

### 3.5 Rewrite the top-level run cell as a sweep over fresh models

Replace the single `train_model(...)` + sweep-at-eval block with:

```python
all_models, all_logs, all_results = {}, {}, {}

for p in PERT_VALUES:
    print(f"\n=== Training at p={p} ===")
    net, log = train_model(train_loader, val_loader, p=p, seed=SEED)
    torch.save(net.state_dict(), f"data/{MODEL_PREFIX}_p{p}.pt")

    result = test_with_repeats(net, test_loader, p=p, num_repeats=NUM_REPEATS)
    all_models[p], all_logs[p], all_results[p] = net, log, result
    print(f"p={p} | test acc = {result['mean']:.4f} ± {result['std']:.4f}")
```

`train_model` itself only needs three changes:

1. Accept `p: float = 0.0` as an argument.
2. In the training inner loop, call `net(x_batch, p=p)` instead of
   `net(x_batch)`.
3. In the validation inner loop, call `net(x_batch, p=p)` so the
   best-checkpoint selection sees the same perturbation it will be
   evaluated under.

Seed handling: re-seed inside `train_model` (already done in most
notebooks via `set_seed`) so each level gets the same starting weights —
otherwise random init noise confounds the curve.

### 3.6.5 GPU-vectorise the perturbation kernel

Replace any `*_hidden_batch` that routes through numpy with a fully
on-device kernel. Two templates cover the operators in this project.

**Template A — uniform-random spike relocation** (used by
`perturb_hidden_batch` in `isi_tau`, `isi_delay`, `ccisi_tau`,
`ccisi_delay`, `coin_tau`, `coin_delay`, `shd_train`, `ssc_train`,
`inverse_train`). For each (batch, neuron) draw a random sort key
over time bins, take the top-`num_to_move` spike positions to remove,
and the top-`num_to_move` empty positions to place new spikes:

```python
@torch.no_grad()
def perturb_hidden_batch(hidden_spikes, f):
    if f <= 0: return hidden_spikes
    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    n_spikes = is_spike.sum(dim=-1, keepdim=True)
    num_to_move = (n_spikes.float() * f).floor().long()

    key = torch.where(is_spike, torch.rand_like(x), torch.full_like(x, 2.0))
    rank = key.argsort(dim=-1).argsort(dim=-1)
    remove_mask = rank < num_to_move
    keep_mask = is_spike & ~remove_mask

    available = ~keep_mask
    key2 = torch.where(available, torch.rand_like(x), torch.full_like(x, 2.0))
    rank2 = key2.argsort(dim=-1).argsort(dim=-1)
    add_mask = rank2 < num_to_move

    return (keep_mask | add_mask).to(hidden_spikes.dtype).view(B, C, H, W, T)
```

Spike count per (batch, neuron) is preserved exactly.

**Template B — per-spike Gaussian operators with collisions** (used by
`jitter_hidden_batch` in `jitter_train`, by `shift_hidden_batch` in
`shift_train`, and by `deletion_hidden_batch` in `deletion_train` —
deletion is the special case where every spike's target is "out of
range" with probability `p_d`). For each spike, sample a target bin
from the operator's distribution and resolve collisions with a
priority-based tiebreaker:

```python
@torch.no_grad()
def jitter_hidden_batch(hidden_spikes, sigma, max_attempts=50):
    if sigma <= 0: return hidden_spikes
    B, C, H, W, T = hidden_spikes.shape
    x = hidden_spikes.view(B, C, T)
    is_spike = x > 0.5

    new_spikes = torch.zeros_like(is_spike)
    unplaced = is_spike.clone()
    t_idx = torch.arange(T, device=x.device).view(1, 1, T)
    inf_tensor = torch.full_like(x, float("inf"))

    for _ in range(max_attempts):
        if not unplaced.any(): break
        target = (t_idx + torch.randn_like(x) * sigma).round().long().clamp(0, T - 1)
        priority = torch.where(unplaced, torch.rand_like(x), inf_tensor)
        min_priority = inf_tensor.clone()
        min_priority.scatter_reduce_(-1, target, priority, reduce="amin", include_self=True)

        wins = unplaced & (priority == min_priority.gather(-1, target)) \
                        & ~new_spikes.gather(-1, target)
        scatter_out = torch.zeros((B, C, T), device=x.device, dtype=torch.uint8)
        scatter_out.scatter_add_(-1, target, wins.to(torch.uint8))
        new_spikes = new_spikes | (scatter_out > 0)
        unplaced = unplaced & ~wins

    # Fallback: any spike that never found a free target stays at its
    # original bin (matches the numpy reference, harmless under OR).
    new_spikes = new_spikes | unplaced
    return new_spikes.to(hidden_spikes.dtype).view(B, C, H, W, T)
```

Spike count is approximately preserved — the retry budget keeps the
preservation rate above what the numpy reference achieves in practice.

**Required for every notebook**, not optional. Status (2026-05-19):
`isi_tau`, `isi_delay`, `ccisi_tau`, `ccisi_delay`, `shd_train`,
`jitter_train` already use the on-device kernel. `coin_tau`,
`coin_delay`, `ssc_train`, `shift_train`, `deletion_train`,
`inverse_train` still need it ported.

### 3.6 Diagnostic block right after the sweep loop

Single cell, no plotting, prints the upstream-weight norm and a binary
check. Catches a re-occurrence of Issue 1 in five seconds rather than
after a full re-run:

```python
print(f"{'p':>6}  {'fc1.norm':>10}  {'psp.norm':>10}  hidden_unique")
for p, net in all_models.items():
    fc1_norm = sum(param.norm().item() for n, param in net.named_parameters()
                   if n.startswith("fc1.weight"))
    psp_norm = (net.psp_filter.weight.norm().item()
                if hasattr(net, "psp_filter") else float("nan"))
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.unsqueeze(2).unsqueeze(3).float().to(device)
        h = net._first_layer(x) if hasattr(net, "_first_layer") else net._first_hidden(x)
        unique = torch.unique(h).cpu().tolist()
    print(f"{p:>6.2f}  {fc1_norm:>10.4f}  {psp_norm:>10.4f}  {unique}")
```

Expected: `fc1.norm` varies between p=0 and p>0 rows (training actually
learned something different); `hidden_unique` is `[0.0, 1.0]` or
`[0.0]`. If two p>0 rows are bit-identical, the STE didn't take — go
back to Section 3.2.

---

## 4. Per-notebook steps

The repeated work is captured by Section 3. The notes below only list
**deltas from that template**. "Apply §3" means: do all of 3.1–3.6 with
no special handling.

### 4.1 Synthetic — ISI / CCISI / Coincidence (6 notebooks)

- `isi_tau`, `ccisi_tau`, `coin_tau`: apply §3. No delay layer, so 3.4
  is a no-op. `perturb_hidden_batch` is already GPU-vectorised; the STE
  wrapper goes around the call site, not around the function itself.
- `isi_delay`, `ccisi_delay`: apply §3. Delay routing is **already
  clean** (`delay1` is on the input side, `delay2` lives in
  `_second_layer`). 3.4 is a no-op. The optimiser already has three
  parameter groups (regular / tau / delay) — leave the LR multipliers
  intact.
- `coin_delay`: apply §3. `delay1` already sits in `_second_layer`, no
  delay-routing change needed.

Coincidence is the slowest synthetic task (1000-step inputs at 32 batch
size) — confirm Issue 3 (Section 7.3) is still resolved before running
the sweep; if `coin_dataset.mat` reports `Time steps: 1000` in the load
cell, it is.

### 4.2 Realistic — SHD / SSC (2 notebooks)

- Apply §3 in full. The §3.4 delay refactor matters here: pick
  **Option B** (move `self.delay1(...)` to the start of
  `_second_hidden_and_output`). The model has two hidden layers, so
  `delay2` continues to live at the start of layer 3 inside the same
  method; no change.
- Architecture stays at `Input → fc1 → spike → [hook] → delay1 →
  fc2 → spike → delay2 → fc3 → spike → output`.
- `tSample=200` padding is load-bearing for the SpikeRate readout
  window — do not touch it (see Section 7.3).
- SHD-whole at `Input(700) → 128 → 128 → 20`, 500 epochs × 6 levels is
  expensive (~5–6 h on the current box). Plan one variant at a time
  rather than dispatching the whole sweep.

### 4.3 Perturbation — jitter / shift / deletion / inverse (4 notebooks)

- `jitter_train`, `shift_train`, `deletion_train`: these **already have**
  the `forward(x, sigma)` / `forward(x, p_d)` plumbing from §3.3 and the
  training loop from §3.5. The work is §3.2 (add STE) and §3.4 (move
  `delay1` out of `_first_hidden` — Option B). After the STE is in
  place, retraining is mandatory for every σ > 0 / p_d > 0 — the
  current cached `data/jitter_*_sigma{1,3,...}.pt` files were trained
  with `fc1` frozen at random init (see Section 7.1.3) and must be
  discarded.
- `inverse_train`: apply §3 in full. Extra step: route
  `forward_with_hidden_reversal` through the same STE wrapper, since
  the train-at-f / eval-at-f protocol should treat reversal as a
  training-time intervention too. Keep `reverse=True/False` and
  `f` as orthogonal axes — the eval grid already sweeps both.

---

## 5. Sanity checks (run once per notebook, after the first p>0 sweep)

These catch every failure mode we have seen:

1. **Diagnostic block from §3.6 prints non-identical `fc1.norm` across
   p>0 rows.** If two rows match to four decimals, the upstream layer
   never got a gradient — Issue 1 is back.
2. **`hidden_unique` is `[0.0, 1.0]`.** Confirms the hook receives
   binary spikes; if it sees fractional values, the delay-routing
   refactor (§3.4) regressed.
3. **Sweep curve is smoothly decaying, not cliff+plateau.** The
   diagnostic signature of the autograd bug (Section 7.1.2).
4. **Wall-clock per epoch at p>0 is within ~1.5× of p=0.** A ~3×
   slowdown specifically at p>0 is the fingerprint of a CPU↔GPU
   round-trip surviving in `*_hidden_batch`. The §3.6.5 GPU kernel
   is now required for every notebook — if this check fails, the
   refactor is incomplete, not merely slow.
5. **Clean baseline (p=0) accuracy matches the existing test-time-only
   p=0 number to within a few %.** The p=0 forward path is unchanged;
   any large gap means the seed / data-loader / optimiser was
   inadvertently altered while editing.

---

## 6. Re-run plan and cost

Trained models live under each notebook's `data/` directory. Old
checkpoints from the test-time-only protocol use names like
`isi_tau_trained.pt` (single file) and need to be replaced with the
per-p set `isi_tau_p0.0.pt`, `isi_tau_p0.2.pt`, …

| Family | Per-model time (current box) | Models | Total |
|---|---|---|---|
| ISI (tau, delay) | ~5 / 7 min | 6 levels × 2 = 12 | ~75 min |
| CCISI (tau, delay) | similar to ISI | 12 | ~80 min |
| Coincidence (tau, delay) | similar to ISI | 12 | ~80 min |
| SHD (whole, part, norm × tau, delay) | 16–50 min depending on variant | 6 × 6 = 36 | one day each variant |
| SSC (part, norm × tau, delay) | similar to SHD | 24 | similar |
| Jitter / shift / deletion (× whole/part/norm × delay/nodelay) | 16–50 min | already sweeps 6 levels per variant | retrain σ>0 / pd>0 = 5 per variant |
| Inverse (× delay/nodelay) | 16–50 min | 6 × 2 = 12 | one afternoon |

Order:

1. **ISI tau** first — fastest, tightest signal, smallest blast radius
   if the refactor template is wrong.
2. **ISI delay**, then **CCISI / Coincidence** in either order — they
   share the refactor template, so any issue surfaces once and gets
   fixed across all of them.
3. **Jitter / shift / deletion** next — they only need §3.2 + §3.4;
   the existing curves under `log/` give a clean before/after to
   verify the fix flipped the cliff+plateau into smooth decay.
4. **Inverse**.
5. **SHD**, then **SSC**. Highest wall-clock cost — only run after
   the template is validated end-to-end on at least one synthetic
   notebook and one perturbation/* notebook.

Discard old checkpoints **only** after the new ones land
(`mv data/ data.bak/` rather than `rm`).

---

## 7. Reference: original bug analysis (preserved)

The diagnoses below are what the refactor in Section 3 is responding to.
Kept verbatim except for light reformatting.

### 7.1 Issue 1 — Hidden-layer perturbation severs the autograd graph during training

#### 7.1.1 Symptom

When a perturbation function applied to hidden-layer spikes goes through
numpy (`.detach().cpu().numpy() → ... → torch.from_numpy(...)`) or
`@torch.no_grad()`, the returned tensor is a fresh leaf with no edge in
the autograd graph. If that tensor is fed into the rest of the forward
pass during **training**, `loss.backward()` propagates gradients only
through layers downstream of the perturbation site. All upstream layers
(e.g. `fc1`, `psp_filter`, `delay1`) receive **zero gradient** for any
perturbation level > 0 and stay frozen at their initialization for the
entire run.

This was previously safe because perturbation was applied only at
evaluation (under `torch.no_grad()`). The train-at-f / eval-at-f
protocol moves perturbation into the training forward pass, which is
when the bug becomes silent-but-fatal.

#### 7.1.2 Diagnostic signature: cliff + plateau

Bug-affected sweep curves share the same shape — a sharp cliff at the
first non-zero perturbation level, then a flat plateau across all
higher levels. Clean sweeps decay gradually.

| Notebook | bug? | curve |
|---|---|---|
| `jitter_part_nodelay`     | yes | σ=0 0.380 → σ=1 0.156 → σ=3 0.145 → σ=5 0.146 → σ=10 0.158 → σ=17 0.182 → σ=25 0.174 |
| `shift_whole_delay`       | yes | σ=0 0.864 → σ=1 0.414 → σ=3 0.343 → σ=5 0.299 → σ=10 0.308 → σ=17 0.295 → σ=25 0.276 |
| `deletion_whole_delay`    | yes | pd=0.0 0.627 → pd=0.2 0.108 → pd=0.4 0.086 → pd=0.6 0.086 → pd=0.8 0.051 |
| `ccisi_tau` (test-time)   | no  | f=0.0 1.000 → f=0.2 0.986 → f=0.4 0.830 → f=0.6 0.672 → f=0.8 0.572 → f=1.0 0.539 |
| `shd_whole_delay` (test)  | no  | f=0.0 0.864 → f=0.2 0.678 → f=0.4 0.522 → f=0.6 0.382 → f=0.8 0.299 → f=1.0 0.287 |

The cliff in the buggy notebooks is huge (0.38→0.16, 0.86→0.41,
0.63→0.11) and is not consistent with biological perturbation strength;
σ=1 ms of jitter destroying half the accuracy should not happen on its
own. The plateau after the cliff is the unmistakable fingerprint of
"the upstream half of the network has been frozen at random init, and
the downstream half is doing whatever it can with random features."

#### 7.1.3 Concrete evidence — `jitter_part_nodelay`

`fc1.weight_g` / `weight_v` mean and std are bit-identical across
σ ∈ {1, 3, 5, 10, 17, 25} and differ from σ=0:

```
sigma=0:   fc1.weight_g mean=14.6847, std=14.5894   fc1.weight_v mean=-0.6226, std=6.1113
sigma=1:   fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=3:   fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=5:   fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=10:  fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=17:  fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
sigma=25:  fc1.weight_g mean=5.7758,  std=0.1568    fc1.weight_v mean=-0.0000, std=0.3861
```

Parameters that share an identical post-init fingerprint across six
independent training runs cannot have received any gradient signal in
any of those runs. Training logs confirm: σ>0 runs all converge to
val_loss ≈ 351–356 and val_acc ≈ 0.15–0.18, independent of σ.
Wall-clock at σ>0 is ~3× the σ=0 time (16 min → 50 min) — the per-batch
CPU/numpy round-trip dominating.

#### 7.1.4 The fix — straight-through estimator (STE)

```python
def _apply_perturbation(self, hidden: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return hidden
    perturbed = perturb_fn(hidden, p)   # numpy round-trip OR @no_grad func
    return hidden + (perturbed - hidden).detach()
```

Forward value equals `perturbed`; backward gradient flows through
`hidden` as if no perturbation had been applied. Standard trick for
non-differentiable discrete ops. Accept that the gradient is biased —
for mild perturbations this is well-behaved.

Reference: `my_project/docs/knowledge_bank/how_to_perturb_hidden_layer_during_training.md`

### 7.2 Issue 2 — "Delay-after-spike" anti-pattern (cosmetic)

`my_project/docs/knowledge_bank/where_to_apply_perturbation_between_layers.md`
warned that if `_first_hidden` ends with `if use_delay: x = self.delay1(x)`,
the perturbation hook receives a fractional tensor (because
`slayer.delay` was described there as doing linear interpolation
between time bins) and `np.where(spike_train == 1)` matches nothing, so
the perturbation is a silent no-op for delay runs.

**Status:** the structural anti-pattern is present in many notebooks
(see audit below), but in this version of `slayerSNN` it does **not**
actually produce the silent no-op. `slayer.delay` floors the delay
before shifting (`slayer.py:351-373`: "*it is floored during actual
delay applicaiton internally*") and `_delayFunction.apply` calls
`slayerCuda.shift(input, delay.data, Ts)`, an integer-step shift.
Therefore `delay(binary_spikes)` is binary spikes shifted by
`floor(delay)` time steps — still strictly binary, so the perturbation
is not a no-op.

Empirically, the SHD/SSC `_delay` sweep curves degrade smoothly (e.g.
`shd_whole_delay`: 0.864 → 0.678 → 0.522 → 0.382 → 0.299 → 0.287),
which is incompatible with a silent no-op. Jitter / shift / deletion
`_delay` variants show the same cliff+plateau as their `_nodelay`
siblings (just from a higher baseline at σ=0), which is Issue 1, not
Issue 2.

We refactor the routing out anyway (Section 3.4) because:

- It's fragile across slayer versions (docstring could change, or a
  "fractional delay" variant could be enabled).
- It mixes routing (delay) with compute (psp+fc+spike) in the same
  method, making the perturbation site harder to audit.
- The synthetic `*_delay` notebooks already follow the cleaner
  pattern; consistency makes future work cheaper.

Notebooks with the anti-pattern: `shd_train` (~line 453), `ssc_train`
(~line 484), `inverse_train` (~line 392), `jitter_train` (~line 481),
`shift_train` (~line 450), `deletion_train` (~line 601).
`isi_delay`, `coin_delay`, `ccisi_delay` already use the cleaner
pattern.

### 7.3 Issue 3 — Coincidence dataset / SLAYER `tSample` mismatch (resolved)

**Resolved 2026-05-18.** `coin_data_gen.py` produced 4000-step samples
but `coin_tau.ipynb` / `coin_delay.ipynb` declared `tSample=1000`.
Fix applied: `N_TIMESTEPS = 1000` in the generator (option 2 — match
data to notebooks), `coin_dataset.mat` regenerated, both notebooks
re-run end-to-end. Discarded the pre-fix per-lambda checkpoints.

Cross-check (2026-05-10):

- **Synthetic ISI**: `tSample=1000`, dataset `(N, 10, 1000)` — clean.
- **Synthetic CCISI**: cached output suggests `Time steps: 10000` vs
  `tSample=1000`. Still flagged as a follow-up — re-run the load cell
  against the current `ccisi_dataset.h5` to confirm whether cached
  output is stale or the bug is live. If live, regenerate at 1000.
- **Realistic SHD/SSC**: `nb_steps=100` raw, padded to `tSample=200`
  by `load_shd_data(..., target_T=200)`. This is the original Beyond
  Rate convention (verified in `temporal_shd_project`); the trailing
  100 zero-bins are settling time for the SpikeRate readout window
  `[0, 200]`. Load-bearing — do not touch.
- **Perturbation** (jitter / shift / deletion / inverse): inherit the
  SHD pipeline with the same 100→200 padding. Not a bug.

Secondary concern carried over from Beyond Rate: `nb_steps=100` over
`max_time=1.4 s` means each SHD/SSC bin represents ~14 ms but
notebooks declare `Ts=1` (1 simulation unit per bin). LIF_PARAMS time
constants and any "learned tau" values reported by the model are in
*bin units*, not milliseconds — flagged for interpretation work.
