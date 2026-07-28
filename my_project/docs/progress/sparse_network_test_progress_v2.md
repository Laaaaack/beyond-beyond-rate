# First Milestone v2 — Execution checklist: does sparsity increase temporal processing?

**Status:** scripts written and verified (2026-07-28); **nothing trained yet**.
Next action: run the **Step 1 probe** (both arms, `QUICK_TEST = True`).
**Owner:** _(you)_
**Why v2 exists:** [sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md) — v1's results and their diagnosis.
**v1 execution history:** [sparse_network_test_progress.md](sparse_network_test_progress.md)
**Full design & rationale:** [sparse_network.md](sparse_network.md)

---

## 1. Why there is a v2 (one paragraph)

v1 answered the question as posed and got a clear negative: in all 8 perturbation ×
arm combinations, sparser networks were **more** perturbation-robust, not less
(Spearman ρ = +0.67…+0.96). But the diagnosis showed v1 could not have tested H1 in
the first place. H1's mechanism assumes that when a layer is sparse each neuron fires
**0–2 spikes**, so *count* carries no resolution and the network is forced onto a
*latency* code. v1 never reached that regime — even its sparsest models fired **3.1
(no-delay) / 4.1 (delay) spikes per _active_ neuron**; the headline
`spikes_per_neuron` of 1.4 / 2.1 was diluted by silent neurons. Worse, v1's one-sided
hinge had **zero gradient at or below its target**, so a neuron could dodge the
penalty entirely by falling silent on a sample. Sparsity therefore arrived partly as
**stimulus selectivity** (silent fraction .25–.33 → .48–.54), which *strengthens* a
population-identity code that every perturbation in the study leaves perfectly
intact. The network always had a rich, perturbation-immune escape hatch.

**v2 closes the escape hatch**, making H1's premise true by construction, so that a
second negative result would refute H1 *on its own terms* rather than refuting a
regime H1 never claimed.

---

## 2. What v2 changes (and what it does not)

**Changed: the sparsity penalty.** A one-sided hinge on the *batch-mean* per-neuron
rate becomes a **two-sided band on the per-(sample, neuron) spike count**:

```python
# v1 — hinge on batch-mean rate
per_neuron_rate = hidden_spikes.sum(dim=-1).mean(dim=0)      # (C, 1, 1)
penalty = relu(per_neuron_rate - target).mean()

# v2 — two-sided band, charged per sample
per_sample_rate = hidden_spikes.sum(dim=-1)                  # (B, C, 1, 1)
penalty = (relu(per_sample_rate - band_hi)
           + under_weight * relu(band_lo - per_sample_rate)).mean()
```

Two consequences, both deliberate:

1. **The lower arm closes the identity escape hatch.** Silence now *costs*
   `under_weight × band_lo`, so neurons are pushed to fire on every sample. Drive
   `silent_fraction` → 0 and the identity channel carries almost nothing; pin the
   count inside a narrow band and the count channel carries ~1 bit. **Timing becomes
   the only rich channel left.**
2. **Charging per sample, not per batch-mean.** v1 averaged over the batch *before*
   the ReLU, so a neuron firing 10 spikes on 1 sample in 128 registered as rate 0.078
   and drew **no penalty at all** — bursty, highly selective neurons were free. This
   is verified numerically: on that exact case v1 charges **0.0000** and v2 charges
   **1.0547**.

**Changed: the swept axis.** v1 swept penalty *strength* at a fixed target, because
the target turned out inert. Both arms of a band bind, so the band *position* sets
achieved spikes-per-active-neuron directly and becomes the natural swept axis. v2
sweeps the **band** at fixed strength.

**Changed: the no-delay warm-up, 15 → 0.** That guard existed only because v1's hinge
had no lower arm to stop firing overshooting into silence. v2's band supplies one
structurally. Side benefit: both arms now run at warm-up 0, removing v1's standing
arm-comparability caveat.

**Unchanged (deliberately, for comparability):** architecture (128–128, SRMALPHA),
dataset and the fixed 60/15/15 splits, `NumSpikes` loss, Nadam, LR 0.1, MultiStepLR
milestone 300, batch 128, `EPOCHS = 1250`, early-stop patience 300, seeds {42,43,44},
and every eval-side perturbation grid.

---

## 3. Experiment setup

| Fixed choice | Value | Why |
|---|---|---|
| Dataset | SHD `whole` (input_dim 700) | one dataset only, as v1 |
| Arms | SGD-delay **and** SGD no-delay | the no-delay arm removes the delays-as-timing-mechanism confound; the gap between arms is how much timing the delays carry |
| Perturbation site | 1st hidden layer output | project's primary site |
| Swept axis | **band** (5 values) | binds from both sides, unlike v1's target |
| Seeds per band | 3 (42, 43, 44) | sparsity interacts strongly with init |
| **Models per arm** | **5 × 3 = 15** (30 across both) | |

**Scripts** (both verified 2026-07-28: syntax, hand-computed penalty values,
activity-stat correctness incl. the all-silent guard, gradient flow from the *lower*
arm through to `fc1`, 2-epoch end-to-end run, the identity
`sp/neuron == (1−silent) × sp/ACTIVE`, and run-tag isolation from v1 — 21/21 pass):

- [sn_train_withDelay_v2.py](../../exp_sparse_network/sn_train_withDelay_v2.py)
- [sn_train_noDelay_v2.py](../../exp_sparse_network/sn_train_noDelay_v2.py)

**Key configuration:**

```python
SPARSITY_BANDS   = [(1,2), (2,3), (3,5), (5,8), (8,12)]   # the swept axis
PENALTY_STRENGTH = 3.0        # fixed; only has to be large enough to bind
UNDER_WEIGHT     = 1.0        # weight on the anti-silencing lower arm
WARMUP_EPOCHS    = 0          # both arms
SEEDS            = [42, 43, 44]
EPOCHS           = 1250
```

`(1,2)` is the regime H1 is actually about. The wider bands supply the sparsity
gradient the headline plot needs.

**Outputs** are tagged `_v2_` throughout and cannot collide with v1 artifacts:

| Artefact | Path |
|---|---|
| Checkpoint | `sn_data/sparse_whole_{delay,nodelay}_v2_band{lo}-{hi}_str3_seed{s}.pt` |
| Per-model log | `sn_log/{run_tag}_training_log.json` — now includes `train_silent`, `train_sp_active` |
| Summary | `sn_log/sparse_whole_{delay,nodelay}_v2_train_summary.json` — now includes `spikes_per_active_neuron` |

Probe runs get a `_probe` suffix on every output, so they can never clobber a real run.

> ⚠️ **Protocol rule (do not break):** sparsity is applied **during training** on
> clean data; every perturbation is applied **only at evaluation**. Never train with
> a perturbation on.

> ⚠️ **Always analyse against *measured* firing statistics**, never the band, the
> strength, or any probe number.

---

## 4. Test steps

### Step 0 — verify the scripts ✅ DONE (2026-07-28)

21/21 checks pass, including the decisive v1-vs-v2 contrast on a bursty neuron
(v1 penalty 0.0000, v2 penalty 1.0547). Nothing to re-run.

### Step 1 — the probe (DO THIS FIRST)

Both scripts ship with `QUICK_TEST = True`, which runs **3 bands ×
seed 42 × 400 epochs**: `[(1,2), (3,5), (8,12)]`. The tight band is first so a
failure surfaces early.

```
python my_project/exp_sparse_network/sn_train_withDelay_v2.py     # ~2.3 h
python my_project/exp_sparse_network/sn_train_noDelay_v2.py       # ~3.8 h
```

The band penalty is **unvalidated** — forcing every neuron to fire on every sample
is a far stronger constraint than v1's hinge and may cost real accuracy. This probe
is a **feasibility check, not a calibration**: reduced-epoch firing *under-estimates*
the full run (v1 lesson — weak penalties re-densify over 1250 epochs), so never lock
a grid from these numbers.

**Acceptance criteria** — all three must hold, per model. The script prints each:

| # | Criterion | Read from | If it fails |
|---|---|---|---|
| 1 | `spikes/ACTIVE` lands **inside** the band | printed `in band` / `OUT OF BAND` | above `band_hi` → raise `PENALTY_STRENGTH` |
| 2 | `silent_fraction` falls **toward 0** | summary + `train_silent` | still high → raise `UNDER_WEIGHT` |
| 3 | `clean_acc` stays **well above chance** (0.05) at the tight `(1,2)` band | summary | collapsed → lower `PENALTY_STRENGTH`, then `UNDER_WEIGHT` |

**Also check** `train_silent` in the per-epoch log never spikes toward 1.0 in the
first epochs — that is v1's no-delay collapse signature. If it does, re-add
`WARMUP_EPOCHS = 10–15` **to both arms** (keep them matched).

> The lower arm is **preventive, not reviving**: SLAYER's surrogate gradient vanishes
> once a neuron's membrane sits far below threshold, so a deeply silent neuron may be
> unrecoverable. That is why v2 runs at warm-up 0 — neurons must never be allowed to
> go deeply silent in the first place.

### Step 2 — lock the grid, run the full sweep

Only once Step 1 passes on **both** arms. Set `QUICK_TEST = False` in both scripts,
adjusting `SPARSITY_BANDS` / `PENALTY_STRENGTH` / `UNDER_WEIGHT` if the probe
demanded it, then run both: **15 models per arm at 1250 epochs** (~35 h delay,
~59 h no-delay; **~94 h total**).

Record the achieved table per arm: band, seed, `spikes_per_neuron`,
**`spikes_per_active_neuron`**, `silent_fraction`, `clean_acc`. Drop any model at
~chance accuracy (a broken net is not "maximally temporal") and say so explicitly.

### Step 3 — the eval sweeps (eval-only, 1st layer)

Four perturbations × two arms, reusing the existing scripts under
`exp_sparse_network/{jitter,shift,shd,deletion}/`. **Two edits needed in each of the
8 scripts first:**

1. `TRAIN_SUMMARY_FILE` → point at `..._v2_train_summary.json`.
2. Add `"spikes_per_active_neuron"` to `SUMMARY_PASSTHROUGH_FIELDS`, so the metric
   that matters reaches the analysis. (`"target_rate"` is absent from v2 summaries;
   the passthrough is guarded by `if field in row`, so it degrades harmlessly.)

The network classes need **no** change — v2 checkpoints have an identical parameter
set to v1's and load unchanged.

| Perturbation | Sweep | Preserves rate? | Role |
|---|---|---|---|
| jitter | σ ∈ {0,1,3,5,10,17,25} | yes | local timing |
| shift | σ ∈ {0,1,3,5,10,17,25} | yes | cross-neuron alignment / onset |
| shd (relocation) | f ∈ {0,…,1.0} | exactly | **all** timing at f=1 — the cleanest probe |
| deletion | p_d ∈ {0,…,0.8} | **no** | **rate / general-robustness control** |

Keep `NUM_REPEATS = 3` and `QUICK_TEST = False`. Cost ~1 h for all eight.

Two grid fixes carried over from v1's caveats — apply them here:

- **Extend deletion** to `p_d ∈ {0.9, 0.95}`. v1's equal-damage numbers were censored
  at the 0.8 grid edge in the delay arm.
- **Refine `shd` at low f** (`{0.02, 0.05, 0.1, 0.15, 0.2}`). All 15 v1 no-delay
  models lost everything by f = 0.2, so that arm's score rested on one grid point.

### Step 4 — analysis

`temporal_score = 1 − (acc(max) − 0.05) / (acc(0) − 0.05)` as in v1.

**4a. Manipulation check — do this before anything else.** v2's whole claim is that
the perturbation-immune channels were closed. Verify it, per checkpoint:

- `silent_fraction` ≈ 0 and `spikes_per_active_neuron` inside its band;
- **decode from identity** (binarised which-neurons-active vector) and **decode from
  count** (per-neuron count vector), both fit on the train split and scored on test.
  Both are exactly invariant to every rate-preserving perturbation, so the count
  decode is the ceiling on what a perturbation-immune readout could score.
  **v1 reference values to beat (i.e. fall below):** identity .453→.513 (no-delay),
  .339→.502 (delay); count .749→.687, .759→.736.

If these do not drop substantially versus v1, **v2 did not do its job** and the
headline result is uninterpretable — fix the manipulation before reading anything else.

**4b. Headline.** `temporal_score` vs **measured** `spikes_per_neuron`, one point per
(band, seed), arms in separate panels. Report Spearman ρ. **H1 predicts ρ < 0.**

**4c. Robustness control.** Repeat v1's decisive analysis: partial correlation
ρ(sp/neuron, timing score | deletion score), the `shd(f=1) − deletion(p=max)`
contrast, and the equal-damage comparison. A raw correlation that vanishes under the
deletion control is general robustness, not timing.

**4d. Cross-version comparison.** v1 vs v2 at matched measured firing rate. This is
the payoff: it isolates what closing the identity channel did to timing dependence.

**4e. Plots.** Headline; `acc(perturbation)` curve families coloured by firing rate;
guard plot (`clean_acc` vs firing rate).

---

## 5. How to read the outcome (fix the interpretations now, not after)

| Outcome | Meaning |
|---|---|
| Manipulation check passes **and** ρ < 0 | **H1 supported** under its own premise. v1's positive ρ was an artifact of the identity escape hatch. Strong result — scale up. |
| Manipulation check passes **and** ρ > 0 | **H1 refuted on its own terms** — much stronger than v1, because the count/identity channels were closed by construction and sparsity *still* did not increase timing dependence. |
| Manipulation check passes **and** ρ ≈ 0 | Timing dependence is invariant to sparsity once the identity channel is closed. Consistent with v1's delay-arm reading (sparsity changes timing *precision*, not *presence*). |
| `clean_acc` collapses at the tight band | The band is unlearnable for this architecture. H1 is **untestable here** — report as such and move to the synthetic ISI task, where the latency regime can be imposed by design. |
| Manipulation check fails | Uninterpretable. Retune `PENALTY_STRENGTH` / `UNDER_WEIGHT` and re-probe. |

**Either direction is a valid finding.** Both are worth reporting; the milestone is
done when the headline plot exists and the manipulation check is documented.

---

## 6. Watch out for (carried lessons)

1. **Reduced-epoch numbers under-estimate the full run.** The 400-ep probe is a
   feasibility check only. Never lock a grid from it — v1 lost a full 15-model run to
   exactly this, and the under-estimate is *worse* for stronger penalties.
2. **`spikes_per_neuron` alone lies.** It is diluted by silent neurons; it hid v1's
   mechanism for all of Step 1. Always report `spikes_per_active_neuron` beside it.
3. **Dead neurons are unrecoverable.** The band's lower arm prevents silence but
   cannot reliably reverse it. Watch `train_silent` from epoch 0.
4. **Never analyse against the band or the strength** — only measured firing.
5. **Never train with a perturbation on.** Clean train, perturb at eval.
6. **A chance-level model is broken, not "maximally temporal."** Drop it and say so.
7. **Expect lower clean accuracy in the no-delay arm** (v1: ~.49–.59 vs ~.78–.89).
   That is why the score is chance-corrected and baseline-normalised, and why arms are
   compared at matched firing rate, never on raw accuracy drops.
8. **Keep the two arms' hyper-parameters matched.** If the probe forces a warm-up back
   on, apply it to both.

---

## 7. Progress checklist

- [x] v1 result recorded and diagnosed
  ([sparse_network_1stLayer_results.md](sparse_network_1stLayer_results.md))
- [x] v2 penalty designed — two-sided per-sample band closing the identity escape hatch
- [x] v2 training scripts written
  ([withDelay](../../exp_sparse_network/sn_train_withDelay_v2.py),
  [noDelay](../../exp_sparse_network/sn_train_noDelay_v2.py))
- [x] v2 scripts verified (21/21: penalty math, activity stats, gradient flow from the
  lower arm, 2-epoch end-to-end, run-tag isolation from v1)
- [ ] **Step 1 — probe both arms** (`QUICK_TEST = True`, ~6 h total); check the three
  acceptance criteria + the `train_silent` collapse signature
- [ ] Step 1b — retune `PENALTY_STRENGTH` / `UNDER_WEIGHT` / warm-up if needed, re-probe
- [ ] Step 2 — lock the band grid; full run, 15 models × 2 arms at 1250 ep (~94 h)
- [ ] Step 3a — patch the 8 eval scripts (`TRAIN_SUMMARY_FILE`,
  `SUMMARY_PASSTHROUGH_FIELDS`); extend the deletion grid; refine the low-f `shd` grid
- [ ] Step 3b — run all four perturbation sweeps × both arms (~1 h)
- [ ] Step 4a — **manipulation check** (silent ≈ 0, sp/ACTIVE in band, identity/count
  decode well below v1's values)
- [ ] Step 4b–4e — headline ρ, deletion control + partial correlation, v1-vs-v2 at
  matched firing, three plots
- [ ] v2 verdict recorded against the table in §5, plus the delay-vs-no-delay comparison

---

## 8. Cost summary

| Stage | Delay arm | No-delay arm | Total |
|---|---|---|---|
| Step 1 probe (3 models, 400 ep) | ~2.3 h | ~3.8 h | **~6 h** |
| Step 2 full run (15 models, 1250 ep) | ~35 h | ~59 h | **~94 h** |
| Step 3 eval sweeps (4 perturbations) | — | — | **~1 h** |

Per-model timings are measured from v1's run logs (~2.3 h delay, ~3.9 h no-delay at
1250 epochs). **Do not skip Step 1** — it costs 6 % of the full run and is the only
thing standing between a bad hyper-parameter and a wasted 94 hours.
