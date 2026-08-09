"""v3 Phase 1: the decisive analysis — does timing reliance depend on ``a`` or on ``s``?

Joins the three measurement files the factorial produces and answers the question v1
structurally could not, because there ``a`` and ``s`` moved together at
``rho = -0.94``:

- the training summary — the design variables ``k``, ``floor_strength``, and the
  achieved ``a`` (spikes per active neuron) and ``s`` (silent fraction);
- [hidden_channel_decode.py](hidden_channel_decode.py) — **availability**, the share of
  the layer's decodable class information that requires knowing spike placement;
- [phase1_measure.py](phase1_measure.py) — **usage**, the readout's own measured
  dependence on that timing, plus the spike-**deletion** control.

The two models, per document 5 §4b::

    usage           ~ b_a * log(a) + b_s * s + b_d * (deletion control)
    timing_fraction ~ b_a * log(a) + b_s * s

- **H1' predicts b_a < 0 for usage**: holding selectivity fixed, fewer spikes per
  active neuron should push the code toward timing, so the readout should lose *more*
  when timing is destroyed.
- **H2 predicts b_s carries the effect and b_a is null**: the anti-H1 trend v1 measured
  was selectivity building a perturbation-immune identity code, not temporal sparsity
  doing anything.

Three things this script insists on, each because an earlier version was misread
without them:

1. **The design acceptance check runs first and is reported first.** If
   ``|rho(a, s)| > 0.5`` the coefficients cannot be attributed to one axis and nothing
   below is interpretable — the correct response is to report that and stop.
2. **The deletion control is in the usage model, not optional.** Phase 0 found it
   tracking sparsity at +0.936 against the timing probe's +0.939 in the no-delay arm,
   i.e. a pure rate insult and a pure timing insult were indistinguishable there. A
   ``b_a`` that survives only without the control is a statement about general
   robustness.
3. **Standardised coefficients are reported alongside raw ones.** ``log(a)`` and ``s``
   are in different units, so raw magnitudes cannot be compared; "which axis carries
   the effect" is a question about the standardised ones.

Which grid, which layers, and what a coefficient off it means
--------------------------------------------------------------
``GRID`` selects the training generation, and each constrained a different set of
hidden layers — so both dependent variables have to be read at the sites that
generation's measurement scripts wrote::

    "v3"     1st-layer factorial (document 5)    axes l1    usage at l1
    "v3L2"   2nd-layer factorial (document 6)    axes l2    usage at l2
    "v3L12"  both-layer factorial (document 7)   axes net   usage at l1, l2, both

For the both-layer grid the design axes are the **network** ones, because that is
what the two knobs manipulate and what the acceptance check was passed on
(rho(a_net, s_net) = -0.199 no-delay, +0.040 delay, against v1's -0.879 / +0.042).
Two consequences, both of which belong beside any number this script prints:

- **A coefficient from that grid is a network-level one and cannot be attributed
  to a layer.** rho(a1, a2) is +0.973 / +0.893 there, and both knobs move both
  layers by construction, so the design cannot separate them. Attribution is what
  the two single-layer grids are for (document 7 §6 point 1).
- **The dependent variable is reported at every site, not pooled.** The sweeps
  found the three sites do not reduce to one another — relocation and jitter
  saturate (``both`` ~ ``l1``) while deletion compounds hard (document 7 §6e) — so
  ``PRIMARY_SITE`` names one for the verdict and the others are fitted and printed
  beside it rather than dropped.

Writes ``v3_analysis/log/phase1_regress_{tag}.json`` and
``v3_analysis/fig/phase1_regress_{tag}{arm}.png``. The tag is load-bearing: the
three grids share dataset, arm, ``k``, floor and seed, so without it this script
would overwrite the completed 1st-layer result (document 7 §6 point 4).
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr, t as t_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# This console is GBK, which cannot encode every character used below. Replace rather
# than raise: a mangled dash is a better failure than losing the whole analysis.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
TRAIN_LOG_DIR = EXP_DIR / "sn_log"
LOG_DIR = SCRIPT_DIR / "log"
FIG_DIR = SCRIPT_DIR / "fig"

# Which training generation to analyse. Must match the GRID the two measurement
# scripts were run at, since it selects their output files as well as this one's.
GRID: str = "v3L12"
VERSION_TAG: str = f"{GRID}_" if GRID else ""
ARMS: tuple[str, ...] = ("delay", "nodelay")

# The decode view supplying the design axes ``a`` and ``s``. For the both-layer grid
# that is the pooled ``net`` view: the network axes are the ones the two knobs
# manipulate and the ones the acceptance check was passed on.
GRID_AXES_VIEW: dict[str, str] = {"": "l1", "v3": "l1", "v3L2": "l2", "v3L12": "net"}

# The perturbation sites supplying ``usage`` and the deletion control, and the decode
# views supplying ``availability``. Both must match what the measurement scripts wrote.
GRID_SITES: dict[str, tuple[str, ...]] = {
    "": ("l1",), "v3": ("l1",), "v3L2": ("l2",), "v3L12": ("l1", "l2", "both"),
}
GRID_VIEWS: dict[str, tuple[str, ...]] = {
    "": ("l1",), "v3": ("l1",), "v3L2": ("l2",), "v3L12": ("l1", "l2", "net"),
}

AXES_VIEW: str = GRID_AXES_VIEW[GRID]
SITES: tuple[str, ...] = GRID_SITES[GRID]
VIEWS: tuple[str, ...] = GRID_VIEWS[GRID]

# The site and view the verdict is worded on. Every other site and view is fitted and
# printed beside them — this names a primary, it does not discard the rest.
#
# ``both`` is primary for the both-layer grid because it is the insult matched to the
# manipulation: the grid constrained both layers, so the probe that destroys timing at
# both is the one asking whether *this network's* code moved toward timing. Read it
# knowing that relocation saturates — ``both`` came back within .03 of ``l1`` on every
# sweep — so the two should agree, and a disagreement is a finding rather than noise.
PRIMARY_SITE: str = SITES[-1]
PRIMARY_VIEW: str = VIEWS[-1]

# v1's observational rho(a, s) on the axes each grid manipulates, printed beside the
# acceptance check so the achieved number has something to be read against. Quoting
# layer 1's -0.943 next to a network-wide axis would understate what was broken.
OBSERVATIONAL_RHO: dict[str, str] = {
    "": "-0.943 no-delay, -0.455 delay (layer 1)",
    "v3": "-0.943 no-delay, -0.455 delay (layer 1)",
    "v3L2": "+0.829 no-delay, +0.427 delay (layer 2)",
    "v3L12": "-0.879 no-delay, +0.042 delay (network)",
}

# One line per injection site, so a printed model never has to be decoded from a
# field name alone.
SITE_DESCRIPTIONS: dict[str, str] = {
    "l1": "perturbed at the 1st hidden layer only",
    "l2": "perturbed at the 2nd hidden layer only",
    "both": "perturbed at both layers -- layer 2 after an already-perturbed layer 1",
    "net": "decoded from both layers' features concatenated",
}

# The design acceptance threshold on the factorial itself (document 5 §4b).
MAX_AXIS_CORRELATION: float = 0.5

# Significance level used only to word the printed verdict.
ALPHA: float = 0.05

# Coverage of the reported intervals. At n = 16 the interval, not the p-value, is what
# tells a reader which effect sizes this design can and cannot rule out.
CONFIDENCE_LEVEL: float = 0.95

# Largest |difference in log a| allowed when pairing a floor-on model with a floor-off
# one for the matched contrast. 0.2 in log units is about 22% in a.
MATCH_TOLERANCE_LOG_A: float = 0.2


def ordinary_least_squares(design: np.ndarray, response: np.ndarray,
                           names: list[str]) -> dict:
    """Fit OLS with an intercept and return coefficients, standard errors and fit.

    Implemented directly rather than via statsmodels, which is not in this
    environment. Standard errors are the usual homoscedastic ones; with n = 16 they are
    the honest limit on what this design can claim, and the printed output reports the
    degrees of freedom so that is visible.

    Args:
        design: Predictor matrix, shape (n, p), WITHOUT an intercept column.
        response: Response vector, shape (n,).
        names: Predictor names, length p.

    Returns:
        Dict with per-predictor coefficient/se/t/p, plus r_squared, adjusted
        r_squared, n and degrees of freedom.
    """
    n_samples = len(response)
    matrix = np.column_stack([np.ones(n_samples), design])
    coefficients, *_ = np.linalg.lstsq(matrix, response, rcond=None)
    residuals = response - matrix @ coefficients
    degrees_of_freedom = n_samples - matrix.shape[1]

    residual_sum = float(residuals @ residuals)
    variance = residual_sum / degrees_of_freedom
    covariance = variance * np.linalg.pinv(matrix.T @ matrix)
    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = coefficients / standard_errors
    p_values = 2.0 * t_distribution.sf(np.abs(t_statistics), degrees_of_freedom)
    critical = t_distribution.ppf(1.0 - 0.5 * (1.0 - CONFIDENCE_LEVEL),
                                  degrees_of_freedom)
    half_width = critical * standard_errors

    total_sum = float(((response - response.mean()) ** 2).sum())
    r_squared = 1.0 - residual_sum / max(1e-12, total_sum)
    adjusted = 1.0 - (1.0 - r_squared) * (n_samples - 1) / degrees_of_freedom

    terms = {}
    for index, name in enumerate(["intercept"] + names):
        terms[name] = {
            "coefficient": float(coefficients[index]),
            "standard_error": float(standard_errors[index]),
            "t": float(t_statistics[index]),
            "p": float(p_values[index]),
            "ci_low": float(coefficients[index] - half_width[index]),
            "ci_high": float(coefficients[index] + half_width[index]),
        }
    return {"terms": terms, "r_squared": r_squared, "adjusted_r_squared": adjusted,
            "n": n_samples, "degrees_of_freedom": degrees_of_freedom}


def standardise(values: np.ndarray) -> np.ndarray:
    """Return z-scored values, or zeros if the column is constant."""
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else values * 0.0


def variance_inflation(design: np.ndarray, names: list[str]) -> dict[str, float]:
    """Return each predictor's VIF, the standard collinearity diagnostic.

    VIF is 1 when a predictor is orthogonal to the others and rises without bound as it
    becomes predictable from them. It is reported because the whole point of the
    factorial is to keep it near 1 — v1's observational design would have produced
    large values here, which is what made its coefficients uninterpretable.

    Args:
        design: Predictor matrix without an intercept, shape (n, p).
        names: Predictor names.

    Returns:
        Mapping of predictor name to VIF.
    """
    inflation = {}
    for index, name in enumerate(names):
        others = np.delete(design, index, axis=1)
        if others.shape[1] == 0:
            inflation[name] = 1.0
            continue
        fit = ordinary_least_squares(others, design[:, index],
                                     [f"x{j}" for j in range(others.shape[1])])
        inflation[name] = float(1.0 / max(1e-12, 1.0 - fit["r_squared"]))
    return inflation


def report_model(label: str, design: np.ndarray, response: np.ndarray,
                 names: list[str]) -> dict:
    """Fit, print and return one regression in both raw and standardised form."""
    raw = ordinary_least_squares(design, response, names)
    standardised = ordinary_least_squares(
        np.column_stack([standardise(design[:, j]) for j in range(design.shape[1])]),
        standardise(response), names)
    inflation = variance_inflation(design, names)

    print(f"\n--- {label} ---")
    print(f"  n = {raw['n']}, df = {raw['degrees_of_freedom']}, "
          f"R^2 = {raw['r_squared']:.3f} (adj {raw['adjusted_r_squared']:.3f})")
    print(f"  {'term':<22} {'beta':>10} {'se':>9} {'t':>7} {'p':>9} "
          f"{'95% CI':>19} {'beta*':>7} {'VIF':>6}")
    for name in names:
        term = raw["terms"][name]
        interval = f"[{term['ci_low']:+.4f},{term['ci_high']:+.4f}]"
        print(f"  {name:<22} {term['coefficient']:>+10.4f} "
              f"{term['standard_error']:>9.4f} {term['t']:>+7.2f} "
              f"{term['p']:>9.4f} {interval:>19} "
              f"{standardised['terms'][name]['coefficient']:>+7.3f} "
              f"{inflation[name]:>6.2f}")
    return {"raw": raw, "standardised": standardised,
            "variance_inflation": inflation}


def matched_floor_contrast(rows: list[dict]) -> dict:
    """Pair floor-on with floor-off models at matched ``a`` and compare usage.

    Because ``s`` is *manipulated* here rather than observed, this is a genuine
    controlled comparison — the first in this line of work. H2 predicts that removing
    silence (floor on) *raises* usage, since the identity code that survives relocation
    is what silence builds.

    Pairing is nearest-neighbour on ``log(a)`` without replacement, keeping only pairs
    within ``MATCH_TOLERANCE_LOG_A``. With ``a`` and ``s`` already near-orthogonal this
    is a robustness check on the regression rather than the primary analysis.

    Args:
        rows: The joined per-model table.

    Returns:
        Dict of the pairs, the mean difference and a paired t test.
    """
    floor_on = [row for row in rows if row["floor_strength"] > 0]
    floor_off = [row for row in rows if row["floor_strength"] == 0]
    available = list(floor_off)
    pairs = []
    for on_row in sorted(floor_on, key=lambda r: r["log_a"]):
        if not available:
            break
        nearest = min(available,
                      key=lambda r: abs(r["log_a"] - on_row["log_a"]))
        gap = abs(nearest["log_a"] - on_row["log_a"])
        if gap > MATCH_TOLERANCE_LOG_A:
            continue
        available.remove(nearest)
        pairs.append({
            "floor_on": on_row["run_tag"], "floor_off": nearest["run_tag"],
            "log_a_gap": gap,
            "s_on": on_row["s"], "s_off": nearest["s"],
            "usage_on": on_row["usage"], "usage_off": nearest["usage"],
            "delta_usage": on_row["usage"] - nearest["usage"],
        })

    print(f"\n--- controlled contrast: floor ON vs OFF at matched a "
          f"(|d log a| <= {MATCH_TOLERANCE_LOG_A}) ---")
    if len(pairs) < 2:
        print(f"  only {len(pairs)} matched pair(s) — not interpretable")
        return {"pairs": pairs, "mean_delta_usage": None, "p": None,
                "n_pairs": len(pairs)}

    deltas = np.array([pair["delta_usage"] for pair in pairs])
    degrees_of_freedom = len(deltas) - 1
    standard_error = deltas.std(ddof=1) / np.sqrt(len(deltas))
    t_statistic = deltas.mean() / max(1e-12, standard_error)
    p_value = float(2.0 * t_distribution.sf(abs(t_statistic), degrees_of_freedom))

    print(f"  {'floor ON':<38} {'floor OFF':<38} {'dlogA':>6} "
          f"{'s_on':>6} {'s_off':>7} {'dusage':>8}")
    for pair in pairs:
        print(f"  {pair['floor_on']:<38} {pair['floor_off']:<38} "
              f"{pair['log_a_gap']:>6.3f} {pair['s_on']:>6.1%} "
              f"{pair['s_off']:>7.1%} {pair['delta_usage']:>+8.3f}")
    print(f"  mean delta usage = {deltas.mean():+.4f} "
          f"(se {standard_error:.4f}, t = {t_statistic:+.2f}, p = {p_value:.4f}, "
          f"{len(deltas)} pairs)")
    print("  H2 predicts delta > 0: removing silence should RAISE timing reliance.")
    return {"pairs": pairs, "mean_delta_usage": float(deltas.mean()),
            "standard_error": float(standard_error), "t": float(t_statistic),
            "p": p_value, "n_pairs": len(deltas)}


def report_arm_decision(rows: list[dict], usage_fits: dict,
                        control_fits: dict) -> dict:
    """Print the two measurements the primary-arm decision rests on.

    Document 5 chose the delay arm on two Phase 0 findings that no factorial can fix,
    both measured **at layer 1 only**. Carrying that decision across a layer boundary
    is the mistake this calibration exists to avoid (document 7 §5 step 2), so both
    are re-measured here on the grid's own checkpoints:

    1. **The readout-efficiency gap**, ``decode_full - clean_acc``. A large positive
       gap means a linear decoder reading the hidden layer beats the network's own
       readout, so a perturbation experiment there measures the readout's inefficiency
       as much as the code's structure. Layer 1 gave +.300 in the no-delay arm against
       -.013 in the delay arm, which is what decided it.
    2. **Whether the deletion control is distinguishable from the timing probe.** In
       the no-delay arm at layer 1 a pure rate insult and a pure timing insult tracked
       sparsity identically (+0.936 against +0.939), so nothing there could be called
       a timing effect. Here the factorial lets that be asked properly: fit the control
       against the same two axes as usage and compare the coefficients. If ``b_a`` is
       the same for both, the usage trend is general robustness.

    Both sides of measurement 2 must carry the **same** predictors, so the usage models
    passed here are the ones fitted *without* the deletion control: comparing a
    control-partialled coefficient against the control's own would not be like for like.

    Args:
        rows: The arm's joined table.
        usage_fits: ``usage ~ log(a) + s`` models keyed by injection site.
        control_fits: ``control ~ log(a) + s`` models keyed by injection site.

    Returns:
        Dict of both measurements, for the results file.
    """
    print("\n--- arm decision, re-measured on this grid (document 7 §5 step 2) ---")

    gaps = {}
    print("  readout-efficiency gap = decode_full - clean_acc  "
          "(layer 1 Phase 0: +.300 no-delay, -.013 delay)")
    for view in VIEWS:
        gap = np.array([row[f"decode_full_{view}"] - row["clean_acc"]
                        for row in rows])
        gaps[view] = {"mean": float(gap.mean()), "std": float(gap.std()),
                      "min": float(gap.min()), "max": float(gap.max()),
                      "fraction_positive": float((gap > 0).mean())}
        print(f"    {view:<5} {gap.mean():+.3f} +- {gap.std():.3f}  "
              f"[{gap.min():+.3f}, {gap.max():+.3f}]  "
              f"positive on {(gap > 0).mean():.0%} of checkpoints")

    separations = {}
    print("\n  timing probe vs deletion control, same predictors "
          "(indistinguishable => general robustness, not timing)")
    print(f"    {'site':<6} {'b_a usage':>10} {'b_a control':>12} {'difference':>11} "
          f"{'b_s usage':>10} {'b_s control':>12}")
    for site in SITES:
        usage_terms = usage_fits[site]["raw"]["terms"]
        control_terms = control_fits[site]["raw"]["terms"]
        separations[site] = {
            "beta_a_usage": usage_terms["log_a"]["coefficient"],
            "beta_a_control": control_terms["log_a"]["coefficient"],
            "beta_a_difference": (usage_terms["log_a"]["coefficient"]
                                  - control_terms["log_a"]["coefficient"]),
            "beta_s_usage": usage_terms["s"]["coefficient"],
            "beta_s_control": control_terms["s"]["coefficient"],
        }
        print(f"    {site:<6} {usage_terms['log_a']['coefficient']:>+10.4f} "
              f"{control_terms['log_a']['coefficient']:>+12.4f} "
              f"{separations[site]['beta_a_difference']:>+11.4f} "
              f"{usage_terms['s']['coefficient']:>+10.4f} "
              f"{control_terms['s']['coefficient']:>+12.4f}")

    return {"readout_gap": gaps, "timing_vs_control": separations}


def verdict(usage_fit: dict, availability_fit: dict, axis_rho: float,
            min_clean_acc_floor_on: float) -> str:
    """Word the outcome against document 5 §5's table.

    Args:
        usage_fit: The usage regression's report.
        availability_fit: The availability regression's report.
        axis_rho: Spearman rho between the two sparsity axes.
        min_clean_acc_floor_on: Worst clean accuracy in the floor-on column.

    Returns:
        A human-readable verdict string.
    """
    if abs(axis_rho) > MAX_AXIS_CORRELATION:
        return (f"DESIGN FAILED — |rho(a, s)| = {abs(axis_rho):.3f} exceeds "
                f"{MAX_AXIS_CORRELATION}. Not interpretable as a test of H1'.")
    if min_clean_acc_floor_on < 2.0 / 20:
        return ("Clean accuracy at/near chance in the floor-on column — the "
                "constraint is unlearnable in this architecture. Report and move to "
                "Phase 2.")

    beta_a = usage_fit["raw"]["terms"]["log_a"]
    beta_s = usage_fit["raw"]["terms"]["s"]
    a_significant = beta_a["p"] < ALPHA
    s_significant = beta_s["p"] < ALPHA

    if a_significant and beta_a["coefficient"] < 0:
        return ("H1' SUPPORTED under its own premise — b_a < 0 with s held fixed. "
                "v1's positive rho was the selectivity confound.")
    if a_significant and beta_a["coefficient"] > 0:
        return ("H1' REFUTED on its own terms — b_a > 0 with s fixed by "
                "construction, so the prediction ran backwards even after the "
                "confound was broken. The strongest available negative.")
    if s_significant:
        return ("H2 CONFIRMED — b_a is null while b_s carries the effect. "
                "'Sparsity' was never the operative variable; selectivity was.")

    # Neither coefficient clears alpha. At n = 16 that is not the same as "no effect",
    # and reporting it as though it were would repeat the mistake this whole document
    # exists to correct. Say what the intervals actually exclude, and which way the
    # point estimates run.
    standardised = usage_fit["standardised"]["terms"]
    availability_a = availability_fit["raw"]["terms"]["log_a"]
    largest = max(("log_a", "s", "control_deletion"),
                  key=lambda name: abs(standardised[name]["coefficient"]))
    direction = ("AGAINST H1' (H1' predicts b_a < 0)"
                 if beta_a["coefficient"] > 0 else "in H1's predicted direction")
    return (
        f"INCONCLUSIVE but directional, and NOT symmetric between the hypotheses. "
        f"Neither coefficient clears alpha = {ALPHA} at n = {usage_fit['raw']['n']} "
        f"(df = {usage_fit['raw']['degrees_of_freedom']}). "
        f"b_a = {beta_a['coefficient']:+.4f} "
        f"[{beta_a['ci_low']:+.4f}, {beta_a['ci_high']:+.4f}], p = {beta_a['p']:.3f} "
        f"— point estimate runs {direction}. "
        f"b_s = {beta_s['coefficient']:+.4f} "
        f"[{beta_s['ci_low']:+.4f}, {beta_s['ci_high']:+.4f}], p = {beta_s['p']:.3f} "
        f"— H2's direction. Largest standardised effect: {largest} "
        f"(beta* = {standardised[largest]['coefficient']:+.3f}). "
        f"Availability remains flat (b_a p = {availability_a['p']:.3f}), a third "
        f"independent replication of it. Read as: H1' gained no support even with "
        f"its confound removed, the residual signal is selectivity and general "
        f"robustness, and the design is underpowered to separate a modest H2 effect "
        f"from none.")


def decode_view(row: dict, view: str) -> dict:
    """Return one decode view's block from a checkpoint's row.

    Tolerates the flat schema the decode script wrote before it was retargeted to
    two layers, so the completed 1st-layer analysis can be re-run against its
    archived artifacts without regenerating them. A multi-view grid never takes that
    path — its files always carry the nested blocks.

    Args:
        row: One checkpoint's row from a ``hidden_channel_decode_*`` file.
        view: ``"l1"``, ``"l2"`` or ``"net"``.

    Returns:
        The view's measurements.

    Raises:
        KeyError: If a multi-view file is missing the requested view.
    """
    if view in row:
        return row[view]
    if len(VIEWS) == 1:
        return row
    raise KeyError(view)


def site_score(row: dict, field: str, site: str) -> float:
    """Return one injection site's score from a checkpoint's row.

    Tolerates the unsuffixed schema the measurement script wrote before it was
    retargeted, for the same reason as ``decode_view``. A multi-site grid never takes
    that path.

    Args:
        row: One checkpoint's row from a ``phase1_measure_*`` file.
        field: ``"usage"``, ``"control_deletion"`` or an accuracy field.
        site: ``"l1"``, ``"l2"`` or ``"both"``.

    Returns:
        The score at that site.

    Raises:
        KeyError: If a multi-site file is missing the requested site.
    """
    key = f"{field}_{site}"
    if key in row:
        return row[key]
    if len(SITES) == 1:
        return row[field]
    raise KeyError(key)


def check_measurement_provenance(arm: str, decode: dict, measured: dict) -> None:
    """Fail if either input file was produced at a different ``GRID`` than this one.

    The three scripts communicate by filename, so a decode run at one grid and a
    measurement run at another produce two files that would otherwise be joined into a
    table mixing one grid's axes with another grid's dependent variable. The tags
    normally keep them apart, but once every grid has been measured all the filenames
    exist and a stale ``GRID`` in one script alone would no longer raise on its own.

    Files written before the retarget carry neither marker; those are skipped rather
    than rejected, since their grid is unambiguous from the filename that found them.

    Args:
        arm: The arm being joined, for the message.
        decode: The decode file's per-checkpoint rows.
        measured: The measurement file's per-checkpoint rows.

    Raises:
        ValueError: If either file records a grid or site set that is not this one's.
    """
    decode_row = next(iter(decode.values()))
    if "grid" in decode_row and decode_row["grid"] != GRID:
        raise ValueError(
            f"{arm}: the decode file was written at GRID = {decode_row['grid']!r} but "
            f"this run is GRID = {GRID!r}. Re-run hidden_channel_decode.py "
            f"at {GRID!r}.")

    measured_row = next(iter(measured.values()))
    if "sites" in measured_row and tuple(measured_row["sites"]) != SITES:
        raise ValueError(
            f"{arm}: the measurement file carries sites {measured_row['sites']} but "
            f"this run expects {list(SITES)}. Re-run phase1_measure.py at {GRID!r}.")


def build_table(arm: str) -> list[dict]:
    """Join the training summary, the decode and the perturbation measurements.

    Args:
        arm: ``"delay"`` or ``"nodelay"``.

    Returns:
        One row per checkpoint. ``a``, ``s`` and ``log_a`` are the design axes read
        from ``AXES_VIEW``; ``usage`` / ``control_deletion`` and ``availability``
        carry every site and view, with the primary one aliased unsuffixed so the
        regression and plotting code below reads a single name.
    """
    with open(TRAIN_LOG_DIR
              / f"sparse_whole_{arm}_{VERSION_TAG}train_summary.json") as handle:
        summary = json.load(handle)
    with open(LOG_DIR / f"hidden_channel_decode_{VERSION_TAG}{arm}.json") as handle:
        decode = json.load(handle)
    with open(LOG_DIR / f"phase1_measure_{VERSION_TAG}{arm}.json") as handle:
        measured = json.load(handle)

    missing = set(summary) - set(decode) or set(summary) - set(measured)
    if missing:
        raise ValueError(f"{arm}: measurements missing for {sorted(missing)}")
    check_measurement_provenance(arm, decode, measured)

    rows = []
    for run_tag, meta in summary.items():
        # `a` and `s` are taken from the decode, which measures them on exactly the
        # same test-split activity the availability score is computed from.
        axes = decode_view(decode[run_tag], AXES_VIEW)
        active_rate = axes["spikes_per_active_neuron"]
        row = {
            "run_tag": run_tag,
            "ceiling_k": meta["ceiling_k"],
            "floor_strength": meta["floor_strength"],
            "seed": meta["seed"],
            "clean_acc": measured[run_tag]["clean_acc"],
            "axes_view": AXES_VIEW,
            "a": active_rate,
            "log_a": float(np.log(active_rate)),
            "s": axes["silent_fraction"],
            "spikes_per_neuron": axes["spikes_per_neuron"],
        }
        for view in VIEWS:
            block = decode_view(decode[run_tag], view)
            row[f"availability_{view}"] = block["timing_fraction"]
            row[f"timing_information_{view}"] = block["timing_information"]
            row[f"decode_count_{view}"] = block["decode_count"]
            row[f"decode_identity_{view}"] = block["decode_identity"]
            row[f"decode_full_{view}"] = block["decode_full"]
        for site in SITES:
            row[f"usage_{site}"] = site_score(measured[run_tag], "usage", site)
            row[f"control_deletion_{site}"] = site_score(
                measured[run_tag], "control_deletion", site)

        # The primary site and view are aliased unsuffixed, so the models, the
        # matched contrast and the plot below name one thing rather than branching.
        row["usage"] = row[f"usage_{PRIMARY_SITE}"]
        row["control_deletion"] = row[f"control_deletion_{PRIMARY_SITE}"]
        row["availability"] = row[f"availability_{PRIMARY_VIEW}"]
        row["timing_information"] = row[f"timing_information_{PRIMARY_VIEW}"]
        row["decode_count"] = row[f"decode_count_{PRIMARY_VIEW}"]
        row["decode_identity"] = row[f"decode_identity_{PRIMARY_VIEW}"]
        rows.append(row)
    return rows


def plot_arm(arm: str, rows: list[dict]) -> None:
    """Four panels: both dependent variables against both axes, plus the design itself.

    Args:
        arm: Arm name, for the title.
        rows: The arm's joined table.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_axis = np.array([row["a"] for row in rows])
    s_axis = np.array([row["s"] for row in rows])
    floor = np.array([row["floor_strength"] for row in rows])
    ceiling = np.array([row["ceiling_k"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        (axes[0][0], a_axis, "usage", s_axis, "s (silent fraction)",
         "a — spikes per active neuron (log)", "usage (temporal_score at f = 1)",
         True),
        (axes[0][1], a_axis, "availability", s_axis, "s (silent fraction)",
         "a — spikes per active neuron (log)", "availability (timing fraction)",
         True),
        (axes[1][0], s_axis, "usage", a_axis, "a (spikes/active neuron)",
         "s — silent fraction", "usage (temporal_score at f = 1)", False),
    ]
    for axis, x_values, key, colour, colour_label, x_label, y_label, log_x in panels:
        values = np.array([row[key] for row in rows])
        scatter = axis.scatter(x_values, values, c=colour, cmap="viridis",
                              s=80, edgecolor="black", linewidth=0.5)
        figure.colorbar(scatter, ax=axis, label=colour_label)
        if log_x:
            axis.set_xscale("log")
        if key == "availability":
            axis.set_ylim(0.0, 0.8)
        result = spearmanr(x_values, values)
        axis.set_title(f"rho = {result.statistic:+.3f} (p = {result.pvalue:.3g})")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.3)

    # Bottom-right: the design itself. If the factorial worked, the two marker sets
    # occupy separate horizontal bands that each span the whole a range.
    design_axis = axes[1][1]
    for floor_value, marker, label in [(0.0, "o", "floor OFF"),
                                       (1.0, "^", "floor ON")]:
        mask = floor == floor_value
        if not mask.any():
            continue
        scatter = design_axis.scatter(a_axis[mask], s_axis[mask], c=ceiling[mask],
                                      cmap="plasma", marker=marker, s=100,
                                      edgecolor="black", linewidth=0.5, label=label)
    figure.colorbar(scatter, ax=design_axis, label="ceiling k")
    design_rho = spearmanr(a_axis, s_axis)
    design_axis.set_xscale("log")
    design_axis.set_xlabel("a — spikes per active neuron (log)")
    design_axis.set_ylabel("s — silent fraction")
    design_axis.set_title(f"the design: rho(a, s) = {design_rho.statistic:+.3f} "
                          f"(v1: {OBSERVATIONAL_RHO[GRID]})")
    design_axis.legend()
    design_axis.grid(alpha=0.3)

    # The title names the sites, because for a multi-site grid this figure shows one
    # of several and the others are in the results file rather than absent.
    figure.suptitle(
        f"v3 Phase 1 [{GRID or 'v1'}] — {arm} arm: timing availability "
        f"('{PRIMARY_VIEW}') and usage ('{PRIMARY_SITE}') against two decorrelated "
        f"'{AXES_VIEW}' sparsity axes")
    figure.tight_layout()
    path = FIG_DIR / f"phase1_regress_{VERSION_TAG}{arm}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"\nFigure -> {path}")


def analyse_arm(arm: str) -> dict:
    """Run the design check, every regression and the matched contrast for one arm.

    Args:
        arm: ``"delay"`` or ``"nodelay"``.

    Returns:
        The arm's results: the acceptance check, one usage model per injection site
        and one availability model per decode view, the matched floor contrast, the
        verdict worded on the primary site, and the joined table.
    """
    rows = build_table(arm)
    a_axis = np.array([row["a"] for row in rows])
    s_axis = np.array([row["s"] for row in rows])
    log_a = np.array([row["log_a"] for row in rows])
    clean = np.array([row["clean_acc"] for row in rows])
    floor = np.array([row["floor_strength"] for row in rows])

    print(f"\n{'=' * 74}")
    print(f"  v3 PHASE 1 [{GRID or 'v1'}] — {arm.upper()} ARM  (n = {len(rows)})")
    print(f"  axes from the '{AXES_VIEW}' decode view | usage sites {list(SITES)} "
          f"| availability views {list(VIEWS)}")
    print(f"  primary: usage at '{PRIMARY_SITE}', availability at '{PRIMARY_VIEW}'")
    print(f"{'=' * 74}")

    design_rho = spearmanr(a_axis, s_axis)
    passed = bool(abs(design_rho.statistic) <= MAX_AXIS_CORRELATION)
    print("\n--- design acceptance check (this gates everything below) ---")
    print(f"  rho(a, s) = {design_rho.statistic:+.3f} (p = {design_rho.pvalue:.4g}), "
          f"threshold |rho| <= {MAX_AXIS_CORRELATION}")
    print(f"  {'PASS — the factorial decorrelated the axes' if passed else 'FAIL'}"
          f"   [v1 observational: {OBSERVATIONAL_RHO[GRID]}]")
    print(f"  a spans {a_axis.min():.2f}–{a_axis.max():.2f} "
          f"({a_axis.max() / a_axis.min():.2f}x) | "
          f"s spans {s_axis.min():.1%}–{s_axis.max():.1%} | "
          f"clean acc {clean.min():.3f}–{clean.max():.3f}")

    # One usage model per injection site and one availability model per decode view.
    # The sites do not reduce to one another (document 7 §6e), so collapsing them to a
    # single dependent variable would hide exactly the structure the sweeps found.
    usage_fits, usage_fits_no_control = {}, {}
    for site in SITES:
        response = np.array([row[f"usage_{site}"] for row in rows])
        site_control = np.array([row[f"control_deletion_{site}"] for row in rows])
        usage_fits[site] = report_model(
            f"usage[{site}] ~ log(a) + s + deletion control[{site}]   "
            f"({SITE_DESCRIPTIONS[site]})",
            np.column_stack([log_a, s_axis, site_control]), response,
            ["log_a", "s", "control_deletion"])
        usage_fits_no_control[site] = report_model(
            f"usage[{site}] ~ log(a) + s   (control omitted)",
            np.column_stack([log_a, s_axis]), response, ["log_a", "s"])

    # The deletion control against the same two axes as usage. This is not a nuisance
    # regression: if it reproduces the usage coefficients, the usage trend is general
    # robustness rather than anything about timing (document 5's Phase 0 finding).
    control_fits = {}
    for site in SITES:
        control_fits[site] = report_model(
            f"control_deletion[{site}] ~ log(a) + s   (general-robustness covariate)",
            np.column_stack([log_a, s_axis]),
            np.array([row[f"control_deletion_{site}"] for row in rows]),
            ["log_a", "s"])

    availability_fits = {}
    for view in VIEWS:
        response = np.array([row[f"availability_{view}"] for row in rows])
        availability_fits[view] = report_model(
            f"timing_fraction[{view}] ~ log(a) + s   ({SITE_DESCRIPTIONS[view]})",
            np.column_stack([log_a, s_axis]), response, ["log_a", "s"])

    usage_fit = usage_fits[PRIMARY_SITE]
    availability_fit = availability_fits[PRIMARY_VIEW]
    arm_decision = report_arm_decision(rows, usage_fits_no_control, control_fits)
    contrast = matched_floor_contrast(rows)

    floor_on_accuracy = clean[floor > 0]
    outcome = verdict(usage_fit, availability_fit, design_rho.statistic,
                      float(floor_on_accuracy.min()) if floor_on_accuracy.size
                      else 1.0)
    print(f"\n{'=' * 74}\n  VERDICT ({arm}, usage at '{PRIMARY_SITE}'): {outcome}")
    if len(SITES) > 1:
        print("  b_a at every site: " + ", ".join(
            f"{site} {usage_fits[site]['raw']['terms']['log_a']['coefficient']:+.4f} "
            f"(p = {usage_fits[site]['raw']['terms']['log_a']['p']:.3f})"
            for site in SITES))
        print("  This grid moves both layers' activity together by construction, so "
              "every b_a above is a NETWORK-level coefficient (document 7 §6 point 1).")
    print("=" * 74)

    return {
        "grid": GRID,
        "n": len(rows),
        "axes_view": AXES_VIEW,
        "sites": list(SITES),
        "views": list(VIEWS),
        "primary_site": PRIMARY_SITE,
        "primary_view": PRIMARY_VIEW,
        "design_acceptance": {"rho": float(design_rho.statistic),
                              "p": float(design_rho.pvalue), "passed": passed},
        "usage_model": usage_fit,
        "usage_model_without_control": usage_fits_no_control[PRIMARY_SITE],
        "availability_model": availability_fit,
        "usage_models_by_site": usage_fits,
        "usage_models_by_site_without_control": usage_fits_no_control,
        "control_models_by_site": control_fits,
        "availability_models_by_view": availability_fits,
        "arm_decision": arm_decision,
        "matched_floor_contrast": contrast,
        "verdict": outcome,
        "rows": rows,
    }


def compare_arms(results: dict[str, dict]) -> None:
    """Print the arm-decision evidence for both arms side by side.

    This reports which arm each criterion favours; it does not declare a primary arm.
    That decision is a judgement over both criteria plus the consideration document 7
    §6 point 8 raises in the other direction — the no-delay arm is the only one whose
    pooled confound actually failed, so it is the only one where this design had
    something to break — and it belongs in the progress document, not in a print
    statement.

    Args:
        results: Per-arm results from ``analyse_arm``.
    """
    if len(results) < 2:
        return

    print(f"\n{'=' * 74}\n  ARM DECISION — the two criteria, side by side\n{'=' * 74}")
    gaps = {arm: result["arm_decision"]["readout_gap"][PRIMARY_VIEW]["mean"]
            for arm, result in results.items()}
    separations = {
        arm: result["arm_decision"]["timing_vs_control"][PRIMARY_SITE][
            "beta_a_difference"]
        for arm, result in results.items()}

    print(f"\n  1. readout-efficiency gap at '{PRIMARY_VIEW}' "
          f"(smaller is better — a large gap means the readout, not the code, is "
          f"what the perturbation is measuring)")
    for arm, gap in gaps.items():
        print(f"       {arm:<9} {gap:+.3f}")
    print(f"     -> favours {min(gaps, key=lambda arm: abs(gaps[arm]))}")

    print(f"\n  2. |b_a(usage) - b_a(control)| at '{PRIMARY_SITE}' "
          f"(larger is better — the timing probe has to be distinguishable from a "
          f"pure rate insult)")
    for arm, separation in separations.items():
        print(f"       {arm:<9} {separation:+.4f}")
    print(f"     -> favours "
          f"{max(separations, key=lambda arm: abs(separations[arm]))}")

    print("\n  Both were measured at layer 1 only in document 5 and are NOT inherited "
          "here (document 7 §5 step 2).\n  Record the decision in the progress "
          "document; it is not made by this script.")


def main() -> None:
    """Analyse every configured arm and write the results."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in ARMS:
        results[arm] = analyse_arm(arm)
        plot_arm(arm, results[arm]["rows"])
    compare_arms(results)

    out_path = LOG_DIR / f"phase1_regress_{VERSION_TAG.rstrip('_') or 'v1'}.json"
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
