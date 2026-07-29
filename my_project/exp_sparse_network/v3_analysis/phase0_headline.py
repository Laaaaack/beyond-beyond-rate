"""v3 Phase 0: availability and usage, against both sparsity axes.

Joins everything Phase 0 measured into one table per arm and asks the two questions
the milestone actually turns on, keeping *availability* and *usage* separate
throughout:

============  =========================================  ==========================
quantity      what it is                                  source
============  =========================================  ==========================
``a``         spikes per **active** neuron per sample      hidden_channel_decode.py
``s``         fraction of (sample, neuron) pairs silent    hidden_channel_decode.py
availability  ``timing_fraction``: the share of the        hidden_channel_decode.py
              layer's decodable class information that
              survives only if spike placement is known
usage         ``temporal_score`` at ``f = 1``: how much    shd eval, corrected
              of the network's own accuracy the readout       window
              loses when hidden timing is destroyed
control       the same score under spike *deletion*,       deletion eval
              which is general robustness, not timing
============  =========================================  ==========================

Two things this script exists to make unmissable.

1. ``spikes_per_neuron`` is the product ``(1 - s) * a`` of two variables that push
   the code in opposite directions, so it is reported but never analysed alone.
   Every plot colours its points by ``s`` and every correlation against ``a`` is
   also reported partialled on ``s``, so a reader can see directly whether the data
   support attributing an effect to one axis rather than the other.
2. Availability and usage are different measurements and are free to disagree.
   Reporting either alone is how v1 became hard to read.

Because ``a`` and ``s`` are strongly collinear in v1's checkpoints, the partial
correlations here are *diagnostic, not decisive* — that is exactly the confound
Phase 1's factorial exists to break. ``rho(a, s)`` is printed first for that reason.

Writes ``v3_analysis/log/phase0_headline_{arm}.csv``,
``v3_analysis/log/phase0_headline.json`` and ``v3_analysis/fig/phase0_headline.png``.
"""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as t_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "log"
FIG_DIR = SCRIPT_DIR / "fig"

NUM_CLASSES: int = 20
CHANCE: float = 1.0 / NUM_CLASSES

# Endpoints of the two perturbation sweeps, as string keys in their result files.
RELOCATION_ENDPOINT: str = "1.0"
DELETION_ENDPOINT: str = "0.8"

# The corrected relocation sweep. The uncorrected one is loaded alongside so the
# size of the padding-window artifact stays visible in the output.
CORRECTED_SUFFIX: str = "_v3window"

ARMS: tuple[str, ...] = ("nodelay", "delay")


def robustness_score(clean_acc: float, perturbed_acc: float) -> float:
    """Return v1's chance-corrected, baseline-normalised perturbation score.

    0 means the perturbation cost nothing; 1 means it drove the network to chance.
    Normalising by each network's own headroom is what makes the score comparable
    across checkpoints whose clean accuracies differ.

    Args:
        clean_acc: Unperturbed test accuracy.
        perturbed_acc: Test accuracy at the sweep's endpoint.

    Returns:
        The score, higher meaning more damage done.
    """
    return 1.0 - (perturbed_acc - CHANCE) / max(1e-9, clean_acc - CHANCE)


def endpoint_score(sweep: dict, endpoint: str) -> float:
    """Score one checkpoint's sweep at its strongest setting.

    Args:
        sweep: The ``f_sweep`` / ``pd_sweep`` dict of setting -> repeat statistics.
        endpoint: Key of the strongest setting.

    Returns:
        The robustness score at that setting.
    """
    clean_key = next(key for key in sweep if float(key) == 0.0)
    return robustness_score(sweep[clean_key]["mean"], sweep[endpoint]["mean"])


def load_json(path: Path) -> dict | None:
    """Load a JSON file, or return None if it has not been produced yet."""
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    with open(path) as handle:
        return json.load(handle)


def sweep_scores(results: dict | None, sweep_key: str,
                 endpoint: str) -> dict[str, float]:
    """Reduce a whole perturbation-sweep result file to one score per checkpoint."""
    if results is None:
        return {}
    return {run_tag: endpoint_score(row[sweep_key], endpoint)
            for run_tag, row in results.items()}


def build_arm_table(arm: str) -> list[dict]:
    """Join the decode, relocation and deletion measurements for one arm.

    Args:
        arm: ``"nodelay"`` or ``"delay"``.

    Returns:
        One row per checkpoint, ordered as the decode results were written.
    """
    decode = load_json(LOG_DIR / f"hidden_channel_decode_{arm}.json")
    if decode is None:
        raise FileNotFoundError(
            f"Run hidden_channel_decode.py first — no decode results for {arm}.")

    relocation_dir = EXP_DIR / "shd/log"
    corrected = sweep_scores(
        load_json(relocation_dir
                  / f"sparse_whole_{arm}_shd_eval{CORRECTED_SUFFIX}.json"),
        "f_sweep", RELOCATION_ENDPOINT)
    uncorrected = sweep_scores(
        load_json(relocation_dir / f"sparse_whole_{arm}_shd_eval.json"),
        "f_sweep", RELOCATION_ENDPOINT)
    deletion = sweep_scores(
        load_json(EXP_DIR / f"deletion/log/sparse_whole_{arm}_deletion_eval.json"),
        "pd_sweep", DELETION_ENDPOINT)

    rows = []
    for run_tag, row in decode.items():
        rows.append({
            "run_tag": run_tag,
            "spikes_per_neuron": row["spikes_per_neuron"],
            "a_spikes_per_active": row["spikes_per_active_neuron"],
            "s_silent_fraction": row["silent_fraction"],
            "decode_count": row["decode_count"],
            "decode_identity": row["decode_identity"],
            "decode_full": row["decode_full"],
            "timing_information": row["timing_information"],
            "availability_timing_fraction": row["timing_fraction"],
            "usage_relocation_corrected": corrected.get(run_tag),
            "usage_relocation_full_window": uncorrected.get(run_tag),
            "control_deletion": deletion.get(run_tag),
        })
    return rows


def column(rows: list[dict], key: str) -> np.ndarray:
    """Return one column as a float array, with missing entries as NaN."""
    return np.array([np.nan if row[key] is None else float(row[key])
                     for row in rows])


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     controls: list[np.ndarray]) -> tuple[float, float, int]:
    """Spearman correlation of x and y with the controls' rank effects removed.

    Ranks every variable, then regresses x and y on the ranked controls and
    correlates the residuals. This is the standard rank partial correlation; with
    ``a`` and ``s`` collinear it is a diagnostic rather than a clean answer, which
    is why it is always reported next to ``rho(a, s)``.

    The p-value is the usual t test on the partial coefficient, with one degree of
    freedom spent per control. At n = 12-15 with two controls there is very little
    left, so a non-significant partial here is weak evidence of absence.

    Args:
        x: First variable.
        y: Second variable.
        controls: Variables to hold fixed.

    Returns:
        Tuple of (partial rho, two-sided p-value, sample size). Rho and p are NaN
        if the inputs are degenerate or too few.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    for control in controls:
        finite &= np.isfinite(control)
    sample_size = int(finite.sum())
    degrees_of_freedom = sample_size - 2 - len(controls)
    if degrees_of_freedom < 1:
        return float("nan"), float("nan"), sample_size

    design = np.column_stack(
        [np.ones(sample_size)] + [rankdata(c[finite]) for c in controls])
    residuals = []
    for variable in (x, y):
        ranked = rankdata(variable[finite])
        coefficients, *_ = np.linalg.lstsq(design, ranked, rcond=None)
        residuals.append(ranked - design @ coefficients)
    if np.std(residuals[0]) == 0 or np.std(residuals[1]) == 0:
        return float("nan"), float("nan"), sample_size

    rho = float(np.corrcoef(residuals[0], residuals[1])[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0, sample_size
    t_statistic = rho * np.sqrt(degrees_of_freedom / (1.0 - rho ** 2))
    p_value = float(2.0 * t_distribution.sf(abs(t_statistic),
                                            degrees_of_freedom))
    return rho, p_value, sample_size


def report_correlations(arm: str, rows: list[dict]) -> dict:
    """Print and return the correlations Phase 0 is meant to deliver.

    Args:
        arm: Arm name, for the printed heading.
        rows: The arm's joined table.

    Returns:
        Dict of every reported statistic, for the JSON output.
    """
    spikes = column(rows, "spikes_per_neuron")
    a_axis = column(rows, "a_spikes_per_active")
    s_axis = column(rows, "s_silent_fraction")
    availability = column(rows, "availability_timing_fraction")
    usage = column(rows, "usage_relocation_corrected")
    usage_old = column(rows, "usage_relocation_full_window")
    control = column(rows, "control_deletion")

    log_a = np.log(a_axis)
    stats: dict[str, dict] = {}

    def record(name: str, x: np.ndarray, y: np.ndarray) -> None:
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            stats[name] = {"rho": None, "p": None, "n": int(finite.sum())}
            print(f"  {name:<46} (no data)")
            return
        result = spearmanr(x[finite], y[finite])
        stats[name] = {"rho": float(result.statistic),
                       "p": float(result.pvalue), "n": int(finite.sum())}
        print(f"  {name:<46} rho={result.statistic:+.3f} "
              f"p={result.pvalue:<9.3g} n={finite.sum()}")

    print(f"\n=== {arm} (n={len(rows)}) ===")
    print("-- the confound: the two sparsity axes against each other --")
    record("rho(a, s)", a_axis, s_axis)
    record("rho(spikes_per_neuron, a)", spikes, a_axis)
    record("rho(spikes_per_neuron, s)", spikes, s_axis)

    print("-- availability: does the layer HOLD more timing information? --")
    record("rho(timing_fraction, a)", a_axis, availability)
    record("rho(timing_fraction, s)", s_axis, availability)
    record("rho(timing_fraction, spikes_per_neuron)", spikes, availability)

    print("-- usage: does the readout DEPEND on hidden timing? --")
    record("rho(usage, a)", a_axis, usage)
    record("rho(usage, s)", s_axis, usage)
    record("rho(usage, spikes_per_neuron)", spikes, usage)
    record("rho(usage_full_window, spikes_per_neuron)", spikes, usage_old)
    record("rho(deletion control, spikes_per_neuron)", spikes, control)

    print("-- the immune channels, as measured covariates --")
    record("rho(decode_count, spikes_per_neuron)", spikes,
           column(rows, "decode_count"))
    record("rho(decode_identity, spikes_per_neuron)", spikes,
           column(rows, "decode_identity"))

    print("-- partials (diagnostic only while a and s are collinear) --")
    partials = {
        "partial rho(timing_fraction, log a | s)":
            partial_spearman(log_a, availability, [s_axis]),
        "partial rho(usage, log a | s)":
            partial_spearman(log_a, usage, [s_axis]),
        "partial rho(usage, s | log a)":
            partial_spearman(s_axis, usage, [log_a]),
        "partial rho(usage, log a | s, deletion)":
            partial_spearman(log_a, usage, [s_axis, control]),
        "partial rho(usage, s | log a, deletion)":
            partial_spearman(s_axis, usage, [log_a, control]),
    }
    for name, (rho, p_value, sample_size) in partials.items():
        print(f"  {name:<46} rho={rho:+.3f} p={p_value:<9.3g} n={sample_size}")

    stats["partials"] = {
        name: {"rho": None if not np.isfinite(rho) else rho,
               "p": None if not np.isfinite(p_value) else p_value,
               "n": sample_size}
        for name, (rho, p_value, sample_size) in partials.items()}
    return stats


def audit_window_correction(arm: str) -> dict:
    """Report what confining the perturbation window to the support actually did.

    Phase 0's correctness fix has to be shown to be a fix rather than a rescue, so
    for each corrected perturbation this reports the accuracy shift it produced at
    the sweep's endpoint and the headline correlation before and after. Relocation
    scatters spikes across the whole window and is strongly affected; jitter is
    local and is barely affected; deletion and shift choose no destination bin and
    are not corrected at all.

    Args:
        arm: ``"nodelay"`` or ``"delay"``.

    Returns:
        Dict per perturbation of accuracy shifts and the two correlations.
    """
    perturbations = [
        ("relocation", EXP_DIR / "shd/log", f"sparse_whole_{arm}_shd_eval",
         "f_sweep", RELOCATION_ENDPOINT),
        ("jitter", EXP_DIR / "jitter/log", f"sparse_whole_{arm}_jitter_eval",
         "sigma_sweep", "25"),
    ]
    print(f"\n-- window correction audit ({arm}) --")
    audit = {}
    for name, directory, stem, sweep_key, endpoint in perturbations:
        full = load_json(directory / f"{stem}.json")
        support = load_json(directory / f"{stem}{CORRECTED_SUFFIX}.json")
        if full is None or support is None:
            continue

        clean_key = next(key for key in full[next(iter(full))][sweep_key]
                         if float(key) == 0.0)
        shifts, rates, before, after = [], [], [], []
        for run_tag, row in full.items():
            clean = row[sweep_key][clean_key]["mean"]
            full_acc = row[sweep_key][endpoint]["mean"]
            support_acc = support[run_tag][sweep_key][endpoint]["mean"]
            shifts.append(support_acc - full_acc)
            rates.append(row["spikes_per_neuron"])
            before.append(robustness_score(clean, full_acc))
            after.append(robustness_score(clean, support_acc))

        rho_before = spearmanr(rates, before)
        rho_after = spearmanr(rates, after)
        audit[name] = {
            "endpoint": endpoint,
            "accuracy_shift_min": float(np.min(shifts)),
            "accuracy_shift_max": float(np.max(shifts)),
            "rho_full_window": float(rho_before.statistic),
            "p_full_window": float(rho_before.pvalue),
            "rho_support_window": float(rho_after.statistic),
            "p_support_window": float(rho_after.pvalue),
        }
        print(f"  {name:<11} acc shift {np.min(shifts):+.3f}..{np.max(shifts):+.3f} "
              f"| rho(spikes/neuron, score) {rho_before.statistic:+.3f} -> "
              f"{rho_after.statistic:+.3f}")
    return audit


def write_table(arm: str, rows: list[dict]) -> None:
    """Write one arm's joined measurements as CSV."""
    path = LOG_DIR / f"phase0_headline_{arm}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  table -> {path}")


def plot_headline(tables: dict[str, list[dict]]) -> None:
    """Plot availability and usage against ``a``, with ``s`` as point colour.

    One row per arm, one column per dependent variable. Colouring by ``s`` is the
    point of the figure: it puts the confound on the page instead of hiding it
    behind a single ``spikes_per_neuron`` axis.

    Args:
        tables: Arm name -> that arm's joined table.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Both dependent variables are fractions on the same 0-1 footing, so both
    # panels are drawn on a fixed y range. Autoscaling the availability panel to
    # its own ~0.05-wide spread would magnify sampling noise into what looks like
    # structure, which is precisely the reading this figure has to prevent.
    panels = [("availability_timing_fraction",
               "availability\n(timing fraction of decodable information)",
               (0.0, 0.8)),
              ("usage_relocation_corrected",
               "usage\n(temporal_score at f = 1, support window)",
               (0.0, 0.8))]

    figure, axes = plt.subplots(len(tables), len(panels),
                                figsize=(11, 4.4 * len(tables)), squeeze=False)
    for row_index, (arm, rows) in enumerate(tables.items()):
        a_axis = column(rows, "a_spikes_per_active")
        s_axis = column(rows, "s_silent_fraction")
        for col_index, (key, label, y_limits) in enumerate(panels):
            axis = axes[row_index][col_index]
            values = column(rows, key)
            finite = np.isfinite(values)
            if not finite.any():
                axis.text(0.5, 0.5, "not measured yet", ha="center",
                          va="center", transform=axis.transAxes)
            else:
                scatter = axis.scatter(a_axis[finite], values[finite],
                                       c=s_axis[finite], cmap="viridis",
                                       s=70, edgecolor="black", linewidth=0.5)
                figure.colorbar(scatter, ax=axis,
                                label="s (silent fraction)")
                result = spearmanr(a_axis[finite], values[finite])
                axis.set_title(f"{arm}: rho = {result.statistic:+.3f} "
                               f"(p = {result.pvalue:.3g}, n = {finite.sum()})")
            axis.set_xscale("log")
            axis.set_ylim(*y_limits)
            axis.set_xlabel("a — spikes per active neuron (log scale)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)

    figure.suptitle("v3 Phase 0 — what the hidden layer holds vs what the readout "
                    "uses,\nagainst temporal sparsity, coloured by selectivity")
    figure.tight_layout()
    path = FIG_DIR / "phase0_headline.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"\nFigure -> {path}")


def main() -> None:
    """Join every Phase 0 measurement, report the correlations and plot them."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tables, stats = {}, {}
    for arm in ARMS:
        print(f"\nLoading {arm} ...")
        rows = build_arm_table(arm)
        tables[arm] = rows
        write_table(arm, rows)
    for arm, rows in tables.items():
        stats[arm] = report_correlations(arm, rows)
        stats[arm]["window_correction_audit"] = audit_window_correction(arm)

    out_path = LOG_DIR / "phase0_headline.json"
    with open(out_path, "w") as handle:
        json.dump({"per_arm_statistics": stats,
                   "per_arm_rows": tables}, handle, indent=2)
    print(f"\nStatistics -> {out_path}")
    plot_headline(tables)


if __name__ == "__main__":
    main()
