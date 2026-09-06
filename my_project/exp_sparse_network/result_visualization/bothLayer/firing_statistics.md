# Network-wide firing statistics — v3L12 both-layer grid

Both hidden layers pooled, mean ± sd over seeds 42/43/44. `none` is the **non-sparse
baseline**: the same architecture trained with neither penalty, one checkpoint per arm.
Only the floor-**ON** column of the factorial is listed.

|        Arm | k1/k2 | Floor |    n | Clean acc |         Hz |    sp/neu |    sp/act |   silent% |
| :---       | ---:  | :---: | ---: | ---:      | ---:       | ---:      | ---:      | ---:      |
|   No Delay |  none |  none |    1 |     57.3% |      30.72 |      6.14 |     11.53 |     46.7% |
|   No Delay |   1/1 |    ON |    3 | 52.5±1.7% |  8.94±0.15 | 1.79±0.03 | 2.16±0.01 | 17.4±0.9% |
|   No Delay |   2/2 |    ON |    3 | 53.5±0.3% | 10.89±0.27 | 2.18±0.05 | 2.46±0.06 | 11.6±0.3% |
|   No Delay |   4/4 |    ON |    3 | 57.2±1.3% | 14.69±0.10 | 2.94±0.02 | 3.17±0.00 |  7.2±0.5% |
|   No Delay |   8/8 |    ON |    3 | 56.8±1.1% | 21.45±0.24 | 4.29±0.05 | 4.49±0.04 |  4.4±0.2% |
| With Delay |  none |  none |    1 |     87.1% |      65.86 |     13.17 |     18.63 |     29.3% |
| With Delay |   1/2 |    ON |    3 | 78.3±2.3% | 12.28±0.09 | 2.46±0.02 | 2.93±0.02 | 16.2±0.2% |
| With Delay |   2/4 |    ON |    3 | 78.2±0.9% | 14.76±0.11 | 2.95±0.02 | 3.32±0.03 | 11.2±0.1% |
| With Delay |   4/8 |    ON |    3 | 80.7±0.7% | 20.80±0.17 | 4.16±0.03 | 4.44±0.04 |  6.4±0.3% |
| With Delay |  8/16 |    ON |    3 | 84.0±0.4% | 31.81±0.81 | 6.36±0.16 | 6.59±0.16 |  3.5±0.2% |

**Columns.** `Hz` is the average neuron firing rate — spikes per neuron per second over
the 200 × 1 ms simulation window; `sp/neu` is the mean spike count per (sample, neuron)
pair; `sp/act` is that count over non-silent pairs only (the availability axis `a`); and
`silent%` is the silence ratio `s`, the pairs that fired no spike at all. `Hz` and
`sp/neu` are the same quantity in different units and both are the product `(1 − s) × a`,
so read `sp/act` and `silent%` together — never `sp/neu` alone.

**Sources.** Grid rows: `sn_log/sparse_whole_{arm}_v3L12_train_summary.json`, measured
on the test set after the 1250-epoch run by `evaluate_firing_statistics()` in
`sn_bothLayer_train_{noDelay,withDelay}_v3.py`. Baseline rows:
`v3_analysis/log/baseline_firing_statistics.json`, written by
`v3_analysis/baseline_firing_statistics.py` from
`exp_fixed_weight_perturbation/code/perturbation/jitter/data/jitter_whole_{arm}_trained.pt`
with the same architecture, data, test split `[0.75, 0.9)` and reductions. Full per-layer
and per-checkpoint tables are in [firing_statistics.ipynb](firing_statistics.ipynb).

**What it shows.** The penalties do not simply turn the network down. Pooled firing falls
by 1.4–5.4× against the baseline and `sp/act` by 2.6–6.4×, but `silent%` moves the
*other* way: every floor-ON cell silences **fewer** neurons than its baseline (4.4–17.4%
against 46.7% in the no-delay arm, 3.5–16.2% against 29.3% with delay). That is the floor
doing its job — it keeps neurons alive while the ceiling caps what each one may spend.
