# Participant prediction benchmark

## Question

Do structural measurements improve prediction of individual functional
connectivity beyond the shared pattern captured by a training FC mean?

## MICs result

| Model | Mean RMSE | Mean FC correlation | R2 vs template |
|---|---:|---:|---:|
| Training FC template | 0.20659 | 0.86244 | 0.00000 |
| Residual OLS | 0.20747 | 0.85995 | -0.01109 |
| Residual Ridge | 0.20730 | 0.86167 | -0.00755 |
| Residual boosting | 0.20767 | 0.86155 | -0.01162 |
| Residual neural network | 0.20816 | 0.86160 | -0.01606 |

The tested models did not outperform the training FC mean. High full-FC
correlations largely reflect shared patterns and do not establish accurate
prediction of individual departures. Negative R2 vs template means greater
squared error than the baseline; it is not ordinary regression R2.

## Protocol

- **Data:** 45 unique MICs participants; Schaefer-400 connectomes with caliber,
  R1, tract length and static BOLD FC.
- **Splits:** five outer participant folds, each with 36 training and 9 test
  people; five inner folds select settings. Each participant is tested once.
- **References:** FC templates, structural means and neural scaling are fitted
  using only the relevant training fold.
- **Target:** residual models predict individual departures from the training
  FC template, then add the template back.
- **Scoring:** all models use the same 655,216 eligible edge rows, with 100%
  retained coverage. Metrics are averaged within participants, then across people.

## Bounded model choices

| Model | Selection budget | Fitting design |
|---|---|---|
| OLS/Ridge | Ridge alpha: 11 values, 1e-4 to 1e6 | Separate network-pair models with interaction terms |
| Boosting | 0/30/60 iterations | Shared shallow ensemble; up to 2,500 rows per person |
| Neural network | 0/10/30 epochs | Shared 16-unit tanh layer with Adam; up to 1,000 rows per person |

Nonlinear models include network-pair metadata and use participant-balanced
weights. Zero iterations/epochs predicts the template alone. Inner validation
selected the template in three boosting folds and four neural folds. The
remaining folds selected 30 boosting iterations or 10 neural epochs.

## Interpretation and scope

These are exploratory results for unseen participants within MICs. Model
architectures, parameter sharing and fitting budgets differ, so the comparison
does not isolate nonlinearity. More edge rows do not create more independent
participants, and a failed prediction benchmark does not prove the relevant
information is absent from structural data.

An additional MWC case study covers MTsat, R1 and tract-specific g-ratio, with
five evaluated participants and paired scans. A small MTsat residual improvement
was not consistent across metrics; prior inspection and a known FC registration
limitation restrict its interpretation. No cross-cohort transfer is claimed.

## Reproduction

The [README](../README.md) provides the public synthetic demo. Its smaller
budgets and generated inputs exercise the software; they are not the real-data
benchmark. Real-data CLI commands, input conventions and output definitions are
in the [user guide](user-guide.md). The table above records the locally reproduced
all-model comparison.
