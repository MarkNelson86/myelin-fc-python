# Predicting individual functional connectivity: benchmark report

## Result

Across the MICs benchmark, the empirical mean FC of training participants
outperformed the tested structural prediction models. More expressive models
did not produce better prediction of unseen participants. This supports retaining
a strong baseline and reporting limitations rather than expanding the search.

| MICs model | Mean RMSE | Mean correlation | R2 vs template |
|---|---:|---:|---:|
| Training FC template | **0.20659** | 0.86244 | 0.00000 |
| Residual OLS | 0.20747 | 0.85995 | -0.01109 |
| Residual Ridge | 0.20730 | 0.86167 | -0.00755 |
| Residual gradient boosting | 0.20767 | 0.86155 | -0.01162 |
| Residual small neural network | 0.20816 | 0.86160 | -0.01606 |

Linear and tree values above use the user's locally reproduced comparison.
Neural values use the fixed protocol run in the development environment.
The local and development tree RMSE differed by approximately 0.000006,
without changing selected iterations or conclusions. Full local three-family
comparison is generated from saved predictions using the command below.
Reported R2 is mean participant error improvement over the template, not an
ordinary group regression R2. Correlation with full FC mostly reflects shared
edge patterns and does not establish reliable prediction of individual FC.

## Question and data

Can a participant's structural connectome improve FC prediction beyond shared
FC patterns? The case study uses tract caliber, myelin-sensitive measurements,
tract length and static BOLD FC on the Schaefer-400 parcellation.

MWC has 26 retained main scans and a five-participant, two-session evaluation
stack. Every scan of an evaluated person is excluded from training. Exploratory
MTsat residual OLS slightly improved the template; R1 and tract-specific g-ratio
did not show a clear advantage. The previously inspected evaluation participants
and known functional-registration limitation constrain interpretation.

MICs supplies 45 aligned unique participant stacks with caliber, R1, length and
FC, with no rescans. Sequential labels identify stack position; the supplied
50-ID list is deliberately unused. Parcellation ordering was confirmed by the
data owner. The supplied group matrices are not used in fitting.

## Evaluation contract

Five fixed outer participant folds evaluate every MICs person once (36 training,
9 test each). Five inner participant folds select the regularization or iteration
budget. Seed 20261006 fixes splits. All FC templates, structural edge references
and neural scaling are rebuilt using only the appropriate training fold.

Residual models predict a departure from the training FC template using
structural departures from training edge means. Zeros in structural measurements
mean unavailable connections; zero departures remain valid. FC zero is valid.
All eligible held-out edges are scored. The comparison checks matching empirical
FC and training templates, intersects eligible edges and averages metrics within
participants, then across participants. The completed linear/tree comparison
contains 655,216 edge rows, with 100% retained coverage for each run.

## Bounded alternatives

| Pipeline | Fixed budget | Architecture |
|---|---|---|
| Residual OLS/Ridge | Existing alpha grid, all fitting rows | Separate models per network pair; three structural predictors and interactions |
| Gradient boosting | 0/30/60 iterations; <=2,500 rows/person | Shared histogram tree ensemble, categorical network pair, depth 3 |
| Small neural network | 0/10/30 epochs; <=1,000 rows/person | Shared 16-unit tanh layer, one-hot pair, weighted feature/target scaling, Adam |

The nonlinear pipelines use equal total participant fitting weights, divided
among valid sessions. Sampling is deterministic and logged by edge-index hash.
Both include a zero-correction candidate that predicts the template exactly.
Automatic estimator validation splits and early stopping are disabled; grouped
inner folds perform selection. Neural fixed epoch budgets do not claim optimizer
convergence. scikit-learn >=1.7 supports the required MLP sample weights.

Trees selected 30 iterations in two outer folds and the template in three.
The neural pipeline selected 10 epochs in one fold and the template in four.
Architecture, parameter sharing, input representation and fitting row budgets
differ, so performance differences cannot be attributed solely to nonlinearity.
Many correlated edges do not increase the independent participant count.
These exploratory within-cohort results do not demonstrate MWC-to-MICs transfer.

## Public reproduction

```bash
python -m pip install -e ".[dev,ml]"
python -m pytest -q
myelin-fc-demo --all-models --out out/public-demo
```

No private inputs or downloads are required. Open
out/public-demo/benchmark/summary.csv for a common-edge comparison of all model
families. This synthetic benchmark uses smaller budgets and a separate main-stack
cohort protocol from the original scan-rescan demonstration. Its metrics are
software demonstration outputs, not real scientific results.

For an existing real-data run:

```bash
python -m myelinfccoupling.compare_runs \
  --run linear=out/mics/residual/predictions.csv \
  --run boosting=out/mics/boosting/predictions.csv \
  --run neural=out/mics/neural/predictions.csv \
  --out out/mics/all-model-comparison
```

## Engineering evidence and scope

The package provides explicit data contracts, CLI entry points, grouped tuning,
training-only preprocessing, participant-balanced fitting, input hashes, fold
assignments and deterministic sample logs. Selected neural models export numeric
weights and fitted scaling as NPZ files; residual references are saved separately.
Tests exercise known synthetic signal learning, deterministic behavior, session
grouping, matched-edge scoring and invariance to changes in outer-test FC.
CI is configured for Python 3.10 and 3.12 and runs the installed all-model demo;
local checks do not imply the remote workflow has already passed.

This is research analysis software and a reproducible portfolio case study.
It does not claim exact MATLAB-paper reproduction, clinically useful individual
prediction, production deployment, or validated cross-cohort transfer. With the
bounded model extensions complete, the useful next step is a clean repository
review and a short employer-facing walkthrough. Paper 3 communication modeling
can remain a separate project.
