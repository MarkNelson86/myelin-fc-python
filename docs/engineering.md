# Engineering decisions

## Data and model boundaries

A manifest maps named measurements to MATLAB files, variable keys and participant
identities. The loader validates dimensions and symmetry before extracting one
row per undirected edge. Repeated sessions share a participant ID; explicit
identity lists support other naming conventions. Parcel ordering must be verified
when preparing inputs: matrix checks cannot establish anatomical correspondence.

Separate modules handle loading, regression, participant validation, nonlinear
estimators and comparison of saved predictions. Tree and neural workflows reuse
the same fold, reference, sampling and evaluation implementation. The original
CSV group-regression workflow remains available; command details are in the
[user guide](user-guide.md).

## Evaluation choices

- Split participants rather than correlated edges or repeat scans. Inner folds
  select model settings; outer folds evaluate unseen participants.
- Recalculate FC templates, structural means and fitted scaling within each
  training fold. Outer-test FC is used only for scoring.
- Give each participant equal total fitting weight, divided among valid sessions.
  The FC template remains a training-scan mean.
- Compare saved predictions on identical participant, scan and edge keys. Reject
  mismatched targets or templates and report retained coverage.
- Include a zero-correction candidate in nonlinear selection. Model complexity
  must compete with the empirical FC template.

## Resource budget

Boosting uses shallow trees; the neural model has one 16-unit hidden layer.
Small, fixed grids and deterministic row caps bound training cost. Evaluation
uses every eligible test edge. Automatic estimator validation splits and early
stopping are disabled to avoid splitting dependent edge rows; grouped inner
validation selects the iteration budget. Fitting runs on one CPU thread.

Linear models fit network pairs separately; nonlinear models share parameters
and condition on network pair. Sampling caps also differ. The benchmark compares
bounded pipelines rather than isolating model family as the only changed factor.

## Verification and traceability

Tests cover input validation, known synthetic signal recovery, weighted Ridge
agreement with scikit-learn, session grouping and deterministic sampling. Leakage
checks alter outer-test FC and verify that predictions and model selection for
that fold remain unchanged. The installed public demo exercises all model
families without private inputs. CI is configured for Python 3.10 and 3.12.

Runs export input hashes, fold assignments, settings, predictions and metrics.
Selected neural models export numeric weights and scaling as NPZ files. Floating
point results can vary across dependency versions and platforms; hashes identify
input bytes rather than guaranteeing identical numeric results.

## Current limits

Participant counts remain small despite many edge rows. Evaluation is exploratory
within each cohort, with no validated cross-cohort transfer. MATLAB v7.3 inputs
require conversion; dependency locking and production deployment are outside the
current implementation. Demo and cohort/nonlinear commands reject nonempty output
directories; legacy group commands can overwrite outputs. Use fresh run paths.
