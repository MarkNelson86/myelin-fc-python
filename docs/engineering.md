# Engineering notes

## Data flow

The manifest maps named features to MATLAB files and variable keys. The loader
validates scan identities, dimensions and matrix symmetry. Modeling extracts
one upper-triangle row per undirected edge. Participant identity, not scan
identity, determines all model-selection splits. The run exports explicit
fold assignments, fitted scaling, coefficients, predictions, metrics and input
hashes so each result can be traced to its inputs and settings.

## Why these choices

- Explicit mappings prevent accidental feature or subject-order inference.
- Canonical participant IDs group the current sub-NN/sub-NNr sessions. Other
  naming schemes can use explicit participant_ids in the manifest.
- An empirical training-FC mean is a useful benchmark for shared edge patterns.
- Group mode averages training connectomes. Subject mode pools subject-edge
  rows and gives participants equal total fitting weight per network pair.
- Inner participant folds choose Ridge alpha. Outer participants provide the
  exploratory evaluation. All fitted preprocessing is rebuilt within folds.
- Deterministic synthetic MATLAB files exercise the actual loader and model,
  allowing the project to run without participant data.
- Output-directory protection in the demo preserves previous runs. The research
  modeling command currently permits output reuse: choose a fresh directory.

## Testing and reproducibility

The tests include regression fixtures, data validation, held-out-data
invariance, grouped participant splits, weighted Ridge equivalence with
scikit-learn, participant/session weighting, deterministic synthetic inputs and
an end-to-end demo. CI is configured to install the package, run tests and run
the installed demo command on Python 3.10 and 3.12. CI results become available
when the workflow runs in the GitHub repository; local validation does not
establish those remote checks have passed.

Demo randomness and generated input bytes are deterministic for a fixed seed
and dependency environment. Floating-point results may differ slightly between
platforms. Input hashes describe exact input bytes, while run configuration
records modeling settings. Full dependency/environment locking is future work.

## Scope and limitations

This is research analysis software. The subject evaluation currently uses five
real held-out participants with paired scans and is exploratory because these
subjects have already been inspected. The MWC functional data also have a known
registration limitation. Dependent edges and repeated scans are
not independent participants. Exact numerical replication of the original
MATLAB paper is a separate task; the predictive Python baseline changes
preprocessing and keeps fitted steps within training folds. MATLAB v7.3 input
requires conversion to v7 or a future HDF5 adapter.

The loader validates matrix and ID structure, but does not prove that a file's
parcellation and edge order match an external label table. That correspondence
must be established when preparing data. Multiple training sessions receive
equal participant fitting weight in subject mode; the FC template remains a
scan mean. The current MWC main stack has one retained scan per participant.

## Portfolio roadmap

Completed: validated loaders, group and subject OLS/Ridge, residual models,
MWC three-metric comparisons, MICs nested participant evaluation, bounded
boosting and a small neural network. The synthetic public demo exercises all
model families. Configurations, input hashes, folds and sampling hashes support
review and reproduction. Benchmark results and limitations are consolidated in
[the report](benchmark-report.md).

Next work should prioritize repository review, a clean public release and a
brief employer-facing walkthrough. Paper 3 communication features remain a
separate optional project rather than an unfinished prerequisite.

The deliverable is reusable, reviewable analysis software. Each added experiment
should demonstrate an engineering or modeling capability and have a defined
stopping point. Scientific improvement is assessed honestly rather than used
as a requirement for completing the repository.

## Nonlinear benchmark design

The bounded tree benchmark learns FC departures from a training-only empirical
FC template using departures from training-only structural edge means. It fits
one histogram gradient boosting ensemble with three numeric predictors and a
categorical network-pair feature. The linear benchmark fits separate models per
network pair, so this experiment changes both model family and parameter sharing.
Performance differences cannot be attributed solely to nonlinearity.

Sampling caps training rows per participant; fitting weights give each person
equal total weight and divide that weight among valid sessions. Edge samples are
deterministic and logged by index hash. Evaluation does not sample test edges.
Independent sample size remains the number of participants, regardless of how
many edges are sampled. A small, fixed iteration grid includes a zero-correction
baseline. Inner validation and outer testing split participants; all references
are recalculated within the relevant training fold. Estimator early stopping is
disabled to avoid an automatic split of correlated edge rows. CPU use is limited
to one thread. These choices favor reproducibility and bounded runtime over an
expansive tuning search.

Within-cohort MICs validation is exploratory; it does not demonstrate transfer
from MWC to MICs. No neural network is warranted merely because more edge rows
are available. The small neural benchmark uses the same shared participant protocol with
weighted scaling, one-hot network pair and fixed epoch candidates. Its smaller
row cap and different architecture mean it is an alternative bounded pipeline;
the comparison does not isolate architecture from fitting budget.
