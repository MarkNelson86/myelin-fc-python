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

Completed: group and subject OLS/Ridge; nested participant folds; synthetic
end-to-end demo; provenance outputs; automated tests and CI configuration.

Completed extension: template-deviation OLS/Ridge and matched-edge comparisons
for MWC tract-specific g-ratio, MTsat and R1. Next: adapt the same data contract
to MICs using shared R1;
add one gradient-boosting benchmark under the same participant splits. A small
neural network and Paper 3 communication features are later extensions.

The deliverable is reusable, reviewable analysis software. Each added experiment
should demonstrate an engineering or modeling capability and have a defined
stopping point. Scientific improvement is assessed honestly rather than used
as a requirement for completing the repository.
