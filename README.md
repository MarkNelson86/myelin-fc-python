# Myelin–FC coupling in Python

A reproducible Python pipeline for predicting functional connectivity from
structural connectomes. It combines data validation, participant-aware model
selection, OLS/Ridge comparisons, and auditable CSV outputs. The neuroscience
case study illustrates predictive modeling with many dependent observations
and relatively few independent participants.

## Try the synthetic subject demo

Python 3.10 or newer; from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ml]"
myelin-fc-demo --out out/demo
```

The demo requires no downloads or participant data. It generates 24-node
connectomes for ten synthetic people, including paired sessions for two held-out
people, and runs three inner participant folds for Ridge selection. It writes
`summary.csv`, `DEMO_REPORT.md`, subject audits, fold exclusions, and a `models/`
folder containing predictions, coefficients, scaling, selected alphas, input
hashes and run configuration. A nonempty output folder is rejected; choose a
new `--out` path for another run. Use `--seed` to generate a different fixture.
Demo metrics illustrate the workflow and carry no scientific interpretation.

## What the project demonstrates

| Capability | Implementation |
|---|---|
| Reusable Python software | Installable package, CLI commands, modular loaders and models |
| Data contracts | Explicit file/variable mappings, matrix shape and symmetry checks, identity validation |
| Leakage control | Every session of a held-out participant excluded; nested participant folds; training-only preprocessing |
| Model evaluation | Absolute-FC and template-deviation OLS/Ridge; FC template benchmark; matched-edge comparisons |
| Reproducibility | Deterministic fixture generation, explicit seeds, saved configurations and SHA-256 input hashes |
| Automated checks | Unit and integration tests; CI configured for Python 3.10 and 3.12 |

See [engineering notes](docs/engineering.md) for design decisions, limitations,
and the portfolio development roadmap. The group-regression workflow and
subject-data analyses are documented below. Private participant data belongs
in the ignored `data/subjects/` directory; generated runs belong in `out/`.

## 1. Install

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
myelin-fc-run --help
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. The `dev` extra installs testing and build tools; use `python -m pip install -e .` for analysis only.

## 2. Run the supplied example

```bash
myelin-fc-run \
  --edges data/edges_fc_BOLDin.csv \
  --out out \
  --fc-label BOLD
```

This input contains 18,332 undirected edges. Results appear in `out/BOLD/`. Expected global R² values are approximately **0.3431063373** for the full model and **0.3287746533** for the reduced model.

To use synthetic data instead:

```bash
python examples/make_demo_data.py
myelin-fc-run --edges data/demo_edges.csv --out out --fc-label DEMO
```

The synthetic example has 24 nodes and 276 edges and is intended to check installation and execution.

## 3. Prepare your own data

Supply a CSV with one row per undirected edge. Either orientation is allowed, but include each edge once and exclude self-loops.

| Column | Meaning |
|---|---|
| `i`, `j` | Positive integer endpoint IDs; IDs need not be consecutive |
| `FC` | Numeric functional connectivity target |
| `caliber` | Numeric tract caliber predictor |
| `myelin` | Numeric myelin predictor |
| `length` | Numeric tract length predictor |
| `rsn_i`, `rsn_j` | Nonempty network labels for the endpoints |

Example schema:

```csv
i,j,FC,caliber,myelin,length,rsn_i,rsn_j
1,2,0.42,8.5,0.018,36.2,Visual,Default
1,3,0.30,7.8,0.021,52.4,Visual,Control
```

These two rows illustrate formatting; use sufficiently many edges per model for estimation and inference. The loader does not log-transform, threshold, or otherwise select edges. Prepare those choices before running the code. Zero measurements are retained; blank values and non-finite numeric measurements are treated as missing.

If one or both endpoint label columns are absent, provide a separate node CSV:

```csv
node_id,rsn
1,Visual
2,Default
3,Control
```

```bash
myelin-fc-run --edges data/your_edges.csv --nodes data/your_nodes.csv \
  --out out --fc-label YOUR_DATA
```

Node IDs must be unique in this table. All endpoints must receive network labels, including when running only the global level.

## 4. Choose analysis settings

By default, both main models run at `global`, `rsn_pairs`, and `nodewise` levels. Reduced models also run in five myelin quantile bins at the global level.

```bash
myelin-fc-run --edges data/edges_fc_BOLDin.csv --out out --fc-label BOLD_GLOBAL \
  --levels-main global --levels-binned global --myelin-bins 5
```

Repeat a level option to select several levels:

```bash
myelin-fc-run --edges data/edges_fc_BOLDin.csv --out out --fc-label BOLD_NETWORKS \
  --levels-main global --levels-main rsn_pairs \
  --levels-binned global --levels-binned rsn_pairs
```

Use a different `--fc-label` or `--out` for each run you want to retain: rerunning the same destination overwrites corresponding CSV files and does not clear older files from other levels.

Both predictors and FC are z-scored within each fitted group using population standard deviation (`ddof=0`). The code first excludes rows missing any regression measurement, then standardizes and creates interactions. Use `--no-standardize-x` and/or `--no-standardize-y` to retain raw units. Quantile bins are defined globally from myelin before fitting within each bin. Ties can reduce the number of bins; a constant myelin predictor yields no usable bins.

## 5. Read the results

The reduced model is:

`FC ~ 1 + caliber + myelin + length`

The full model adds:

`myelin × caliber` and `myelin × length`

| File | Contents |
|---|---|
| `global_full.csv`, `global_reduced.csv` | One model across all edges |
| `rsn_pairs_full.csv`, `rsn_pairs_reduced.csv` | One model per unordered network pair, including within-network pairs |
| `nodewise_full.csv`, `nodewise_reduced.csv` | One model per node, pooling edges where it appears as either endpoint |
| `global_bins.csv` | Reduced models within myelin bins |
| `global_bin_corr.csv` | Pairwise-complete Pearson FC–predictor correlations within bins |

Additional binned levels produce `rsn_pairs_bins.csv` or `nodewise_bins.csv`.

| Column prefix/name | Interpretation |
|---|---|
| `key`, `level`, `bin` | Group and, where applicable, bin identifiers |
| `n_edges`, `n_obs` | Input edges in the group and complete rows actually fitted |
| `R2`, `R2_adj` | In-sample ordinary and adjusted R² |
| `B_…`, `p_…` | OLS coefficients and conventional coefficient p-values |
| `dom_…`, `domfrac_…` | Contribution values and their normalized percentages |
| `design_rank`, `df_resid`, `status` | Fit diagnostics |

The contribution metric averages individual and marginal R² contributions. For interactions, the individual contribution is the R² increment above its two main effects. Removing a main effect for its marginal contribution also removes its interactions. This is the source project's custom metric; it is **not** an exhaustive subset-based Shapley decomposition, and contribution values need not sum to model R². Percentages normalize by the sum of those contribution values.

Check `status` before interpreting a row: `ok`, `rank_deficient`, `no_residual_df`, `constant_response`, or `no_complete_rows`. Degenerate response groups return counts and missing fit values. Other ill-conditioned models may still return coefficients; check diagnostics and group size. Standard OLS p-values assume the usual residual model and do not account for dependence between brain-network edges; they are not held-out predictive evaluation or a permutation/spatial-null analysis.

## 6. Use the Python API

```python
from myelinfccoupling.config import Config
from myelinfccoupling.io import load_inputs
from myelinfccoupling.model import run
from myelinfccoupling.binning import run_binned

edges = load_inputs("data/edges_fc_BOLDin.csv")
cfg = Config(myelin_num_bins=5)
full = run(edges, "global", with_interactions=True, cfg=cfg)
binned = run_binned(edges, "global", cfg=cfg)
print(full[["R2", "R2_adj", "n_obs", "status"]])
```

The CLI uses the fixed CSV schema above. `Config` permits alternate measurement/group column names for direct API calls; prepare that DataFrame yourself when using a custom schema.

## 7. Test the installation

```bash
python -m pytest -q
```

Tests check saved reference results from the original Python implementation, recovery of known synthetic coefficients, grouping, input validation, missing-data handling, and CLI output creation. The supplied reference CSVs live in `tests/fixtures/`.

For the original MATLAB implementation and broader data release, see [Modeling-Myelin-FC](https://github.com/TardifLab/Modeling-Myelin-FC). This package runs from CSV without MATLAB. Population-standardized Python coefficients can differ from MATLAB's sample-standardized coefficients, particularly intercepts and interactions. Code is distributed under GPL-3.0; see `LICENSE`.

## Subject-stack input audit

The subject workflow currently validates MATLAB stacks and participant exclusions.
A corrected group-trained prediction baseline is also available below. Exact MATLAB Figure 6 reproduction remains unverified.

Keep your local subject files under `data/subjects/` (ignored by Git).
Copy `examples/subject_manifest.example.json` to `data/subjects/manifest.json`
and edit the file paths and MATLAB variable keys to match your data.
Paths in the manifest are relative to the manifest itself. The example uses
`main/` and `holdout/` subfolders, with each feature stored as `Dts`
(nodes × nodes × scans), and subject-list files containing `Ss` and `Ss_ho`.

```bash
python -m myelinfccoupling.subjects --manifest data/subjects/manifest.json --out out/subject_audit
```

Outputs are `subject_audit.csv` (nonfinite and zero edge counts per scan)
and `fold_exclusions.csv` (which main-stack sessions each held-out person excludes).
Anatomical edge ordering still requires the original parcel labels; shape and symmetry
checks alone cannot prove that node order agrees. Zeros are counted, not automatically
classified as missing. MATLAB v7.3 stacks require conversion or a future HDF5 adapter.
For arbitrary scan naming, provide an explicit `participant_ids` list in each batch.
The default parser supports this project's `sub-22` / `sub-22r` convention.

## Subject generalizability baseline

Extract `subject-data.zip` under `data/subjects/` so its `subject-data/` folder
contains `mwc/` and `parcellations/`. Copy one of the `examples/mwc-*.json`
files into that `subject-data/` folder.

```bash
cp examples/mwc-MTsat-tm.json data/subjects/subject-data/
python -m myelinfccoupling.generalizability \
  --manifest data/subjects/subject-data/mwc-MTsat-tm.json \
  --lut data/subjects/subject-data/parcellations/lut/lut_schaefer-400_mics.csv \
  --out out/generalizability/MTsat --permutations 1000
```

For R1 or g-ratio, use the corresponding example manifest and a different
output folder. All manifests select one myelin metric alongside caliber and length.
Use `--permutations 0` for a fast run without the permutation reference.

The model fits group-average edges separately within Yeo-7 network pairs,
with caliber, myelin, length, caliber × myelin and myelin × length. It excludes
every main-stack session of the held-out person, then predicts each holdout scan.
Structural group averages ignore zero entries. FC averages retain zero correlations.
Predictor scaling uses the training group's network-pair mean and sample SD;
FC stays in its supplied units. Each undirected edge is used once, without the diagonal.
Training structure defines edge availability; test FC never selects prediction edges.
Rank-deficient blocks and blocks with fewer than ten residual degrees of freedom are skipped.

Outputs include per-scan/network metrics, edge predictions, training FC templates,
coefficients, fitted scaling, model status, fold exclusions, input SHA-256 digests
and run settings. Individual-deviation correlation compares predicted minus
training-mean FC with empirical minus training-mean FC. `own_percentile` and
`own_minus_other` describe matching against training participants; they are
not independently validated identification accuracy or specificity p-values.
The within-network edge permutations are a conditional reference and do not
fully preserve spatial or shared-node dependence.

This baseline intentionally differs from the original MATLAB script: it does
not automatically log-transform skewed stacks, does not z-score test FC, and
uses training-fitted predictor scaling and fold-specific masks. It therefore
must not be described as exact replication of Figure 6.

## Nested Ridge comparison

```bash
python -m pip install -e ".[ml]"
python -m myelinfccoupling.ridge \
  --manifest data/subjects/subject-data/mwc-MTsat-tm.json \
  --lut data/subjects/subject-data/parcellations/lut/lut_schaefer-400_mics.csv \
  --out out/ridge/MTsat
```

Each outer fold excludes one holdout participant from the main stack. Five inner
folds split unique training participants, rebuilding group means, masks, scaling
and models inside each fold. Ridge alpha is selected by mean participant RMSE;
sessions of a person stay together. The default eleven positive alpha candidates
span 10^-4 to 10^6. A single alpha is shared across network pairs per outer fold.
Main predictors and the five-term interaction design are scaled using training
statistics only. The intercept is unpenalized. Ridge uses cached linear-system
solutions equivalent to scikit-learn's Ridge sum-of-squares objective, tested
against that estimator. OLS, Ridge and the training FC template use the same
eligible edges and block gates. No edge-permutation tuning is used.

Results include metrics.csv, selected_alpha.csv, inner_scores.csv,
fold_assignments.csv (scan_index is zero-based in the main stack), coefficients,
scaling, model status, predictions, input hashes and run configuration. Inspect
the `all` rows in metrics.csv; averages over sessions should first be reduced to
participant-level summaries before inference. Runs are exploratory: earlier
holdout baseline outcomes have already been inspected.

Use `--inner-folds` and `--alphas` only for a predefined development experiment;
do not repeatedly pick settings by outer holdout scores. Change the output
folder for each configuration. This implementation trains on group-average
edges; training directly on participant-edge rows is a subsequent extension.

The preprocessing module translates the supplied MATLAB helpers for reference
work. It is not applied in the raw-data Ridge baseline. Exact MATLAB runtime
agreement and full historical Figure 6 reproduction remain unverified. The
positive-real log translation uses an omit-NaN minimum for shifting; negative
inputs are rejected rather than producing MATLAB complex values.

## Training on subject-level SC–FC pairs

Version 0.5 adds `--training-mode subject` to the Ridge comparison. Within each
network pair it pools rows from each training subject's own caliber, myelin,
length and FC, with the same two interaction terms. Each participant receives
equal total weight per network pair; multiple valid sessions share that weight.
Weighted means and population standard deviations are fitted on training rows.
Weights are rescaled to sum to the row count, so the Ridge objective is weighted
SSE plus alpha times squared slopes. Alpha values therefore depend on training
row count and are not directly comparable between group and subject modes.

```bash
python -m myelinfccoupling.ridge \
  --manifest data/subjects/subject-data/mwc-MTsat-tm.json \
  --lut data/subjects/subject-data/parcellations/lut/lut_schaefer-400_mics.csv \
  --out out/subject-ridge/MTsat \
  --training-mode subject
```

Outer participant exclusions and five inner participant folds apply to all
training, scaling and alpha selection. The FC template remains the raw mean of
training scans (one retained scan per participant in the current MWC main
stack). Held-out FC is used only for evaluation. Subjects with more edges do
not gain more total fitting weight. Edges from the same person are dependent;
pooled rows do not increase the number of independent participants.

Both modes evaluate OLS, Ridge and Template on identical eligible edges within
a run. Subject pooling may make a network pair estimable that group training
skipped; compare modes using common `(participant_id, scan_id, i, j)` rows from
`predictions.csv`, rather than assuming their evaluation edge sets are equal.
A subject-trained model still learns shared coefficients across subjects; it
is not a separately fitted model for the held-out person. This phase tests
whether retaining subject variation in training improves prediction. Results
remain exploratory because the holdout outcomes have already been inspected.

## Predicting departures from the FC template

Version 0.7 adds `--training-mode residual`. Inside each training fold, compute
an edge-wise FC template and nonzero structural-feature means. Fit each network
pair to subject SC deviations from those means and FC deviations from the
template, with the same equal participant fitting weights as subject mode.
The three main terms are structural deviations; the two interactions are
products of caliber/myelin deviations and myelin/length deviations.

At prediction time, the held-out person's raw SC is centered using the training
references. The predicted FC deviation is added to the training FC template.
A zero structural deviation is valid; a zero raw structural value continues to
mean an unavailable edge. Test FC is used only in evaluation. All references,
scaling, model coefficients and alpha choices are rebuilt inside inner folds.

```bash
python -m myelinfccoupling.ridge \
  --manifest data/subjects/subject-data/mwc-gratio-ts.json \
  --lut data/subjects/subject-data/parcellations/lut/lut_schaefer-400_mics.csv \
  --out out/residual/gratio \
  --training-mode residual
```

Residual runs also export `references.csv`: each outer fold's FC template and
structural means keyed by participant and edge. Predictions include
`OLS_deviation` and `Ridge_deviation` in addition to final absolute FC.
`R2_vs_template` is 1 minus model squared error divided by template squared
error. Positive values indicate improvement over the template; zero matches
its error; negative values indicate worse error. It differs from conventional
R² relative to a scalar target mean. Individual-deviation correlation can be
positive without improving squared error, so inspect both measures.

### Compare modes and feature sets fairly

```bash
python -m myelinfccoupling.compare_runs \
  --run direct=out/subject-ridge/MTsat/predictions.csv \
  --run residual=out/residual/MTsat/predictions.csv \
  --out out/comparison/MTsat
```

Provide two or more named runs. Every model is scored on the intersection of
participant/scan/edge rows across all supplied files. The tool requires matching
scan sets, empirical FC and training templates, and rejects duplicate edge
keys. It exports per-scan metrics, metrics averaged within each participant,
participant-averaged summaries, edge coverage and comparison configuration with
input hashes. A single template benchmark is included. Inspect coverage:
intersecting runs with different structural metrics can change the evaluated
edge population. This tool compares matched cohorts and exclusions; it is not
a cross-cohort transfer evaluator.

## One-cohort participant cross-validation

Version 0.8 adds `myelinfccoupling.cohort` for a single cohort with no separate
holdout stack. A manifest containing only a `main` batch is sufficient. Five
outer folds split unique participants, and five inner participant folds choose
Ridge alpha using only each outer training set. Every scan is evaluated once;
all sessions of a participant stay in the same outer fold. Subject and residual
training modes use the existing weighting and preprocessing implementations.

```bash
python -m myelinfccoupling.cohort \
  --manifest data/subjects/subject-data/mics-R1.json \
  --lut data/subjects/subject-data/parcellations/lut/lut_schaefer-400_mics.csv \
  --out out/mics/residual \
  --training-mode residual
```

`examples/mics-R1.example.json` illustrates the mapping, using placeholder
filenames/keys that must be checked against actual files. If scan IDs are not
in the current `sub-NN`/`sub-NNr` format, add an explicit `participant_ids` list
inside the batch in the same order as the IDs and stack columns. For a cohort
with one scan per person, each participant ID can be its exact scan ID. For
repeat scans, all sessions of a person must share the same participant ID.

Outputs include per-scan/network metrics, participant-averaged summary,
outer and inner fold assignments, selected alpha, coefficients and scaling,
predictions, audit, fitted residual references, input hashes and configuration.
Fold and scan indices in these outputs are zero-based. A nonempty output
directory is rejected. Use `--outer-folds`, `--inner-folds`, `--seed` and
`--alphas` for a predefined development protocol. Evaluate subject and residual
modes with identical outer settings, then use `compare_runs` on their saved
predictions.

This protocol estimates prediction for unseen participants within a cohort.
Training on MWC and testing on MICs would be a separate external-transfer
protocol. The fixed protocol has been run on 45 aligned MICs participants with R1,
caliber, length and FC. Cohort integration tests use entirely synthetic data.
Sequential MICs labels identify stack positions rather than original study IDs. Reusing outer
results to repeatedly select settings turns those results into development
feedback, so keep the benchmark bounded and report it accordingly.
