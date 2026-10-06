# Myelin–FC coupling in Python

Fit functional connectivity (FC) from tract caliber, myelin, and tract length. Run models across all edges, network pairs, or individual nodes, then export coefficients, R², p-values, and interaction-aware contribution metrics to CSV.

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
Prediction and Figure 6 reproduction are not implemented yet.

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
