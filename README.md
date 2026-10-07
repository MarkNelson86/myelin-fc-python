# Myelin–FC coupling in Python

An installable Python pipeline for testing whether structural brain measurements
improve prediction of an individual's functional connectivity. The case study
combines multimodal data validation, linear and nonlinear modeling, and evaluation
with many correlated observations but few independent participants.

## Run the public demo

Python 3.10+; from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ml]"
myelin-fc-demo --all-models --out out/demo
```

No participant data or downloads are required. The command generates synthetic
connectomes and compares OLS, Ridge, gradient boosting and a small neural network.
Open `out/demo/benchmark/summary.csv` and `out/demo/DEMO_REPORT.md`. Synthetic
scores demonstrate execution, not scientific performance. Choose a fresh output
directory for each run.

Preview the [all-model demo results](examples/demo-results/benchmark/summary.csv)
and [demo report](examples/demo-results/DEMO_REPORT.md) without installing the
package. These outputs use synthetic data and demonstrate the workflow only.

## Engineering highlights

| Concern | Implementation |
|---|---|
| Input integrity | Explicit file mappings; matrix shape, symmetry and participant identity checks |
| Evaluation leakage | Nested participant folds; repeated sessions kept together; training-only preprocessing |
| Fair comparisons | Strong training-FC baseline; participant-balanced fitting; common-edge scoring |
| Reproducibility | Fixed seeds, input hashes, saved folds, configurations and sampling logs |
| Usability | Modular package, CLI commands and a self-contained synthetic demo |
| Verification | 47 tests; CI configured for Python 3.10 and 3.12, including the installed demo |

## Benchmark finding

On 45 MICs participants, the training FC mean achieved **0.20659 RMSE**. Residual
Ridge, boosting and the neural network achieved **0.20730**, **0.20767** and
**0.20816**, respectively. The tested models did not improve on the baseline.
The evaluation demonstrates why shared patterns must be distinguished from
individual predictive information; it does not establish that structure cannot
predict individual function.

See the [benchmark report](docs/benchmark-report.md) for the protocol and results,
[engineering notes](docs/engineering.md) for implementation choices, and
[analysis user guide](docs/user-guide.md) for input schemas and CLI options.

## Development

```bash
python -m pip install -e ".[dev,ml]"
python -m pytest -q
python -m build
```

The package uses NumPy, pandas, SciPy, statsmodels and scikit-learn. Private
participant inputs belong in the ignored `data/subjects/` directory; generated
outputs belong in `out/`. Real participant stacks are not bundled with the demo.

Derived from [Modeling-Myelin-FC](https://github.com/TardifLab/Modeling-Myelin-FC).
The predictive workflows change preprocessing and validation; exact reproduction
of the original analysis is not claimed. Licensed under [GPL-3.0](LICENSE).
