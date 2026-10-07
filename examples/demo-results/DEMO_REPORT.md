# Synthetic subject-modeling demo

All inputs are generated fixtures. Metrics demonstrate execution only; they are not scientific results. Ten synthetic people; two held-out people with paired sessions; nine training people per outer fold; three inner participant folds. Sessions are averaged within people before summary.

See summary.csv, subject_audit.csv, fold_exclusions.csv and models/ for metrics, predictions, coefficients, scaling, selected alphas, fold assignments, input hashes and run configuration.

The optional all-model benchmark uses the ten main-stack people in three outer and two inner participant folds. Its separate benchmark/summary.csv compares residual OLS, Ridge, boosting and a small neural network on common edges. Budgets are smaller than the real-data protocol. Synthetic metrics demonstrate execution only.
