# myelinfccoupling/model.py

from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import Config


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute R^2 for OLS y ~ X (with intercept) using linear algebra.

    Assumes X, y are already finite and aligned.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]

    n = X.shape[0]
    if n == 0:
        return np.nan

    # Design matrix with intercept
    X_design = np.column_stack([np.ones(n), X])

    # Solve least squares
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def _dominance_interaction(
    df: pd.DataFrame, cfg: Config, with_interactions: bool
) -> Dict[str, tuple]:
    """
    Dominance analysis adapted to handle interactions (see dominance_interaction.m)

    Uses:
        x1 = caliber
        x2 = myelin
        x3 = length
        interactions: x2*x1 (myelin:caliber) and x2*x3 (myelin:length)

    Returns a dict:
        { variable_name: (dominance_value, dominance_percentage) }

    When with_interactions=False, only the 3 main effects are included.
    When with_interactions=True, both main effects and interactions are included.
    """
    cols = [cfg.col_cal, cfg.col_myelin, cfg.col_len, cfg.col_FC]
    D = df[cols].dropna().copy()
    if D.empty:
        return {}

    X_main = D[[cfg.col_cal, cfg.col_myelin, cfg.col_len]].to_numpy()
    y = D[cfg.col_FC].to_numpy()
    N = X_main.shape[1]
    names_main = [cfg.col_cal, cfg.col_myelin, cfg.col_len]

    # --- Lower-order terms only ---
    R2_full_lo = _ols_r2(X_main, y)
    R2_indi_lo = np.array([_ols_r2(X_main[:, i], y) for i in range(N)])
    R2_rest_lo = np.array(
        [_ols_r2(np.delete(X_main, i, axis=1), y) for i in range(N)]
    )
    R2_marg_lo = R2_full_lo - R2_rest_lo
    dom_lo = (R2_marg_lo + R2_indi_lo) / 2.0

    if np.isfinite(dom_lo).all() and dom_lo.sum() != 0:
        dom_lo_frac = dom_lo / dom_lo.sum() * 100.0
    else:
        dom_lo_frac = np.full_like(dom_lo, np.nan, dtype=float)

    result_lo = {
        names_main[i]: (float(dom_lo[i]), float(dom_lo_frac[i]))
        for i in range(N)
    }

    if not with_interactions:
        return result_lo

    # --- Including interactions: myelin:caliber and myelin:length ---
    # x1 = cal, x2 = myelin, x3 = length
    x1, x2, x3 = X_main.T
    inter_defs = [
        (f"{cfg.col_myelin}:{cfg.col_cal}", (1, 0)),  # myelin:caliber (x2*x1)
        (f"{cfg.col_myelin}:{cfg.col_len}", (1, 2)),  # myelin:length  (x2*x3)
    ]
    inter_cols = [x2 * x1, x2 * x3]

    Xt_full = np.column_stack([X_main] + inter_cols)
    N_tot = Xt_full.shape[1]

    R2_full_hi = _ols_r2(Xt_full, y)

    # Individual R2 (hi): copy lower-order for first N, then interaction increments
    R2_indi_hi = np.zeros(N_tot)
    R2_indi_hi[:N] = R2_indi_lo

    for j, (_, (idx_my, idx_other)) in enumerate(inter_defs, start=N):
        # Model with only main effects involved in this interaction
        S = [idx_my, idx_other]
        X_lo = X_main[:, S]
        R2_lo = _ols_r2(X_lo, y)

        # Model with main effects + interaction term
        X_hi = np.column_stack([X_lo, Xt_full[:, j]])
        R2_hi = _ols_r2(X_hi, y)

        R2_indi_hi[j] = R2_hi - R2_lo

    # Marginal R2 for hi model: drop each term (and associated interactions for mains)
    R2_rest_hi = np.zeros(N_tot)
    all_idx = list(range(N_tot))

    # Which interaction columns involve each main effect:
    # main index -> list of interaction indices
    inter_map = {
        0: [N],       # caliber in myelin:caliber
        1: [N, N + 1],# myelin in both interactions
        2: [N + 1],   # length in myelin:length
    }

    # Main effects: drop main + its interactions
    for i in range(N):
        drop = [i] + inter_map.get(i, [])
        keep = [k for k in all_idx if k not in drop]
        if keep:
            X_rest = Xt_full[:, keep]
            R2_rest_hi[i] = _ols_r2(X_rest, y)
        else:
            # No predictors left
            R2_rest_hi[i] = 0.0

    # Interactions: drop only the interaction term
    for j in range(N, N_tot):
        keep = [k for k in all_idx if k != j]
        X_rest = Xt_full[:, keep]
        R2_rest_hi[j] = _ols_r2(X_rest, y)

    R2_marg_hi = R2_full_hi - R2_rest_hi
    dom_hi = (R2_marg_hi + R2_indi_hi) / 2.0

    if dom_hi.sum() != 0:
        dom_hi_frac = dom_hi / dom_hi.sum() * 100.0
    else:
        dom_hi_frac = np.full_like(dom_hi, np.nan, dtype=float)

    names_all = names_main + [name for name, _ in inter_defs]
    result_hi = {
        names_all[i]: (float(dom_hi[i]), float(dom_hi_frac[i]))
        for i in range(N_tot)
    }

    return result_hi


def _standardize(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Z-score columns in-place (ddof=0)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            std = df[c].std(ddof=0)
            if std != 0:
                df[c] = (df[c] - df[c].mean()) / std
    return df


def _fit(df: pd.DataFrame, cfg: Config, with_interactions: bool) -> pd.Series:
    """
    Fit regression model for a single group of edges and return a Series
    with R², coefficients, p-values, and dominance metrics.
    """
    cols = dict(FC=cfg.col_FC, cal=cfg.col_cal, my=cfg.col_myelin, ln=cfg.col_len)
    work = df[[cols["FC"], cols["cal"], cols["my"], cols["ln"]]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    if work.empty or work[cols["FC"]].nunique() < 2:
        return pd.Series({"n_edges": len(df), "n_obs": len(work),
                          "R2": np.nan, "R2_adj": np.nan,
                          "status": "no_complete_rows" if work.empty else "constant_response"})

    # Standardize predictors and/or response
    if cfg.standardize_predictors:
        work = _standardize(work, [cols["cal"], cols["my"], cols["ln"]])
    if cfg.standardize_response:
        work = _standardize(work, [cols["FC"]])

    # Formula with or without interactions
    if with_interactions:
        formula = (
            f"{cols['FC']} ~ 1 + "
            f"{cols['my']}*{cols['cal']} + {cols['my']}*{cols['ln']}"
        )
    else:
        formula = (
            f"{cols['FC']} ~ 1 + "
            f"{cols['cal']} + {cols['my']} + {cols['ln']}"
        )

    # Fit OLS
    model = smf.ols(formula, data=work).fit()

    r: Dict[str, float] = {}
    r["n_edges"] = int(df.shape[0])
    r["n_obs"] = int(model.nobs)
    r["R2"] = float(model.rsquared)
    r["R2_adj"] = float(model.rsquared_adj)
    rank = int(np.linalg.matrix_rank(model.model.exog))
    r["design_rank"] = rank
    r["df_resid"] = float(model.df_resid)
    r["status"] = ("rank_deficient" if rank < model.model.exog.shape[1]
                   else "no_residual_df" if model.df_resid <= 0 else "ok")

    # Coefficients and p-values
    for name, val in model.params.items():
        safe = name.replace(":", "_x_").replace(" ", "_")
        r[f"B_{safe}"] = float(val)
        r[f"p_{safe}"] = float(model.pvalues.get(name, np.nan))

    # Dominance analysis
    dom = _dominance_interaction(work, cfg, with_interactions=with_interactions)
    for var, (dom_val, dom_frac) in dom.items():
        safe = var.replace(":", "_x_")
        r[f"dom_{safe}"] = dom_val
        r[f"domfrac_{safe}"] = dom_frac

    return pd.Series(r)


def run(
    edges: pd.DataFrame,
    level: str,
    with_interactions: bool = True,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Run regression + dominance for a given level:

    level in {'global', 'rsn_pairs', 'nodewise'}.

    Returns a DataFrame with one row per group at that level:
    - global:   single row
    - rsn_pairs: one row per RSN pair (unordered)
    - nodewise: one row per node (i/j pooled)
    """
    if cfg is None:
        cfg = Config()

    rows = []

    if level == "global":
        s = _fit(edges, cfg, with_interactions)
        s["level"] = "global"
        s["key"] = "global"
        rows.append(s)

    elif level == "rsn_pairs":
        # Unordered RSN pair label (min–max)
        rsn_i = edges[cfg.col_rsn_i].astype(str)
        rsn_j = edges[cfg.col_rsn_j].astype(str)
        p = np.where(
            rsn_i < rsn_j,
            rsn_i + "–" + rsn_j,
            rsn_j + "–" + rsn_i,
        )
        edges2 = edges.copy()
        edges2["pair"] = p

        for key, sub in edges2.groupby("pair"):
            s = _fit(sub, cfg, with_interactions)
            s["level"] = "rsn_pairs"
            s["key"] = key
            rows.append(s)

    elif level == "nodewise":
        # Pool edges incident on each node (i or j)
        edges_i = edges.rename(columns={cfg.col_i: "node"})
        edges_j = edges.rename(columns={cfg.col_j: "node"})
        both = pd.concat([edges_i, edges_j], ignore_index=True)

        for key, sub in both.groupby("node"):
            s = _fit(sub, cfg, with_interactions)
            s["level"] = "nodewise"
            s["key"] = key
            rows.append(s)
    else:
        raise ValueError("level must be 'global', 'rsn_pairs', or 'nodewise'")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
