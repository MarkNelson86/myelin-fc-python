# myelinfccoupling/binning.py

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from . import model


def bin_by_myelin(edges: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Add a 'my_bin' column to edges based on quantiles of myelin.
    """
    edges_b = edges.copy()
    if cfg.myelin_num_bins < 1:
        raise ValueError("myelin_num_bins must be at least 1")
    q = np.linspace(0.0, 1.0, cfg.myelin_num_bins + 1)
    edges_b["my_bin"] = pd.qcut(
        edges_b[cfg.col_myelin],
        q,
        duplicates="drop",
    )
    return edges_b


def run_binned(
    edges: pd.DataFrame,
    level: str,
    cfg: Optional[Config] = None,
    with_interactions: bool = False,
) -> pd.DataFrame:
    """
    Run models within myelin bins for the given level.

    Returns a DataFrame with all groups x bins stacked, where each row has
    a 'bin' column indicating the myelin bin label.
    """
    if cfg is None:
        cfg = Config()

    edges_b = bin_by_myelin(edges, cfg)
    out = []

    # Only occupied bins are analyzed
    for b, sub in edges_b.groupby("my_bin", observed=True):
        M = model.run(sub, level, with_interactions=with_interactions, cfg=cfg)
        if not M.empty:
            M = M.copy()
            M["bin"] = str(b)
            out.append(M)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


def bin_correlations(edges: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    """
    Compute Pearson correlations between FC and each predictor
    (caliber, myelin, length) within myelin bins.
    """
    if cfg is None:
        cfg = Config()

    edges_b = bin_by_myelin(edges, cfg)
    rows = []

    for b, sub in edges_b.groupby("my_bin", observed=True):
        record = {"bin": str(b)}
        y = sub[cfg.col_FC]

        for v in [cfg.col_cal, cfg.col_myelin, cfg.col_len]:
            x = sub[v]
            ok = x.notna() & y.notna()
            r_name = f"{v}_r"
            n_name = f"{v}_n"

            if ok.sum() > 0:
                r = x[ok].corr(y[ok])
                n = int(ok.sum())
            else:
                r = np.nan
                n = 0

            record[r_name] = float(r)
            record[n_name] = n

        rows.append(record)

    return pd.DataFrame(rows, columns=["bin", "caliber_r", "caliber_n", "myelin_r", "myelin_n", "length_r", "length_n"])
