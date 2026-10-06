# myelinfccoupling/util.py

from pathlib import Path
from typing import Dict, List
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def sanitize_label(label: str) -> str:
    """
    Make a label safe for folder/filenames.

    - trims spaces
    - replaces spaces with underscores
    - replaces path separators with '-'
    """
    s = str(label).strip()
    s = s.replace(" ", "_")
    s = s.replace("/", "-").replace("\\", "-")
    if not s or s in {".", ".."}:
        raise ValueError("FC label must be a nonempty folder name, not . or ..")
    return s


def save_struct_by_level(
    results: Dict[str, pd.DataFrame],
    levels: List[str],
    out_dir: Path,
    suffix: str,
) -> None:
    """
    Save per-level results as CSVs.

    results: dict with keys 'global', 'rsn_pairs', 'nodewise', etc.
    levels: which levels to consider (subset)
    suffix: e.g., 'full', 'reduced', 'bins'
    """
    for lev in levels:
        if lev in results and results[lev] is not None:
            df = results[lev]
            if isinstance(df, pd.DataFrame) and not df.empty:
                fname = out_dir / f"{lev}_{suffix}.csv"
                df.to_csv(fname, index=False)
