# myelinfccoupling/config.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    Configuration for Myelin–FC coupling modeling (Python).

    Mirrors the logic in the MATLAB pipeline.
    """

    # Column names in edges.csv
    col_i: str = "i"
    col_j: str = "j"
    col_FC: str = "FC"
    col_cal: str = "caliber"
    col_myelin: str = "myelin"
    col_len: str = "length"
    col_rsn_i: str = "rsn_i"
    col_rsn_j: str = "rsn_j"

    # Whether to z-score predictors / response before fitting
    standardize_predictors: bool = True
    standardize_response: bool = True

    # Myelin binning
    myelin_num_bins: int = 5

    # FC modality label; used to create out/<fc_label>/ subfolder
    fc_label: str = "BOLD"

    # Levels to run in main and binned analyses
    # (used in CLI; model.run itself is per-level)
    levels_main: List[str] = field(
        default_factory=lambda: ["global", "rsn_pairs", "nodewise"]
    )
    levels_binned: List[str] = field(
        default_factory=lambda: ["global"]
    )
