"""Generate deterministic synthetic edges; not a scientific dataset."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_edges(n_nodes=24, seed=42):
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n_nodes, k=1)
    caliber, myelin, length = rng.normal(size=(3, len(i)))
    fc = (0.5 * caliber - 0.3 * myelin + 0.2 * length
          + 0.4 * myelin * caliber + rng.normal(scale=0.15, size=len(i)))
    labels = np.array(["Visual", "Default", "Control"])
    return pd.DataFrame(dict(i=i+1, j=j+1, FC=fc, caliber=caliber,
                             myelin=myelin, length=length,
                             rsn_i=labels[i % 3], rsn_j=labels[j % 3]))


if __name__ == "__main__":
    path = Path("data/demo_edges.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    make_edges().to_csv(path, index=False)
    print(f"Wrote {path}")
