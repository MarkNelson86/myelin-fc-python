"""Read and validate undirected edge tables."""
import numpy as np
import pandas as pd


def load_inputs(edges_path, nodes_path=None) -> pd.DataFrame:
    """Load one row per undirected edge, optionally joining endpoint RSNs.

    Non-finite numeric measurements are treated as missing. Node IDs must be
    positive integers; self-loops, duplicate undirected edges, and missing RSNs
    are rejected. No transformation or structural edge filtering is performed.
    """
    edges = pd.read_csv(edges_path)
    required = ["i", "j", "FC", "caliber", "myelin", "length"]
    missing = [c for c in required if c not in edges]
    if missing:
        raise ValueError(f"Edges table missing columns: {missing}")
    if edges.empty:
        raise ValueError("Edges table has no rows")
    for c in required:
        edges[c] = pd.to_numeric(edges[c], errors="raise")
    for c in ["i", "j"]:
        v = edges[c].to_numpy(dtype=float)
        if not np.all(np.isfinite(v) & (v > 0) & (v == np.floor(v))):
            raise ValueError(f"{c} must contain positive integer node IDs")
    if (edges.i == edges.j).any():
        raise ValueError("Self-loops are not supported")
    pairs = np.sort(edges[["i", "j"]].to_numpy(), axis=1)
    if pd.DataFrame(pairs).duplicated().any():
        raise ValueError("Duplicate undirected edges: supply each edge once")
    for c in ["FC", "caliber", "myelin", "length"]:
        edges[c] = edges[c].replace([np.inf, -np.inf], np.nan)
    missing_rsn = [c for c in ["rsn_i", "rsn_j"] if c not in edges]
    if missing_rsn:
        if nodes_path is None:
            raise ValueError(f"Missing {missing_rsn}; provide --nodes with node_id,rsn")
        nodes = pd.read_csv(nodes_path)
        if not {"node_id", "rsn"} <= set(nodes):
            raise ValueError("Nodes table must contain node_id and rsn")
        nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="raise")
        if nodes.node_id.isna().any() or nodes.node_id.duplicated().any():
            raise ValueError("Nodes table must contain unique, nonmissing node_id values")
        for endpoint, label in [("i", "rsn_i"), ("j", "rsn_j")]:
            if label in missing_rsn:
                labels = nodes[["node_id", "rsn"]].rename(columns={"node_id": endpoint, "rsn": label})
                edges = edges.merge(labels, on=endpoint, how="left", validate="many_to_one")
    for c in ["rsn_i", "rsn_j"]:
        if edges[c].isna().any() or edges[c].astype(str).str.strip().eq("").any():
            raise ValueError(f"{c} has missing network labels")
    return edges
