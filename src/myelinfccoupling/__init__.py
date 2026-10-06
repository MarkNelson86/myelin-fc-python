# myelinfccoupling/__init__.py

"""
Python companion library for the Myelin–FC coupling analysis.

Core modules:
- config: configuration / options
- io:     loading edges/nodes tables
- model:  regression + dominance analysis (global, rsn_pairs, nodewise)
- binning: myelin-binned modeling and FC–predictor correlations
- cli:    command-line entry point (myelin-fc-run)
"""

__all__ = ["config", "io", "model", "binning"]
