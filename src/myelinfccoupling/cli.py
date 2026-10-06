# myelinfccoupling/cli.py

import click
from pathlib import Path

from .io import load_inputs
from .config import Config
from . import model, binning
from .util import ensure_dir, sanitize_label, save_struct_by_level


@click.command()
@click.option(
    "--edges",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Path to edges.csv",
)
@click.option(
    "--nodes",
    type=click.Path(exists=True, readable=True),
    required=False,
    default=None,
    help="Optional nodes.csv (to derive rsn_i / rsn_j if needed).",
)
@click.option(
    "--out",
    type=click.Path(),
    required=True,
    help="Root output directory (modality subfolder will be created).",
)
@click.option(
    "--fc-label",
    type=str,
    default="BOLD",
    help="Label for FC modality (used as subdirectory under --out).",
)
@click.option(
    "--levels-main",
    multiple=True,
    type=click.Choice(["global", "rsn_pairs", "nodewise"]),
    default=["global", "rsn_pairs", "nodewise"],
    help="Levels for main regression models.",
)
@click.option(
    "--levels-binned",
    multiple=True,
    type=click.Choice(["global", "rsn_pairs", "nodewise"]),
    default=["global"],
    help="Levels for binned models.",
)
@click.option(
    "--no-standardize-x",
    is_flag=True,
    help="Disable z-scoring of predictors (caliber, myelin, length).",
)
@click.option(
    "--no-standardize-y",
    is_flag=True,
    help="Disable z-scoring of FC.",
)
@click.option(
    "--myelin-bins",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Number of myelin bins.",
)
def main(
    edges,
    nodes,
    out,
    fc_label,
    levels_main,
    levels_binned,
    no_standardize_x,
    no_standardize_y,
    myelin_bins,
):
    """
    Command-line entry point for Myelin–FC coupling (Python).

    Example:
        myelin-fc-run \\
          --edges ../data/edges_BOLD.csv \\
          --nodes ../data/nodes.csv \\
          --out   ../out \\
          --fc-label BOLD
    """
    edges_path = Path(edges)
    nodes_path = Path(nodes) if nodes else None
    out_root = Path(out)

    # Load data
    try:
        E = load_inputs(edges_path, nodes_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Build config
    cfg = Config()
    cfg.fc_label = fc_label
    cfg.levels_main = list(levels_main)
    cfg.levels_binned = list(levels_binned)
    cfg.standardize_predictors = not no_standardize_x
    cfg.standardize_response = not no_standardize_y
    cfg.myelin_num_bins = myelin_bins

    # FC-modality-specific output directory
    try:
        out_dir = out_root / sanitize_label(cfg.fc_label)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    ensure_dir(out_dir)

    click.echo(f'Running Myelin–FC coupling for modality "{cfg.fc_label}"')
    click.echo(f"Output directory: {out_dir}")

    # --- Main models (full and reduced) ---
    res_full = {}
    res_reduced = {}
    for lev in cfg.levels_main:
        res_full[lev] = model.run(E, lev, with_interactions=True, cfg=cfg)
        res_reduced[lev] = model.run(E, lev, with_interactions=False, cfg=cfg)

    # --- Binned models ---
    res_bins = {}
    for lev in cfg.levels_binned:
        res_bins[lev] = binning.run_binned(
            E, lev, cfg=cfg, with_interactions=False
        )

    # --- Binned FC–predictor correlations (global-style) ---
    bin_corr = binning.bin_correlations(E, cfg=cfg)
    bin_corr_path = out_dir / "global_bin_corr.csv"
    bin_corr.to_csv(bin_corr_path, index=False)

    # --- Save main & binned model results ---
    save_struct_by_level(res_full, cfg.levels_main, out_dir, "full")
    save_struct_by_level(res_reduced, cfg.levels_main, out_dir, "reduced")
    save_struct_by_level(res_bins, cfg.levels_binned, out_dir, "bins")

    click.echo("Done.")
    click.echo(f"Full model CSVs written under:    {out_dir}")
    click.echo(f"Reduced model CSVs written under: {out_dir}")
    click.echo(f"Binned model CSVs written under:  {out_dir}")
    click.echo(f"Binned FC–predictor correlations: {bin_corr_path}")
