"""Write RFind-sc scores into the cell annotations of an AnnData object (0.3.7)."""
import warnings
from collections.abc import Mapping

import pandas as pd


def add_rfind_scores(adata, scores, prefix="RFind_", overwrite=True):
    """Add RFind-sc scores to ``adata.obs``, one column per panel.

    The counterpart of the R function ``add_rfind_scores()``, which writes into
    the metadata of a Seurat object: the cells x panels scores of
    :func:`run_pipeline`, :func:`score_panel_pairs`, :func:`score_genesets` or
    :func:`score_cells` are written into the cell annotations, matched by cell
    name, so that they can be plotted and grouped like any other annotation
    (for example ``scanpy.pl.umap(adata, color="RFind_DAM")``). The scores
    themselves are not changed.

    Parameters
    ----------
    adata : AnnData or pandas.DataFrame
        An AnnData object (its ``obs`` is changed in place), or a DataFrame
        with one row per cell, indexed by cell name (changed in place).
    scores : dict or pandas.DataFrame
        The dict returned by a scoring function (its ``"scores"`` entry is
        used), or a DataFrame of cells x panels indexed by cell name.
    prefix : str
        Prefix of the new column names. Default ``"RFind_"`` (column
        ``RFind_DAM`` for the panel ``DAM``).
    overwrite : bool
        Replace columns of the same name that are already there. Default
        True; with False, such columns raise an error.

    Returns
    -------
    AnnData or pandas.DataFrame
        The same object, with one new column per panel. Cells that are not in
        ``scores`` get NaN (with a warning).
    """
    if isinstance(scores, Mapping):
        scores = scores["scores"]
    scores = pd.DataFrame(scores)
    obs = adata.obs if hasattr(adata, "obs") and hasattr(adata, "obs_names") else adata
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("`adata` must be an AnnData object or a DataFrame with one row per cell.")
    if not scores.index.is_unique:
        raise ValueError("The cell names (index) of `scores` must be unique.")
    cols = [f"{prefix}{c}" for c in scores.columns]
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        raise ValueError("Duplicated column names: " + ", ".join(dup))
    found = obs.index.isin(scores.index)
    if not found.any():
        raise ValueError("No cell of `adata` is in `scores` (the index of `scores` must be cell names).")
    existing = [c for c in cols if c in obs.columns]
    if existing and not overwrite:
        raise ValueError("Columns already present: " + ", ".join(existing)
                         + " (use overwrite=True to replace them).")
    if not found.all():
        warnings.warn(f"{int((~found).sum())} cell(s) of `adata` are not in `scores`; their scores are NaN.",
                      UserWarning, stacklevel=2)
    aligned = scores.reindex(obs.index)
    for panel, col in zip(scores.columns, cols):
        obs[col] = aligned[panel].to_numpy(dtype=float)
    return adata
