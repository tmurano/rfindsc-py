"""Panels: bundled references and DEG tables (R: bundled.R, panels.R)."""
from __future__ import annotations

import json
import math
from importlib import resources

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- bundled

def _side_from_json(d) -> pd.DataFrame:
    return pd.DataFrame({
        "gene": pd.Series(d["gene"], dtype=object),
        "diff": np.array([np.nan if v is None else v for v in d["diff"]], dtype=np.float64),
        "rank": np.array(d["rank"], dtype=np.int64),
    })


def _load_bundle(name: str) -> dict:
    raw = json.loads((resources.files("rfindsc") / "data" / name).read_text("utf-8"))
    out = {}
    for pname, p in raw["panels"].items():
        out[pname] = {k: (_side_from_json(v) if k in ("up", "down") else v)
                      for k, v in p.items()}
    out["meta"] = raw["meta"]
    return out


def default_biosets() -> dict:
    """The 4 Hammond axes of the RFind-sc paper.

    Signed references (500 up- and 500 down-regulated genes): ``DAM``
    (Keren-Shaul 2017), ``Maturity`` (Matcovitch-Natan 2016). Gene groups
    (MSigDB Hallmark, UP only): ``Interferon``, ``G2M``; a gene group has no
    order (``ranked=False``) and is used whole at scoring. Mouse symbols in
    reference rank order (gene groups: alphabetical). Each panel is a dict with
    ``up`` / ``down`` data frames (``gene``, ``diff``, ``rank``) and ``label``,
    ``type``, ``ranked``, ``source``; ``"meta"`` describes the bundle. Same
    content as RFindsc 0.3.6: the two signed references also carry
    ``eligible`` (the genes their study reported), used for the pair universe.
    """
    return _load_bundle("biosets.json")


def load_dedonno_biosets() -> dict:
    """The 4 De Donno axes of the RFind-sc paper: ``Glucocorticoid``
    (corticosterone vs vehicle, hippocampus, GSE184924; signed, 500 + 500),
    ``Seizure`` (kainate seizure 6 h vs saline, hippocampal excitatory
    neurons, GSE125068; top 200 up-regulated genes, ranked; type
    ``"ranked_up"``), ``UPR`` and ``NF-kB`` (MSigDB Hallmark, UP only; gene
    groups with ``ranked=False``). Same form as :func:`default_biosets`."""
    return _load_bundle("biosets_dedonno.json")


# --------------------------------------------------------------------------- DEG table

def build_panel_from_deg(deg, gene_col="gene", log2fc_col="log2FC", top_n=math.inf,
                         label="user_panel", source=None, padj_col=None,
                         padj_thresh=None) -> dict:
    """Build a signed panel from a DEG table.

    Genes with a positive log2 fold change form ``up`` (largest first), genes
    with a negative one form ``down`` (most negative first); rows with a
    missing / empty gene or a missing fold change are dropped. By default all
    genes are kept: scoring keeps the top ``top_n`` (500) per side and then the
    genes of the pair universe, as in the RFind-sc paper (0.3.8; before, the
    restriction came first).

    The panel also records ``eligible``: every gene of the table with a
    finite, non-zero log2 fold change (before ``padj_thresh`` and ``top_n``),
    i.e. the genes the study measured. Scoring uses the dataset's genes that
    are in ``eligible`` as the background (the pair universe; 0.3.5). Pass the
    full DEG table (all tested genes) for this to hold; delete
    ``panel["eligible"]`` to use all genes of the dataset instead.

    Parameters
    ----------
    deg : pandas.DataFrame
    gene_col, log2fc_col : str
    top_n : int or float
        Cap per side (default ``math.inf``: no cap).
    label, source : str
        Stored as metadata.
    padj_col : str, optional
    padj_thresh : float, optional
        If both are given and the column exists, rows with ``padj >
        padj_thresh`` are removed (rows with a missing padj are kept, as in R).

    Returns
    -------
    dict
        ``up`` / ``down`` data frames (``gene``, ``diff``, ``rank``),
        ``eligible`` (list; 0.3.5), ``type="deg"``, ``label``, ``source``.
    """
    if (not isinstance(deg, pd.DataFrame) or gene_col not in deg.columns
            or log2fc_col not in deg.columns):
        raise ValueError("`deg` must be a data frame with the gene and log2FC columns.")
    gene = deg[gene_col].map(lambda v: None if v is None or v is pd.NA
                             or (isinstance(v, float) and math.isnan(v)) else str(v))
    fc = pd.to_numeric(deg[log2fc_col], errors="coerce")
    ok = (gene.notna() & (gene != "") & fc.notna()).to_numpy()
    d = pd.DataFrame({"gene": gene.to_numpy()[ok], "log2fc": fc.to_numpy(dtype=np.float64)[ok]})
    measured = np.isfinite(d["log2fc"].to_numpy()) & (d["log2fc"].to_numpy() != 0)
    eligible = list(pd.unique(d["gene"].to_numpy()[measured]))   # genes the study measured
    if padj_col is not None and padj_thresh is not None and padj_col in deg.columns:
        padj = pd.to_numeric(deg[padj_col], errors="coerce").to_numpy(dtype=np.float64)[ok]
        d = d[np.isnan(padj) | (padj <= padj_thresh)]

    def side(sub, key):
        sub = sub.iloc[np.argsort(key, kind="stable")]
        if not math.isinf(float(top_n)):
            sub = sub.iloc[: int(top_n)]
        return pd.DataFrame({"gene": pd.Series(sub["gene"].to_numpy(), dtype=object),
                             "diff": sub["log2fc"].to_numpy(dtype=np.float64),
                             "rank": np.arange(1, len(sub) + 1, dtype=np.int64)})

    up = d[d["log2fc"] > 0]
    dn = d[d["log2fc"] < 0]
    return {"up": side(up, -up["log2fc"].to_numpy()),
            "down": side(dn, dn["log2fc"].to_numpy()),
            "eligible": eligible, "type": "deg", "label": label, "source": source}
