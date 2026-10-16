"""Per-cell scoring against panels (R: scoring.R, pipeline.R)."""
from __future__ import annotations

import math
import multiprocessing
import warnings
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import rfindsc_config
from .running_fisher import _is_missing, _running_fisher_many, to_b2
from .signature import _cap, _signature_from_resid

_LAYOUTS = ("genes_x_cells", "cells_x_genes")
_BLOCK = 16          # cells per block (dense residual block and one batched kernel call)


# --------------------------------------------------------------------------- input

def _is_anndata(x) -> bool:
    return all(hasattr(x, a) for a in ("X", "obs_names", "var_names", "layers"))


def extract_counts(input, genes=None, cells=None, layout=None, layer=None):
    """Counts matrix as genes x cells, with gene and cell names.

    Parameters
    ----------
    input : array, scipy sparse matrix, pandas DataFrame or AnnData
        Raw counts. Matrices and data frames are genes x cells by default (as
        in R); pass ``layout="cells_x_genes"`` for the transposed orientation.
        AnnData objects are cells x genes; ``layer`` selects
        ``adata.layers[layer]`` instead of ``adata.X``.
    genes, cells : sequence, optional
        Gene / cell names for a plain matrix (genes are required). Data frames
        and AnnData carry their own names.
    layout : {"genes_x_cells", "cells_x_genes"}, optional
    layer : str, optional

    Returns
    -------
    (matrix, genes, cells)
        A ``scipy.sparse.csc_matrix`` or dense ``ndarray`` (float64, genes x
        cells) and two ``pandas.Index``.
    """
    if layout is not None and layout not in _LAYOUTS:
        raise ValueError(f"`layout` must be one of {_LAYOUTS}.")
    if _is_anndata(input):
        if layout not in (None, "cells_x_genes"):
            raise ValueError("AnnData objects are cells x genes; do not set `layout`.")
        if genes is not None or cells is not None:
            raise ValueError("`genes` / `cells` are taken from the AnnData object.")
        mat = input.X if layer is None else input.layers[layer]
        genes, cells, layout = input.var_names, input.obs_names, "cells_x_genes"
    elif isinstance(input, pd.DataFrame):
        if genes is not None or cells is not None:
            raise ValueError("`genes` / `cells` are taken from the data frame.")
        layout = layout or "genes_x_cells"
        rows, cols = input.index, input.columns
        genes, cells = (rows, cols) if layout == "genes_x_cells" else (cols, rows)
        if len(input.columns) and all(isinstance(t, pd.SparseDtype) for t in input.dtypes):
            mat = input.sparse.to_coo()
        else:
            mat = input.to_numpy(dtype=np.float64)
    else:
        layout = layout or "genes_x_cells"
        mat = input
    if layout == "cells_x_genes":
        mat = mat.T
    if sp.issparse(mat):
        mat = sp.csc_matrix(mat, dtype=np.float64)
        mat.sum_duplicates()
    else:
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError("counts must be a 2-D matrix.")
    if genes is None:
        raise ValueError("gene names are required: pass `genes` (one per gene).")
    genes = pd.Index(genes)
    if len(genes) != mat.shape[0]:
        raise ValueError(f"{len(genes)} gene names for {mat.shape[0]} genes.")
    cells = pd.RangeIndex(mat.shape[1]) if cells is None else pd.Index(cells)
    if len(cells) != mat.shape[1]:
        raise ValueError(f"{len(cells)} cell names for {mat.shape[1]} cells.")
    return mat, genes, cells


# --------------------------------------------------------------------------- scoring

@dataclass
class _Context:
    X: object              # universe genes x cells (csc or dense)
    lib: np.ndarray        # library size per cell
    rates: np.ndarray      # baseline rate per universe gene x baseline group
    group: np.ndarray      # baseline group index per cell
    codes: np.ndarray      # symbol id per universe gene (lower case)
    panels: list           # per panel: (lookup_up, lookup_down), None for an empty side
    set_sizes: list        # per panel: (K_up, K_down), None for a ranked panel or an empty side
    masks: list            # per panel: boolean array over the universe (pair universe), None = all genes
    Ps: list               # per panel: size of the pair universe (distinct lower-case symbols)
    n_cell: object
    clip: float


def _lookup(codes, n_codes):
    """Position (first occurrence) of each symbol id in a panel side, -1 if absent."""
    lk = np.full(n_codes, -1, dtype=np.int64)
    if len(codes):
        uniq, first = np.unique(np.asarray(codes, dtype=np.int64), return_index=True)
        lk[uniq] = first
    return lk


def _baseline(X, lib, group, n_groups, nonzero_rate):
    """Universe mask and baseline expected-count rate per gene and group.

    rate = mean over the group's cells of count / libsize (R: rowMeans(cpm) / 1e6).
    """
    n_genes, n_cells = X.shape
    if sp.issparse(X):
        col = np.repeat(np.arange(n_cells), np.diff(X.indptr))
        cpm = (X.data / lib[col]) * 1e6
        detected = np.bincount(X.indices[cpm > 0], minlength=n_genes)
        sums = np.bincount(X.indices.astype(np.int64) * n_groups + group[col],
                           weights=cpm, minlength=n_genes * n_groups)
        sums = sums.reshape(n_genes, n_groups)
    else:
        cpm = (X / lib) * 1e6
        detected = (cpm > 0).sum(axis=1)
        sums = np.column_stack([cpm[:, group == k].sum(axis=1) for k in range(n_groups)])
    keep = detected / n_cells >= nonzero_rate
    size = np.bincount(group, minlength=n_groups)
    rates = (sums / size) / 1e6
    return keep, rates


def _score_block(ctx: _Context, cols: np.ndarray):
    sub = ctx.X[:, cols]
    obs = sub.toarray() if sp.issparse(sub) else np.asarray(sub, dtype=np.float64)
    expected = ctx.rates[:, ctx.group[cols]] * ctx.lib[cols]
    resid = (obs - expected) / np.sqrt(expected + 1)
    ups, downs, pos, sizes, Ps = [], [], [], [], []
    for j in range(len(cols)):
        r = resid[:, j]
        up, dn = _signature_from_resid(r, ctx.n_cell)
        ups.append(up)
        downs.append(dn)
        for (lk_up, lk_dn), (k_up, k_dn), mask, P in zip(ctx.panels, ctx.set_sizes, ctx.masks, ctx.Ps):
            # the cell's lists are cut inside the panel's pair universe (0.3.5)
            u, d = (up, dn) if mask is None else _signature_from_resid(r, ctx.n_cell, mask)
            cu, cd = ctx.codes[u], ctx.codes[d]
            for c1 in (cu, cd):
                for lk, k in ((lk_up, k_up), (lk_dn, k_dn)):
                    pos.append(None if lk is None or c1.size == 0 else lk[c1])
                    sizes.append(k)
                    Ps.append(P)
    rf = _running_fisher_many(pos, Ps, ctx.clip, sizes).reshape(len(cols), len(ctx.panels), 4)
    s = rf[..., 0] - rf[..., 1] - rf[..., 2] + rf[..., 3]   # UP.UP - UP.DOWN - DOWN.UP + DOWN.DOWN
    return np.maximum(-ctx.clip, np.minimum(s, ctx.clip)), ups, downs


_WORKER_CTX = None


def _score_block_in_worker(cols):
    return _score_block(_WORKER_CTX, cols)


def _run_blocks(ctx, blocks, n_cores):
    global _WORKER_CTX
    if (n_cores > 1 and len(blocks) > 1
            and "fork" in multiprocessing.get_all_start_methods()):
        _WORKER_CTX = ctx          # inherited by the forked workers, not pickled
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)   # fork + threads notice
                with ProcessPoolExecutor(max_workers=min(n_cores, len(blocks)),
                                         mp_context=multiprocessing.get_context("fork")) as ex:
                    return list(ex.map(_score_block_in_worker, blocks))
        finally:
            _WORKER_CTX = None
    return [_score_block(ctx, b) for b in blocks]


def _head(x, n):
    n = _cap(n)
    return list(x) if n is None else list(x)[:n]


def _panel_side(side, ranked, top_n, in_universe):
    """The genes of one side of a panel used in scoring (R: prepare_panel): a ranked list keeps its top ``top_n``
    genes and then those in the pair universe (order kept; 0.3.8, before which the restriction came first); a gene
    set (``ranked`` false) is used whole, restricted to the pair universe."""
    cand = _head(side, top_n) if ranked else list(side)
    return [g for g in cand if in_universe(g)]


def score_cells(counts, biosets, config=None, return_signatures=False, libsize=None, *,
                genes=None, cells=None, layout=None, layer=None, baseline_groups=None) -> dict:
    """Score all cells against all panels.

    For each cell, genes of the scoring universe are ranked by the P2 residual
    ``(x - e) / sqrt(e + 1)`` with ``e = pi_g * N_cell`` (``pi_g`` = mean over
    the baseline cells of count / libsize); the top ``n_cell`` positive and
    negative residuals form the cell's UP / DOWN signature, and each panel is
    scored with :func:`rfind_score_cell`.

    Pair universe (0.3.5): the background of each comparison is the scoring
    universe intersected with the panel's ``eligible`` genes (the genes the
    reference's study reported, set by :func:`build_panel_from_deg` and in the
    bundled ranked references); a panel without ``eligible`` (a gene set) uses
    the whole scoring universe. Each cell's UP / DOWN are restricted to it
    before the top ``n_cell`` genes are taken; a ranked panel keeps its top
    ``top_n`` genes per side and then those in the pair universe (0.3.8), and a
    gene set is used whole, as in the RFind-sc paper. Universe sizes count
    distinct lower-case symbols.

    Parameters
    ----------
    counts : array, sparse matrix, DataFrame or AnnData
        Raw counts; see :func:`extract_counts` for ``genes``, ``cells``,
        ``layout`` and ``layer``. The library size is the column sum, so pass
        all genes (or give ``libsize``).
    biosets : dict
        Panels by name (see :func:`default_biosets`); a ``"meta"`` entry is
        ignored. A panel with ``ranked=False`` (a membership gene set, e.g. the
        bundled ``Interferon`` group) is used whole; other panels are ranked
        lists (see :func:`to_b2`). A panel with ``eligible`` is scored in its
        pair universe (see above).
    config : dict, optional
        From :func:`rfindsc_config`.
    return_signatures : bool
        Also return the per-cell UP / DOWN indices.
    libsize : array-like, optional
        Total UMI per cell. Default: column sums of ``counts``. Pass the totals
        over all genes when ``counts`` is already restricted to a gene universe
        (as in the RFind-sc paper).
    baseline_groups : array-like, optional
        One label per cell. When given, each cell's expected counts use the
        mean over the cells of its own group (e.g. per cell type, as the paper
        does for De Donno et al.) instead of over all cells. The scoring
        universe is still chosen over all cells. Default None: one baseline
        over all cells. Same argument as ``baseline_groups`` in the R package
        (RFindsc >= 0.3.3).

    Returns
    -------
    dict
        ``scores``: DataFrame (cells x panels); ``universe``: list of universe
        genes; ``universe_size``: Series, the pair universe size per panel
        (0.3.5); ``signatures`` (if requested): ``up_idx`` / ``down_idx`` lists
        of 0-based index arrays into ``universe`` (top ``n_cell`` of the whole
        scoring universe), ``n_cell``, ``universe_size``.
    """
    config = rfindsc_config() if config is None else config
    if not isinstance(biosets, Mapping) or len(biosets) == 0:
        raise ValueError("`biosets` must be a non-empty dict of panels.")
    axis_names = [k for k in biosets if k != "meta"]
    if not axis_names:
        raise ValueError("`biosets` has no panels.")

    X, gene_ids, cell_ids = extract_counts(counts, genes=genes, cells=cells,
                                           layout=layout, layer=layer)
    n_genes, n_cells = X.shape
    if libsize is None:
        lib = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    else:
        lib = np.asarray(libsize, dtype=np.float64).ravel()
        if lib.size != n_cells:
            raise ValueError("`libsize` must have one value per cell.")
    if not np.all(lib > 0):
        raise ValueError("every cell needs a positive library size.")

    if baseline_groups is None:
        group, n_groups = np.zeros(n_cells, dtype=np.int64), 1
    else:
        labels = list(baseline_groups)
        if len(labels) != n_cells:
            raise ValueError("`baseline_groups` must have one label per cell.")
        group, uniq = pd.factorize(pd.Series(labels, dtype=object), sort=True)
        if (group < 0).any():
            raise ValueError("`baseline_groups` must not contain missing values.")
        group, n_groups = group.astype(np.int64), len(uniq)

    keep, rates = _baseline(X, lib, group, n_groups, config["nonzero_rate"])
    keep_idx = np.flatnonzero(keep)
    universe = [gene_ids[i] for i in keep_idx]
    P = len(universe)
    gene_names = [str(g).lower() for g in universe]
    codes, uniq_names = pd.factorize(pd.Series(gene_names, dtype=object))
    name_code = {g: i for i, g in enumerate(uniq_names)}

    # Each panel is scored in its pair universe (R: prepare_panel): the genes of
    # the dataset in the panel's `eligible`, or all of them for a panel without
    # `eligible` (a gene set). A ranked panel keeps its top top_n genes per side
    # and then those in the pair universe (order kept; 0.3.8), as in the RFind-sc
    # paper. A gene set (ranked = False) has no order: it is used whole, with K =
    # the number of its distinct genes in the universe.
    panels, set_sizes, masks, Ps = [], [], [], []
    for nm in axis_names:
        b = to_b2(biosets[nm])
        if b["eligible"] is None:
            name_ok, mask = None, None
            Ps.append(len(uniq_names))
        else:
            name_ok = np.fromiter((g in b["eligible"] for g in uniq_names), dtype=bool,
                                  count=len(uniq_names))
            mask = name_ok[codes]
            Ps.append(int(name_ok.sum()))
        sides, sizes = [], []
        in_u = lambda g: g in name_code and (name_ok is None or name_ok[name_code[g]])   # noqa: E731
        for side in (b["up"], b["down"]):
            kept = _panel_side(side, b["ranked"], config["top_n"], in_u)
            sides.append(_lookup([name_code[g] for g in kept], len(uniq_names)) if kept else None)
            sizes.append(None if b["ranked"] or not kept else len(set(kept)))
        panels.append(tuple(sides))
        set_sizes.append(tuple(sizes))
        masks.append(mask)

    ctx = _Context(X=X[keep_idx, :], lib=lib, rates=rates[keep_idx], group=group,
                   codes=codes.astype(np.int64), panels=panels, set_sizes=set_sizes,
                   masks=masks, Ps=Ps,
                   n_cell=_cap(config["n_cell"]), clip=float(config["clip"]))
    n_cores = max(1, int(config.get("n_cores", 1)))
    size = max(1, min(_BLOCK, math.ceil(n_cells / n_cores)))
    blocks = [np.arange(s, min(s + size, n_cells)) for s in range(0, n_cells, size)]
    results = _run_blocks(ctx, blocks, n_cores)

    scores = (np.vstack([r[0] for r in results]) if results
              else np.empty((0, len(axis_names))))
    out = {"scores": pd.DataFrame(scores, index=cell_ids, columns=axis_names),
           "universe": universe,
           "universe_size": pd.Series(Ps, index=axis_names, dtype=np.int64)}
    if return_signatures:
        out["signatures"] = {
            "up_idx": [u for r in results for u in r[1]],
            "down_idx": [d for r in results for d in r[2]],
            "n_cell": config["n_cell"],
            "universe_size": P,
        }
    return out


# --------------------------------------------------------------------------- wrappers

def run_pipeline(input, biosets=None, config=None, return_signatures=False, **kwargs) -> dict:
    """End-to-end scoring: counts (matrix, DataFrame or AnnData) + panels in,
    cell x panel scores out.

    ``biosets`` defaults to :func:`default_biosets` (the 4 Hammond axes of the
    RFind-sc paper). Extra keyword arguments (``genes``, ``cells``, ``layout``,
    ``layer``, ``libsize``, ``baseline_groups``) go to :func:`score_cells`.
    """
    if biosets is None:
        from .panels import default_biosets
        biosets = default_biosets()
    return score_cells(input, biosets, config=config,
                       return_signatures=return_signatures, **kwargs)


def _gene_vector(g) -> list:
    """R: as.character(g), dropping NA and empty strings."""
    if isinstance(g, str):
        g = [g]
    return [str(v) for v in g if not _is_missing(v) and str(v) != ""]


def _empty_side() -> pd.DataFrame:
    return pd.DataFrame({"gene": pd.Series([], dtype=object),
                         "diff": pd.Series([], dtype=np.float64),
                         "rank": pd.Series([], dtype=np.int64)})


def _side_frame(genes, diff) -> pd.DataFrame:
    return pd.DataFrame({"gene": pd.Series(genes, dtype=object),
                         "diff": np.full(len(genes), diff, dtype=np.float64),
                         "rank": np.arange(1, len(genes) + 1, dtype=np.int64)})


def score_genesets(input, gene_sets, config=None, return_signatures=False, **kwargs) -> dict:
    """Score cells against single-list gene sets (AUCell-style input).

    ``gene_sets`` maps names to gene lists; each set is scored with an empty
    DOWN side, i.e. ``RF(UP, set) - RF(DOWN, set)``. A set is a membership set
    with no order (``ranked=False``): it is used whole, so the order of its
    genes does not matter (0.3.4). Use :func:`score_panel_pairs` for signed
    (UP, DOWN) panels or a ranked UP-only list. Keyword arguments as in
    :func:`run_pipeline`.
    """
    if not isinstance(gene_sets, Mapping):
        if isinstance(gene_sets, (list, tuple)) and len(gene_sets) > 0:
            raise ValueError("`gene_sets` must have unique non-empty names for each gene set.")
        raise ValueError("`gene_sets` must be a non-empty named list of gene vectors.")
    if len(gene_sets) == 0:
        raise ValueError("`gene_sets` must be a non-empty named list of gene vectors.")
    if any(not isinstance(k, str) or k == "" for k in gene_sets):
        raise ValueError("`gene_sets` must have unique non-empty names for each gene set.")
    biosets = {nm: {"up": _side_frame(_gene_vector(g), np.nan), "down": _empty_side(),
                    "type": "gene_set", "ranked": False, "label": None, "source": None}
               for nm, g in gene_sets.items()}
    return score_cells(input, biosets, config=config,
                       return_signatures=return_signatures, **kwargs)


_SIDE_ERROR = "Each panel `up`/`down` must be a character vector or a data.frame with `gene`."


def _as_side(x, sign) -> pd.DataFrame:
    if x is None:
        return _empty_side()
    if isinstance(x, pd.DataFrame):
        if "gene" in x.columns:
            return x
        raise ValueError(_SIDE_ERROR)
    if isinstance(x, str):
        x = [x]
    try:
        vals = list(x)
    except TypeError:
        raise ValueError(_SIDE_ERROR) from None
    if not vals:
        return _empty_side()
    if all(isinstance(v, str) or _is_missing(v) for v in vals):
        return _side_frame(_gene_vector(vals), sign * 1.0)
    raise ValueError(_SIDE_ERROR)


def score_panel_pairs(input, panels, config=None, return_signatures=False, **kwargs) -> dict:
    """Score cells against (UP, DOWN) panel pairs (signed, four-term score).

    ``panels`` maps names to dicts with ``up`` (gene list or data frame with a
    ``gene`` column) and optionally ``down``. The lists are taken as ranked,
    most extreme first; add ``"ranked": False`` to a panel whose lists are
    unordered membership sets, so that they are used whole, and ``"eligible"``
    (the genes the reference's study measured) to score a panel in its pair
    universe (0.3.5). Keyword arguments as in :func:`run_pipeline`.
    """
    if not isinstance(panels, Mapping) or len(panels) == 0:
        raise ValueError("`panels` must be a non-empty named list.")
    if any(not isinstance(k, str) or k == "" for k in panels):
        raise ValueError("`panels` must have unique non-empty names.")
    biosets = {}
    for nm, p in panels.items():
        up, dn = _as_side(p.get("up"), 1), _as_side(p.get("down"), -1)
        biosets[nm] = {"up": up, "down": dn, "type": "deg" if len(dn) else "gene_set",
                       "label": None, "source": None}
        if p.get("ranked") is not None:
            biosets[nm]["ranked"] = p["ranked"]
        if p.get("eligible") is not None:
            biosets[nm]["eligible"] = list(p["eligible"])
    return score_cells(input, biosets, config=config,
                       return_signatures=return_signatures, **kwargs)
