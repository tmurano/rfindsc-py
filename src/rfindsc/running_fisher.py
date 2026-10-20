"""Running Fisher kernel and the four-term RFind-sc score (R: running_fisher.R)."""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ._hyper import log_upper_tail

_LN10 = math.log(10.0)


class DuplicateGeneNamesWarning(UserWarning):
    """A gene list names a gene more than once (0.3.9; R: class ``rfindsc_duplicated_names``)."""


def _corners(pos2: np.ndarray):
    """Cut-off pairs to evaluate for one pair of lists.

    ``pos2[i]`` is the 0-based position in the second list of element ``i`` of
    the first list (-1 when absent; first occurrence, as R's ``match``).
    Returns ``None`` when the lists share nothing, else ``(M, G, K, top)``:
    the overlap ``M`` of the top ``G`` of list 1 and the top ``K`` of list 2 at
    the evaluated corners, and the largest ``G`` / ``K`` over all corners.

    The corners are the pairs of ranks of shared genes (R evaluates all
    ``m x m`` of them). A corner whose overlap equals that of the corner one
    shared gene earlier in either list has a larger ``G`` or ``K`` at the same
    ``M`` and so a p that is not smaller; those corners are skipped, which
    leaves the minimum unchanged.
    """
    sel = np.flatnonzero(pos2 >= 0)
    if sel.size == 0:
        return None
    r1 = sel + 1                             # ranks in list 1 (increasing)
    r2 = pos2[sel] + 1                       # their ranks in list 2
    K = np.sort(r2)
    M = np.cumsum(r2[:, None] <= K[None, :], axis=0)   # M[a, b]: top r1[a] x top K[b]
    keep = ((np.diff(M, axis=0, prepend=-1) != 0)
            & (np.diff(M, axis=1, prepend=-1) != 0))
    a, b = np.nonzero(keep)
    return M[a, b], r1[a], K[b], max(int(r1[-1]), int(K[-1]))


def _set_corners(pos2: np.ndarray, set_size: int):
    """Cut-offs to evaluate when the second list is an unordered gene set.

    The set is used whole (``K`` = ``set_size``, the number of distinct genes
    in it) and only the top-``G`` cut-offs of list 1 are scanned: for ``G`` at
    the rank of the ``j``-th shared gene the overlap is ``M = j`` (R:
    ``running_fisher(g2_ranked = FALSE)``). Same return value as ``_corners``.
    """
    sel = np.flatnonzero(pos2 >= 0)
    if sel.size == 0:
        return None
    set_size = int(set_size)
    M = np.arange(1, sel.size + 1, dtype=np.int64)
    return M, sel + 1, np.full(sel.size, set_size, dtype=np.int64), max(int(sel[-1]) + 1, set_size)


def _running_fisher_many(pos2_list, universe_size, clip, set_sizes=None) -> np.ndarray:
    """Running Fisher scores of many list pairs at once.

    Each element of ``pos2_list`` is a ``pos2`` array (see ``_corners``) or
    ``None`` for an empty list. ``universe_size`` is one size for all pairs or
    one per pair (each panel has its own pair universe since 0.3.5; pairs are
    grouped by size). ``set_sizes``, if given, has one entry per pair: ``None``
    when the second list is ranked, or the number of distinct genes of the
    second list when it is an unordered gene set (see ``_set_corners``). All
    tail probabilities of one universe size are evaluated in one vectorised
    call.
    """
    if np.ndim(universe_size) == 0:
        return _running_fisher_one_universe(pos2_list, universe_size, clip, set_sizes)
    Ps = np.rint(np.asarray(universe_size, dtype=np.float64)).astype(np.int64)
    if Ps.size != len(pos2_list):
        raise ValueError("`universe_size` must be a scalar or have one value per pair.")
    out = np.zeros(len(pos2_list))
    for P in np.unique(Ps):
        idx = np.flatnonzero(Ps == P)
        out[idx] = _running_fisher_one_universe(
            [pos2_list[t] for t in idx], int(P), clip,
            None if set_sizes is None else [set_sizes[t] for t in idx])
    return out


def _running_fisher_one_universe(pos2_list, universe_size, clip, set_sizes=None) -> np.ndarray:
    P = int(round(universe_size))
    out = np.zeros(len(pos2_list))
    Ms, Gs, Ks, owner = [], [], [], []
    for t, pos2 in enumerate(pos2_list):
        if pos2 is None or len(pos2) == 0:
            continue
        k_set = None if set_sizes is None else set_sizes[t]
        c = _corners(pos2) if k_set is None else _set_corners(pos2, k_set)
        if c is None:
            continue
        M, G, K, top = c
        if top > P:
            # A list is longer than the universe: R's phyper() returns NaN, and
            # a non-finite score is returned as `clip`.
            out[t] = clip
            continue
        Ms.append(M)
        Gs.append(G)
        Ks.append(K)
        owner.append(t)
    if owner:
        sizes = np.fromiter((m.size for m in Ms), dtype=np.int64, count=len(Ms))
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        lp = log_upper_tail(np.concatenate(Ms), np.concatenate(Gs),
                            np.concatenate(Ks), P)
        s = -np.minimum.reduceat(lp, starts) / _LN10
        out[owner] = np.where(np.isfinite(s), np.minimum(s, clip), clip)
    return out


def _positions(g1: Sequence, g2: Sequence) -> np.ndarray:
    """0-based position in ``g2`` (first occurrence) of each element of ``g1``."""
    first = {}
    for i, g in enumerate(g2):
        first.setdefault(g, i)
    return np.fromiter((first.get(g, -1) for g in g1), dtype=np.int64,
                       count=len(g1))


def running_fisher(g1, g2, universe_size, clip=300.0, g2_ranked=True) -> float:
    """Running Fisher score of two ranked gene lists.

    ``-log10`` of the smallest hypergeometric upper-tail probability
    ``P(X >= M)`` over every pair of cut-offs: the top ``G`` genes of ``g1``
    and the top ``K`` genes of ``g2`` (``M`` = genes shared by the two
    prefixes; Kupershmidt et al. 2010, G x K form). The probabilities are
    intermediate quantities of the similarity score, computed on the log scale.
    Only cut-offs at ranks of shared genes are evaluated, which gives the same
    minimum as the full G x K scan.

    A membership gene set (e.g. an MSigDB Hallmark set) has no order. With
    ``g2_ranked=False`` it is used whole, as a biogroup in the original Running
    Fisher: ``K`` is fixed at the number of distinct genes in ``g2`` and only
    the top-``G`` cut-offs of ``g1`` are scanned, so the score does not depend
    on the order of ``g2`` (added in 0.3.4, as in RFindsc 0.3.4).

    Parameters
    ----------
    g1 : sequence of str
        Ranked gene symbols, most extreme first.
    g2 : sequence of str
        Ranked gene symbols, or a membership gene set when ``g2_ranked=False``.
    universe_size : int
        Size ``P`` of the gene universe (hypergeometric null).
    clip : float
        Score cap (default 300).
    g2_ranked : bool
        ``False`` when ``g2`` is an unordered gene set (default ``True``).

    Returns
    -------
    float
        The score, clipped to ``[0, clip]``; 0 when either list is empty or the
        lists share no gene.

    Notes
    -----
    Each list should name each gene once: a name given twice counts as two
    genes and inflates the overlap ``M``. Since 0.3.9 such input gives a
    :class:`DuplicateGeneNamesWarning`; the score is computed as before.
    """
    g1 = list(g1)
    g2 = list(g2)
    if not g1 or not g2:
        return 0.0
    if len(set(g1)) < len(g1) or len(set(g2)) < len(g2):
        warnings.warn("running_fisher(): a gene list names a gene more than once; each copy counts as a separate gene "
                      "and inflates the overlap. Remove the duplicates before matching.", DuplicateGeneNamesWarning,
                      stacklevel=2)
    sizes = None if g2_ranked else [len(set(g2))]
    return float(_running_fisher_many([_positions(g1, g2)], universe_size, clip, sizes)[0])


def bid_running_fisher(g1, g2, universe_size, clip=300.0, g2_ranked=True) -> float:
    """Former name of :func:`running_fisher` (identical since 0.3.0)."""
    return running_fisher(g1, g2, universe_size, clip, g2_ranked=g2_ranked)


def _is_missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _is_false(v) -> bool:
    """R: isFALSE(v) (a single logical FALSE)."""
    return v is False or (isinstance(v, np.bool_) and not v)


# Panel types that are membership gene sets (no order) when a panel has no
# ``ranked`` field: bundled MSigDB Hallmark groups and the panels of RFindsc's
# msigdbr_to_panels() / c2_cgp_auto_pair().
_UNRANKED_TYPES = ("gene_group", "msigdb_type_b", "msigdb_cgp_paired")


def _panel_is_ranked(panel) -> bool:
    """R: panel_is_ranked(). ``False`` for an unordered membership gene set."""
    ranked = panel.get("ranked")
    if ranked is not None:
        return not _is_false(ranked)
    t = panel.get("type")
    return not (isinstance(t, str) and t in _UNRANKED_TYPES)


def _side_genes(side) -> list:
    """Gene column of one panel side (data frame with ``gene``, or a list)."""
    if side is None:
        return []
    if isinstance(side, pd.DataFrame):
        return side["gene"].tolist()
    if isinstance(side, Mapping):
        return list(side["gene"])
    if isinstance(side, str):
        return [side]
    return list(side)


def _clean_symbol(v):
    """R: tolower(trimws(x)); missing values stay missing (None)."""
    if _is_missing(v):
        return None
    return str(v).strip(" \t\r\n").lower()


def to_b2(panel) -> dict:
    """Convert a panel into scoring form.

    Parameters
    ----------
    panel : dict
        With ``up`` and ``down`` (data frames with a ``gene`` column, as in
        :func:`default_biosets`; a plain list of symbols is also accepted; a
        missing ``down`` is an empty side). An optional ``ranked=False`` marks
        a membership gene set, whose lists have no order and are used whole;
        without it, panels of type ``"gene_group"``, ``"msigdb_type_b"`` or
        ``"msigdb_cgp_paired"`` are gene sets and all other panels are ranked
        lists. An optional ``eligible`` (the genes the reference's study
        reported; 0.3.5) sets the pair universe (see :func:`score_cells`).

    Returns
    -------
    dict
        ``up`` / ``down`` lists of lower-case, trimmed symbols, their sizes
        ``n_up`` / ``n_down``, ``ranked``, and ``eligible`` (a set of
        lower-case symbols, or None).
    """
    up = [_clean_symbol(g) for g in _side_genes(panel.get("up"))]
    dn = [_clean_symbol(g) for g in _side_genes(panel.get("down"))]
    el = panel.get("eligible")
    if el is not None:
        if isinstance(el, str):
            el = [el]
        el = {c for c in (_clean_symbol(g) for g in el) if c}
    return {"up": up, "down": dn, "n_up": len(up), "n_down": len(dn),
            "ranked": _panel_is_ranked(panel), "eligible": el}


def rfind_score_cell(b1_up, b1_down, b2, universe_size, clip=300.0) -> float:
    """Four-term RFind-sc score of one cell against one panel.

    ``S = RF(b1_up, up) - RF(b1_up, down) - RF(b1_down, up) + RF(b1_down, down)``
    with ``RF`` = :func:`running_fisher`. For a gene set without a DOWN side
    this is ``RF(b1_up, set) - RF(b1_down, set)``. Positive = aligned with the
    panel's case direction, negative = aligned with control. When
    ``b2["ranked"]`` is ``False`` (a membership gene set) the panel lists are
    used whole in all four terms (``g2_ranked=False``).

    Parameters
    ----------
    b1_up, b1_down : sequence of str
        Per-cell UP / DOWN signatures (ranked).
    b2 : dict
        Panel in scoring form (see :func:`to_b2`); a missing ``ranked`` means
        ranked lists.
    universe_size : int
        Size of the gene universe.
    clip : float
        Score magnitude cap (symmetric, default 300).
    """
    rk = not _is_false(b2.get("ranked"))      # b2["ranked"] = False for a membership gene set
    s = (running_fisher(b1_up, b2["up"], universe_size, clip, rk)
         - running_fisher(b1_up, b2["down"], universe_size, clip, rk)
         - running_fisher(b1_down, b2["up"], universe_size, clip, rk)
         + running_fisher(b1_down, b2["down"], universe_size, clip, rk))
    return float(np.maximum(-clip, np.minimum(s, clip)))
