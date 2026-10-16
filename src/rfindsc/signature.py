"""Per-cell UP/DOWN signatures from the P2 Pearson residual (Delta)."""
from __future__ import annotations

import math

import numpy as np


def _cap(n):
    """Signature / panel length cap: None means no cap (R: head(x, Inf))."""
    if n is None:
        return None
    n = float(n)
    return None if math.isinf(n) else int(n)


def _head_desc(idx: np.ndarray, v: np.ndarray, n):
    """``idx`` ordered by decreasing ``v`` (ties in index order), first ``n``.

    Equivalent to a full stable sort followed by ``head(n)`` (as R's
    ``order(..., decreasing = TRUE)``), but only the candidates that can reach
    the top ``n`` are sorted.
    """
    if n is not None and idx.size > n:
        if n <= 0:
            return idx[:0]
        kth = np.partition(v, idx.size - n)[idx.size - n]   # n-th largest value
        keep = v >= kth
        idx, v = idx[keep], v[keep]
    order = np.argsort(-v, kind="stable")
    out = idx[order]
    return out if n is None else out[:n]


def _signature_from_resid(resid: np.ndarray, n, mask=None):
    """UP / DOWN of one cell, first ``n`` each. With ``mask`` (a boolean array
    over the universe; 0.3.5) only the genes of a pair universe are ranked,
    which equals ranking all genes, keeping those in the mask and then cutting
    (R: score_cell_ranking)."""
    if mask is None:
        pos = np.flatnonzero(resid > 0)
        neg = np.flatnonzero(resid < 0)
    else:
        pos = np.flatnonzero((resid > 0) & mask)
        neg = np.flatnonzero((resid < 0) & mask)
    return _head_desc(pos, resid[pos], n), _head_desc(neg, -resid[neg], n)


def build_cell_signature(obs, expected, n_cell=500) -> dict:
    """Per-cell UP/DOWN signature from Pearson residuals (Delta = P2).

    Ranks the genes of one cell by ``Delta = (obs - exp) / sqrt(exp + 1)``,
    where ``exp`` is the count expected under the baseline rate scaled to the
    cell's library size. UP = genes with a positive residual, largest first;
    DOWN = genes with a negative residual, most negative first; at most
    ``n_cell`` each (fewer when fewer residuals have that sign; ties keep gene
    order).

    Parameters
    ----------
    obs : array-like
        Observed counts of one cell, one entry per universe gene.
    expected : array-like
        Expected counts (baseline rate x library size), same order.
    n_cell : int or float
        Maximum genes per side (default 500; ``math.inf`` for no cap).

    Returns
    -------
    dict
        ``up_idx`` and ``down_idx``: 0-based integer indices into the universe
        (the R package returns 1-based indices).
    """
    obs = np.asarray(obs, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    resid = (obs - expected) / np.sqrt(expected + 1)
    up, down = _signature_from_resid(resid, _cap(n_cell))
    return {"up_idx": up, "down_idx": down}
