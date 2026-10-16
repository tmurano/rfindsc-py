"""Hyperparameter defaults (R: config.R)."""
from __future__ import annotations


def rfindsc_config(top_n=500, n_cell=500, nonzero_rate=0.01, clip=300.0, n_cores=1) -> dict:
    """RFind-sc configuration defaults.

    Parameters
    ----------
    top_n : int or float
        Cap on each side of a ranked panel, applied before the restriction to
        the pair universe (0.3.8; default 500, the RFind-sc paper; ``math.inf``
        for no cap). Gene sets are used whole.
    n_cell : int or float
        Cap on each side of the per-cell signature (default 500).
    nonzero_rate : float
        Minimum fraction of cells in which a gene is non-zero to enter the
        scoring universe (default 0.01).
    clip : float
        Score magnitude cap, symmetric (default 300).
    n_cores : int
        Worker processes. Values > 1 fork workers (POSIX only, like R's
        ``mclapply``); results do not depend on it. Default 1 (the R package
        defaults to the number of cores minus one; 1 on Windows).

    Returns
    -------
    dict
    """
    return {"top_n": top_n, "n_cell": n_cell, "nonzero_rate": nonzero_rate,
            "clip": clip, "n_cores": n_cores}
