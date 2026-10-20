"""Scoring tests (ports of tests/testthat/test-scoring.R, plus input handling)."""
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rfindsc import (build_cell_signature, rfindsc_config, score_cells, score_genesets,
                     score_panel_pairs)


def test_signature_ranks_by_p2_residual():
    obs = np.array([50, 0, 5, 20, 0])
    expected = np.array([10, 8, 5, 40, 1])
    resid = (obs - expected) / np.sqrt(expected + 1)      # +, -, 0, -, -
    sig = build_cell_signature(obs, expected, n_cell=500)
    np.testing.assert_array_equal(sig["up_idx"], np.flatnonzero(resid > 0))
    neg = np.flatnonzero(resid < 0)
    np.testing.assert_array_equal(sig["down_idx"], neg[np.argsort(resid[neg], kind="stable")])
    assert 2 not in set(sig["up_idx"]) | set(sig["down_idx"])   # residual 0 is in neither


def test_signature_caps_each_side():
    rng = np.random.default_rng(7)
    obs = np.concatenate([rng.poisson(20, 300), np.zeros(200)])
    sig = build_cell_signature(obs, np.full(500, 5.0), n_cell=50)
    assert len(sig["up_idx"]) <= 50 and len(sig["down_idx"]) <= 50


def test_signature_ties_keep_gene_order():
    sig = build_cell_signature([3, 3, 0, 3, 0], np.ones(5), n_cell=2)
    np.testing.assert_array_equal(sig["up_idx"], [0, 1])
    np.testing.assert_array_equal(sig["down_idx"], [2, 4])


def _toy(seed, n_genes=200, n_cells=40):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2, size=(n_genes, n_cells)).astype(float)
    genes = [f"g{i}" for i in range(1, n_genes + 1)]
    cells = [f"c{i}" for i in range(1, n_cells + 1)]
    return counts, genes, cells


def _panels():
    rng = np.random.default_rng(0)
    side = lambda genes, diff: pd.DataFrame({"gene": genes, "diff": diff,
                                             "rank": np.arange(1, len(genes) + 1)})
    return {
        "P1": {"up": side([f"g{i}" for i in range(1, 21)], rng.random(20)),
               "down": side([f"g{i}" for i in range(21, 41)], -rng.random(20)), "type": "deg"},
        "P2": {"up": side([f"g{i}" for i in range(41, 61)], rng.random(20)),
               "down": side([], []), "type": "gene_set"},
    }


CFG = rfindsc_config(n_cell=50, n_cores=1, nonzero_rate=0)


def test_score_cells_returns_cell_by_panel():
    counts, genes, cells = _toy(42)
    out = score_cells(counts, _panels(), CFG, genes=genes, cells=cells)
    assert out["scores"].shape == (40, 2)
    assert list(out["scores"].columns) == ["P1", "P2"]
    assert list(out["scores"].index) == cells


def test_score_genesets_accepts_gene_lists():
    counts, genes, cells = _toy(43, n_cells=30)
    gene_sets = {"set_a": [f"g{i}" for i in range(1, 21)], "set_b": [f"g{i}" for i in range(50, 76)]}
    out = score_genesets(counts, gene_sets, CFG, genes=genes, cells=cells)
    assert out["scores"].shape == (30, 2)
    assert list(out["scores"].columns) == ["set_a", "set_b"]


def test_score_panel_pairs_accepts_gene_lists():
    counts, genes, cells = _toy(44, n_cells=30)
    panels = {"pair1": {"up": [f"g{i}" for i in range(1, 21)], "down": [f"g{i}" for i in range(21, 41)]},
              "pair2": {"up": [f"g{i}" for i in range(41, 61)]}}
    out = score_panel_pairs(counts, panels, CFG, genes=genes, cells=cells)
    assert out["scores"].shape == (30, 2)
    assert list(out["scores"].columns) == ["pair1", "pair2"]


def test_score_genesets_validates_input():
    counts, genes, _ = _toy(45, n_genes=20, n_cells=5)
    cfg = rfindsc_config(n_cell=5, n_cores=1, nonzero_rate=0)
    with pytest.raises(ValueError, match="non-empty named list"):
        score_genesets(counts, {}, cfg, genes=genes)
    with pytest.raises(ValueError, match="unique non-empty names"):
        score_genesets(counts, [["g1", "g2"]], cfg, genes=genes)


def test_input_layouts_agree():
    counts, genes, cells = _toy(46, n_cells=25)
    ref = score_cells(counts, _panels(), CFG, genes=genes, cells=cells)["scores"]
    variants = {
        "csr": sp.csr_matrix(counts),
        "cells_x_genes": counts.T,
        "data_frame": pd.DataFrame(counts, index=genes, columns=cells),
        "anndata_like": SimpleNamespace(X=None, layers={"counts": sp.csr_matrix(counts.T)},
                                        obs_names=pd.Index(cells), var_names=pd.Index(genes)),
    }
    for name, x in variants.items():
        kw = {}
        if name in ("csr", "cells_x_genes"):
            kw = {"genes": genes, "cells": cells}
        if name == "cells_x_genes":
            kw["layout"] = "cells_x_genes"
        if name == "anndata_like":
            kw = {"layer": "counts"}
        got = score_cells(x, _panels(), CFG, **kw)["scores"]
        pd.testing.assert_frame_equal(got, ref, check_exact=True, check_index_type=False)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs fork")
def test_n_cores_gives_identical_scores():
    counts, genes, cells = _toy(47, n_cells=70)
    one = score_cells(counts, _panels(), CFG, genes=genes, cells=cells)["scores"]
    two = score_cells(counts, _panels(), dict(CFG, n_cores=2), genes=genes, cells=cells)["scores"]
    pd.testing.assert_frame_equal(one, two, check_exact=True)


def test_single_baseline_group_equals_default():
    counts, genes, cells = _toy(48)
    a = score_cells(counts, _panels(), CFG, genes=genes, cells=cells)["scores"]
    b = score_cells(counts, _panels(), CFG, genes=genes, cells=cells,
                    baseline_groups=["x"] * len(cells))["scores"]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_argument_checks():
    counts, genes, cells = _toy(49, n_cells=10)
    with pytest.raises(ValueError, match="libsize"):
        score_cells(counts, _panels(), CFG, libsize=np.ones(3), genes=genes)
    with pytest.raises(ValueError, match="gene names"):
        score_cells(counts, _panels(), CFG)
    with pytest.raises(ValueError, match="baseline_groups"):
        score_cells(counts, _panels(), CFG, genes=genes, baseline_groups=["a", None] * 5)


def test_signatures_are_returned():
    counts, genes, cells = _toy(50, n_cells=12)
    out = score_cells(counts, _panels(), CFG, genes=genes, cells=cells, return_signatures=True)
    sig = out["signatures"]
    assert len(sig["up_idx"]) == len(sig["down_idx"]) == 12
    assert sig["universe_size"] == len(out["universe"])
    assert all(len(u) <= 50 and (u < len(out["universe"])).all() for u in sig["up_idx"])


def test_score_genesets_uses_each_set_whole():
    counts, genes, cells = _toy(51, n_cells=30)
    counts[:20, :10] += np.random.default_rng(51).poisson(6, size=(20, 10))
    sets = {"set_a": [f"g{i}" for i in [*range(1, 21), *range(101, 111)]],
            "set_b": [f"g{i}" for i in range(50, 76)]}
    out = score_genesets(counts, sets, CFG, genes=genes, cells=cells)["scores"]
    shuffled = {k: v[::-1] for k, v in sets.items()}
    pd.testing.assert_frame_equal(score_genesets(counts, shuffled, CFG, genes=genes, cells=cells)["scores"],
                                  out, check_exact=True)
    # the same sets as UP-only panels marked ranked=False give the same scores
    pairs = {k: {"up": sorted(v), "ranked": False} for k, v in sets.items()}
    pd.testing.assert_frame_equal(score_panel_pairs(counts, pairs, CFG, genes=genes, cells=cells)["scores"],
                                  out, check_exact=True)
    # as ranked UP-only panels the order matters
    ranked = score_panel_pairs(counts, {k: {"up": v} for k, v in sets.items()}, CFG,
                               genes=genes, cells=cells)["scores"]
    assert not ranked.equals(out)
    if "fork" in multiprocessing.get_all_start_methods():
        two = score_genesets(counts, sets, dict(CFG, n_cores=2), genes=genes, cells=cells)["scores"]
        pd.testing.assert_frame_equal(two, out, check_exact=True)


def test_pair_universe_of_a_panel_with_eligible():
    """0.3.5: a panel with `eligible` is scored in its pair universe (R: same test in test-scoring.R)."""
    from rfindsc.running_fisher import rfind_score_cell, to_b2
    from rfindsc.signature import _signature_from_resid
    rng = np.random.default_rng(45)
    n_genes, n_cells = 300, 20
    counts = rng.poisson(2, size=(n_genes, n_cells)).astype(float)
    genes = [f"g{i}" for i in range(1, n_genes + 1)]
    up, dn = genes[:40], genes[40:80]
    el = genes[:80] + genes[150:250]
    bs = {"A": {"up": up, "down": dn, "type": "deg", "eligible": el},
          "B": {"up": up, "down": dn, "type": "deg"}}
    cfg = rfindsc_config(n_cell=30, n_cores=1, nonzero_rate=0)
    out = score_cells(counts, bs, cfg, genes=genes)
    assert out["universe_size"].tolist() == [180, 300]
    # by hand: restrict each cell's ranking to the pair universe, then cut to n_cell
    lib = counts.sum(axis=0)
    rate = ((counts / lib) * 1e6).mean(axis=1) / 1e6
    mask = np.isin(np.arange(n_genes), list(range(80)) + list(range(150, 250)))
    b = to_b2(bs["A"])
    gl = [g.lower() for g in genes]
    for j in range(n_cells):
        e = rate * lib[j]
        u, d = _signature_from_resid((counts[:, j] - e) / np.sqrt(e + 1), 30, mask)
        want = rfind_score_cell([gl[i] for i in u], [gl[i] for i in d], b, universe_size=180)
        assert out["scores"]["A"].iloc[j] == pytest.approx(want, abs=1e-10)
    # `eligible` covering every gene gives the same scores as no `eligible`
    bs["C"] = dict(bs["B"], eligible=genes)
    out2 = score_cells(counts, bs, cfg, genes=genes)
    np.testing.assert_allclose(out2["scores"]["C"], out2["scores"]["B"], rtol=0, atol=0)


def test_universe_size_counts_distinct_symbols():
    """0.3.5: symbols that differ only in case count once in the universe size (as in the paper)."""
    rng = np.random.default_rng(3)
    genes = [f"g{i}" for i in range(1, 51)] + ["G1"]
    counts = rng.poisson(3, size=(len(genes), 12)).astype(float)
    out = score_cells(counts, {"S": {"up": genes[:10], "ranked": False}},
                      rfindsc_config(n_cell=20, n_cores=1, nonzero_rate=0), genes=genes)
    assert len(out["universe"]) == 51 and out["universe_size"]["S"] == 50


def test_panel_side_cut_before_pair_universe():
    """0.3.8: a ranked panel keeps its top top_n genes before the pair universe; a gene set is used whole
    (R: the same test in test-scoring.R)."""
    from rfindsc.scoring import _panel_side
    genes = {f"g{i}" for i in range(1, 11)}
    in_u = genes.__contains__
    assert _panel_side(["g11", "g1", "g2", "g3"], True, 2, in_u) == ["g1"]      # the top 2 are g11 and g1
    assert _panel_side(["g4", "g12", "g5"], True, 2, in_u) == ["g4"]             # the top 2 are g4 and g12
    assert _panel_side(["g11", "g1", "g2", "g3"], False, 2, in_u) == ["g1", "g2", "g3"]   # a gene set is used whole


def test_score_cells_warns_once_about_names_that_differ_only_in_case():
    # 0.3.9: names are matched without regard to case, so PISD and Pisd would count as the same gene twice
    import warnings
    rng = np.random.default_rng(3)
    counts = rng.poisson(3, size=(60, 30)).astype(float)
    genes = ["Gdup", "GDUP"] + [f"g{i}" for i in range(3, 61)]
    cells = [f"c{i}" for i in range(1, 31)]
    bs = {"P": {"up": [f"g{i}" for i in range(3, 13)], "down": [f"g{i}" for i in range(13, 23)]}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score_cells(counts, bs, config=rfindsc_config(n_cores=1), genes=genes, cells=cells, layout="genes_x_cells")
    assert sum("occur in more than one row" in str(x.message) for x in w) == 1
