"""Kernel tests (ports of the R package's tests/testthat/test-running-fisher.R)."""
import numpy as np
import pytest
from scipy import stats

from rfindsc import bid_running_fisher, rfind_score_cell, running_fisher, to_b2
from rfindsc._hyper import log_upper_tail


def test_empty_input_returns_zero():
    assert running_fisher([], ["a", "b", "c"], 100) == 0
    assert running_fisher(["a", "b", "c"], [], 100) == 0


def test_bounded_by_clip():
    g = [f"g{i}" for i in range(1, 11)]
    s = running_fisher(g, g, universe_size=1000, clip=50)
    assert 0 < s <= 50
    assert running_fisher(g, g, universe_size=1000, clip=500) >= s - 1e-8


def test_symmetric_up_to_fp_noise():
    rng = np.random.default_rng(1)
    pool = np.array([f"g{i}" for i in range(1, 501)])
    g1, g2 = rng.choice(pool, 50, replace=False), rng.choice(pool, 60, replace=False)
    assert running_fisher(g1, g2, 500) == pytest.approx(running_fisher(g2, g1, 500), abs=1e-8)


def _brute(g1, g2, P):
    """-log10 of the smallest P(X >= M) over every (top G, top K) pair."""
    best = 0.0
    for G in range(1, len(g1) + 1):
        for K in range(1, len(g2) + 1):
            M = len(set(g1[:G]) & set(g2[:K]))
            if M > 0:
                best = max(best, -stats.hypergeom(P, K, G).logsf(M - 1) / np.log(10))
    return best


def test_equals_full_gxk_scan():
    rng = np.random.default_rng(2)
    pool = [f"g{i}" for i in range(1, 301)]
    for _ in range(20):
        g1 = list(rng.choice(pool, rng.integers(5, 41), replace=False))
        g2 = list(dict.fromkeys(list(rng.choice(g1, 4, replace=False))
                                + list(rng.choice(pool, rng.integers(5, 41), replace=False))))
        assert running_fisher(g1, g2, 300) == pytest.approx(min(_brute(g1, g2, 300), 300), abs=1e-8)


def _brute_set(g1, gene_set, P):
    """-log10 of the smallest P(X >= M) over the top-G cut-offs of g1, K = set size."""
    K = len(set(gene_set))
    best = 0.0
    for G in range(1, len(g1) + 1):
        M = len(set(g1[:G]) & set(gene_set))
        if M > 0:
            best = max(best, -stats.hypergeom(P, K, G).logsf(M - 1) / np.log(10))
    return best


def test_unordered_set_equals_scan_over_g_only():
    rng = np.random.default_rng(5)
    pool = [f"g{i}" for i in range(1, 301)]
    for _ in range(20):
        g1 = list(rng.choice(pool, rng.integers(5, 61), replace=False))
        gene_set = list(dict.fromkeys(list(rng.choice(g1, 4, replace=False))
                                      + list(rng.choice(pool, rng.integers(5, 61), replace=False))))
        s = running_fisher(g1, gene_set, 300, g2_ranked=False)
        assert s == pytest.approx(min(_brute_set(g1, gene_set, 300), 300), abs=1e-8)
        # the order in which the set is stored does not matter
        assert running_fisher(g1, list(rng.permutation(gene_set)), 300, g2_ranked=False) == s
        assert running_fisher(g1, sorted(gene_set), 300, g2_ranked=False) == s
    assert running_fisher([], ["a"], 100, g2_ranked=False) == 0
    assert running_fisher(["a", "b"], ["c"], 100, g2_ranked=False) == 0
    assert running_fisher(list("abcde"), list("abcde"), 1000, clip=5, g2_ranked=False) == 5


def test_ranked_default_unchanged():
    rng = np.random.default_rng(6)
    pool = [f"g{i}" for i in range(1, 401)]
    for _ in range(10):
        g1, g2 = list(rng.choice(pool, 80, replace=False)), list(rng.choice(pool, 60, replace=False))
        assert running_fisher(g1, g2, 400, g2_ranked=True) == running_fisher(g1, g2, 400)


def test_alias():
    rng = np.random.default_rng(3)
    pool = [f"g{i}" for i in range(1, 201)]
    g1, g2 = rng.choice(pool, 30, replace=False), rng.choice(pool, 30, replace=False)
    assert bid_running_fisher(g1, g2, 200) == running_fisher(g1, g2, 200)


def test_log_upper_tail_matches_scipy():
    rng = np.random.default_rng(4)
    P = 5000
    K = rng.integers(1, 600, 3000)
    G = rng.integers(1, 600, 3000)
    M = np.minimum(np.minimum(K, G), rng.integers(1, 80, 3000))
    ours = log_upper_tail(M, G, K, P)
    ref = stats.hypergeom(P, K, G).logsf(M - 1)
    ok = ref > -700                    # scipy's logsf loses accuracy only near underflow
    np.testing.assert_allclose(ours[ok], ref[ok], rtol=1e-9, atol=1e-12)


def test_four_term_score():
    b1u, b1d = ["a", "b", "c", "d"], ["x", "y", "z"]
    b2 = to_b2({"up": ["A", "c", "q"], "down": [" Y ", "x"]})
    assert b2["up"] == ["a", "c", "q"] and b2["down"] == ["y", "x"]
    rf = lambda g1, g2: running_fisher(g1, g2, 100)
    expected = rf(b1u, b2["up"]) - rf(b1u, b2["down"]) - rf(b1d, b2["up"]) + rf(b1d, b2["down"])
    assert rfind_score_cell(b1u, b1d, b2, 100) == pytest.approx(expected, abs=1e-12)
    # unsigned set: RF(UP, set) - RF(DOWN, set)
    b2u = to_b2({"up": ["a", "x"]})
    assert b2u["down"] == []
    assert rfind_score_cell(b1u, b1d, b2u, 100) == pytest.approx(
        rf(b1u, ["a", "x"]) - rf(b1d, ["a", "x"]), abs=1e-12)


def test_four_term_score_with_an_unordered_set():
    rng = np.random.default_rng(7)
    pool = [f"g{i}" for i in range(1, 501)]
    b1u = list(rng.choice(pool, 60, replace=False))
    b1d = list(rng.choice([g for g in pool if g not in b1u], 60, replace=False))
    up = list(rng.choice(pool, 40, replace=False))
    dn = list(rng.choice([g for g in pool if g not in up], 30, replace=False))
    rf = lambda g1, g2: running_fisher(g1, g2, 500, g2_ranked=False)
    b2 = {"up": up, "down": dn, "ranked": False}
    assert rfind_score_cell(b1u, b1d, b2, 500) == pytest.approx(
        rf(b1u, up) - rf(b1u, dn) - rf(b1d, up) + rf(b1d, dn), abs=1e-12)
    assert rfind_score_cell(b1u, b1d, {"up": up[::-1], "down": sorted(dn), "ranked": False}, 500) == \
        rfind_score_cell(b1u, b1d, b2, 500)
    rr = lambda g1, g2: running_fisher(g1, g2, 500)
    assert rfind_score_cell(b1u, b1d, {"up": up, "down": dn}, 500) == pytest.approx(
        rr(b1u, up) - rr(b1u, dn) - rr(b1d, up) + rr(b1d, dn), abs=1e-12)


def test_to_b2_marks_membership_sets_unranked():
    assert to_b2({"up": ["A"], "down": ["B"], "type": "deg"})["ranked"] is True
    assert to_b2({"up": ["A"], "type": "gene_set"})["ranked"] is True
    assert to_b2({"up": ["A"]})["ranked"] is True
    for t in ("gene_group", "msigdb_type_b", "msigdb_cgp_paired"):
        assert to_b2({"up": ["A"], "type": t})["ranked"] is False
    assert to_b2({"up": ["A"], "type": "gene_set", "ranked": False})["ranked"] is False
    assert to_b2({"up": ["A"], "type": "gene_group", "ranked": True})["ranked"] is True
    assert to_b2({"up": ["A"], "type": "gene_group", "ranked": np.bool_(True)})["ranked"] is True
    assert to_b2({"up": ["A"], "ranked": np.bool_(False)})["ranked"] is False


def test_duplicated_names_warn_and_score_as_before():
    # 0.3.9 (R: class rfindsc_duplicated_names): a name given twice counts twice; the warning says so
    from rfindsc.running_fisher import DuplicateGeneNamesWarning
    with pytest.warns(DuplicateGeneNamesWarning):
        s = running_fisher(["a", "a", "b", "c"], ["a", "d", "b"], 50)
    assert np.isfinite(s)
