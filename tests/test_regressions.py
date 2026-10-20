"""Regressions for numerical zeroes, invalid inputs and optional signatures."""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import rfindsc.scoring as scoring
from rfindsc import build_cell_signature, rfindsc_config, score_cells, score_genesets, score_panel_pairs
from rfindsc.running_fisher import DuplicateGeneNamesWarning


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_cells", [10, 40, 1000])
def test_identical_cells_have_zero_scores_and_empty_signatures(sparse, n_cells):
    values = np.r_[np.full(20, 5), np.full(80, 3)]
    counts = np.repeat(values[:, None], n_cells, axis=1)
    genes = [f"g{i}" for i in range(100)]
    if sparse:
        counts = sp.csc_matrix(counts)
    result = score_genesets(counts, {"markers": genes[:20]}, genes=genes, return_signatures=True)
    assert (result["scores"].to_numpy() == 0).all()
    assert all(a.size == 0 for side in ("up_idx", "down_idx") for a in result["signatures"][side])


def test_proportional_cells_and_singleton_baselines_have_zero_scores():
    rng = np.random.default_rng(21)
    genes = [f"g{i}" for i in range(100)]
    counts = rng.integers(1, 10, 100)[:, None] * np.array([1, 2, 3, 7, 10])
    result = score_genesets(counts, {"markers": genes[:20]}, genes=genes)
    assert (result["scores"].to_numpy() == 0).all()
    counts = rng.poisson(3, (100, 40))
    result = score_genesets(counts, {"markers": genes[:20]}, genes=genes,
                            baseline_groups=list(range(40)), return_signatures=True)
    assert (result["scores"].to_numpy() == 0).all()
    assert all(a.size == 0 for side in ("up_idx", "down_idx") for a in result["signatures"][side])


def test_roundoff_guard_preserves_resolvable_positive_and_negative_differences():
    sig = build_cell_signature([5 + 1e-14, 5 + 1e-8, 5 - 1e-8], [5, 5, 5])
    np.testing.assert_array_equal(sig["up_idx"], [1])
    np.testing.assert_array_equal(sig["down_idx"], [2])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1])
@pytest.mark.parametrize("sparse", [False, True])
def test_invalid_counts_are_rejected_even_with_external_libsize(bad, sparse):
    counts = np.array([[bad, 1], [1, 2]], dtype=float)
    if sparse:
        counts = sp.csc_matrix(counts)
    with pytest.raises(ValueError, match="finite, non-negative"):
        score_genesets(counts, {"A": ["a"]}, genes=["a", "b"], libsize=[10, 10])


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0, -1])
def test_invalid_library_sizes_are_rejected(bad):
    with pytest.raises(ValueError, match="finite, positive"):
        score_genesets(np.ones((2, 2)), {"A": ["a"]}, genes=["a", "b"], libsize=[bad, 10])


@pytest.mark.parametrize("bad", [None, np.nan, "", " "])
def test_missing_gene_symbols_are_rejected(bad):
    with pytest.raises(ValueError, match="gene names"):
        score_genesets(np.ones((2, 2)), {"A": ["a"]}, genes=["a", bad])


@pytest.mark.parametrize("n_cores", [1, 2])
def test_reference_duplicates_warn_once_before_workers_and_keep_legacy_ranks(n_cores):
    counts = np.array([[10, 1], [1, 10], [5, 5], [2, 2]])
    cfg = rfindsc_config(n_cores=n_cores)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = score_panel_pairs(counts, {"A": {"up": ["a", " A ", "b"]}}, cfg,
                                    genes=["a", "b", "c", "d"])
    duplicate_warnings = [w for w in seen if issubclass(w.category, DuplicateGeneNamesWarning)]
    assert len(duplicate_warnings) == 1
    assert "Reference panel" in str(duplicate_warnings[0].message)
    np.testing.assert_allclose(result["scores"].A, [np.log10(3), -np.log10(3)], atol=1e-12)


@pytest.mark.parametrize("n_cores", [1, 2])
def test_signatures_are_not_retained_when_they_are_not_requested(monkeypatch, n_cores):
    original = scoring._run_blocks
    retained = []

    def inspect(ctx, blocks, cores):
        results = original(ctx, blocks, cores)
        retained.append(sum(a.nbytes for result in results for side in result[1:] for a in side))
        return results

    monkeypatch.setattr(scoring, "_run_blocks", inspect)
    counts = np.random.default_rng(9).poisson(2, (200, 40))
    kwargs = dict(genes=[f"g{i}" for i in range(200)], config=rfindsc_config(n_cores=n_cores))
    plain = score_genesets(counts, {"A": ["g1", "g2"]}, **kwargs)
    signed = score_genesets(counts, {"A": ["g1", "g2"]}, return_signatures=True, **kwargs)
    assert retained[0] == 0 and retained[1] > 0
    pd.testing.assert_frame_equal(plain["scores"], signed["scores"], check_exact=True)


def test_invalid_anndata_layer_is_rejected():
    ad = pytest.importorskip("anndata")
    obj = ad.AnnData(X=np.ones((2, 2)), var=pd.DataFrame(index=["a", "b"]))
    obj.layers["counts"] = np.array([[np.nan, 1], [1, 2]])
    with pytest.raises(ValueError, match="finite, non-negative"):
        score_cells(obj, {"A": {"up": ["a"]}}, layer="counts", libsize=[10, 10])
