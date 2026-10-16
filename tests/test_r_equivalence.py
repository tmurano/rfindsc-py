"""Numerical equivalence with the R package RFindsc 0.3.8.

The reference values in tests/data/ were produced by tests/data/make_fixtures.R
with RFindsc 0.3.8 (versions in tests/data/fixtures_meta.json). Scores must agree
to 1e-8 and signatures and universes exactly. Run ``pytest -s`` to print the
maximum differences.
"""
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp

from rfindsc import (default_biosets, load_dedonno_biosets, rfindsc_config,
                     running_fisher, score_cells, score_genesets, score_panel_pairs)

DATA = Path(__file__).parent / "data"
TOL = 1e-8
CFG = rfindsc_config(n_cores=1)


@pytest.fixture(scope="module")
def toy():
    with gzip.open(DATA / "toy_counts.mtx.gz", "rb") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # scipy's spmatrix -> sparray notice
        X = sp.csc_matrix(sio.mmread(fh), dtype=np.float64)
    genes = (DATA / "toy_genes.txt").read_text(encoding="utf-8").splitlines()
    cells = pd.read_csv(DATA / "toy_cells.tsv", sep="\t")
    return X, genes, cells


def _r_scores(name):
    return pd.read_csv(DATA / name, sep="\t", index_col=0, float_precision="round_trip")


def _compare(py, name):
    r = _r_scores(name)
    assert list(py.columns) == list(r.columns)
    assert [str(c) for c in py.index] == list(r.index)
    d = float(np.abs(py.to_numpy() - r.to_numpy()).max())
    print(f"\n[{name}] {r.shape[0]} cells x {r.shape[1]} panels: max |Python - R| = {d:.3g}")
    assert d < TOL


def _compare_signatures(res, name):
    ref = json.loads((DATA / name).read_text())
    assert res["universe"] == ref["universe"]
    for key, rkey in (("up_idx", "up"), ("down_idx", "down")):
        assert len(res["signatures"][key]) == len(ref[rkey])
        for got, want in zip(res["signatures"][key], ref[rkey]):
            np.testing.assert_array_equal(got + 1, np.asarray(want, dtype=np.int64))


def test_kernel_matches_r():
    cases = json.loads((DATA / "kernel_cases.json").read_text())
    names = lambda v: [f"g{i}" for i in np.atleast_1d(np.asarray(v, dtype=np.int64))]
    d = [abs(running_fisher(names(c["g1"]), names(c["g2"]), c["P"], c["clip"],
                            g2_ranked=c.get("ranked", True)) - float(c["score"]))
         for c in cases]
    print(f"\n[kernel] {len(cases)} list pairs: max |Python - R| = {max(d):.3g}")
    assert max(d) < TOL


def test_hammond_axes_and_signatures(toy):
    X, genes, cells = toy
    res = score_cells(X, default_biosets(), CFG, return_signatures=True, genes=genes, cells=cells["cell"])
    _compare(res["scores"], "scores_hammond.tsv")
    _compare_signatures(res, "signatures_hammond.json")
    assert res["universe_size"].to_dict() == json.loads((DATA / "universe_size_hammond.json").read_text())


def test_dedonno_axes(toy):
    X, genes, cells = toy
    _compare(score_cells(X, load_dedonno_biosets(), CFG, genes=genes, cells=cells["cell"])["scores"],
             "scores_dedonno.tsv")


def test_libsize_on_a_restricted_matrix(toy):
    X, genes, cells = toy
    sub = (DATA / "libsize_genes.txt").read_text(encoding="utf-8").splitlines()
    rows = [genes.index(g) for g in sub]
    lib = np.asarray(X.sum(axis=0)).ravel()
    res = score_cells(X[rows, :], default_biosets(), CFG, libsize=lib, genes=sub, cells=cells["cell"])
    _compare(res["scores"], "scores_libsize.tsv")


def _both():
    dd = load_dedonno_biosets()
    return {**default_biosets(), **{k: v for k, v in dd.items() if k != "meta"}}


def test_config_variant(toy):
    X, genes, cells = toy
    cfg = rfindsc_config(top_n=100, n_cell=60, nonzero_rate=0.05, clip=12, n_cores=1)
    res = score_cells(X, _both(), cfg, return_signatures=True, genes=genes, cells=cells["cell"])
    _compare(res["scores"], "scores_config.tsv")
    _compare_signatures(res, "signatures_config.json")


def test_score_genesets(toy):
    X, genes, cells = toy
    gene_sets = json.loads((DATA / "genesets.json").read_text())
    _compare(score_genesets(X, gene_sets, CFG, genes=genes, cells=cells["cell"])["scores"],
             "scores_genesets.tsv")


def test_score_panel_pairs(toy):
    X, genes, cells = toy
    pairs = json.loads((DATA / "panel_pairs.json").read_text())
    bs = default_biosets()
    pairs["pair_df"] = {"up": bs["Maturity"]["up"].iloc[:150], "down": bs["Maturity"]["down"].iloc[:150]}
    _compare(score_panel_pairs(X, pairs, CFG, genes=genes, cells=cells["cell"])["scores"],
             "scores_panel_pairs.tsv")


def test_per_group_baseline(toy):
    X, genes, cells = toy
    res = score_cells(X, _both(), CFG, genes=genes, cells=cells["cell"],
                      baseline_groups=cells["group"])
    _compare(res["scores"], "scores_grouped.tsv")


def test_dense_input_matches_sparse(toy):
    X, genes, cells = toy
    a = score_cells(X, default_biosets(), CFG, genes=genes)["scores"]
    b = score_cells(X.toarray(), default_biosets(), CFG, genes=genes)["scores"]
    assert float(np.abs(a.to_numpy() - b.to_numpy()).max()) < TOL
