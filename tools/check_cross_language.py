"""Run both source trees on shared inputs; check scores, names and signatures.

Usage: python tools/check_cross_language.py --r-source ../RFindsc --report parity.json
R needs Matrix and jsonlite. RFINDSC_RSCRIPT can select a particular Rscript.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import scipy.io
import scipy.sparse as sp


def make_cases(py_source):
    rng = np.random.default_rng(20261020)
    genes = [f"g{i}" for i in range(160)]
    counts = rng.poisson(2, (160, 64))
    counts[:20, :32] += rng.poisson(4, (20, 32))
    panels = {
        "ranked": {"up": genes[:20], "down": genes[20:40]},
        "eligible": {"up": genes[10:40], "down": genes[40:60],
                     "eligible": genes[:90]},
        "set": {"up": genes[:60], "ranked": False},
        "cut_first": {"up": ["outside"] * 10 + genes[:20], "eligible": genes[:100]},
    }
    cases = []

    def add(name, x=counts, g=genes, **kwargs):
        config = {"n_cores": 1, "top_n": 10, "n_cell": 30, "nonzero_rate": 0.01}
        config.update(kwargs.pop("config", {}))
        case = {"name": name, "counts": x.tolist(), "genes": list(g),
                "cells": [f"c{i}" for i in range(x.shape[1])],
                "config": config, "panels": panels, "api": "cells", "sparse": False}
        case.update(kwargs)
        cases.append(case)

    add("dense_ranked_sets_and_pair_universes")
    add("sparse_ranked_sets_and_pair_universes", sparse=True)
    add("grouped_baseline_with_singletons", sparse=True,
        baseline_groups=["A"] * 31 + ["B"] * 31 + ["single1", "single2"])
    add("restricted_counts_external_library", x=counts[:90], g=genes[:90],
        libsize=counts.sum(axis=0).tolist(), sparse=True)
    add("gene_set_wrapper", api="genesets", gene_sets={"A": genes[:60], "B": genes[60:110]})
    add("panel_pair_wrapper", api="pairs", panels={"A": panels["ranked"], "B": panels["eligible"]})
    add("clipped_scores", config={"clip": 2, "n_cell": 20, "nonzero_rate": 0.5})
    add("zero_ranked_cap_sets_still_whole", config={"top_n": 0})
    add("zero_signature_cap", config={"n_cell": 0}, expect_zero=True)
    add("unlimited_caps", config={"top_n": "Inf", "n_cell": "Inf"})
    add("empty_scoring_universe", x=np.eye(4, dtype=int), g=genes[:4], api="genesets",
        gene_sets={"markers": genes[:2]}, config={"nonzero_rate": 1}, expect_zero=True)
    if os.name != "nt":
        add("two_cores", config={"n_cores": 2})
    for sparse in [False, True]:
        x = np.repeat(np.r_[np.full(20, 5), np.full(80, 3)][:, None], 10, axis=1)
        add(f"identical_cells_{'sparse' if sparse else 'dense'}", x=x, g=genes[:100],
            api="genesets", gene_sets={"markers": genes[:20]}, sparse=sparse, expect_zero=True)
    x = np.arange(1, 101)[:, None] * np.array([1, 2, 3, 7, 10])
    add("proportional_cells", x=x, g=genes[:100], api="genesets",
        gene_sets={"markers": genes[:20]}, expect_zero=True)
    add("all_singleton_baselines", baseline_groups=list(range(64)), expect_zero=True)
    add("duplicate_reference_warning", panels={"A": {"up": ["g0", " G0 ", "g1"]}},
        expect_warnings=1)
    data = py_source / "tests" / "data"
    with gzip.open(data / "toy_counts.mtx.gz", "rb") as fh:
        toy = scipy.io.mmread(fh).toarray()
    toy_genes = (data / "toy_genes.txt").read_text().splitlines()
    # Source data, not precomputed expected scores; each implementation loads
    # its own native reference bundle. Both default and grouped baselines run.
    for bundle in ["hammond", "dedonno", "both"]:
        add(f"bundled_{bundle}", x=toy, g=toy_genes, bundle=bundle, sparse=True,
            config={"top_n": 500, "n_cell": 500})
    add("bundled_both_grouped", x=toy, g=toy_genes, bundle="both", sparse=True,
        baseline_groups=["A"] * 80 + ["B"] * 80 + ["C"] * 80,
        config={"top_n": 100, "n_cell": 60})
    return cases


def python_result(case, rf):
    x = np.asarray(case["counts"], dtype=float)
    if case["sparse"]:
        x = sp.csc_matrix(x)
    config = {k: float("inf") if v == "Inf" else v for k, v in case["config"].items()}
    kwargs = {"genes": case["genes"], "cells": case["cells"], "return_signatures": True,
              "config": rf.rfindsc_config(**config)}
    for key in ("libsize", "baseline_groups"):
        if key in case:
            kwargs[key] = case[key]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if "bundle" in case:
            bs = rf.default_biosets() if case["bundle"] == "hammond" else rf.load_dedonno_biosets()
            if case["bundle"] == "both":
                bs = {**rf.default_biosets(), **{k: v for k, v in bs.items() if k != "meta"}}
            out = rf.score_cells(x, bs, **kwargs)
        elif case["api"] == "genesets":
            out = rf.score_genesets(x, case["gene_sets"], **kwargs)
        elif case["api"] == "pairs":
            out = rf.score_panel_pairs(x, case["panels"], **kwargs)
        else:
            out = rf.score_cells(x, case["panels"], **kwargs)
    return out, len(seen)


def check(py_source, r_source, report=None):
    sys.path.insert(0, str(py_source / "src"))
    import rfindsc as rf

    if Path(rf.__file__).resolve().parent != (py_source / "src" / "rfindsc").resolve():
        raise RuntimeError("The requested Python source tree was not imported.")
    cases = make_cases(py_source)
    kernels = json.loads((py_source / "tests/data/kernel_cases.json").read_text())
    with tempfile.TemporaryDirectory(prefix="rfindsc-parity-") as temp:
        request, response = Path(temp) / "request.json", Path(temp) / "response.json"
        request.write_text(json.dumps({"cases": cases, "kernels": kernels}, allow_nan=False))
        rscript = os.environ.get("RFINDSC_RSCRIPT") or shutil.which("Rscript")
        if not rscript:
            raise RuntimeError("Rscript is required for the live equivalence test.")
        subprocess.run([rscript, "--vanilla", str(py_source / "tools/export_r_results.R"),
                        str(r_source), str(request), str(response)], check=True)
        r = json.loads(response.read_text())
    metrics = []
    for case, ref in zip(cases, r["cases"]):
        assert case["name"] == ref["name"]
        out, warning_count = python_result(case, rf)
        scores = out["scores"].to_numpy()
        expected = np.asarray(ref["scores"], dtype=float).reshape(scores.shape)
        assert np.isfinite(scores).all() and np.isfinite(expected).all()
        error = float(np.max(np.abs(scores - expected)))
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-8)
        assert out["universe"] == ref["universe"]
        assert out["universe_size"].to_dict() == ref["universe_size"]
        assert out["scores"].columns.tolist() == ref["panels"]
        assert out["scores"].index.tolist() == ref["cells"]
        for side in ("up_idx", "down_idx"):
            assert [a.tolist() for a in out["signatures"][side]] == ref[side]
        assert warning_count == ref["warning_count"]
        if case.get("expect_zero"):
            assert (scores == 0).all() and (expected == 0).all()
            assert all(a.size == 0 for side in ("up_idx", "down_idx") for a in out["signatures"][side])
        if "expect_warnings" in case:
            assert warning_count == case["expect_warnings"]
        metrics.append({"name": case["name"], "cells": scores.shape[0], "panels": scores.shape[1],
                        "max_abs_diff": error, "signatures_equal": True, "warnings": warning_count})
        print(f"PASS {case['name']}: max |Python - R| = {error:.3g}", flush=True)
    assert len(r["cases"]) == len(cases)
    kernel_diffs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case, ref in zip(kernels, r["kernels"]):
            names = lambda x: [f"g{int(g)}" for g in np.atleast_1d(x)]
            value = rf.running_fisher(names(case["g1"]), names(case["g2"]), case["P"], case["clip"],
                                      g2_ranked=case.get("ranked", True))
            kernel_diffs.append(abs(value - ref))
    assert len(r["kernels"]) == len(kernels) and max(kernel_diffs) < 1e-8
    result = {"passed": True, "python_version": sys.version.split()[0], "r_version": r["r_version"],
              "cases": metrics, "kernel_cases": len(kernels), "kernel_max_abs_diff": max(kernel_diffs),
              "score_max_abs_diff": max(c["max_abs_diff"] for c in metrics),
              "score_values_compared": sum(c["cells"] * c["panels"] for c in metrics),
              "signature_lists_compared": 2 * sum(c["cells"] for c in metrics),
              "source_sha256": {"python": {}, "R": {}}}
    for language, root, pattern in [("python", py_source, "src/rfindsc/*.py"), ("R", r_source, "R/*.R")]:
        result["source_sha256"][language] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob(pattern))}
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "source_sha256")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--r-source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check(args.python_source.resolve(), args.r_source.resolve(), args.report)
