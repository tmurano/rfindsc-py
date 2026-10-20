"""Panel tests: bundled references and DEG tables."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rfindsc
from rfindsc import build_panel_from_deg, default_biosets, load_dedonno_biosets

DATA = Path(__file__).parent / "data"


def test_default_biosets_are_the_four_hammond_axes():
    bs = default_biosets()
    assert set(bs) - {"meta"} == {"DAM", "Maturity", "Interferon", "G2M"}
    assert bs["meta"]["axes"] == ["DAM", "Maturity", "Interferon", "G2M"]
    assert list(bs["DAM"]["up"].columns) == ["gene", "diff", "rank"]
    assert len(bs["DAM"]["up"]) == 500
    assert len(bs["Interferon"]["down"]) == 0
    # gene groups have no order and are used whole (0.3.4)
    assert {k: p["ranked"] for k, p in bs.items() if k != "meta"} == {
        "DAM": True, "Maturity": True, "Interferon": False, "G2M": False}


def test_dedonno_biosets_are_the_four_axes():
    dd = load_dedonno_biosets()
    assert set(dd) - {"meta"} == {"Glucocorticoid", "Seizure", "UPR", "NF-kB"}
    assert len(dd["Seizure"]["down"]) == 0
    assert {k: p["label"] for k, p in dd.items() if k != "meta"} == {
        "Glucocorticoid": "Glucocorticoid", "Seizure": "Seizure", "UPR": "UPR", "NF-kB": "NF-kB"}
    assert {k: p["ranked"] for k, p in dd.items() if k != "meta"} == {
        "Glucocorticoid": True, "Seizure": True, "UPR": False, "NF-kB": False}


def _md5(genes):
    """md5 of the file R's writeLines(genes) produces (empty for no genes)."""
    genes = list(genes)
    text = "".join(g + "\n" for g in genes)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def test_bundled_gene_lists_match_r():
    summ = json.loads((DATA / "bundled_summary.json").read_text())
    loaders = {"biosets": default_biosets, "biosets_dedonno": load_dedonno_biosets}
    for name, load in loaders.items():
        b = load()
        assert [k for k in b if k != "meta"] == list(summ[name])
        for pname, s in summ[name].items():
            assert len(b[pname]["up"]) == s["n_up"] and len(b[pname]["down"]) == s["n_down"]
            assert _md5(b[pname]["up"]["gene"]) == s["md5_up"]
            assert _md5(b[pname]["down"]["gene"]) == s["md5_down"]


def test_build_panel_from_deg_splits_and_caps():
    rng = np.random.default_rng(0)
    deg = pd.DataFrame({"gene": [f"g{i}" for i in range(1, 31)],
                        "log2FC": np.concatenate([rng.uniform(0.1, 3, 15), -rng.uniform(0.1, 3, 15)])})
    p = build_panel_from_deg(deg, top_n=10, label="test")
    assert len(p["up"]) == 10 and len(p["down"]) == 10
    assert (p["up"]["diff"] > 0).all() and (p["down"]["diff"] < 0).all()
    assert p["type"] == "deg"


def test_build_panel_from_deg_matches_r():
    deg = pd.read_csv(DATA / "deg_table.tsv", sep="\t", keep_default_na=False, na_values=["NA"],
                      dtype={"gene": str, "log2FC": str})
    ref = json.loads((DATA / "deg_panels.json").read_text())
    got = {"filtered": build_panel_from_deg(deg, top_n=5, label="a", padj_col="padj", padj_thresh=0.05),
           "all": build_panel_from_deg(deg)}
    for key in ("filtered", "all"):
        for side in ("up", "down"):
            r, p = ref[key][side], got[key][side]
            assert p["gene"].tolist() == r["gene"]
            np.testing.assert_array_equal(p["diff"].to_numpy(), np.array(r["diff"], dtype=float))
            assert p["rank"].tolist() == r["rank"]
        assert got[key]["eligible"] == ref[key]["eligible"]      # genes the study measured (0.3.5)


def test_version():
    assert rfindsc.__version__ == "1.0.0"
