"""add_rfind_scores(): scores into AnnData.obs (port of tests/testthat/test-add-scores.R)."""
import numpy as np
import pandas as pd
import pytest

from rfindsc import add_rfind_scores, score_genesets


def _scores():
    return pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [-1.0, -2.0, -3.0]}, index=["c1", "c2", "c3"])


def test_adds_one_column_per_panel_matched_by_cell_name():
    obs = pd.DataFrame({"x": [1, 2, 3, 4]}, index=["c3", "c1", "c2", "c4"])
    with pytest.warns(UserWarning, match="1 cell"):
        out = add_rfind_scores(obs, {"scores": _scores()})
    assert out is obs                                        # changed in place
    np.testing.assert_array_equal(obs["RFind_A"], [3.0, 1.0, 2.0, np.nan])
    np.testing.assert_array_equal(obs["RFind_B"], [-3.0, -1.0, -2.0, np.nan])
    assert list(obs["x"]) == [1, 2, 3, 4]                    # other columns kept

    with pytest.warns(UserWarning):
        add_rfind_scores(obs, _scores() * 10)                 # overwrite (default)
    np.testing.assert_array_equal(obs["RFind_A"], [30.0, 10.0, 20.0, np.nan])
    with pytest.raises(ValueError, match="already present"):
        add_rfind_scores(obs, _scores(), overwrite=False)
    with pytest.warns(UserWarning):
        add_rfind_scores(obs, _scores(), prefix="idx_")
    assert {"idx_A", "idx_B"} <= set(obs.columns)


def test_rejects_scores_that_do_not_name_the_cells():
    obs = pd.DataFrame({"x": [1, 2]}, index=["c1", "c2"])
    with pytest.raises(ValueError, match="No cell"):
        add_rfind_scores(obs, pd.DataFrame({"A": [1.0]}, index=["z"]))
    with pytest.raises(TypeError, match="AnnData"):
        add_rfind_scores([1, 2], _scores())


def test_writes_the_scores_of_an_anndata_object_into_obs():
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(42)
    cells = [f"c{i}" for i in range(1, 41)]
    genes = [f"g{i}" for i in range(1, 201)]
    ad = anndata.AnnData(X=rng.poisson(2, size=(40, 200)).astype(float),
                         obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    out = score_genesets(ad, {"setA": genes[:20], "setB": genes[20:40]})
    assert add_rfind_scores(ad, out) is ad
    for panel in ("setA", "setB"):
        np.testing.assert_allclose(ad.obs[f"RFind_{panel}"].to_numpy(),
                                   out["scores"].loc[ad.obs_names, panel].to_numpy())
