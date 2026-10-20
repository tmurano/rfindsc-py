"""Optional live integration test against the matching R source checkout."""
import importlib.util
import os
from pathlib import Path
import shutil

import pytest


def test_live_r_python_equivalence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    r_source = Path(os.environ.get("RFINDSC_R_SOURCE", root.parent / "RFindsc"))
    if not (r_source / "R/scoring.R").is_file():
        pytest.skip("set RFINDSC_R_SOURCE to the companion R source checkout")
    if not (os.environ.get("RFINDSC_RSCRIPT") or shutil.which("Rscript")):
        pytest.skip("Rscript with Matrix and jsonlite is required")
    spec = importlib.util.spec_from_file_location("cross_language", root / "tools/check_cross_language.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.check(root, r_source.resolve(), tmp_path / "parity.json")
    assert result["passed"] and result["kernel_cases"] == 384
