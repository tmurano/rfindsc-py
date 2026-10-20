"""RFind-sc: direction-aware panel scoring for single-cell RNA-seq.

Python implementation of the R package RFindsc 1.0.1 (same algorithm, same
bundled references, numerically identical scores).
"""
from .annotate import add_rfind_scores
from .config import rfindsc_config
from .panels import build_panel_from_deg, default_biosets, load_dedonno_biosets
from .running_fisher import bid_running_fisher, rfind_score_cell, running_fisher, to_b2
from .scoring import (extract_counts, run_pipeline, score_cells, score_genesets,
                      score_panel_pairs)
from .signature import build_cell_signature

__version__ = "1.0.1"

__all__ = [
    "add_rfind_scores",
    "bid_running_fisher",
    "build_cell_signature",
    "build_panel_from_deg",
    "default_biosets",
    "extract_counts",
    "load_dedonno_biosets",
    "rfind_score_cell",
    "rfindsc_config",
    "run_pipeline",
    "running_fisher",
    "score_cells",
    "score_genesets",
    "score_panel_pairs",
    "to_b2",
]
