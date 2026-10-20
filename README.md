# rfindsc

**Direction-aware panel scoring for single-cell RNA-seq (RFind-sc), in Python.**

> RFind-sc has three repositories: the R package [RFindsc](https://github.com/tmurano/RFindsc) (for Seurat objects), this Python package (**rfindsc**, for AnnData; the same scores) and the [analysis code of the RFind-sc paper](https://github.com/tmurano/RFind-sc).

`rfindsc` is the Python implementation of the R package
[RFindsc](https://github.com/tmurano/RFindsc). Version 1.0.1 matches
**RFindsc 1.0.1**: the same algorithm, the same bundled references and the same
scores (the test suite checks agreement with R to 1e-8; the observed differences
are below 1e-12; see [Numerical equivalence](#numerical-equivalence)).

## What it computes

For every cell and every reference panel (a case-vs-control pair of ranked gene
lists, UP and DOWN, or a single gene set):

1. **Per-cell signature (Delta = P2 residual).** For gene *g* in cell *j*,
   `Delta = (x_gj - e_gj) / sqrt(e_gj + 1)` with `e_gj = pi_g * N_j`: `N_j` is
   the cell's total UMI and `pi_g` the mean of `x_gj / N_j` over the baseline
   cells (all cells by default). The cell's UP list holds the genes with a
   positive Delta, DOWN those with a negative Delta, most extreme first, up to
   500 each.
2. **Running Fisher, G x K.** Two ranked lists are compared through the
   smallest hypergeometric upper-tail probability over every pair of cut-offs
   (top *G* genes of one list x top *K* of the other); the kernel returns
   `-log10` of that probability (log scale throughout, capped at 300). Only
   cut-offs at ranks of shared genes are evaluated, which gives the same
   minimum as the full scan. A membership gene set (e.g. an MSigDB Hallmark
   set, `ranked=False`) has no order: it is used whole (K = its size) and only
   the cut-offs of the cell's list are scanned.
3. **RFind-sc score.**
   `S = RF(UP, up) + RF(DOWN, down) - RF(UP, down) - RF(DOWN, up)`
   (cell lists in capitals, panel lists in lower case): positive = case-like,
   negative = control-like. For a gene set without a DOWN list,
   `S = RF(UP, set) - RF(DOWN, set)`.

The scoring universe is the genes detected in at least 1% of cells. Each panel
is scored in its pair universe: the scoring universe intersected with
the panel's `eligible` genes (the genes its source study reported, set by
`build_panel_from_deg()` and carried by the bundled ranked references), or the
whole scoring universe for a panel without `eligible` (a gene set). Each cell's
UP / DOWN lists are restricted to it before the top 500 genes per side are
taken; a ranked panel keeps its top 500 genes per side and then those in the
pair universe, and a gene set is used whole.

## Installation

```bash
pip install git+https://github.com/tmurano/rfindsc-py@v1.0.1
# optional AnnData support
pip install "rfindsc[anndata] @ git+https://github.com/tmurano/rfindsc-py@v1.0.1"
```

Requires Python >= 3.9, numpy, scipy and pandas.

## Quick start

```python
import numpy as np
import rfindsc

panels = rfindsc.default_biosets()                  # the 4 Hammond axes of the paper
genes = sorted({g for name, p in panels.items() if name != "meta"
                for side in ("up", "down") for g in p[side]["gene"]})
rng = np.random.default_rng(0)
counts = rng.poisson(0.5, size=(len(genes), 200))   # raw UMI counts, genes x cells

res = rfindsc.score_cells(counts, panels, genes=genes)
res["scores"].head()                                # cells x panels (DataFrame)
```

With an AnnData object (cells x genes; raw counts in a layer or in `X`):

```python
import anndata as ad

adata = ad.read_h5ad("microglia.h5ad")
res = rfindsc.run_pipeline(adata, layer="counts")    # default: the 4 Hammond axes
rfindsc.add_rfind_scores(adata, res)                 # adata.obs: RFind_DAM, RFind_Maturity, ...

import scanpy as sc                                  # for example, plot them on the UMAP
sc.pl.umap(adata, color=["RFind_DAM", "RFind_Maturity"])
```

`add_rfind_scores()` writes the scores into `adata.obs`, one column per panel, matched by cell name (the counterpart
of the R function of the same name, which writes into the metadata of a Seurat object).

### Input

* `counts`: raw UMI counts as a numpy array, a scipy sparse matrix, a pandas
  data frame or an AnnData object. Matrices and data frames are genes x cells
  by default (as in R); pass `layout="cells_x_genes"` otherwise. Plain
  matrices need `genes=` (and optionally `cells=`).
* The library size is the total count per cell. If `counts` is already
  restricted to a gene universe, pass `libsize=` with the totals over all genes
  (as in the RFind-sc paper, which also fixes the universe beforehand: use
  `rfindsc_config(nonzero_rate=0)`).
* `baseline_groups=`: one label per cell, e.g. the cell type. Each cell is then
  compared with the mean of its own group instead of all cells, as the paper
  does for the De Donno et al. dataset (same argument in the R package). The scoring universe is still chosen over all cells.

### Panels

```python
rfindsc.default_biosets()        # Hammond: DAM, Maturity, Interferon, G2M
rfindsc.load_dedonno_biosets()   # De Donno: Glucocorticoid, Seizure, UPR, NF-kB

# your own DEG table (columns gene, log2FC, padj)
import pandas as pd
deg = pd.read_csv("my_deg.csv")
panel = rfindsc.build_panel_from_deg(deg, padj_col="padj", padj_thresh=0.05, label="my_condition")
res = rfindsc.score_cells(adata, {"my_condition": panel}, layer="counts")

# plain gene lists: unordered sets (used whole) or ranked (UP, DOWN) lists
rfindsc.score_genesets(adata, {"markers": ["Tmem119", "P2ry12", "Cx3cr1"]}, layer="counts")
rfindsc.score_panel_pairs(adata, {"contrast": {"up": up_genes, "down": down_genes}}, layer="counts")
# a panel of unordered sets: add "ranked": False
```

Gene symbols are matched case-insensitively, as in R.

A panel is either ranked (most extreme gene first) or an unordered membership
set with `"ranked": False` (the bundled gene groups `Interferon`, `G2M`, `UPR`
and `NF-kB`, and every set given to `score_genesets()`). A set is used whole,
so the order of its genes does not matter.

### Output

`score_cells()` and the wrappers return a dict: `scores` (cells x panels
DataFrame), `universe` (the scoring genes), `universe_size` (the pair universe
size per panel) and, with
`return_signatures=True`, `signatures` (per-cell `up_idx` / `down_idx`,
0-based indices into `universe`).

### Configuration

`rfindsc_config()` holds the defaults of the paper:

| parameter | default | meaning |
|---|---|---|
| `top_n` | 500 | genes per side of a ranked panel, taken before the restriction to the pair universe |
| `n_cell` | 500 | genes per side of each cell's signature |
| `nonzero_rate` | 0.01 | a gene enters the universe if non-zero in >= 1% of cells |
| `clip` | 300 | cap on the score magnitude |
| `n_cores` | 1 | worker processes (> 1 forks workers; POSIX only) |

## API (R -> Python)

| RFindsc (R) | rfindsc (Python) |
|---|---|
| `run_pipeline()` | `run_pipeline()` (AnnData instead of Seurat) |
| `score_cells()` | `score_cells()` |
| `score_genesets()`, `score_panel_pairs()` | same names |
| `running_fisher()`, `bid_running_fisher()` | same names |
| `build_cell_signature()`, `rfind_score_cell()`, `to_b2()` (internal in R) | same names |
| `build_panel_from_deg()` | `build_panel_from_deg()` |
| `default_biosets()`, `load_dedonno_biosets()` | same names |
| `rfindsc_config()` | `rfindsc_config()` |
| `add_rfind_scores()` (Seurat metadata) | `add_rfind_scores()` (`AnnData.obs`) |
| `msigdbr_to_panels()`, `c2_cgp_auto_pair()` | not ported (they need the R package msigdbr) |

Differences from the R package:

* Signature indices are 0-based (1-based in R).
* `n_cores` defaults to 1 (R: number of cores minus one; 1 on Windows). The
  results do not depend on it.
* Python-only input options: `genes=`, `cells=`, `layout=`, `layer=` (AnnData,
  data frames, plain matrices). `libsize=` and `baseline_groups=` exist in both.

## Numerical zeroes and input checks

A count difference (observed minus expected) is zero when it is within the
floating-point error of the baseline,
`8 * eps * max(8, N) * max(1, abs(observed), abs(expected))`, where `eps` is the
float64 machine epsilon and `N` is the number of cells in the cell's baseline
group (`N = 1` in `build_cell_signature()`). A cell identical to its baseline,
such as the only cell of a baseline group, therefore gets empty UP/DOWN lists
and a score of 0. The R package uses the same rule.

Counts must be finite and non-negative, library sizes finite and positive, and
gene names present and non-empty. A gene named twice in a prepared reference
list gives one warning before scoring; the list is scored as given, so remove
the duplicate from the reference.

## Numerical equivalence

`tests/test_r_equivalence.py` compares the package with reference outputs of
RFindsc, generated by `tests/data/make_fixtures.R`: 384 random and edge-case
list pairs for the kernel (132 of them with an unordered gene set), and a
simulated dataset (240 cells x 1,156 mouse genes) scored with the bundled
Hammond and De Donno panels, gene sets, panel pairs, a restricted matrix with
`libsize`, other `rfindsc_config()` settings and a per-cell-type baseline.
Scores must agree to 1e-8 and signatures and universes exactly; the largest
observed difference is 4.3e-13 (kernel) and 5.7e-14 (cell scores).

The fixtures were made with RFindsc 0.3.8 (see `tests/data/fixtures_meta.json`)
by `tests/data/make_fixtures.R`.

The kernel of RFindsc 0.3.8 (unchanged since 0.3.5) is identical to `analysis/running_fisher.R` in the
paper repository [RFind-sc](https://github.com/tmurano/RFind-sc) (ranked lists
and gene sets): on the 384 kernel cases the largest difference is 0
(`engine_max_abs_diff` in `fixtures_meta.json`, computed with RFindsc 0.3.8).

To run the tests:

```bash
pip install -e ".[test]"
pytest -s            # prints the maximum differences
```

To run this package and RFindsc on the same inputs, check out
[RFindsc](https://github.com/tmurano/RFindsc) beside this repository and install
R with Matrix and jsonlite (`RFINDSC_RSCRIPT` selects the Rscript):

```bash
RFINDSC_R_SOURCE=../RFindsc pytest -q
python tools/check_cross_language.py --r-source ../RFindsc --report parity.json
```

This compares the scores (to 1e-8), the UP/DOWN lists, the universes and the
384 kernel cases; the report records the largest differences.

## Performance

2,000 Hammond et al. microglia x the 4 bundled panels (8,960-gene universe) on
an 8-core Apple M3 laptop: about 490 cells/s on one core and 2,400 cells/s with
`n_cores=8` (2026-09-29).

## Citation

Murano et al. (in preparation). Until the preprint is available, please cite
this repository (the paper used version 1.0.1, git tag `v1.0.1`) and the R package.

## License

MIT (c) 2026 Tomoyuki Murano (Miyakawa Lab). See [LICENSE](LICENSE).
