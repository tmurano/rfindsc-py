# Changelog

Versions follow the R package [RFindsc](https://github.com/tmurano/RFindsc): a
given version of `rfindsc` (Python) gives the same scores as the same version
of RFindsc (R).

## 1.0.0 (2026-10-20)

First stable release, the version of the RFind-sc paper. Matches RFindsc 1.0.0; the code and the scores are those
of 0.3.9 (only the version number changed).

## 0.3.9 (2026-10-19)

Matches RFindsc 0.3.9; scores are unchanged.

* `running_fisher()` warns (`DuplicateGeneNamesWarning`) when a list names a gene more than once, which counts the gene
  twice and inflates the overlap; `score_cells()` warns once when names of the scoring universe differ only in case
  (e.g. PISD and Pisd), since names are matched without regard to case.
* The faster multi-core scoring of RFindsc 0.3.9 concerns the R package only (this package already scaled with cores).

## 0.3.8 (2026-10-16)

Matches RFindsc 0.3.8.

* The bundled references (`default_biosets()`: DAM, Maturity, Interferon, G2M; `load_dedonno_biosets()`: Glucocorticoid,
  Seizure, UPR, NF-kB) are rebuilt from the final references of the RFind-sc paper (2026-10-16): counts per million over the
  nuclear genes, genes without informative names, mitochondrial and haemoglobin genes left out and ribosomal protein genes
  kept, the top 500 genes per direction of each published list taken before the genes of a dataset, and the top 200
  up-regulated genes for Seizure without an expression floor. Scores against DAM, Maturity, Glucocorticoid and Seizure
  change accordingly; the gene sets are unchanged.
* `score_cells()` and the functions built on it (`run_pipeline()`, `score_panel_pairs()`): A ranked panel now keeps its top `top_n` genes per side before its genes are restricted to the pair universe, as in the
  paper; before, the lists were restricted first and then cut, so lower-ranked genes of a reference took the place of top
  genes that the dataset did not measure. Gene sets are used whole (before, they were also cut to `top_n` after the
  restriction, which mattered only for sets of more than 500 genes). A panel whose top `top_n` genes per side are all in the
  pair universe gives the same scores as before.

## 0.3.7 (2026-10-07)

Matches RFindsc 0.3.7. First public release.

* New `add_rfind_scores()`: writes the cells x panels scores of `run_pipeline()`, `score_panel_pairs()`,
  `score_genesets()` or `score_cells()` into `adata.obs` (or a DataFrame of cells), one column per panel
  (`RFind_<panel>`), matched by cell name, so that they can be plotted like any other annotation (for example
  `scanpy.pl.umap(adata, color="RFind_DAM")`). The counterpart of the R function of the same name. Scores are not
  changed.
* Removed `read_registry()`, `read_reference()` and `REGISTRY_URL`, which read the rfind-panels registry (no longer
  maintained), and their tests. Scores of the other functions are unchanged.
* The bundled `biosets.json` was not re-exported: its `meta.source` still names the earlier location of the Hammond
  file (`Reference/output/references_panel9.rds`), now `data/Hammond2019/output/references/references_panel9.rds` in
  the RFind-sc paper repository (https://github.com/tmurano/RFind-sc).

## 0.3.6 (2026-09-29)

Matches RFindsc 0.3.6.

* Removed the bundled DAM2 (Sala Frigerio 2019) and Cholesterol (MSigDB Hallmark) references, which the RFind-sc
  paper does not use. `default_biosets()` now returns the 4 Hammond axes (`DAM`, `Maturity`, `Interferon`, `G2M`);
  their gene lists and scores, and those of `load_dedonno_biosets()`, are unchanged.
* Tests: the columns and entries of the two references were dropped from the R fixtures in `tests/data/` (made with
  RFindsc 0.3.5, R 4.6.1; see `fixtures_meta.json`); all other values are unchanged, and `make_fixtures.R` with
  RFindsc 0.3.6 reproduces them. It now reads the inputs that had been drawn from the two references (the genes of the
  simulated dataset and the `pair_chr` lists) from the fixtures.

## 0.3.5 (2026-09-18)

Matches RFindsc 0.3.5.

* **Pair universe (scoring change for ranked references).** The background of each comparison is
  the scoring universe intersected with the panel's new optional `eligible` genes (the genes the
  reference's source study reported with a finite, non-zero change), as in the RFind-sc paper. Both
  the panel's lists and each cell's UP / DOWN lists are restricted to it before the top `top_n` /
  `n_cell` genes are taken. A panel without `eligible` (a gene set) uses the whole scoring universe.
* The universe size counts distinct lower-case symbols (`PISD` and `Pisd` count once).
* `build_panel_from_deg()` records `eligible` (genes with a finite, non-zero log2 fold change, before
  `padj_thresh` and `top_n`); `score_panel_pairs()` passes it through; `to_b2()` keeps it.
* `score_cells()` returns `universe_size` (a Series, per panel).
* Bundled references: the five ranked references carry `eligible` (`data-raw/export_bundled.R`).
* Internals: `_running_fisher_many()` accepts one universe size per pair; `_signature_from_resid()`
  takes an optional mask (the pair universe).
* Tests: the R fixtures were regenerated with RFindsc 0.3.5 (`tests/data/make_fixtures.R`); two new
  tests for the pair universe. Python and R agree to 4e-13.

## 0.3.4 (2026-09-17)

Matches RFindsc 0.3.4.

* **Gene sets are used whole (scoring fix).** A membership gene set (an MSigDB Hallmark set such
  as the bundled `Interferon`, `G2M`, `Cholesterol`, `UPR` and `NF-kB` groups) has no order, but
  until 0.3.3 it was scanned as if it were ranked, in the order stored (alphabetical for the bundled
  groups), so the score depended on the alphabetical order of gene names. As for a biogroup in the
  original Running Fisher, a set is now used whole: K is fixed at the number of its distinct genes
  and only the top-G cut-offs of the cell's ranked list are scanned. Scores of gene sets change;
  scores of ranked references (signed lists, and UP-only lists ranked by fold change such as
  `Seizure`) are unchanged (bitwise).
* `running_fisher()` / `bid_running_fisher()` gain `g2_ranked` (default `True`).
  `rfind_score_cell()` reads `b2["ranked"]`; `to_b2()` sets it from a panel's `ranked` field, or,
  without one, from its `type` (`"gene_group"`, `"msigdb_type_b"`, `"msigdb_cgp_paired"` are sets).
* `score_genesets()` marks every set `ranked=False`; `score_panel_pairs()` passes a panel's
  `ranked` field through (default ranked). `read_reference()` returns `ranked=False` for a registry
  reference whose `reference.yaml` has `score_method: binary` (a curated +1 / -1 list without order;
  it now also reads `reference.yaml`).
* Bundled references: every panel has `ranked` (and `meta["ranked"]`); the type of `Seizure` is now
  `"ranked_up"` instead of `"gene_group"`. Gene lists unchanged.
* The R-equivalence fixtures were regenerated with RFindsc 0.3.4: 132 kernel cases with an unordered
  set were added (384 in total) and `panel_pairs.json` has a `ranked=False` panel. The simulated
  dataset and the fixtures of ranked panels are unchanged.

## 0.3.3 (2026-09-16)

Matches RFindsc 0.3.3.

* **Bundled references regenerated from the final references of the RFind-sc paper** (the pipelines
  were re-run on 2026-09-16 with a gene-name fix). The Hammond gene universe now holds 8,960 genes:
  90 names that had lost their dash in an earlier import are restored (`H2.D1` -> `H2-D1`), so the UP
  lists of `DAM`, `DAM2` and `Maturity` now include MHC genes such as `H2-D1`, `H2-K1` and `H2-Q7`
  (500 genes per side as before; the DOWN lists and the `Interferon`, `G2M` and `Cholesterol` groups
  are unchanged). The De Donno axes (`Glucocorticoid`, `Seizure`, `UPR`, `NF-kB`) are unchanged.
* `baseline_groups` is no longer Python-only: RFindsc 0.3.3 has the same argument with the same
  semantics, so the per-group baseline fixtures are now produced by the R package's own
  `score_cells(baseline_groups = )` instead of a stand-in written out in `make_fixtures.R`.
* The R-equivalence fixtures were regenerated with RFindsc 0.3.3 (R 4.5.1).

## 0.3.2 (2026-09-15)

Matches RFindsc 0.3.2.

* **Bundled MSigDB Hallmark set removed**, as in RFindsc 0.3.2:
  `load_hallmark_biosets()` and `src/rfindsc/data/biosets_msigdb_hallmark.json`
  (50 Hallmark sets, shipped in 0.3.1) are gone. Gene sets of your own can
  still be scored with `score_cells()`, `score_genesets()` and
  `score_panel_pairs()`.
* The `label` fields of `load_dedonno_biosets()` equal the axis names
  (`Glucocorticoid`, `Seizure`), now also in the R source data, so
  `data-raw/export_bundled.R` no longer overrides them. The bundled JSON files
  are unchanged.
* The R-equivalence fixtures were regenerated with RFindsc 0.3.2 (the toy
  dataset no longer draws genes from the Hallmark sets).

## 0.3.1 (2026-09-15)

First release of the Python package, matching RFindsc 0.3.1.

* **De Donno axes under the paper's index names**, as in RFindsc 0.3.1:
  `load_dedonno_biosets()` returns `Glucocorticoid`, `Seizure`, `UPR` and
  `NF-kB` (RFindsc 0.3.0 used `CORT` and `Hyperexcitability` for the first
  two). The gene lists are unchanged.
* **Running Fisher kernel, G x K form** (the RFind-sc paper definition):
  `-log10` of the smallest hypergeometric upper-tail probability over every
  pair of cut-offs, top `G` of one list x top `K` of the other, evaluated at
  the ranks of shared genes, on the log scale, clipped at `clip` (300).
* **Per-cell Delta = P2 Pearson residual** `(x - e) / sqrt(e + 1)`,
  `e = pi_g * N_cell`; UP / DOWN = the 500 most positive / most negative
  residuals.
* **Score** `S = RF(UP, up) + RF(DOWN, down) - RF(UP, down) - RF(DOWN, up)`;
  for a gene set without a DOWN side, `RF(UP, set) - RF(DOWN, set)`.
* Bundled references identical to RFindsc 0.3.1: `default_biosets()`
  (6 Hammond axes), `load_dedonno_biosets()` (4 De Donno axes).
* Input: genes x cells or cells x genes matrices (dense or sparse), pandas
  data frames, AnnData (`pip install "rfindsc[anndata]"`).
* Python-only additions: `baseline_groups` in `score_cells()` (per-group
  baseline, e.g. per cell type as in the paper's De Donno analysis) and
  `read_registry()` / `read_reference()` for the rfind-panels registry
  (removed in 0.3.7).
* Not ported: `msigdbr_to_panels()` and `c2_cgp_auto_pair()` (they depend on
  the R package msigdbr).
* Numerical equivalence with RFindsc 0.3.1 is tested on fixtures generated in
  R (`tests/data/make_fixtures.R`).

An earlier, unreleased Python port (0.1.0) implemented the pre-0.3.0 kernel
(average of two one-sided scans) and a log2-difference Delta; it is superseded
by this release.
