## Reference outputs of RFindsc 0.3.8 (R) for the Python equivalence tests (tests/test_r_equivalence.py).
## Run from the rfindsc-py folder:
##   RFINDSC_SRC=<RFindsc 0.3.8 source folder> [RFIND_ENGINE=<running_fisher.R>] Rscript tests/data/make_fixtures.R
## RFIND_ENGINE (optional): analysis/running_fisher.R of the RFind-sc paper repository
## (https://github.com/tmurano/RFind-sc), whose kernel is compared with running_fisher() on every kernel case.
## Reads the fixed inputs toy_genes.txt (the genes of the simulated dataset) and the `pair_chr` lists of panel_pairs.json.
## Writes into tests/data/:
##   kernel_cases.json          random / edge-case list pairs and running_fisher() values
##   toy_counts.mtx.gz, toy_cells.tsv   simulated counts over the genes of toy_genes.txt
##   scores_*.tsv               score_cells() / score_genesets() / score_panel_pairs() outputs (cells x panels)
##   signatures_*.json          per-cell UP/DOWN indices (1-based) and the scoring universe
##   universe_size_hammond.json, libsize_genes.txt, genesets.json, panel_pairs.json   pair universe sizes and inputs
##   deg_table.tsv, deg_panels.json   build_panel_from_deg() input / outputs
##   bundled_summary.json       sizes and md5 of the bundled gene lists
##   fixtures_meta.json         versions and the kernel check against RFIND_ENGINE (engine_max_abs_diff)
suppressPackageStartupMessages({ library(Matrix); library(jsonlite) })
src <- Sys.getenv("RFINDSC_SRC")
if (nzchar(src)) pkgload::load_all(src, quiet = TRUE, export_all = FALSE) else library(RFindsc)
stopifnot(packageVersion("RFindsc") >= "0.3.8")          # 0.3.8: bundled references and the order of the panel cut
ns <- asNamespace("RFindsc")
OUT <- "tests/data"
f17 <- function(x) sprintf("%.17g", x)
wjson <- function(x, file) write_json(x, file.path(OUT, file), auto_unbox = TRUE, digits = NA, na = "null")
set.seed(20260915)

## ---- (a) kernel: random and edge-case list pairs ----
cases <- list()
add <- function(g1, g2, P, clip = 300, tag = "random", ranked = TRUE) {
  s <- suppressWarnings(running_fisher(paste0("g", g1), paste0("g", g2), P, clip, g2_ranked = ranked))
  cc <- list(tag = tag, g1 = I(as.integer(g1)), g2 = I(as.integer(g2)), P = P, clip = clip, score = f17(s))
  if (!ranked) cc$ranked <- FALSE                     # g2 is an unordered gene set (RFindsc >= 0.3.4)
  cases[[length(cases) + 1L]] <<- cc
}
for (i in 1:240) {
  P <- sample(c(40, 300, 3000, 20000), 1)
  n1 <- sample.int(min(P, 500), 1); n2 <- sample.int(min(P, 500), 1)
  g1 <- sample.int(P, n1); k <- sample(0:min(n1, n2), 1)
  g2 <- switch(i %% 3 + 1,
               c(sample(g1, k), sample.int(P, n2)),    # shared genes anywhere in list 1
               c(g1[seq_len(k)], sample.int(P, n2)),   # shared genes at the top of both lists
               sample.int(P, n2))                      # chance overlap only
  g2 <- head(unique(g2), 500)
  add(g1, g2, P, clip = sample(c(300, 300, 300, 50, 5, 1000), 1))
}
add(1:200, 1:200, 20000, 1000, "identical, beyond 1e-308")
add(1:200, 1:200, 20000, 300, "identical, clipped")
add(5, 5, 9000, 300, "top gene shared (p = 1/P)")
add(c(7, 1:30), c(3, 7, 40:60), 9000, 300, "single shared gene")
add(c(1, 2, 2, 3, 1, 4), c(2, 2, 5, 1), 50, 300, "duplicated genes")
add(integer(0), 1:10, 100, 300, "empty g1")
add(1:10, integer(0), 100, 300, "empty g2")
add(1:20, 21:40, 100, 300, "no overlap")
add(1:60, 1:60, 50, 300, "lists longer than the universe")
add(1:40, sample.int(40, 10), 40, 300, "g1 = whole universe")
add(sample.int(40), sample.int(40), 40, 300, "both = whole universe, shuffled")
add(sample.int(300, 250), sample.int(300, 250), 300, 300, "large chance overlap")
## g2 as an unordered gene set (g2_ranked = FALSE, RFindsc >= 0.3.4). Drawn from a separate random stream, so the
## cases above and the toy dataset below are the same as with RFindsc 0.3.3.
rng_state <- .Random.seed
set.seed(20260917)
for (i in 1:120) {
  P <- sample(c(40, 300, 3000, 20000), 1)
  n1 <- sample.int(min(P, 500), 1); n2 <- sample.int(min(P, 300), 1)
  g1 <- sample.int(P, n1); k <- sample(0:min(n1, n2), 1)
  g2 <- switch(i %% 3 + 1, c(sample(g1, k), sample.int(P, n2)), c(g1[seq_len(k)], sample.int(P, n2)), sample.int(P, n2))
  g2 <- head(unique(g2), 300)
  add(g1, g2, P, clip = sample(c(300, 300, 300, 50, 5, 1000), 1), tag = "set, random", ranked = FALSE)
}
add(1:200, sample(1:200), 20000, 1000, "set, identical, beyond 1e-308", ranked = FALSE)
add(1:200, 1:200, 20000, 300, "set, identical, clipped", ranked = FALSE)
add(5, c(9, 5, 3), 9000, 300, "set, top gene shared", ranked = FALSE)
add(c(1, 2, 2, 3, 1, 4), c(2, 2, 5, 1), 50, 300, "set, duplicated genes", ranked = FALSE)
add(c(1, 1, 1), 1, 50, 300, "set, duplicated genes beyond the set size", ranked = FALSE)
add(integer(0), 1:10, 100, 300, "set, empty g1", ranked = FALSE)
add(1:10, integer(0), 100, 300, "set, empty g2", ranked = FALSE)
add(1:20, 21:40, 100, 300, "set, no overlap", ranked = FALSE)
add(1:30, 1:60, 50, 300, "set larger than the universe", ranked = FALSE)
add(1:60, 1:30, 50, 300, "set, list longer than the universe", ranked = FALSE)
add(sample.int(40), sample.int(40), 40, 300, "set = whole universe", ranked = FALSE)
add(sample.int(300, 250), sample.int(300, 250), 300, 300, "set, large chance overlap", ranked = FALSE)
.Random.seed <- rng_state
eng_diff <- NA_real_
eng <- Sys.getenv("RFIND_ENGINE")
if (nzchar(eng)) {
  E <- new.env(); sys.source(eng, envir = E)
  eng_diff <- max(vapply(cases, function(cc) abs(suppressWarnings(
    running_fisher(paste0("g", cc$g1), paste0("g", cc$g2), cc$P, cc$clip, g2_ranked = is.null(cc$ranked)) -
      E$running_fisher(paste0("g", cc$g1), paste0("g", cc$g2), cc$P, cc$clip, g2_ranked = is.null(cc$ranked)))), 0))
}
wjson(cases, "kernel_cases.json")

## ---- (b) toy dataset ----
hb <- default_biosets(); dd <- load_dedonno_biosets()
top <- function(df, n) head(as.character(df$gene), n)
## The genes are fixed (toy_genes.txt): mouse symbols drawn from the panels bundled with RFindsc 0.3.5, two of which
## were removed in 0.3.6, then a second row whose name differs only in case, 150 fillers and 30 rare genes. Drawing
## them used no random numbers. Since 0.3.8 the bundled lists also hold genes outside these genes (the top 500 of each
## published list are taken before the genes of a dataset), so program() skips the genes that the toy dataset lacks.
genes <- readLines(file.path(OUT, "toy_genes.txt"))
G <- length(genes)
groups <- rep(c("A", "B", "C"), c(100, 80, 60)); n <- length(groups)
rate <- rgamma(G, shape = 0.5, rate = 1) * matrix(exp(rnorm(G * 3, 0, 0.7)), G, 3, dimnames = list(NULL, c("A", "B", "C")))[, groups]
idx <- function(g) match(tolower(g), tolower(genes))
program <- function(rate, up, down, cells, f) {
  up <- idx(up); up <- up[!is.na(up)]; down <- idx(down); down <- down[!is.na(down)]   # genes of the toy dataset only
  rate[up, cells] <- rate[up, cells] * f; rate[down, cells] <- rate[down, cells] / f; rate }
rate <- program(rate, top(hb$DAM$up, 150), top(hb$DAM$down, 150), which(groups == "A")[1:50], 5)
rate <- program(rate, top(dd$Glucocorticoid$up, 80), top(dd$Glucocorticoid$down, 80), which(groups == "B")[1:40], 4)
rare <- grep("^Rare", genes); rate[rare, ] <- 0
rate <- sweep(rate, 2, colSums(rate), "/")
lib <- round(exp(rnorm(n, log(2500), 0.5)))
counts <- matrix(rnbinom(G * n, size = 2, mu = rate * rep(lib, each = G)), G, n)
for (k in seq_along(rare)) counts[rare[k], sample.int(n, (k - 1) %% 4)] <- 1   # detected in 0-3 cells
dimnames(counts) <- list(genes, sprintf("cell%03d", seq_len(n)))
counts <- as(Matrix(counts, sparse = TRUE), "CsparseMatrix")
tm <- as(counts, "TsparseMatrix"); con <- gzfile(file.path(OUT, "toy_counts.mtx.gz"), "w")
writeLines(c("%%MatrixMarket matrix coordinate integer general", sprintf("%d %d %d", G, n, length(tm@x)),
             sprintf("%d %d %d", tm@i + 1L, tm@j + 1L, as.integer(tm@x))), con); close(con)
write.table(data.frame(cell = colnames(counts), group = groups), file.path(OUT, "toy_cells.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

write_scores <- function(S, file) {
  df <- data.frame(cell = rownames(S), matrix(f17(S), nrow(S), dimnames = list(NULL, colnames(S))), check.names = FALSE)
  write.table(df, file.path(OUT, file), sep = "\t", quote = FALSE, row.names = FALSE)
}
write_sig <- function(res, file) wjson(list(up = lapply(res$signatures$up_idx, function(v) I(as.integer(v))),
                                            down = lapply(res$signatures$down_idx, function(v) I(as.integer(v))),
                                            universe = I(res$universe)), file)
cfg1 <- rfindsc_config(n_cores = 1)
r1 <- score_cells(counts, hb, cfg1, return_signatures = TRUE); write_scores(r1$scores, "scores_hammond.tsv")
wjson(as.list(r1$universe_size), "universe_size_hammond.json")                     # pair universe per panel (0.3.5)
write_sig(r1, "signatures_hammond.json")
write_scores(score_cells(counts, dd, cfg1)$scores, "scores_dedonno.tsv")
sub <- rownames(counts)[Matrix::rowMeans(counts > 0) >= 0.05]
writeLines(sub, file.path(OUT, "libsize_genes.txt"))
write_scores(score_cells(counts[sub, ], hb, cfg1, libsize = Matrix::colSums(counts))$scores, "scores_libsize.tsv")
both <- c(hb, dd[setdiff(names(dd), "meta")])
cfg5 <- rfindsc_config(top_n = 100, n_cell = 60, nonzero_rate = 0.05, clip = 12, n_cores = 1)
r5 <- score_cells(counts, both, cfg5, return_signatures = TRUE); write_scores(r5$scores, "scores_config.tsv")
write_sig(r5, "signatures_config.json")
gene_sets <- list(set_a = c(top(hb$DAM$up, 30), "NotAGene", NA, ""), set_b = toupper(top(dd$UPR$up, 40)),
                  set_c = c(top(hb$G2M$up, 25), top(hb$G2M$up, 5)))
wjson(lapply(gene_sets, I), "genesets.json")
write_scores(score_genesets(counts, gene_sets, cfg1)$scores, "scores_genesets.tsv")
## pair_chr: two fixed character vectors (drawn in 0.3.5 from a bundled reference removed in 0.3.6), read back
pair_chr <- lapply(read_json(file.path(OUT, "panel_pairs.json"))$pair_chr, function(g) as.character(unlist(g)))
pairs <- list(pair_chr = list(up = pair_chr$up, down = pair_chr$down),
              pair_df = list(up = hb$Maturity$up[1:150, ], down = hb$Maturity$down[1:150, ]),
              up_only = list(up = top(dd$Glucocorticoid$up, 100)),
              set_pair = list(up = sort(top(hb$DAM$up, 80)), down = sort(top(hb$DAM$down, 80)), ranked = FALSE))
wjson(list(pair_chr = lapply(pairs$pair_chr, I), pair_df = list(up = I(top(hb$Maturity$up, 150)), down = I(top(hb$Maturity$down, 150))),
           up_only = list(up = I(pairs$up_only$up)),
           set_pair = list(up = I(pairs$set_pair$up), down = I(pairs$set_pair$down), ranked = FALSE)), "panel_pairs.json")
write_scores(score_panel_pairs(counts, pairs, cfg1)$scores, "scores_panel_pairs.tsv")

## Per-group baseline: score_cells(baseline_groups = ) (RFindsc >= 0.3.3; the paper's De Donno rule,
## analysis/dedonno/02_signature.R of the RFind-sc repository). Cross-checked against the same rule written out with
## the package's own steps (as used for these fixtures before 0.3.3); with a single group it reproduces score_cells()
## exactly.
grouped <- function(counts, groups, biosets, config) {
  libsize <- Matrix::colSums(counts)
  cpm <- Matrix::t(Matrix::t(counts) / libsize) * 1e6
  keep <- Matrix::rowMeans(cpm > 0) >= config$nonzero_rate
  X <- as.matrix(counts[keep, , drop = FALSE]); cpm <- cpm[keep, , drop = FALSE]
  gn <- tolower(rownames(cpm)); P <- length(gn); ax <- setdiff(names(biosets), "meta")
  ## pair universe per panel (RFindsc >= 0.3.5): prepare_panel(), and each cell's lists cut inside it
  b2 <- lapply(ax, function(nm) ns$prepare_panel(ns$to_b2(biosets[[nm]]), gn, config$top_n))
  lv <- sort(unique(groups)); gi <- match(groups, lv)
  rates <- vapply(lv, function(g) as.numeric(Matrix::rowMeans(cpm[, groups == g, drop = FALSE])) / 1e6, numeric(P))
  S <- t(vapply(seq_len(ncol(X)), function(j) {
    r <- ns$rank_cell_residuals(X[, j], rates[, gi[j]] * libsize[j])
    vapply(seq_along(ax), function(a) ns$score_cell_ranking(r, b2[[a]], gn, config$n_cell, config$clip), 0)
  }, numeric(length(ax))))
  dimnames(S) <- list(colnames(counts), ax); S
}
stopifnot(max(abs(grouped(counts, rep("all", n), hb, cfg1) - r1$scores)) == 0)
rg <- score_cells(counts, both, cfg1, baseline_groups = groups)$scores
stopifnot(max(abs(grouped(counts, groups, both, cfg1) - rg)) == 0)
write_scores(rg, "scores_grouped.tsv")

## ---- build_panel_from_deg ----
deg <- data.frame(gene = c(sprintf("G%02d", 1:24), NA, "", "Gx", "Gy", "Gz"),
                  log2FC = c(as.character(c(round(rnorm(20, 0, 2), 1), 1.5, 1.5, -1.5, -1.5)), "1", "2", NA, "abc", "0"),
                  padj = c(round(runif(24, 0, 0.1), 3), 0.01, 0.01, 0.01, NA, 0.2))
deg$padj[c(3, 9)] <- NA
write.table(deg, file.path(OUT, "deg_table.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
side_out <- function(df) list(gene = I(df$gene), diff = I(f17(df$diff)), rank = I(df$rank))
dp <- list(filtered = suppressWarnings(build_panel_from_deg(deg, top_n = 5, label = "a", padj_col = "padj", padj_thresh = 0.05)),
           all = suppressWarnings(build_panel_from_deg(deg)))
wjson(lapply(dp, function(p) list(up = side_out(p$up), down = side_out(p$down), eligible = I(p$eligible))), "deg_panels.json")

## ---- bundled gene lists ----
md5 <- function(g) { f <- tempfile(); writeLines(as.character(g), f); unname(tools::md5sum(f)) }
summ <- function(b) lapply(b[setdiff(names(b), "meta")], function(p)
  list(n_up = nrow(p$up), n_down = nrow(p$down), md5_up = md5(p$up$gene), md5_down = md5(p$down$gene),
       n_eligible = length(p$eligible), md5_eligible = if (length(p$eligible)) md5(p$eligible) else NA))
wjson(list(biosets = summ(hb), biosets_dedonno = summ(dd)), "bundled_summary.json")

wjson(list(rfindsc_version = as.character(packageVersion("RFindsc")), r_version = R.version.string,
           created = format(Sys.time(), "%Y-%m-%d %H:%M:%S"), n_kernel_cases = length(cases),
           engine_max_abs_diff = eng_diff, toy_genes = G, toy_cells = n, toy_nnz = length(tm@x)), "fixtures_meta.json")
message(sprintf("fixtures written; kernel vs RFIND_ENGINE max |diff| = %s", format(eng_diff)))
