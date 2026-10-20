args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1], mustWork = TRUE)
for (f in list.files(file.path(root, "R"), pattern = "\\.R$", full.names = TRUE)) source(f)
request <- jsonlite::fromJSON(args[2], simplifyVector = FALSE)

panel_list <- function(panels) lapply(panels, function(p) {
  b <- list(up = data.frame(gene = as.character(unlist(p$up))),
            down = data.frame(gene = as.character(unlist(p$down))))
  if (!is.null(p$ranked)) b$ranked <- p$ranked
  if (!is.null(p$eligible)) b$eligible <- as.character(unlist(p$eligible))
  b
})
results <- lapply(request$cases, function(case) {
  cat("R computing", case$name, "\n"); flush.console()
  genes <- unlist(case$genes, use.names = FALSE)
  cells <- unlist(case$cells, use.names = FALSE)
  counts <- matrix(unlist(case$counts), nrow = length(genes), byrow = TRUE,
                   dimnames = list(genes, cells))
  if (isTRUE(case$sparse)) counts <- Matrix::Matrix(counts, sparse = TRUE)
  config <- do.call(rfindsc_config, lapply(case$config, function(x) if (identical(x, "Inf")) Inf else x))
  common <- list(config = config, return_signatures = TRUE)
  if (!is.null(case$libsize)) common$libsize <- unlist(case$libsize)
  if (!is.null(case$baseline_groups)) common$baseline_groups <- unlist(case$baseline_groups)
  warnings <- character()
  out <- withCallingHandlers({
    if (!is.null(case$bundle)) {
      bs <- readRDS(file.path(root, "inst/extdata", if (case$bundle == "hammond")
                               "biosets.rds" else "biosets_dedonno.rds"))
      if (case$bundle == "both") bs <- c(readRDS(file.path(root, "inst/extdata/biosets.rds")),
                                         bs[names(bs) != "meta"])
      do.call(score_cells, c(list(counts = counts, biosets = bs), common))
    } else if (case$api == "genesets") {
      do.call(score_genesets, c(list(input = counts, gene_sets = lapply(case$gene_sets, unlist)), common))
    } else if (case$api == "pairs") {
      do.call(score_panel_pairs, c(list(input = counts, panels = panel_list(case$panels)), common))
    } else do.call(score_cells, c(list(counts = counts, biosets = panel_list(case$panels)), common))
  }, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  })
  list(name = case$name, scores = unname(out$scores), panels = as.list(colnames(out$scores)),
       cells = as.list(rownames(out$scores)), universe = as.list(out$universe),
       universe_size = as.list(out$universe_size),
       up_idx = lapply(out$signatures$up_idx, function(x) as.list(as.integer(x) - 1L)),
       down_idx = lapply(out$signatures$down_idx, function(x) as.list(as.integer(x) - 1L)),
       warning_count = length(warnings))
})
names_for <- function(x) if (length(x)) paste0("g", unlist(x)) else character()
cat("R computing", length(request$kernels), "kernel cases\n"); flush.console()
kernels <- lapply(request$kernels, function(k) suppressWarnings(running_fisher(
  names_for(k$g1), names_for(k$g2), k$P, clip = k$clip,
  g2_ranked = if (is.null(k$ranked)) TRUE else k$ranked)))
jsonlite::write_json(list(cases = results, kernels = kernels, r_version = R.version.string),
                      args[3], auto_unbox = TRUE, digits = NA, pretty = FALSE)
