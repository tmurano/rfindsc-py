## Export the reference biosets bundled with RFindsc 0.3.8 (inst/extdata/*.rds) to the JSON files shipped with the
## Python package (src/rfindsc/data/). Not run at install time. Run from the rfindsc-py folder:
##   RFINDSC_SRC=<RFindsc source folder> Rscript data-raw/export_bundled.R
src <- Sys.getenv("RFINDSC_SRC", "../RFindsc")
stopifnot(package_version(read.dcf(file.path(src, "DESCRIPTION"))[, "Version"]) >= "0.3.8")   # bundle rebuilt in 0.3.8
out_dir <- "src/rfindsc/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

side <- function(df) list(gene = I(as.character(df$gene)), diff = I(as.numeric(df$diff)), rank = I(as.integer(df$rank)))
plain <- function(x) if (is.atomic(x) && !is.null(names(x))) as.list(x) else x
for (f in c("biosets", "biosets_dedonno")) {
  b <- readRDS(file.path(src, "inst", "extdata", paste0(f, ".rds")))
  nm <- setdiff(names(b), "meta")
  panels <- lapply(nm, function(k) {
    p <- b[[k]]; out <- lapply(p, plain); out$up <- side(p$up); out$down <- side(p$down)
    if (!is.null(p$eligible)) out$eligible <- I(as.character(p$eligible))   # pair universe (0.3.5)
    out
  })
  names(panels) <- nm
  x <- list(panels = panels, meta = lapply(b$meta, plain))
  jsonlite::write_json(x, file.path(out_dir, paste0(f, ".json")), auto_unbox = TRUE, digits = NA, na = "null")
  message(sprintf("%s: %d panels", f, length(panels)))
}
