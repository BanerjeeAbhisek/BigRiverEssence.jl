# generate_srbct.R
# One-time: extract the SRBCT dataset from mixOmics and save as CSVs.
# Run once locally where R and mixOmics are available; commit the resulting CSVs.

suppressMessages(library(mixOmics))
data(srbct)

Xg <- srbct$gene                    # 63 × 2308 gene expression
yg <- as.character(srbct$class)     # 63 class labels (EWS, BL, NB, RMS)

outdir <- "reference_Data/srbctdata"

write.table(Xg, file = file.path(outdir, "gene.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
# labels are strings — write one per line, no quotes
writeLines(yg, con = file.path(outdir, "class.csv"))

cat("Saved SRBCT CSVs to", outdir, "\n")
cat("gene: ", nrow(Xg), "x", ncol(Xg), "\n")
cat("class:", length(yg), "\n")