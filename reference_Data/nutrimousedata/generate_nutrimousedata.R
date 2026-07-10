# generate_nutrimousedata.R
# One-time: extract the nutrimouse dataset from mixOmics and save as CSVs.
# Used by BOTH the cca and scca tutorials. Run once; commit the CSVs.

suppressMessages(library(mixOmics))
data(nutrimouse)

genes  <- as.matrix(nutrimouse$gene)    # 40 × 120 liver gene expression
lipids <- as.matrix(nutrimouse$lipid)   # 40 × 21 lipid concentrations

outdir <- "reference_Data/nutrimousedata"

# save data with a header row so gene/lipid NAMES are preserved (the tutorials use them)
write.table(genes, file = file.path(outdir, "genes.csv"),
            sep = ",", row.names = FALSE, col.names = TRUE)
write.table(lipids, file = file.path(outdir, "lipids.csv"),
            sep = ",", row.names = FALSE, col.names = TRUE)

cat("Saved nutrimouse CSVs to", outdir, "\n")
cat("genes: ", nrow(genes), "x", ncol(genes), "\n")
cat("lipids:", nrow(lipids), "x", ncol(lipids), "\n")