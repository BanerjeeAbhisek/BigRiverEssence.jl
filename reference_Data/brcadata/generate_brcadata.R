# generate_brcadata.R
# One-time: extract the BRCA multi-omics data from r.jive and save as CSVs.
# Run once locally where R and r.jive are available; commit the resulting CSVs.

suppressMessages(library(r.jive))
data(BRCA_data)

X1 <- Data[[1]]     # Expression block (features × samples)
X2 <- Data[[2]]     # Methylation block
X3 <- Data[[3]]     # miRNA block
cl <- clusts        # cluster labels

outdir <- "reference_Data/brcadata"

write.table(X1, file = file.path(outdir, "expression.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(X2, file = file.path(outdir, "methylation.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(X3, file = file.path(outdir, "mirna.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(cl, file = file.path(outdir, "clusts.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)

cat("Saved BRCA CSVs to", outdir, "\n")
cat("Expression: ", nrow(X1), "x", ncol(X1), "\n")
cat("Methylation:", nrow(X2), "x", ncol(X2), "\n")
cat("miRNA:      ", nrow(X3), "x", ncol(X3), "\n")
cat("clusts:     ", length(cl), "\n")