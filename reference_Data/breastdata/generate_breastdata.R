# generate_breastdata.R
# One-time script: download PMA's breastdata and save dna, chrom, nuc as CSVs.
# Run this once (locally, where R and PMA are available); commit the resulting CSVs.
# The tutorials then read the CSVs directly.

suppressMessages(suppressWarnings({
  library(PMA)
  breast <- download_breast_data(url = "https://tibshirani.su.domains/PMA/breastdata.rda")
}))

# Extract the pieces used by the tutorial
dna_t <- t(breast$dna)     # transpose: samples on rows, CGH spots on columns (89 × 2149)
chrom <- breast$chrom      # 2149 chromosome labels
nuc   <- breast$nuc        # 2149 genomic positions

# Output directory 
outdir <- "reference_Data/breastdata"


# Write CSVs. row.names = FALSE and col.names = FALSE keep them as plain numeric
# matrices/vectors, matching how readdlm reads them in Julia.
write.table(dna_t, file = file.path(outdir, "dna.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(chrom, file = file.path(outdir, "chrom.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
write.table(nuc, file = file.path(outdir, "nuc.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)

cat("Saved breastdata CSVs to", outdir, "\n")
cat("dna:  ", nrow(dna_t), "x", ncol(dna_t), "\n")
cat("chrom:", length(chrom), "\n")
cat("nuc:  ", length(nuc), "\n")