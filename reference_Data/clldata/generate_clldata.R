# generate_clldata.R
# One-time: export a compact real CLL example from the official MOFAdata files.
# Run from the BigRiverEssence repository root. Documentation reads the saved
# CSVs and does not require R, Python, or an internet connection.

outdir <- "reference_Data/clldata"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
source_url <- "https://raw.githubusercontent.com/bioFAM/MOFAdata/master/data/"
source_dir <- tempfile("mofa-cll-source-")
dir.create(source_dir)

for (name in c("CLL_data.RData", "CLL_covariates.RData")) {
    download.file(paste0(source_url, name), file.path(source_dir, name), mode = "wb")
}
source_data <- new.env()
load(file.path(source_dir, "CLL_data.RData"), envir = source_data)
load(file.path(source_dir, "CLL_covariates.RData"), envir = source_data)

samples <- colnames(source_data$CLL_data$Drugs)
metadata <- source_data$CLL_covariates
stopifnot(!anyDuplicated(samples), all(samples %in% rownames(metadata)))
metadata <- metadata[samples, , drop = FALSE]
metadata <- data.frame(sample = samples, gender = as.character(metadata$Gender),
                       diagnosis = as.character(metadata$Diagnosis))
write.table(metadata, file.path(outdir, "samples.csv"), sep = ",",
            row.names = FALSE, col.names = TRUE, quote = TRUE, na = "NaN")

view_names <- c("Drugs", "mRNA", "Methylation")
file_names <- c("drugs.csv", "rna.csv", "methylation.csv")
feature_limits <- c(310L, 500L, 500L)
overview <- data.frame()
for (m in seq_along(view_names)) {
    X <- source_data$CLL_data[[view_names[m]]]
    stopifnot(!anyDuplicated(colnames(X)), !anyDuplicated(rownames(X)),
              setequal(samples, colnames(X)))
    X <- X[, samples, drop = FALSE]
    feature_variance <- apply(X, 1L, var, na.rm = TRUE)
    eligible <- which(is.finite(feature_variance) & feature_variance > 0)
    if (view_names[m] == "Drugs") {
        selected <- eligible
    } else {
        selected <- eligible[order(-feature_variance[eligible], rownames(X)[eligible])]
        selected <- head(selected, feature_limits[m])
    }
    X <- t(X[selected, , drop = FALSE])
    stopifnot(nrow(X) == length(samples), ncol(X) == feature_limits[m],
              all(is.finite(X[!is.na(X)])))
    write.table(X, file.path(outdir, file_names[m]), sep = ",",
                row.names = FALSE, col.names = TRUE, quote = TRUE, na = "NaN")
    overview <- rbind(overview, data.frame(view = view_names[m], samples = nrow(X),
        features = ncol(X), observed_samples = sum(rowSums(!is.na(X)) > 0),
        missing_entries = sum(is.na(X))))
}
write.table(overview, file.path(outdir, "overview.csv"), sep = ",",
            row.names = FALSE, col.names = TRUE, quote = TRUE)
print(overview)
cat("Saved aligned CLL matrices and sample IDs to", outdir, "\n")
