# CLL reference data (`generate_clldata.R`)

The `generate_clldata.R` script exports the real chronic lymphocytic leukaemia
(CLL) data distributed by `MOFAdata` for use in `mofa_tutorial.md`. This is the
same cohort used by the official Python MOFA tutorial linked below. The CSVs
are stored in the repository so documentation can run without R, Python,
MOFAdata, or network access.

## It performs the following tasks:

- It downloads `CLL_data.RData` and `CLL_covariates.RData` from the official
  [MOFAdata repository](https://github.com/bioFAM/MOFAdata/tree/master/data).
- It retains all 200 patients and aligns every view to the drug matrix's
  patient IDs, checking unique identifiers and matching sample sets.
- It retains all 310 drug/concentration features. Among features with finite,
  positive observed variance, it selects the 500 most variable RNA genes and
  500 most variable methylation sites; ties are resolved by feature ID.
- It transposes the data to samples in rows and features in columns and writes
  `drugs.csv` (200×310), `rna.csv` (200×500), and `methylation.csv` (200×500).
  Header rows preserve the source feature IDs.
- It writes `samples.csv`, containing patient IDs, sex (`gender` in the source),
  and diagnosis, in the exact row order shared by all three matrices.
- It writes `overview.csv`, recording dimensions, numbers of observed
  patients, and numbers of missing entries.

The source RNA values are already normalized and the methylation values are
M-values. No additional normalization, imputation, artificial missingness, or
simulated measurements are introduced by the export. Missing values are written
as `NaN`. Drug, RNA, and methylation data are available for 184, 136, and 196
patients respectively; absent views retain their rows of missing values.

The binary mutation view is deliberately excluded because BigRiverEssence's
current MOFA implementation supports Gaussian likelihoods only. The tutorial
uses five factors and this compact subset, not the full four-view, 15-factor
model from the published analysis. Its factor labels are not claimed to match
the published model.

## Regenerating the files

From the BigRiverEssence repository root, run:

```sh
Rscript reference_Data/clldata/generate_clldata.R
```

The generator uses base R and an internet connection; no R package installation
is required. Regeneration overwrites only the five named CSV files in this
directory. Normal documentation builds only read those files.

## References and attribution

1. Dietrich, S., Oleś, M., Lu, J., et al. (2018). Drug-perturbation-based
   stratification of blood cancer. *Journal of Clinical Investigation*,
   128(1), 427–445. <https://doi.org/10.1172/JCI93801>.
2. Argelaguet, R., Velten, B., Arnol, D., et al. (2018). Multi-Omics Factor
   Analysis—a framework for unsupervised integration of multi-omics data sets.
   *Molecular Systems Biology*, 14, e8124.
   <https://doi.org/10.15252/msb.20178124>.
3. [MOFAdata](https://github.com/bioFAM/MOFAdata) and its
   [CLL data description](https://github.com/bioFAM/MOFAdata/blob/master/man/CLL_data.Rd).
   MOFAdata credits Ricard Argelaguet, Britta Velten, Damien Arnol, Florian
   Buettner, Wolfgang Huber, and Oliver Stegle; its DESCRIPTION declares
   LGPL-3. These external data are attributed to their original providers,
   not presented as newly generated BigRiverEssence data. Copies of the upstream
   declared license texts accompany the export as `LICENSE.LGPL-3` and
   `LICENSE.GPL-3`.
4. [Official Python CLL tutorial](https://muon-tutorials.readthedocs.io/en/latest/CLL.html),
   linked from the [MOFA tutorials page](https://biofam.github.io/MOFA2/tutorials.html).
   It uses Gaussian drug/RNA/methylation views and a Bernoulli mutation view.
