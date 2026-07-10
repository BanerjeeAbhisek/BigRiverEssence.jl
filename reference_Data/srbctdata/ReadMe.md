# SRBCT reference data (`generate_srbct.R`)

The `generate_srbct.R` script extracts the SRBCT (Small Round Blue Cell Tumors) dataset
from the `mixOmics` R package and saves it as CSV files. These files are read by
`splsda_tutorial.md` so the tutorial can load the data directly, without requiring R or the
`mixOmics` package at build time. The script only needs to be run once; the resulting CSVs
are committed to the repository.

## It performs the following tasks:

- It loads the R package `mixOmics` and the `srbct` dataset.
- It extracts the gene expression measurements and the tumor class labels for 63 samples.
- It writes the gene expression matrix (63 samples × 2308 genes) as `gene.csv`.
- It writes the 63 tumor class labels (EWS, BL, NB, RMS) as `class.csv`.

The gene expression is stored as a plain numeric CSV. The class labels are text, so they
are written one per line, allowing them to be read back as strings in Julia.

## Where it is used

The CSV files are read by `splsda_tutorial.md`, which loads the gene expression and tumor
labels and applies `splsda` to classify the four tumor types while selecting a small subset
of discriminating genes. Storing the data as CSVs keeps the tutorial reproducible and
removes any dependency on R when the documentation is built.

## References

The SRBCT (Small Round Blue Cell Tumors) dataset contains expression levels of 2308 genes
on 63 samples across four tumor classes — Ewing sarcoma (EWS), Burkitt lymphoma (BL),
neuroblastoma (NB), and rhabdomyosarcoma (RMS) — from Khan et al. (2001) [1], obtained via
the `mixOmics` R package [2].

[1] Khan, J., Wei, J. S., Ringnér, M., et al. (2001). Classification and diagnostic
    prediction of cancers using gene expression profiling and artificial neural networks.
    *Nature Medicine*, 7(6), 673–679.

[2] Rohart, F., Gautier, B., Singh, A., & Lê Cao, K.-A. (2017). mixOmics: An R package for
    'omics feature selection and multiple data integration. *PLoS Computational Biology*,
    13(11), e1005752.


