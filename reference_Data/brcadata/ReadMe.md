# BRCA reference data (`generate_brcadata.R`)

The `generate_brcadata.R` script extracts the BRCA (breast cancer) multi-source dataset
from the `r.jive` R package and saves it as CSV files. These files are read by
`jive_tutorial.md` so the tutorial can load the data directly, without requiring R or the
`r.jive` package at build time. The script only needs to be run once; the resulting CSVs
are committed to the repository.

## It performs the following tasks:

- It loads the R package `r.jive` and the `BRCA_data` object.
- It extracts the three data blocks — gene expression, DNA methylation, and miRNA
  expression — each measured on the same 348 breast tumor samples, along with the
  cluster labels.
- It writes the gene expression block (645 genes × 348 samples) as `expression.csv`.
- It writes the DNA methylation block (574 CpG sites × 348 samples) as `methylation.csv`.
- It writes the miRNA expression block (423 miRNAs × 348 samples) as `mirna.csv`.
- It writes the cluster labels for the 348 samples as `clusts.csv`.

Each data block is stored with features in rows and samples in columns (the format JIVE
expects), and the 348 sample columns are shared and aligned across all three blocks.

## Where it is used

The CSV files are read by `jive_tutorial.md`, which loads the three blocks and cluster
labels and applies `jive` to decompose the multi-source data into joint and individual
structure. Storing the data as CSVs keeps the tutorial reproducible and removes any
dependency on R when the documentation is built.

## References

The BRCA dataset contains gene expression, DNA methylation, and miRNA expression measured
on the same 348 breast tumor samples from The Cancer Genome Atlas [1]. It is obtained via
the `r.jive` R package [3], which implements the JIVE method [2].

[1] Cancer Genome Atlas Network (2012). Comprehensive molecular portraits of human breast
    tumours. *Nature*, 490(7418), 61–70.

[2] Lock, E. F., Hoadley, K. A., Marron, J. S., & Nobel, A. B. (2013). Joint and Individual
    Variation Explained (JIVE) for integrated analysis of multiple data types. *The Annals
    of Applied Statistics*, 7(1), 523–542.

[3] O'Connell, M. J., & Lock, E. F. (2016). R.JIVE for exploration of multi-source molecular
    data. *Bioinformatics*, 32(18), 2877–2879.



