# breastdata reference data (`generate_breastdata.R`)

The `generate_breastdata.R` script downloads the breast cancer dataset from the `PMA` R
package and saves it as CSV files. These files are read by `pmd_tutorial.md` and
`spc_tutorial.md` so the tutorials can load the data directly, without requiring R or the
`PMA` package at build time. The script only needs to be run once; the resulting CSVs are
committed to the repository.

## It performs the following tasks:

- It loads the R package `PMA` and downloads the `breastdata` object.
- It extracts the DNA copy-number data, transposing it so samples are on rows and CGH
  spots on columns, along with the chromosome labels and genomic positions.
- It writes the DNA copy-number matrix (89 samples × 2149 CGH spots) as `dna.csv`.
- It writes the 2149 chromosome labels as `chrom.csv`.
- It writes the 2149 genomic positions as `nuc.csv`.

The data is stored as plain numeric CSVs (no headers or row names), matching how `readdlm`
reads them in Julia.

## Where it is used

The CSV files are read by `pmd_tutorial.md` and `spc_tutorial.md`, which load the DNA
copy-number data and apply `pmd` and `spc` respectively to find sparse copy-number
signatures across the genome. Storing the data as CSVs keeps the tutorials reproducible and
removes any dependency on R when the documentation is built.

## References

The breast cancer dataset (CGH copy-number and gene expression on 89 samples) is from
Chin et al. (2006) [1], obtained via the `PMA` R package, where it serves as the worked
example for the penalized matrix decomposition of Witten, Tibshirani & Hastie (2009) [2].

[1] Chin, K., DeVries, S., Fridlyand, J., et al. (2006). Genomic and transcriptional
    aberrations linked to breast cancer pathophysiologies. *Cancer Cell*, 10, 529–541.

[2] Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix decomposition,
    with applications to sparse principal components and canonical correlation analysis.
    *Biostatistics*, 10(3), 515–534.


