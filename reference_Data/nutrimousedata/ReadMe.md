# nutrimouse reference data (`generate_nutrimousedata.R`)

The `generate_nutrimousedata.R` script extracts the nutrimouse dataset from the `mixOmics`
R package and saves it as CSV files. These files are read by `cca_tutorial.md` and
`scca_tutorial.md` so the tutorials can load the data directly, without requiring R or the
`mixOmics` package at build time. The script only needs to be run once; the resulting CSVs
are committed to the repository.

## It performs the following tasks:

- It loads the R package `mixOmics` and the `nutrimouse` dataset.
- It extracts the gene expression and lipid concentration measurements for 40 mice.
- It writes the gene expression matrix (40 mice × 120 genes) as `genes.csv`.
- It writes the lipid concentration matrix (40 mice × 21 lipids) as `lipids.csv`.

Unlike the other reference datasets, these files are written with a header row, so the gene
and lipid names are preserved — the tutorials use those names to interpret which genes and
lipids drive the canonical correlations.

## Where it is used

The CSV files are read by `cca_tutorial.md` and `scca_tutorial.md`, which load the gene and
lipid data and apply `cca` and `scca` respectively to find correlated combinations of genes
and lipids across the mice. Storing the data as CSVs keeps the tutorials reproducible and
removes any dependency on R when the documentation is built.

## References

The nutrimouse dataset comes from a nutrigenomic study in mice (Martin et al., 2007) [1],
containing the expression of 120 genes and the concentrations of 21 hepatic fatty acids
measured on the same 40 mice. It is obtained via the `mixOmics` R package [2].

[1] Martin, P. G. P., Guillou, H., Lasserre, F., Déjean, S., Lan, A., Pascussi, J.-M.,
    San Cristobal, M., Legrand, P., Besse, P., & Pineau, T. (2007). Novel aspects of
    PPARα-mediated regulation of lipid and xenobiotic metabolism revealed through a
    nutrigenomic study. *Hepatology*, 54, 767–777.

[2] Rohart, F., Gautier, B., Singh, A., & Lê Cao, K.-A. (2017). mixOmics: An R package for
    'omics feature selection and multiple data integration. *PLoS Computational Biology*,
    13(11), e1005752.


