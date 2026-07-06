# BigRiverEssence

Paste here CI, codecov, MIT License, Stable, Pkg Status


## Description


This package provides efficient implementations of matrix decomposition
and multivariate dimension reduction methods, including Principal
Component Analysis (PCA)[^1], Sparse Principal Component Analysis
(SPCA)[^2], Penalized Matrix Decomposition (PMD)[^2], Canonical
Correlation Analysis (CCA)[^3], Sparse Canonical Correlation Analysis
(SCCA)[^4], Joint and Individual Variation Explained (JIVE)[^5],
Sparse Partial Least Squares Discriminant Analysis (SPLSDA)[^6][^7], and
Partial Least Squares Kernel Regression (PLSkern)[^8].

These methods are useful for extracting low-dimensional structure from
high-dimensional data, identifying sparse latent components, and
analyzing shared and individual variation across multiple datasets.
The package is intended for high-dimensional applications such as
high-throughput biological data.

The core routines rely on matrix operations for fast computation and
are designed to support exploratory data analysis, feature extraction,
and integrative analysis of complex datasets.







## Installation 

The `BigRiverEssence` package can be installed by running: 

```
using Pkg
Pkg.add("BigRiverEssence")
```

or from the julia REPL, press `]` to enter pkg mode, and execute the following command:

```
add BigRiverEssence
```

For the most recent (development) version, use:
```
using Pkg
Pkg.add(url = "https://github.com/senresearch/BigRiverEssence.jl", rev="main")
```

## Contributing

We appreciate contributions from users including reporting bugs, fixing issues, improving performance and adding new features.

## Questions

If you have questions about contributing or using `BigRiverEssence` package, please communicate with the authors via GitHub.

## Citing `BigRiverEssence`

????

## References

[^1]: Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space*. Philosophical Magazine, 2(11), 559–572.

[^2]: Witten, D. M., Tibshirani, R., & Hastie, T. (2009). *A Penalized Matrix Decomposition, with Applications to Sparse Principal Components and Canonical Correlation Analysis*. Biostatistics, 10(3), 515–534.

[^3]: Weenink, D. (2003). *Canonical Correlation Analysis*. Institute of Phonetic Sciences, University of Amsterdam, Proceedings 25, 81–99.

[^4]: Witten, D. M., & Tibshirani, R. (2009). *Extensions of Sparse Canonical Correlation Analysis with Applications to Genomic Data*. Statistical Applications in Genetics and Molecular Biology, 8(1), Article 28.

[^5]: Lock, E. F., Hoadley, K. A., Marron, J. S., & Nobel, A. B. (2013). *Joint and Individual Variation Explained (JIVE) for Integrated Analysis of Multiple Data Types*. Annals of Applied Statistics, 7(1), 523–542.

[^6]: Lê Cao, K.-A., Boitard, S., & Besse, P. (2011). *Sparse PLS Discriminant Analysis: Biologically Relevant Feature Selection and Graphical Displays for Multiclass Problems*. BMC Bioinformatics, 12, 253. 

[^7]: Lê Cao, K.-A., Rossouw, D., Robert-Granié, C., & Besse, P. (2008). *A Sparse PLS for Variable Selection when Integrating Omics Data*. Statistical Applications in Genetics and Molecular Biology, 7(1), Article 35. 

[^8]: Dayal, B. S., & MacGregor, J. F. (1997). *Improved PLS Algorithms*. Journal of Chemometrics, 11(1), 73–85.

