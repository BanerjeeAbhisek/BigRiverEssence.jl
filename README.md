# BigRiverEssence


[![CI](https://github.com/senresearch/BigRiverEssence.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/senresearch/BigRiverEssence.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/senresearch/BigRiverEssence.jl/branch/main/graph/badge.svg?token=uHM6utUQoi)](https://codecov.io/gh/senresearch/BigRiverEssence.jl)
[![Stable](https://img.shields.io/badge/docs-dev-blue.svg)](https://senresearch.github.io/BigRiverEssence.jl/dev)
[![Pkg Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)


## Description

BigRiverEssence implements different methods for capturing the essential structure 
of high-dimensional data matrices. It provides a broad range of methods, from standard 
techniques to specialized algorithms for sparse, supervised, and integrative analyses.

> **Why “Essence”?** The name reflects the package’s goal of reducing
> large, high-dimensional data matrices to their essential structure
> through matrix decomposition and dimension reduction.  It is a
> component of the BigRiver Julia package ecosystem.

These methods can be used to extract low-dimensional structure, identify sparse latent components, 
and characterize shared and dataset-specific variation across multiple data sources. The package 
is designed for high-dimensional data, particularly high-throughput biological and omics 
data.

The core routines prioritize computational efficiency in mind, with an emphasis on 
reducing memory usage. They rely primarily on optimized matrix 
operations to support fast exploratory data analysis, feature extraction, supervised dimension 
reduction, and integrative analysis of complex datasets.

The package currently implements the following methods:

* Principal Component Analysis (PCA)[^1]
* Sparse Principal Component Analysis (SPCA)[^2]
* Penalized Matrix Decomposition (PMD)[^2]
* Nonnegative Matrix Factorization (NMF)[^3][^4]
* Canonical Correlation Analysis (CCA)[^5]
* Sparse Canonical Correlation Analysis (SCCA)[^6]
* Joint and Individual Variation Explained (JIVE)[^7]
* Gaussian Multi-Omics Factor Analysis (MOFA)[^12][^13]
* Partial Least Squares Kernel Regression (PLSkern)[^8]
* Partial Least Squares Discriminant Analysis (PLSDA)[^9]
* Sparse Partial Least Squares Discriminant Analysis (SPLSDA)[^10][^11]

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

We appreciate contributions from users. GitHub Issues are the preferred venue for 
reporting bugs and requesting features, while GitHub Pull Requests are the preferred 
venue for contributing code, such as bug fixes, performance improvements, or new features.

## Questions

If you have questions about contributing or using `BigRiverEssence` package, please communicate with the authors via GitHub.


## References

[^1]: Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space*. Philosophical Magazine, 2(11), 559–572.

[^2]: Witten, D. M., Tibshirani, R., & Hastie, T. (2009). *A Penalized Matrix Decomposition, with Applications to Sparse Principal Components and Canonical Correlation Analysis*. Biostatistics, 10(3), 515–534.

[^3]: Cichocki, A., & Phan, A.-H. (2009). *Fast Local Algorithms for Large Scale Nonnegative Matrix and Tensor Factorizations*. IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, E92-A(3), 708–721.

[^4]: Boutsidis, C., & Gallopoulos, E. (2008). *SVD Based Initialization: A Head Start for Nonnegative Matrix Factorization*. Pattern Recognition, 41(4), 1350–1362.

[^5]: Weenink, D. (2003). *Canonical Correlation Analysis*. Institute of Phonetic Sciences, University of Amsterdam, Proceedings 25, 81–99.

[^6]: Witten, D. M., & Tibshirani, R. (2009). *Extensions of Sparse Canonical Correlation Analysis with Applications to Genomic Data*. Statistical Applications in Genetics and Molecular Biology, 8(1), Article 28.

[^7]: Lock, E. F., Hoadley, K. A., Marron, J. S., & Nobel, A. B. (2013). *Joint and Individual Variation Explained (JIVE) for Integrated Analysis of Multiple Data Types*. Annals of Applied Statistics, 7(1), 523–542.

[^8]: Dayal, B. S., & MacGregor, J. F. (1997). *Improved PLS Algorithms*. Journal of Chemometrics, 11(1), 73–85.

[^9]: Pérez-Enciso, M., Tenenhaus, M. Prediction of clinical outcome with microarray data: a partial least squares discriminant analysis (PLS-DA) approach. Hum Genet 112, 581–592 (2003). https://doi.org/10.1007/s00439-003-0921-9

[^10]: Lê Cao, K.-A., Boitard, S., & Besse, P. (2011). *Sparse PLS Discriminant Analysis: Biologically Relevant Feature Selection and Graphical Displays for Multiclass Problems*. BMC Bioinformatics, 12, 253. 

[^11]: Lê Cao, K.-A., Rossouw, D., Robert-Granié, C., & Besse, P. (2008). *A Sparse PLS for Variable Selection when Integrating Omics Data*. Statistical Applications in Genetics and Molecular Biology, 7(1), Article 35. 

[^12]: Argelaguet, R., Velten, B., Arnol, D., et al. (2018). *Multi-Omics Factor Analysis—a framework for unsupervised integration of multi-omics data sets*. Molecular Systems Biology, 14, e8124. https://doi.org/10.15252/msb.20178124

[^13]: Argelaguet, R., Arnol, D., Bredikhin, D., et al. (2020). *MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data*. Genome Biology, 21, 111. https://doi.org/10.1186/s13059-020-02015-1

