# SCCA reference fixtures (`scca.R`)

The `scca.R` is used in context of testing the outputs of `scca` function of `BigRiverEssence.jl` with the outputs of the original R implementation of `PMA::CCA`. It produces all the simulated data matrices and outputs of `PMA::CCA` which are used in `scca_test.jl` to test similarity of outputs with `PMA::CCA`.

## It performs the following tasks:

- It loads the R package `PMA` and fixes a random seed for reproducibility.
- It simulates two data matrix `X` and `Z` and saves it them as `X.csv` and `Z.csv` respectively.
- It fits `PMA::CCA` to `X` and `Z` matrices with fixed penalties `penaltyx` and `penaltyz`, fixed number of components `K` and fixed number of iterations `niter`.
- It writes the outputs of `PMA::CCA`: the canonical vectors `u` and `v` as `u.csv` and`v.csv`, the weights `d` as `d.csv` and the canonical correlations `cors` as `cors.csv`.
- It records the `R` version and the `PMA` package versions as `sessionInfo.txt` and `session_meta.csv`. 
- It creates `meta.csv` containing parameters the `R` fixture was generated it. 
