# SPC reference fixtures (`spc.R`)

The `spc.R` is used in context of testing the outputs of `spc` function of `BigRiverEssence.jl` with the outputs of the original R implementation of `PMA::SPC`. It produces all the simulated data matrices and outputs of `PMA::SPC` which are used in `spc_test.jl` to test similarity of outputs with `PMA::SPC`.

## It performs the following tasks:

- It loads the R package `PMA` and fixes a random seed for reproducibility.
- It simulates a data matrix `X` and saves it as `X.csv`.
- It fits `PMA::SPC` to the `X` matrix with a fixed sparsity budget `sumabsv` and a fixed number of components `K`.
  - It uses `orth = FALSE` in `PMA::SPC` and writes the outputs: the loadings as `v_spc.csv` and the
    singular values as `d_spc.csv`.
  - It uses `orth = TRUE` in `PMA::SPC` and writes the outputs: the loadings as `v_spc_orth.csv` and the
    singular values as `d_spc_orth.csv`.  
- It records the `R` version and the `PMA` package versions as `sessionInfo.txt` and `session_meta.csv`. 
- It creates `meta.csv` containing parameters the `R` fixture was generated it. 
