# PMD reference fixtures (`pmd.R`)

The `pmd.R` is used in context of testing the outputs of `pmd` function of `BigRiverEssence.jl` with the outputs of the original R implementation of `PMA::PMD`. It produces all the simulated data matrices and outputs of `PMA::PMD` which are used in `pmd_test.jl` to test similarity of outputs with `PMA::PMD`.

## It performs the following tasks:

- It loads the R package `PMA` and fixes a random seed for reproducibility.
- It simulates a data matrix `X` and saves it as `X.csv`.
- It fits `PMA::PMD` to the `X` matrix with a fixed sparsity budget `sumabs` and a fixed number of components `K`.
- It writes the outputs of `PMA::PMD`: the left and right factors `u` and `v`, and the weights `d` as `u_pmd.csv`, `v_pmd.csv` and `d_pmd.csv` respectively. 
- It records the `R` version and the `PMA` package versions as `sessionInfo.txt` and `session_meta.csv`. 
- It creates `meta.csv` containing parameters the `R` fixture was generated it. 
