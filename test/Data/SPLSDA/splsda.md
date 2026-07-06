# SPLSDA reference fixtures (`splsda.R`)

The `splsda.R` is used in context of testing the outputs of `splsda` function of `BigRiverEssence.jl` with the outputs of the original R implementation `mixOmics::splsda`. It produces all the simulated data matrices and outputs of `mixOmics::splsda` which are used in `splsda_test.jl` to test similarity of outputs with `mixOmics::splsda`.

## It performs the following tasks:

- It loads the R package `mixOmics` and fixes a random seed for reproducibility.
- It simulates a predictor matrix `X` and a class level vector `Y` and saves it as `X.csv` and `Y.csv` respectively. It also stores the ordered class levels of `Y` as `levels.csv`.
- It fits `mixOmics::splsda` with a fixed number of components `ncomp` and a fixed per-component variable budget `keepX`. 
- It writes the outputs: the X and Y loadings as `lx.csv` and`ly.csv`, the X and Y variates as `vx.csv` and `vy.csv`.
- It records the `R` version and the `PMA` package versions as `sessionInfo.txt` and `session_meta.csv`. 
- It creates `meta.csv` containing parameters the `R` fixture was generated it. 
