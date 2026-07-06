# JIVE reference fixtures (`jive.R`)

The `jive.R` is used in context of testing the outputs of `jive` function of `BigRiverEssence.jl` with the outputs of the original R implementation  `r.jive`. It produces all the simulated data matrices and outputs of `r.jive` which are used in `jive_test.jl` to test similarity of outputs with `r.jive`.

## It performs the following tasks:

- It loads the R package `r.jive` and fixes a random seed for reproducibility.
- It simulates a data matrices in two stages:
   - It simulates two data matrices `X1` and `X2` with known joint and indivisual structures with known joint rank `rT` and indivisual ranks `r1T` and `r2T`. It writes the matrices as `X1.csv` and `X2.csv`. 
   - It simulates two more matrices `X1s` and `X2s` which are more strongly separated and have ranks `X1s.csv` and `X2s.csv` respectively. 
- It fits `r.jive` with given ranks `method="given"` on `X1` and `X2` and stores the outputs: the joint matrices as `J1.csv` and `J2.csv`, the individual matrices as `A1.csv` and `A2.csv`. It also stores the scaled data used ny `r.jive` as `D1.csv` and `D2.csv`. 
- It fits `r.jive` with `method="perm"` for estimating the ranks on `X1s` and `X2s` and writes the joint matrices  as `J1p.csv` and  `J2p.csv` and the estimated ranks as `ranks_perm.csv`. 
- It records the `R` version and the `r.jive` package versions as `sessionInfo.txt` and `session_meta.csv`. 
- It creates `meta.csv` containing parameters the `R` fixture was generated it. 
