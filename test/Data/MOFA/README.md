# MOFA reference fixtures (`generate_mofa_reference.py`)

The `generate_mofa_reference.py` is used in context of testing the outputs of
`mofa` in `BigRiverEssence.jl` against the original Python implementation
`mofapy2`. It produces simulated data and reference outputs used in
`mofa_test.jl` to compare complete fits and individual ELBO contributions.

## It performs the following tasks:

- It loads `mofapy2` and fixes the random seed to 160.
- It simulates two aligned views with two sample groups, missing entries, an
  entirely missing sample, and an entirely missing group/feature block. It
  saves `X1.csv`, `X2.csv`, `groups.csv`, and shared initial factors `Z_init.csv`.
- It uses the original group-centering routine and creates all eight combinations
  of loading ARD, factor ARD, and sparse loading weights.
- It runs the original `BayesNet.iterate` training loop for 30 update sweeps
  (`maxiter=31` in Python), with Beta updates starting at sweep three and ELBO
  checks after every sweep. Initial loadings are zero and variances are one.
- In each case folder it writes all ELBO components and their total as
  `elbo.csv`, posterior parameters as `posterior.csv`, and total/single-factor
  explained variation as `r2.csv` and `factor_r2.csv`.
- It also runs early-stopping fits with five-sweep ELBO intervals and saves
  their actual update counts, convergence flags, and final ELBO in `stopping.csv`.
- It records Python and package versions in `session_info.txt` and settings in
  `meta.csv`. No Julia code is used to generate the reference outputs.

The source is mofapy2 0.7.5 at
[`90418e5`](https://github.com/bioFAM/mofapy2/tree/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7).
No reference algorithm was modified. Normal package tests read the saved files
and do not require Python. To regenerate from this directory in a separate
Python environment:

```sh
python -m pip install -r requirements.txt
python generate_mofa_reference.py
```

`elbo.csv` columns are iteration, Y, W, Z, Tau, AlphaW, AlphaZ, ThetaW, total;
disabled nodes contribute zero. `posterior.csv` concatenates arrays in
column-major order: Z mean/variance; each view's W mean/second moment and Tau
shape/rate; each view's AlphaW shape/rate if enabled; AlphaZ shape/rate if
enabled; each view's inclusion probabilities and Beta a/b if enabled.
`factor_r2.csv` flattens a group×view×factor array in column-major order.

Identical custom initial factors avoid RNG and PCA sign differences between
languages. Checks compare factor means, uncertainty, hyperparameters,
reconstructions, and ELBOs rather than expecting bitwise identical floats.
