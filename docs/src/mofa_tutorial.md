# MOFA: Multi-Omics Factor Analysis

Multi-Omics Factor Analysis or MOFA is an unsupervised method for finding
low-dimensional structure across several data matrices measured on the same
samples. Each matrix is a **view**, for example gene expression, proteins, or
metabolites. Views may have different numbers of features. Their sample rows
must be aligned before fitting: `mofa` does not join samples by identifier.

The model learns sample factors shared across views and a separate loading
matrix for each view. Some factors may be important in several views, whereas
others may be useful primarily in one. Unlike PCA, MOFA also estimates
uncertainty, noise precisions, and optional shrinkage/sparsity parameters.

In this tutorial we demonstrate `BigRiverEssence.mofa` using simulated Gaussian
measurements with known shared structure, two sample groups, and missing data.

> **Implementation note.** This implementation adapts the Gaussian, full-batch
> variational-inference path of
> [`mofapy2` 0.7.5](https://github.com/bioFAM/mofapy2/tree/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7)
> to Julia, following its initialization, prior distributions, sequential
> updates, sparse-loading moments, ELBO, and convergence criterion. Compact
> group/feature noise storage and reusable vector scratch buffers avoid full
> residual and expanded precision matrices during training. This is not a
> claim of measured speed superiority or complete mofapy2 feature parity.

## The dataset

We simulate 60 samples and two views with 12 and 8 features. Both views depend
on two latent variables, but have different loadings and independent noise.
The groups could represent two study cohorts. Groups partition samples; they
are not the same thing as views, which partition measurement types.

```@example mofa
using BigRiverEssence, Random, LinearAlgebra, Statistics, Plots

Random.seed!(160)
n, k = 60, 2
groups = repeat(["cohort_a", "cohort_b"], 30)
true_z = randn(n, k)
X1 = true_z * randn(k, 12) + 0.1 * randn(n, 12)
X2 = true_z * randn(k, 8) + 0.1 * randn(n, 8)

# Add group-specific offsets, which preprocessing will remove.
X1[groups .== "cohort_b", :] .+= 3.0
X2[groups .== "cohort_b", :] .-= 2.0

# Missing measurements are excluded, not treated as observed zeros.
X1[3, 2] = NaN
X2[5, :] .= NaN
views = [X1, X2]
size.(views)
```

`missing` and `NaN` are supported. Infinite or non-real entries are rejected.
An entire group/feature block may be missing if the feature is observed in
another group. A feature missing in every sample is rejected. A sample missing
across every view can be retained, but its factors are informed only by the
prior, not by measurements.

## The method summary

- Center features within sample groups, with optional scalar view/group scaling.
- Initialize sample factors using PCA, random values, or supplied custom values.
- Alternately update loadings, optional shrinkage and inclusion parameters,
  sample factors, and Gaussian noise precisions.
- Monitor the variational objective (ELBO) and stop after confirmed convergence.
- Return posterior means, uncertainty, preprocessing information, and diagnostics.

For sample group ``g`` and view ``m``, the Gaussian model is

```math
Y^{(g,m)}_{nd} \sim
\mathcal N\!\left(\sum_{j=1}^{K} Z^{(g)}_{nj}W^{(m)}_{dj},\,
\tau_{gmd}^{-1}\right).
```

Here ``Z`` contains sample factors, ``W`` contains feature loadings, and
``\tau`` is noise **precision**, the inverse of noise variance. Only observed
entries contribute to the likelihood. Stored preprocessing zeros at missing
positions are never treated as observations.

Automatic relevance determination (**ARD**) learns a separate precision per
view/factor for loadings and, optionally, per group/factor for sample factors.
Large precision shrinks the corresponding factor's values toward zero.
Spike-and-slab loadings additionally estimate an inclusion probability for each
feature/factor pair. ARD and feature-wise inclusion are different mechanisms.

The solver uses mean-field variational inference: it fits distributions for
the unknown quantities instead of only finding a pair of low-rank matrices.
Expected squared reconstruction error includes posterior uncertainty and is
not just the squared error of the posterior means. Gamma/Beta expected logs
use `digamma` from `SpecialFunctions`.

## Fitting the model

```@example mofa
model = mofa(views; k = 2, groups,
    ard_weights = true, ard_factors = true,
    spikeslab_weights = true, maxiter = 1000)

(model.niter, model.converged, model.elbo[end])
```

The fitted `Mofa` object contains:

- `model.z`: a 60×2 matrix of sample-factor posterior means;
- `model.w[1]` and `model.w[2]`: 12×2 and 8×2 effective loading means;
- `model.posterior`: factor/loading uncertainty, noise distributions, ARD,
  active-slab parameters, and loading inclusion probabilities;
- `model.means`, `model.scales`, and group information: how to undo preprocessing;
- `model.elbo` and `model.elbo_iterations`: objective values and their sweep numbers;
- `model.niter` and `model.converged`: training progress and stopping status; and
- `model.r2` and `model.factor_r2`: total and single-factor explained variation.

The result does not retain copied training matrices, missing masks, or scratch
buffers. `model.z` and `model.w` share their arrays with the corresponding
posterior means; they are not stored twice. Treat these fitted arrays as read
only, since changing means alone would leave cached second moments inconsistent.

## Reconstruction and original units

`model.z * transpose(model.w[m])` reconstructs a view in its **processed** units.
For sparse loadings, `model.w[m]` is already the inclusion-weighted mean; do
not multiply it by inclusion probabilities a second time.

```@example mofa
processed_reconstruction = model.z * transpose(model.w[1])
original_reconstruction = [
    processed_reconstruction[i, j] * model.scales[model.group_indices[i], 1] +
    model.means[1][model.group_indices[i], j]
    for i in axes(X1, 1), j in axes(X1, 2)
]

observed = .!isnan.(X1)
relative_error = norm(original_reconstruction[observed] - X1[observed]) /
                 norm(X1[observed])
relative_error
```

Predictions at missing entries are posterior-mean imputations, not newly
observed data or confidence intervals. Their reliability depends on the
information available in the other measurements and on model assumptions.

```@example mofa
scatter(X1[observed], original_reconstruction[observed];
    xlabel = "Observed value", ylabel = "Reconstructed value",
    title = "MOFA reconstruction of view 1", label = "Measurements",
    markeralpha = 0.6)
```

## Sample factors and sparse loadings

```@example mofa
scatter(model.z[:, 1], model.z[:, 2]; group = groups,
    xlabel = "Factor 1", ylabel = "Factor 2",
    title = "MOFA sample-factor means", markeralpha = 0.7)
```

Group centering removes group-specific feature means, so the factors summarize
variation remaining within groups rather than simply reproducing the offsets.

```@example mofa
inclusion = model.posterior.spike_w[1].inclusion
heatmap(inclusion;
    xlabel = "Factor", ylabel = "Feature in view 1",
    title = "Posterior loading inclusion probabilities", clims = (0, 1))
```

An inclusion probability near one means the loading is likely active under
the fitted approximation. It does not by itself measure effect size or prove
biological importance. Inspect the loading mean and uncertainty as well.

## Explained variation

`model.r2[g,m]` is
``1-\sum_{\mathrm{observed}}(Y-ZW^\top)^2/\sum_{\mathrm{observed}}Y^2``
on the processed data. `model.factor_r2[g,m,j]` uses only factor ``j`` in the
reconstruction. These are descriptive posterior-mean diagnostics, not the ELBO.

```@example mofa
model.r2
```

The factor-wise values need not add up, since factors are not constrained to
be orthogonal. They can be negative. A group/view block with no nonzero
observed processed values has undefined R², reported as `NaN`.

## Training and convergence

The update order is `W → AlphaW → ThetaW → Z → AlphaZ → Tau`, omitting disabled
nodes. `start_sparsity=50` delays **Beta hyperparameter** updates until sweep
50. It does not disable ARD or the individual loading-inclusion updates during
the first 49 sweeps.

```@example mofa
plot(model.elbo_iterations, model.elbo;
    xlabel = "Completed update sweeps", ylabel = "ELBO",
    title = "MOFA variational training", marker = :circle, label = "ELBO")
```

The default checks begin at sweep one and occur every five sweeps. Convergence
requires two consecutive eligible checks whose percentage ELBO change relative
to the initial ELBO falls below the selected threshold: `:fast` = 0.0005,
`:medium` = 0.00005, `:slow` = 0.000005. `miniter` can delay eligibility.

For an equal-work timing comparison, use `forceiter=true`. Julia's `maxiter`
counts actual update sweeps. Python's `iter` includes the initial state, so
Julia `maxiter=100` corresponds to Python `iter=101`. The last Julia objective
is always recorded, even if the last sweep is not a scheduled stopping check.

## Changing the model settings

| Setting | Meaning |
|---|---|
| `k` | Number of factors; not automatically selected or pruned |
| `groups` | Optional sample-group labels; one group by default |
| `center_groups=true` | Remove group/feature means |
| `scale_views=false` | Optional single scale per view, not per-feature standardization |
| `scale_groups=false` | Optional single scale per group/view block |
| `ard_weights=true` | Learn loading shrinkage by view and factor |
| `ard_factors=false` | Learn sample-factor shrinkage by group and factor |
| `spikeslab_weights=true` | Estimate sparse loading inclusions |
| `init=:pca` | PCA initialization; `:random` and `:custom` are alternatives |
| `maxiter=1000` | Maximum actual update sweeps |
| `start_elbo=1`, `freq_elbo=5` | ELBO checking schedule |
| `start_sparsity=50` | First Beta hyperparameter update |
| `miniter=0`, `convergence_mode=:fast` | Convergence eligibility and threshold |
| `forceiter=false`, `verbose=false` | Fixed-work fitting and progress messages |

For an entirely dense model:

```@example mofa
dense_model = mofa(views; k = 2, groups,
    ard_weights = false, ard_factors = false, spikeslab_weights = false,
    maxiter = 20, forceiter = true)
size(dense_model.z)
```

For reproducible random initialization, supply an RNG. Custom initial values
are copied without rescaling; supplying `z_init` is required for `init=:custom`.

```@example mofa
random_model = mofa(views; k = 2, groups, init = :random,
    rng = MersenneTwister(160), maxiter = 20, forceiter = true)

custom_model = mofa(views; k = 2, groups, init = :custom,
    z_init = copy(model.z), w_init = copy.(model.w),
    maxiter = 20, forceiter = true)
custom_model.niter
```

## Validation and limitations

Tests compare complete 30-sweep fits and all ELBO components with the original
Python engine using identical initial factors. They cover all eight combinations
of loading ARD, factor ARD, and sparse loadings, with groups and missing data.
Additional Julia tests check mathematical identities, numerical edge cases,
stopping behavior, preprocessing, reconstruction, and input preservation.

This implementation supports Float64 Gaussian views, dense sample factors,
equal view weights, and full-batch CPU inference. It does not support Bernoulli
or Poisson likelihoods, sparse sample factors, stochastic inference, GPU
execution, MEFISTO, automatic factor dropping, or out-of-sample transformation.
Counts require an appropriate preprocessing/modeling decision before they are
used as Gaussian observations. A factor count larger than the available PCA
rank requires another initialization and careful interpretation.

Factor signs, ordering, and scale are not separately identifiable. Comparing
raw factor entries from independently initialized runs can therefore be
misleading; compare reconstructions, the objective, and aligned factors.
As with other variational latent-factor models, initialization can lead to
different local optima, and variational uncertainty is not exact posterior uncertainty.

## References

1. Argelaguet, R., Velten, B., Arnol, D., et al. (2018).
   *Multi-Omics Factor Analysis—a framework for unsupervised integration of
   multi-omics data sets*. Molecular Systems Biology, 14, e8124.
   [Paper](https://doi.org/10.15252/msb.20178124).
2. Argelaguet, R., Arnol, D., Bredikhin, D., et al. (2020).
   *MOFA+: a statistical framework for comprehensive integration of multi-modal
   single-cell data*. Genome Biology, 21, 111.
   [Paper](https://doi.org/10.1186/s13059-020-02015-1).
3. [mofapy2 reference source](https://github.com/bioFAM/mofapy2/tree/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7):
   `core/BayesNet.py`, `core/nodes/`, and `build_model/init_model.py`.
