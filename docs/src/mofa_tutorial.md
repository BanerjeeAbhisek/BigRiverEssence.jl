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

In this tutorial we demonstrate `BigRiverEssence.mofa` using real measurements
from 200 patients with chronic lymphocytic leukaemia (CLL). We integrate drug
response, RNA expression, and DNA methylation, retaining the naturally missing
measurements. This is the CLL cohort used in the official
[Python MOFA tutorial](https://muon-tutorials.readthedocs.io/en/latest/CLL.html)
and the original MOFA application [1,4].

> **Implementation note.** This implementation adapts the Gaussian, full-batch
> variational-inference path of
> [`mofapy2` 0.7.5](https://github.com/bioFAM/mofapy2/tree/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7)
> to Julia, following its initialization, prior distributions, sequential
> updates, sparse-loading moments, ELBO, and convergence criterion. Compact
> group/feature noise storage and reusable scratch buffers avoid full
> residual and expanded precision matrices during training.

Small-rank predictions use direct SIMD loops. From eight factors onward,
updates reuse a prefix/suffix-sum cache of size `max(n, sum(pᵢ)) × (k+1)`:
updated earlier factors form the prefix, and original later factors form the
suffix. This excludes the current factor without subtracting its old
contribution from a large residual. The cache is allocated once per fit and
discarded afterwards; update order and convergence checks remain unchanged.

Large problems (`n * sum(pᵢ) * k ≥ 10^7`, with at least eight factors) also
split independent loading rows and sample rows between Julia worker tasks.
The number of workers is limited to the available Julia threads, sample/feature
counts, and eight workers. Each worker has its own prefix/suffix and coefficient
scratch; the first reuses the serial buffers. No reconstruction-sized matrix is
introduced, and no floating-point reduction crosses workers. The coordinates
within each row are still updated sequentially, so worker partitioning does not
change their arithmetic. Small problems and single-threaded Julia use the
serial path. Additional workers trade a little extra scratch memory and task
allocation for speed; this is not an allocation-free parallel implementation.

To make four Julia threads available, start Julia with `julia --threads=4`
(or set `JULIA_NUM_THREADS=4` before starting Julia). Changing BLAS thread counts
alone does not enable these Julia worker tasks. The solver does not change
either global thread setting. Noise and diagnostic calculations remain serial.

## The dataset

We use the processed `CLL_data` matrices distributed by
[`MOFAdata`](https://github.com/bioFAM/MOFAdata) [5]. The upstream dataset has
four views: drug response, RNA expression, DNA methylation, and somatic
mutations. The Python tutorial uses Gaussian likelihoods for the first three
and a Bernoulli likelihood for mutations. We omit the mutation view because
our current implementation supports Gaussian observations only.

For a compact, runnable example, we retain all 200 patients and all 310
drug/concentration features, together with the 500 most variable RNA genes
and 500 most variable methylation sites. Variance is calculated from observed
values, with feature names breaking ties. This is an illustrative subset,
not a reproduction of the published full-data, 15-factor analysis. The RNA
values are already normalized and methylation is supplied as M-values; we
do not treat raw RNA counts or binary mutations as Gaussian observations.

The matrices are stored locally in `reference_Data/clldata`, just like the
real datasets used by the other tutorials. They have already been transposed
and aligned by patient ID: rows are patients and columns are features. No
imputation, artificial missingness, or simulated values are introduced. The
one-time export script and its provenance are described in that folder's
README; building this tutorial requires neither R/Python nor a download.

```@example mofa
using BigRiverEssence, DelimitedFiles, DataFrames
using Random, LinearAlgebra, Statistics, Plots

datadir = joinpath(pkgdir(BigRiverEssence), "reference_Data", "clldata")
X1, drug_names = readdlm(joinpath(datadir, "drugs.csv"), ',', Float64; header = true)
X2, gene_names = readdlm(joinpath(datadir, "rna.csv"), ',', Float64; header = true)
X3, cpg_names = readdlm(joinpath(datadir, "methylation.csv"), ',', Float64; header = true)
sample_table, _ = readdlm(joinpath(datadir, "samples.csv"), ','; header = true)

sample_ids = String.(sample_table[:, 1])
sex = String.(sample_table[:, 2])
groups = String.(sample_table[:, 3])   # one cohort: "CLL"
views = [X1, X2, X3]
view_names = ["Drugs", "RNA", "Methylation"]
feature_names = [vec(String.(drug_names)), vec(String.(gene_names)), vec(String.(cpg_names))]
n, k = length(sample_ids), 5
@assert length(unique(sample_ids)) == n
@assert all(size(X, 1) == n for X in views)

overview = DataFrame(
    view = view_names,
    patients = [size(X, 1) for X in views],
    features = [size(X, 2) for X in views],
    observed_patients = [count(i -> any(!isnan, X[i, :]), axes(X, 1)) for X in views],
    missing_percent = [100 * count(isnan, X) / length(X) for X in views],
)
overview
```

The drug, RNA, and methylation views have observations for 184, 136, and 196
patients respectively. All matrices retain 200 rows; a patient absent from a
view has `NaN` in that entire row, not zero. The missing-entry proportions are
8%, 32%, and 2% respectively.

```@example mofa
bar(view_names, overview.missing_percent;
    xlabel = "View", ylabel = "Missing measurements (%)",
    title = "Naturally missing measurements in the CLL data", legend = false)
```

Here all patients belong to a single sample group, `"CLL"`. Views partition
measurement types; groups would partition patients into cohorts or batches.
We do not invent groups or use sex as a centering group: sex is only used to
color a later plot and is never supplied as a predictor.

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

We fit five factors and scale each view by its overall population standard
deviation (`scale_views=true`), since the assays have different units. This
is view-wise scaling, not gene-by-gene standardization. Loading ARD and sparse
loadings are enabled; factor ARD is left off for this single-group example.
The default PCA initialization temporarily fills missing entries with zero in
the centered working matrix. This does not replace missing observations in
the likelihood: their masks remain in place throughout MOFA fitting.

```@example mofa
model = mofa(views; k, groups, scale_views = true,
    ard_weights = true, ard_factors = false,
    spikeslab_weights = true, maxiter = 1000)

(model.niter, model.converged, model.elbo[end])
```

The fitted `Mofa` object contains:

- `model.z`: a 200×5 matrix of sample-factor posterior means;
- `model.w[1]`, `model.w[2]`, and `model.w[3]`: 310×5, 500×5, and 500×5
  effective loading means for drugs, RNA, and methylation;
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
# Thin the plotted points only; relative_error above uses every observed entry.
observed_indices = findall(observed)
stride = max(1, cld(length(observed_indices), 2000))
plot_indices = observed_indices[1:stride:end]
lower, upper = extrema(X1[observed])
plt = scatter(X1[plot_indices], original_reconstruction[plot_indices];
    xlabel = "Observed value", ylabel = "Reconstructed value",
    title = "CLL drug-response reconstruction", label = "Measurements",
    markeralpha = 0.6)
plot!(plt, [lower, upper], [lower, upper]; label = "Exact reconstruction",
    color = :black, linestyle = :dash)
plt
```

Unlike the simulated low-rank case, real measurements contain substantial
noise and view-specific variation. Five factors need not reconstruct every
entry closely. This plot describes an in-sample fit, not held-out predictive
accuracy or clinical validation; Gaussian predictions are not constrained to
the original assay's range.

## Sample factors and sparse loadings

```@example mofa
scatter(model.z[:, 1], model.z[:, 2]; group = sex,
    xlabel = "Factor 1", ylabel = "Factor 2",
    title = "CLL patient factors", legendtitle = "Sex (metadata)", markeralpha = 0.7)
```

Each point is a real patient. The metadata colors do not influence fitting,
and no association with sex is assumed. Factor signs and ordering are not
identified; factor 1 here is not necessarily factor 1 in the published model.
With one group, centering removes each feature's cohort mean.

```@example mofa
inclusion = model.posterior.spike_w[1].inclusion
heatmap(inclusion;
    xlabel = "Factor", ylabel = "Drug/concentration feature", yticks = false,
    title = "CLL drug loading inclusion probabilities", clims = (0, 1),
    left_margin = 10 * Plots.mm)
```

An inclusion probability near one means the loading is likely active under
the fitted approximation. It does not by itself measure effect size or prove
biological importance. Inspect the loading mean and uncertainty as well.

To see which recorded features drive a factor, inspect its largest absolute
loadings. We use the source feature IDs, rather than assigning biological
names or drug identities that are not present in the exported data.

```@example mofa
view_index, factor_index = 2, 1   # RNA loadings for factor 1
top = sortperm(abs.(model.w[view_index][:, factor_index]); rev = true)[1:10]
top_loadings = DataFrame(
    feature = feature_names[view_index][top],
    loading = model.w[view_index][top, factor_index],
    inclusion_probability = model.posterior.spike_w[view_index].inclusion[top, factor_index],
)
top_loadings
```

## Explained variation

`model.r2[g,m]` is
``1-\sum_{\mathrm{observed}}(Y-ZW^\top)^2/\sum_{\mathrm{observed}}Y^2``
on the processed data. `model.factor_r2[g,m,j]` uses only factor ``j`` in the
reconstruction. These are descriptive posterior-mean diagnostics, not the ELBO.

```@example mofa
DataFrame(view = view_names, variance_explained_percent = 100 .* vec(model.r2))
```

For this subset, the five-factor fit explains approximately 50.5% of drug-response
variation, 21.5% of RNA variation, and 32.1% of methylation variation. These
values describe this in-sample, three-view subset and preprocessing choice;
they are not the results reported for the original full-data model.

```@example mofa
factor_r2 = 100 .* permutedims(model.factor_r2[1, :, :])
heatmap(view_names, 1:k, factor_r2;
    xlabel = "View", ylabel = "Factor",
    title = "CLL variance explained by each factor", colorbar_title = "%")
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
dense_model = mofa(views; k, groups, scale_views = true,
    ard_weights = false, ard_factors = false, spikeslab_weights = false,
    maxiter = 20, forceiter = true)
size(dense_model.z)
```

For reproducible random initialization, supply an RNG. Custom initial values
are copied without rescaling; supplying `z_init` is required for `init=:custom`.

```@example mofa
random_model = mofa(views; k, groups, scale_views = true, init = :random,
    rng = MersenneTwister(160), maxiter = 20, forceiter = true)

custom_model = mofa(views; k, groups, scale_views = true, init = :custom,
    z_init = copy(model.z), w_init = copy.(model.w),
    maxiter = 20, forceiter = true)
custom_model.niter
```

The 20-sweep alternative fits above illustrate the API; they are not used to
rank initializations or claim convergence. Keep preprocessing and model options
matched when comparing complete fits. To warm-start both mean matrices, supply
`w_init` as well as `z_init`. Variances and hyperparameters are initialized
again: this is not a continuation of the full fitted variational state.

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
4. Dietrich, S., Oleś, M., Lu, J., et al. (2018).
   *Drug-perturbation-based stratification of blood cancer*.
   Journal of Clinical Investigation, 128(1), 427–445.
   [Paper](https://doi.org/10.1172/JCI93801).
5. [MOFAdata CLL dataset](https://github.com/bioFAM/MOFAdata): processed data
   distributed for MOFA tutorials. See also the
   [Python CLL analysis](https://muon-tutorials.readthedocs.io/en/latest/CLL.html)
   and the [MOFA2 CLL vignette](https://github.com/bioFAM/MOFA2_tutorials/blob/master/R_tutorials/CLL.Rmd).
