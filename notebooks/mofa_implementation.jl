# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# + [markdown]
# # MOFA implementation: theory and the mofapy2 algorithm
#
# This notebook is the background and algorithm specification for adding
# Multi-Omics Factor Analysis (MOFA) to BigRiverEssence.jl. It explains the
# statistical model first, then connects its quantities to the implementation
# in Python's **mofapy2 0.7.5**.
#
# This is a reading notebook. All cells are explanatory Markdown, equations,
# or pseudocode; it does not fit a model or install dependencies. The eventual
# Julia implementation, tests, and method tutorial will be separate work.
#
# The aim is to preserve the selected MOFA model and inference procedure while
# reducing computational cost, peak memory, and unnecessary allocations.
# Performance improvements will need measurements against the same Python
# configuration and comparable fit quality.
#
# Read the four sections in order:
#
# 1. [What is MOFA](#what-is-mofa): the problem, inputs, and outputs.
# 2. [Reference papers](#reference-papers): where the model and algorithm come from.
# 3. [Method](#method): the theory and mathematical quantities.
# 4. [Algorithm](#algorithm): the actual mofapy2 training procedure.
# -

# + [markdown]
# <a id="what-is-mofa"></a>
# ## What is MOFA
#
# MOFA is an **unsupervised probabilistic factor model** for learning the main
# sources of variation in several datasets measured on a common set of samples.
# Each dataset is called a **view**. Views can contain different features and
# different numbers of features.
#
# For example, consider the following illustrative dataset:
#
# | View | Samples | Features | Matrix shape |
# |---|---:|---:|---|
# | RNA expression | 100 patients | 5,000 genes | $100\times5{,}000$ |
# | DNA methylation | The same 100 patients | 3,000 CpG sites | $100\times3{,}000$ |
# | Protein abundance | The same 100 patients | 200 proteins | $100\times200$ |
#
# MOFA might summarize these matrices using five **latent factors**. Every
# patient receives five factor values, and every feature receives five loadings
# describing its association with those factors. The patient labels or outcomes
# are not needed to learn the factors.
#
# A factor is initially a numerical pattern, not a known biological mechanism.
# We interpret it afterward using the samples, loadings, explained variation,
# and available biological annotations.
# -

# + [markdown]
# ### Shared, partially shared, and view-specific patterns
#
# A factor can be active in different combinations of views:
#
# | Illustrative factor | RNA | Methylation | Protein | Interpretation |
# |---|---|---|---|---|
# | Factor 1 | Active | Active | Active | Shared across all views |
# | Factor 2 | Active | Inactive | Active | Shared across a subset of views |
# | Factor 3 | Inactive | Active | Inactive | Specific to methylation |
#
# Here, "active" means that the factor has loadings and explains appreciable
# variation in that view. It is not a hard biological label supplied as input.
#
# Some entries, or an entire assay for some patients, can be missing. The
# samples must still be aligned across views; MOFA does not automatically match
# unrelated samples or discover which patient a row belongs to.
#
# **MOFA+** also handles **groups**, which partition the samples. For example,
# patients from two studies can form two groups, each with the same collection
# of views and a consistent feature definition within each view. This lets the
# model describe factor activity across both views and sample groups.
# -

# + [markdown]
# ### What we obtain from a fitted model
#
# | Result | What it tells us |
# |---|---|
# | Factor means | Each sample's low-dimensional representation |
# | Loading means per view | Which features are associated with each factor |
# | Posterior variances | Uncertainty represented by the variational approximation |
# | Inclusion probabilities, when enabled | How likely an individual loading or factor value is to be active |
# | Noise parameters | How much unexplained variation remains per feature and group |
# | Explained variation | Where each factor contributes across views and groups |
# | ELBO history and convergence information | How training progressed |
# | Reconstructed values | Denoised or predicted measurements, including missing entries |
#
# Explained variation is calculated from the fitted model; it is not the
# variational training objective. Factor signs and ordering can change between
# fits. Comparisons should account for these ambiguities and examine
# reconstructions, matched factors, and model fit as appropriate.
# -

# + [markdown]
# <a id="reference-papers"></a>
# ## Reference papers
#
# ### The two main papers
#
# **[1] Argelaguet, R., Velten, B., Arnol, D., et al. (2018).**
# *Multi-Omics Factor Analysis—a framework for unsupervised integration of
# multi-omics data sets.* Molecular Systems Biology, 14, e8124.
# [Paper and DOI](https://doi.org/10.15252/msb.20178124).
#
# This introduces the original multi-view model, view-wise ARD, sparse
# loadings, mixed likelihoods, missing-data handling, and variational training.
# Its **Materials and Methods** and **Appendix, Sections 2–5** are especially
# relevant to implementation.
#
# **[2] Argelaguet, R., Arnol, D., Bredikhin, D., et al. (2020).**
# *MOFA+: a statistical framework for comprehensive integration of multi-modal
# single-cell data.* Genome Biology, 21, 111.
# [Paper and DOI](https://doi.org/10.1186/s13059-020-02015-1).
#
# This extends the framework to multiple sample groups and scalable inference.
# The **Methods** and **Additional file 2: Supplementary Methods** explain the
# model, priors, variational updates, and stochastic inference.
#
# | Technical supplement | Sections to read |
# |---|---|
# | [Original MOFA appendix](https://media.springernature.com/original/springer-static/esm/art%3A10.15252%2Fmsb.20178124/MediaObjects/44320_2018_BFMSB178124_MOESM1_ESM.pdf) | 2: model; 3: Gaussian inference; 4: non-Gaussian inference; 5: training considerations |
# | [MOFA+ supplementary methods](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13059-020-02015-1/MediaObjects/13059_2020_2015_MOESM2_ESM.pdf) | 2.1: group/factor regularization; 2.2.1: VI; 2.2.3: update equations; 2.2.4: SVI |
# -

# + [markdown]
# ### Software reference and terminology
#
# **MOFA** is the method family; **MOFA+** is the extended model; **MOFA2** is
# the R package; **mofapy2** is the Python training engine. R MOFA2 calls that
# Python engine. The old Python package named mofapy corresponds to the earlier
# generation; our reference here is mofapy2.
#
# Sources:
#
# - [Official MOFA2 documentation](https://biofam.github.io/MOFA2/).
# - [Installation and the R/Python relationship](https://biofam.github.io/MOFA2/installation.html).
# - [mofapy2 source repository](https://github.com/bioFAM/mofapy2).
# - [Reference source snapshot for this notebook, version 0.7.5](https://github.com/bioFAM/mofapy2/tree/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7).
#
# The papers define the statistical ideas. The software snapshot defines the
# initialization, defaults, update order, and stopping behavior described below.
# These details should be recorded explicitly when comparing implementations.
# -

# + [markdown]
# <a id="method"></a>
# ## Method
#
# ### Notation and matrix orientation
#
# We use the papers' observations-by-features convention throughout:
#
# | Symbol | Meaning | Dimensions |
# |---|---|---|
# | $M$ | Number of views | Scalar |
# | $G$ | Number of sample groups | Scalar |
# | $N_g$ | Number of samples in group $g$ | Scalar |
# | $D_m$ | Number of features in view $m$ | Scalar |
# | $K$ | Number of factors | Scalar |
# | $Y^{(g,m)}$ | Data for group $g$ and view $m$ | $N_g\times D_m$ |
# | $Z^{(g)}$ | Sample factors for group $g$ | $N_g\times K$ |
# | $W^{(m)}$ | Feature loadings for view $m$ | $D_m\times K$ |
# | $\tau_{gmd}$ | Gaussian noise precision | One value per group, view, and feature |
# | $O^{(g,m)}_{nd}$ | Observation indicator: 1 observed, 0 missing | $N_g\times D_m$ |
#
# Groups partition samples. Within a view, feature identities must match across
# groups. The features in two different views do not need to match.
#
# Python stores groups concatenated along the sample axis of each view matrix;
# $g(n)$ identifies the group of row $n$. This is equivalent to the block
# notation used here.
# -

# + [markdown]
# ### The generative model
#
# For centered Gaussian data, the basic model is
#
# $$
# Y^{(g,m)}=Z^{(g)}(W^{(m)})^\top+E^{(g,m)}.
# $$
#
# Equivalently, for an observed entry,
#
# $$
# y^{(g,m)}_{nd}\mid Z,W,\tau
# \sim\mathcal N\!\left(
# \sum_{k=1}^{K}z^{(g)}_{nk}w^{(m)}_{dk},
# \tau_{gmd}^{-1}
# \right).
# $$
#
# A **precision** is the inverse of a variance: large $\tau_{gmd}$ means small
# residual variance. Different features and groups may have different noise
# levels.
#
# A factor column $z^{(g)}_{:k}$ is reused across views in that group. A loading
# column $w^{(m)}_{:k}$ is reused across groups for that view. This is how the
# model connects the blocks.
#
# If preprocessing removed feature means or scaled views, reconstruction is
# initially in those processed units. Original-unit predictions must undo the
# recorded preprocessing.
#
# Sources: [1], Materials and Methods; [2], Methods, model equation.
# -

# + [markdown]
# ### Priors: controlling which factors and features matter
#
# **ARD on weights.** For a non-spike loading with ARD,
#
# $$
# w^{(m)}_{dk}\mid\alpha^W_{mk}
# \sim\mathcal N(0,(\alpha^W_{mk})^{-1}),
# \qquad
# \alpha^W_{mk}\sim\operatorname{Gamma}(a^W_0,b^W_0).
# $$
#
# The same precision $\alpha^W_{mk}$ controls all features in a loading column.
# Large precision encourages that factor to have small weights in that view.
#
# **Spike-and-slab on weights.** Introduce an inclusion variable and an
# underlying Gaussian weight:
#
# $$
# w^{(m)}_{dk}=s^{W,(m)}_{dk}\widehat w^{(m)}_{dk},\qquad
# s^{W,(m)}_{dk}\sim\operatorname{Bernoulli}(\theta^W_{mk}),
# $$
# $$
# \widehat w^{(m)}_{dk}\mid\alpha^W_{mk}
# \sim\mathcal N(0,(\alpha^W_{mk})^{-1}),\qquad
# \theta^W_{mk}\sim\operatorname{Beta}(a^\theta_0,b^\theta_0).
# $$
#
# The product here is entrywise. If $s=0$, the effective loading is zero.
# ARD regulates a whole factor/view combination; spike-and-slab provides
# feature-level selection within it.
#
# MOFA+ offers corresponding choices for sample factors:
# $\alpha^Z_{gk}$ controls factor activity within group $g$, while an optional
# inclusion variable can select individual sample-factor values.
#
# These are configurable priors. A Gaussian prior alone does not produce
# exact-zero point mass, and enabling ARD does not imply that spike-and-slab
# must also be enabled.
#
# Sources: [1], Appendix, Section 2; [2], Supplementary Methods, Section 2.1.
# -

# + [markdown]
# ### Missing entries and preprocessing
#
# Only observed values contribute to the likelihood:
#
# $$
# \log p(Y\mid Z,W,\tau)
# =\sum_{g,m,n,d:\,O^{(g,m)}_{nd}=1}
# \log p(y^{(g,m)}_{nd}\mid Z,W,\tau).
# $$
#
# A missing value therefore contributes neither an error term nor an observation
# count. Replacing NaNs by zero is safe only when an accompanying mask removes
# them from every relevant calculation.
#
# mofapy2 uses a Boolean mask with **True meaning missing**, the opposite of
# our mathematical $O$ convention. Its Gaussian data node stores zeros in
# missing positions and retains the mask.
#
# For Gaussian data, the Python default centers each feature within each
# group. Optional view scaling divides by the overall standard deviation of
# the view; optional group scaling uses a scalar standard deviation for that
# group/view block. These are not per-feature unit-variance scaling.
#
# Initialization is a separate issue: mofapy2 can mean-impute a temporary
# concatenated matrix for PCA initialization. Those temporary values do not
# become observations in the fitted likelihood.
#
# Sources: [process_data](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/utils.py),
# [Y_Node](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Y_nodes.py),
# and [init_model](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/init_model.py).
# -

# + [markdown]
# ### Variational inference and the training objective
#
# Let $\Theta$ denote the unobserved factors, loadings, noise precisions, and
# enabled regularization quantities. The posterior $p(\Theta\mid Y)$ is
# approximated by a tractable distribution $q(\Theta)$.
#
# The training objective is the evidence lower bound:
#
# $$
# \mathcal L(q)=
# \mathbb E_q[\log p(Y,\Theta)]-\mathbb E_q[\log q(\Theta)]
# =\log p(Y)-\operatorname{KL}(q(\Theta)\Vert p(\Theta\mid Y)).
# $$
#
# Increasing the ELBO improves this approximation within the chosen family.
# The coordinate update for one variational block is
#
# $$
# \log q_i^*(\Theta_i)
# =\mathbb E_{q_{-i}}[\log p(Y,\Theta)]+\text{constant}.
# $$
#
# Update one block using the current others, then continue through the schedule.
# The spike variable and its underlying Gaussian variable are treated jointly
# at each coordinate; they are not assumed independent of one another.
#
# The implementation stores **means and second moments**, not just factor
# matrices. For a non-spike Gaussian with mean $\mu$ and variance $v$,
#
# $$
# \mathbb E[x]=\mu,\qquad \mathbb E[x^2]=\mu^2+v.
# $$
#
# Replacing $\mathbb E[x^2]$ by $\mathbb E[x]^2$ would remove uncertainty terms
# and change the algorithm.
#
# Sources: [1], Appendix, Section 3; [2], Supplementary Methods, Section 2.2.
# -

# + [markdown]
# ### Moments for spike-and-slab variables
#
# Let $x=s\widehat x$, $\gamma=q(s=1)$, and let the active Gaussian have
# conditional mean $\mu$ and variance $v$. Then
#
# $$
# \mathbb E[x]=\gamma\mu,\qquad
# \mathbb E[x^2]=\gamma(\mu^2+v).
# $$
#
# The underlying, ungated Gaussian has a different second moment:
#
# $$
# \mathbb E[\widehat x^2]
# =\gamma(\mu^2+v)+(1-\gamma)v_0,
# $$
#
# where $v_0$ is its variance conditional on being inactive; in a full-batch
# update with zero prior mean, $v_0=1/\mathbb E[\alpha]$.
#
# Python distinguishes these quantities explicitly:
#
# | Python expectation key | Mathematical quantity | Example use |
# |---|---|---|
# | E | $\mathbb E[s\widehat x]$ | Predicted signal |
# | E2 | $\mathbb E[(s\widehat x)^2]$ | Factor/loading updates and noise |
# | EB | $\mathbb E[s]=\gamma$ | Inclusion-probability hyperparameters |
# | EN | $\mathbb E[\widehat x\mid s=1]=\mu$ | Active Gaussian mean |
# | ENN | $\mathbb E[\widehat x^2]$ | ARD update |
#
# Confusing E2 with ENN would change the shrinkage behavior.
#
# Source: [BernoulliGaussian.updateExpectations](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/distributions/bernoulli_gaussian.py).
# -

# + [markdown]
# ### Gaussian coordinate updates: the quantities we must preserve
#
# We first write the non-spike, zero-prior-mean, full-batch equations with
# view weighting disabled. Bars denote posterior means, superscript $(2)$
# denotes a second moment, and $A^W_{mk}$ and $A^Z_{gk}$ are expected prior
# precisions. When ARD is disabled in this baseline, the prior precision is 1.
#
# Define the expected residual excluding component $k$:
#
# $$
# r^{(-k)}_{gmnd}
# =y^{(g,m)}_{nd}-\sum_{j\ne k}\bar z^{(g)}_{nj}\bar w^{(m)}_{dj}.
# $$
#
# For a loading coordinate,
#
# $$
# v^W_{mdk}
# =\left(A^W_{mk}+\sum_{g,n}O_{gmnd}\bar\tau_{gmd}z^{(2)}_{gnk}\right)^{-1},
# $$
# $$
# \bar w_{mdk}
# =v^W_{mdk}\sum_{g,n}
# O_{gmnd}\bar\tau_{gmd}\bar z_{gnk}r^{(-k)}_{gmnd}.
# $$
#
# For a sample-factor coordinate,
#
# $$
# v^Z_{gnk}
# =\left(A^Z_{gk}+\sum_{m,d}O_{gmnd}\bar\tau_{gmd}w^{(2)}_{mdk}\right)^{-1},
# $$
# $$
# \bar z_{gnk}
# =v^Z_{gnk}\sum_{m,d}
# O_{gmnd}\bar\tau_{gmd}\bar w_{mdk}r^{(-k)}_{gmnd}.
# $$
#
# Each variance denominator includes uncertainty through the second moments.
# Within a node, coordinates are updated in sequence: later coordinates use the
# newly updated means of earlier coordinates.
#
# Sources: [W_Node](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/W_nodes.py),
# [Z_Node](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Z_nodes.py),
# and [1], Appendix, Section 3.2.
# -

# + [markdown]
# ### Extending a coordinate to spike-and-slab
#
# For a loading coordinate, write the data contribution to its precision as
# $c=\sum O\bar\tau z^{(2)}$ and its residual projection as
# $b=\sum O\bar\tau\bar z r^{(-k)}$. In a full-batch sparse-weight update,
#
# $$
# v=(A+c)^{-1},\qquad \mu=vb,
# $$
# $$
# \gamma=\operatorname{sigmoid}\!\left(
# \mathbb E[\log\theta]-\mathbb E[\log(1-\theta)]+\tfrac12\log(Av)+\tfrac12\mu^2/v
# \right).
# $$
#
# The effective loading mean is $\gamma\mu$, rather than $\mu$. The sample-side
# sparse update has the analogous structure with sample/group quantities.
#
# For ARD on weights, using Gamma **shape–rate** parameters,
#
# $$
# a^W_{mk}=a^W_0+D_m/2,\qquad
# b^W_{mk}=b^W_0+\tfrac12\sum_d\mathbb E[\widehat w_{mdk}^2].
# $$
#
# For non-spike weights, the underlying and effective weights coincide.
# For spike-and-slab inclusion hyperparameters,
#
# $$
# a^\theta_{mk}=a^\theta_0+\sum_d\gamma_{mdk},\qquad
# b^\theta_{mk}=b^\theta_0+D_m-\sum_d\gamma_{mdk}.
# $$
#
# Corresponding group-wise updates sum over the samples of group $g$.
# The expected Gamma precision is $a/b$; the expected log precision is
# $\psi(a)-\log b$. These are distinct quantities.
#
# Sources: [SW_Node](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/W_nodes.py),
# [Alpha nodes](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Alpha_nodes.py),
# and [Theta nodes](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Theta_nodes.py).
# -

# + [markdown]
# ### Noise updates and the expected reconstruction error
#
# For a feature in a group/view block, the Gamma posterior has
#
# $$
# a^\tau_{gmd}=a^\tau_0+\tfrac12\sum_n O_{gmnd},\qquad
# b^\tau_{gmd}=b^\tau_0+
# \tfrac12\sum_n O_{gmnd}\,
# \mathbb E\!\left[\left(y_{gmnd}-\sum_k z_{gnk}w_{mdk}\right)^2\right].
# $$
#
# Under the coordinate-wise variational factorization, the expected squared
# error for an entry is
#
# $$
# \left(y_{gmnd}-\sum_k\bar z_{gnk}\bar w_{mdk}\right)^2+\sum_k\left(
# z^{(2)}_{gnk}w^{(2)}_{mdk}-\bar z_{gnk}^{\,2}\bar w_{mdk}^{\,2}\right).
# $$
#
# The second term accounts for posterior uncertainty. Using only the residual
# of the posterior means would underestimate this expected error.
#
# The Gaussian likelihood part of the ELBO uses this same expected error,
# together with $\mathbb E[\log\tau]$, not $\log\mathbb E[\tau]$.
#
# Source: [TauD_Node](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Tau_nodes.py)
# and [Y_Node.calculateELBO](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Y_nodes.py).
# -

# + [markdown]
# ### Other likelihoods, interpretation, and limitations
#
# The additive residual equation above describes the Gaussian case. For a
# non-Gaussian view, the same linear predictor
# $\eta_{gmnd}=\sum_k z_{gnk}w_{mdk}$ enters a different likelihood:
#
# | Data type | Likelihood in the reference formulation |
# |---|---|
# | Continuous | $\mathcal N(\eta,\tau^{-1})$ |
# | Binary | $\operatorname{Bernoulli}(\operatorname{sigmoid}(\eta))$ |
# | Counts | $\operatorname{Poisson}(\operatorname{softplus}(\eta))$ |
#
# Here $\operatorname{softplus}(x)=\log(1+e^x)$. The Poisson rate in this
# formulation is not the conventional exponential link. The Python
# non-Gaussian nodes use variational bounds and pseudo-observations to reuse
# Gaussian-style updates. Extra quantities and additional memory are involved;
# selecting a different distribution alone is not a complete implementation.
#
# MOFA remains a linear latent-factor model. It is not a clustering algorithm,
# does not impose PCA-style orthogonal factors, and does not automatically
# identify a factor with a biological cause. Group-wise centering also means
# that multi-group activity concerns within-group variation, rather than simply
# separating group means.
#
# Sources: [1], Appendix, Section 4;
# [non-Gaussian nodes](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/nongaussian_nodes.py);
# [official FAQ](https://biofam.github.io/MOFA2/faq.html).
# -

# + [markdown]
# <a id="algorithm"></a>
# ## Algorithm
#
# ### What "following mofapy2" means
#
# We will use the **coordinate-wise mean-field variational inference implemented
# by mofapy2**, beginning with the Gaussian full-batch path. The preceding
# sections also explain the broader MOFA+ model so that later options fit the
# same framework.
#
# The Python configuration determines which nodes exist:
#
# | Python node | Meaning |
# |---|---|
# | Y | Observations; Gaussian values stay fixed, non-Gaussian pseudo-data can update |
# | W | Effective loading distributions, one block per view |
# | AlphaW | Optional ARD precisions for weights |
# | ThetaW | Optional loading-inclusion probability hyperparameters |
# | Z | Effective sample-factor distributions |
# | AlphaZ | Optional ARD precisions for sample factors |
# | ThetaZ | Optional sample-factor inclusion hyperparameters |
# | Tau | Gaussian noise-precision distributions |
#
# A node update refreshes both its variational parameters and the expectations
# consumed by later nodes. Julia can use a more compact representation without
# reproducing Python's class hierarchy.
# -

# + [markdown]
# ### Step 1: prepare the data and configuration
#
# 1. Align sample identifiers, view features, group labels, and missing masks.
# 2. Select a likelihood for each view explicitly for a reproducible reference.
# 3. Apply the selected Gaussian centering and scalar scaling.
# 4. Record $K$, enabled ARD/spike-and-slab priors, precision, seed, iteration
#    controls, and optional factor-dropping threshold.
# 5. Construct the appropriate variational nodes.
#
# The following are defaults of the inspected Python low-level entry point,
# not a final promise about the future BigRiverEssence API:
#
# | Category | Python defaults |
# |---|---|
# | Model | factors=10; ard_weights=True; spikeslab_weights=True; ard_factors=False; spikeslab_factors=False |
# | Gaussian preprocessing | center_groups=True; scale_views=False; scale_groups=False |
# | Numeric storage option | use_float32=False |
# | Training | iter=1000; startELBO=1; freqELBO=5; startSparsity=50 |
# | Factor dropping | dropR2=None, meaning disabled |
# | Convergence | convergence_mode="fast"; nostop=False |
# | Execution | gpu_mode=False; full-batch inference unless stochastic options are set |
#
# ARD on factors can be enabled for multiple groups; it is not silently enabled
# by these low-level defaults. The code prints a recommendation when multiple
# groups are used without it.
#
# Source: [entry_point option setters](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/run/entry_point.py).
# -

# + [markdown]
# ### Step 2: initialize factors, loadings, and hyperparameters
#
# The standard model builder initializes Z from PCA of the concatenated views,
# using a temporarily mean-imputed copy if necessary. For the non-spike Z path,
# the PCA scores are subsequently rescaled to the interval [-1, 1] per factor.
# The sparse SZ initialization follows a separate path and does not perform
# that same rescaling in this reference snapshot.
#
# The builder initializes loading means to zero. Initial variational
# variances are positive (1 in these standard paths), and spike-and-slab
# inclusion probabilities initially equal 1.
#
# The reference initialization uses:
#
# | Quantity | Prior or initial setting |
# |---|---|
# | Standard non-ARD Gaussian Z/W | Prior mean 0, variance 1 |
# | ARD precision | Gamma prior shape=rate=$10^{-14}$; initial q shape=rate=1 |
# | Gaussian noise precision | Gamma prior shape=rate=$10^{-3}$; initial q shape=rate=1 |
# | Inclusion hyperparameter | Beta prior a=b=1; initial q a=b=1 |
#
# The initial inclusion expectations and the Beta hyperparameter distribution
# are separate pieces of state. When a node is built, its prior can subsequently
# be supplied by the connected hyperparameter node.
#
# Compute initial expectations, observed counts, constants, and the initial
# ELBO before starting training.
#
# Sources: [buildBiofam](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/build_model.py),
# [initModel](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/init_model.py),
# and [BayesNet.precompute](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/BayesNet.py).
# -

# + [markdown]
# ### Step 3: construct the actual Python update schedule
#
# The base schedule is:
#
#     Y -> W -> Z -> Tau
#
# When all ARD and spike-and-slab options are enabled, the constructed schedule is:
#
#     Y -> W -> AlphaW -> ThetaW -> Z -> AlphaZ -> ThetaZ -> Tau
#
# For the default single-group configuration it is:
#
#     Y -> W -> AlphaW -> ThetaW -> Z -> Tau
#
# Omit optional nodes when the corresponding prior is disabled. The actual
# Gaussian Y values do not change during this schedule.
#
# For a W update, hold the current Z and hyperparameter expectations fixed while
# sweeping through loading components. Refresh W's means and moments before
# its hyperparameter updates and before Z is updated. Apply the corresponding
# logic during the Z update.
#
# **Warm-up detail:** before startSparsity, the training loop skips updates of
# ThetaW and ThetaZ. It does not skip W, Z, or the ARD nodes. Describing this as
# "turning off all regularization for 50 iterations" would be inaccurate.
#
# Sources: [set_train_options](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/run/entry_point.py)
# and [BayesNet.iterate](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/BayesNet.py).
# -

# + [markdown]
# ### Step 4: full-batch training pseudocode
#
# This is an explanatory transcription of the reference control flow. The
# Gaussian coordinate equations in Method supply the W, Z, and Tau updates.
#
#     prepare processed views, group labels, masks, options, and initial state
#     construct the schedule from enabled model nodes
#     precompute constants and current moments
#     L_initial = calculate the initial ELBO
#     convergence_token = 1
#
#     for i = 1, ..., maxiter - 1:          # Python range(1, maxiter)
#         if factor dropping is enabled and its scheduled check is due:
#             calculate factor-wise explained variation
#             remove at most one eligible inactive factor, as in the reference
#             keep every node's factor dimensions consistent
#
#         for node in schedule:
#             if node is ThetaW or ThetaZ and i < startSparsity:
#                 continue
#             update this node's variational parameters
#             refresh this node's expectations for subsequent updates
#
#         if i >= startELBO and (i - startELBO) is divisible by freqELBO:
#             L_current = calculate ELBO from the current state
#             delta = L_current - the preceding scheduled ELBO
#                     (use L_initial at the first scheduled check)
#             update convergence_token using the reference percentage criterion
#             if convergence is confirmed and nostop is false:
#                 stop training
#
#         record training information
#
#     return posterior state and training information
#
# The Python maximum includes an initial state: its loop has at most
# maxiter-1 update sweeps. This indexing detail should be made explicit in
# fixed-iteration comparisons rather than silently comparing unequal work.
# ELBO changes need not be monotone across changes to the model structure
# such as factor removal; SVI also has its own stochastic behavior.
# -

# + [markdown]
# ### Step 5: monitor convergence and optional factor removal
#
# Python calculates the ELBO on scheduled checks rather than every iteration.
# The convergence statistic is
#
# $$
# 100\left|\frac{\mathcal L_{\mathrm{current}}-\mathcal L_{\mathrm{previous\ checked}}}
# {\mathcal L_{\mathrm{initial}}}\right|.
# $$
#
# | convergence_mode | Threshold for the percentage statistic above |
# |---|---:|
# | fast | 0.0005 |
# | medium | 0.00005 |
# | slow | 0.000005 |
#
# These numbers are **percentage thresholds**, and the denominator is the
# initial ELBO. In the reference, a token initialized to 1 reaches 3 after
# two consecutive qualifying checks. A failed check resets the token to 1.
# Early stopping is also gated by the initial-check/minimum-iteration controls.
#
# For centered Gaussian data, factor-wise explained variation uses
#
# $$
# R^2_{gmk}=1-
# \frac{\sum_{n,d:\,O=1}
# (y_{gmnd}-\bar z_{gnk}\bar w_{mdk})^2}
# {\sum_{n,d:\,O=1}y_{gmnd}^{\,2}}.
# $$
#
# This single-factor statistic need not add up across non-orthogonal factors.
# Total-model R-squared is calculated separately using all factors.
#
# If dropR2 is enabled, a factor is eligible when it fails to exceed the
# threshold in every group/view combination. The inspected implementation
# randomly selects at most one eligible factor at each removal check. The
# default dropR2=None leaves this explicit removal disabled.
#
# Source: [assess_convergence, calculate_variance_explained, and removeInactiveFactors](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/BayesNet.py).
# -

# + [markdown]
# ### The stochastic and non-Gaussian extensions
#
# **Stochastic inference.** At the configured starting iteration, the Python
# implementation takes sample mini-batches from a shuffled index sequence.
# Within each epoch these batches are taken without replacement. Its code uses
# the same constructed node schedule; comments suggesting moving Z first are
# not active schedule-changing statements in this snapshot.
#
# The active step-size function is
#
# $$
# \rho_e=\frac{\text{learning_rate}}
# {(1+\text{forgetting_rate}\,e)^{3/4}},
# $$
#
# where $e$ is the epoch index. Node-specific updates scale batch statistics:
# W uses total-sample/batch-sample scaling, and group-specific quantities use
# group-specific counts. Z updates only the selected samples.
#
# It is insufficient to average every stored matrix using one generic rule:
# W/SW, Z/SZ, AlphaW/AlphaZ, and ThetaW/ThetaZ have different update behavior.
# For example, AlphaW and ThetaW ignore the supplied damping step, whereas
# group-side updates use scaled batch quantities. Explicit factor dropping is
# disabled by set_stochastic_options.
#
# **Non-Gaussian views.** Y represents the appropriate pseudo-data node, which
# updates local variational-bound parameters and pseudo-observations. Those
# updates are part of the training work and must be included in a matching
# implementation and performance comparison.
#
# Sources: [StochasticBayesNet](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/BayesNet.py),
# [Alpha nodes](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Alpha_nodes.py),
# [Theta nodes](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Theta_nodes.py),
# and [non-Gaussian nodes](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/nongaussian_nodes.py).
# -

# + [markdown]
# ### Connecting the algorithm to an efficient Julia implementation
#
# The mathematical work consists largely of weighted projections, second-moment
# sums, residual updates, and reductions over observed entries. This suggests
# the following design directions for later profiling:
#
# | Computation | Opportunity to investigate | What must stay equivalent |
# |---|---|---|
# | Leave-one-component-out predictions | Reuse a residual workspace or equivalent sufficient statistics | Sequential component updates and current means |
# | Matrix projections | In-place mul! into preallocated arrays | Masks, view weights, and precision |
# | Hyperparameter expansion | Index compact group/view vectors instead of materializing repeated matrices | The correct precision/inclusion quantity at every coordinate |
# | Expected reconstruction errors | Blocked or fused reductions | Posterior second-moment correction |
# | Repeated fitting | Separate reusable workspaces from retained fitted results | Complete required output state and input ownership |
# | ELBO monitoring | Reuse valid sufficient statistics, as the reference already does in places | The same objective and checking schedule |
#
# A full residual workspace costs one data-sized matrix per retained view.
# A blocked approach may reduce peak memory but perform more computation.
# Allocating the final factors, variances, loadings, masks, and returned
# information is unavoidable; the objective is to minimize avoidable temporary
# allocations and unnecessary passes.
#
# These are proposed engineering directions, not benchmark findings. A change
# of precision, prior, likelihood, iteration schedule, stopping threshold, or
# fitted outputs must be reported as a changed configuration, rather than
# credited automatically as an implementation speedup.
# -

# + [markdown]
# ### Source map for subsequent implementation and validation
#
# All links below refer to the inspected mofapy2 0.7.5 source snapshot.
#
# | Topic | Reference file |
# |---|---|
# | Data/model/training options and schedule assembly | [run/entry_point.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/run/entry_point.py) |
# | Gaussian preprocessing | [build_model/utils.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/utils.py) |
# | Node construction | [build_model/build_model.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/build_model.py) |
# | Initialization | [build_model/init_model.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/init_model.py) |
# | Training, convergence, SVI, and factor removal | [core/BayesNet.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/BayesNet.py) |
# | Loading updates | [core/nodes/W_nodes.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/W_nodes.py) |
# | Sample-factor updates | [core/nodes/Z_nodes.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Z_nodes.py) |
# | Noise updates | [core/nodes/Tau_nodes.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Tau_nodes.py) |
# | ARD and inclusion hyperparameters | [Alpha_nodes.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Alpha_nodes.py), [Theta_nodes.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Theta_nodes.py) |
# | Observations and Gaussian likelihood ELBO | [core/nodes/Y_nodes.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/nodes/Y_nodes.py) |
# | Sparse-variable moments | [core/distributions/bernoulli_gaussian.py](https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/core/distributions/bernoulli_gaussian.py) |
#
# Validation should start from matching processed data and initial states,
# compare individual updates and ELBO components, then compare complete fits.
# Factor signs and permutations must be aligned where necessary. A common
# random seed alone does not guarantee identical PCA initialization or random
# sequences across Python and Julia.
#
# No Julia mofa fitting API, numerical equivalence result, or performance
# advantage is claimed by this notebook. It establishes the reference model
# and algorithm that the forthcoming implementation can be checked against.
# -
