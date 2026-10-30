# Implementation Note:
# Gaussian preprocessing follows mofapy2's group-wise centering, followed by
# optional view and group scaling (population standard deviations).
# https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/utils.py
# Dense Gaussian initialization and moments follow the same reference's
# build_model/init_model.py and core/distributions/univariate_gaussian.py.
# These internal functions prepare and initialize a model; they do not fit MOFA.

"""
	MofaData{T,L}

Internal container for Gaussian MOFA inputs, as returned by `_mofa_preprocess`
# Fields
- `data::Vector{Matrix{T}}`: The processed n×pᵢ matrices, one per view, with
  samples in their original row order and zeros in missing positions
- `missing_mask::Vector{BitMatrix}`: One n×pᵢ mask per view; true means missing.
  The stored zeros at these positions must be excluded from model calculations
- `group_indices::Vector{Int}`: The group index of each sample (length n)
- `group_labels::Vector{L}`: The original group labels, in first-appearance order
- `group_rows::Vector{Vector{Int}}`: The original row indices of each group
- `means::Vector{Matrix{T}}`: The removed group-by-feature means, one G×pᵢ
  matrix per view; zero when centering is disabled or a group/feature is missing
- `scales::Matrix{T}`: The G×M combined view/group scale divisors; one when
  scaling is disabled. Undo preprocessing using data*scale + mean entrywise
- `observed_counts::Vector{Matrix{Int}}`: The G×pᵢ observed-entry counts per view
"""
struct MofaData{T,L}
	data::Vector{Matrix{T}}
	missing_mask::Vector{BitMatrix}
	group_indices::Vector{Int}
	group_labels::Vector{L}
	group_rows::Vector{Vector{Int}}
	means::Vector{Matrix{T}}
	scales::Matrix{T}
	observed_counts::Vector{Matrix{Int}}
end

"""
	MofaGaussian{T}

Internal variational Gaussian parameters and moments for MOFA factors or weights
# Fields
- `mean::Matrix{T}`: The n×k factor means or pᵢ×k loading means; also the first
  moments E[x], without storing a duplicate expectation matrix
- `variance::Matrix{T}`: The positive variational variances, matching mean
- `second_moment::Matrix{T}`: The cached second moments E[x²] = mean² + variance;
  refreshed by `_mofa_gaussian_moments!` after parameter changes
"""
struct MofaGaussian{T}
	mean::Matrix{T}
	variance::Matrix{T}
	second_moment::Matrix{T}
end

"""
	MofaState{T}

Internal starting state for dense Gaussian MOFA, as returned by `_mofa_initialize`
# Fields
- `z::MofaGaussian{T}`: The n×k sample-factor distribution, shared across views
- `w::Vector{MofaGaussian{T}}`: One pᵢ×k loading distribution per view, shared
  across sample groups
- `tau_shape::Vector{Matrix{T}}`: The Gamma noise-posterior shapes, one G×pᵢ
  matrix per view; initialized to one
- `tau_rate::Vector{Matrix{T}}`: The Gamma noise-posterior rates, matching
  tau_shape; initialized to one. These are rates, not scales
- `tau_mean::Vector{Matrix{T}}`: The cached noise precisions E[τ] = shape/rate,
  kept per group/feature rather than expanded into n×pᵢ matrices
- `tau_prior_shape::T`: The common Gamma noise-prior shape, 1e-3
- `tau_prior_rate::T`: The common Gamma noise-prior rate, 1e-3

This state does not yet contain ARD or spike-and-slab distributions. The dense
factor and loading priors are zero-mean, unit-variance Gaussians
"""
struct MofaState{T}
	z::MofaGaussian{T}
	w::Vector{MofaGaussian{T}}
	tau_shape::Vector{Matrix{T}}
	tau_rate::Vector{Matrix{T}}
	tau_mean::Vector{Matrix{T}}
	tau_prior_shape::T
	tau_prior_rate::T
end

"""
	MofaWorkspace{T}

Internal reusable scratch buffers, allocated once by `_mofa_initialize`
# Fields
- `sample_buffer::Vector{T}`: Length-n scratch for sample-wise calculations
- `feature_buffer::Vector{T}`: Scratch of length max(pᵢ), reused across views;
  only the current view's first pᵢ entries are used
- `factor_buffer::Vector{T}`: Length-k scratch for factor-wise calculations

Scratch is separate from model parameters and starts at zero. No full n×pᵢ
residual matrix is reserved; update routines will be implemented separately
"""
struct MofaWorkspace{T}
	sample_buffer::Vector{T}
	feature_buffer::Vector{T}
	factor_buffer::Vector{T}
end

"""
	_mofa_gaussian_moments!(q::MofaGaussian{Float64})

Refresh Gaussian second moments without allocating a replacement matrix
# Arguments
- `q::MofaGaussian{Float64}`: Matching mean, positive variance, and second-moment
  matrices; the cached second moments are overwritten
# Value
The supplied q with E[x²] = mean² + variance. The first moment is q.mean itself.
Throws a `DimensionMismatch` for inconsistent shapes or an `ArgumentError` for
nonfinite means, nonpositive/nonfinite variances, or overflowing second moments
"""
function _mofa_gaussian_moments!(q::MofaGaussian{Float64})
	size(q.mean) == size(q.variance) == size(q.second_moment) ||
		throw(DimensionMismatch("MOFA Gaussian parameter and moment shapes must match"))
	for i in eachindex(q.mean)
		mu, variance = q.mean[i], q.variance[i]
		isfinite(mu) && isfinite(variance) && variance > 0 ||
			throw(ArgumentError("MOFA Gaussian means must be finite and variances finite and positive"))
		second = mu * mu + variance
		isfinite(second) || throw(ArgumentError("MOFA Gaussian second moment overflowed"))
		q.second_moment[i] = second
	end
	return q
end

"""
	_mofa_gamma_mean!(expected::Matrix{Float64}, shape::Matrix{Float64},
		rate::Matrix{Float64})

Refresh the expected Gamma noise precision using shape-rate parameters
# Arguments
- `expected::Matrix{Float64}`: Cached group-by-feature expectations, overwritten
- `shape::Matrix{Float64}`: Finite positive Gamma posterior shapes
- `rate::Matrix{Float64}`: Finite positive Gamma posterior rates
# Value
The supplied expected matrix containing shape/rate, without allocating a
replacement. Throws a `DimensionMismatch` for inconsistent shapes or an
`ArgumentError` for invalid parameters or nonfinite/nonpositive expectations.
This computes E[τ], not E[log τ]; log moments will accompany the later noise/ELBO
implementation rather than use a custom special-function approximation
"""
function _mofa_gamma_mean!(expected::Matrix{Float64}, shape::Matrix{Float64},
	rate::Matrix{Float64})
	size(expected) == size(shape) == size(rate) ||
		throw(DimensionMismatch("MOFA Gamma parameter and expectation shapes must match"))
	for i in eachindex(expected)
		a, b = shape[i], rate[i]
		isfinite(a) && a > 0 && isfinite(b) && b > 0 ||
			throw(ArgumentError("MOFA Gamma shapes and rates must be finite and positive"))
		value = a / b
		isfinite(value) && value > 0 ||
			throw(ArgumentError("MOFA Gamma expectation must be finite and positive"))
		expected[i] = value
	end
	return expected
end

"""
	_mofa_rescale_factors!(Z::Matrix{Float64})

Rescale each initial factor column to [-1,1], following mofapy2's dense Z path
# Arguments
- `Z::Matrix{Float64}`: Initial factor means, overwritten column by column
# Value
The supplied Z with each column's minimum equal to -1 and maximum equal to 1.
Throws an `ArgumentError` for empty, constant, nonfinite, or overflowing factor
columns. Used for PCA and random initialization only; supplied custom factors
are not rescaled
"""
function _mofa_rescale_factors!(Z::Matrix{Float64})
	isempty(Z) && throw(ArgumentError("MOFA initial factors must be nonempty"))
	for j in axes(Z, 2)
		lower, upper = extrema(@view Z[:, j])
		width = upper - lower
		isfinite(lower) && isfinite(upper) && isfinite(width) && width > 0 ||
			throw(ArgumentError("MOFA initial factor columns must be finite and nonconstant"))
		for i in axes(Z, 1)
			Z[i, j] = 2 * ((Z[i, j] - lower) / width) - 1
		end
	end
	return Z
end

"""
	_mofa_pca_factors(data::MofaData{Float64}, k::Int)

Initialize dense sample factors using PCA of the concatenated processed views
# Arguments
- `data::MofaData{Float64}`: Validated Gaussian views with missing-entry masks
- `k::Int`: Number of numerically nonzero centered principal components;
  1 ≤ k ≤ min(n-1, sum(pᵢ))
# Value
An n×k matrix of factor means rescaled to [-1,1] per column. Missing entries
are temporarily imputed with observed feature means, then every feature is
centered, matching sklearn PCA's preprocessing. One concatenated centered
matrix is built and overwritten by an SVD; data and masks remain unchanged.
Whitening multiplies each score column by a positive scalar that cancels during
the final rescaling, so the left singular vectors suffice. Signs use the
largest-magnitude loading, as in sklearn's full-SVD PCA path. Throws an
`ArgumentError` for invalid k, numerical rank deficiency, or nonfinite working
data; use random or custom initialization when PCA is not well-defined
"""
function _mofa_pca_factors(data::MofaData{Float64}, k::Int)
	n = size(first(data.data), 1)
	p = sum(size(Y, 2) for Y in data.data)
	1 <= k <= min(n - 1, p) ||
		throw(ArgumentError("MOFA PCA initialization requires 1 <= k <= min(n-1, total features)"))
	centered = Matrix{Float64}(undef, n, p)
	offset = 0
	for m in eachindex(data.data)
		Y, mask = data.data[m], data.missing_mask[m]
		for j in axes(Y, 2)
			average, count = 0.0, 0
			for i in 1:n
				mask[i, j] && continue
				count += 1
				average += (Y[i, j] - average) / count
			end
			for i in 1:n
				centered[i, offset + j] = mask[i, j] ? 0.0 : Y[i, j] - average
			end
		end
		offset += size(Y, 2)
	end
	all(isfinite, centered) || throw(ArgumentError("MOFA PCA working data must be finite"))
	F = svd!(centered; full = false)
	F.S[k] > eps(Float64) * max(n, p) * F.S[1] ||
		throw(ArgumentError("MOFA PCA initialization requires k numerically nonzero centered components; use :random or :custom"))
	Z = Matrix(@view F.U[:, 1:k])
	for component in 1:k
		_, pivot = findmax(abs, @view F.Vt[component, :])
		if F.Vt[component, pivot] < 0
			@views Z[:, component] .*= -1
		end
	end
	return _mofa_rescale_factors!(Z)
end

"""
	_mofa_initialize(data::MofaData{Float64}, k::Int; init::Symbol = :pca,
		rng::AbstractRNG = Random.GLOBAL_RNG, z_init = nothing, w_init = nothing)

Create the starting dense Gaussian MOFA parameters and reusable scratch buffers
# Arguments
- `data::MofaData{Float64}`: Validated, preprocessed Gaussian views
- `k::Int`: Positive number of latent factors, shared by every view
- `init::Symbol`: `:pca` for concatenated-view PCA (the Python builder default),
  `:random` for standard-normal draws, or `:custom` for supplied factor means.
  PCA/random columns are rescaled to [-1,1]; custom columns are not
- `rng::AbstractRNG`: Generator used only by :random. Defaults to Julia's global
  RNG. Equal seeds do not imply identical draws across Julia and Python
- `z_init`: An n×k finite real matrix, required only with :custom; copied
- `w_init`: Optional vector of pᵢ×k finite real loading matrices with :custom;
  copied. Otherwise loading means start at zero, matching the Python builder
# Value
A tuple `(state, workspace)` containing a `MofaState{Float64}` and a
`MofaWorkspace{Float64}`. Gaussian variances start at one and second moments are
computed from their means. Noise Gamma posteriors start at shape=rate=1, distinct
from the shape=rate=1e-3 priors. State, input data, supplied means, and scratch
buffers do not share storage. Throws an `ArgumentError` for invalid options,
nonfinite values, or degenerate PCA/random factors, and a `DimensionMismatch`
for incorrectly shaped custom means. No parameters are trained; ARD and
spike-and-slab initialization will be added with their separate update steps
"""
function _mofa_initialize(data::MofaData{Float64}, k::Int; init::Symbol = :pca,
	rng::AbstractRNG = Random.GLOBAL_RNG, z_init = nothing, w_init = nothing)
	k >= 1 || throw(ArgumentError("MOFA requires at least one factor"))
	init in (:pca, :random, :custom) || throw(ArgumentError("init must be :pca, :random, or :custom"))
	n = size(first(data.data), 1)
	if init === :custom
		z_init isa AbstractMatrix{<:Real} || throw(ArgumentError(":custom requires a real z_init matrix"))
		size(z_init) == (n, k) || throw(DimensionMismatch("z_init must have size (n,k)"))
		zmean = Matrix{Float64}(z_init)
	else
		z_init === nothing && w_init === nothing ||
			throw(ArgumentError("z_init and w_init can only be supplied with init=:custom"))
		zmean = init === :pca ? _mofa_pca_factors(data, k) : _mofa_rescale_factors!(randn(rng, n, k))
	end
	z = MofaGaussian(zmean, ones(n, k), Matrix{Float64}(undef, n, k))
	_mofa_gaussian_moments!(z)
	nviews, ngroups = length(data.data), length(data.group_labels)
	w_init === nothing || (w_init isa AbstractVector && length(w_init) == nviews) ||
		throw(ArgumentError("w_init must contain one loading matrix per view"))
	w = Vector{MofaGaussian{Float64}}(undef, nviews)
	tau_shape = Vector{Matrix{Float64}}(undef, nviews)
	tau_rate = Vector{Matrix{Float64}}(undef, nviews)
	tau_mean = Vector{Matrix{Float64}}(undef, nviews)
	for m in eachindex(data.data)
		p = size(data.data[m], 2)
		if w_init === nothing
			wmean = zeros(p, k)
		else
			w_init[m] isa AbstractMatrix{<:Real} || throw(ArgumentError("each w_init entry must be a real matrix"))
			size(w_init[m]) == (p, k) || throw(DimensionMismatch("w_init[$m] must have size ($p,$k)"))
			wmean = Matrix{Float64}(w_init[m])
		end
		w[m] = MofaGaussian(wmean, ones(p, k), Matrix{Float64}(undef, p, k))
		_mofa_gaussian_moments!(w[m])
		tau_shape[m] = ones(ngroups, p)
		tau_rate[m] = ones(ngroups, p)
		tau_mean[m] = Matrix{Float64}(undef, ngroups, p)
		_mofa_gamma_mean!(tau_mean[m], tau_shape[m], tau_rate[m])
	end
	state = MofaState(z, w, tau_shape, tau_rate, tau_mean, 1e-3, 1e-3)
	workspace = MofaWorkspace(zeros(n), zeros(maximum(size(Y, 2) for Y in data.data)), zeros(k))
	return state, workspace
end

"""
	_mofa_groups(n::Int, groups::Union{Nothing,AbstractVector})

Index sample groups without changing sample order
# Arguments
- `n::Int`: The number of aligned samples, at least one
- `groups::Union{Nothing,AbstractVector}`: One label per sample, or nothing for
  a single group. Labels must not be missing or nothing; numeric labels must
  be finite
# Value
A tuple `(indices, labels, rows)` containing the group index per sample, unique
labels in first-appearance order, and row indices per group. Throws an
`ArgumentError` for invalid labels or n, or a `DimensionMismatch` when the label
count differs from n. Does not modify the supplied labels
"""
function _mofa_groups(n::Int, groups::Union{Nothing,AbstractVector})
	n >= 1 || throw(ArgumentError("MOFA requires at least one sample"))
	groups === nothing && return (ones(Int, n), [1], [collect(1:n)])
	Base.require_one_based_indexing(groups)
	length(groups) == n || throw(DimensionMismatch("groups must contain one label per sample"))
	all(label -> !ismissing(label) && label !== nothing &&
		(!(label isa Number) || isfinite(label)), groups) ||
		throw(ArgumentError("group labels must be nonmissing; numeric labels must be finite"))
	labels = unique(groups)
	lookup = Dict(label => g for (g, label) in enumerate(labels))
	indices = Vector{Int}(undef, n)
	rows = [Int[] for _ in labels]
	for i in 1:n
		g = lookup[groups[i]]
		indices[i] = g
		push!(rows[g], i)
	end
	return indices, labels, rows
end

"""
	_mofa_scale!(Y::Matrix{Float64}, mask::BitMatrix,
		rows::AbstractVector{Int})

Divide observed entries in selected rows by their population standard deviation
# Arguments
- `Y::Matrix{Float64}`: A processed view, modified in place at observed entries
- `mask::BitMatrix`: Missing-entry mask matching Y; true entries are excluded
- `rows::AbstractVector{Int}`: Unique, valid row indices of the view or group
# Value
Float; the scalar population standard deviation of all observed entries in
the selected block, computed about their overall mean with divisor N, not N-1.
Missing positions remain unchanged. Throws an `ArgumentError` when the block
has no observations or its scale is zero or nonfinite. Internal callers supply
matching matrix dimensions and valid row indices
"""
function _mofa_scale!(Y::Matrix{Float64}, mask::BitMatrix,
	rows::AbstractVector{Int})
	average = 0.0
	count = 0
	for j in axes(Y, 2), i in rows
		mask[i, j] && continue
		count += 1
		average += (Y[i, j] - average) / count
	end
	count > 0 || throw(ArgumentError("cannot scale a MOFA block with no observed entries"))
	# Accumulate the scaled Euclidean norm without allocating a residual block.
	root_count = sqrt(count)
	scale = 0.0
	for j in axes(Y, 2), i in rows
		mask[i, j] && continue
		scale = hypot(scale, (Y[i, j] - average) / root_count)
	end
	isfinite(scale) && scale > 0 ||
		throw(ArgumentError("MOFA scaling requires a finite, positive block standard deviation"))
	for j in axes(Y, 2), i in rows
		mask[i, j] || (Y[i, j] /= scale)
	end
	return scale
end

"""
	_mofa_preprocess(views::AbstractVector; groups = nothing,
		center_groups::Bool = true, scale_views::Bool = false,
		scale_groups::Bool = false)

Validate and prepare aligned Gaussian data views for MOFA without modifying input
# Arguments
- `views::AbstractVector`: A nonempty vector of real-valued matrices, one per
  view; samples in rows and features in columns. All views must have the same
  nonzero number of rows and at least one feature. `missing` and NaN denote
  missing observations; infinities and entirely missing features are rejected
- `groups`: One nonmissing label per sample, or nothing for one group. Rows must
  refer to the same samples in the same order across views; alignment cannot be
  inferred from numeric matrices. Groups need not occupy consecutive rows
- `center_groups::Bool`: Whether to remove each feature's observed mean within
  each group. Defaults to true
- `scale_views::Bool`: Whether to divide each view by its overall observed
  population standard deviation after centering. Defaults to false
- `scale_groups::Bool`: Whether to divide each group/view block by its observed
  population standard deviation after view scaling. Defaults to false
# Value
A `MofaData{Float64}` containing copied data, missing-entry masks, group indices,
and preprocessing statistics. Scaling uses one scalar per view or group/view
block, not per-feature standardization. Entirely missing group/feature blocks
have mean zero and count zero. Entirely missing group/view blocks are allowed
unless group scaling is requested. Constant features are retained, but a block
with zero variance cannot be scaled. Throws an `ArgumentError` for invalid data
or undefined scaling and a `DimensionMismatch` for inconsistent sample counts.
This internal function prepares Gaussian data only; it does not fit MOFA
"""
function _mofa_preprocess(views::AbstractVector; groups = nothing,
	center_groups::Bool = true, scale_views::Bool = false,
	scale_groups::Bool = false)
	Base.require_one_based_indexing(views)
	isempty(views) && throw(ArgumentError("MOFA requires at least one view"))
	all(X -> X isa AbstractMatrix, views) ||
		throw(ArgumentError("each MOFA view must be a matrix"))
	n = size(first(views), 1)
	indices, labels, rows = _mofa_groups(n, groups)
	ngroups, nviews = length(labels), length(views)
	data = Vector{Matrix{Float64}}(undef, nviews)
	masks = Vector{BitMatrix}(undef, nviews)
	means = Vector{Matrix{Float64}}(undef, nviews)
	counts = Vector{Matrix{Int}}(undef, nviews)
	scales = ones(Float64, ngroups, nviews)
	for m in eachindex(views)
		X = views[m]
		Base.require_one_based_indexing(X)
		size(X, 1) == n || throw(DimensionMismatch("all MOFA views must have the same number of samples"))
		p = size(X, 2)
		p >= 1 || throw(ArgumentError("MOFA view $m has no features"))
		Y = zeros(Float64, n, p)
		mask = falses(n, p)
		group_means = zeros(Float64, ngroups, p)
		group_counts = zeros(Int, ngroups, p)
		for j in 1:p, i in 1:n
			value = X[i, j]
			if ismissing(value) || (value isa Real && isnan(value))
				mask[i, j] = true
			else
				value isa Real && isfinite(value) ||
					throw(ArgumentError("MOFA view $m must contain finite real values, missing, or NaN"))
				y = Float64(value)
				isfinite(y) || throw(ArgumentError("MOFA values must be representable as finite Float64"))
				Y[i, j] = y
				g = indices[i]
				group_counts[g, j] += 1
				if center_groups
					group_means[g, j] += (y - group_means[g, j]) / group_counts[g, j]
				end
			end
		end
		for j in 1:p
			sum(@view group_counts[:, j]) > 0 ||
				throw(ArgumentError("feature $j in MOFA view $m is entirely missing"))
		end
		if center_groups
			for j in 1:p, i in 1:n
				mask[i, j] || (Y[i, j] -= group_means[indices[i], j])
			end
		end
		if scale_views
			view_scale = _mofa_scale!(Y, mask, Base.OneTo(n))
			scales[:, m] .= view_scale
		end
		if scale_groups
			for g in 1:ngroups
				scales[g, m] *= _mofa_scale!(Y, mask, rows[g])
			end
		end
		all(isfinite, Y) && all(isfinite, group_means) &&
			all(s -> isfinite(s) && s > 0, @view scales[:, m]) ||
			throw(ArgumentError("MOFA preprocessing produced nonfinite values or invalid scales"))
		data[m], masks[m], means[m], counts[m] = Y, mask, group_means, group_counts
	end
	return MofaData(data, masks, indices, labels, rows, means, scales, counts)
end
