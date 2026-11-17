# Implementation Note:
# Gaussian preprocessing follows mofapy2's group-wise centering, followed by
# optional view and group scaling (population standard deviations).
# https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/utils.py
# Dense Gaussian initialization and moments follow the same reference's
# build_model/init_model.py and core/distributions/univariate_gaussian.py.
# Dense Gaussian weight/factor sweeps follow core/nodes/W_nodes.py and
# core/nodes/Z_nodes.py (full batch, equal view weights,
# zero-mean priors). Gamma noise/ARD and sparse-weight/Beta updates follow
# Tau_nodes.py, Alpha_nodes.py, W_nodes.py, and Theta_nodes.py in that snapshot.
# ELBO and full-batch scheduling follow core/BayesNet.py and these same nodes.
# Only Gaussian observations, dense sample factors, and equal view weights
# are supported; non-Gaussian likelihoods, SVI, MEFISTO, and factor dropping
# are not implemented. Public maxiter counts update sweeps, not the initial state.
# Coordinate equations/order are unchanged. Small ranks use direct SIMD
# prediction; higher ranks reuse prefix/suffix sums so a coordinate's old
# contribution never needs to be subtracted from a large residual.
# Large sweeps distribute independent loading/sample rows across Julia tasks,
# each with private scratch and unchanged within-row coordinate arithmetic.

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

Internal Gaussian parameters and moment storage for MOFA factors or weights
# Fields
- `mean::Matrix{T}`: The n×k factor means or pᵢ×k loading means; also the first
  moments E[x], without storing a duplicate expectation matrix
- `variance::Matrix{T}`: The positive variational variances, matching mean
- `second_moment::Matrix{T}`: The cached second moments E[x²] = mean² + variance;
  refreshed by `_mofa_gaussian_moments!` after parameter changes

When sparse weights are enabled, the loading entries store the effective
mixture mean, variance, and second moment instead. Their active Gaussian
parameters live in `MofaSpikeSlab`; use `_mofa_spike_moments!` to refresh them
"""
struct MofaGaussian{T}
	mean::Matrix{T}
	variance::Matrix{T}
	second_moment::Matrix{T}
end

"""
	MofaGamma{T}

Internal Gamma posterior and cached expectations for MOFA ARD precisions
# Fields
- `shape::Matrix{T}`: Posterior shapes; one row per view or sample group
- `rate::Matrix{T}`: Posterior rates, not scales, matching shape
- `mean::Matrix{T}`: Cached E[x] = shape/rate
- `logmean::Matrix{T}`: Cached E[log x] = digamma(shape) - log(rate)
- `prior_shape::T`: Positive common prior shape, 1e-14 for ARD
- `prior_rate::T`: Positive common prior rate, 1e-14 for ARD
"""
struct MofaGamma{T}
	shape::Matrix{T}
	rate::Matrix{T}
	mean::Matrix{T}
	logmean::Matrix{T}
	prior_shape::T
	prior_rate::T
end

"""
	MofaBeta{T}

Internal Beta posterior for loading inclusion probabilities, one view×factor entry
# Fields
- `a::Matrix{T}`: Posterior first shape parameters
- `b::Matrix{T}`: Posterior second shape parameters
- `mean::Matrix{T}`: Cached E[θ] = a/(a+b)
- `logmean::Matrix{T}`: Cached E[log θ] = digamma(a) - digamma(a+b)
- `logcomplement::Matrix{T}`: Cached E[log(1-θ)] = digamma(b) - digamma(a+b)
- `prior_a::T`: Common first prior shape, one
- `prior_b::T`: Common second prior shape, one
"""
struct MofaBeta{T}
	a::Matrix{T}
	b::Matrix{T}
	mean::Matrix{T}
	logmean::Matrix{T}
	logcomplement::Matrix{T}
	prior_a::T
	prior_b::T
end

"""
	MofaSpikeSlab{T}

Internal active-slab parameters and inclusion probabilities for one loading view
# Fields
- `mean::Matrix{T}`: The pᵢ×k active Gaussian means μ, not effective loading means
- `variance::Matrix{T}`: The pᵢ×k positive active Gaussian variances v
- `inactive_variance::Vector{T}`: Length-k inactive Gaussian variances, 1/E[α]
  at the last loading update; shared across features instead of expanded
- `inclusion::Matrix{T}`: The pᵢ×k Bernoulli expectations γ, in [0,1]

Effective loading moments are stored in state.w. The underlying second moment
used by ARD is γ*(μ²+v) + (1-γ)*inactive_variance, computed when needed rather
than stored as another pᵢ×k matrix. It differs from the effective second moment
γ*(μ²+v) used in factor/noise updates
"""
struct MofaSpikeSlab{T}
	mean::Matrix{T}
	variance::Matrix{T}
	inactive_variance::Vector{T}
	inclusion::Matrix{T}
end

"""
	MofaState{T}

Internal variational state for Gaussian MOFA, created by `_mofa_initialize`
# Fields
- `z::MofaGaussian{T}`: The n×k sample-factor distribution, shared across views
- `w::Vector{MofaGaussian{T}}`: One pᵢ×k effective loading-moment container per
  view, shared across groups; Gaussian parameters when sparse weights are off
- `tau_shape::Vector{Matrix{T}}`: The Gamma noise-posterior shapes, one G×pᵢ
  matrix per view; initialized to one
- `tau_rate::Vector{Matrix{T}}`: The Gamma noise-posterior rates, matching
  tau_shape; initialized to one. These are rates, not scales
- `tau_mean::Vector{Matrix{T}}`: The cached noise precisions E[τ] = shape/rate,
  kept per group/feature rather than expanded into n×pᵢ matrices
- `tau_prior_shape::T`: The common Gamma noise-prior shape, 1e-3
- `tau_prior_rate::T`: The common Gamma noise-prior rate, 1e-3
- `tau_logmean::Vector{Matrix{T}}`: Group-by-feature E[log τ] per view, distinct
  from log(tau_mean)
- `alpha_w::Union{Nothing,MofaGamma{T}}`: Optional view×factor ARD posterior
- `alpha_z::Union{Nothing,MofaGamma{T}}`: Optional group×factor ARD posterior
- `spike_w::Union{Nothing,Vector{MofaSpikeSlab{T}}}`: Optional sparse-weight state
- `theta_w::Union{Nothing,MofaBeta{T}}`: Optional view×factor inclusion posterior

Without ARD the corresponding Gaussian prior precision remains one. Sample
factors are dense; spike-and-slab is currently supported for loading weights only
"""
struct MofaState{T}
	z::MofaGaussian{T}
	w::Vector{MofaGaussian{T}}
	tau_shape::Vector{Matrix{T}}
	tau_rate::Vector{Matrix{T}}
	tau_mean::Vector{Matrix{T}}
	tau_prior_shape::T
	tau_prior_rate::T
	tau_logmean::Vector{Matrix{T}}
	alpha_w::Union{Nothing,MofaGamma{T}}
	alpha_z::Union{Nothing,MofaGamma{T}}
	spike_w::Union{Nothing,Vector{MofaSpikeSlab{T}}}
	theta_w::Union{Nothing,MofaBeta{T}}
end

"""
	MofaState(z, w, tau_shape, tau_rate, tau_mean, tau_prior_shape, tau_prior_rate)

Construct an internal dense state using the original preparation-stage arguments
# Arguments
- `z::MofaGaussian{Float64}`: Sample-factor parameter and moment storage
- `w::Vector{MofaGaussian{Float64}}`: Loading parameter and moment storage
- `tau_shape::Vector{Matrix{Float64}}`: Group-by-feature posterior noise shapes
- `tau_rate::Vector{Matrix{Float64}}`: Matching posterior noise rates
- `tau_mean::Vector{Matrix{Float64}}`: Existing matching expected noise precisions
- `tau_prior_shape::Float64`: Common positive prior noise shape
- `tau_prior_rate::Float64`: Common positive prior noise rate
# Value
A `MofaState{Float64}` reusing the supplied arrays, computing cached noise log
expectations, and disabling ARD and sparse weights. This compatibility
constructor is for internal state assembly; `_mofa_initialize` constructs owned
arrays and supports the optional model settings
"""
function MofaState(z::MofaGaussian{Float64}, w::Vector{MofaGaussian{Float64}},
	tau_shape::Vector{Matrix{Float64}}, tau_rate::Vector{Matrix{Float64}},
	tau_mean::Vector{Matrix{Float64}}, tau_prior_shape::Float64, tau_prior_rate::Float64)
	logs = [similar(a) for a in tau_shape]
	for m in eachindex(logs)
		size(logs[m]) == size(tau_rate[m]) || throw(DimensionMismatch("noise shapes and rates must match"))
		for i in eachindex(logs[m])
			a, b = tau_shape[m][i], tau_rate[m][i]
			isfinite(a) && a > 0 && isfinite(b) && b > 0 || throw(ArgumentError("noise shapes/rates must be finite and positive"))
			value = digamma(a) - log(b)
			isfinite(value) || throw(ArgumentError("noise log expectations must be finite"))
			logs[m][i] = value
		end
	end
	return MofaState(z, w, tau_shape, tau_rate, tau_mean, tau_prior_shape,
		tau_prior_rate, logs, nothing, nothing, nothing, nothing)
end

"""
	MofaWorkspace{T}

Internal reusable scratch buffers, allocated once by `_mofa_initialize`
# Fields
- `sample_buffer::Vector{T}`: Length-n scratch for sample-wise calculations
- `feature_buffer::Vector{T}`: Scratch of length max(pᵢ), reused across views;
  only the current view's first pᵢ entries are used
- `factor_buffer::Vector{T}`: Length-k scratch for factor-wise calculations
- `coordinate_sums::Union{Nothing,Matrix{T}}`: Optional max(n,sum(pᵢ))×(k+1)
  prefix/suffix cache. Used from rank eight onward; small ranks avoid its
  storage and preparation overhead and use direct prediction instead
- `parallel_scratch::Vector{Tuple{Matrix{T},Vector{T}}}`: Independent suffix
  and coefficient buffers for large sweeps; empty for serial execution. The
  first pair reuses coordinate_sums and factor_buffer

Scratch is separate from model parameters and starts at zero. Prefix/suffix
sums exclude the current coordinate without subtract/add-back cancellation.
Storage grows with factor/loading counts, not with n×pᵢ residual matrices
"""
struct MofaWorkspace{T}
	sample_buffer::Vector{T}
	feature_buffer::Vector{T}
	factor_buffer::Vector{T}
	coordinate_sums::Union{Nothing,Matrix{T}}
	parallel_scratch::Vector{Tuple{Matrix{T},Vector{T}}}
end

"""
	MofaWorkspace(sample_buffer::Vector{T}, feature_buffer::Vector{T},
		factor_buffer::Vector{T}, coordinate_sums::Union{Nothing,Matrix{T}}) where {T}

Construct explicitly serial MOFA scratch with optional coordinate sums
# Arguments
- `sample_buffer::Vector{T}`: Length-n sample scratch, reused without copying
- `feature_buffer::Vector{T}`: Feature scratch, reused without copying
- `factor_buffer::Vector{T}`: Length-k coefficient scratch, reused without copying
- `coordinate_sums::Union{Nothing,Matrix{T}}`: Prefix/suffix scratch or nothing
# Value
A `MofaWorkspace` reusing the supplied arrays with no parallel workers.
`_mofa_initialize` enables independent worker scratch only for large problems
"""
function MofaWorkspace(sample_buffer::Vector{T}, feature_buffer::Vector{T},
	factor_buffer::Vector{T}, coordinate_sums::Union{Nothing,Matrix{T}}) where {T}
	return MofaWorkspace(sample_buffer, feature_buffer, factor_buffer, coordinate_sums,
		Tuple{Matrix{T},Vector{T}}[])
end

"""
	MofaWorkspace(sample_buffer::Vector{T}, feature_buffer::Vector{T},
		factor_buffer::Vector{T}) where {T}

Construct reusable MOFA scratch from three independently supplied buffers
# Arguments
- `sample_buffer::Vector{T}`: Length-n sample buffer, reused without copying
- `feature_buffer::Vector{T}`: Feature scratch; its length also reserves suffix rows
- `factor_buffer::Vector{T}`: Length-k coefficients, reused without copying
# Value
A `MofaWorkspace` with an additional max(n,p)×(k+1) allocation at rank ≥ 8.
`_mofa_initialize` reserves max(n,sum(pᵢ)) rows for complete multiview updates
"""
function MofaWorkspace(sample_buffer::Vector{T}, feature_buffer::Vector{T},
	factor_buffer::Vector{T}) where {T}
	sums = length(factor_buffer) >= 8 ?
		zeros(T, max(length(sample_buffer), length(feature_buffer)), length(factor_buffer) + 1) : nothing
	return MofaWorkspace(sample_buffer, feature_buffer, factor_buffer, sums)
end

"""
	Mofa{T,L,O}

Container for a fitted Gaussian multi-omics factor analysis, as returned by `mofa`
# Fields
- `z::Matrix{T}`: The n×k posterior sample-factor means, shared across views
- `w::Vector{Matrix{T}}`: The pᵢ×k effective loading means, one matrix per view;
  reconstruct processed view i with z*transpose(w[i])
- `posterior::MofaState{T}`: Variational variances, second moments, noise
  precisions, optional ARD, and optional sparse-loading probabilities. Its
  means share storage with z and w; no duplicate factor matrices are stored
- `group_indices::Vector{Int}`: Length-n group indices in original sample order
- `group_labels::Vector{L}`: Original group labels in first-appearance order
- `means::Vector{Matrix{T}}`: Removed G×pᵢ group/feature means, one per view
- `scales::Matrix{T}`: The G×M preprocessing divisors; undo scaling before adding means
- `elbo::Vector{T}`: Finite objective values at initialization, scheduled checks,
  and the final sweep if it was not a scheduled check
- `elbo_iterations::Vector{Int}`: Sweep number for each ELBO value; starts at zero
- `niter::Int`: Number of update sweeps actually performed, excluding initialization
- `converged::Bool`: Whether two eligible consecutive ELBO checks met the
  reference percentage-change criterion; false when forceiter=true
- `r2::Matrix{T}`: G×M total-model explained variation on observed processed
  entries. NaN denotes a group/view with no nonzero observed data
- `factor_r2::Array{T,3}`: G×M×k single-factor explained variation; not additive
  across nonorthogonal factors and not clipped to [0,1]
- `options::O`: Named tuple recording preprocessing, model, and training settings

The fitted object does not retain training views, masks, or scratch buffers.
Factor signs, order, and scale are not individually identifiable
"""
struct Mofa{T,L,O}
	z::Matrix{T}
	w::Vector{Matrix{T}}
	posterior::MofaState{T}
	group_indices::Vector{Int}
	group_labels::Vector{L}
	means::Vector{Matrix{T}}
	scales::Matrix{T}
	elbo::Vector{T}
	elbo_iterations::Vector{Int}
	niter::Int
	converged::Bool
	r2::Matrix{T}
	factor_r2::Array{T,3}
	options::O
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
This computes E[τ] only; `_mofa_gamma_moments!` refreshes both expectations
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
	_mofa_gamma_moments!(expected, logexpected, shape, rate)

Refresh Gamma expectations in preallocated matrices using shape-rate parameters
# Arguments
- `expected::Matrix{Float64}`: E[x] output buffer, overwritten
- `logexpected::Matrix{Float64}`: E[log x] output buffer, overwritten
- `shape::Matrix{Float64}`: Finite positive posterior shapes
- `rate::Matrix{Float64}`: Matching finite positive posterior rates
# Value
Nothing; fills shape/rate and digamma(shape)-log(rate), respectively. Throws a
`DimensionMismatch` for incompatible buffers or an `ArgumentError` for invalid
parameters or nonfinite moments. The input parameters are not modified
"""
function _mofa_gamma_moments!(expected::Matrix{Float64}, logexpected::Matrix{Float64},
	shape::Matrix{Float64}, rate::Matrix{Float64})
	size(logexpected) == size(shape) || throw(DimensionMismatch("Gamma log-moment shape must match parameters"))
	_mofa_gamma_mean!(expected, shape, rate)
	for i in eachindex(shape)
		value = digamma(shape[i]) - log(rate[i])
		isfinite(value) || throw(ArgumentError("Gamma log expectation must be finite"))
		logexpected[i] = value
	end
	return nothing
end

"""
	_mofa_gamma_moments!(q::MofaGamma{Float64})

Refresh the cached expectations of an ARD Gamma posterior
# Arguments
- `q::MofaGamma{Float64}`: Posterior shape/rate arrays and output caches
# Value
The supplied q after refreshing E[x] and E[log x]. Throws an `ArgumentError`
for invalid prior parameters; posterior validation follows the matrix overload
"""
function _mofa_gamma_moments!(q::MofaGamma{Float64})
	isfinite(q.prior_shape) && q.prior_shape > 0 && isfinite(q.prior_rate) && q.prior_rate > 0 ||
		throw(ArgumentError("Gamma prior shapes and rates must be finite and positive"))
	_mofa_gamma_moments!(q.mean, q.logmean, q.shape, q.rate)
	return q
end

"""
	_mofa_check_gamma(q, dims::Tuple{Int,Int})

Validate an optional, already refreshed ARD posterior before an update
# Arguments
- `q::Union{Nothing,MofaGamma{Float64}}`: Optional ARD posterior; nothing disables it
- `dims::Tuple{Int,Int}`: Required view/group-by-factor dimensions
# Value
Nothing. Throws a `DimensionMismatch` for incorrect dimensions or an
`ArgumentError` for invalid priors, parameters, or cached moments. Does not
refresh caches or change the posterior
"""
function _mofa_check_gamma(q::Union{Nothing,MofaGamma{Float64}}, dims::Tuple{Int,Int})
	q === nothing && return nothing
	size(q.shape) == size(q.rate) == size(q.mean) == size(q.logmean) == dims ||
		throw(DimensionMismatch("ARD posterior dimensions must match the model"))
	all(x -> isfinite(x) && x > 0, q.shape) && all(x -> isfinite(x) && x > 0, q.rate) &&
		all(x -> isfinite(x) && x > 0, q.mean) && all(isfinite, q.logmean) &&
		isfinite(q.prior_shape) && q.prior_shape > 0 && isfinite(q.prior_rate) && q.prior_rate > 0 ||
		throw(ArgumentError("ARD priors, parameters, and expectations must be valid"))
	return nothing
end

"""
	_mofa_beta_moments!(q::MofaBeta{Float64})

Refresh Beta inclusion-hyperparameter expectations without replacement arrays
# Arguments
- `q::MofaBeta{Float64}`: View-by-factor positive shape parameters and caches
# Value
The supplied q holding a/(a+b), digamma(a)-digamma(a+b), and
digamma(b)-digamma(a+b). Throws a `DimensionMismatch` for inconsistent shapes or
an `ArgumentError` for invalid prior/posterior parameters or nonfinite moments
"""
function _mofa_beta_moments!(q::MofaBeta{Float64})
	size(q.a) == size(q.b) == size(q.mean) == size(q.logmean) == size(q.logcomplement) ||
		throw(DimensionMismatch("Beta posterior parameter and cache shapes must match"))
	isfinite(q.prior_a) && q.prior_a > 0 && isfinite(q.prior_b) && q.prior_b > 0 ||
		throw(ArgumentError("Beta prior shapes must be finite and positive"))
	for i in eachindex(q.a)
		a, b = q.a[i], q.b[i]
		isfinite(a) && a > 0 && isfinite(b) && b > 0 && isfinite(a + b) ||
			throw(ArgumentError("Beta posterior shapes must be finite and positive"))
		total = digamma(a + b)
		q.mean[i] = a / (a + b)
		q.logmean[i] = digamma(a) - total
		q.logcomplement[i] = digamma(b) - total
		isfinite(q.logmean[i]) && isfinite(q.logcomplement[i]) ||
			throw(ArgumentError("Beta log expectations must be finite"))
	end
	return q
end

"""
	_mofa_check_beta(q::MofaBeta{Float64}, dims::Tuple{Int,Int})

Validate a refreshed inclusion-hyperparameter posterior without changing caches
# Arguments
- `q::MofaBeta{Float64}`: Positive Beta parameters and cached expectations
- `dims::Tuple{Int,Int}`: Required view-by-factor dimensions
# Value
Nothing. Throws a `DimensionMismatch` for incompatible arrays or an
`ArgumentError` for invalid parameters, priors, or nonfinite cached expectations
"""
function _mofa_check_beta(q::MofaBeta{Float64}, dims::Tuple{Int,Int})
	size(q.a) == size(q.b) == size(q.mean) == size(q.logmean) == size(q.logcomplement) == dims ||
		throw(DimensionMismatch("inclusion posterior dimensions must match the model"))
	all(x -> isfinite(x) && x > 0, q.a) && all(x -> isfinite(x) && x > 0, q.b) &&
		all(x -> isfinite(x) && 0 <= x <= 1, q.mean) && all(isfinite, q.logmean) && all(isfinite, q.logcomplement) &&
		isfinite(q.prior_a) && q.prior_a > 0 && isfinite(q.prior_b) && q.prior_b > 0 ||
		throw(ArgumentError("inclusion posterior parameters and moments must be valid"))
	return nothing
end

"""
	_mofa_check_spike(q::MofaSpikeSlab{Float64}, dims::Tuple{Int,Int})

Validate the active-slab parameters and inclusion probabilities for one view
# Arguments
- `q::MofaSpikeSlab{Float64}`: Active means/variances, inactive variances, and γ
- `dims::Tuple{Int,Int}`: Required feature-by-factor dimensions
# Value
Nothing. Throws a `DimensionMismatch` for incorrect dimensions or an
`ArgumentError` for nonfinite means, nonpositive/nonfinite variances, or γ
outside [0,1]. Does not modify the supplied arrays
"""
function _mofa_check_spike(q::MofaSpikeSlab{Float64}, dims::Tuple{Int,Int})
	size(q.mean) == size(q.variance) == size(q.inclusion) == dims &&
		length(q.inactive_variance) == dims[2] || throw(DimensionMismatch("sparse loading dimensions must match the model"))
	all(isfinite, q.mean) && all(x -> isfinite(x) && x > 0, q.variance) &&
		all(x -> isfinite(x) && x > 0, q.inactive_variance) &&
		all(x -> isfinite(x) && 0 <= x <= 1, q.inclusion) ||
		throw(ArgumentError("sparse loading parameters must be finite with positive variances and inclusion in [0,1]"))
	return nothing
end

"""
	_mofa_spike_moments!(effective::MofaGaussian{Float64}, q::MofaSpikeSlab{Float64})

Refresh effective sparse-loading moments while retaining active-slab parameters
# Arguments
- `effective::MofaGaussian{Float64}`: Effective mean/variance/second-moment buffers
- `q::MofaSpikeSlab{Float64}`: Active means μ, variances v, and inclusion γ
# Value
The supplied effective storage with mean γμ, second moment γ(μ²+v), and
variance γv+γ(1-γ)μ². Exact γ=0 and γ=1 are supported without clipping.
Throws a `DimensionMismatch` for incompatible arrays or an `ArgumentError`
for invalid sparse parameters or overflowing moments. Does not modify q
"""
function _mofa_spike_moments!(effective::MofaGaussian{Float64}, q::MofaSpikeSlab{Float64})
	size(effective.mean) == size(effective.variance) == size(effective.second_moment) ||
		throw(DimensionMismatch("effective sparse-loading moment shapes must match"))
	_mofa_check_spike(q, size(effective.mean))
	for i in eachindex(q.mean)
		mu, v, g = q.mean[i], q.variance[i], q.inclusion[i]
		second = mu * mu + v
		isfinite(second) || throw(ArgumentError("sparse loading second moment overflowed"))
		effective.mean[i] = g * mu
		effective.second_moment[i] = g * second
		effective.variance[i] = g * v + g * (1 - g) * mu * mu
	end
	return effective
end

"""
	_mofa_logistic(x::Float64)

Evaluate an inclusion probability without overflow in the exponential
# Arguments
- `x::Float64`: A log-odds value; positive/negative infinity are allowed
# Value
Float in [0,1]. Returns one/zero at the respective infinities and throws an
`ArgumentError` for NaN. No clipping or approximation is used
"""
function _mofa_logistic(x::Float64)
	isnan(x) && throw(ArgumentError("inclusion log odds must not be NaN"))
	x >= 0 && return inv(1 + exp(-x))
	e = exp(x)
	return e / (1 + e)
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
		rng::AbstractRNG = Random.GLOBAL_RNG, z_init = nothing, w_init = nothing,
		ard_weights::Bool = false, ard_factors::Bool = false,
		spikeslab_weights::Bool = false)

Create Gaussian MOFA parameters, optional shrinkage state, and reusable buffers
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
- `ard_weights::Bool`: Enable view/factor ARD; defaults to false to retain the
  dense preparation-stage behavior
- `ard_factors::Bool`: Enable group/factor ARD; defaults to false
- `spikeslab_weights::Bool`: Enable sparse loading weights and Beta inclusion
  hyperparameters; defaults to false. Sample factors remain dense
# Value
A tuple `(state, workspace)` containing a `MofaState{Float64}` and a
`MofaWorkspace{Float64}`. Gaussian variances start at one and second moments are
computed from their means. Noise Gamma posteriors start at shape=rate=1, distinct
from the shape=rate=1e-3 priors. State, input data, supplied means, and scratch
buffers do not share storage. Throws an `ArgumentError` for invalid options,
nonfinite values, or degenerate PCA/random factors, and a `DimensionMismatch`
for incorrectly shaped custom means. ARD priors have shape=rate=1e-14 and
posteriors shape=rate=1. Inclusion hyperparameters start at Beta(1,1), while
individual loading inclusions start at one, as in the reference builder.
This internal initializer keeps dense defaults. The public `mofa` function
supplies its own model defaults and runs training; this initializer does not.
For k ≥ 8 and n*sum(pᵢ)*k ≥ 10^7, it reserves private scratch for up to
min(Threads.nthreads(), n, sum(pᵢ), 8) workers. Single-threaded and smaller fits
keep serial scratch only. The first worker reuses the serial suffix/coefficient
buffers; extra workers add O(max(n,sum(pᵢ))*k) storage each
"""
function _mofa_initialize(data::MofaData{Float64}, k::Int; init::Symbol = :pca,
	rng::AbstractRNG = Random.GLOBAL_RNG, z_init = nothing, w_init = nothing,
	ard_weights::Bool = false, ard_factors::Bool = false,
	spikeslab_weights::Bool = false)
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
	tau_logmean = Vector{Matrix{Float64}}(undef, nviews)
	spike_w = spikeslab_weights ? Vector{MofaSpikeSlab{Float64}}(undef, nviews) : nothing
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
		if spikeslab_weights
			spike_w[m] = MofaSpikeSlab(copy(wmean), ones(p, k), ones(k), ones(p, k))
			_mofa_spike_moments!(w[m], spike_w[m])
		end
		tau_shape[m] = ones(ngroups, p)
		tau_rate[m] = ones(ngroups, p)
		tau_mean[m] = Matrix{Float64}(undef, ngroups, p)
		tau_logmean[m] = Matrix{Float64}(undef, ngroups, p)
		_mofa_gamma_moments!(tau_mean[m], tau_logmean[m], tau_shape[m], tau_rate[m])
	end
	alpha_w = ard_weights ? _mofa_gamma_moments!(MofaGamma(ones(nviews, k),
		ones(nviews, k), ones(nviews, k), zeros(nviews, k), 1e-14, 1e-14)) : nothing
	alpha_z = ard_factors ? _mofa_gamma_moments!(MofaGamma(ones(ngroups, k),
		ones(ngroups, k), ones(ngroups, k), zeros(ngroups, k), 1e-14, 1e-14)) : nothing
	theta_w = spikeslab_weights ? _mofa_beta_moments!(MofaBeta(ones(nviews, k),
		ones(nviews, k), zeros(nviews, k), zeros(nviews, k), zeros(nviews, k), 1.0, 1.0)) : nothing
	state = MofaState(z, w, tau_shape, tau_rate, tau_mean, 1e-3, 1e-3,
		tau_logmean, alpha_w, alpha_z, spike_w, theta_w)
	sums = k >= 8 ? zeros(max(n, sum(size(Y, 2) for Y in data.data)), k + 1) : nothing
	workspace = MofaWorkspace(zeros(n), zeros(maximum(size(Y, 2) for Y in data.data)), zeros(k), sums)
	# Parallelize only independent samples/features. Each worker owns its scratch;
	# no BLAS calls or floating-point reductions are shared between workers.
	total_features = sum(size(Y, 2) for Y in data.data)
	if sums !== nothing && Threads.nthreads() > 1 && min(n, total_features) > 1 && Float64(n) * total_features * k >= 1e7
		nworkers = min(Threads.nthreads(), n, total_features, 8)
		push!(workspace.parallel_scratch, (sums, workspace.factor_buffer))
		for _ in 2:nworkers
			push!(workspace.parallel_scratch, (zeros(size(sums)), zeros(k)))
		end
	end
	return state, workspace
end

"""
	_mofa_predict!(prediction::AbstractVector{Float64}, basis::Matrix{Float64},
		coefficients::Vector{Float64})

Multiply a factor basis by coordinate coefficients using reusable output storage
# Arguments
- `prediction::AbstractVector{Float64}`: Length-n destination, overwritten
- `basis::Matrix{Float64}`: The n×k sample-factor or loading-mean matrix
- `coefficients::Vector{Float64}`: Length-k coefficients; a zero coordinate is
  excluded from the prediction without subtracting an old contribution
# Value
The destination containing basis*coefficients. Uses column-contiguous SIMD
loops for small ranks, avoiding repeated tiny BLAS calls; larger ranks use
`mul!`. No temporary arrays are allocated. Inputs must not alias the destination.
Throws a `DimensionMismatch` when the supplied dimensions do not agree
"""
function _mofa_predict!(prediction::AbstractVector{Float64}, basis::Matrix{Float64},
	coefficients::Vector{Float64})
	length(prediction) == size(basis, 1) && length(coefficients) == size(basis, 2) ||
		throw(DimensionMismatch("MOFA prediction dimensions must match"))
	if size(basis, 2) > 16
		return mul!(prediction, basis, coefficients)
	end
	fill!(prediction, 0.0)
	@inbounds for component in axes(basis, 2)
		coefficient = coefficients[component]
		iszero(coefficient) && continue
		@simd for i in axes(basis, 1)
			prediction[i] += basis[i, component] * coefficient
		end
	end
	return prediction
end

"""
	_mofa_coordinate_sums!(sums::Matrix{Float64}, basis::Matrix{Float64},
		coefficients::Vector{Float64}, offset::Int = 0)

Prepare frozen suffixes and an empty updated prefix for one coordinate sweep
# Arguments
- `sums::Matrix{Float64}`: Reusable destination with at least offset+n rows
  and k+1 columns; rows outside the selected block are left unchanged
- `basis::Matrix{Float64}`: Fixed n×k sample factors or loading means
- `coefficients::Vector{Float64}`: The original length-k coordinate means
- `offset::Int`: Nonnegative row offset, used to concatenate multiple views
# Value
The destination with column one set to zero and column c+1 containing the
contribution of original coordinates c+1:k. The last column is zero. Updating
only the prefix after each new coordinate gives its exclusion prediction as
prefix+suffix, without ever subtracting that coordinate's old contribution.
Requires finite validated inputs and no input/output aliasing. Allocates no
temporary arrays. Throws a `DimensionMismatch` for insufficient scratch or
inconsistent basis/coefficient dimensions, or `ArgumentError` for negative offset
"""
function _mofa_coordinate_sums!(sums::Matrix{Float64}, basis::Matrix{Float64},
	coefficients::Vector{Float64}, offset::Int = 0)
	n, k = size(basis)
	offset >= 0 || throw(ArgumentError("MOFA coordinate offset must be nonnegative"))
	length(coefficients) == k && size(sums, 1) >= offset + n && size(sums, 2) >= k + 1 ||
		throw(DimensionMismatch("MOFA coordinate scratch dimensions must match"))
	@inbounds @simd for i in 1:n
		sums[offset + i, 1] = 0.0
		sums[offset + i, k + 1] = 0.0
	end
	@inbounds for component in k:-1:2
		coefficient = coefficients[component]
		@inbounds @simd for i in 1:n
			sums[offset + i, component] = sums[offset + i, component + 1] + basis[i, component] * coefficient
		end
	end
	return sums
end

"""
	_mofa_check_parallel_scratch(workspace::MofaWorkspace{Float64}, rows::Int, k::Int)

Validate private scratch before any parallel coordinate sweep
# Arguments
- `workspace::MofaWorkspace{Float64}`: Reusable serial and worker buffers
- `rows::Int`: Required prefix/suffix rows across both kinds of update
- `k::Int`: Number of factors; suffix storage requires at least k+1 columns
# Value
Nothing. Empty worker storage selects serial execution. Otherwise each worker
must have sufficient storage and length-k coefficients, with no aliasing within
or between workers. Throws a `DimensionMismatch` for insufficient storage or
an `ArgumentError` for aliased worker arrays. No arrays are modified
"""
function _mofa_check_parallel_scratch(workspace::MofaWorkspace{Float64}, rows::Int, k::Int)
	for worker in eachindex(workspace.parallel_scratch)
		sums, factors = workspace.parallel_scratch[worker]
		size(sums, 1) >= rows && size(sums, 2) >= k + 1 && length(factors) == k ||
			throw(DimensionMismatch("MOFA parallel scratch must fit samples, features, and factors"))
		Base.mightalias(sums, factors) && throw(ArgumentError("MOFA worker buffers must not alias"))
		for previous in 1:worker - 1
			other_sums, other_factors = workspace.parallel_scratch[previous]
			(Base.mightalias(sums, other_sums) || Base.mightalias(factors, other_factors) ||
				Base.mightalias(sums, other_factors) || Base.mightalias(factors, other_sums)) &&
				throw(ArgumentError("MOFA workers require independent scratch"))
		end
	end
	return nothing
end

"""
	_mofa_update_weights!(data::MofaData{Float64}, state::MofaState{Float64},
		workspace::MofaWorkspace{Float64})

Perform one full-batch coordinate sweep over dense or sparse loading distributions
# Arguments
- `data::MofaData{Float64}`: Preprocessed aligned views and missing-entry masks;
  only observed entries contribute to the update
- `state::MofaState{Float64}`: Current sample-factor means and second moments,
  loading distributions, and group-by-feature expected noise precisions. Factor
  moments must be current, as produced by `_mofa_gaussian_moments!`. Loading
  means, variances, and second moments are overwritten; factors and noise are
  held fixed
- `workspace::MofaWorkspace{Float64}`: Reusable length-n sample and length-k
  factor buffers plus prefix/suffix scratch, overwritten during the sweep;
  the feature buffer is unused
# Value
The supplied state after one loading sweep with zero-mean priors and no
stochastic damping. Prior precision A is the current ARD expectation when
enabled, otherwise one. Each active loading variance is
1/(A + sum(τ*E[z²])); its mean is that variance times sum(τ*E[z]*r), where r
excludes the current factor and sums are over observed samples. Coordinates are
updated in order, so later coordinates use earlier coordinates' new means.
With sparse weights, the inclusion log odds combine the current expected Beta
log odds, 0.5*log(A*variance), and 0.5*active_mean²/variance. Effective means γμ
are used immediately by later coordinates. Inactive variance is 1/A at this
update, not retroactively changed by a subsequent ARD update. This function
does not update ARD or Beta hyperparameters; callers control their order.
Features are conditionally independent, allowing feature-major traversal for
Julia's column-major data. Prediction excludes the current coordinate using
updated prefix and frozen suffix sums, never subtracting its old contribution.
No n×pᵢ residual or expanded precision matrix is allocated. Throws a
`DimensionMismatch` for incompatible dimensions or an `ArgumentError` for
invalid group indices, nonfinite means, nonpositive/nonfinite factor second
moments or noise precisions, or nonfinite update results. Numeric failure can
leave earlier loading coordinates updated; no rollback is attempted. This is
one update, not a complete MOFA fit. When parallel scratch is present, disjoint
loading rows are updated concurrently; no floating-point reduction crosses
workers. Numeric exceptions inside tasks are wrapped by Julia's thread scheduler
"""
function _mofa_update_weights!(data::MofaData{Float64}, state::MofaState{Float64},
	workspace::MofaWorkspace{Float64})
	n, k = size(state.z.mean)
	n >= 1 && k >= 1 || throw(ArgumentError("MOFA requires samples and factors"))
	size(state.z.variance) == size(state.z.second_moment) == (n, k) ||
		throw(DimensionMismatch("MOFA factor parameter and moment shapes must match"))
	length(workspace.sample_buffer) == n && length(workspace.factor_buffer) == k ||
		throw(DimensionMismatch("MOFA weight workspace requires length-n and length-k buffers"))
	workspace.coordinate_sums === nothing ||
		(size(workspace.coordinate_sums, 1) >= n && size(workspace.coordinate_sums, 2) >= k + 1) ||
		throw(DimensionMismatch("MOFA weight coordinate scratch must fit samples and factors"))
	nviews, ngroups = length(data.data), length(data.group_labels)
	nviews >= 1 && ngroups >= 1 || throw(ArgumentError("MOFA requires views and groups"))
	_mofa_check_gamma(state.alpha_w, (nviews, k))
	if state.spike_w === nothing
		state.theta_w === nothing || throw(ArgumentError("inclusion posterior requires sparse loading state"))
	else
		length(state.spike_w) == nviews && state.theta_w !== nothing ||
			throw(DimensionMismatch("sparse loading state and inclusion posterior must match views"))
		_mofa_check_beta(state.theta_w, (nviews, k))
	end
	length(data.missing_mask) == length(state.w) == length(state.tau_mean) == nviews ||
		throw(DimensionMismatch("MOFA data, masks, weights, and noise must have matching view counts"))
	length(data.group_indices) == n || throw(DimensionMismatch("MOFA group indices must match the sample count"))
	all(g -> 1 <= g <= ngroups, data.group_indices) ||
		throw(ArgumentError("MOFA sample group indices are out of range"))
	all(isfinite, state.z.mean) && all(x -> isfinite(x) && x > 0, state.z.second_moment) ||
		throw(ArgumentError("MOFA factor means must be finite and second moments finite and positive"))
	# Check every view before changing any loading, so shape errors do not cause
	# a partially updated earlier view. The cached factor moments are read only.
	for m in 1:nviews
		Y, W, tau = data.data[m], state.w[m], state.tau_mean[m]
		p = size(Y, 2)
		size(Y, 1) == n && size(data.missing_mask[m]) == size(Y) &&
			size(W.mean) == size(W.variance) == size(W.second_moment) == (p, k) &&
			size(tau) == (ngroups, p) ||
			throw(DimensionMismatch("MOFA view $m has incompatible data, weight, mask, or noise dimensions"))
		all(isfinite, W.mean) && all(x -> isfinite(x) && x > 0, tau) ||
			throw(ArgumentError("MOFA loading means must be finite and noise precisions finite and positive"))
		state.spike_w === nothing || _mofa_check_spike(state.spike_w[m], (p, k))
	end
	total_features = sum(size(Y, 2) for Y in data.data)
	_mofa_check_parallel_scratch(workspace, max(n, total_features), k)
	# Shared per-view sparse priors are initialized once, outside worker tasks.
	if state.spike_w !== nothing
		@inbounds for m in 1:nviews
			sparse = state.spike_w[m]
			for component in 1:k
				A = state.alpha_w === nothing ? 1.0 : state.alpha_w.mean[m, component]
				inactive = inv(A)
				isfinite(inactive) && inactive > 0 || throw(ArgumentError("inactive loading variance must be finite and positive"))
				sparse.inactive_variance[component] = inactive
			end
		end
	end
	if isempty(workspace.parallel_scratch)
		_mofa_weight_sweep!(data, state, workspace.sample_buffer, workspace.factor_buffer,
			workspace.coordinate_sums, 1:total_features)
	else
		nworkers = length(workspace.parallel_scratch)
		Threads.@threads for worker in 1:nworkers
			sums, factors = workspace.parallel_scratch[worker]
			features = (div((worker - 1) * total_features, nworkers) + 1):div(worker * total_features, nworkers)
			_mofa_weight_sweep!(data, state, workspace.sample_buffer, factors, sums, features)
		end
	end
	return state
end

"""
	_mofa_weight_sweep!(data, state, prediction, factors, sums, features)

Update a disjoint range of loading rows after `_mofa_update_weights!` validation
# Arguments
- `data::MofaData{Float64}`: Validated views, masks, and group indices
- `state::MofaState{Float64}`: State whose loading rows in features are overwritten
- `prediction::Vector{Float64}`: Length-n scratch, written only when sums is nothing
- `factors::Vector{Float64}`: Private length-k coefficient scratch
- `sums::Union{Nothing,Matrix{Float64}}`: Private validated prefix/suffix scratch
- `features::UnitRange{Int}`: Loading-row indices across concatenated views
# Value
The updated state. Callers validate dimensions and moments and initialize sparse
inactive variances before calling. Concurrent calls must use disjoint feature
ranges and independent coefficient/suffix buffers. All coordinate operations
within one loading row retain their serial order; no cross-worker reduction is
used. Cached calls do not read or write prediction
"""
function _mofa_weight_sweep!(data::MofaData{Float64}, state::MofaState{Float64},
	prediction::Vector{Float64}, factors::Vector{Float64}, sums,
	features::UnitRange{Int})
	n, k = size(state.z.mean)
	offset = 0
	@inbounds for m in eachindex(data.data)
		Y, mask, W, tau = data.data[m], data.missing_mask[m], state.w[m], state.tau_mean[m]
		sparse = state.spike_w === nothing ? nothing : state.spike_w[m]
		p = size(Y, 2)
		for j in max(1, first(features) - offset):min(p, last(features) - offset)
			for component in 1:k
				factors[component] = W.mean[j, component]
			end
			sums === nothing || _mofa_coordinate_sums!(sums, state.z.mean, factors)
			for component in 1:k
				if sums === nothing
					factors[component] = 0.0
					_mofa_predict!(prediction, state.z.mean, factors)
				end
				A = state.alpha_w === nothing ? 1.0 : state.alpha_w.mean[m, component]
				precision, numerator = A, 0.0
				if sums === nothing
					@inbounds @simd for i in 1:n
						if !mask[i, j]
							noise = tau[data.group_indices[i], j]
							precision += noise * state.z.second_moment[i, component]
							numerator += noise * state.z.mean[i, component] * (Y[i, j] - prediction[i])
						end
					end
				else
					@inbounds @simd for i in 1:n
						if !mask[i, j]
							noise = tau[data.group_indices[i], j]
							precision += noise * state.z.second_moment[i, component]
							other = sums[i, 1] + sums[i, component + 1]
							numerator += noise * state.z.mean[i, component] * (Y[i, j] - other)
						end
					end
				end
				isfinite(precision) && isfinite(numerator) ||
					throw(ArgumentError("MOFA weight update produced nonfinite sufficient statistics"))
				variance = inv(precision)
				mu = variance * numerator
				second = mu * mu + variance
				isfinite(mu) && isfinite(second) && variance > 0 ||
					throw(ArgumentError("MOFA weight update produced invalid Gaussian moments"))
				if sparse === nothing
					W.mean[j, component] = mu
					W.variance[j, component] = variance
					W.second_moment[j, component] = second
				else
					logodds = state.theta_w.logmean[m, component] - state.theta_w.logcomplement[m, component] +
						0.5 * (log(A) - log(precision)) + 0.5 * mu * numerator
					g = _mofa_logistic(logodds)
					sparse.mean[j, component] = mu
					sparse.variance[j, component] = variance
					sparse.inclusion[j, component] = g
					W.mean[j, component] = g * mu
					W.variance[j, component] = g * variance + g * (1 - g) * mu * mu
					W.second_moment[j, component] = g * second
				end
				updated = W.mean[j, component]
				if sums === nothing
					factors[component] = updated
				else
					@inbounds @simd for i in 1:n
						sums[i, 1] += state.z.mean[i, component] * updated
					end
				end
			end
		end
		offset += p
	end
	return state
end

"""
	_mofa_update_factors!(data::MofaData{Float64}, state::MofaState{Float64},
		workspace::MofaWorkspace{Float64})

Perform one full-batch coordinate sweep over dense Gaussian sample-factor distributions
# Arguments
- `data::MofaData{Float64}`: Preprocessed aligned views and missing-entry masks;
  only observed entries across all views contribute for each sample
- `state::MofaState{Float64}`: Current factor means, loading distributions, and
  group-by-feature expected noise precisions. Loading second moments must be
  current, as produced by `_mofa_gaussian_moments!` or `_mofa_update_weights!`.
  Factor means, variances, and second moments are overwritten; weights and
  noise are held fixed
- `workspace::MofaWorkspace{Float64}`: Reusable feature buffer of length at least
  max(pᵢ), length-k coefficients, and prefix/suffix scratch with sum(pᵢ) rows.
  The scratch and coefficients are overwritten; the sample buffer is unused
# Value
The supplied state after one dense-factor sweep with zero-mean priors, equal
view weights, and no stochastic damping. Prior precision A is the current
group/factor ARD expectation when enabled, otherwise one. Each factor
variance is 1/(A + sum(τ*E[w²])); its mean is that variance times sum(τ*E[w]*r),
summing over observed features in all views. The residual r excludes the current
factor, and later coordinates use earlier coordinates' newly updated means.
Sparse weights contribute their effective moments, not their active-slab means.
Factor spike-and-slab is not implemented; ARD itself is not updated here.
Samples are conditionally independent, so sample-major traversal preserves the
reference's coordinate updates. A sample missing every view returns to the
group's prior, with zero mean and variance 1/A. Prediction excluding the
current coordinate combines updated prefix and frozen suffix sums, without
subtract/add-back cancellation or an n×pᵢ residual/precision allocation. Throws a
`DimensionMismatch` for incompatible dimensions or an `ArgumentError` for invalid
group indices, nonfinite means, invalid loading second moments (zero is allowed
for sparse weights), or nonpositive/nonfinite
noise precisions, or nonfinite update results. Numeric failure can leave earlier
factor coordinates updated; no rollback is attempted. This is one update, not a
complete MOFA fit. When parallel scratch is present, disjoint sample rows are
updated concurrently with unchanged per-row arithmetic. Numeric exceptions
inside tasks are wrapped by Julia's thread scheduler
"""
function _mofa_update_factors!(data::MofaData{Float64}, state::MofaState{Float64},
	workspace::MofaWorkspace{Float64})
	n, k = size(state.z.mean)
	n >= 1 && k >= 1 || throw(ArgumentError("MOFA requires samples and factors"))
	size(state.z.variance) == size(state.z.second_moment) == (n, k) ||
		throw(DimensionMismatch("MOFA factor parameter and moment shapes must match"))
	length(workspace.factor_buffer) == k ||
		throw(DimensionMismatch("MOFA factor workspace requires a length-k factor buffer"))
	nviews, ngroups = length(data.data), length(data.group_labels)
	nviews >= 1 && ngroups >= 1 || throw(ArgumentError("MOFA requires views and groups"))
	_mofa_check_gamma(state.alpha_z, (ngroups, k))
	length(data.missing_mask) == length(state.w) == length(state.tau_mean) == nviews ||
		throw(DimensionMismatch("MOFA data, masks, weights, and noise must have matching view counts"))
	length(data.group_indices) == n || throw(DimensionMismatch("MOFA group indices must match the sample count"))
	all(g -> 1 <= g <= ngroups, data.group_indices) ||
		throw(ArgumentError("MOFA sample group indices are out of range"))
	all(isfinite, state.z.mean) || throw(ArgumentError("MOFA factor means must be finite"))
	# Check all views before changing factors, including the largest scratch need.
	for m in 1:nviews
		Y, W, tau = data.data[m], state.w[m], state.tau_mean[m]
		p = size(Y, 2)
		size(Y, 1) == n && size(data.missing_mask[m]) == size(Y) &&
			size(W.mean) == size(W.variance) == size(W.second_moment) == (p, k) &&
			size(tau) == (ngroups, p) ||
			throw(DimensionMismatch("MOFA view $m has incompatible data, weight, mask, or noise dimensions"))
		length(workspace.feature_buffer) >= p ||
			throw(DimensionMismatch("MOFA factor workspace must fit every view's features"))
		all(isfinite, W.mean) && all(x -> isfinite(x) && (x > 0 || (x == 0 && state.spike_w !== nothing)), W.second_moment) &&
			all(x -> isfinite(x) && x > 0, tau) ||
			throw(ArgumentError("MOFA loading means must be finite and second moments/noise precisions finite and positive"))
	end
	sums = workspace.coordinate_sums
	sums === nothing ||
		(size(sums, 1) >= sum(size(Y, 2) for Y in data.data) && size(sums, 2) >= k + 1) ||
		throw(DimensionMismatch("MOFA factor coordinate scratch must fit all views and factors"))
	_mofa_check_parallel_scratch(workspace, max(n, sum(size(Y, 2) for Y in data.data)), k)
	if isempty(workspace.parallel_scratch)
		_mofa_factor_sweep!(data, state, workspace.feature_buffer, workspace.factor_buffer, sums, 1:n)
	else
		nworkers = length(workspace.parallel_scratch)
		Threads.@threads for worker in 1:nworkers
			worker_sums, factors = workspace.parallel_scratch[worker]
			samples = (div((worker - 1) * n, nworkers) + 1):div(worker * n, nworkers)
			_mofa_factor_sweep!(data, state, workspace.feature_buffer, factors, worker_sums, samples)
		end
	end
	return state
end

"""
	_mofa_factor_sweep!(data, state, feature_buffer, factors, sums, samples)

Update a disjoint range of factor rows after `_mofa_update_factors!` validation
# Arguments
- `data::MofaData{Float64}`: Validated aligned views, masks, and group indices
- `state::MofaState{Float64}`: State whose sample rows in samples are overwritten
- `feature_buffer::Vector{Float64}`: Prediction scratch, written only without sums
- `factors::Vector{Float64}`: Private length-k coefficient scratch
- `sums::Union{Nothing,Matrix{Float64}}`: Private validated prefix/suffix scratch
- `samples::UnitRange{Int}`: Disjoint sample rows assigned to this sweep
# Value
The updated state. Dimensions/moments must already be validated. Concurrent calls
must use disjoint sample ranges and independent coefficient/suffix buffers.
The coordinate order and per-sample reductions are unchanged. Cached calls do
not read or write feature_buffer; weights and noise are read only
"""
function _mofa_factor_sweep!(data::MofaData{Float64}, state::MofaState{Float64},
	feature_buffer::Vector{Float64}, factors::Vector{Float64}, sums,
	samples::UnitRange{Int})
	n, k = size(state.z.mean)
	nviews = length(data.data)
	@inbounds for i in samples
		group = data.group_indices[i]
		for component in 1:k
			factors[component] = state.z.mean[i, component]
		end
		if sums !== nothing
			offset = 0
			for m in 1:nviews
				_mofa_coordinate_sums!(sums, state.w[m].mean, factors, offset)
				offset += size(state.w[m].mean, 1)
			end
		end
		for component in 1:k
			sums === nothing && (factors[component] = 0.0)
			A = state.alpha_z === nothing ? 1.0 : state.alpha_z.mean[group, component]
			precision, numerator = A, 0.0
			offset = 0
			for m in 1:nviews
				Y, mask, W, tau = data.data[m], data.missing_mask[m], state.w[m], state.tau_mean[m]
				p = size(Y, 2)
				prediction = @view feature_buffer[1:p]
				if sums === nothing
					_mofa_predict!(prediction, W.mean, factors)
					@inbounds @simd for j in 1:p
						if !mask[i, j]
							noise = tau[group, j]
							precision += noise * W.second_moment[j, component]
							numerator += noise * W.mean[j, component] * (Y[i, j] - prediction[j])
						end
					end
				else
					@inbounds @simd for j in 1:p
						if !mask[i, j]
							noise = tau[group, j]
							precision += noise * W.second_moment[j, component]
							other = sums[offset + j, 1] + sums[offset + j, component + 1]
							numerator += noise * W.mean[j, component] * (Y[i, j] - other)
						end
					end
				end
				offset += p
			end
			isfinite(precision) && isfinite(numerator) ||
				throw(ArgumentError("MOFA factor update produced nonfinite sufficient statistics"))
			variance = inv(precision)
			mu = variance * numerator
			second = mu * mu + variance
			isfinite(mu) && isfinite(second) && variance > 0 ||
				throw(ArgumentError("MOFA factor update produced invalid Gaussian moments"))
			state.z.mean[i, component] = mu
			state.z.variance[i, component] = variance
			state.z.second_moment[i, component] = second
			if sums === nothing
				factors[component] = mu
			else
				offset = 0
				for m in 1:nviews
					W = state.w[m]
					@inbounds @simd for j in axes(W.mean, 1)
						sums[offset + j, 1] += W.mean[j, component] * mu
					end
					offset += size(W.mean, 1)
				end
			end
		end
	end
	return state
end

"""
	_mofa_check_moments(data::MofaData{Float64}, state::MofaState{Float64})

Validate aligned Gaussian/effective moments before noise or ARD updates
# Arguments
- `data::MofaData{Float64}`: Views, masks, and sample-group indices
- `state::MofaState{Float64}`: Current factor and effective loading moments
# Value
Nothing. Throws a `DimensionMismatch` for inconsistent dimensions or an
`ArgumentError` for invalid groups, nonfinite means, or negative/nonfinite
variances or second moments. Sparse effective variances/moments may be zero.
All arrays are read only; moment consistency is the caller's responsibility
"""
function _mofa_check_moments(data::MofaData{Float64}, state::MofaState{Float64})
	n, k = size(state.z.mean)
	M, G = length(data.data), length(data.group_labels)
	n > 0 && k > 0 && M > 0 && G > 0 || throw(ArgumentError("MOFA requires samples, factors, views, and groups"))
	size(state.z.variance) == size(state.z.second_moment) == (n, k) &&
		length(data.group_indices) == n && length(data.missing_mask) == length(state.w) == M ||
		throw(DimensionMismatch("MOFA moment dimensions must match the data"))
	all(g -> 1 <= g <= G, data.group_indices) || throw(ArgumentError("MOFA group indices are out of range"))
	all(isfinite, state.z.mean) && all(x -> isfinite(x) && x >= 0, state.z.variance) &&
		all(x -> isfinite(x) && x >= 0, state.z.second_moment) || throw(ArgumentError("MOFA factor moments are invalid"))
	for m in 1:M
		Y, W = data.data[m], state.w[m]
		p = size(Y, 2)
		size(Y, 1) == n && size(data.missing_mask[m]) == size(Y) &&
			size(W.mean) == size(W.variance) == size(W.second_moment) == (p, k) ||
			throw(DimensionMismatch("MOFA loading moments must match each view"))
		all(isfinite, W.mean) && all(x -> isfinite(x) && x >= 0, W.variance) &&
			all(x -> isfinite(x) && x >= 0, W.second_moment) || throw(ArgumentError("MOFA loading moments are invalid"))
	end
	return nothing
end

"""
	_mofa_update_noise!(data::MofaData{Float64}, state::MofaState{Float64},
		workspace::MofaWorkspace{Float64})

Update Gaussian group-by-feature Gamma noise posteriors in one full-batch sweep
# Arguments
- `data::MofaData{Float64}`: Processed views and masks; counts are taken from
  current masks rather than assuming stored zeros are observations
- `state::MofaState{Float64}`: Current factor/effective-loading moments and
  positive noise-prior shape/rate; only the noise posterior and caches change
- `workspace::MofaWorkspace{Float64}`: Length-n sample and length-k factor
  buffers, overwritten; no full reconstruction or residual matrix is stored
# Value
The supplied state with shape = prior_shape + observed_count/2 and rate =
prior_rate + expected_squared_error/2. Expected error includes uncertainty:
residual² + sum(var(z)*E[w²] + E[z]²*var(w)). This nonnegative expression avoids
cancellation in the expanded squared-error identity and works for sparse
effective weights. Entirely missing group/feature blocks retain their priors.
Refreshes both E[τ] and E[log τ]. Throws a `DimensionMismatch` for invalid
dimensions or an `ArgumentError` for invalid moments/priors or nonfinite
statistics. Numeric failure may leave part of the noise state updated
"""
function _mofa_update_noise!(data::MofaData{Float64}, state::MofaState{Float64},
	workspace::MofaWorkspace{Float64})
	_mofa_check_moments(data, state)
	n, k = size(state.z.mean)
	M, G = length(data.data), length(data.group_labels)
	length(workspace.sample_buffer) == n && length(workspace.factor_buffer) == k ||
		throw(DimensionMismatch("noise workspace requires length-n and length-k buffers"))
	length(state.tau_shape) == length(state.tau_rate) == length(state.tau_mean) == length(state.tau_logmean) == M ||
		throw(DimensionMismatch("noise posterior view counts must match data"))
	isfinite(state.tau_prior_shape) && state.tau_prior_shape > 0 &&
		isfinite(state.tau_prior_rate) && state.tau_prior_rate > 0 || throw(ArgumentError("noise prior shapes/rates must be finite and positive"))
	for m in 1:M
		p = size(data.data[m], 2)
		size(state.tau_shape[m]) == size(state.tau_rate[m]) == size(state.tau_mean[m]) == size(state.tau_logmean[m]) == (G, p) ||
			throw(DimensionMismatch("noise posterior dimensions must match groups and features"))
	end
	@inbounds for m in 1:M
		Y, mask, W = data.data[m], data.missing_mask[m], state.w[m]
		fill!(state.tau_shape[m], state.tau_prior_shape)
		fill!(state.tau_rate[m], state.tau_prior_rate)
		for j in axes(Y, 2)
			for component in 1:k
				workspace.factor_buffer[component] = W.mean[j, component]
			end
			_mofa_predict!(workspace.sample_buffer, state.z.mean, workspace.factor_buffer)
			for i in 1:n
				mask[i, j] && continue
				residual = Y[i, j] - workspace.sample_buffer[i]
				error = residual * residual
				for component in 1:k
					error += state.z.variance[i, component] * W.second_moment[j, component] +
						state.z.mean[i, component]^2 * W.variance[j, component]
				end
				isfinite(error) || throw(ArgumentError("expected MOFA reconstruction error must be finite"))
				g = data.group_indices[i]
				state.tau_shape[m][g, j] += 0.5
				state.tau_rate[m][g, j] += 0.5 * error
			end
		end
		_mofa_gamma_moments!(state.tau_mean[m], state.tau_logmean[m], state.tau_shape[m], state.tau_rate[m])
	end
	return state
end

"""
	_mofa_update_ard!(data::MofaData{Float64}, state::MofaState{Float64};
		weights::Bool = true, factors::Bool = true)

Update enabled view-wise loading and group-wise factor ARD Gamma posteriors
# Arguments
- `data::MofaData{Float64}`: Aligned views and sample groups
- `state::MofaState{Float64}`: Current Gaussian/effective moments and optional
  ARD posteriors; only enabled ARD parameters and their caches are overwritten
- `weights::Bool`: Update loading ARD, if enabled; defaults to true
- `factors::Bool`: Update factor ARD, if enabled; defaults to true. Selectors
  allow loading ARD after W and factor ARD after Z, preserving the Python order
# Value
The supplied state. Each shape is prior_shape + number_of_values/2 and each
rate is prior_rate + sum(second_moments)/2. For sparse weights, use the
underlying second moment γ*(μ²+v)+(1-γ)*inactive_variance, not effective E[w²].
Factor ARD counts every sample in its group, including samples with missing
views. Disabled ARD is a no-op. Throws a `DimensionMismatch` or `ArgumentError`
for incompatible state or invalid moments/parameters. This does not rerun
weight/factor updates or change sparse inactive variances
"""
function _mofa_update_ard!(data::MofaData{Float64}, state::MofaState{Float64};
	weights::Bool = true, factors::Bool = true)
	(!weights || state.alpha_w === nothing) && (!factors || state.alpha_z === nothing) && return state
	_mofa_check_moments(data, state)
	n, k = size(state.z.mean)
	M, G = length(data.data), length(data.group_labels)
	_mofa_check_gamma(state.alpha_w, (M, k))
	_mofa_check_gamma(state.alpha_z, (G, k))
	if state.spike_w !== nothing
		length(state.spike_w) == M || throw(DimensionMismatch("sparse loading view counts must match data"))
		for m in 1:M
			_mofa_check_spike(state.spike_w[m], size(state.w[m].mean))
		end
	end
	if weights && state.alpha_w !== nothing
		q = state.alpha_w
		for m in 1:M, component in 1:k
			p = size(state.w[m].mean, 1)
			total = 0.0
			for j in 1:p
				if state.spike_w === nothing
					total += state.w[m].second_moment[j, component]
				else
					s = state.spike_w[m]
					g = s.inclusion[j, component]
					total += g * (s.mean[j, component]^2 + s.variance[j, component]) +
						(1 - g) * s.inactive_variance[component]
				end
			end
			q.shape[m, component] = q.prior_shape + 0.5 * p
			q.rate[m, component] = q.prior_rate + 0.5 * total
		end
		_mofa_gamma_moments!(q)
	end
	if factors && state.alpha_z !== nothing
		q = state.alpha_z
		for g in 1:G, component in 1:k
			total, count = 0.0, 0
			for i in 1:n
				data.group_indices[i] == g || continue
				total += state.z.second_moment[i, component]
				count += 1
			end
			q.shape[g, component] = q.prior_shape + 0.5 * count
			q.rate[g, component] = q.prior_rate + 0.5 * total
		end
		_mofa_gamma_moments!(q)
	end
	return state
end

"""
	_mofa_update_inclusions!(state::MofaState{Float64})

Update the Beta hyperparameters of sparse loading inclusions
# Arguments
- `state::MofaState{Float64}`: Sparse loading inclusions and their optional
  view-by-factor Beta posterior; other model parameters are held fixed
# Value
The supplied state with a = prior_a + sum(γ) and b = prior_b + sum(1-γ) per
view/factor, followed by refreshed Beta expectations. Dense models are a no-op.
Throws a `DimensionMismatch` or `ArgumentError` for incompatible or invalid
sparse state. Scheduling this update, including mofapy2's sparsity warm-up,
belongs to the later training loop; the loading inclusion γ is updated by
`_mofa_update_weights!` even when this hyperparameter update is not called
"""
function _mofa_update_inclusions!(state::MofaState{Float64})
	state.spike_w === nothing && state.theta_w === nothing && return state
	state.spike_w !== nothing && state.theta_w !== nothing || throw(ArgumentError("sparse weights require an inclusion posterior"))
	M, k = length(state.w), size(state.z.mean, 2)
	length(state.spike_w) == M || throw(DimensionMismatch("sparse loading view counts must match weights"))
	q = state.theta_w
	_mofa_check_beta(q, (M, k))
	for m in 1:M
		_mofa_check_spike(state.spike_w[m], size(state.w[m].mean))
	end
	for m in 1:M, component in 1:k
		a, b = q.prior_a, q.prior_b
		for j in axes(state.spike_w[m].inclusion, 1)
			g = state.spike_w[m].inclusion[j, component]
			a += g
			b += 1 - g
		end
		q.a[m, component], q.b[m, component] = a, b
	end
	_mofa_beta_moments!(q)
	return state
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

"""
	_mofa_gamma_elbo(shape, rate, expected, logexpected, prior_shape, prior_rate)

Calculate the Gamma log-prior minus log-posterior contribution to the MOFA ELBO
# Arguments
- `shape`, `rate`: Matching posterior parameter matrices, with rate parameterization
- `expected`, `logexpected`: Matching cached E[x] and E[log x]
- `prior_shape`, `prior_rate`: Positive common prior shape and rate
# Value
Float; minus the Gamma KL divergence, summed over entries. Does not modify
arrays. Throws a `DimensionMismatch` for inconsistent sizes or an
`ArgumentError` for invalid parameters, expectations, or nonfinite results
"""
function _mofa_gamma_elbo(shape, rate, expected, logexpected, prior_shape, prior_rate)
	size(shape) == size(rate) == size(expected) == size(logexpected) ||
		throw(DimensionMismatch("Gamma ELBO parameter dimensions must match"))
	isfinite(prior_shape) && prior_shape > 0 && isfinite(prior_rate) && prior_rate > 0 ||
		throw(ArgumentError("Gamma ELBO priors must be finite and positive"))
	constant = prior_shape * log(prior_rate) - loggamma(prior_shape)
	value = 0.0
	for i in eachindex(shape)
		a, b, e, le = shape[i], rate[i], expected[i], logexpected[i]
		isfinite(a) && a > 0 && isfinite(b) && b > 0 && isfinite(e) && e > 0 && isfinite(le) ||
			throw(ArgumentError("Gamma ELBO parameters and expectations are invalid"))
		value += constant - a * log(b) + loggamma(a) +
			(prior_shape - a) * le + (b - prior_rate) * e
	end
	isfinite(value) || throw(ArgumentError("Gamma ELBO overflowed"))
	return value
end

"""
	_mofa_beta_elbo(q::MofaBeta{Float64})

Calculate the Beta log-prior minus log-posterior contribution to the MOFA ELBO
# Arguments
- `q::MofaBeta{Float64}`: Inclusion posterior with valid cached expectations
# Value
Float; minus the summed Beta KL divergence, without changing q. Throws a
`DimensionMismatch` or `ArgumentError` for invalid parameters or nonfinite results
"""
function _mofa_beta_elbo(q::MofaBeta{Float64})
	_mofa_check_beta(q, size(q.a))
	prior_logbeta = loggamma(q.prior_a) + loggamma(q.prior_b) - loggamma(q.prior_a + q.prior_b)
	value = 0.0
	for i in eachindex(q.a)
		a, b = q.a[i], q.b[i]
		value += loggamma(a) + loggamma(b) - loggamma(a + b) - prior_logbeta +
			(q.prior_a - a) * q.logmean[i] + (q.prior_b - b) * q.logcomplement[i]
	end
	isfinite(value) || throw(ArgumentError("Beta ELBO overflowed"))
	return value
end

"""
	_mofa_elbo(data::MofaData{Float64}, state::MofaState{Float64},
		workspace::MofaWorkspace{Float64}; noise_updated::Bool = false)

Evaluate Gaussian MOFA's evidence lower bound and its individual node contributions
# Arguments
- `data::MofaData{Float64}`: Processed views, group indices, and missing masks
- `state::MofaState{Float64}`: Current posterior parameters and consistent moments
- `workspace::MofaWorkspace{Float64}`: Reusable length-n and length-k buffers;
  scratch may change, but data and posterior parameters are read only
- `noise_updated::Bool`: Use the reference's fast noise-rate identity only when
  `_mofa_update_noise!` was the last parameter update. Defaults to false, which
  explicitly evaluates expected squared errors and is valid at initialization
# Value
A named tuple `(y, w, z, tau, alpha_w, alpha_z, theta_w, total)` of finite
Float64 values. Disabled node contributions are zero. Includes observation
normalizing constants, Gaussian prior/entropy terms, Gamma/Beta KL terms, and
Bernoulli inclusion terms. No n×p reconstruction or residual is allocated.
The sparse-weight entropy follows mofapy2, using the current expected ARD
precision for inactive conditional variances, while the prior uses the cached
underlying second moments from the preceding W update. Throws a
`DimensionMismatch` or `ArgumentError` for incompatible or invalid state
"""
function _mofa_elbo(data::MofaData{Float64}, state::MofaState{Float64},
	workspace::MofaWorkspace{Float64}; noise_updated::Bool = false)
	_mofa_check_moments(data, state)
	n, k = size(state.z.mean)
	M, G = length(data.data), length(data.group_labels)
	length(workspace.sample_buffer) == n && length(workspace.factor_buffer) == k ||
		throw(DimensionMismatch("ELBO workspace requires length-n and length-k buffers"))
	_mofa_check_gamma(state.alpha_w, (M, k))
	_mofa_check_gamma(state.alpha_z, (G, k))
	length(state.tau_shape) == length(state.tau_rate) == length(state.tau_mean) == length(state.tau_logmean) == M ||
		throw(DimensionMismatch("ELBO noise view counts must match data"))
	(state.spike_w === nothing) == (state.theta_w === nothing) ||
		throw(ArgumentError("sparse ELBO requires both weight and inclusion state"))
	if state.spike_w !== nothing
		length(state.spike_w) == M || throw(DimensionMismatch("sparse ELBO view counts must match data"))
		_mofa_check_beta(state.theta_w, (M, k))
	end
	y, w, z, tau = 0.0, 0.0, 0.0, 0.0
	for m in 1:M
		Y, mask, W = data.data[m], data.missing_mask[m], state.w[m]
		p = size(Y, 2)
		size(state.tau_shape[m]) == size(state.tau_rate[m]) == size(state.tau_mean[m]) == size(state.tau_logmean[m]) == (G, p) ||
			throw(DimensionMismatch("ELBO noise dimensions must match groups and features"))
		tau += _mofa_gamma_elbo(state.tau_shape[m], state.tau_rate[m], state.tau_mean[m],
			state.tau_logmean[m], state.tau_prior_shape, state.tau_prior_rate)
		if noise_updated
			for j in 1:p, g in 1:G
				count = 2 * (state.tau_shape[m][g, j] - state.tau_prior_shape)
				y += 0.5 * count * (state.tau_logmean[m][g, j] - log(2pi)) -
					state.tau_mean[m][g, j] * (state.tau_rate[m][g, j] - state.tau_prior_rate)
			end
		else
			for j in 1:p
				for component in 1:k
					workspace.factor_buffer[component] = W.mean[j, component]
				end
				_mofa_predict!(workspace.sample_buffer, state.z.mean, workspace.factor_buffer)
				for i in 1:n
					mask[i, j] && continue
					error = (Y[i, j] - workspace.sample_buffer[i])^2
					for component in 1:k
						error += state.z.variance[i, component] * W.second_moment[j, component] +
							state.z.mean[i, component]^2 * W.variance[j, component]
					end
					g = data.group_indices[i]
					y += 0.5 * (state.tau_logmean[m][g, j] - log(2pi) - state.tau_mean[m][g, j] * error)
				end
			end
		end
		if state.spike_w === nothing
			for component in 1:k, j in 1:p
				A = state.alpha_w === nothing ? 1.0 : state.alpha_w.mean[m, component]
				logA = state.alpha_w === nothing ? 0.0 : state.alpha_w.logmean[m, component]
				v = W.variance[j, component]
				v > 0 || throw(ArgumentError("dense ELBO variances must be positive"))
				w += 0.5 * (logA - A * W.second_moment[j, component] + log(v) + 1)
			end
		else
			s, theta = state.spike_w[m], state.theta_w
			_mofa_check_spike(s, (p, k))
			for component in 1:k, j in 1:p
				g = s.inclusion[j, component]
				A = state.alpha_w === nothing ? 1.0 : state.alpha_w.mean[m, component]
				logA = state.alpha_w === nothing ? 0.0 : state.alpha_w.logmean[m, component]
				underlying = g * (s.mean[j, component]^2 + s.variance[j, component]) +
					(1 - g) * s.inactive_variance[component]
				w += 0.5 * (logA - A * underlying + 1 + g * log(s.variance[j, component]) - (1 - g) * log(A))
				# x*log(x) tends to zero at x=0; do not clip the posterior probability.
				w += g * theta.logmean[m, component] + (1 - g) * theta.logcomplement[m, component]
				g > 0 && (w -= g * log(g))
				g < 1 && (w -= (1 - g) * log1p(-g))
			end
		end
	end
	for component in 1:k, i in 1:n
		g = data.group_indices[i]
		A = state.alpha_z === nothing ? 1.0 : state.alpha_z.mean[g, component]
		logA = state.alpha_z === nothing ? 0.0 : state.alpha_z.logmean[g, component]
		v = state.z.variance[i, component]
		v > 0 || throw(ArgumentError("factor ELBO variances must be positive"))
		z += 0.5 * (logA - A * state.z.second_moment[i, component] + log(v) + 1)
	end
	aw = state.alpha_w
	alpha_w = aw === nothing ? 0.0 : _mofa_gamma_elbo(aw.shape, aw.rate, aw.mean, aw.logmean, aw.prior_shape, aw.prior_rate)
	az = state.alpha_z
	alpha_z = az === nothing ? 0.0 : _mofa_gamma_elbo(az.shape, az.rate, az.mean, az.logmean, az.prior_shape, az.prior_rate)
	theta_w = state.theta_w === nothing ? 0.0 : _mofa_beta_elbo(state.theta_w)
	total = y + w + z + tau + alpha_w + alpha_z + theta_w
	isfinite(total) || throw(ArgumentError("MOFA ELBO is nonfinite"))
	return (; y, w, z, tau, alpha_w, alpha_z, theta_w, total)
end

"""
	_mofa_convergence(delta::Float64, initial::Float64, token::Int, mode::Symbol)

Apply mofapy2's percentage-change convergence criterion to one eligible ELBO check
# Arguments
- `delta::Float64`: Change since the preceding ELBO check
- `initial::Float64`: Initial ELBO, the fixed normalization denominator
- `token::Int`: Confirmation counter, initialized to one and reset to one on failure
- `mode::Symbol`: `:fast`, `:medium`, or `:slow`, with percentage thresholds
  0.0005, 0.00005, and 0.000005 respectively
# Value
A tuple `(token, converged)`. Two consecutive qualifying checks confirm
convergence. The initial denominator is bounded below by eps(Float64) only
to handle an exactly zero initial ELBO. Throws an `ArgumentError` for invalid
inputs. Eligibility and minimum-iteration gates belong to the fitting loop
"""
function _mofa_convergence(delta::Float64, initial::Float64, token::Int, mode::Symbol)
	isfinite(delta) && isfinite(initial) && token in (1, 2) ||
		throw(ArgumentError("convergence requires finite ELBO values and token 1 or 2"))
	mode in (:fast, :medium, :slow) || throw(ArgumentError("convergence_mode must be :fast, :medium, or :slow"))
	threshold = mode === :fast ? 5e-4 : mode === :medium ? 5e-5 : 5e-6
	if 100 * abs(delta) / max(abs(initial), eps(Float64)) < threshold
		token += 1
		return token, token == 3
	end
	return 1, false
end

"""
	_mofa_variance_explained(data::MofaData{Float64}, state::MofaState{Float64},
		workspace::MofaWorkspace{Float64})

Calculate total-model and single-factor explained variation on observed entries
# Arguments
- `data::MofaData{Float64}`: Processed views, masks, and sample-group indices
- `state::MofaState{Float64}`: Fitted posterior factor/effective-loading means
- `workspace::MofaWorkspace{Float64}`: Length-n and length-k scratch buffers
# Value
A tuple `(total, factors)` containing a G×M matrix and G×M×k array of
1 - residual_sum_of_squares/data_sum_of_squares. Entirely missing or zero-energy
group/view blocks return NaN. Single-factor values are not additive or clipped.
Uses posterior means, not expected error including uncertainty. Does not change
data/state. Throws a `DimensionMismatch` or `ArgumentError` for invalid inputs
"""
function _mofa_variance_explained(data::MofaData{Float64}, state::MofaState{Float64},
	workspace::MofaWorkspace{Float64})
	_mofa_check_moments(data, state)
	n, k = size(state.z.mean)
	G, M = length(data.group_labels), length(data.data)
	length(workspace.sample_buffer) == n && length(workspace.factor_buffer) == k ||
		throw(DimensionMismatch("explained-variation workspace dimensions must match"))
	energy, residuals = zeros(G, M), zeros(G, M)
	factors = zeros(G, M, k)
	for m in 1:M, j in axes(data.data[m], 2)
		for component in 1:k
			workspace.factor_buffer[component] = state.w[m].mean[j, component]
		end
		_mofa_predict!(workspace.sample_buffer, state.z.mean, workspace.factor_buffer)
		for i in 1:n
			data.missing_mask[m][i, j] && continue
			g, y = data.group_indices[i], data.data[m][i, j]
			energy[g, m] += y * y
			residuals[g, m] += (y - workspace.sample_buffer[i])^2
			for component in 1:k
				factors[g, m, component] += (y - state.z.mean[i, component] * state.w[m].mean[j, component])^2
			end
		end
	end
	for m in 1:M, g in 1:G
		isfinite(energy[g, m]) && isfinite(residuals[g, m]) || throw(ArgumentError("explained variation overflowed"))
		residuals[g, m] = iszero(energy[g, m]) ? NaN : 1 - residuals[g, m] / energy[g, m]
		for component in 1:k
			isfinite(factors[g, m, component]) || throw(ArgumentError("single-factor explained variation overflowed"))
			factors[g, m, component] = iszero(energy[g, m]) ? NaN : 1 - factors[g, m, component] / energy[g, m]
		end
	end
	return residuals, factors
end

"""
	mofa(views::AbstractVector; k::Int = 10, groups = nothing,
		center_groups::Bool = true, scale_views::Bool = false,
		scale_groups::Bool = false, init::Symbol = :pca,
		ard_weights::Bool = true, ard_factors::Bool = false,
		spikeslab_weights::Bool = true, maxiter::Int = 1000,
		start_elbo::Int = 1, freq_elbo::Int = 5, start_sparsity::Int = 50,
		miniter::Int = 0, convergence_mode::Symbol = :fast,
		forceiter::Bool = false, verbose::Bool = false,
		rng::AbstractRNG = Random.GLOBAL_RNG, z_init = nothing, w_init = nothing)

Fit Gaussian multi-omics factor analysis using full-batch variational inference
# Arguments
- `views::AbstractVector`: One n×pᵢ real matrix per view, with aligned samples
  in rows and features in columns; missing and NaN entries are excluded
- `k::Int`: Number of latent factors, at least one; defaults to 10. PCA
  initialization requires k <= min(n-1, sum(pᵢ)) and nonzero selected components
- `groups`: Optional length-n sample-group labels; nothing means one group
- `center_groups::Bool`: Remove each group/feature mean; defaults to true
- `scale_views::Bool`: Scale each view by its observed population standard
  deviation, not separately per feature; defaults to false
- `scale_groups::Bool`: Similarly scale each group/view block; defaults to false
- `init::Symbol`: `:pca` (default), `:random`, or `:custom`
- `ard_weights::Bool`: Enable view/factor ARD; defaults to true
- `ard_factors::Bool`: Enable group/factor ARD; defaults to false
- `spikeslab_weights::Bool`: Enable sparse loadings; defaults to true. Sample
  factors remain dense. ARD shrinks factors but does not explicitly delete them
- `maxiter::Int`: Maximum number of actual update sweeps; defaults to 1000.
  For equal work, use iter=maxiter+1 in mofapy2, whose count includes initialization
- `start_elbo::Int`: First scheduled ELBO check, at least one; defaults to one
- `freq_elbo::Int`: Positive interval between checks; defaults to five
- `start_sparsity::Int`: First sweep updating Beta inclusion hyperparameters;
  defaults to 50. Individual loading inclusions are updated even before this
- `miniter::Int`: Nonnegative minimum sweep gate for convergence; defaults to zero
- `convergence_mode::Symbol`: `:fast`, `:medium`, or `:slow`; defaults to :fast
- `forceiter::Bool`: Perform exactly maxiter sweeps without early stopping;
  defaults to false. In this mode converged remains false, not assessed
- `verbose::Bool`: Log scheduled ELBO checks; defaults to false
- `rng::AbstractRNG`: Generator used only by random initialization
- `z_init`, `w_init`: Initial n×k factors and optional vector of pᵢ×k loadings
  with init=:custom; copied, not rescaled. Custom initialization requires z_init
# Value
A `Mofa` with posterior means/uncertainty, group preprocessing statistics,
ELBO history, convergence information, and observed-data explained variation.
Each sweep follows W -> AlphaW -> ThetaW -> Z -> AlphaZ -> Tau, omitting
disabled updates and delaying ThetaW until start_sparsity. The initial ELBO
uses explicit expected errors; later checks reuse the freshly updated noise
rates. Convergence uses two eligible consecutive checks of the percentage
change relative to the initial ELBO, after start_elbo and miniter. Inputs are
not modified. The result does not retain data, masks, or workspaces.

Only Float64 Gaussian, equal-view-weight, full-batch inference is supported.
Non-Gaussian likelihoods, sparse sample factors, SVI, GPU execution, MEFISTO,
and explicit factor dropping are not supported. For preprocessing or option
errors throws an `ArgumentError` or `DimensionMismatch`. Warns if maxiter is
reached without convergence unless forceiter=true
"""
function mofa(views::AbstractVector; k::Int = 10, groups = nothing,
	center_groups::Bool = true, scale_views::Bool = false, scale_groups::Bool = false,
	init::Symbol = :pca, ard_weights::Bool = true, ard_factors::Bool = false,
	spikeslab_weights::Bool = true, maxiter::Int = 1000, start_elbo::Int = 1,
	freq_elbo::Int = 5, start_sparsity::Int = 50, miniter::Int = 0,
	convergence_mode::Symbol = :fast, forceiter::Bool = false, verbose::Bool = false,
	rng::AbstractRNG = Random.GLOBAL_RNG, z_init = nothing, w_init = nothing)
	maxiter >= 1 && start_elbo >= 1 && freq_elbo >= 1 && start_sparsity >= 1 && miniter >= 0 ||
		throw(ArgumentError("maxiter, start_elbo, freq_elbo, and start_sparsity must be positive; miniter nonnegative"))
	convergence_mode in (:fast, :medium, :slow) || throw(ArgumentError("convergence_mode must be :fast, :medium, or :slow"))
	data = _mofa_preprocess(views; groups, center_groups, scale_views, scale_groups)
	state, workspace = _mofa_initialize(data, k; init, rng, z_init, w_init,
		ard_weights, ard_factors, spikeslab_weights)
	elbo = [_mofa_elbo(data, state, workspace).total]
	elbo_iterations = [0]
	# Reserve only the number of scheduled checks, rather than one state per sweep.
	checks = start_elbo > maxiter ? 0 : (maxiter - start_elbo) ÷ freq_elbo + 1
	sizehint!(elbo, checks + 2)
	sizehint!(elbo_iterations, checks + 2)
	token, niter, converged = 1, 0, false
	for iteration in 1:maxiter
		_mofa_update_weights!(data, state, workspace)
		_mofa_update_ard!(data, state; weights = true, factors = false)
		iteration >= start_sparsity && _mofa_update_inclusions!(state)
		_mofa_update_factors!(data, state, workspace)
		_mofa_update_ard!(data, state; weights = false, factors = true)
		_mofa_update_noise!(data, state, workspace)
		niter = iteration
		if iteration >= start_elbo && (iteration - start_elbo) % freq_elbo == 0
			value = _mofa_elbo(data, state, workspace; noise_updated = true).total
			delta = value - elbo[end]
			push!(elbo, value)
			push!(elbo_iterations, iteration)
			verbose && @info "MOFA ELBO" iteration elbo=value delta
			if !forceiter && iteration > start_elbo && iteration > miniter
				token, converged = _mofa_convergence(delta, elbo[1], token, convergence_mode)
				converged && break
			end
		end
	end
	if elbo_iterations[end] != niter
		push!(elbo, _mofa_elbo(data, state, workspace; noise_updated = true).total)
		push!(elbo_iterations, niter)
	end
	r2, factor_r2 = _mofa_variance_explained(data, state, workspace)
	options = (; k, center_groups, scale_views, scale_groups, init, ard_weights,
		ard_factors, spikeslab_weights, maxiter, start_elbo, freq_elbo,
		start_sparsity, miniter, convergence_mode, forceiter, verbose)
	converged || forceiter || @warn "MOFA reached maxiter=$maxiter before convergence; increase maxiter or inspect the ELBO history."
	return Mofa(state.z.mean, [q.mean for q in state.w], state, data.group_indices,
		data.group_labels, data.means, data.scales, elbo, elbo_iterations, niter,
		converged, r2, factor_r2, options)
end
