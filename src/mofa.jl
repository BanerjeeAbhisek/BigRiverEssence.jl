# Implementation Note:
# Gaussian preprocessing follows mofapy2's group-wise centering, followed by
# optional view and group scaling (population standard deviations).
# https://github.com/bioFAM/mofapy2/blob/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7/mofapy2/build_model/utils.py
# This first implementation step prepares data only; it does not fit MOFA.

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
