"""
	Splsda{T}

Container for a fitted sparse PLS discriminant analysis, as returned by `splsda`
# Fields
- `variates_X::Matrix{T}`: The n×ncomp X-scores (sample coordinates for plotting/classification)
- `variates_Y::Matrix{T}`: The n×ncomp Y-scores (the class-side variates)
- `loadings_X::Matrix{T}`: The p×ncomp sparse X-loadings; the nonzero entries per
  column are the variables selected for that component
- `loadings_Y::Matrix{T}`: The K×ncomp Y-loadings (K = number of classes)
- `ncomp::Int`: The number of components fitted
- `keepX::Vector{Int}`: The number of variables kept per component (the sparsity budget)
- `Y_dummy::Matrix{T}`: The n×K one-hot encoding of the class labels
- `classes::Vector`: The class labels, in the column order of Y_dummy
"""
struct Splsda{T}
	variates_X::Matrix{T}
	variates_Y::Matrix{T}
	loadings_X::Matrix{T}
	loadings_Y::Matrix{T}
	ncomp::Int
	keepX::Vector{Int}
	Y_dummy::Matrix{T}
	classes::Vector
end

"""
	_center_scale(M::AbstractMatrix; scale = true)

Center, and optionally scale, the columns of a matrix
# Arguments
- `M::AbstractMatrix`: 2d array of floats; the matrix to center/scale
- `scale::Bool`: Whether to divide each column by its (n−1) standard deviation
  after centering. Defaults to true
# Value
2d array of floats; the column-centered (and, if scale=true, unit-variance)
matrix. Constant columns (zero standard deviation) are guarded: their scale is
treated as 1 and the column is set to all zeros rather than producing NaN. The
n−1 (corrected) standard deviation matches mixOmics' colSds
"""
function _center_scale(M::AbstractMatrix; scale = true)
	Mc = M .- mean(M, dims = 1)
	if scale
		s = std(M, dims = 1; corrected = true)         # n−1 SD, matching mixOmics
		s[s .== 0] .= 1.0                            # guard: don't divide constant columns by 0
		Mc = Mc ./ s
		zerocols = vec(std(M, dims = 1; corrected = true) .== 0)
		Mc[:, zerocols] .= 0.0                       # a constant column carries no information
	end
	return Mc
end

"""
	_unmap(y::AbstractVector; levels = nothing)

One-hot encode a vector of class labels into a dummy indicator matrix
# Arguments
- `y::AbstractVector`: The class labels, one per observation
- `levels`: An optional ordering of the classes; when given, it fixes the column
  order of the output and must list each class exactly once. Defaults to nothing
  (classes sorted by `sort(unique(y))`)
# Value
A tuple `(Yd, classes)` where `Yd` is the n×K one-hot matrix (`Yd[i,k] = 1` when
observation i is in class k, else 0) and `classes` is the class labels in column
order. This is mixOmics' `unmap`, the dummy encoding that turns classification
into a PLS regression onto the indicator matrix
"""
function _unmap(y::AbstractVector; levels = nothing)
	classes = levels === nothing ? sort(unique(y)) : collect(levels)   # column order of the dummy matrix
	length(classes) == length(unique(y)) ||
		throw(ArgumentError("`levels` must list each class exactly once (got $(length(classes)) levels for $(length(unique(y))) classes)"))
	k = length(classes)
	Yd = zeros(Float64, length(y), k)
	for (i, yi) in enumerate(y)
		idx = findfirst(==(yi), classes)
		idx === nothing && throw(ArgumentError("class $(yi) not found in supplied `levels`"))
		Yd[i, idx] = 1.0
	end
	return Yd, classes
end

"""
	_soft_threshold_L1!(out, x, nx::Int, absx, ord, ranks)

Soft-threshold a loading vector to keep only its largest-magnitude entries,
writing the result into a preallocated buffer
# Arguments
- `out`: 1d array of floats; the destination buffer, overwritten in place
- `x`: 1d array of floats; the loading vector to threshold
- `nx::Int`: The number of SMALLEST-magnitude entries to drop (= p − keepX). If
  nx ≤ 0 the vector is copied through unchanged
- `absx`: 1d array of floats; preallocated scratch for the entry magnitudes
- `ord`: 1d array of ints; preallocated scratch for the sort permutation
- `ranks`: 1d array of ints; preallocated scratch for the magnitude ranks
# Value
The buffer `out`, holding the thresholded vector: the `keepX = p − nx` largest
entries survive (shrunk by λ = the largest dropped magnitude), the rest are
zeroed. Implements the L1 penalty P_λ(u) = sign(u)(|u| − λ)₊ of Lê Cao et al.
(2011), reparameterized as a keep-count rather than a λ (mixOmics
`soft_thresholding_L1`). The rank computation handles magnitude ties so that
tied entries are dropped or kept together
"""
function _soft_threshold_L1!(out, x, nx::Int, absx, ord, ranks)
	p = length(x)
	if nx <= 0
		copyto!(out, x)                            # keep everything: nothing to drop
		return out
	end
	@. absx = abs(x)
	sortperm!(ord, absx)                           # ascending order of magnitudes
	fill!(ranks, 0)
	# assign each entry a rank = its position in ascending magnitude, ties sharing
	# the highest position in the tie group (so ties are dropped/kept as a block)
	i = 1
	while i <= p
		j = i
		while j < p && absx[ord[j+1]] == absx[ord[i]]
			j += 1
		end
		for m in i:j
			;
			ranks[ord[m]] = j;
		end
		i = j + 1
	end
	# λ = the largest magnitude among the entries we're dropping (the nx smallest)
	lambda = 0.0;
	anydrop = false
	@inbounds for t in 1:p
		if ranks[t] <= nx
			anydrop = true
			absx[t] > lambda && (lambda = absx[t])
		end
	end
	if !anydrop                                    # ties pushed everything above nx ⇒ drop nothing
		copyto!(out, x)
		return out
	end
	# survivors (rank > nx) shrink by λ; dropped entries (rank ≤ nx) become 0
	@inbounds for t in 1:p
		out[t] = ranks[t] > nx ? sign(x[t]) * (absx[t] - lambda) : 0.0
	end
	return out
end

"""
	_sqdiff(a, b)

Compute the squared ℓ₂ distance between two vectors
# Arguments
- `a`: 1d array of floats
- `b`: 1d array of floats
# Value
Float; Σ(aᵢ − bᵢ)². Used as the convergence criterion between successive loading
iterates
"""
function _sqdiff(a, b)
	s = zero(eltype(a))
	@inbounds @simd for i in eachindex(a)
		d = a[i] - b[i];
		s += d * d
	end
	return s
end

"""
	splsda(X::Matrix{Float64}, y::Vector, ncomp::Int, keepX::Vector{Int};
		   scale = true, tol = 1e-6, max_iter = 100, levels = nothing)

Fit a sparse PLS discriminant analysis (sPLS-DA) for multiclass problems, after
Lê Cao, Boitard & Besse (2011)
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the n×p predictor matrix
- `y::Vector`: The class labels, one per observation (n total)
- `ncomp::Int`: The number of components to extract
- `keepX::Vector{Int}`: The number of variables to keep per component; must have
  length ncomp (one sparsity budget per component)
- `scale::Bool`: Whether to scale columns to unit variance in addition to
  centering. Defaults to true
- `tol::Real`: The convergence tolerance on the loading iterates. Defaults to 1e-6
- `max_iter::Int`: The maximum number of inner iterations per component.
  Defaults to 100
- `levels`: An optional class ordering passed to the dummy encoding; fixes the
  column order of the classes. Defaults to nothing (sorted unique labels)
# Value
An `Splsda` holding the X and Y variates (scores), the sparse X and
dense Y loadings, the number of components, the per-component keepX, the one-hot
class encoding, and the class labels. The class labels are dummy-encoded into an
indicator matrix and the problem solved as a sparse PLS regression of X onto that
matrix: each component is the penalized rank-1 approximation of the cross-product
Mₕ = XₕᵀYₕ, solved by alternating soft-thresholded power iteration, after which X
(only) is deflated by regression on its variate (Y is not deflated — the
discriminant regression mode)
"""
function splsda(X::Matrix{Float64}, y::Vector, ncomp::Int, keepX::Vector{Int};
	scale = true, tol = 1e-6, max_iter = 100, levels = nothing)
	n, p = size(X)
	Yd, classes = _unmap(y; levels = levels)             # one-hot encode the labels (Step 0a)
	k = size(Yd, 2)
	length(keepX) == ncomp || throw(ArgumentError("keepX must have length ncomp"))

	# center/scale both blocks (Step 0b)
	Xc = _center_scale(Matrix{Float64}(X); scale = scale)
	Yc = _center_scale(Yd; scale = scale)

	TX = zeros(n, ncomp);
	TY = zeros(n, ncomp)         # X- and Y-variates (scores)
	PX = zeros(p, ncomp);
	PY = zeros(k, ncomp)         # X- (sparse) and Y-loadings

	R  = copy(Xc)                                      # X residual, deflated each component
	Ry = copy(Yc)                                      # Y stays fixed (no deflation in DA mode)

	# scratch reused across components
	M = Matrix{Float64}(undef, p, k)
	uh = Vector{Float64}(undef, p);
	uh_old = Vector{Float64}(undef, p)
	vh = Vector{Float64}(undef, k);
	vh_old = Vector{Float64}(undef, k)
	tX = Vector{Float64}(undef, n)
	tY = Vector{Float64}(undef, n)
	uraw = Vector{Float64}(undef, p)                  # X-loading before thresholding
	pX = Vector{Float64}(undef, p)
	absx = Vector{Float64}(undef, p)
	ord = Vector{Int}(undef, p)
	ranks = Vector{Int}(undef, p)

	for comp in 1:ncomp
		# Step 1: initialize uh, vh from the leading singular vectors of the
		# current (deflated) cross-product Mₕ = RᵀRy
		mul!(M, transpose(R), Ry)
		F = svd(M)
		copyto!(uh, @view F.U[:, 1])
		copyto!(vh, @view F.V[:, 1])
		copyto!(uh_old, uh);
		copyto!(vh_old, vh)

		# Step 2: alternating soft-thresholded power iteration until the loadings settle
		iter = 1
		while true
			mul!(tY, Ry, vh)                           # Y-variate  tY = Ry·vh
			mul!(uraw, transpose(R), tY)               # raw X-loading  Rᵀ·tY
			_soft_threshold_L1!(uh, uraw, p - keepX[comp], absx, ord, ranks)  # L1 penalty (keep keepX)
			uh ./= sqrt(sum(abs2, uh))                 # normalize
			mul!(tX, R, uh)                            # X-variate  tX = R·uh
			mul!(vh, transpose(Ry), tX)                # Y-loading  Ryᵀ·tX
			vh ./= sqrt(sum(abs2, vh))                 # normalize (no penalty — DA keeps all classes)

			dX = _sqdiff(uh, uh_old)
			dY = _sqdiff(vh, vh_old)
			(max(dX, dY) < tol || iter > max_iter) && break
			copyto!(uh_old, uh);
			copyto!(vh_old, vh)
			iter += 1
		end

		# Step 3: store this component's variates and loadings
		mul!(tX, R, uh);
		mul!(tY, Ry, vh)
		@views TX[:, comp] .= tX;
		@views TY[:, comp] .= tY
		@views PX[:, comp] .= uh;
		@views PY[:, comp] .= vh

		# deflate X ONLY, by regression on its variate (Y is left untouched)
		mul!(pX, transpose(R), tX)
		pX ./= dot(tX, tX)                             # pₕ = Rᵀ·tX / ‖tX‖²
		BLAS.ger!(-1.0, tX, pX, R)                     # R ← R − tX·pₕᵀ
	end

	return Splsda(TX, TY, PX, PY, ncomp, keepX, Yd, classes)
end
