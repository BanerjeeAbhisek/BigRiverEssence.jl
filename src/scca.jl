"""
	Scca{T}

Container for a fitted sparse canonical correlation analysis, as returned by `scca`
# Fields
- `u::Matrix{T}`: The dx×K sparse X canonical vectors (one unit-ℓ₂, L1-penalized
  column per component); nonzero entries are the selected X features
- `v::Matrix{T}`: The dy×K sparse Y canonical vectors; nonzero entries are the
  selected Y features
- `d::Vector{T}`: The K weights dₖ = uₖᵀ Xᵀ Z vₖ (the penalized-covariance objective values)
- `cors::Vector{T}`: The K sample correlations between the paired canonical variates
- `penaltyx::Float64`: The L1 penalty applied to the X vectors, in (0,1]
- `penaltyz::Float64`: The L1 penalty applied to the Y vectors, in (0,1]
- `K::Int`: The number of components
"""
struct Scca{T}
	u::Matrix{T}
	v::Matrix{T}
	d::Vector{T}
	cors::Vector{T}
	penaltyx::Float64
	penaltyz::Float64
	K::Int
end

"""
	_softcca!(out, a, d)

Soft-threshold a vector into a preallocated buffer
# Arguments
- `out`: 1d array of floats; the destination buffer, overwritten in place
- `a`: 1d array of floats; the vector to threshold
- `d`: Float; the nonnegative threshold
# Value
The buffer `out`, holding sign(aᵢ)·max(|aᵢ| − d, 0). The elementwise L1-penalized
update of Witten, Tibshirani & Hastie (2009)
"""
function _softcca!(out, a, d)
	@inbounds @simd for i in eachindex(a)
		ai = a[i]
		out[i] = sign(ai) * max(abs(ai) - d, 0.0)
	end
	return out
end

"""
	_l2n_val(a)

Compute the ℓ₂ norm of a vector with a zero-guard
# Arguments
- `a`: 1d array of floats
# Value
Float; the Euclidean norm ‖a‖₂, or 0.05 when a is all zeros. The guard mirrors
PMA's convention and prevents division by zero in the normalization steps
"""
function _l2n_val(a)
	s = 0.0
	@inbounds @simd for i in eachindex(a)
		s += a[i] * a[i]
	end
	r = sqrt(s)
	return r == 0 ? 0.05 : r
end

"""
	_l1_of_norm_soft(a, d)

Compute the ℓ₁/ℓ₂ ratio of a vector AFTER soft-thresholding it by d, without
materializing the thresholded vector
# Arguments
- `a`: 1d array of floats; the unthresholded vector
- `d`: Float; the soft-threshold to apply
# Value
Float; ‖S(a,d)‖₁ / ‖S(a,d)‖₂ where S is the soft-threshold operator. Used inside
the binary search to test whether a candidate threshold meets the L1 budget
"""
function _l1_of_norm_soft(a, d)
	s2 = 0.0;
	s1 = 0.0
	@inbounds @simd for i in eachindex(a)
		si = sign(a[i]) * max(abs(a[i]) - d, 0.0)
		s2 += si * si
		s1 += abs(si)
	end
	nrm = sqrt(s2)
	nrm == 0 && (nrm = 0.05)
	return s1 / nrm
end

"""
	_l1_of_norm(a)

Compute the ℓ₁/ℓ₂ ratio of a vector
# Arguments
- `a`: 1d array of floats
# Value
Float; ‖a‖₁ / ‖a‖₂. This ratio is the quantity the L1 budget constrains; it
ranges from 1 (maximally sparse) to √length(a) (fully dense)
"""
function _l1_of_norm(a)
	nrm = _l2n_val(a)
	s1 = 0.0
	@inbounds @simd for i in eachindex(a)
		s1 += abs(a[i])
	end
	return s1 / nrm
end

"""
	_l1diff(a, b)

Compute the ℓ₁ distance between two vectors
# Arguments
- `a`: 1d array of floats
- `b`: 1d array of floats
# Value
Float; ‖a − b‖₁. Used as the convergence criterion between successive iterates
"""
function _l1diff(a, b)
	s = 0.0
	@inbounds @simd for i in eachindex(a)
		s += abs(a[i] - b[i])
	end
	return s
end

"""
	_binary_search_opt(argu, sumabs)

Find the soft-threshold that makes a soft-thresholded vector meet a target L1
budget
# Arguments
- `argu`: 1d array of floats; the vector to be thresholded
- `sumabs`: Float; the target ℓ₁/ℓ₂ ratio (the L1 budget)
# Value
Float; the threshold d ≥ 0 such that ‖S(argu,d)‖₁/‖S(argu,d)‖₂ ≈ sumabs. Returns
0 when the vector already satisfies the budget. Implements the bisection step 
 bracketing d in [0, max|argu|] and halving
for up to 150 iterations or until the bracket is below 1e-6
"""
function _binary_search_opt(argu, sumabs)
	(_l2n_val(argu) == 0 || _l1_of_norm(argu) <= sumabs) && return 0.0
	lam1 = 0.0
	lam2 = maximum(abs, argu) - 1e-5
	iter = 1
	while iter < 150
		mid = (lam1 + lam2) / 2
		if _l1_of_norm_soft(argu, mid) < sumabs
			lam2 = mid
		else
			lam1 = mid
		end
		(lam2 - lam1) < 1e-6 && return (lam1 + lam2) / 2
		iter += 1
	end
	return (lam1 + lam2) / 2
end

"""
	_matsqrt(A)

Compute the symmetric square root of a symmetric positive-semidefinite matrix
# Arguments
- `A`: 2d array of floats; a symmetric (positive-semidefinite) matrix
# Value
2d array of floats; the symmetric matrix R such that R·R = A, formed from the
eigendecomposition with eigenvalues floored at zero (so tiny negative roundoff
eigenvalues don't produce complex results)
"""
function _matsqrt(A)
	E = eigen(Symmetric(A))
	vals = sqrt.(max.(E.values, 0.0))             # floor at 0 to avoid sqrt of roundoff negatives
	return E.vectors * Diagonal(vals) * transpose(E.vectors)
end

"""
	_fast_init_v(x, z, K)

Initialize the Y canonical vectors for wide data via an SVD construction
# Arguments
- `x`: 2d array of floats; the (observations × features) X block
- `z`: 2d array of floats; the (observations × features) Z block
- `K`: Int; the number of components to initialize
# Value
2d array of floats; the K leading left singular vectors of zᵀ·(xxᵀ)^½, used as the
starting v for the sparse iteration when both blocks are wider than they are tall.
Building the init from xxᵀ (observations × observations) keeps the SVD small in
the high-dimensional case
"""
function _fast_init_v(x, z, K)
	xx = x * transpose(x)                         # observations × observations (small when wide)
	xx_sqrt = _matsqrt(xx)
	y = transpose(z) * xx_sqrt
	F = svd(y)
	return F.U[:, 1:K]
end

"""
	_sparse_cca_single_opt!(u, v, x, z, v0, penaltyx, penaltyz, niter,
							vold, zv_buf, xu_buf, argu, argv, su, sv)

Compute one rank-1 sparse canonical vector pair by alternating, soft-thresholded
power iteration
# Arguments
- `u`: 1d array of floats; the X canonical vector, written in place
- `v`: 1d array of floats; the Z canonical vector, written in place
- `x`: 2d array of floats; the (observations × features) X block for this component
- `z`: 2d array of floats; the (observations × features) Z block for this component
- `v0`: 1d array of floats; the initial v (from `_fast_init_v` or an SVD)
- `penaltyx`: Float; the X penalty in (0,1], scaled internally to the budget penaltyx·√(#X features)
- `penaltyz`: Float; the Z penalty in (0,1], scaled to penaltyz·√(#Z features)
- `niter`: Int; the maximum number of power-iteration steps
- `vold`: 1d array of floats; scratch holding the previous v iterate
- `zv_buf`, `xu_buf`: 1d arrays of floats; scratch for z·v and x·u (sized for the
  deflated block height)
- `argu`, `argv`: 1d arrays of floats; scratch for the raw (pre-threshold) canonical vectors
- `su`, `sv`: 1d arrays of floats; scratch for the soft-thresholded vectors
# Value
Float; the weight d = uᵀ xᵀ z v = ⟨x u, z v⟩. On return `u` and `v` hold the
unit-ℓ₂, L1-penalized canonical vectors. Each step alternates u ← normalize(S(xᵀz v, λᵤ))
and v ← normalize(S(zᵀx u, λᵥ)), with the thresholds chosen by `_binary_search_opt`
to meet the penalty budgets — the penalized-CCA update of Witten, Tibshirani &
Hastie (2009). The randn primer on `vold` only forces the first iteration to run
and does not affect the converged fixed point
"""
function _sparse_cca_single_opt!(u, v, x, z, v0, penaltyx, penaltyz, niter,
	vold, zv_buf, xu_buf, argu, argv, su, sv)
	nr = size(x, 1)                               # current block height (grows with deflation)
	p1 = size(x, 2);
	p2 = size(z, 2)
	c1 = penaltyx * sqrt(p1)                      # L1 budget on u
	c2 = penaltyz * sqrt(p2)                      # L1 budget on v
	zv = @view zv_buf[1:nr]                       # views sized to the current block height
	xu = @view xu_buf[1:nr]
	copyto!(v, v0)
	@inbounds for i in eachindex(vold)
		;
		vold[i] = randn();
	end   # primer: forces the first iteration to run
	fill!(u, 0.0)
	for _ in 1:niter
		if _l1diff(vold, v) > 1e-6
			mul!(zv, z, v)                        # z v
			mul!(argu, transpose(x), zv)         # xᵀ(z v)  → raw u
			lamu = _binary_search_opt(argu, c1)
			_softcca!(su, argu, lamu)            # soft-threshold to the X budget
			su ./= _l2n_val(su)                  # normalize
			copyto!(u, su)
			copyto!(vold, v)
			mul!(xu, x, u)                        # x u
			mul!(argv, transpose(z), xu)         # zᵀ(x u)  → raw v
			lamv = _binary_search_opt(argv, c2)
			_softcca!(sv, argv, lamv)            # soft-threshold to the Z budget
			sv ./= _l2n_val(sv)                  # normalize
			copyto!(v, sv)
		end
	end
	mul!(zv, z, v);
	mul!(xu, x, u)
	return dot(xu, zv)                           # d = ⟨x u, z v⟩
end

"""
	scca(X::Matrix{Float64}, Y::Matrix{Float64}; penaltyx::Real = 0.3,
		 penaltyz::Real = 0.3, K::Int = 1, niter::Int = 15, standardize::Bool = true)

Fit a sparse canonical correlation analysis (sparse CCA) between two sets of
variables, after Witten, Tibshirani & Hastie (2009)
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the first variable set, with variables
  in ROWS and observations in COLUMNS (dx×n)
- `Y::Matrix{Float64}`: 2d array of floats; the second variable set (dy×n), same
  number of observations (columns) as X
- `penaltyx::Real`: The L1 penalty on the X canonical vectors, in (0,1]; smaller
  values give sparser vectors. Defaults to 0.3
- `penaltyz::Real`: The L1 penalty on the Y canonical vectors, in (0,1]. Defaults to 0.3
- `K::Int`: The number of canonical components to extract; must be in 1:min(dx,dy).
  Defaults to 1
- `niter::Int`: The maximum power-iteration steps per component. Defaults to 15
- `standardize::Bool`: Whether to center and scale each variable to unit variance
  before fitting. Defaults to true
# Value
An `Scca` holding the sparse X and Y canonical vectors (u, v), the
component weights d, the sample correlations of the paired variates, the penalties,
and K. Sparse CCA finds pairs of sparse vectors, one in X-space and one in Y-space,
whose projected variates are maximally correlated under an L1 penalty (so each
vector selects only a few variables). It uses the diagonal-covariance penalized
formulation of Witten, Tibshirani & Hastie (2009): each component is a rank-1
penalized approximation of XᵀY solved by alternating soft-thresholded power
iteration, after which the data are deflated before the next component
"""
function scca(X::Matrix{Float64}, Y::Matrix{Float64};
	penaltyx::Real = 0.3, penaltyz::Real = 0.3,
	K::Int = 1, niter::Int = 15, standardize::Bool = true)
	dx, n  = size(X)                              # variables × observations
	dy, n2 = size(Y)
	n == n2 || throw(DimensionMismatch("X and Y must share the number of columns (observations)."))
	dx >= 2 && dy >= 2 || throw(ArgumentError("need at least two features in each of X and Y"))
	(0 < penaltyx <= 1 && 0 < penaltyz <= 1) || throw(ArgumentError("penaltyx, penaltyz must be in (0,1]"))
	1 <= K <= min(dx, dy) || throw(ArgumentError("K must be in 1:min(dx,dy)"))

	# the sparse routines work observations × features, so transpose into that layout
	x = Matrix{Float64}(transpose(X))
	z = Matrix{Float64}(transpose(Y))

	if standardize
		sdx = std(x, dims = 1; corrected = true)
		sdz = std(z, dims = 1; corrected = true)
		any(sdx .== 0) && throw(ArgumentError("a column of X has zero std; cannot standardize"))
		any(sdz .== 0) && throw(ArgumentError("a column of Y has zero std; cannot standardize"))
		x .= (x .- mean(x, dims = 1)) ./ sdx
		z .= (z .- mean(z, dims = 1)) ./ sdz
	end

	# initialize v: a cheap SVD construction when both blocks are wide, else a direct SVD
	Vinit = if dx > n && dy > n
		_fast_init_v(x, z, K)
	else
		svd(transpose(x)*z).V[:, 1:K]
	end

	U = zeros(dx, K);
	V = zeros(dy, K);
	D = zeros(K);
	C = zeros(K)

	# scratch buffers reused across components; zv/xu are oversized (n+K-1) to
	# accommodate the block growing as deflation rows are appended
	nmax = n + K - 1
	u    = Vector{Float64}(undef, dx)
	v    = Vector{Float64}(undef, dy)
	vold = Vector{Float64}(undef, dy)
	zv   = Vector{Float64}(undef, nmax)
	xu   = Vector{Float64}(undef, nmax)
	argu = Vector{Float64}(undef, dx)
	argv = Vector{Float64}(undef, dy)
	su   = Vector{Float64}(undef, dx)
	sv   = Vector{Float64}(undef, dy)

	xres = copy(x);
	zres = copy(z)                # working blocks, grown by deflation
	for k in 1:K
		d = _sparse_cca_single_opt!(u, v, xres, zres, @view(Vinit[:, k]),
			penaltyx, penaltyz, niter,
			vold, zv, xu, argu, argv, su, sv)
		@views U[:, k] .= u
		@views V[:, k] .= v
		D[k] = d
		if any(!iszero, u) && any(!iszero, v)
			C[k] = cor(x * u, z * v)              # sample correlation of the paired variates
		end
		# deflate by appending weighted rows that absorb this component, so the
		# next component is orthogonal to it (the PMA::CCA deflation scheme)
		if k < K
			xres = vcat(xres, sqrt(d) .* transpose(u))
			zres = vcat(zres, -sqrt(d) .* transpose(v))
		end
	end

	return Scca(U, V, D, C, Float64(penaltyx), Float64(penaltyz), K)
end
