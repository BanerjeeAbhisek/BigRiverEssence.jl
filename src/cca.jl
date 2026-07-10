"""
	CcaStructure{T}

Container for a fitted canonical correlation analysis, as returned by `cca`
# Fields
- `xmean::Vector{T}`: The mean of each X variable (length dx), removed during centering
- `ymean::Vector{T}`: The mean of each Y variable (length dy), removed during centering
- `xproj::Matrix{T}`: The dx×p X canonical directions (columns), one per component
- `yproj::Matrix{T}`: The dy×p Y canonical directions (columns), one per component
- `corrs::Vector{T}`: The p canonical correlations, in [0,1] and descending
- `nobs::Int`: The number of observations used (or −1 when fit via the covariance
  path, which doesn't carry n through)
"""
struct CcaStructure{T}
	xmean::Vector{T}
	ymean::Vector{T}
	xproj::Matrix{T}
	yproj::Matrix{T}
	corrs::Vector{T}
	nobs::Int
end

"""
	cca_transform(M::CcaStructure, Z::AbstractMatrix, c::Symbol)

Project data onto the canonical directions of a fitted CCA model
# Arguments
- `M::CcaStructure`: A fitted CCA model, as returned by `cca`
- `Z::AbstractMatrix`: 2d array of floats; the data to project, with variables in
  rows and observations in columns, matching the side selected by `c`
- `c::Symbol`: Which side to project, `:x` (using the X directions and X mean) or
  `:y` (using the Y directions and Y mean)
# Value
2d array of floats; the p×n matrix of canonical variates (each observation's
coordinates along the canonical directions). Throws an `ArgumentError` if `c` is
neither `:x` nor `:y`
"""
function cca_transform(M::CcaStructure, Z::AbstractMatrix, c::Symbol)
	if c === :x
		return transpose(M.xproj) * (Z .- M.xmean)        # project onto X canonical directions
	elseif c === :y
		return transpose(M.yproj) * (Z .- M.ymean)        # project onto Y canonical directions
	else
		throw(ArgumentError("component must be :x or :y"))
	end
end

"""
	_cca_svd_opt(Zx, Zy, xmean, ymean, p::Int)

Solve CCA directly from the centered data via SVD (no covariance matrices formed)
# Arguments
- `Zx`: 2d array of floats; the centered X data (dx×n), OVERWRITTEN by svd!
- `Zy`: 2d array of floats; the centered Y data (dy×n), OVERWRITTEN by svd!
- `xmean`: 1d array of floats; the X means, stored in the result
- `ymean`: 1d array of floats; the Y means, stored in the result
- `p::Int`: The number of canonical components to return
# Value
A `CcaStructure` with the top-p canonical directions and correlations. Works from
the SVDs of Zx and Zy directly rather than forming covariance matrices, which is
the numerically stable route advocated by Weenink (2003): the canonical
correlations are the singular values of the overlap between the two
right-singular-vector bases, and the canonical directions are recovered by
rescaling the left singular vectors so the variates have unit variance
"""
function _cca_svd_opt(Zx, Zy, xmean, ymean, p::Int)
	n = size(Zx, 2)

	Sx = svd!(Zx)                                         # SVD of centered X (Zx destroyed)
	Sy = svd!(Zy)                                         # SVD of centered Y (Zy destroyed)

	inner = Sx.Vt * transpose(Sy.Vt)                      # overlap of the two row-space bases
	S = svd!(inner)                                       # its singular values are the canonical correlations

	ord = sortperm(S.S; rev = true)
	si  = ord[1:p]                                        # indices of the top-p correlations

	# rescale the left singular vectors so the directions give unit-variance variates
	scale = sqrt(n - 1)
	rmul!(Sx.U, Diagonal(scale ./ Sx.S))
	rmul!(Sy.U, Diagonal(scale ./ Sy.S))
	Px = Sx.U * @view S.U[:, si]                          # X canonical directions
	Py = Sy.U * @view S.V[:, si]                          # Y canonical directions

	corrs = S.S[si]
	return CcaStructure(xmean, ymean, Px, Py, corrs, n)
end

"""
	_cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, p::Int)

Solve CCA from the covariance matrices via a generalized eigenproblem
# Arguments
- `Cxx`: 2d array of floats; the dx×dx X covariance
- `Cyy`: 2d array of floats; the dy×dy Y covariance
- `Cxy`: 2d array of floats; the dx×dy cross-covariance
- `xmean`: 1d array of floats; the X means, stored in the result
- `ymean`: 1d array of floats; the Y means, stored in the result
- `p::Int`: The number of canonical components to return
# Value
A `CcaStructure` with the top-p canonical directions and correlations (its `nobs`
is −1, since this path works from covariances and doesn't carry n). Solves the
classical canonical-correlation generalized eigenproblem (Weenink 2003), reducing
through whichever side is smaller (dx ≤ dy or dx > dy) for efficiency: the squared
canonical correlations are the eigenvalues, one side's directions are the
eigenvectors, and the other side's are recovered and normalized to unit variance
by `_qnormalize!`
"""
function _cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, p::Int)
	dx = size(Cxx, 1)
	dy = size(Cyy, 1)

	if dx <= dy                                           # reduce through the smaller side (X)
		G    = cholesky(Symmetric(Cyy)) \ transpose(Cxy)    # Cyy⁻¹ Cyx
		A    = Cxy * G                                       # Cxy Cyy⁻¹ Cyx
		E    = eigen(Symmetric(A), Symmetric(Cxx))           # generalized eigenproblem vs Cxx
		ord  = sortperm(E.values; rev = true)[1:p]
		eigs = E.values[ord]
		Px   = E.vectors[:, ord]                             # X directions (eigenvectors)
		Py   = G * Px                                        # Y directions recovered from X
		_qnormalize!(Py, Cyy)                              # normalize Y dirs to unit variance
	else                                                  # reduce through the smaller side (Y)
		H    = cholesky(Symmetric(Cxx)) \ Cxy                # Cxx⁻¹ Cxy
		A    = transpose(Cxy) * H                            # Cyx Cxx⁻¹ Cxy
		E    = eigen(Symmetric(A), Symmetric(Cyy))
		ord  = sortperm(E.values; rev = true)[1:p]
		eigs = E.values[ord]
		Py   = E.vectors[:, ord]
		Px   = H * Py
		_qnormalize!(Px, Cxx)
	end

	corrs = sqrt.(clamp.(eigs, 0.0, Inf))                 # canonical corr = √eigenvalue (clamped ≥ 0)
	return CcaStructure(xmean, ymean, Px, Py, corrs, -1)
end

"""
	_qnormalize!(P, C)

Normalize the columns of a direction matrix to unit variance under a covariance
metric, in place
# Arguments
- `P`: 2d array of floats; the directions (columns), overwritten in place
- `C`: 2d array of floats; the covariance defining the metric
# Value
The matrix `P`, with each column pⱼ rescaled so that pⱼᵀ C pⱼ = 1 (unit variance
of the corresponding canonical variate). Used to normalize the recovered side's
directions in the covariance-based CCA solver
"""
function _qnormalize!(P, C)
	d, p = size(P)
	cp = Vector{eltype(P)}(undef, d)                      # scratch for C·pⱼ
	@inbounds for j in 1:p
		pj = @view P[:, j]
		mul!(cp, C, pj)
		s = sqrt(dot(pj, cp))                             # √(pⱼᵀ C pⱼ)
		pj ./= s                                          # rescale to unit variance
	end
	return P
end

"""
	cca(X::Matrix{Float64}, Y::Matrix{Float64}; method::Symbol = :svd,
		outdim::Int = min(size(X,1), size(Y,1)))

Fit a canonical correlation analysis (CCA) between two sets of variables, after
Weenink (2003)
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the first variable set, with variables
  in ROWS and observations in COLUMNS (dx×n)
- `Y::Matrix{Float64}`: 2d array of floats; the second variable set (dy×n), with
  the same number of observations (columns) as X
- `method::Symbol`: The solver, `:svd` (works directly from the centered data,
  more stable) or `:cov` (works from covariance matrices). Both give the same
  result. Defaults to `:svd`
- `outdim::Int`: The number of canonical components to return; must be in
  1:min(dx, dy). Defaults to min(dx, dy)
# Value
A `CcaStructure` holding the X and Y means, the canonical directions for each
side, and the canonical correlations (descending). CCA finds pairs of directions,
one in X-space and one in Y-space, whose projected variates are maximally
correlated; the k-th pair is the most correlated subject to being uncorrelated
with the previous k−1. The default solver follows the SVD-based approach of
Weenink (2003), which avoids forming covariance matrices for numerical stability.
Warns when n is not comfortably larger than dx and dy, where CCA becomes unstable
"""
function cca(X::Matrix{Float64}, Y::Matrix{Float64};
	method::Symbol = :svd, outdim::Int = min(size(X, 1), size(Y, 1)))
	dx, n  = size(X)                                       # variables × observations
	dy, n2 = size(Y)
	n == n2 || throw(DimensionMismatch("X and Y must have the same number of columns."))
	1 <= outdim <= min(dx, dy) || throw(ArgumentError("outdim must be in 1:min(dx,dy)"))
	(n > dx && n > dy) || @warn "CCA unstable when n ≤ dx or n ≤ dy (n=$n, dx=$dx, dy=$dy)."

	xmean = vec(mean(X, dims = 2))                           # mean per variable (across observations)
	ymean = vec(mean(Y, dims = 2))
	Zx = X .- xmean                                        # centered X
	Zy = Y .- ymean

	if method === :svd
		return _cca_svd_opt(Zx, Zy, xmean, ymean, outdim)
	elseif method === :cov
		# form the covariance and cross-covariance matrices, then solve the eigenproblem
		Cxx = rmul!(Zx * transpose(Zx), 1.0 / (n - 1))
		Cyy = rmul!(Zy * transpose(Zy), 1.0 / (n - 1))
		Cxy = rmul!(Zx * transpose(Zy), 1.0 / (n - 1))
		return _cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, outdim)
	else
		throw(ArgumentError("method must be :svd or :cov"))
	end
end
