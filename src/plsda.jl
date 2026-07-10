"""
	PlsdaStructure{T}

Container for a fitted PLS discriminant analysis, as returned by `plsda`.
# Fields
- `variates_X::Matrix{T}`: n×ncomp X-scores (sample coordinates)
- `variates_Y::Matrix{T}`: n×ncomp Y-scores
- `loadings_X::Matrix{T}`: p×ncomp dense X-loadings
- `loadings_Y::Matrix{T}`: K×ncomp Y-loadings (K = number of classes)
- `ncomp::Int`: number of components
- `Y_dummy::Matrix{T}`: n×K one-hot encoding of the labels
- `classes::Vector`: class labels, in Y_dummy column order
"""
struct PlsdaStructure{T}
	variates_X::Matrix{T}
	variates_Y::Matrix{T}
	loadings_X::Matrix{T}
	loadings_Y::Matrix{T}
	ncomp::Int
	Y_dummy::Matrix{T}
	classes::Vector
end

"""
	plsda(X::Matrix{Float64}, y::Vector, ncomp::Int;
	      scale = true, tol = 1e-6, max_iter = 100, levels = nothing)

Fit a PLS discriminant analysis (PLS-DA) for multiclass problems — the non-sparse
counterpart of `splsda`. The class labels are dummy-encoded and the problem solved
as a PLS regression of X onto the indicator matrix: each component is the rank-1
approximation of the cross-product Mₕ = XₕᵀYₕ by alternating power iteration, after
which X (only) is deflated by regression on its variate. Unlike `splsda`, no L1
penalty is applied, so every variable contributes to each loading.
"""
function plsda(X::Matrix{Float64}, y::Vector, ncomp::Int;
	scale = true, tol = 1e-6, max_iter = 100, levels = nothing)
	n, p = size(X)
	Yd, classes = _unmap(y; levels = levels)
	k = size(Yd, 2)

	Xc = _center_scale(Matrix{Float64}(X); scale = scale)
	Yc = _center_scale(Yd; scale = scale)

	TX = zeros(n, ncomp); TY = zeros(n, ncomp)
	PX = zeros(p, ncomp); PY = zeros(k, ncomp)

	R  = copy(Xc)
	Ry = copy(Yc)

	M = Matrix{Float64}(undef, p, k)
	uh = Vector{Float64}(undef, p); uh_old = Vector{Float64}(undef, p)
	vh = Vector{Float64}(undef, k); vh_old = Vector{Float64}(undef, k)
	tX = Vector{Float64}(undef, n)
	tY = Vector{Float64}(undef, n)
	uraw = Vector{Float64}(undef, p)
	pX = Vector{Float64}(undef, p)

	for comp in 1:ncomp
		mul!(M, transpose(R), Ry)
		F = svd(M)
		copyto!(uh, @view F.U[:, 1])
		copyto!(vh, @view F.V[:, 1])
		copyto!(uh_old, uh); copyto!(vh_old, vh)

		iter = 1
		while true
			mul!(tY, Ry, vh)
			mul!(uraw, transpose(R), tY)
			copyto!(uh, uraw)                 # ← no soft-threshold: keep ALL variables
			uh ./= sqrt(sum(abs2, uh))
			mul!(tX, R, uh)
			mul!(vh, transpose(Ry), tX)
			vh ./= sqrt(sum(abs2, vh))

			dX = _sqdiff(uh, uh_old)
			dY = _sqdiff(vh, vh_old)
			(max(dX, dY) < tol || iter > max_iter) && break
			copyto!(uh_old, uh); copyto!(vh_old, vh)
			iter += 1
		end

		mul!(tX, R, uh); mul!(tY, Ry, vh)
		@views TX[:, comp] .= tX; @views TY[:, comp] .= tY
		@views PX[:, comp] .= uh; @views PY[:, comp] .= vh

		mul!(pX, transpose(R), tX)
		pX ./= dot(tX, tX)
		BLAS.ger!(-1.0, tX, pX, R)
	end

	return PlsdaStructure(TX, TY, PX, PY, ncomp, Yd, classes)
end