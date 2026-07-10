"""
	SpcStructure{T}

Container for a fitted sparse principal component analysis, as returned by
`spc` or `spc_orth`
# Fields
- `mean::Vector{T}`: The column means of the training data (length p), removed
  during centering
- `scale::Vector{T}`: The column scales (length p) — the column standard
  deviations when `standardize=true`, otherwise ones
- `loadings::Matrix{T}`: The p×k sparse loadings (one unit-ℓ₂, L1-penalized
  column per component), signs fixed by `_sign_consistency_opt!`
- `variances::Vector{T}`: The variance associated with each of the k components
  (dₖ² / (n-1), where dₖ is the component weight)
- `propOFvar::Vector{T}`: The cumulative proportion of variance explained by the
  first k components (see `_prop_var_explained`)
"""
struct SpcStructure{T}
	mean::Vector{T}
	scale::Vector{T}
	loadings::Matrix{T}
	variances::Vector{T}
	propOFvar::Vector{T}
end

"""
	_l1_diff(a, b)

Compute the ℓ₁ distance between two vectors
# Arguments
- `a`: 1d array of floats
- `b`: 1d array of floats
# Value
Float; ‖a − b‖₁. Used as the convergence criterion between successive iterates
"""
function _l1_diff(a, b)
	s = zero(eltype(a))
	@inbounds @simd for i in eachindex(a)
		s += abs(a[i] - b[i])
	end
	return s
end

"""
	_finding_v!(v, s, z, c)

Project a vector onto the intersection of the unit ℓ₂ ball and the L1 budget,
writing the result in place
# Arguments
- `v`: 1d array of floats; the destination buffer for the result, overwritten in place
- `s`: 1d array of floats; preallocated scratch for the soft-thresholded vector
- `z`: 1d array of floats; the raw direction to project (e.g. Xᵀu)
- `c`: Float; the L1 budget (in [1, √p]); the returned v satisfies ‖v‖₂ = 1 and
  ‖v‖₁ ≤ c
# Value
The buffer `v`, holding the unit-ℓ₂ vector closest to z whose ℓ₁ norm meets the
budget. If the plain unit-normalized z already satisfies the budget it is
returned unchanged; otherwise a soft-threshold δ is found by bisection (up to 150
steps) so that the normalized, thresholded vector has ‖v‖₁ ≈ c. This is the
penalized update of the SPC criterion in Witten, Tibshirani & Hastie (2009)
"""
function _finding_v!(v, s, z, c)
	Tz = eltype(z)
	nz = norm(z);
	iszero(nz) && (nz = Tz(0.05))
	@. v = z / nz
	sum(abs, v) <= c && return v
	lo = zero(Tz)
	hi = maximum(abs, z) - Tz(1e-5)
	for _ in 1:150
		delta = (lo + hi) / 2
		@. s = sign(z) * max(abs(z) - delta, zero(Tz))   # soft-threshold by δ
		ns = norm(s);
		iszero(ns) && (ns = Tz(0.05))
		@. v = s / ns                                    # renormalize to unit ℓ₂
		sum(abs, v) < c ? (hi = delta) : (lo = delta)    # bisect on the L1 budget
		(hi - lo) < Tz(1e-6) && break
	end
	return v
end

"""
	_init_rsv(Xc, k::Int)

Compute the top-k right singular vectors of a centered data matrix, used to
initialize the sparse power iteration
# Arguments
- `Xc`: 2d array of floats; the centered (and optionally scaled) data matrix
- `k::Int`: The number of leading right singular vectors to return
# Value
2d array of floats; the p×k matrix of leading right singular vectors of `Xc`.
Two branches keep the eigenproblem small: for tall data (p ≤ n) it eigendecomposes
the p×p Gram matrix XcᵀXc directly, otherwise it eigendecomposes the n×n Gram
matrix XcXcᵀ and back-projects. Both use a partial eigendecomposition (only the
top k eigenpairs) and return columns in descending eigenvalue order
"""
function _init_rsv(Xc, k)
	n, p = size(Xc);
	T = eltype(Xc)
	if p <= n
		E = eigen(Symmetric(transpose(Xc) * Xc), (p-k+1):p)    # top k eigenpairs of the p×p Gram
		return E.vectors[:, k:-1:1]                          # reorder to descending
	else
		E  = eigen(Symmetric(Xc * transpose(Xc)), (n-k+1):n)   # top k of the n×n Gram
		Vk = transpose(Xc) * E.vectors[:, k:-1:1]            # back-project to feature space
		@inbounds for j in 1:k
			nrm = norm(@view Vk[:, j]);
			iszero(nrm) && (nrm = T(0.05))
			@views Vk[:, j] ./= nrm                          # renormalize each column
		end
		return Vk
	end
end

"""
	_prop_var_explained(Xc, V)

Compute the cumulative proportion of variance explained by sparse components
# Arguments
- `Xc`: 2d array of floats; the centered data matrix
- `V`: 2d array of floats; the p×K matrix of (possibly non-orthonormal) sparse
  loadings
# Value
1d array of floats of length K; entry k is the proportion of total variance
explained by the first k loadings jointly. Because sparse loadings need not be
orthonormal, the adjusted variance uses the projection ‖Xc·Vₖ(VₖᵀVₖ)⁻¹Vₖᵀ‖²_F /
‖Xc‖²_F, evaluated here in the equivalent trace form tr((VₖᵀVₖ)⁻¹ VₖᵀXcᵀXcVₖ) /
‖Xc‖²_F. This is the adjusted-variance measure of Shen & Huang / Witten et al.
for non-orthogonal sparse PCs
"""
function _prop_var_explained(Xc, V)
	K = size(V, 2);
	T = eltype(Xc)
	totsq = sum(abs2, Xc)
	A = Xc * V              # Xc·V  (n×K)
	M = transpose(V) * V    # VᵀV   (K×K)
	B = transpose(A) * A    # VᵀXcᵀXcV  (K×K)
	pve = Vector{T}(undef, K)
	@inbounds for k in 1:K
		pve[k] = tr(Symmetric(M[1:k, 1:k]) \ B[1:k, 1:k]) / totsq   # adjusted cumulative variance
	end
	return pve
end

"""
	_spca_component!(v, X, c, u, Xv, Xtu, s, vold; tol = 1e-7, niter = 20)

Compute one sparse principal component by soft-thresholded power iteration
(deflation variant)
# Arguments
- `v`: 1d array of floats; the right factor, initialized by the caller and
  overwritten in place with the converged loading
- `X`: 2d array of floats; the (deflated) data matrix for this component
- `c`: Float; the L1 budget on v (in [1, √p])
- `u`: 1d array of floats; preallocated buffer for the left factor (scores direction)
- `Xv`: 1d array of floats; preallocated scratch of length n
- `Xtu`: 1d array of floats; preallocated scratch of length p
- `s`: 1d array of floats; preallocated scratch for `_finding_v!`
- `vold`: 1d array of floats; preallocated scratch holding the previous v iterate
- `tol`: Float; the ℓ₁ convergence tolerance on v. Defaults to 1e-7
- `niter::Int`: The maximum number of power-iteration steps. Defaults to 20
# Value
Float; the component weight d = uᵀ X v. On return `v` holds the unit-ℓ₂,
L1-penalized loading and `u` the corresponding unit-ℓ₂ scores direction. Each
step sets u ← normalize(X v) then v ← project(Xᵀ u) onto the unit ball and L1
budget via `_finding_v!`, the SPC update of Witten, Tibshirani & Hastie (2009)
"""
function _spca_component!(v, X, c, u, Xv, Xtu, s, vold; tol = 1e-7, niter = 20)
	T = eltype(v)
	for _ in 1:niter
		copyto!(vold, v)
		mul!(u, X, v);
		nu = norm(u);
		iszero(nu) && (nu = T(0.05));
		u ./= nu   # u ← normalize(X v)
		mul!(Xtu, transpose(X), u)
		_finding_v!(v, s, Xtu, c)                                              # v ← project(Xᵀ u)
		_l1_diff(v, vold) < tol && break
	end
	mul!(Xv, X, v)
	return dot(u, Xv)                                                         # d = uᵀ X v
end

"""
	_spca_component_orth!(v, X, c, U_prev, u, uold, Xv, Xtu, s, vold, proj;
						 tol = 1e-6, niter = 20)

Compute one sparse principal component whose scores are orthogonal to all
previously extracted components
# Arguments
- `v`: 1d array of floats; the right factor, initialized by the caller and
  overwritten in place with the converged loading
- `X`: 2d array of floats; the (uncentered-residual) data matrix — note the
  orthogonal variant decomposes the original Xc, not a deflated residual
- `c`: Float; the L1 budget on v (in [1, √p])
- `U_prev`: 2d array of floats; the n×(j-1) matrix of previously extracted scores
  directions that the new u must be orthogonal to (empty for the first component)
- `u`: 1d array of floats; preallocated buffer for the left factor, written in place
- `uold`: 1d array of floats; preallocated scratch holding the previous u iterate
- `Xv`: 1d array of floats; preallocated scratch of length n
- `Xtu`: 1d array of floats; preallocated scratch of length p
- `s`: 1d array of floats; preallocated scratch for `_finding_v!`
- `vold`: 1d array of floats; preallocated scratch holding the previous v iterate
- `proj`: 1d array of floats; preallocated scratch for the orthogonal projection
  coefficients
- `tol`: Float; the ℓ₁ convergence tolerance on u and v. Defaults to 1e-6
- `niter::Int`: The maximum number of power-iteration steps. Defaults to 20
# Value
Float; the component weight d = uᵀ X v. Differs from `_spca_component!` by
projecting u onto the orthogonal complement of `U_prev` each iteration
(u ← u − U_prev U_prevᵀ u before normalizing), which enforces orthogonal scores
across components instead of relying on deflation — the orthogonal SPC variant of
Witten, Tibshirani & Hastie (2009)
"""
function _spca_component_orth!(v, X, c, U_prev, u, uold, Xv, Xtu, s, vold, proj;
	tol = 1e-6, niter = 20)
	T = eltype(v);
	fill!(u, zero(T));
	m = size(U_prev, 2)
	for _ in 1:niter
		copyto!(vold, v);
		copyto!(uold, u)
		mul!(u, X, v)
		if m > 0
			pj = view(proj, 1:m)
			mul!(pj, transpose(U_prev), u)
			mul!(u, U_prev, pj, -one(T), one(T))   # u -= U_prev (U_prevᵀ u): orthogonalize
		end
		nu = norm(u);
		iszero(nu) && (nu = T(0.05));
		u ./= nu
		mul!(Xtu, transpose(X), u)
		_finding_v!(v, s, Xtu, c)
		(_l1_diff(v, vold) < tol && _l1_diff(u, uold) < tol) && break
	end
	mul!(Xv, X, v)
	return dot(u, Xv)
end

"""
	spc(X::Matrix{Float64}; k::Int = 2, c::Real = sqrt(size(X,2))/2,
		standardize::Bool = false, tol::Real = 1e-7, niter::Int = 20)

Fit a sparse principal component analysis using the SPC criterion (deflation
variant)
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the n×p data matrix
- `k::Int`: The number of sparse components to extract. Defaults to 2
- `c::Real`: The L1 budget on each loading, in [1, √p]; smaller values give
  sparser loadings, c = √p gives ordinary (dense) PCA. Defaults to √p / 2
- `standardize::Bool`: Whether to scale each column to unit standard deviation
  in addition to centering. Defaults to false (center only)
- `tol::Real`: The ℓ₁ convergence tolerance per component. Defaults to 1e-7
- `niter::Int`: The maximum number of power-iteration steps per component.
  Defaults to 20
# Value
An `SpcStructure` holding the column means, scales, k sparse loadings, component
variances, and cumulative proportion of variance explained. Components are found
sequentially: each loading is fit on the residual, then removed by rank-1
deflation before the next is extracted
"""
function spc(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
	tol = 1e-7, niter = 20)
	n, p = size(X)                                     # n observations, p features
	# the L1 budget c lives between 1 (maximally sparse) and √p (fully dense)
	1 <= c <= sqrt(p) || throw(ArgumentError("c (=sumabsv) must be in [1, √p]=[1,$(sqrt(p))], got $c"))

	T = eltype(float(X))                              # work in floats even if X is integer
	means = T.(vec(mean(X, dims = 1)))                 # column means (p-vector)
	sigma = standardize ? T.(vec(std(X, dims = 1))) : ones(T, p)   # column std devs, or ones if not standardizing
	# center, and scale too when standardizing — this is the matrix SPC decomposes
	Xc = standardize ? (X .- means') ./ sigma' : T.(X .- means')

	Vinit = _init_rsv(Xc, k)                            # SVD-based starting loadings for all k components
	Rmat  = copy(Xc)                                   # residual matrix, deflated after each component

	# outputs: k loadings (V), k weights (d)
	V = zeros(T, p, k);
	d = zeros(T, k)

	# scratch buffers, allocated once and reused across all k components (avoids
	# per-component allocation; passed by reference into the rank-1 core)
	u = Vector{T}(undef, n);
	Xv = Vector{T}(undef, n)  # left factor + length-n scratch
	Xtu = Vector{T}(undef, p);
	s = Vector{T}(undef, p) # length-p scratch (Xᵀu, and _finding_v! workspace)
	vold = Vector{T}(undef, p);
	v = Vector{T}(undef, p)# current loading + its previous iterate

	for j in 1:k
		copyto!(v, view(Vinit, :, j))                  # warm-start this component at its SVD direction
		# solve for the j-th sparse rank-1 factor on the current residual;
		# returns the weight d[j], leaves the loading in v and scores dir in u
		d[j] = _spca_component!(v, Rmat, c, u, Xv, Xtu, s, vold; tol = tol, niter = niter)
		@views V[:, j] .= v                            # store the loading
		BLAS.ger!(-d[j], u, v, Rmat)                   # rank-1 deflation: Rmat ← Rmat − d·u·vᵀ
	end                                                # so the next component sees only what's left

	_sign_consistency_opt!(V)                            # fix arbitrary per-column signs for reproducibility
	vars = d .^ 2 ./ (n - 1)                           # variance carried by each component (dₖ² / (n-1))
	# PVE uses the adjusted-variance measure since sparse loadings aren't orthonormal
	return SpcStructure{T}(means, sigma, V, vars, _prop_var_explained(Xc, V))
end

"""
	spc_orth(X::Matrix{Float64}; k::Int = 2, c::Real = sqrt(size(X,2))/2,
			 standardize::Bool = false, tol::Real = 1e-6, niter::Int = 20)

Fit a sparse principal component analysis with orthogonal scores (orthogonal
variant of the SPC criterion)
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the n×p data matrix
- `k::Int`: The number of sparse components to extract. Defaults to 2
- `c::Real`: The L1 budget on each loading, in [1, √p]; smaller values give
  sparser loadings. Defaults to √p / 2
- `standardize::Bool`: Whether to scale each column to unit standard deviation
  in addition to centering. Defaults to false (center only)
- `tol::Real`: The ℓ₁ convergence tolerance per component. Defaults to 1e-6
- `niter::Int`: The maximum number of power-iteration steps per component.
  Defaults to 20
# Value
An `SpcStructure` holding the column means, scales, k sparse loadings, component
variances, and cumulative proportion of variance explained. Unlike `spc`, each
component's scores are constrained to be orthogonal to all previous components'
scores (via projection rather than deflation), so the score directions form an
orthonormal set
"""
function spc_orth(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
	tol = 1e-6, niter = 20)
	n, p = size(X)                                     # n observations, p features
	# the L1 budget c lives between 1 (maximally sparse) and √p (fully dense)
	1 <= c <= sqrt(p) || throw(ArgumentError("c (=sumabsv) must be in [1, √p]=[1,$(sqrt(p))], got $c"))

	T = eltype(float(X))                              # work in floats even if X is integer
	means = T.(vec(mean(X, dims = 1)))                 # column means (p-vector)
	sigma = standardize ? T.(vec(std(X, dims = 1))) : ones(T, p)   # column std devs, or ones if not standardizing
	# center, and scale too when standardizing — this is the matrix SPC decomposes
	Xc = standardize ? (X .- means') ./ sigma' : T.(X .- means')

	Vinit = _init_rsv(Xc, k)                            # SVD-based starting loadings for all k components

	# outputs: k loadings (V), k weights (d), AND the k score directions (U).
	# unlike spc, U is kept because each new component is orthogonalized against it.
	V = zeros(T, p, k);
	d = zeros(T, k);
	U = Matrix{T}(undef, n, k)

	# scratch buffers, allocated once and reused across all k components
	u = Vector{T}(undef, n);
	uold = Vector{T}(undef, n);
	Xv = Vector{T}(undef, n)  # left factor, its prev iterate, length-n scratch
	Xtu = Vector{T}(undef, p);
	s = Vector{T}(undef, p)                             # length-p scratch (Xᵀu, _finding_v! workspace)
	vold = Vector{T}(undef, p);
	v = Vector{T}(undef, p);
	proj = Vector{T}(undef, k)# current loading, its prev iterate, projection coeffs

	for j in 1:k
		copyto!(v, view(Vinit, :, j))                  # warm-start this component at its SVD direction
		Uprev = @view U[:, 1:(j-1)]                       # scores of all previously extracted components
		# solve for the j-th sparse factor on the ORIGINAL Xc (not a deflated residual),
		# but force its scores u orthogonal to Uprev inside the core. returns weight d[j].
		d[j] = _spca_component_orth!(v, Xc, c, Uprev, u, uold, Xv, Xtu, s, vold, proj;
			tol = tol, niter = niter)
		@views V[:, j] .= v                            # store the loading
		@views U[:, j] .= u                            # store the scores direction (needed to orthogonalize later components)
	end

	_sign_consistency_opt!(V)                            # fix arbitrary per-column signs for reproducibility
	vars = d .^ 2 ./ (n - 1)                           # variance carried by each component (dₖ² / (n-1))
	# PVE uses the adjusted-variance measure since sparse loadings aren't orthonormal
	return SpcStructure{T}(means, sigma, V, vars, _prop_var_explained(Xc, V))
end
