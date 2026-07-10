"""
	Pmd{T}

Container for a fitted penalized matrix decomposition, as returned by `pmd`
# Fields
- `u::Matrix{T}`: The n×K left factors (one unit-ℓ₂, L1-penalized column per component)
- `v::Matrix{T}`: The p×K right factors (one unit-ℓ₂, L1-penalized column per component)
- `d::Vector{T}`: The K nonnegative singular-value-like weights, dₖ = uₖᵀ X vₖ
- `sumabsu::Float64`: The resolved L1 budget on each u column (in [1, √n])
- `sumabsv::Float64`: The resolved L1 budget on each v column (in [1, √p])
- `K::Int`: The number of components
- `meanx::Float64`: The grand mean removed during centering, or NaN if center=false
"""
struct Pmd{T}
	u::Matrix{T}
	v::Matrix{T}
	d::Vector{T}
	sumabsu::Float64
	sumabsv::Float64
	K::Int
	meanx::Float64       # grand mean removed when centering, NaN otherwise
end

"""
	_pmd_soft(a::Float64, λ::Float64)

Soft-threshold a scalar: shrink toward zero by λ, clamping at zero
# Arguments
- `a::Float64`: The value to threshold
- `λ::Float64`: The nonnegative threshold
# Value
Float; sign(a)·max(|a| − λ, 0). The elementwise solution to the L1-penalized
update in Witten, Tibshirani & Hastie (2009)
"""
_pmd_soft(a, λ) = sign(a) * max(abs(a) - λ, 0.0)

"""
	_pmd_l2n(a)

Compute the ℓ₂ norm of a vector with a zero-guard
# Arguments
- `a`: 1d array of floats
# Value
Float; the Euclidean norm ‖a‖₂, or 0.05 when a is all zeros. The guard mirrors the R package
PMA's convention  and prevents division by zero in the normalization steps
"""
function _pmd_l2n(a)
	s = 0.0
	@inbounds @simd for i in eachindex(a)
		s += a[i] * a[i]
	end
	r = sqrt(s)
	return r == 0 ? 0.05 : r
end

"""
	_pmd_l1_norm_soft(a, λ::Float64)

Compute the ℓ₁/ℓ₂ ratio of a vector AFTER soft-thresholding it by λ, without
materializing the thresholded vector
# Arguments
- `a`: 1d array of floats; the unthresholded vector
- `λ::Float64`: The soft-threshold to apply
# Value
Float; ‖S(a,λ)‖₁ / ‖S(a,λ)‖₂ where S is the soft-threshold operator. Used inside
the binary search to test whether a candidate λ meets the L1 budget
"""
function _pmd_l1_norm_soft(a, λ)
	s2 = 0.0;
	s1 = 0.0
	@inbounds @simd for i in eachindex(a)
		si = sign(a[i]) * max(abs(a[i]) - λ, 0.0)
		s2 += si * si;
		s1 += abs(si)
	end
	nrm = sqrt(s2);
	nrm == 0 && (nrm = 0.05)
	return s1 / nrm
end

"""
	_pmd_l1_norm(a)

Compute the ℓ₁/ℓ₂ ratio of a vector
# Arguments
- `a`: 1d array of floats
# Value
Float; ‖a‖₁ / ‖a‖₂. This ratio is the quantity the L1 budget constrains; it
ranges from 1 (maximally sparse) to √length(a) (fully dense)
"""
function _pmd_l1_norm(a)
	nrm = _pmd_l2n(a);
	s1 = 0.0
	@inbounds @simd for i in eachindex(a)
		s1 += abs(a[i])
	end
	return s1 / nrm
end

"""
	_pmd_l1diff(a, b)

Compute the ℓ₁ distance between two vectors
# Arguments
- `a`: 1d array of floats
- `b`: 1d array of floats
# Value
Float; ‖a − b‖₁. Used as the convergence criterion between successive v iterates
"""
function _pmd_l1diff(a, b)
	s = 0.0
	@inbounds @simd for i in eachindex(a)
		s += abs(a[i] - b[i])
	end
	return s
end

"""
	_pmd_binary_search(argu, sumabs::Float64)

Find the soft-threshold λ that makes a soft-thresholded vector meet a target
L1 budget
# Arguments
- `argu`: 1d array of floats; the vector to be thresholded
- `sumabs::Float64`: The target ℓ₁/ℓ₂ ratio (the L1 budget)
# Value
Float; the threshold λ ≥ 0 such that ‖S(argu,λ)‖₁/‖S(argu,λ)‖₂ ≈ sumabs. Returns
0 when the vector already satisfies the budget. Implements the bisection step of
Witten, Tibshirani & Hastie (2009), bracketing λ in [0, max|argu|] and halving
for up to 150 iterations or until the bracket is below 1e-6
"""
function _pmd_binary_search(argu, sumabs)
	(_pmd_l2n(argu) == 0 || _pmd_l1_norm(argu) <= sumabs) && return 0.0
	lam1 = 0.0
	lam2 = maximum(abs, argu) - 1e-5
	iter = 1
	while iter < 150
		mid = (lam1 + lam2) / 2
		_pmd_l1_norm_soft(argu, mid) < sumabs ? (lam2 = mid) : (lam1 = mid)
		(lam2 - lam1) < 1e-6 && return (lam1 + lam2) / 2
		iter += 1
	end
	return (lam1 + lam2) / 2
end

"""
	_pmd_soft_normalize!(out, arg, λ::Float64)

Soft-threshold a vector by λ and rescale it to unit ℓ₂ norm, writing into a
preallocated buffer
# Arguments
- `out`: 1d array of floats; the destination buffer, overwritten in place
- `arg`: 1d array of floats; the vector to threshold
- `λ::Float64`: The soft-threshold
# Value
The buffer `out`, holding S(arg,λ) normalized to unit ℓ₂ norm (or all zeros,
guarded against division by zero). This is the factor update of the PMD power
iteration
"""
function _pmd_soft_normalize!(out, arg, λ)
	nrm = 0.0
	@inbounds @simd for i in eachindex(arg)
		si = sign(arg[i]) * max(abs(arg[i]) - λ, 0.0)
		out[i] = si
		nrm += si * si
	end
	nrm = sqrt(nrm);
	nrm == 0 && (nrm = 0.05)
	@inbounds @simd for i in eachindex(out)
		out[i] /= nrm
	end
	return out
end

"""
	_pmd_check_v(x::Matrix{Float64}, K::Int)

Initialize the right factors from the leading right singular vectors of the data
# Arguments
- `x::Matrix{Float64}`: 2d array of floats; the (already preprocessed) data matrix
- `K::Int`: The number of components to initialize
# Value
2d array of floats; the p×K matrix of leading right singular vectors of `x`. Two
branches keep the eigenproblem small: for wide data (p > n) it decomposes the
n×n Gram matrix x·xᵀ and back-projects, otherwise it decomposes the p×p Gram
matrix xᵀ·x directly. SVD initialization is the deterministic starting point
recommended in Witten, Tibshirani & Hastie (2009)
"""
function _pmd_check_v(x, K)
	n, p = size(x)
	if p > n
		F = svd(x * transpose(x))                  # n×n Gram; cheaper when wide
		V = transpose(x) * F.V[:, 1:K]             # back-project to feature space
		for j in 1:K
			V[:, j] ./= _pmd_l2n(@view V[:, j])    # renormalize each column
		end
		return V
	else
		F = svd(transpose(x) * x)                  # p×p Gram; cheaper when tall
		return F.V[:, 1:K]
	end
end

"""
	_pmd_smd!(x, v0, sumabsu::Float64, sumabsv::Float64, niter::Int,
			  u, v, vold, argu, argv)

Compute one rank-1 sparse factor pair by alternating, soft-thresholded power
iteration
# Arguments
- `x`: 2d array of floats; the (deflated) data matrix for this component
- `v0`: 1d array of floats; the initial right factor (from `_pmd_check_v`)
- `sumabsu::Float64`: The L1 budget on u (in [1, √n])
- `sumabsv::Float64`: The L1 budget on v (in [1, √p])
- `niter::Int`: The maximum number of power-iteration steps
- `u`: 1d array of floats; preallocated buffer for the left factor, written in place
- `v`: 1d array of floats; preallocated buffer for the right factor, written in place
- `vold`: 1d array of floats; preallocated scratch holding the previous v iterate
- `argu`: 1d array of floats; preallocated scratch of length n
- `argv`: 1d array of floats; preallocated scratch of length p
# Value
Float; the weight d = uᵀ x v for the fitted component. On return `u` and `v` hold
the unit-ℓ₂, L1-penalized factors. Each step alternates u ← normalize(S(x v, λᵤ))
and v ← normalize(S(xᵀ u, λᵥ)), with λ chosen by `_pmd_binary_search` to meet the
budgets; this is the single-factor PMD(L1,L1) update of Witten, Tibshirani &
Hastie (2009). The randn primer on `vold` only forces the first iteration to run
and does not affect the converged fixed point
"""
function _pmd_smd!(x, v0, sumabsu, sumabsv, niter, u, v, vold, argu, argv)
	copyto!(v, v0)
	@inbounds for i in eachindex(vold)
		;
		vold[i] = randn();
	end   # PMA loop primer
	fill!(u, 0.0)
	for _ in 1:niter
		if _pmd_l1diff(vold, v) > 1e-7
			copyto!(vold, v)
			mul!(argu, x, v)                          # x v
			lamu = _pmd_binary_search(argu, sumabsu)
			_pmd_soft_normalize!(u, argu, lamu)       # u ← normalize(S(x v, λᵤ))
			mul!(argv, transpose(x), u)               # xᵀ u
			lamv = _pmd_binary_search(argv, sumabsv)
			_pmd_soft_normalize!(v, argv, lamv)       # v ← normalize(S(xᵀ u, λᵥ))
		end
	end
	mul!(argu, x, v)                                  # recompute x v at the optimum
	return dot(u, argu)                               # d = uᵀ x v
end


"""
	pmd(X::Matrix{Float64}; sumabs::Real = 0.4,
		sumabsu::Union{Nothing,Real} = nothing,
		sumabsv::Union{Nothing,Real} = nothing,
		K::Int = 1, niter::Int = 20, center::Bool = true)

Fit a penalized matrix decomposition with L1 penalties on both factors, PMD(L1,L1)
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the n×p data matrix
- `sumabs::Real`: The shared sparsity budget in (0,1], scaled internally to give
  the per-column L1 budgets su = √n·sumabs on u and sv = √p·sumabs on v. Smaller
  values give sparser factors. Ignored when both `sumabsu` and `sumabsv` are
  supplied. Defaults to 0.4
- `sumabsu::Union{Nothing,Real}`: An explicit L1 budget on each u column, in
  [1, √n]; overrides `sumabs` when given together with `sumabsv`. Defaults to
  nothing
- `sumabsv::Union{Nothing,Real}`: An explicit L1 budget on each v column, in
  [1, √p]; overrides `sumabs` when given together with `sumabsu`. Defaults to
  nothing
- `K::Int`: The number of components to extract; must be in 1:min(n,p).
  Defaults to 1
- `niter::Int`: The maximum number of power-iteration steps per component.
  Defaults to 20
- `center::Bool`: Whether to subtract the grand mean of `X` before decomposing.
  Defaults to true
# Value
A `Pmd` holding the n×K left factors u, the p×K right factors v, the K
weights d, the resolved budgets su and sv, the number of components K, and the
grand mean removed (or NaN when center=false)
"""
function pmd(X::Matrix{Float64}; sumabs::Real = 0.4,
	sumabsu::Union{Nothing, Real} = nothing,
	sumabsv::Union{Nothing, Real} = nothing,
	K::Int = 1, niter::Int = 20, center::Bool = true)
	n, p = size(X)
	1 <= K <= min(n, p) || throw(ArgumentError("K must be in 1:min(n,p)"))

	# resolve the sparsity budgets: either derive both from sumabs, or take the
	# explicit per-factor budgets when supplied
	if sumabsu === nothing || sumabsv === nothing
		0 < sumabs <= 1 || throw(ArgumentError("sumabs must be in (0,1]"))
		su = sqrt(n) * sumabs                     # L1 budget on u, in [1,√n]
		sv = sqrt(p) * sumabs                     # L1 budget on v, in [1,√p]
	else
		su = Float64(sumabsu);
		sv = Float64(sumabsv)
	end
	1 <= su <= sqrt(n) || throw(ArgumentError("sumabsu must be in [1,√n]=[1,$(sqrt(n))]"))
	1 <= sv <= sqrt(p) || throw(ArgumentError("sumabsv must be in [1,√p]=[1,$(sqrt(p))]"))

	meanx = center ? mean(X) : NaN                # grand mean removed when centering
	Xc = center ? X .- meanx : copy(X)

	U = zeros(n, K);
	V = zeros(p, K);
	D = zeros(K)
	Vinit = _pmd_check_v(Xc, K)                   # SVD-based initialization for all K

	# caller-owned scratch buffers, reused across components
	u    = Vector{Float64}(undef, n)
	v    = Vector{Float64}(undef, p)
	vold = Vector{Float64}(undef, p)
	argu = Vector{Float64}(undef, n)
	argv = Vector{Float64}(undef, p)

	R = copy(Xc)                                  # residual, deflated after each component
	for k in 1:K
		d = _pmd_smd!(R, @view(Vinit[:, k]), su, sv, niter, u, v, vold, argu, argv)
		@views U[:, k] .= u
		@views V[:, k] .= v
		D[k] = d
		BLAS.ger!(-d, u, v, R)                    # rank-1 deflation: R ← R − d·u·vᵀ
	end

	return Pmd(U, V, D, su, sv, K, center ? meanx : NaN)
end
