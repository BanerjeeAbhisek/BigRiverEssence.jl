# Abhisek Banerjee
# PMD — optimized version (Witten, Tibshirani & Hastie 2009, "SPC" method)
# Optimizations: preallocated buffers in hot loops, mul! for matrix-vector
# products, in-place broadcasts, BLAS.ger! for rank-1 deflation.



# ‖a - b‖ without allocating the difference vector
function norm_diff(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        d = a[i] - b[i]
        s += d * d
    end
    return sqrt(s)
end

# finding_v! — binary search for Δ giving ‖v‖₁ ≤ c, writing into preallocated
# buffers v and s (no per-bisection allocation). z is the target (X'u).
function finding_v!(v, s, z, c)
    nz = norm(z)
    @. v = z / nz                                  # v0 = z/‖z‖ into buffer v
    sum(abs, v) <= c && return v                   # already sparse enough
    lo = zero(eltype(z)); hi = maximum(abs, z)
    for _ in 1:100                                 # 100 bisections → machine precision
        delta = (lo + hi) / 2
        @. s = sign(z) * max(abs(z) - delta, zero(eltype(z)))   # soft-threshold into s
        ns = norm(s)
        if ns == 0
            @. v = s
        else
            @. v = s / ns                          # normalize into v
        end
        sum(abs, v) < c ? (hi = delta) : (lo = delta)
    end
    return v
end


# ONE SPARSE COMPONENT (deflation variant) — Algorithm 3, with preallocated buffers and in-place BLAS
function spca_component_opt(X, c; tol = 1e-6, maxiter = 500)
    n, p = size(X)
    T = eltype(X)
    v    = randn(T, p); v ./= norm(v)
    vold = similar(v)
    u    = Vector{T}(undef, n)
    Xv   = Vector{T}(undef, n)         # buffer for X*v
    Xtu  = Vector{T}(undef, p)         # buffer for X'*u
    s    = Vector{T}(undef, p)         # buffer for finding_v!'s soft-threshold

    # power-method warmup (reuse buffers)
    for _ in 1:5
        mul!(Xv, X, v)                  # Xv = X*v
        mul!(v, transpose(X), Xv)       # v = X'*(X*v)
        v ./= norm(v)
    end

    for _ in 1:maxiter
        copyto!(vold, v)
        mul!(u, X, v); u ./= norm(u)            # u = X*v / ‖·‖
        mul!(Xtu, transpose(X), u)              # Xtu = X'*u
        finding_v!(v, s, Xtu, c)                # v ← soft-threshold(X'u), into v
        norm_diff(v, vold) < tol && break
    end

    mul!(Xv, X, v)
    d = dot(u, Xv)                              # d = u'Xv  (= u'·(X*v))
    return d, u, v
end

# SPARSE PCA via deflation (Algorithm 2) — with preallocated buffers and in-place BLAS
function pmd_opt(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
                 tol = 1e-6, maxiter = 500)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c must be in [1, √p] = [1, $(sqrt(p))], you entered $c"))

    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'
    T  = eltype(Xc)

    R = copy(Xc)                                # residual; deflated each component
    V = zeros(T, p, k)
    d = zeros(T, k)
    for j in 1:k
        dj, uj, vj = spca_component_opt(R, c; tol = tol, maxiter = maxiter)
        V[:, j] = vj
        d[j]    = dj
        BLAS.ger!(-dj, uj, vj, R)               # R += (-dj)·uj·vj'  rank-1, in place
    end

    SignConsistency!(V)
    vars  = d .^ 2 ./ (n - 1)
    total = sum(abs2, Xc) / (n - 1)
    return pcaStructure{T}(T.(means), T.(sigma), V, vars, vars ./ total)
end

# ONE SPARSE COMPONENT (orthogonal-u variant) — Section 3.2, with preallocated buffers and in-place BLAS
function spca_component_orth_opt(X, c, U_prev; tol = 1e-6, maxiter = 500)
    n, p = size(X)
    T = eltype(X)
    v    = randn(T, p); v ./= norm(v)
    vold = similar(v)
    u    = Vector{T}(undef, n)
    Xv   = Vector{T}(undef, n)
    Xtu  = Vector{T}(undef, p)
    s    = Vector{T}(undef, p)
    proj = isempty(U_prev) ? nothing : Vector{T}(undef, size(U_prev, 2))  # buffer for U'u

    for _ in 1:5
        mul!(Xv, X, v)
        mul!(v, transpose(X), Xv)
        v ./= norm(v)
    end

    for _ in 1:maxiter
        copyto!(vold, v)
        mul!(u, X, v)                            # u = X*v
        if !isempty(U_prev)                      # project u ⟂ previous u's:  u -= U(U'u)
            mul!(proj, transpose(U_prev), u)     # proj = U'u
            mul!(u, U_prev, proj, -1.0, 1.0)     # u = u - U*proj   (5-arg mul!: u = -U*proj + u)
        end
        u ./= norm(u)
        mul!(Xtu, transpose(X), u)
        finding_v!(v, s, Xtu, c)
        norm_diff(v, vold) < tol && break
    end

    mul!(Xv, X, v)
    d = dot(u, Xv)
    return d, u, v
end


# SPARSE PCA with orthogonal u's (Section 3.2) — no deflation, with preallocated buffers and in-place BLAS
function pmd_orth_opt(X; k = 2, c = sqrt(size(X, 2)) / 2, standardize = false,
                      tol = 1e-6, maxiter = 500)
    n, p = size(X)
    1 <= c <= sqrt(p) || throw(ArgumentError("c must be in [1, √p] = [1, $(sqrt(p))], you entered $c"))

    means = vec(mean(X, dims = 1))
    sigma = standardize ? vec(std(X, dims = 1)) : ones(eltype(means), p)
    Xc = (X .- means') ./ sigma'
    T  = eltype(Xc)

    V = zeros(T, p, k)
    d = zeros(T, k)
    U = Matrix{T}(undef, n, k)                  # preallocate all u's; fill column by column
    for j in 1:k
        Uprev = @view U[:, 1:j-1]               # previous u's (empty view when j==1)
        dj, uj, vj = spca_component_orth_opt(Xc, c, Uprev; tol = tol, maxiter = maxiter)
        V[:, j] = vj
        d[j]    = dj
        U[:, j] = uj                            # store this u (no hcat reallocation)
    end

    SignConsistency!(V)
    vars  = d .^ 2 ./ (n - 1)
    total = sum(abs2, Xc) / (n - 1)
    return pcaStructure{T}(T.(means), T.(sigma), V, vars, vars ./ total)
end