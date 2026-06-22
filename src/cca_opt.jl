# Abhisek Banerjee
# cca — OPTIMIZED. Same algorithm/results as cca; in-place svd!/mul!,
# preallocated buffers, rmul! for in-place scaling, avoids per-column
# products in the cov normalize loop.
# Uses CcaResult, xprojection, yprojection, correlations, cca_transform from cca.jl.



# :svd method — Weenink §2.2.2
function _cca_svd_opt(Zx, Zy, xmean, ymean, p::Int)
    n = size(Zx, 2)

    Sx = svd!(Zx)                             # svd! overwrites Zx (already a fresh copy)
    Sy = svd!(Zy)

    # inner = Vx * Vy'  (col-major form of the paper's Ux'Uy);
    # its singular values ARE the canonical correlations.
    inner = Sx.Vt * transpose(Sy.Vt)
    S = svd!(inner)

    ord = sortperm(S.S; rev=true)
    si  = ord[1:p]

    # Px = Ux * Dx^-1 * U_inner[:,si], scaled by sqrt(n-1);
    # fold Dx^-1 and the scale into Ux in place.
    scale = sqrt(n - 1)
    rmul!(Sx.U, Diagonal(scale ./ Sx.S))      # Ux .= Ux * diag(scale/Dx)  (in place)
    rmul!(Sy.U, Diagonal(scale ./ Sy.S))
    Px = Sx.U * @view S.U[:, si]
    Py = Sy.U * @view S.V[:, si]

    corrs = S.S[si]
    return CcaResult(xmean, ymean, Px, Py, corrs, n)
end


# :cov method — Weenink §2.2.1
function _cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, p::Int)
    dx = size(Cxx, 1)
    dy = size(Cyy, 1)

    if dx <= dy
        # solve X-side: (Cxy Cyy^-1 Cyx) Px = ρ² Cxx Px ; recover Py = Cyy^-1 Cyx Px
        G  = cholesky(Symmetric(Cyy)) \ transpose(Cxy)    # Cyy^-1 Cyx  (dy×dx)
        A  = Cxy * G                                       # dx×dx
        E  = eigen(Symmetric(A), Symmetric(Cxx))
        ord  = sortperm(E.values; rev=true)[1:p]
        eigs = E.values[ord]
        Px = E.vectors[:, ord]                             # eigen gives Px'CxxPx = I
        Py = G * Px                                        # recover (eq.22 payoff)
        _qnormalize!(Py, Cyy)                              # enforce Py'CyyPy = I
    else
        # solve Y-side: (Cyx Cxx^-1 Cxy) Py = ρ² Cyy Py ; recover Px = Cxx^-1 Cxy Py
        H  = cholesky(Symmetric(Cxx)) \ Cxy                # Cxx^-1 Cxy  (dx×dy)
        A  = transpose(Cxy) * H                            # dy×dy
        E  = eigen(Symmetric(A), Symmetric(Cyy))
        ord  = sortperm(E.values; rev=true)[1:p]
        eigs = E.values[ord]
        Py = E.vectors[:, ord]
        Px = H * Py
        _qnormalize!(Px, Cxx)
    end

    corrs = sqrt.(clamp.(eigs, 0.0, Inf))     # ρ = √eigenvalues (clamp tiny negatives)
    return CcaResult(xmean, ymean, Px, Py, corrs, -1)
end

# in-place Σ-normalization: scale each column j of P so Pⱼ'·C·Pⱼ = 1.
# uses one preallocated buffer for C*Pⱼ instead of allocating per column.
function _qnormalize!(P, C)
    d, p = size(P)
    cp = Vector{eltype(P)}(undef, d)          # reused buffer
    @inbounds for j in 1:p
        pj = @view P[:, j]
        mul!(cp, C, pj)                       # cp = C * Pⱼ  (no per-column alloc)
        s = sqrt(dot(pj, cp))
        pj ./= s
    end
    return P
end


# public interface
"""
    cca_opt(X, Y; method=:svd, outdim=min(dx,dy))

Optimized CCA. Same results as `cca`. Each COLUMN of `X` (dx×n) and `Y` (dy×n)
is an observation; both must share the same number of columns.

- `method = :svd` (default): Weenink §2.2.2, SVD of the data (numerically stable).
- `method = :cov`: Weenink §2.2.1, covariance + Cholesky + generalized eigen.
- `outdim`: number of canonical pairs (default `min(dx, dy)`).
"""
function cca_opt(X::Matrix{Float64}, Y::Matrix{Float64};
                 method::Symbol=:svd, outdim::Int=min(size(X,1), size(Y,1)))
    dx, n  = size(X)
    dy, n2 = size(Y)
    n == n2 || throw(DimensionMismatch("X and Y must have the same number of columns."))
    1 <= outdim <= min(dx, dy) || throw(ArgumentError("outdim must be in 1:min(dx,dy)"))
    (n > dx && n > dy) || @warn "CCA unstable when n ≤ dx or n ≤ dy (n=$n, dx=$dx, dy=$dy)."

    xmean = vec(mean(X, dims=2))
    ymean = vec(mean(Y, dims=2))
    Zx = X .- xmean                           # fresh copies (svd!/cov consume them)
    Zy = Y .- ymean

    if method === :svd
        return _cca_svd_opt(Zx, Zy, xmean, ymean, outdim)
    elseif method === :cov
        # rmul! scales in place (one fewer alloc per matrix than `./ (n-1)`)
        Cxx = rmul!(Zx * transpose(Zx), 1.0 / (n - 1))
        Cyy = rmul!(Zy * transpose(Zy), 1.0 / (n - 1))
        Cxy = rmul!(Zx * transpose(Zy), 1.0 / (n - 1))
        return _cca_cov_opt(Cxx, Cyy, Cxy, xmean, ymean, outdim)
    else
        throw(ArgumentError("method must be :svd or :cov"))
    end
end