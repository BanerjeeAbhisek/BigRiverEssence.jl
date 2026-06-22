# Abhisek Banerjee
# cca — Canonical Correlation Analysis.
# Two algorithms from Weenink (2003) "Canonical Correlation Analysis",
# Inst. of Phonetic Sciences, Univ. of Amsterdam, Proceedings 25, 81-99:
#   :svd → §2.2.2 (SVD of data matrices; numerically stable; DEFAULT)
#   :cov → §2.2.1 (covariance + Cholesky + generalized eigen)
# Column-major convention: each COLUMN of X and Y is an observation


using LinearAlgebra, Statistics

struct CcaResult{T}
    xmean::Vector{T}     # mean of X (length dx)
    ymean::Vector{T}     # mean of Y (length dy)
    xproj::Matrix{T}     # X projection matrix Px (dx × p)
    yproj::Matrix{T}     # Y projection matrix Py (dy × p)
    corrs::Vector{T}     # canonical correlations (length p)
    nobs::Int            # number of observations
end

#  accessors (mirroring the pca style) 
xprojection(M::CcaResult) = M.xproj
yprojection(M::CcaResult) = M.yproj
correlations(M::CcaResult) = M.corrs

# project observations into the common space.
# Z: dx×m (for :x) or dy×m (for :y); each column an observation.
# this function allows us to take new data (Z) and project it onto the canonical components learned from the original data (M),
# for either the X or Y side, depending on the value of c. It uses the stored means and projection matrices from M to perform this transformation, 
# enabling us to see how new observations relate to the canonical space defined by the original CCA model.
function cca_transform(M::CcaResult, Z::AbstractMatrix, c::Symbol)
    if c === :x
        return transpose(M.xproj) * (Z .- M.xmean)
    elseif c === :y
        return transpose(M.yproj) * (Z .- M.ymean)
    else
        throw(ArgumentError("component must be :x or :y"))
    end
end


# :svd method — Weenink §2.2.2 (SVD of centered data matrices)
# Zx, Zy already centered, dx×n and dy×n.
# we do not need to explicitly compute the covariance matrices; the SVD of the data matrices suffices to get the canonical projections and correlations.
function _cca_svd(Zx, Zy, xmean, ymean, p::Int)
    n = size(Zx, 2)
    Sx = svd(Zx)                              # Zx = Ux Dx Vx'   (col-major)
    Sy = svd(Zy)                              # Zy = Uy Dy Vy'
    # now take the SVD of the product of the right singular vectors (Vx' * Vy), which gives us the canonical correlations
    S = svd!(Sx.Vt * transpose(Sy.Vt))        # = svd(Vx * Vy')
    # order the canonical correlations and corresponding projections; we take the top p components based on the largest singular values from the SVD of Vx' * Vy, which correspond to the strongest canonical correlations between the two datasets.
    ord = sortperm(S.S; rev=true)
    si  = ord[1:p] # indices of the top p canonical correlations

    # recover the directions (projections): Px = Ux Dx^-1 U_inner ;  Py = Uy Dy^-1 V_inner
    Px = (Sx.U * Diagonal(1.0 ./ Sx.S)) * S.U[:, si]
    Py = (Sy.U * Diagonal(1.0 ./ Sy.S)) * S.V[:, si]

    # scale so Px'CxxPx = I, Py'CyyPy = I  (Cxx = ZxZx'/(n-1)) such that the canonical variates have unit variance
    Px .*= sqrt(n - 1)
    Py .*= sqrt(n - 1)

    corrs = S.S[si]
    return CcaResult(xmean, ymean, Px, Py, corrs, n)
end


# :cov method — Weenink §2.2.1 (covariance + Cholesky + generalized eigen)
# Cxx (dx×dx), Cyy (dy×dy), Cxy (dx×dy).
function _cca_cov(Cxx, Cyy, Cxy, xmean, ymean, p::Int)
    dx = size(Cxx, 1)
    dy = size(Cyy, 1)

    if dx <= dy
        # solve X-side: (Cxy Cyy^-1 Cyx) Px = ρ² Cxx Px ; recover Py = Cyy^-1 Cyx Px
        G  = cholesky(Symmetric(Cyy)) \ transpose(Cxy)    # G = Cyy^-1 * Cyx  (dy×dx)
        E  = eigen(Symmetric(Cxy * G), Symmetric(Cxx))    # generalized eig
        ord = sortperm(E.values; rev=true)[1:p]
        eigs = E.values[ord]
        Px = E.vectors[:, ord]
        Py = G * Px
        # normalize Py so Py'CyyPy = I
        # we loop through each column of Py, and for each column, we compute the quadratic form Py[:, j]' * Cyy * Py[:, j], which gives us the variance of the j-th canonical variate on the Y side. We then divide the entire column Py[:, j] by the square root of this variance to ensure that the resulting canonical variate has unit variance with respect to the covariance matrix Cyy. 
        for j in 1:p
            Py[:, j] ./= sqrt(dot(@view(Py[:, j]), Cyy * @view(Py[:, j])))
        end
    else
        # solve Y-side: (Cyx Cxx^-1 Cxy) Py = ρ² Cyy Py ; recover Px = Cxx^-1 Cxy Py
        H  = cholesky(Symmetric(Cxx)) \ Cxy               # H = Cxx^-1 * Cxy  (dx×dy)
        E  = eigen(Symmetric(transpose(Cxy) * H), Symmetric(Cyy))
        ord = sortperm(E.values; rev=true)[1:p]
        eigs = E.values[ord]
        Py = E.vectors[:, ord]
        Px = H * Py
        for j in 1:p
            Px[:, j] ./= sqrt(dot(@view(Px[:, j]), Cxx * @view(Px[:, j])))
        end
    end

    corrs = sqrt.(clamp.(eigs, 0.0, Inf))     # ρ = √eigenvalues (clamp tiny negatives)
    return CcaResult(xmean, ymean, Px, Py, corrs, -1)
end

# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------
"""
    cca(X, Y; method=:svd, outdim=min(dx,dy))

Canonical Correlation Analysis. Each COLUMN of `X` (dx×n) and `Y` (dy×n) is an
observation; both must share the same number of columns `n`.

- `method = :svd` (default): Weenink §2.2.2, SVD of the data (numerically stable).
- `method = :cov`: Weenink §2.2.1, covariance + Cholesky + generalized eigen.
- `outdim`: number of canonical pairs to return (default `min(dx, dy)`).

Returns a `CcaResult` with `xproj`, `yproj`, `corrs`.
"""
function cca(X::AbstractMatrix, Y::AbstractMatrix;
             method::Symbol=:svd, outdim::Int=min(size(X,1), size(Y,1)))
    dx, n  = size(X)
    dy, n2 = size(Y)
    n == n2 || throw(DimensionMismatch("X and Y must have the same number of columns (observations)."))
    1 <= outdim <= min(dx, dy) || throw(ArgumentError("outdim must be in 1:min(dx,dy)"))
    (n > dx && n > dy) || @warn "CCA is unstable when n ≤ dx or n ≤ dy (n=$n, dx=$dx, dy=$dy)."

    xmean = vec(mean(X, dims=2))
    ymean = vec(mean(Y, dims=2))
    Zx = X .- xmean                         # centered (dx×n)
    Zy = Y .- ymean

    if method === :svd
        return _cca_svd(Zx, Zy, xmean, ymean, outdim)
    elseif method === :cov
        Cxx = (Zx * transpose(Zx)) ./ (n - 1)
        Cyy = (Zy * transpose(Zy)) ./ (n - 1)
        Cxy = (Zx * transpose(Zy)) ./ (n - 1)
        return _cca_cov(Cxx, Cyy, Cxy, xmean, ymean, outdim)
    else
        throw(ArgumentError("method must be :svd or :cov"))
    end
end