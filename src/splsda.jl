# Abhisek Banerjee
# splsda — faithful transcription of mixOmics' sPLS-DA (single-block path).
# Reproduces splsda(X, Y, ncomp, keepX): PLS NIPALS with L1 variable selection
# on X-loadings, regression deflation on X. Matches mixOmics bit-for-bit.
#
# To match mixOmics EXACTLY (including Y-loadings), pass `levels` = R's factor
# level order, e.g. levels = levels(factor(Y)) from R. Default sorts classes
# alphabetically (R's default for an unordered factor).

using LinearAlgebra, Statistics

struct SplsdaResult{T}
    variates_X::Matrix{T}     # X scores (n × ncomp)
    variates_Y::Matrix{T}     # Y scores (n × ncomp)
    loadings_X::Matrix{T}     # X loadings (p × ncomp), sparse
    loadings_Y::Matrix{T}     # Y loadings (k × ncomp)
    ncomp::Int
    keepX::Vector{Int}
    Y_dummy::Matrix{T}        # the one-hot indicator
    classes::Vector           # class levels (in dummy-column order)
end

# one-hot encode a class vector (mixOmics' unmap).
# `levels`: dummy-column class order. Pass R's factor levels to match mixOmics
# exactly. Defaults to sorted order (R's default for unordered factors).
function _unmap(y::AbstractVector; levels=nothing)
    classes = levels === nothing ? sort(unique(y)) : collect(levels)
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

# center + scale columns (scale = unbiased SD, n-1, matching colSds)
function _center_scale(M::AbstractMatrix; scale=true)
    Mc = M .- mean(M, dims=1)
    if scale
        s = std(M, dims=1; corrected=true)        # n-1 SD, like colSds
        s[s .== 0] .= 1.0                           # avoid div by zero (mixOmics zeros those cols)
        Mc = Mc ./ s
        zerocols = vec(std(M, dims=1; corrected=true) .== 0)
        Mc[:, zerocols] .= 0.0
    end
    return Mc
end

# soft_thresholding_L1: keep the keepX largest-|·| entries, shrink them by the
# largest eliminated magnitude; zero the rest. nx = p - keepX entries to drop.
function _soft_threshold_L1(x::AbstractVector, nx::Int)
    nx <= 0 && return copy(x)
    absx = abs.(x)
    # rank with ties="max": an entry's rank = number of entries ≤ it.
    # keep entries whose rank > nx (the keepX largest).
    p = length(x)
    ord = sortperm(absx)
    ranks = zeros(Int, p)
    i = 1
    while i <= p
        j = i
        while j < p && absx[ord[j+1]] == absx[ord[i]]
            j += 1
        end
        for m in i:j; ranks[ord[m]] = j; end        # ties get the max rank
        i = j + 1
    end
    keep = ranks .> nx                               # TRUE = keep
    all(keep) && return copy(x)
    lambda = maximum(absx[.!keep])                   # largest dropped magnitude
    out = similar(x)
    for i in 1:p
        out[i] = keep[i] ? sign(x[i]) * (absx[i] - lambda) : 0.0
    end
    return out
end

l2norm(x) = x ./ sqrt(sum(abs2, x))

function splsda(X::AbstractMatrix, y::AbstractVector, ncomp::Int, keepX::Vector{Int};
                scale=true, tol=1e-6, max_iter=100, levels=nothing)
    n, p = size(X)
    Yd, classes = _unmap(y; levels=levels)
    k = size(Yd, 2)
    length(keepX) == ncomp || throw(ArgumentError("keepX must have length ncomp"))

    Xc = _center_scale(Matrix{Float64}(X); scale=scale)
    Yc = _center_scale(Yd; scale=scale)

    TX = zeros(n, ncomp); TY = zeros(n, ncomp)
    PX = zeros(p, ncomp); PY = zeros(k, ncomp)

    R = copy(Xc)                                     # X residual (deflated each comp)
    Ry = copy(Yc)                                    # Y residual (not deflated for DA)

    for comp in 1:ncomp
        # --- init via SVD of XᵀY ---
        M = R' * Ry
        F = svd(M)
        aX = F.U[:, 1]
        aY = F.V[:, 1]

        aX_old = copy(aX); aY_old = copy(aY)
        iter = 1
        while true
            tY = Ry * aY
            # block X: outer weight, sparsity, normalize
            aX = R' * tY
            aX = _soft_threshold_L1(aX, p - keepX[comp])
            aX = l2norm(aX)
            tX = R * aX
            # block Y: outer weight, normalize (no sparsity)
            aY = Ry' * tX
            aY = l2norm(aY)

            dX = sum(abs2, aX .- aX_old)
            dY = sum(abs2, aY .- aY_old)
            (max(dX, dY) < tol || iter > max_iter) && break
            aX_old = copy(aX); aY_old = copy(aY)
            iter += 1
        end

        tX = R * aX; tY = Ry * aY
        TX[:, comp] = tX; TY[:, comp] = tY
        PX[:, comp] = aX; PY[:, comp] = aY

        # --- regression deflation of X by its own variate tX ---
        pX = (R' * tX) / (tX' * tX)
        R = R .- tX * pX'
        # Y not deflated for DA (mode="regression")
    end

    return SplsdaResult(TX, TY, PX, PY, ncomp, keepX, Yd, classes)
end