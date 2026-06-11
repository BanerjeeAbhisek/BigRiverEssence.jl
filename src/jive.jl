# Abhisek Banerjee
# JIVE — Joint and Individual Variation Explained.
# Lock, Hoadley, Marron & Nobel (2013), Ann. Appl. Stat. 7(1):523-542,
# + supplement (Section 3 pseudocode, Section 4 SVD speedup).
#
# Decomposes k datasets (same n columns/samples, different rows/variables) into:
#   Xᵢ = Jᵢ (joint, shared across datasets) + Aᵢ (individual to i) + noise.

# result struct: joint & individual matrices AND their PCA-style factorization.
struct JiveResult{T}
    J::Vector{Matrix{T}}      # joint structure per dataset       (pᵢ × n)
    A::Vector{Matrix{T}}      # individual structure per dataset  (pᵢ × n)
    S::Matrix{T}              # shared joint scores               (r × n)
    U::Vector{Matrix{T}}      # joint loadings per dataset         (pᵢ × r)
    Si::Vector{Matrix{T}}     # individual scores per dataset      (rᵢ × n)
    Wi::Vector{Matrix{T}}     # individual loadings per dataset    (pᵢ × rᵢ)
    r::Int                    # joint rank
    ri::Vector{Int}           # individual ranks
end

"""
    jive(Xs, r, ri; standardize = true, tol = 1e-10, maxiter = 1000)

JIVE decomposition of datasets `Xs = [X₁, X₂, …]`, each pᵢ × n with the SAME n
columns (samples). Ranks are REQUIRED inputs:
- `r`  : joint rank (number of shared components).
- `ri` : vector of individual ranks, one per dataset.
- `standardize` : row-center each dataset and scale by its Frobenius norm (paper §2.1).

Returns a `JiveResult` with joint `J` / individual `A` matrices and their PCA-style
factorization (`S`, `U`, `Si`, `Wi`) per the model Xᵢ ≈ Uᵢ S + Wᵢ Sᵢ + Rᵢ (eq. 3.1).
"""
function jive(Xs::Vector{<:AbstractMatrix}, r::Int, ri::Vector{Int};   # Xs is a vector of matrices
              standardize = true, tol = 1e-10, maxiter = 1000)
    k = length(Xs)                                                     # number of datasets
    Xs = [Matrix{Float64}(X) for X in Xs]                              # convert to Float64 matrices
    n = size(Xs[1], 2)                                                 # number of samples (columns) should have same number for all Xs
    all(size(X, 2) == n for X in Xs) || throw(ArgumentError("All datasets must share the same number of columns (samples)"))
    length(ri) == k || throw(ArgumentError("ri must have one rank per dataset (length $k), got length $(length(ri))"))
    T_ = Float64

    #  preprocessing (paper §2.1): row-center, then scale each block by its Frobenius norm 
    Xc = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        Xi = Xs[i] .- mean(Xs[i], dims = 2)        # row-center (samples are columns → dims=2)
        standardize && (Xi ./= norm(Xi))           # Frobenius norm → ‖Xᵢ‖ = 1
        Xc[i] = Xi
    end

    pis = [size(X, 1) for X in Xc]                 # row counts per dataset

    stack(blocks) = reduce(vcat, blocks)           # function to takes a list of matrices and stacks them on top of each other into one tall matrix.
    function rowblocks(M)                          # function to split a tall matrix into blocks corresponding to each dataset
        idx = 1; out = Matrix{T_}[]
        for pᵢ in pis
            push!(out, M[idx:idx+pᵢ-1, :]); idx += pᵢ
        end
        out
    end

    #  initialize: joint working matrix = full stacked data (supplement §3) 
    Xjoint = stack(Xc)
    A = [zeros(T_, pis[i], n) for i in 1:k]
    J = [zeros(T_, pis[i], n) for i in 1:k]

    #  STAGE 1: alternating estimation loop (supplement §3) 
    prev_norm = Inf
    for _ in 1:maxiter
        # STEP 1: joint = rank-r SVD of Xjoint
        F = svd(Xjoint)
        Jfull = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        J = rowblocks(Jfull)
        V = F.Vt[1:r, :]'                          # joint row-space basis for projection

        # STEP 2: individual structure, per dataset
        for i in 1:k
            Xindiv = Xc[i] .- J[i]
            proj = Xindiv .- (Xindiv * V) * V'     # project away from joint space (J ⊥ Aᵢ)
            Fi = svd(proj)
            rri = ri[i]
            A[i] = Fi.U[:, 1:rri] * Diagonal(Fi.S[1:rri]) * Fi.Vt[1:rri, :]
        end

        # rebuild & check convergence
        Xjoint = stack([Xc[i] .- A[i] for i in 1:k])
        R = stack([Xc[i] .- J[i] .- A[i] for i in 1:k])
        cur = norm(R)
        abs(prev_norm - cur) < tol && break
        prev_norm = cur
    end

    #  STAGE 2: factorize into scores & loadings (paper §3.1 pca part) 
    # joint:  J_full = U S   (one shared S across datasets)
    Fj = svd(stack(J))
    S  = Fj.Vt[1:r, :]                             # shared joint scores (r × n)
    Ufull = Fj.U[:, 1:r] * Diagonal(Fj.S[1:r])     # joint loadings, stacked
    U = rowblocks(Ufull)                           # split into per-dataset blocks

    # individual:  Aᵢ = Wᵢ Sᵢ
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = svd(A[i]); rri = ri[i]
        push!(Si, Fi.Vt[1:rri, :])                          # individual scores
        push!(Wi, Fi.U[:, 1:rri] * Diagonal(Fi.S[1:rri]))   # individual loadings
    end

    return JiveResult{T_}(J, A, S, U, Si, Wi, r, ri)
end


# jive_fast — JIVE with the SVD-reduction speedup (supplement §4).
# Identical results to jive(), much faster for WIDE data (pᵢ > n): each dataset
# is compressed to rank(Xᵢ)×n via SVD before decomposing, then mapped back.

"""
    jive_fast(Xs, r, ri; standardize = true, tol = 1e-10, maxiter = 1000)

Same JIVE decomposition as `jive`, but uses the supplement §4 SVD-reduction:
each dataset is reduced to an n×rank(Xᵢ) representation before the alternating
loop, then mapped back. Gives identical results, far faster when pᵢ ≫ n.
"""
function jive_fast(Xs::Vector{<:AbstractMatrix}, r::Int, ri::Vector{Int};
                   standardize = true, tol = 1e-10, maxiter = 1000)
    k = length(Xs)
    Xs = [Matrix{Float64}(X) for X in Xs]
    n = size(Xs[1], 2)
    all(size(X, 2) == n for X in Xs) || throw(ArgumentError("All datasets must share the same number of columns (samples)"))
    length(ri) == k || throw(ArgumentError("ri must have one rank per dataset (length $k), got length $(length(ri))"))
    T_ = Float64

    #  preprocessing (same as jive): row-center, optional Frobenius scale 
    Xc = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        Xi = Xs[i] .- mean(Xs[i], dims = 2)
        standardize && (Xi ./= norm(Xi))
        Xc[i] = Xi
    end

    #  §4 SVD-REDUCTION: compress each Xᵢ to Xᵢ⊥ = Λᵢ Vᵢᵀ, remember Uᵢ to map back 
    Ubig    = Vector{Matrix{T_}}(undef, k)         # left singular vectors (pᵢ × rᵢ_full), for mapping back
    Xred    = Vector{Matrix{T_}}(undef, k)         # reduced data Xᵢ⊥ (rank_i × n)
    for i in 1:k
        Fi = svd(Xc[i])
        tolσ = maximum(Fi.S) * max(size(Xc[i])...) * eps(T_)   # numerical rank threshold
        rank_i = count(>(tolσ), Fi.S)              # numerical rank of Xᵢ (≤ n)
        Ubig[i] = Fi.U[:, 1:rank_i]                # pᵢ × rank_i
        Xred[i] = Diagonal(Fi.S[1:rank_i]) * Fi.Vt[1:rank_i, :]   # Λᵢ Vᵢᵀ  (rank_i × n)
    end

    pis = [size(X, 1) for X in Xred]               # NOW these are the REDUCED row counts (rank_i)
    stack(blocks) = reduce(vcat, blocks)
    function rowblocks(M)
        idx = 1; out = Matrix{T_}[]
        for pᵢ in pis
            push!(out, M[idx:idx+pᵢ-1, :]); idx += pᵢ
        end
        out
    end

    # --- STAGE 1: alternating loop, but on the SMALL reduced matrices ---
    Xjoint = stack(Xred)
    A⊥ = [zeros(T_, pis[i], n) for i in 1:k]        # reduced individual structures
    J⊥ = [zeros(T_, pis[i], n) for i in 1:k]
    prev_norm = Inf
    for _ in 1:maxiter
        F = svd(Xjoint)
        Jfull = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        J⊥ = rowblocks(Jfull)
        V = F.Vt[1:r, :]'
        for i in 1:k
            Xindiv = Xred[i] .- J⊥[i]
            proj = Xindiv .- (Xindiv * V) * V'
            Fi = svd(proj)
            rri = ri[i]
            A⊥[i] = Fi.U[:, 1:rri] * Diagonal(Fi.S[1:rri]) * Fi.Vt[1:rri, :]
        end
        Xjoint = stack([Xred[i] .- A⊥[i] for i in 1:k])
        R = stack([Xred[i] .- J⊥[i] .- A⊥[i] for i in 1:k])
        cur = norm(R)
        abs(prev_norm - cur) < tol && break
        prev_norm = cur
    end

    # --- MAP BACK to full variable space: Jᵢ = Uᵢ Jᵢ⊥, Aᵢ = Uᵢ Aᵢ⊥ (supplement §4) ---
    J = [Ubig[i] * J⊥[i] for i in 1:k]
    A = [Ubig[i] * A⊥[i] for i in 1:k]

    # --- STAGE 2: factorize (same as jive, now on full-size J and A) ---
    Fj = svd(stack(J))
    S  = Fj.Vt[1:r, :]
    Ufull = Fj.U[:, 1:r] * Diagonal(Fj.S[1:r])
    pis_full = [size(Ji, 1) for Ji in J]
    U = Matrix{T_}[]; idx = 1
    for pᵢ in pis_full
        push!(U, Ufull[idx:idx+pᵢ-1, :]); idx += pᵢ
    end
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = svd(A[i]); rri = ri[i]
        push!(Si, Fi.Vt[1:rri, :])
        push!(Wi, Fi.U[:, 1:rri] * Diagonal(Fi.S[1:rri]))
    end

    return JiveResult{T_}(J, A, S, U, Si, Wi, r, ri)
end