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

























# Abhisek Banerjee
# jive_similar — JIVE matching r.jive's CURRENT algorithm (O'Connell & Lock).
# Same as jive(), but ALSO enforces orthogonality between the individual estimates
# (not just joint ⊥ individual). The vignette notes r.jive added this constraint:
# "also enforcing the individual estimates to be orthogonal to each other improves
#  convergence and robustness." This shifts variance from individual toward joint.

function jive_similar(Xs::Vector{<:AbstractMatrix}, r::Int, ri::Vector{Int};
                      standardize = true, tol = 1e-10, maxiter = 1000)
    k = length(Xs)
    Xs = [Matrix{Float64}(X) for X in Xs]
    n = size(Xs[1], 2)
    all(size(X, 2) == n for X in Xs) || throw(ArgumentError("All datasets must share the same number of columns (samples)"))
    length(ri) == k || throw(ArgumentError("ri must have one rank per dataset (length $k), got length $(length(ri))"))
    T_ = Float64

    # --- preprocessing (paper §2.1): row-center, scale each block by Frobenius norm ---
    Xc = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        Xi = Xs[i] .- mean(Xs[i], dims = 2)
        standardize && (Xi ./= norm(Xi))
        Xc[i] = Xi
    end

    pis = [size(X, 1) for X in Xc]
    stack(blocks) = reduce(vcat, blocks)
    function rowblocks(M)
        idx = 1; out = Matrix{T_}[]
        for pᵢ in pis
            push!(out, M[idx:idx+pᵢ-1, :]); idx += pᵢ
        end
        out
    end

    # --- initialize ---
    Xjoint = stack(Xc)
    A = [zeros(T_, pis[i], n) for i in 1:k]
    J = [zeros(T_, pis[i], n) for i in 1:k]

    # --- STAGE 1: alternating loop ---
    prev_norm = Inf
    for _ in 1:maxiter
        # STEP 1: joint = rank-r SVD of Xjoint
        F = svd(Xjoint)
        Jfull = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        J = rowblocks(Jfull)
        V = F.Vt[1:r, :]'                          # joint row-space basis (n × r)

        # STEP 2: individual structure — now orthogonal to joint AND other individuals
        for i in 1:k
            # collect bases to orthogonalize against: joint V + every OTHER individual's row space
            bases = Matrix{T_}[V]
            for j in 1:k
                if j != i && norm(A[j]) > 0        # skip zero (not-yet-estimated) individuals
                    Vj = svd(A[j]).Vt[1:ri[j], :]' # j-th individual's row space (n × rⱼ)
                    push!(bases, Vj)
                end
            end
            B  = reduce(hcat, bases)               # combined basis (n × m)
            Qb = Matrix(qr(B).Q)[:, 1:size(B, 2)]  # orthonormalize the combined basis

            Xindiv = Xc[i] .- J[i]
            proj   = Xindiv .- (Xindiv * Qb) * Qb' # remove joint AND other individuals
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

    # --- STAGE 2: factorize (same as jive) ---
    Fj = svd(stack(J))
    S  = Fj.Vt[1:r, :]
    Ufull = Fj.U[:, 1:r] * Diagonal(Fj.S[1:r])
    U = rowblocks(Ufull)

    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = svd(A[i]); rri = ri[i]
        push!(Si, Fi.Vt[1:rri, :])
        push!(Wi, Fi.U[:, 1:rri] * Diagonal(Fi.S[1:rri]))
    end

    return JiveResult{T_}(J, A, S, U, Si, Wi, r, ri)
end




# Abhisek Banerjee
# jive_rjive — replicates r.jive's jive(method="given", scale=TRUE, est=TRUE, orthIndiv=TRUE).
# Matches r.jive's source exactly: scaling = norm(Xi,'fro')*sqrt(sum of all element counts),
# SVD-reduction before the loop with map-back, and individual⊥individual orthogonality.


#=
function jive_rjive(Xs::Vector{<:AbstractMatrix}, r::Int, ri::Vector{Int};
                    scale = true, center = true, tol = nothing, maxiter = 1000)
    k = length(Xs)
    Xs = [Matrix{Float64}(X) for X in Xs]
    n = size(Xs[1], 2)
    all(size(X,2) == n for X in Xs) || throw(ArgumentError("all datasets need the same number of columns"))
    T_ = Float64

    # element counts per block, and their sum (r.jive's `n` and `sum(n)`)
    nel = [size(X,1)*size(X,2) for X in Xs]
    sum_n = sum(nel)

    # --- preprocessing: center by row mean, scale by norm*sqrt(sum_n)  (r.jive's exact formula) ---
    Xc = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        Xi = center ? Xs[i] .- mean(Xs[i], dims=2) : copy(Xs[i])
        if scale
            Xi ./= (norm(Xi) * sqrt(sum_n))        # scaleValues[i] = norm(Xi,'f')*sqrt(sum(n))
        end
        Xc[i] = Xi
    end

    # r.jive's default convergence tol: 1e-6 * ‖stacked data‖_F
    conv = tol === nothing ? 1e-6 * norm(reduce(vcat, Xc)) : tol

    # --- SVD-reduction (est=TRUE): reduce each block to ΛVᵀ, keep U to map back ---
    Ubig = Vector{Matrix{T_}}(undef, k)
    Xr   = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        if size(Xc[i],1) > size(Xc[i],2)           # only reduce when rows > cols (r.jive's condition)
            F = svd(Xc[i])
            nc = size(Xc[i], 2)
            Xr[i] = Diagonal(F.S[1:nc]) * F.Vt[1:nc, :]   # nc × n
            Ubig[i] = F.U[:, 1:nc]
        else
            Xr[i] = Xc[i]
            Ubig[i] = Matrix{T_}(I, size(Xc[i],1), size(Xc[i],1))
        end
    end

    pis = [size(X,1) for X in Xr]
    stack(b) = reduce(vcat, b)
    function rowblocks(M)
        idx=1; out=Matrix{T_}[]
        for p in pis; push!(out, M[idx:idx+p-1, :]); idx+=p; end
        out
    end

    # --- jive.iter on the reduced data ---
    A = [zeros(T_, pis[i], n) for i in 1:k]
    J = [zeros(T_, pis[i], n) for i in 1:k]
    Vind = [zeros(T_, n, ri[i]) for i in 1:k]
    Xtot = stack(Xr)
    Jtot = fill(-1.0, size(Xtot)); Atot = fill(-1.0, size(Xtot))

    nrun = 0; converged = false
    while nrun < maxiter && !converged
        Jlast = copy(Jtot); Alast = copy(Atot)

        # --- joint: rank-r SVD of (Xtot - Atot) ---
        if r > 0
            tmp = Xtot .- Atot
            s = svd(tmp)
            Jtot = s.U[:,1:r] * Diagonal(s.S[1:r]) * s.Vt[1:r,:]
            V = s.Vt[1:r,:]'                       # n × r
        else
            Jtot = zeros(T_, size(Xtot)); V = zeros(T_, n, 0)
        end
        J = rowblocks(Jtot)

        # --- individual: project away from joint AND other individuals ---
        for i in 1:k
            if ri[i] > 0
                tmp = (Xr[i] .- J[i]) * (I - V*V')             # remove joint
                if nrun > 0                                    # remove other individuals (orthIndiv)
                    for j in 1:k
                        j == i && continue
                        tmp = tmp * (I - Vind[j]*Vind[j]')
                    end
                end
                s = svd(tmp)
                Vind[i] = s.Vt[1:ri[i], :]'
                A[i] = s.U[:,1:ri[i]] * Diagonal(s.S[1:ri[i]]) * s.Vt[1:ri[i],:]
            else
                A[i] = zeros(T_, pis[i], n)
            end
        end

        # first-iteration special handling (r.jive's nrun==0 block): re-orthogonalize
        if nrun == 0
            for i in 1:k, j in 1:k
                j == i && continue
                A[i] = A[i] * (I - Vind[j]*Vind[j]')
            end
            for i in 1:k
                if ri[i] > 0
                    s = svd(A[i]); Vind[i] = s.Vt[1:ri[i], :]'
                end
            end
        end

        Atot = stack(A)
        if norm(Jtot .- Jlast) <= conv && norm(Atot .- Alast) <= conv
            converged = true
        end
        nrun += 1
    end

    # --- map reduced J, A back to full variable space:  Jᵢ = Uᵢ Jᵢ,  Aᵢ = Uᵢ Aᵢ ---
    Jfull = [Ubig[i] * J[i] for i in 1:k]
    Afull = [Ubig[i] * A[i] for i in 1:k]

    # --- factorize (paper §3.1) ---
    Fj = svd(reduce(vcat, Jfull))
    S = Fj.Vt[1:r, :]
    pis_full = [size(Ji,1) for Ji in Jfull]
    Ufull = Fj.U[:,1:r] * Diagonal(Fj.S[1:r])
    U = Matrix{T_}[]; idx=1
    for p in pis_full; push!(U, Ufull[idx:idx+p-1,:]); idx+=p; end
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = svd(Afull[i])
        push!(Si, Fi.Vt[1:ri[i], :])
        push!(Wi, Fi.U[:,1:ri[i]] * Diagonal(Fi.S[1:ri[i]]))
    end

    return JiveResult{T_}(Jfull, Afull, S, U, Si, Wi, r, ri)
end
=#



# ============================================================================
# jive_rjive — replicates r.jive exactly.
#
# THREE functions below:
#   1. _jive_rjive_core   — internal. The actual JIVE computation (your validated
#                            algorithm). Takes already-preprocessed data + ranks.
#   2. _jive_perm_ranks   — internal. Estimates ranks via permutation test
#                            (r.jive's method="perm") when you don't supply them.
#   3. jive_rjive         — THE FUNCTION YOU CALL. Preprocesses, then either uses
#                            your given ranks or estimates them, then runs the core.
#
# USAGE:
#   jive_rjive(Xs, r, ri)   → given ranks  (identical to the validated version)
#   jive_rjive(Xs)          → auto ranks   (permutation, like r.jive's default)
# ============================================================================

# --- robust SVD (mirrors r.jive's svdwrapper: fall back if fast routine fails) ---
function safe_svd(A)
    try
        return svd(A)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        return svd(A; alg = LinearAlgebra.QRIteration())
    end
end

function safe_svdvals(A)
    try
        return svdvals(A)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        return svdvals(A; alg = LinearAlgebra.QRIteration())
    end
end

# ============================================================================
# (1) INTERNAL CORE — the JIVE computation. Not called directly by you.
#     Takes preprocessed data Xc (already centered + scaled) and ranks.
# ============================================================================
function _jive_rjive_core(Xc::Vector{Matrix{Float64}}, n::Int, r::Int, ri::Vector{Int};
                          conv::Float64, maxiter::Int)
    T_ = Float64
    k = length(Xc)

    # SVD-reduction (est=TRUE): reduce each block to ΛVᵀ, keep U to map back
    Ubig = Vector{Matrix{T_}}(undef, k)
    Xr   = Vector{Matrix{T_}}(undef, k)
    for i in 1:k
        if size(Xc[i],1) > size(Xc[i],2)
            F = safe_svd(Xc[i]); nc = size(Xc[i], 2)
            Xr[i] = Diagonal(F.S[1:nc]) * F.Vt[1:nc, :]
            Ubig[i] = F.U[:, 1:nc]
        else
            Xr[i] = Xc[i]
            Ubig[i] = Matrix{T_}(I, size(Xc[i],1), size(Xc[i],1))
        end
    end

    pis = [size(X,1) for X in Xr]
    stack(b) = reduce(vcat, b)
    function rowblocks(M)
        idx=1; out=Matrix{T_}[]
        for p in pis; push!(out, M[idx:idx+p-1, :]); idx+=p; end
        out
    end

    # alternating estimation loop (jive.iter)
    A = [zeros(T_, pis[i], n) for i in 1:k]
    J = [zeros(T_, pis[i], n) for i in 1:k]
    Vind = [zeros(T_, n, ri[i]) for i in 1:k]
    Xtot = stack(Xr)
    Jtot = fill(-1.0, size(Xtot)); Atot = fill(-1.0, size(Xtot))

    nrun = 0; converged = false
    while nrun < maxiter && !converged
        Jlast = copy(Jtot); Alast = copy(Atot)

        # joint: rank-r SVD of (Xtot - Atot)
        if r > 0
            tmp = Xtot .- Atot
            s = safe_svd(tmp)
            Jtot = s.U[:,1:r] * Diagonal(s.S[1:r]) * s.Vt[1:r,:]
            V = s.Vt[1:r,:]'
        else
            Jtot = zeros(T_, size(Xtot)); V = zeros(T_, n, 0)
        end
        J = rowblocks(Jtot)

        # individual: project away from joint AND other individuals
        for i in 1:k
            if ri[i] > 0
                tmp = (Xr[i] .- J[i]) * (I - V*V')
                if nrun > 0
                    for j in 1:k
                        j == i && continue
                        tmp = tmp * (I - Vind[j]*Vind[j]')
                    end
                end
                s = safe_svd(tmp)
                Vind[i] = s.Vt[1:ri[i], :]'
                A[i] = s.U[:,1:ri[i]] * Diagonal(s.S[1:ri[i]]) * s.Vt[1:ri[i],:]
            else
                A[i] = zeros(T_, pis[i], n)
            end
        end

        # first-iteration re-orthogonalization (r.jive's nrun==0 block)
        if nrun == 0
            for i in 1:k, j in 1:k
                j == i && continue
                A[i] = A[i] * (I - Vind[j]*Vind[j]')
            end
            for i in 1:k
                if ri[i] > 0
                    s = safe_svd(A[i]); Vind[i] = s.Vt[1:ri[i], :]'
                end
            end
        end

        Atot = stack(A)
        if norm(Jtot .- Jlast) <= conv && norm(Atot .- Alast) <= conv
            converged = true
        end
        nrun += 1
    end

    # map reduced J, A back to full variable space
    Jfull = [Ubig[i] * J[i] for i in 1:k]
    Afull = [Ubig[i] * A[i] for i in 1:k]

    # factorize (paper §3.1)
    Fj = safe_svd(reduce(vcat, Jfull))
    S = Fj.Vt[1:r, :]
    pis_full = [size(Ji,1) for Ji in Jfull]
    Ufull = Fj.U[:,1:r] * Diagonal(Fj.S[1:r])
    U = Matrix{T_}[]; idx=1
    for p in pis_full; push!(U, Ufull[idx:idx+p-1,:]); idx+=p; end
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = safe_svd(Afull[i])
        push!(Si, Fi.Vt[1:ri[i], :])
        push!(Wi, Fi.U[:,1:ri[i]] * Diagonal(Fi.S[1:ri[i]]))
    end
    return JiveResult{T_}(Jfull, Afull, S, U, Si, Wi, r, ri)
end

# ============================================================================
# (2) INTERNAL PERMUTATION RANK SELECTION (r.jive's jive.perm).
#     Estimates (r, ri) from preprocessed data. Not called directly by you.
# ============================================================================
function _jive_perm_ranks(Xc::Vector{Matrix{Float64}}, n::Int;
                          nperm::Int, alpha::Float64, conv::Float64,
                          maxiter::Int, maxrounds::Int = 10)
    k = length(Xc)
    Jperp = [zeros(size(Xc[i])) for i in 1:k]
    Aperp = [zeros(size(Xc[i])) for i in 1:k]
    last = fill(-2, k+1); current = fill(-1, k+1)
    rJ = 0; rA = zeros(Int, k); nrun = 0

    while last != current && nrun < maxrounds
        last = copy(current)

        # joint rank: individual removed, permute columns within each block
        full = [Xc[i] .- Aperp[i] for i in 1:k]
        actual = safe_svdvals(reduce(vcat, full))
        nsv = min(n, sum(size(X,1) for X in Xc))
        perms = zeros(nperm, nsv)
        for p in 1:nperm
            permuted = [full[i][:, randperm(n)] for i in 1:k]
            sv = safe_svdvals(reduce(vcat, permuted))
            perms[p, 1:min(length(sv),nsv)] = sv[1:min(length(sv),nsv)]
        end
        rJ = 0
        for i in 1:nsv
            actual[i] > quantile(perms[:,i], 1-alpha) ? (rJ += 1) : break
        end
        rJ = max(rJ, last[1])

        # individual ranks: joint removed, permute within each row
        for i in 1:k
            ind = Xc[i] .- Jperp[i]
            actual_i = safe_svdvals(ind)
            nsv_i = min(n, size(ind,1))
            perms_i = zeros(nperm, nsv_i)
            for p in 1:nperm
                permuted = similar(ind)
                for row in 1:size(ind,1)
                    permuted[row, :] = ind[row, randperm(n)]
                end
                sv = safe_svdvals(permuted)
                perms_i[p, 1:min(length(sv),nsv_i)] = sv[1:min(length(sv),nsv_i)]
            end
            ra = 0
            for j in 1:nsv_i
                actual_i[j] > quantile(perms_i[:,j], 1-alpha) ? (ra += 1) : break
            end
            rA[i] = ra
        end

        current = vcat(rJ, rA)

        # refit at new ranks to update Jperp/Aperp, if ranks changed
        if last != current && rJ > 0
            fit = _jive_rjive_core(Xc, n, rJ, rA; conv=conv, maxiter=maxiter)
            Jperp = fit.J; Aperp = fit.A
        end
        nrun += 1
    end
    return rJ, rA
end

# ============================================================================
# (3) PUBLIC jive_rjive — THE FUNCTION YOU CALL.
#     Does preprocessing, picks ranks (given or estimated), runs the core.
# ============================================================================
function jive_rjive(Xs::Vector{<:AbstractMatrix};
                    r = nothing, ri = nothing,
                    scale = true, center = true, tol = nothing,
                    maxiter = 1000, nperm = 100, alpha = 0.05)
    k = length(Xs)
    Xs = [Matrix{Float64}(X) for X in Xs]
    n = size(Xs[1], 2)
    all(size(X,2) == n for X in Xs) || throw(ArgumentError("all datasets need the same number of columns"))

    # preprocessing: center by row mean + r.jive scaling (norm * sqrt(sum_n))
    nel = [size(X,1)*size(X,2) for X in Xs]; sum_n = sum(nel)
    Xc = Vector{Matrix{Float64}}(undef, k)
    for i in 1:k
        Xi = center ? Xs[i] .- mean(Xs[i], dims=2) : copy(Xs[i])
        scale && (Xi ./= (norm(Xi) * sqrt(sum_n)))
        Xc[i] = Xi
    end
    conv = tol === nothing ? 1e-6 * norm(reduce(vcat, Xc)) : tol

    # estimate ranks if not supplied (r.jive's default method="perm")
    if r === nothing || ri === nothing
        println("Estimating ranks via permutation test...")
        r, ri = _jive_perm_ranks(Xc, n; nperm=nperm, alpha=alpha, conv=conv, maxiter=maxiter)
        println("Estimated joint rank: $r, individual ranks: $ri")
    end

    return _jive_rjive_core(Xc, n, r, ri; conv=conv, maxiter=maxiter)
end

# positional form: jive_rjive(Xs, r, ri; ...) — for when ranks are known
jive_rjive(Xs::Vector{<:AbstractMatrix}, r::Int, ri::Vector{Int}; kwargs...) =
    jive_rjive(Xs; r=r, ri=ri, kwargs...)