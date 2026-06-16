# Abhisek Banerjee
# JIVE (r.jive replica) — OPTIMIZED versions, kept alongside the originals.
# Optimizations: (I-VV') projections rewritten as X-(X*V)*V' (no n×n matrices),
# preallocated buffers, 5-arg mul! for in-place projections, svd! in-place input.
# Robustness (LAPACK fallback) preserved via safe_svd! for degenerate matrices.
#
# Uses (already defined in jive.jl, NOT redefined here):
#   JiveResult, safe_svd, safe_svdvals

using LinearAlgebra, Statistics, Random

# in-place SVD with the same LAPACK fallback as safe_svd (mirrors r.jive svdwrapper).
# svd! overwrites its input A (saves the input copy); falls back to QRIteration on failure.
function safe_svd!(A)
    try
        return svd!(A)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        return svd(A; alg = LinearAlgebra.QRIteration())   # A may be partially overwritten; recompute robustly
    end
end


# (1) OPTIMIZED CORE — Algorithm 1, with preallocated buffers and in-place BLAS
function _jive_rjive_core_opt2(Xc::Vector{Matrix{Float64}}, n::Int, r::Int, ri::Vector{Int};
                               conv::Float64, maxiter::Int)
    T_ = Float64
    k = length(Xc)

    # SVD-reduction (once, not in loop)
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
    rowranges = (let rr = Vector{UnitRange{Int}}(undef, k); idx=1
                     for i in 1:k; rr[i]=idx:idx+pis[i]-1; idx+=pis[i]; end; rr end)
    ptot = sum(pis)

    A    = [zeros(T_, pis[i], n) for i in 1:k]
    J    = [zeros(T_, pis[i], n) for i in 1:k]
    Vind = [zeros(T_, n, ri[i]) for i in 1:k]
    Xtot = reduce(vcat, Xr)

    # preallocated buffers reused every iteration
    Jtot  = fill(-1.0, ptot, n)
    Atot  = fill(-1.0, ptot, n)
    Jlast = similar(Jtot)
    Alast = similar(Atot)
    tmpJ  = Matrix{T_}(undef, ptot, n)
    V     = Matrix{T_}(undef, n, r)
    USj   = Matrix{T_}(undef, ptot, r)
    tmpi  = [Matrix{T_}(undef, pis[i], n) for i in 1:k]
    projr = [Matrix{T_}(undef, pis[i], r) for i in 1:k]

    nrun = 0; converged = false
    while nrun < maxiter && !converged
        copyto!(Jlast, Jtot); copyto!(Alast, Atot)

        #  joint: rank-r SVD of (Xtot - Atot) 
        if r > 0
            @. tmpJ = Xtot - Atot
            s = safe_svd!(tmpJ)                          # in-place input (tmpJ rewritten next iter)
            @views mul!(USj, s.U[:,1:r], Diagonal(s.S[1:r]))
            @views mul!(Jtot, USj, s.Vt[1:r,:])
            @views copyto!(V, transpose(s.Vt[1:r,:]))
        else
            fill!(Jtot, 0.0)
        end
        for i in 1:k
            @views J[i] .= Jtot[rowranges[i], :]
        end

        #  individual: project ⟂ joint and ⟂ other individuals, then SVD 
        for i in 1:k
            if ri[i] > 0
                tmp = tmpi[i]
                @. tmp = Xr[i] - J[i]
                if r > 0
                    mul!(projr[i], tmp, V)                       # projr = tmp*V
                    mul!(tmp, projr[i], transpose(V), -1.0, 1.0) # tmp -= projr*V'
                end
                if nrun > 0
                    for j in 1:k
                        j == i && continue
                        Vj = Vind[j]
                        pj = tmp * Vj                            # pis × ri[j]
                        mul!(tmp, pj, transpose(Vj), -1.0, 1.0)  # tmp -= pj*Vj'
                    end
                end
                s = safe_svd!(tmp)                                # tmp rewritten next iter
                @views copyto!(Vind[i], transpose(s.Vt[1:ri[i], :]))
                @views mul!(A[i], s.U[:,1:ri[i]] * Diagonal(s.S[1:ri[i]]), s.Vt[1:ri[i],:])
            else
                fill!(A[i], 0)
            end
        end

        # first-iteration re-orthogonalization (keep copy: A[i] used later)
        if nrun == 0
            for i in 1:k, j in 1:k
                j == i && continue
                Vj = Vind[j]
                pj = A[i] * Vj
                mul!(A[i], pj, transpose(Vj), -1.0, 1.0)
            end
            for i in 1:k
                if ri[i] > 0
                    s = safe_svd(A[i])                            # NON-destructive: A[i] needed
                    @views copyto!(Vind[i], transpose(s.Vt[1:ri[i], :]))
                end
            end
        end

        for i in 1:k
            @views Atot[rowranges[i], :] .= A[i]
        end

        if norm(Jtot .- Jlast) <= conv && norm(Atot .- Alast) <= conv
            converged = true
        end
        nrun += 1
    end

    # map back & factorize (once)
    Jfull = [Ubig[i] * J[i] for i in 1:k]
    Afull = [Ubig[i] * A[i] for i in 1:k]
    Fj = safe_svd(reduce(vcat, Jfull))
    S = Matrix(@view Fj.Vt[1:r, :])
    pis_full = [size(Ji,1) for Ji in Jfull]
    Ufull = Fj.U[:,1:r] * Diagonal(Fj.S[1:r])
    U = Matrix{T_}[]; idx=1
    for p in pis_full; push!(U, Ufull[idx:idx+p-1,:]); idx+=p; end
    Si = Matrix{T_}[]; Wi = Matrix{T_}[]
    for i in 1:k
        Fi = safe_svd(Afull[i])
        push!(Si, Matrix(@view Fi.Vt[1:ri[i], :]))
        push!(Wi, Fi.U[:,1:ri[i]] * Diagonal(Fi.S[1:ri[i]]))
    end
    return JiveResult{T_}(Jfull, Afull, S, U, Si, Wi, r, ri)
end


# (2) OPTIMIZED PERMUTATION RANK SELECTION — Algorithm 2, with preallocated buffers and in-place BLAS

function _jive_perm_ranks_opt(Xc::Vector{Matrix{Float64}}, n::Int;
                              nperm::Int, alpha::Float64, conv::Float64,
                              maxiter::Int, maxrounds::Int = 10)
    k = length(Xc)
    Jperp = [zeros(size(Xc[i])) for i in 1:k]
    Aperp = [zeros(size(Xc[i])) for i in 1:k]
    last = fill(-2, k+1); current = fill(-1, k+1)
    rJ = 0; rA = zeros(Int, k); nrun = 0

    ptot = sum(size(X,1) for X in Xc)
    # preallocated buffers reused across permutations
    fullstack = Matrix{Float64}(undef, ptot, n)    # stacked (Xc - Aperp), permuted column-wise
    permcols  = Vector{Int}(undef, n)

    while last != current && nrun < maxrounds
        last = copy(current)

        #  joint rank: individual removed, permute columns within each block 
        full = [Xc[i] .- Aperp[i] for i in 1:k]
        actual = safe_svdvals(reduce(vcat, full))
        nsv = min(n, ptot)
        perms = zeros(nperm, nsv)
        rowr = (let rr=Vector{UnitRange{Int}}(undef,k); idx=1
                    for i in 1:k; rr[i]=idx:idx+size(full[i],1)-1; idx+=size(full[i],1); end; rr end)
        for p in 1:nperm
            for i in 1:k
                randperm!(permcols)                         # in-place random permutation of 1:n
                @views fullstack[rowr[i], :] .= full[i][:, permcols]
            end
            sv = safe_svdvals(fullstack)
            m = min(length(sv), nsv)
            @views perms[p, 1:m] .= sv[1:m]
        end
        rJ = 0
        for i in 1:nsv
            actual[i] > quantile(@view(perms[:,i]), 1-alpha) ? (rJ += 1) : break
        end
        rJ = max(rJ, last[1])

        #  individual ranks: joint removed, permute within each row 
        for i in 1:k
            ind = Xc[i] .- Jperp[i]
            pi_ = size(ind, 1)
            actual_i = safe_svdvals(ind)
            nsv_i = min(n, pi_)
            perms_i = zeros(nperm, nsv_i)
            permbuf = Matrix{Float64}(undef, pi_, n)
            for p in 1:nperm
                for row in 1:pi_
                    randperm!(permcols)
                    @views permbuf[row, :] .= ind[row, permcols]
                end
                sv = safe_svdvals(permbuf)
                m = min(length(sv), nsv_i)
                @views perms_i[p, 1:m] .= sv[1:m]
            end
            ra = 0
            for j in 1:nsv_i
                actual_i[j] > quantile(@view(perms_i[:,j]), 1-alpha) ? (ra += 1) : break
            end
            rA[i] = ra
        end

        current = vcat(rJ, rA)

        # refit at new ranks (uses the OPTIMIZED core)
        if last != current && rJ > 0
            fit = _jive_rjive_core_opt2(Xc, n, rJ, rA; conv=conv, maxiter=maxiter)
            Jperp = fit.J; Aperp = fit.A
        end
        nrun += 1
    end
    return rJ, rA
end


# (3) OPTIMIZED PUBLIC WRAPPER — user-facing function, with optional rank estimation via permutation test.
function jive_rjive_opt(Xs::Vector{<:AbstractMatrix};
                        r = nothing, ri = nothing,
                        scale = true, center = true, tol = nothing,
                        maxiter = 1000, nperm = 100, alpha = 0.05)
    k = length(Xs)
    Xs = [Matrix{Float64}(X) for X in Xs]
    n = size(Xs[1], 2)
    all(size(X,2) == n for X in Xs) || throw(ArgumentError("all datasets need the same number of columns"))

    nel = [size(X,1)*size(X,2) for X in Xs]; sum_n = sum(nel)
    Xc = Vector{Matrix{Float64}}(undef, k)
    for i in 1:k
        Xi = center ? Xs[i] .- mean(Xs[i], dims=2) : copy(Xs[i])
        scale && (Xi ./= (norm(Xi) * sqrt(sum_n)))
        Xc[i] = Xi
    end
    conv = tol === nothing ? 1e-6 * norm(reduce(vcat, Xc)) : tol

    if r === nothing || ri === nothing
        println("Estimating ranks via permutation test...")
        r, ri = _jive_perm_ranks_opt(Xc, n; nperm=nperm, alpha=alpha, conv=conv, maxiter=maxiter)
        println("Estimated joint rank: $r, individual ranks: $ri")
    end

    return _jive_rjive_core_opt2(Xc, n, r, ri; conv=conv, maxiter=maxiter)
end

# positional form
jive_rjive_opt(Xs::Vector{<:AbstractMatrix}, r::Int, ri::Vector{Int}; kwargs...) =
    jive_rjive_opt(Xs; r=r, ri=ri, kwargs...)