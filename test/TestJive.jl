# test/TestJive.jl

using BigRiverSchneider, RCall, BenchmarkTools
using LinearAlgebra, Statistics, Random
Random.seed!(1234)


# ---------------------------------------------------------------------------
# Build data with KNOWN joint + individual structure (supplement §6.2 generative model):
#   X₁ = U₁S + W₁S₁ ,  X₂ = U₂S + W₂S₂   — shared scores S, individual scores Sᵢ.
# ---------------------------------------------------------------------------
n = 100
r, r1, r2 = 2, 3, 3            # true ranks: joint=2, individual=3 each
p1, p2 = 40, 30                # variables per dataset

S  = randn(r, n)               # shared JOINT scores (same across both datasets)
U1 = randn(p1, r); U2 = randn(p2, r)     # joint loadings
S1 = randn(r1, n); W1 = randn(p1, r1)    # individual to dataset 1
S2 = randn(r2, n); W2 = randn(p2, r2)    # individual to dataset 2

X1 = U1*S + W1*S1              # dataset 1 = joint + individual (no noise)
X2 = U2*S + W2*S2              # dataset 2

println("X1 is $p1×$n,  X2 is $p2×$n,  true ranks: joint=$r, indiv=[$r1,$r2]\n")

res = jive([X1, X2], r, [r1, r2]; standardize = false)

# ---------------------------------------------------------------------------
# TEST 1 — Exact reconstruction (supplement §6.2 anchor).
# No noise + true ranks ⇒ JIVE must recover the data to machine precision.
# ---------------------------------------------------------------------------
Xc1 = X1 .- mean(X1, dims = 2)         # compare against row-centered data
Xc2 = X2 .- mean(X2, dims = 2)
recon_err = norm(Xc1 .- (res.J[1] .+ res.A[1]))^2 +
            norm(Xc2 .- (res.J[2] .+ res.A[2]))^2
println("TEST 1  exact reconstruction (noiseless, true ranks)")
println("  ‖X − (J+A)‖²        : ", round(recon_err, digits = 12), "   (want ≈ 0)\n")

# ---------------------------------------------------------------------------
# TEST 2 — Orthogonality constraint  J Aᵢᵀ = 0  (the defining JIVE property).
# Joint and individual row spaces must be orthogonal (enforced by the I−VVᵀ projection).
# ---------------------------------------------------------------------------
Jfull = vcat(res.J[1], res.J[2])
ortho1 = norm(Jfull * res.A[1]')       # J Aᵢᵀ should be ≈ 0
ortho2 = norm(Jfull * res.A[2]')
println("TEST 2  orthogonality of joint & individual  (J Aᵢᵀ = 0)")
println("  ‖J·A₁ᵀ‖             : ", round(ortho1, digits = 10))
println("  ‖J·A₂ᵀ‖             : ", round(ortho2, digits = 10), "   (want ≈ 0)\n")

# ---------------------------------------------------------------------------
# TEST 3 — Ranks of the estimated structures match the requested ranks.
# ---------------------------------------------------------------------------
rank_tol = 1e-8
joint_rank = rank(Jfull, rank_tol)
indiv_rank1 = rank(res.A[1], rank_tol)
indiv_rank2 = rank(res.A[2], rank_tol)
println("TEST 3  estimated ranks match requested")
println("  joint rank          : ", joint_rank,  "  (want $r)")
println("  individual rank 1   : ", indiv_rank1, "  (want $r1)")
println("  individual rank 2   : ", indiv_rank2, "  (want $r2)\n")

# ---------------------------------------------------------------------------
# TEST 4 — Factorization consistency: J should equal U·S, Aᵢ should equal Wᵢ·Sᵢ.
# Checks jive_factorize / the returned scores & loadings reconstruct the matrices.
# ---------------------------------------------------------------------------
fac_err_joint = norm(res.J[1] .- res.U[1]*res.S) + norm(res.J[2] .- res.U[2]*res.S)
fac_err_ind   = norm(res.A[1] .- res.Wi[1]*res.Si[1]) + norm(res.A[2] .- res.Wi[2]*res.Si[2])
println("TEST 4  factorization consistency")
println("  ‖J − U·S‖           : ", round(fac_err_joint, digits = 10), "   (want ≈ 0)")
println("  ‖A − W·S_indiv‖     : ", round(fac_err_ind,   digits = 10), "   (want ≈ 0)\n")

# ---------------------------------------------------------------------------
# TEST 5 — Recovery of the JOINT SUBSPACE (the scientifically meaningful part).
# JIVE's joint scores S should span the same space as the true shared S.
# Compare via principal angles: the subspaces should align (canonical correlations ≈ 1).
# ---------------------------------------------------------------------------
Strue = S .- mean(S, dims = 2)          # center true joint scores for fair comparison
Sest  = res.S
# subspace alignment: singular values of (orthonormal basis of Strueᵀ)ᵀ (orthonormal basis of Sestᵀ)
Qtrue = Matrix(qr(Strue').Q)[:, 1:r]
Qest  = Matrix(qr(Sest').Q)[:, 1:r]
canon = svd(Qtrue' * Qest).S            # canonical correlations between the two joint subspaces
println("TEST 5  joint subspace recovery")
println("  canonical correlations: ", round.(canon, digits = 6), "   (want all ≈ 1.0)")




# test/CompareJive.jl — JIVE: our vs r.jive vs ground truth, + benchmark.

#  data with KNOWN joint + individual structure, noiseless (ground truth) 
n = 100; r, r1, r2 = 2, 3, 3; p1, p2 = 40, 30
S  = randn(r, n)                          # shared joint scores (the TRUTH)
U1 = randn(p1, r); U2 = randn(p2, r)
S1 = randn(r1, n); W1 = randn(p1, r1)
S2 = randn(r2, n); W2 = randn(p2, r2)
X1 = U1*S + W1*S1
X2 = U2*S + W2*S2
Xc1 = X1 .- mean(X1, dims = 2)            # centered (what both methods decompose)
Xc2 = X2 .- mean(X2, dims = 2)

#  fit our jive 
res = jive([X1, X2], r, [r1, r2]; standardize = false)

#  fit r.jive on the same data, same given ranks 
@rput X1 X2 r r1 r2
R"""
library(r.jive)
fit <- jive(list(X1, X2), rankJ = r, rankA = c(r1, r2),
            method = "given", scale = FALSE, center = TRUE, showProgress = FALSE)
J1r <- fit$joint[[1]];      J2r <- fit$joint[[2]]
A1r <- fit$individual[[1]]; A2r <- fit$individual[[2]]
d1  <- fit$data[[1]];       d2  <- fit$data[[2]]      # r.jive's CENTERED data
"""
@rget J1r J2r A1r A2r d1 d2

# helper: fraction of variance reconstructed (1.0 = perfect)
varfrac(data, recon) = 1 - norm(data .- recon)^2 / norm(data)^2
# helper: joint-subspace basis from a stacked joint matrix
jbasis(Jblocks...) = Matrix(qr(svd(vcat(Jblocks...)).Vt[1:r, :]').Q)[:, 1:r]

println("="^60)
println("JIVE VALIDATION:  yours  vs  r.jive  vs  ground truth")
println("="^60)

# ---- 1. GROUND TRUTH: does each recover the noiseless data exactly? ----
println("\n[1] Reconstruction of data (fraction of variance, want 1.0)")
println("  yours  D1: ", round(varfrac(Xc1, res.J[1].+res.A[1]), digits=6),
        "   D2: ", round(varfrac(Xc2, res.J[2].+res.A[2]), digits=6))
println("  r.jive D1: ", round(varfrac(d1, J1r.+A1r), digits=6),
        "   D2: ", round(varfrac(d2, J2r.+A2r), digits=6))

# sanity: r.jive's centered data must equal ours (same input)
println("  (input check: ‖r.jive data − your centered‖² = ",
        round(norm(d1 .- Xc1)^2 + norm(d2 .- Xc2)^2, digits=10), ", want 0)")

# ---- 2. JOINT SUBSPACE: the well-determined, convention-robust comparison ----
Strue = S .- mean(S, dims=2)
Qtrue = Matrix(qr(Strue').Q)[:, 1:r]
Qmine = jbasis(res.J[1], res.J[2])
Qrjv  = jbasis(J1r, J2r)
println("\n[2] Joint subspace alignment (canonical correlations, want ≈1)")
println("  yours  vs ground truth : ", round.(svd(Qmine' * Qtrue).S, digits=6))
println("  r.jive vs ground truth : ", round.(svd(Qrjv'  * Qtrue).S, digits=6))
println("  yours  vs r.jive       : ", round.(svd(Qmine' * Qrjv ).S, digits=6))

# ---- 3. ORTHOGONALITY: the defining JIVE property (ours) ----
Jm = vcat(res.J[1], res.J[2])
println("\n[3] Orthogonality  J·Aᵢᵀ = 0  (ours, want ≈0)")
println("  ‖J·A₁ᵀ‖ = ", round(norm(Jm*res.A[1]'), digits=8),
        "   ‖J·A₂ᵀ‖ = ", round(norm(Jm*res.A[2]'), digits=8))

# ---- 4. BENCHMARK ----
println("\n[4] Timing")
print("  yours (Julia) : ")
@btime jive([$X1, $X2], $r, [$r1, $r2]; standardize=false);
R"""
library(microbenchmark)
mb <- microbenchmark(
  jive(list(X1, X2), rankJ=r, rankA=c(r1,r2),
       method="given", scale=FALSE, center=TRUE, showProgress=FALSE),
  times = 20)
cat("  r.jive (R)    :  median", round(median(mb$time)/1e6, 2), "ms\n")
"""


# jive_fast — JIVE with the SVD-reduction speedup (supplement §4).
# Identical results to jive(), much faster for WIDE data (pᵢ > n): each dataset

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