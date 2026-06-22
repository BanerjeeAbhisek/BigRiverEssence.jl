# test/TestCCA.jl — validity (ground truth) + agreement vs MultivariateStats + benchmark
using BigRiverSchneider
using MultivariateStats          # reference implementation
using LinearAlgebra, Statistics, Random
using BenchmarkTools


# PART 1 — GROUND TRUTH: does CCA recover a KNOWN correlation?
# Construct X and Y that share a planted latent direction, so we KNOW
# the leading canonical correlation should be ~1 and the rest ~0.
println("="^60); println("PART 1 — GROUND TRUTH"); println("="^60)

Random.seed!(42)
dx, dy, n = 5, 4, 300
# shared latent signal (1 common factor) + independent noise per block
z = randn(1, n)                                  # the shared latent factor (1×n)
Ax = randn(dx, 1); Ay = randn(dy, 1)             # how each block loads on it
X = Ax * z .+ 0.05 .* randn(dx, n)               # X strongly driven by z
Y = Ay * z .+ 0.05 .* randn(dy, n)               # Y strongly driven by the SAME z

M = cca(X, Y; method=:svd)
println("Canonical correlations: ", round.(M.corrs, digits=4))
println("Leading corr (expect ≈1):       ", round(M.corrs[1], digits=4))
println("Remaining corrs (expect small): ", round.(M.corrs[2:end], digits=4))

# check the canonical variates are actually that correlated
Zx = cca_transform(M, X, :x)
Zy = cca_transform(M, Y, :y)
println("corr(Zx[1,:], Zy[1,:]) (should match corrs[1]): ",
        round(abs(cor(Zx[1,:], Zy[1,:])), digits=4))


# PART 2 — AGREEMENT vs MultivariateStats (both methods)
println("\n", "="^60); println("PART 2 — vs MultivariateStats"); println("="^60)

Random.seed!(7)
dx, dy, n = 6, 5, 400
X = randn(dx, n); Y = randn(dy, n)               # columns = observations

# add some real cross-correlation so the problem is non-degenerate
shared = randn(2, n)
X[1:2, :] .+= shared; Y[1:2, :] .+= shared

# reference (MultivariateStats)
Mref_svd = fit(CCA, X, Y; method=:svd)
Mref_cov = fit(CCA, X, Y; method=:cov)
ref_corrs = correlations(Mref_svd)
ref_Px = xprojection(Mref_svd); ref_Py = yprojection(Mref_svd)

function compare_to_ref(M, ref_corrs, ref_Px, ref_Py; label="")
    println("--- $label ---")
    # correlations are sign-invariant → expect ~1e-14
    println("  corrs ‖diff‖   : ", norm(sort(M.corrs) .- sort(ref_corrs)))
    # projections: per-column sign arbitrary → compare via abs, column by column
    p = length(M.corrs)
    pxd = maximum(norm(abs.(M.xproj[:,j]) .- abs.(ref_Px[:,j])) for j in 1:p)
    pyd = maximum(norm(abs.(M.yproj[:,j]) .- abs.(ref_Py[:,j])) for j in 1:p)
    println("  Px max|diff|abs: ", pxd)
    println("  Py max|diff|abs: ", pyd)
end

M_svd = cca(X, Y; method=:svd)
M_cov = cca(X, Y; method=:cov)
compare_to_ref(M_svd, ref_corrs, ref_Px, ref_Py; label="mine :svd  vs ref :svd")
compare_to_ref(M_cov, ref_corrs, ref_Px, ref_Py; label="mine :cov  vs ref :svd")

# also: my :svd vs my :cov should agree with each other (internal consistency)
println("--- mine :svd vs mine :cov (internal) ---")
println("  corrs ‖diff‖: ", norm(sort(M_svd.corrs) .- sort(M_cov.corrs)))

# ============================================================
# PART 3 — BENCHMARK: mine vs MultivariateStats
# ============================================================
println("\n", "="^60); println("PART 3 — BENCHMARK"); println("="^60)

# small
Random.seed!(1); dx, dy, n = 6, 5, 400
Xs = randn(dx, n); Ys = randn(dy, n)
println("\n[small: dx=$dx dy=$dy n=$n]")
print("mine :svd : "); @btime cca($Xs, $Ys; method=:svd);
print("mine :cov : "); @btime cca($Xs, $Ys; method=:cov);
print("MStats:svd: "); @btime fit(CCA, $Xs, $Ys; method=:svd);
print("MStats:cov: "); @btime fit(CCA, $Xs, $Ys; method=:cov);

# bigger (more variables + observations)
Random.seed!(2); dx, dy, n = 50, 40, 2000
Xb = randn(dx, n); Yb = randn(dy, n)
println("\n[big: dx=$dx dy=$dy n=$n]")
print("mine :svd : "); @btime cca($Xb, $Yb; method=:svd);
print("mine :cov : "); @btime cca($Xb, $Yb; method=:cov);
print("MStats:svd: "); @btime fit(CCA, $Xb, $Yb; method=:svd);
print("MStats:cov: "); @btime fit(CCA, $Xb, $Yb; method=:cov);

#=
============================================================
PART 1 — GROUND TRUTH
============================================================
Canonical correlations: [0.9979, 0.218, 0.1317, 0.0173]
Leading corr (expect ≈1):       0.9979
Remaining corrs (expect small): [0.218, 0.1317, 0.0173]
corr(Zx[1,:], Zy[1,:]) (should match corrs[1]): 0.9979

============================================================
PART 2 — vs MultivariateStats
============================================================
--- mine :svd  vs ref :svd ---
  corrs ‖diff‖   : 9.625073773737373e-16
  Px max|diff|abs: 0.0
  Py max|diff|abs: 0.0
--- mine :cov  vs ref :svd ---
  corrs ‖diff‖   : 2.9153616370250645e-15
  Px max|diff|abs: 3.745842837840578e-14
  Py max|diff|abs: 8.171443474260569e-15
--- mine :svd vs mine :cov (internal) ---
  corrs ‖diff‖: 2.787624521554686e-15

============================================================
PART 3 — BENCHMARK
============================================================

[small: dx=6 dy=5 n=400]
mine :svd :   117.833 μs (89 allocations: 121.59 KiB)
mine :cov :   21.833 μs (88 allocations: 42.88 KiB)
MStats:svd:   121.875 μs (93 allocations: 153.20 KiB)
MStats:cov:   21.166 μs (68 allocations: 41.55 KiB)

[big: dx=50 dy=40 n=2000]
mine :svd :   6.163 ms (103 allocations: 4.61 MiB)
mine :cov :   699.791 μs (242 allocations: 1.67 MiB)
MStats:svd:   6.439 ms (105 allocations: 5.83 MiB)
MStats:cov:   697.166 μs (81 allocations: 1.62 MiB)
=#













# test/TestCCAOpt.jl — three-way: cca_opt vs cca vs MultivariateStats, + benchmark
using BigRiverSchneider
using MultivariateStats
using LinearAlgebra, Statistics, Random
using BenchmarkTools

# per-column abs comparison (per-column sign is arbitrary in SVD/eigen)
proj_absdiff(A, B) = maximum(norm(abs.(A[:,j]) .- abs.(B[:,j])) for j in 1:size(A,2))

# ============================================================
# CORRECTNESS: cca_opt vs cca vs MultivariateStats
# ============================================================
function check(X, Y; label="")
    println("="^60); println("CORRECTNESS — $label"); println("="^60)
    for meth in (:svd, :cov)
        mine     = cca(X, Y; method=meth)
        opt      = cca_opt(X, Y; method=meth)
        ref      = fit(CCA, X, Y; method=meth)
        rc       = correlations(ref)
        rPx, rPy = xprojection(ref), yprojection(ref)
        p        = length(mine.corrs)

        println("--- method = :$meth ---")
        # opt vs original (should be ~1e-14, ideally 0.0 — both deterministic)
        println("  opt vs orig  | corrs ‖diff‖: ", norm(opt.corrs .- mine.corrs),
                "  Px: ", proj_absdiff(opt.xproj, mine.xproj),
                "  Py: ", proj_absdiff(opt.yproj, mine.yproj))
        # orig vs MultivariateStats
        println("  orig vs MVS  | corrs ‖diff‖: ", norm(sort(mine.corrs) .- sort(rc)),
                "  Px: ", proj_absdiff(mine.xproj, rPx),
                "  Py: ", proj_absdiff(mine.yproj, rPy))
        # opt vs MultivariateStats
        println("  opt  vs MVS  | corrs ‖diff‖: ", norm(sort(opt.corrs) .- sort(rc)),
                "  Px: ", proj_absdiff(opt.xproj, rPx),
                "  Py: ", proj_absdiff(opt.yproj, rPy))
    end
end

# small problem with planted cross-correlation
Random.seed!(7); dx, dy, n = 6, 5, 400
Xs = randn(dx, n); Ys = randn(dy, n)
sh = randn(2, n); Xs[1:2,:] .+= sh; Ys[1:2,:] .+= sh
check(Xs, Ys; label="small (dx=6, dy=5, n=400)")

# big problem
Random.seed!(2); dxb, dyb, nb = 50, 40, 2000
Xb = randn(dxb, nb); Yb = randn(dyb, nb)
shb = randn(5, nb); Xb[1:5,:] .+= shb; Yb[1:5,:] .+= shb
check(Xb, Yb; label="big (dx=50, dy=40, n=2000)")

# ============================================================
# BENCHMARK: all three, both methods, both sizes
# ============================================================
println("\n", "="^60); println("BENCHMARK"); println("="^60)
for (lbl, Xt, Yt) in (("small (6×5, n=400)", Xs, Ys), ("big (50×40, n=2000)", Xb, Yb))
    println("\n[$lbl]")
    for meth in (:svd, :cov)
        println("  method = :$meth")
        print("    orig: "); @btime cca($Xt, $Yt; method=$meth);
        print("    opt : "); @btime cca_opt($Xt, $Yt; method=$meth);
        print("    MVS : "); @btime fit(CCA, $Xt, $Yt; method=$meth);
    end
end


#=
============================================================
CORRECTNESS — small (dx=6, dy=5, n=400)
============================================================
--- method = :svd ---
  opt vs orig  | corrs ‖diff‖: 0.0  Px: 2.306595211858756e-16  Py: 1.5709546994741626e-16
  orig vs MVS  | corrs ‖diff‖: 9.625073773737373e-16  Px: 0.0  Py: 0.0
  opt  vs MVS  | corrs ‖diff‖: 9.625073773737373e-16  Px: 2.306595211858756e-16  Py: 1.5709546994741626e-16
--- method = :cov ---
  opt vs orig  | corrs ‖diff‖: 3.928799415222704e-15  Px: 6.53487914565665e-14  Py: 4.713717804298947e-15
  orig vs MVS  | corrs ‖diff‖: 2.760110231135371e-15  Px: 6.53487914565665e-14  Py: 4.713717804298947e-15
  opt  vs MVS  | corrs ‖diff‖: 1.338506384008595e-15  Px: 1.1464934308770758e-16  Py: 0.0
============================================================
CORRECTNESS — big (dx=50, dy=40, n=2000)
============================================================
--- method = :svd ---
  opt vs orig  | corrs ‖diff‖: 0.0  Px: 4.658370492820766e-16  Py: 4.592183321777766e-16
  orig vs MVS  | corrs ‖diff‖: 1.2579221424368276e-15  Px: 0.0  Py: 0.0
  opt  vs MVS  | corrs ‖diff‖: 1.2579221424368276e-15  Px: 4.658370492820766e-16  Py: 4.592183321777766e-16
--- method = :cov ---
  opt vs orig  | corrs ‖diff‖: 1.6966979060436248e-15  Px: 2.0076579779082758e-13  Py: 2.0685300556901982e-13
  orig vs MVS  | corrs ‖diff‖: 1.6563548170316412e-15  Px: 2.007687979855362e-13  Py: 2.0685300556901982e-13
  opt  vs MVS  | corrs ‖diff‖: 1.0135417890801775e-15  Px: 1.3559169519204648e-16  Py: 0.0

============================================================
BENCHMARK
============================================================

[small (6×5, n=400)]
  method = :svd
    orig:   116.208 μs (89 allocations: 121.59 KiB)
    opt :   114.792 μs (75 allocations: 85.22 KiB)
    MVS :   119.083 μs (93 allocations: 153.20 KiB)
  method = :cov
    orig:   20.666 μs (88 allocations: 42.88 KiB)
    opt :   20.458 μs (64 allocations: 40.94 KiB)
    MVS :   20.541 μs (68 allocations: 41.55 KiB)

[big (50×40, n=2000)]
  method = :svd
    orig:   6.191 ms (103 allocations: 4.61 MiB)
    opt :   6.098 ms (85 allocations: 3.15 MiB)
    MVS :   6.500 ms (105 allocations: 5.83 MiB)
  method = :cov
    orig:   699.375 μs (242 allocations: 1.67 MiB)
    opt :   698.875 μs (75 allocations: 1.59 MiB)
    MVS :   699.042 μs (81 allocations: 1.62 MiB)
=#