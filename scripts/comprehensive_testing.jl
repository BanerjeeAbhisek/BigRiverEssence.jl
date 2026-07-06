# benchmark_all_extensive.jl — multi-size similarity + benchmark sweep, BigRiverSchneider.jl
# Per method, per size:
#   (1) CORRECTNESS — verify results are identical to the reference (printed first)
#   (2) @btime  — Julia time + allocations + bytes (native @btime line, all three shown)
#   (3) microbenchmark — R median (where the reference is an R package)
#
# === R REPEAT POLICY ===
#  • R microbenchmark reps are ADAPTIVE: 20 reps for small/cheap cases, 3 reps for large.
#    Decided by matrix size (number of cells) via `rreps`. The printed line reports the
#    rep count it used, so output stays self-documenting.
#  • A hard SKIP guard avoids R benchmarks above `RSKIP_CELLS` (the Julia side still runs
#    and correctness is still checked); prints "skipped (too large for R timing)".
#
# === WARNINGS — read before running ===
#  • Julia huge tier (10000×5000) is fine (seconds). The R side is the risk: even at 3
#    reps, r.jive / PMA / mixOmics on huge matrices can run for minutes and spike RAM.
#    The SKIP guard is there to prevent hangs; raise RSKIP_CELLS if you really want them.
#  • Julia (@btime) and R (microbenchmark) times are SEPARATE-runtime medians — not a
#    head-to-head ratio. R times exclude RCall marshalling.
#  • Similarity is sign-invariant where the method has sign ambiguity (|cos|/|dot|).
#
# Run:  julia --project benchmark_all_extensive.jl

using BigRiverSchneider
using LinearAlgebra, Statistics, Random, BenchmarkTools, Printf
import MultivariateStats; const MVS = MultivariateStats
using RCall                        # unconditional at top level so @rput / R"..." parse

const BRS = BigRiverSchneider

# --- R timing policy -------------------------------------------------------
const RSMALL_REPS = 20             # reps for small/cheap cases
const RBIG_REPS   = 3              # reps for large cases
const RREP_CUTOFF = 300_000        # cells (n*p) at/above which we drop to RBIG_REPS
const RSKIP_CELLS = 20_000_000     # cells at/above which we SKIP the R benchmark entirely

# how many R reps for a problem of this size (many when cheap, few when huge)
rreps(cells) = cells ≥ RREP_CUTOFF ? RBIG_REPS : RSMALL_REPS

# --- optional deps ---------------------------------------------------------
const HAS_JCHEMO = let
    try; @eval import Jchemo; true; catch; false; end
end
const R_OK = Dict{String,Bool}()
for pk in ("PMA", "r.jive", "mixOmics", "microbenchmark")
    R_OK[pk] = rcopy(reval("requireNamespace('$pk', quietly = TRUE)"))
    R_OK[pk] || @warn "R package '$pk' not installed; its section will be skipped."
end

# --- helpers ---------------------------------------------------------------
hr() = println("="^92)
section(name) = (println(); hr(); println("  ", name); hr())

colmisalign(A, B) = maximum(1 - abs(dot(normalize(A[:,j]), normalize(B[:,j]))) for j in 1:size(A,2))
selset(col) = Set(findall(!iszero, col))

# print the correctness verdict line prominently
function verdict(misalign; thresh = 1e-2, sets = nothing)
    ok = misalign < thresh && (sets === nothing || sets)
    tag = ok ? "IDENTICAL ✓" : "DIFFER ✗"
    s = sets === nothing ? "" : (sets ? "  (selected sets match)" : "  (SELECTED SETS DIFFER)")
    @printf("    result: %-12s  max misalignment = %.2e%s\n", tag, misalign, s)
end

# run an R microbenchmark, print its median ms. `cells` sets the rep count and the skip guard.
# (data must already be @rput into R)
function rtime(expr::String, cells::Int)
    R_OK["microbenchmark"] || (println("      R: microbenchmark not available"); return)
    if cells ≥ RSKIP_CELLS
        @printf("      R  microbenchmark: skipped (too large for R timing, %d cells)\n", cells)
        return
    end
    times = rreps(cells)
    res = reval("""
        suppressMessages(library(microbenchmark))
        mb <- microbenchmark($expr, times = $times)
        median(mb\$time) / 1e6
    """)
    @printf("      R  microbenchmark median: %.3f ms (in-R, %d reps)\n", rcopy(res), times)
end

println("\nBigRiverSchneider.jl — EXTENSIVE multi-size similarity + benchmark sweep")
println("Jchemo: $(HAS_JCHEMO ? "yes" : "NO")    Julia: all sizes run.  R: adaptive reps (",
        RSMALL_REPS, "→", RBIG_REPS, "), skip ≥ ", RSKIP_CELLS, " cells.")
println("@btime shows time + allocations + bytes for every Julia fit; R uses microbenchmark.")
println("Julia and R times are separate-runtime medians, not a direct ratio.\n")

# ===========================================================================
# 1. PCA  vs MultivariateStats.PCA   (X is obs×var)
# ===========================================================================
section("PCA  vs  MultivariateStats.PCA   [X = n×p, obs×var]")
for (n, p, k) in [(200,40,6), (1000,100,10), (5000,200,20),
                  (200,500,20), (2000,1000,30), (10000,5000,50)]
    Random.seed!(1)
    X = randn(n, p); Xt = permutedims(X)
    @printf("\n  n=%d p=%d k=%d   (auto picks %s)\n", n, p, k, n >= p ? ":cov" : ":svd")
    m  = BRS.pca(X; k=k, method=:auto)
    rm = MVS.fit(MVS.PCA, Xt; maxoutdim=k, pratio=1.0)
    verdict(colmisalign(m.loadings, MVS.projection(rm)); thresh = 1e-6)
    print("      BRS.pca (:auto) : "); @btime BRS.pca($X; k=$k, method=:auto);
    print("      MVS.fit(PCA)    : "); @btime MVS.fit(MVS.PCA, $Xt; maxoutdim=$k, pratio=1.0);
end

# ===========================================================================
# 2. PLSKERN  vs Jchemo.plskern   (X is obs×var, y vector)
# ===========================================================================
section("PLSKERN  vs  Jchemo.plskern   [X = n×p, y = n]")
if !HAS_JCHEMO
    println("    Jchemo not installed; skipping.")
else
    for (n, p, nlv) in [(200,30,8), (1000,50,12), (5000,100,20),
                        (400,300,30), (2000,500,25), (10000,2000,40)]
        Random.seed!(1234)
        X = randn(n, p); y = randn(n)
        @printf("\n  n=%d p=%d nlv=%d\n", n, p, nlv)
        m = BRS.plskern(copy(X), reshape(copy(y), :, 1); nlv=nlv, method=:algo1)
        B_mine, _ = BRS.plskerncoef(m)
        mod = Jchemo.plskern(; nlv=nlv); Jchemo.fit!(mod, X, y)
        verdict(maximum(abs.(B_mine .- Jchemo.coef(mod).B)); thresh = 1e-8)
        print("      BRS.plskern :algo1 : "); @btime BRS.plskern(Xc, Yc; nlv=$nlv, method=:algo1) setup=(Xc=copy($X); Yc=reshape(copy($y),:,1));
        print("      BRS.plskern :algo2 : "); @btime BRS.plskern(Xc, Yc; nlv=$nlv, method=:algo2) setup=(Xc=copy($X); Yc=reshape(copy($y),:,1));
        print("      Jchemo.plskern     : "); @btime Jchemo.fit!(Jchemo.plskern(; nlv=$nlv), $X, $y);
    end
end

# ===========================================================================
# 3. CCA  vs MultivariateStats.CCA   (X,Y are var×obs)
# ===========================================================================
section("CCA  vs  MultivariateStats.CCA   [X = dx×n, Y = dy×n, var×obs]")
for (dx, dy, n) in [(6,5,300), (20,15,1000), (50,40,5000),
                    (100,80,2000), (200,150,3000), (500,400,10000)]
    Random.seed!(7)
    X = randn(dx, n); Y = randn(dy, n)
    ns = min(dx, dy, 3); sh = randn(ns, n); X[1:ns,:] .+= sh; Y[1:ns,:] .+= sh
    @printf("\n  dx=%d dy=%d n=%d\n", dx, dy, n)
    m  = BRS.cca(X, Y; method=:svd)
    rm = MVS.fit(MVS.CCA, X, Y; method=:svd)
    dc = norm(sort(m.corrs) .- sort(MVS.correlations(rm)))
    dP = colmisalign(m.xproj, MVS.xprojection(rm))
    verdict(max(dc, dP); thresh = 1e-8)
    print("      BRS.cca       : "); @btime BRS.cca($X, $Y; method=:svd);
    print("      MVS.fit(CCA)  : "); @btime MVS.fit(MVS.CCA, $X, $Y; method=:svd);
end

# ===========================================================================
# 4. PMD  vs PMA::PMD (R)   (X is n×p)
# ===========================================================================
section("PMD  vs  PMA::PMD (R)   [X = n×p]")
if !R_OK["PMA"]
    println("    PMA not available; skipping.")
else
    for (n, p) in [(60,40), (200,150), (1000,500),
                   (300,800), (2000,1000), (5000,3000)]
        Random.seed!(11)
        X = randn(n, p); sumabs = 0.4; K = 3
        @printf("\n  n=%d p=%d\n", n, p)
        @rput X sumabs K
        R"""
        suppressMessages(library(PMA))
        rj <- PMD(X, type='standard', sumabs=sumabs, K=K, center=TRUE, trace=FALSE)
        """
        rv = rcopy(R"rj$v"); ru = rcopy(R"rj$u")
        m = BRS.pmd(copy(X); sumabs=sumabs, K=K, center=true)
        verdict(max(colmisalign(m.v, rv), colmisalign(m.u, ru)))
        print("      BRS.pmd : "); @btime BRS.pmd(Xc; sumabs=$sumabs, K=$K, center=true) setup=(Xc=copy($X));
        rtime("PMD(X, type='standard', sumabs=sumabs, K=K, center=TRUE, trace=FALSE)", n*p)
    end
end

# ===========================================================================
# 5. SPC  vs PMA::SPC (R)   (X is n×p)
# ===========================================================================
section("SPC  vs  PMA::SPC (R)   [X = n×p]")
if !R_OK["PMA"]
    println("    PMA not available; skipping.")
else
    for (n, p) in [(80,30), (300,100), (1000,400),
                   (200,600), (2000,800), (5000,3000)]
        Random.seed!(13)
        X = randn(n, p); sumabsv = sqrt(p)/2; K = 3
        @printf("\n  n=%d p=%d\n", n, p)
        @rput X sumabsv K
        R"""
        suppressMessages(library(PMA))
        rj <- SPC(scale(X, center=TRUE, scale=FALSE), sumabsv=sumabsv, K=K, trace=FALSE)
        """
        rv = rcopy(R"rj$v")
        m = BRS.spc(copy(X); k=K, c=sumabsv)
        sm = all(selset(m.loadings[:,j]) == selset(rv[:,j]) for j in 1:K)
        verdict(colmisalign(m.loadings, rv); sets = sm)
        print("      BRS.spc : "); @btime BRS.spc(Xc; k=$K, c=$sumabsv) setup=(Xc=copy($X));
        rtime("SPC(scale(X, center=TRUE, scale=FALSE), sumabsv=sumabsv, K=K, trace=FALSE)", n*p)
    end
end

# ===========================================================================
# 6. SCCA  vs PMA::CCA (R)   (PMA layout: Xr,Zr obs×var; BRS wants var×obs)
# ===========================================================================
section("SCCA  vs  PMA::CCA (R)   [Xr = n×p1, Zr = n×p2, obs×var]")
if !R_OK["PMA"]
    println("    PMA not available; skipping.")
else
    for (n, p1, p2) in [(100,40,50), (300,150,200), (500,800,1000),
                        (1000,500,600), (2000,1500,2000), (3000,4000,5000)]
        Random.seed!(17)
        Xr = randn(n,p1); Zr = randn(n,p2); px=0.3; pz=0.3; K=2; niter=15
        @printf("\n  n=%d p1=%d p2=%d\n", n, p1, p2)
        @rput Xr Zr px pz K niter
        R"""
        suppressMessages(library(PMA))
        rj <- CCA(Xr, Zr, typex='standard', typez='standard',
                  penaltyx=px, penaltyz=pz, K=K, niter=niter, trace=FALSE)
        """
        ru = rcopy(R"rj$u"); rv = rcopy(R"rj$v")
        Xc_ = Matrix(transpose(Xr)); Zc_ = Matrix(transpose(Zr))
        m = BRS.scca(Xc_, Zc_; penaltyx=px, penaltyz=pz, K=K, niter=niter)
        sm = all(selset(m.u[:,j])==selset(ru[:,j]) && selset(m.v[:,j])==selset(rv[:,j]) for j in 1:K)
        verdict(max(colmisalign(m.u, ru), colmisalign(m.v, rv)); sets = sm)
        print("      BRS.scca : "); @btime BRS.scca($Xc_, $Zc_; penaltyx=$px, penaltyz=$pz, K=$K, niter=$niter);
        rtime("CCA(Xr, Zr, typex='standard', typez='standard', penaltyx=px, penaltyz=pz, K=K, niter=niter, trace=FALSE)", n*(p1+p2))
    end
end

# ===========================================================================
# 7. JIVE  vs r.jive (R)   (blocks var×obs; given ranks)
# ===========================================================================
section("JIVE  vs  r.jive (R)   [X1 = p1×n, X2 = p2×n, var×obs; given ranks]")
if !R_OK["r.jive"]
    println("    r.jive not available; skipping.")
else
    canon(B1,B2,d) = (Q1=Matrix(qr(B1).Q)[:,1:d]; Q2=Matrix(qr(B2).Q)[:,1:d]; svdvals(Q1'Q2))
    for (n, p1, p2) in [(80,60,50), (200,150,120), (500,400,300),
                        (300,800,700), (1000,1000,800), (2000,3000,2500)]
        rT,r1T,r2T = 2,3,3
        Random.seed!(2024)
        S=randn(rT,n); U1=randn(p1,rT); U2=randn(p2,rT)
        S1=randn(r1T,n); W1=randn(p1,r1T); S2=randn(r2T,n); W2=randn(p2,r2T)
        X1 = U1*S + W1*S1 .+ 0.3.*randn(p1,n)
        X2 = U2*S + W2*S2 .+ 0.3.*randn(p2,n)
        @printf("\n  n=%d p1=%d p2=%d\n", n, p1, p2)
        @rput X1 X2 rT r1T r2T
        R"""
        suppressMessages(library(r.jive))
        rj <- jive(list(X1,X2), rankJ=rT, rankA=c(r1T,r2T), method='given',
                   conv=1e-6, maxiter=1000, scale=TRUE, center=TRUE, showProgress=FALSE)
        """
        rJ1 = rcopy(R"rj$joint[[1]]"); rJ2 = rcopy(R"rj$joint[[2]]")
        Xblocks = [Matrix{Float64}(X1), Matrix{Float64}(X2)]
        m = BRS.jive(Xblocks, rT, [r1T, r2T])
        cc = minimum(canon(reduce(vcat, m.J)', vcat(rJ1,rJ2)', rT))
        # report as (1 − min canonical corr) so smaller = more identical, like the others
        verdict(1 - cc; thresh = 1e-2)
        print("      BRS.jive : "); @btime BRS.jive($Xblocks, $rT, [$r1T, $r2T]);
        rtime("jive(list(X1,X2), rankJ=rT, rankA=c(r1T,r2T), method='given', conv=1e-6, maxiter=1000, scale=TRUE, center=TRUE, showProgress=FALSE)", n*(p1+p2))
    end
end

# ===========================================================================
# 8. SPLSDA  vs mixOmics::splsda (R)   (X is n×p, y labels)
# ===========================================================================
section("SPLSDA  vs  mixOmics::splsda (R)   [X = n×p, 3 classes]")
if !R_OK["mixOmics"]
    println("    mixOmics not available; skipping.")
else
    for (nper, p, kx) in [(20,200,10), (50,500,20), (200,1000,30),
                          (30,1500,15), (100,2000,40), (300,5000,50)]
        n = 3*nper
        Random.seed!(123)
        classes = ["A","B","C"]; y = repeat(classes, inner=nper)
        X = randn(n, p) .* 0.5
        for (ci,cls) in enumerate(classes); X[findall(==(cls),y), 1:10] .+= ci*2.0; end
        ncomp=2; keepX=[kx,kx]; yR = y
        @printf("\n  n=%d p=%d keepX=%d\n", n, p, kx)
        @rput X yR ncomp keepX
        R"""
        suppressMessages(library(mixOmics))
        rj <- splsda(X, factor(yR), ncomp=ncomp, keepX=keepX)
        """
        rlx = rcopy(R"rj$loadings$X")
        m = BRS.splsda(copy(X), y, ncomp, keepX)
        sm = all(selset(m.loadings_X[:,j]) == selset(rlx[:,j]) for j in 1:ncomp)
        verdict(colmisalign(m.loadings_X, rlx); sets = sm)
        print("      BRS.splsda : "); @btime BRS.splsda(Xc, $y, $ncomp, $keepX) setup=(Xc=copy($X));
        rtime("splsda(X, factor(yR), ncomp=ncomp, keepX=keepX)", n*p)
    end
end

hr()
println("  Done. Julia ran every size; R used adaptive reps (", RSMALL_REPS, "→", RBIG_REPS,
        ") and skipped any case ≥ ", RSKIP_CELLS, " cells.")
println("  '@btime' lines show time + allocations + bytes; R lines are in-R microbenchmark medians.")
println("  Julia and R times are separate-runtime medians, not a direct ratio.")
hr()

#=
BigRiverSchneider.jl — EXTENSIVE multi-size similarity + benchmark sweep
Jchemo: yes    Julia: all sizes run.  R: adaptive reps (20→3), skip ≥ 20000000 cells.
@btime shows time + allocations + bytes for every Julia fit; R uses microbenchmark.
Julia and R times are separate-runtime medians, not a direct ratio.


============================================================================================
  PCA  vs  MultivariateStats.PCA   [X = n×p, obs×var]
============================================================================================

  n=200 p=40 k=6   (auto picks :cov)
    result: IDENTICAL ✓   max misalignment = 2.22e-16
      BRS.pca (:auto) :   127.083 μs (41 allocations: 56.61 KiB)
      MVS.fit(PCA)    :   150.709 μs (41 allocations: 120.55 KiB)

  n=1000 p=100 k=10   (auto picks :cov)
    result: IDENTICAL ✓   max misalignment = 3.33e-16
      BRS.pca (:auto) :   722.625 μs (42 allocations: 335.73 KiB)
      MVS.fit(PCA)    :   1.161 ms (42 allocations: 1.11 MiB)

  n=5000 p=200 k=20   (auto picks :cov)
    result: IDENTICAL ✓   max misalignment = 4.44e-16
      BRS.pca (:auto) :   4.235 ms (43 allocations: 1.06 MiB)
      MVS.fit(PCA)    :   6.495 ms (43 allocations: 8.71 MiB)

  n=200 p=500 k=20   (auto picks :svd)
    result: IDENTICAL ✓   max misalignment = 8.88e-16
      BRS.pca (:auto) :   9.687 ms (42 allocations: 4.02 MiB)
      MVS.fit(PCA)    :   9.656 ms (39 allocations: 4.02 MiB)

  n=2000 p=1000 k=30   (auto picks :cov)
    result: IDENTICAL ✓   max misalignment = 1.78e-15
      BRS.pca (:auto) :   78.706 ms (46 allocations: 23.63 MiB)
      MVS.fit(PCA)    :   156.916 ms (46 allocations: 38.90 MiB)

  n=10000 p=5000 k=50   (auto picks :cov)
    result: IDENTICAL ✓   max misalignment = 3.11e-15
      BRS.pca (:auto) :   10.711 s (46 allocations: 576.16 MiB)
      MVS.fit(PCA)    :   18.402 s (46 allocations: 957.64 MiB)

============================================================================================
  PLSKERN  vs  Jchemo.plskern   [X = n×p, y = n]
============================================================================================

  n=200 p=30 nlv=8
    result: IDENTICAL ✓   max misalignment = 1.11e-16
      BRS.plskern :algo1 :   20.916 μs (45 allocations: 23.30 KiB)
      BRS.plskern :algo2 :   24.291 μs (48 allocations: 30.88 KiB)
      Jchemo.plskern     :   98.083 μs (498 allocations: 111.38 KiB)

  n=1000 p=50 nlv=12
    result: IDENTICAL ✓   max misalignment = 5.55e-17
      BRS.plskern :algo1 :   252.458 μs (49 allocations: 123.44 KiB)
      BRS.plskern :algo2 :   232.667 μs (52 allocations: 143.52 KiB)
      Jchemo.plskern     :   405.167 μs (843 allocations: 602.89 KiB)

  n=5000 p=100 nlv=20
    result: IDENTICAL ✓   max misalignment = 8.33e-17
      BRS.plskern :algo1 :   1.541 ms (49 allocations: 919.12 KiB)
      BRS.plskern :algo2 :   1.539 ms (52 allocations: 1015.20 KiB)
      Jchemo.plskern     :   2.563 ms (1803 allocations: 5.06 MiB)

  n=400 p=300 nlv=30
    result: IDENTICAL ✓   max misalignment = 5.00e-16
      BRS.plskern :algo1 :   1.454 ms (55 allocations: 404.64 KiB)
      BRS.plskern :algo2 :   1.685 ms (58 allocations: 1.08 MiB)
      Jchemo.plskern     :   1.823 ms (3549 allocations: 1.49 MiB)

  n=2000 p=500 nlv=25
    result: IDENTICAL ✓   max misalignment = 1.11e-16
      BRS.plskern :algo1 :   3.712 ms (55 allocations: 842.09 KiB)
      BRS.plskern :algo2 :   7.324 ms (58 allocations: 2.76 MiB)
      Jchemo.plskern     :   5.736 ms (2605 allocations: 8.65 MiB)

  n=10000 p=2000 nlv=40
    result: IDENTICAL ✓   max misalignment = 1.01e-16
      BRS.plskern :algo1 :   294.845 ms (55 allocations: 5.13 MiB)
      BRS.plskern :algo2 :   621.218 ms (58 allocations: 37.13 MiB)
      Jchemo.plskern     :   343.432 ms (5892 allocations: 158.35 MiB)

============================================================================================
  CCA  vs  MultivariateStats.CCA   [X = dx×n, Y = dy×n, var×obs]
============================================================================================

  dx=6 dy=5 n=300
    result: IDENTICAL ✓   max misalignment = 6.15e-16
      BRS.cca       :   89.667 μs (75 allocations: 68.22 KiB)
      MVS.fit(CCA)  :   93.417 μs (93 allocations: 119.70 KiB)

  dx=20 dy=15 n=1000
    result: IDENTICAL ✓   max misalignment = 1.49e-15
      BRS.cca       :   1.154 ms (79 allocations: 633.53 KiB)
      MVS.fit(CCA)  :   1.198 ms (98 allocations: 1.15 MiB)

  dx=50 dy=40 n=5000
    result: IDENTICAL ✓   max misalignment = 1.22e-15
      BRS.cca       :   14.158 ms (85 allocations: 7.27 MiB)
      MVS.fit(CCA)  :   14.533 ms (105 allocations: 13.83 MiB)

  dx=100 dy=80 n=2000
    result: IDENTICAL ✓   max misalignment = 3.23e-15
      BRS.cca       :   17.926 ms (85 allocations: 6.74 MiB)
      MVS.fit(CCA)  :   18.496 ms (105 allocations: 12.15 MiB)

  dx=200 dy=150 n=3000
    result: IDENTICAL ✓   max misalignment = 2.64e-15
      BRS.cca       :   87.721 ms (85 allocations: 20.23 MiB)
      MVS.fit(CCA)  :   99.588 ms (105 allocations: 35.58 MiB)

  dx=500 dy=400 n=10000
    result: IDENTICAL ✓   max misalignment = 3.45e-15
      BRS.cca       :   1.319 s (95 allocations: 166.96 MiB)
      MVS.fit(CCA)  :   1.444 s (116 allocations: 300.97 MiB)

============================================================================================
  PMD  vs  PMA::PMD (R)   [X = n×p]
============================================================================================

  n=60 p=40
    result: IDENTICAL ✓   max misalignment = 1.11e-16
      BRS.pmd :   311.667 μs (44 allocations: 161.42 KiB)
┌ Warning: RCall.jl: Warning in microbenchmark(PMD(X, type = "standard", sumabs = sumabs, K = K,  :
│   less accurate nanosecond times to avoid potential integer overflows
└ @ RCall ~/.julia/packages/RCall/fTLHT/src/io.jl:166
      R  microbenchmark median: 14.398 ms (in-R, 20 reps)

  n=200 p=150
    result: IDENTICAL ✓   max misalignment = 3.33e-16
      BRS.pmd :   4.522 ms (47 allocations: 1.81 MiB)
      R  microbenchmark median: 36.434 ms (in-R, 20 reps)

  n=1000 p=500
    result: IDENTICAL ✓   max misalignment = 2.22e-16
      BRS.pmd :   80.050 ms (53 allocations: 21.30 MiB)
      R  microbenchmark median: 583.364 ms (in-R, 3 reps)

  n=300 p=800
    result: IDENTICAL ✓   max misalignment = 1.11e-16
      BRS.pmd :   27.082 ms (65 allocations: 8.65 MiB)
      R  microbenchmark median: 170.143 ms (in-R, 20 reps)

  n=2000 p=1000
    result: IDENTICAL ✓   max misalignment = 1.11e-16
      BRS.pmd :   424.638 ms (53 allocations: 91.92 MiB)
      R  microbenchmark median: 3471.133 ms (in-R, 3 reps)

  n=5000 p=3000
    result: IDENTICAL ✓   max misalignment = 8.88e-16
      BRS.pmd :   9.886 s (53 allocations: 710.47 MiB)
      R  microbenchmark median: 80049.706 ms (in-R, 3 reps)

============================================================================================
  SPC  vs  PMA::SPC (R)   [X = n×p]
============================================================================================

  n=80 p=30
    result: IDENTICAL ✓   max misalignment = 1.07e-13  (selected sets match)
      BRS.spc :   213.584 μs (99 allocations: 82.09 KiB)
      R  microbenchmark median: 8.763 ms (in-R, 20 reps)

  n=300 p=100
    result: IDENTICAL ✓   max misalignment = 2.40e-14  (selected sets match)
      BRS.spc :   1.540 ms (104 allocations: 863.80 KiB)
      R  microbenchmark median: 41.965 ms (in-R, 20 reps)

  n=1000 p=400
    result: IDENTICAL ✓   max misalignment = 1.01e-14  (selected sets match)
      BRS.spc :   17.821 ms (113 allocations: 10.13 MiB)
      R  microbenchmark median: 1890.683 ms (in-R, 3 reps)

  n=200 p=600
    result: IDENTICAL ✓   max misalignment = 2.45e-14  (selected sets match)
      BRS.spc :   6.102 ms (113 allocations: 2.97 MiB)
      R  microbenchmark median: 251.823 ms (in-R, 20 reps)

  n=2000 p=800
    result: IDENTICAL ✓   max misalignment = 8.88e-15  (selected sets match)
      BRS.spc :   68.980 ms (113 allocations: 39.64 MiB)
      R  microbenchmark median: 14919.128 ms (in-R, 3 reps)

  n=5000 p=3000
    result: IDENTICAL ✓   max misalignment = 2.33e-14  (selected sets match)
      BRS.spc :   3.065 s (113 allocations: 436.64 MiB)
      R  microbenchmark median: 502512.335 ms (in-R, 3 reps)

============================================================================================
  SCCA  vs  PMA::CCA (R)   [Xr = n×p1, Zr = n×p2, obs×var]
============================================================================================

  n=100 p1=40 p2=50
    result: IDENTICAL ✓   max misalignment = 0.00e+00  (selected sets match)
      BRS.scca :   338.167 μs (120 allocations: 431.83 KiB)
      R  microbenchmark median: 8.895 ms (in-R, 20 reps)

  n=300 p1=150 p2=200
    result: IDENTICAL ✓   max misalignment = 2.22e-16  (selected sets match)
      BRS.scca :   5.946 ms (129 allocations: 4.00 MiB)
      R  microbenchmark median: 38.517 ms (in-R, 20 reps)

  n=500 p1=800 p2=1000
    result: IDENTICAL ✓   max misalignment = 0.00e+00  (selected sets match)
      BRS.scca :   148.435 ms (177 allocations: 51.98 MiB)
      R  microbenchmark median: 2128.726 ms (in-R, 3 reps)

  n=1000 p1=500 p2=600
    result: IDENTICAL ✓   max misalignment = 2.22e-16  (selected sets match)
      BRS.scca :   87.534 ms (145 allocations: 40.12 MiB)
      R  microbenchmark median: 606.168 ms (in-R, 3 reps)

  n=2000 p1=1500 p2=2000
    result: IDENTICAL ✓   max misalignment = 2.22e-16  (selected sets match)
      BRS.scca :   2.281 s (145 allocations: 301.15 MiB)
      R  microbenchmark median: 13016.697 ms (in-R, 3 reps)

  n=3000 p1=4000 p2=5000
    result: IDENTICAL ✓   max misalignment = 4.44e-16  (selected sets match)
      BRS.scca :   21.477 s (177 allocations: 1.57 GiB)
      R  microbenchmark: skipped (too large for R timing, 27000000 cells)

============================================================================================
  JIVE  vs  r.jive (R)   [X1 = p1×n, X2 = p2×n, var×obs; given ranks]
============================================================================================

  n=80 p1=60 p2=50
    result: IDENTICAL ✓   max misalignment = 1.97e-09
      BRS.jive :   36.445 ms (1359 allocations: 17.88 MiB)
      R  microbenchmark median: 2325.424 ms (in-R, 20 reps)

  n=200 p1=150 p2=120
    result: IDENTICAL ✓   max misalignment = 9.64e-09
      BRS.jive :   427.361 ms (1515 allocations: 96.79 MiB)
      R  microbenchmark median: 2895.696 ms (in-R, 20 reps)

  n=500 p1=400 p2=300
    result: IDENTICAL ✓   max misalignment = 1.09e-07
      BRS.jive :   3.991 s (1410 allocations: 529.95 MiB)
      R  microbenchmark median: 10540.084 ms (in-R, 3 reps)

  n=300 p1=800 p2=700
    result: IDENTICAL ✓   max misalignment = 9.57e-08
      BRS.jive :   1.865 s (1584 allocations: 362.27 MiB)
      R  microbenchmark median: 6761.173 ms (in-R, 3 reps)

  n=1000 p1=1000 p2=800
    result: IDENTICAL ✓   max misalignment = 8.46e-07
      BRS.jive :   34.673 s (1590 allocations: 3.47 GiB)
      R  microbenchmark median: 115617.339 ms (in-R, 3 reps)

  n=2000 p1=3000 p2=2500
    result: IDENTICAL ✓   max misalignment = 3.04e-06
      BRS.jive :   245.271 s (1584 allocations: 15.40 GiB)
      R  microbenchmark median: 1023389.684 ms (in-R, 3 reps)

============================================================================================
  SPLSDA  vs  mixOmics::splsda (R)   [X = n×p, 3 classes]
============================================================================================

  n=60 p=200 keepX=10
    result: IDENTICAL ✓   max misalignment = 0.00e+00  (selected sets match)
      BRS.splsda :   105.750 μs (184 allocations: 461.03 KiB)
      R  microbenchmark median: 2.849 ms (in-R, 20 reps)

  n=150 p=500 keepX=20
    result: IDENTICAL ✓   max misalignment = 1.11e-16  (selected sets match)
      BRS.splsda :   536.041 μs (208 allocations: 2.55 MiB)
      R  microbenchmark median: 6.537 ms (in-R, 20 reps)

  n=600 p=1000 keepX=30
    result: IDENTICAL ✓   max misalignment = 2.22e-16  (selected sets match)
      BRS.splsda :   2.719 ms (204 allocations: 18.73 MiB)
      R  microbenchmark median: 40.303 ms (in-R, 3 reps)

  n=90 p=1500 keepX=15
    result: IDENTICAL ✓   max misalignment = 0.00e+00  (selected sets match)
      BRS.splsda :   999.041 μs (203 allocations: 4.69 MiB)
      R  microbenchmark median: 9.620 ms (in-R, 20 reps)

  n=300 p=2000 keepX=40
    result: IDENTICAL ✓   max misalignment = 2.22e-16  (selected sets match)
      BRS.splsda :   3.890 ms (210 allocations: 19.06 MiB)
      R  microbenchmark median: 40.565 ms (in-R, 3 reps)

  n=900 p=5000 keepX=50
    result: IDENTICAL ✓   max misalignment = 1.11e-16  (selected sets match)
      BRS.splsda :   35.723 ms (204 allocations: 139.23 MiB)
      R  microbenchmark median: 258.954 ms (in-R, 3 reps)
============================================================================================
  Done. Julia ran every size; R used adaptive reps (20→3) and skipped any case ≥ 20000000 cells.
  '@btime' lines show time + allocations + bytes; R lines are in-R microbenchmark medians.
  Julia and R times are separate-runtime medians, not a direct ratio.
============================================================================================
=#