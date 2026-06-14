# test/TestJiveRjiveFull.jl — comprehensive jive_rjive validation on SIMULATED data.
# Tests: (A) given-ranks similarity+speed vs r.jive, (B) ground-truth recovery,
#        (C) auto-ranks similarity+speed vs r.jive, (D) do given & auto agree.
using BigRiverSchneider, RCall, BenchmarkTools
using LinearAlgebra, Statistics, Random
Random.seed!(2024)

# ----------------------------------------------------------------
# Simulated data with KNOWN structure (supplement §6.2 generative model).
# Moderate size so the permutation path runs in reasonable time.
# Two datasets, shared joint scores S, individual scores S1/S2.
# ----------------------------------------------------------------
n = 80                       # samples (shared columns)
rT, r1T, r2T = 2, 3, 3       # TRUE ranks
p1, p2 = 60, 50              # variables per dataset
S  = randn(rT, n)
U1 = randn(p1, rT); U2 = randn(p2, rT)
S1 = randn(r1T, n); W1 = randn(p1, r1T)
S2 = randn(r2T, n); W2 = randn(p2, r2T)
X1 = U1*S + W1*S1 .+ 0.3 .* randn(p1, n)     # mild noise (realistic; perm test needs some)
X2 = U2*S + W2*S2 .+ 0.3 .* randn(p2, n)
nm = ["Dataset1","Dataset2"]
println("Simulated: X1 $(size(X1)), X2 $(size(X2)); true ranks joint=$rT, indiv=[$r1T,$r2T]\n")

# push to R once
@rput X1 X2

# helper: variance explained
ve(J,A,D) = (norm(J)^2/norm(D)^2, norm(A)^2/norm(D)^2, norm(D.-J.-A)^2/norm(D)^2)
# helper: joint subspace basis
jb(b...) = Matrix(qr(svd(vcat(b...)).Vt[1:2,:]').Q)[:,1:2]

# ================================================================
# PART A — GIVEN RANKS: similarity to r.jive
# ================================================================
println("="^66)
println("PART A — GIVEN RANKS (2, [3,3]): jive_rjive vs r.jive")
println("="^66)

resA = jive_rjive([X1,X2], rT, [r1T,r2T])
R"""
fitA <- jive(list(X1,X2), rankJ=2, rankA=c(3,3), method="given",
             scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE, showProgress=FALSE)
J1A<-fitA$joint[[1]]; J2A<-fitA$joint[[2]]
A1A<-fitA$individual[[1]]; A2A<-fitA$individual[[2]]
d1A<-fitA$data[[1]]; d2A<-fitA$data[[2]]
"""
@rget J1A J2A A1A A2A d1A d2A
JrA=[J1A,J2A]; ArA=[A1A,A2A]; DrA=[d1A,d2A]

println("\ninput match (scaled data, want 0):")
nelA=[p1*n, p2*n]; sumnA=sum(nelA)
XcA = [ let Xi=X.-mean(X,dims=2); Xi./(norm(Xi)*sqrt(sumnA)); end for X in (X1,X2) ]
for i in 1:2
    println("  $(nm[i]): ", round(norm(XcA[i].-DrA[i]),digits=10))
end
println("\nJ / A differences (want 0):")
for i in 1:2
    println("  $(nm[i]): ‖J diff‖=", round(norm(resA.J[i].-JrA[i]),digits=8),
            "  ‖A diff‖=", round(norm(resA.A[i].-ArA[i]),digits=8))
end
println("\nvariance explained:")
for i in 1:2
    println("  $(nm[i]): yours ", round.(ve(resA.J[i],resA.A[i],DrA[i]),digits=4),
            "  r.jive ", round.(ve(JrA[i],ArA[i],DrA[i]),digits=4))
end
println("\njoint subspace canon corr: ", round.(svd(jb(resA.J...)'*jb(JrA...)).S, digits=6))

# ================================================================
# PART B — GROUND TRUTH: does given-ranks recover planted structure?
# (use noiseless version for exact check)
# ================================================================
println("\n", "="^66)
println("PART B — GROUND TRUTH (noiseless, true ranks)")
println("="^66)
X1n = U1*S + W1*S1; X2n = U2*S + W2*S2          # noiseless
resB = jive_rjive([X1n,X2n], rT, [r1T,r2T]; scale=false)
Gc = [X1n .- mean(X1n,dims=2), X2n .- mean(X2n,dims=2)]
recon = sum(norm(Gc[i].-resB.J[i].-resB.A[i])^2 for i in 1:2)
ortho = norm(vcat(resB.J...)*resB.A[1]') + norm(vcat(resB.J...)*resB.A[2]')
Strue = S .- mean(S,dims=2)
Qt = Matrix(qr(Strue').Q)[:,1:rT]
Qm = Matrix(qr(svd(vcat(resB.J...)).Vt[1:rT,:]').Q)[:,1:rT]
println("  reconstruction ‖X−(J+A)‖² : ", round(recon, digits=10), "  (want ≈0)")
println("  orthogonality  ‖J·Aᵀ‖     : ", round(ortho, digits=10), "  (want ≈0)")
println("  joint subspace vs truth   : ", round.(svd(Qm'*Qt).S, digits=6), "  (want ≈1)")

# ================================================================
# PART C — AUTO RANKS: does permutation find the true ranks, and
#          do yours and r.jive agree on the estimated ranks?
# ================================================================
println("\n", "="^66)
println("PART C — AUTO RANKS via permutation")
println("="^66)
Random.seed!(2024)
resC = jive_rjive([X1,X2]; nperm=100)
println("\n  yours  estimated: joint ", resC.r, ", indiv ", resC.ri)
R"""
set.seed(2024)
fitC <- jive(list(X1,X2), method="perm", est=TRUE, orthIndiv=TRUE, showProgress=FALSE)
rJ_r <- fitC$rankJ; rA_r <- fitC$rankA
J1C<-fitC$joint[[1]]; J2C<-fitC$joint[[2]]
A1C<-fitC$individual[[1]]; A2C<-fitC$individual[[2]]
d1C<-fitC$data[[1]]; d2C<-fitC$data[[2]]
"""
@rget rJ_r rA_r J1C J2C A1C A2C d1C d2C
println("  r.jive estimated: joint ", Int(rJ_r), ", indiv ", Int.(rA_r))
println("  true ranks were : joint ", rT, ", indiv [", r1T, ",", r2T, "]")

# if both estimated the SAME ranks, compare their decompositions too
if resC.r == Int(rJ_r) && resC.ri == Int.(rA_r)
    println("\n  → same ranks estimated; comparing decompositions:")
    JrC=[J1C,J2C]; ArC=[A1C,A2C]; DrC=[d1C,d2C]
    for i in 1:2
        println("    $(nm[i]): ‖J diff‖=", round(norm(resC.J[i].-JrC[i]),digits=6),
                "  ‖A diff‖=", round(norm(resC.A[i].-ArC[i]),digits=6))
    end
    println("    joint subspace canon corr: ", round.(svd(jb(resC.J...)'*jb(JrC...)).S, digits=6))
else
    println("\n  → ranks differ (permutation is statistical; RNG differs across languages).")
    println("    comparing joint SUBSPACES instead (robust to rank diff):")
    println("    canon corr: ", round.(svd(jb(resC.J...)'*jb(J1C,J2C)).S, digits=6))
end

# ================================================================
# PART D — SPEED: both paths, vs r.jive
# ================================================================
println("\n", "="^66)
println("PART D — TIMING")
println("="^66)

println("\n[given ranks]")
print("  jive_rjive (Julia): ")
@btime jive_rjive([$X1,$X2], $rT, [$r1T,$r2T]);
R"""
library(microbenchmark)
mbG <- microbenchmark(
  jive(list(X1,X2), rankJ=2, rankA=c(3,3), method="given",
       scale=TRUE, center=TRUE, est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=20)
cat("  r.jive (R)        :  median", round(median(mbG$time)/1e6,2), "ms\n")
"""

println("\n[auto ranks — permutation, fewer reps since it's slow]")
print("  jive_rjive (Julia): ")
@btime jive_rjive([$X1,$X2]; nperm=100) samples=3 evals=1;
R"""
mbP <- microbenchmark(
  jive(list(X1,X2), method="perm", est=TRUE, orthIndiv=TRUE, showProgress=FALSE),
  times=3)
cat("  r.jive (R)        :  median", round(median(mbP$time)/1e6,2), "ms\n")
"""




#=
==================================================================
PART A — GIVEN RANKS (2, [3,3]): jive_rjive vs r.jive
==================================================================

input match (scaled data, want 0):
  Dataset1: 0.0
  Dataset2: 0.0

J / A differences (want 0):
  Dataset1: ‖J diff‖=0.0  ‖A diff‖=0.0
  Dataset2: ‖J diff‖=0.0  ‖A diff‖=0.0

variance explained:
  Dataset1: yours (0.536, 0.4235, 0.0405)  r.jive (0.536, 0.4235, 0.0405)
  Dataset2: yours (0.4409, 0.502, 0.0571)  r.jive (0.4409, 0.502, 0.0571)

joint subspace canon corr: [1.0, 1.0]

==================================================================
PART B — GROUND TRUTH (noiseless, true ranks)
==================================================================
  reconstruction ‖X−(J+A)‖² : 618.1879298603  (want ≈0)
  orthogonality  ‖J·Aᵀ‖     : 1.39549e-5  (want ≈0)
  joint subspace vs truth   : [0.999436, 0.997297]  (want ≈1)

==================================================================
PART C — AUTO RANKS via permutation
==================================================================
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]

  yours  estimated: joint 2, indiv [3, 3]
  r.jive estimated: joint 2, indiv [3, 3]
  true ranks were : joint 2, indiv [3,3]

  → same ranks estimated; comparing decompositions:
    Dataset1: ‖J diff‖=0.0  ‖A diff‖=0.0
    Dataset2: ‖J diff‖=0.0  ‖A diff‖=0.0
    joint subspace canon corr: [1.0, 1.0]

==================================================================
PART D — TIMING
==================================================================

[given ranks]
  jive_rjive (Julia):   39.517 ms (4571 allocations: 52.11 MiB)
  r.jive (R)        :  median 618.81 ms

[auto ranks — permutation, fewer reps since it's slow]
  jive_rjive (Julia): Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
Estimating ranks via permutation test...
Estimated joint rank: 2, individual ranks: [3, 3]
  714.210 ms (236660 allocations: 690.62 MiB)
  r.jive (R)        :  median 4439.36 ms
RObject{NilSxp}
NULL
=#