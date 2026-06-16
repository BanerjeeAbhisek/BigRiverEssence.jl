using RCall, BigRiverSchneider
using LinearAlgebra, Statistics, Random

R"""
library(mixOmics)
data(srbct)
X <- srbct$gene[1:60, 1:200]
Y <- srbct$class[1:60]
res <- splsda(X, Y, ncomp=2, keepX=c(15,15))
vx <- res$variates$X; lx <- res$loadings$X
vy <- res$variates$Y; ly <- res$loadings$Y
levs <- levels(srbct$class)
"""
@rget X Y vx lx vy ly levs
levs = string.(levs)

# fit with mixOmics' class ordering
mine = splsda(Float64.(X), string.(Y), 2, [15,15]; levels=levs)

# verify all five outputs (abs handles arbitrary per-component SVD signs)
for c in 1:2
    sel_match = Set(findall(!iszero, mine.loadings_X[:,c])) == Set(findall(!iszero, lx[:,c]))
    println("comp $c | X-load: ", round(abs(cor(mine.loadings_X[:,c], lx[:,c])), digits=6),
            "  X-var: ", round(abs(cor(mine.variates_X[:,c], vx[:,c])), digits=6),
            "  Y-load: ", round(abs(cor(mine.loadings_Y[:,c], ly[:,c])), digits=6),
            "  Y-var: ", round(abs(cor(mine.variates_Y[:,c], vy[:,c])), digits=6),
            "  sel: ", sel_match)
end



using BenchmarkTools

# --- Julia timing ---
println("="^55); println("BENCHMARK: splsda — mine (Julia) vs mixOmics (R)"); println("="^55)

print("mine (Julia): ")
@btime splsda(Float64.($X), string.($Y), 2, [15,15]; levels=$levs);

# --- mixOmics timing via microbenchmark ---
R"""
library(microbenchmark)
mb <- microbenchmark(
  splsda(X, Y, ncomp=2, keepX=c(15,15)),
  times=20)
cat("mixOmics (R): median", round(median(mb$time)/1e6, 3), "ms\n")
"""
#=
=======================================================
BENCHMARK: splsda — mine (Julia) vs mixOmics (R)
=======================================================
mine (Julia):   225.791 μs (915 allocations: 1.26 MiB)
┌ Warning: RCall.jl: Warning in microbenchmark(splsda(X, Y, ncomp = 2, keepX = c(15, 15)), times = 20) :
│   less accurate nanosecond times to avoid potential integer overflows
└ @ RCall ~/.julia/packages/RCall/fTLHT/src/io.jl:166
mixOmics (R): median 4.577 ms
RObject{NilSxp}
NULL
=#