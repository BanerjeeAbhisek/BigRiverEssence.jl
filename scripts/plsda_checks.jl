using BigRiverSchneider, RCall, BenchmarkTools, LinearAlgebra, Statistics

# ===============================================================
# Setup: SRBCT data + fit both implementations once (for correctness)
# ===============================================================
R"""
suppressMessages(library(mixOmics))
suppressMessages(library(microbenchmark))
data(srbct)
Xg  <- srbct$gene
Ycl <- srbct$class
lv  <- levels(srbct$class)
ncomp <- 3
res <- plsda(Xg, Ycl, ncomp = ncomp, scale = TRUE)
var_mix  <- res$variates$X
load_mix <- res$loadings$X
"""
X        = rcopy(R"Xg")
y        = rcopy(R"as.character(Ycl)")
levels   = rcopy(R"lv")
ncomp    = Int(rcopy(R"ncomp"))
var_mix  = rcopy(R"var_mix")
load_mix = rcopy(R"load_mix")

m = BigRiverSchneider.plsda(X, y, ncomp; scale = true, levels = levels)

# ===============================================================
# PART 1 — CORRECTNESS: are the two implementations the same?
# ===============================================================
println("="^60)
println("PART 1 — CORRECTNESS  (BigRiverSchneider.plsda vs mixOmics)")
println("="^60)

tol = 1e-6
all_match = true
for c in 1:ncomp
    # each PLS component is defined only up to sign; align before comparing
    sL = sign(dot(m.loadings_X[:, c], load_mix[:, c])); sL = sL == 0 ? 1.0 : sL
    sV = sign(dot(m.variates_X[:, c], var_mix[:, c]));  sV = sV == 0 ? 1.0 : sV

    dL = maximum(abs.(sL .* m.loadings_X[:, c] .- load_mix[:, c]))
    dV = maximum(abs.(sV .* m.variates_X[:, c] .- var_mix[:, c]))

    okL = dL < tol
    okV = dV < tol
    (okL && okV) || (all_match = false)

    println("comp $c:  Δloadings = ", rpad(round(dL, sigdigits=3), 10),
            " ", okL ? "✓" : "✗",
            "   Δvariates = ", rpad(round(dV, sigdigits=3), 10),
            " ", okV ? "✓" : "✗")
end

# also check the dummy encoding / class order agrees
println("\nclasses (ours):    ", m.classes)
println("levels (mixOmics): ", levels)

println("\nOVERALL: ", all_match ? "✓ MATCH (within $tol)" : "✗ DIFFER — investigate")

# ===============================================================
# PART 2 — SPEED: Julia @btime vs R microbenchmark
# ===============================================================
println("\n" * "="^60)
println("PART 2 — SPEED")
println("="^60)

println("\nJulia  BigRiverSchneider.plsda  (@btime, reports minimum):")
@btime BigRiverSchneider.plsda($X, $y, $ncomp; scale = true, levels = $levels);

println("\nR  mixOmics::plsda  (microbenchmark, 100 runs):")
R"""
mb <- microbenchmark(
    plsda(Xg, Ycl, ncomp = ncomp, scale = TRUE),
    times = 100L
)
print(mb)
"""

println("\n" * "="^60)
println("NOTE: @btime reports the MINIMUM; compare against microbenchmark's")
println("'min' column. Watch the UNIT in the microbenchmark table header")
println("(µs vs ms) before concluding which is faster.")
println("="^60)






#=

============================================================
PART 1 — CORRECTNESS  (BigRiverSchneider.plsda vs mixOmics)
============================================================
comp 1:  Δloadings = 4.34e-16   ✓   Δvariates = 1.19e-13   ✓
comp 2:  Δloadings = 3.4e-16    ✓   Δvariates = 1.23e-13   ✓
comp 3:  Δloadings = 9.71e-17   ✓   Δvariates = 3.15e-14   ✓

classes (ours):    ["EWS", "BL", "NB", "RMS"]
levels (mixOmics): ["EWS", "BL", "NB", "RMS"]

OVERALL: ✓ MATCH (within 1.0e-6)

============================================================
PART 2 — SPEED
============================================================

Julia  BigRiverSchneider.plsda  (@btime, reports minimum):
  1.111 ms (191 allocations: 5.41 MiB)

R  mixOmics::plsda  (microbenchmark, 100 runs):
┌ Warning: RCall.jl: Warning in microbenchmark(plsda(Xg, Ycl, ncomp = ncomp, scale = TRUE), times = 100L) :
│   less accurate nanosecond times to avoid potential integer overflows
└ @ RCall ~/.julia/packages/RCall/fTLHT/src/io.jl:166
Unit: milliseconds
                                        expr      min       lq     mean
 plsda(Xg, Ycl, ncomp = ncomp, scale = TRUE) 9.078876 10.15625 13.46446
   median       uq      max neval
 11.83467 12.74698 98.13625   100

============================================================
NOTE: @btime reports the MINIMUM; compare against microbenchmark's
'min' column. Watch the UNIT in the microbenchmark table header
(µs vs ms) before concluding which is faster.
============================================================
=#





