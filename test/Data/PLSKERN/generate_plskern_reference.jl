# generate_plskern_reference.jl — generate Jchemo reference fixtures for plskern tests.
#
# Fits Jchemo's plskern on simulated data and writes the inputs and Jchemo's outputs
# as CSVs. plskern_test.jl loads these to compare the BigRiverEssence plskern against
# Jchemo without importing Jchemo at test time.
#
# Run from a REPL (or environment) that has Jchemo available:
#     include("test/Data/PLSKERN/generate_plskern_reference.jl")

using Jchemo, Random, DelimitedFiles

const OUTDIR = @__DIR__            # write CSVs next to this script (test/Data/PLSKERN)

# ---- parameters (single source of truth; meta.csv records them) -------------------
const seed = 1276523
const N    = 400
const P    = 50
const NLV  = 12
const Q    = 3                     # multi-response width

Random.seed!(seed)

# ---- simulate inputs --------------------------------------------------------------
X = randn(N, P)
y = randn(N)                       # single response (length-N vector)
Y = randn(N, Q)                    # multi response (N×Q)

# ---- fit Jchemo: single response --------------------------------------------------
mod = Jchemo.plskern(; nlv = NLV)  # scal=false ↔ standardize=false
Jchemo.fit!(mod, X, y)
B_jc      = Jchemo.coef(mod).B                  # (P, 1)
pred_jc   = Jchemo.predict(mod, X).pred         # (N, 1)
transf_jc = Jchemo.transf(mod, X)               # (N, NLV)

# ---- fit Jchemo: multi response ---------------------------------------------------
modY = Jchemo.plskern(; nlv = NLV)
Jchemo.fit!(modY, X, Y)
BY_jc    = Jchemo.coef(modY).B                  # (P, Q)
predY_jc = Jchemo.predict(modY, X).pred         # (N, Q)

# ---- write everything -------------------------------------------------------------
writedlm(joinpath(OUTDIR, "X.csv"),          X,         ',')
writedlm(joinpath(OUTDIR, "y.csv"),          y,         ',')
writedlm(joinpath(OUTDIR, "Y_multi.csv"),    Y,         ',')
writedlm(joinpath(OUTDIR, "B.csv"),          B_jc,      ',')
writedlm(joinpath(OUTDIR, "pred.csv"),       pred_jc,   ',')
writedlm(joinpath(OUTDIR, "transf.csv"),     transf_jc, ',')
writedlm(joinpath(OUTDIR, "B_multi.csv"),    BY_jc,     ',')
writedlm(joinpath(OUTDIR, "pred_multi.csv"), predY_jc,  ',')

# meta.csv: parameters the fixtures were generated with (read back by the test)
writedlm(joinpath(OUTDIR, "meta.csv"),
    ["seed" seed; "n" N; "p" P; "nlv" NLV; "q" Q], ',')

println("Wrote plskern fixtures to ", OUTDIR)