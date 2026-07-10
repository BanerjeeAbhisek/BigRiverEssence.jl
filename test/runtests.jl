# Library
using Aqua
using MultivariateStats: MultivariateStats
using BigRiverEssence
using Test
using LinearAlgebra, Statistics, Random, DelimitedFiles

# ── shared tolerances ───────────────────────────────────────────────────────
# Each is a DIFFERENCE magnitude (smaller = stricter). Use directly as a bound
# on norms / rtol / atol, and as `1 - tol` for correlation/dot agreement.
#
#   tol_ord   — tight. Things that should agree to ~machine precision:
#               deterministic internal math, :svd-vs-:cov, algo1-vs-algo2,
#               transform round-trips, our-vs-MultivariateStats (pure-Julia, exact).
#   tol_julia — medium. Comparing against a Julia reference whose algorithm or
#               convergence differs slightly (iterative / different code path).
#   tol_r     — loose. Cross-language vs R (PMA / r.jive / mixOmics): different
#               RNG, BLAS, binary-search loops ⇒ agree in substance, not bits.
const tol_ord   = 1e-8
const tol_julia = 1e-5
const tol_r     = 1e-3
# ─────────────────────────────────────────────────────────────────────────────

@testset "BigRiverEssence.jl" begin

	@testset "Aqua" begin
        Aqua.test_all(BigRiverEssence)
    end
	@testset "Utility" begin
		include("utils.jl")
	end
	@testset "Principal Component Analysis (pca)" begin
		include("pca_test.jl")
	end
	@testset "Penalized Matrix Decomposition (pmd)" begin
		include("pmd_test.jl")
	end
	@testset "Sparse Principal Component Analysis (spc)" begin
		include("spc_test.jl")
	end
	@testset "Kernel Partial Least Squares (plskern)" begin
		include("plskern_test.jl")
	end
	@testset "Joint and Individual Variation Explained (jive)" begin
		include("jive_test.jl")
	end
	@testset "Partial Least Squares Discriminant Analysis (plsda)" begin
		include("plsda_test.jl")
	end
	@testset "Sparse Partial Least Squares Discriminant Analysis (splsda)" begin
		include("splsda_test.jl")
	end
	@testset "Canonical Correlation Analysis (cca)" begin
		include("cca_test.jl")
	end
	@testset "Sparse Canonical Correlation Analysis (scca)" begin
		include("scca_test.jl")
	end
end
