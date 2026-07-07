# Test/plskern_test.jl — tests for plskern (Dayal & MacGregor 1997 kernel PLS) and
# its companions plskerncoef / plskernpredict / plskerntransform.
# Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl: tol_ord for exact
# linear-algebra identities, tol_julia for the cross-implementation Jchemo checks.
#
# NOTE: plskern does NOT modify its inputs — it centers/scales internal copies, so
# the caller's X and Y survive the fit untouched. A single response is passed as an
# n×1 matrix via reshape(y, :, 1); no defensive copy is needed since nothing is
# written back through it.








const HAS_JCHEMO = let
	try
		; @eval import Jchemo; true;
	catch
		; false;
	end
end
HAS_JCHEMO || @info "Jchemo not available; cross-implementation tests will be skipped."

@testset "output structure & invariants" begin
	# Basic contract: right type and factor-matrix shapes, the stored centering stats
	# match the data, and the weight vectors are unit-norm.
	Random.seed!(1)
	n, p, q, nlv = 80, 30, 4, 5
	X = randn(n, p);
	Y = randn(n, q)
	m = BigRiverEssence.plskern(X, Y; nlv = nlv)

	@test m isa BigRiverEssence.plskernStructure
	@test size(m.W) == (p, nlv)            # weights
	@test size(m.P) == (p, nlv)            # X loadings
	@test size(m.Q) == (q, nlv)            # Y loadings
	@test size(m.R) == (p, nlv)            # the projection (T = Xc·R)
	@test size(m.T) == (n, nlv)            # scores
	@test m.xmeans ≈ vec(mean(X, dims = 1))   # X is untouched by the fit, so compare directly
	@test m.ymeans ≈ vec(mean(Y, dims = 1))
	@test all(m.xscales .== 1.0)           # standardize=false default ⇒ scales are all 1
	@test all(m.yscales .== 1.0)
	for a in 1:nlv
		@test isapprox(norm(m.W[:, a]), 1.0; atol = tol_ord)   # each weight vector is unit-norm
	end
	@test all(isfinite, m.T)               # no NaN/Inf leaked into the scores
end

@testset "nlv is clamped to min(nlv, n, p)" begin
	# You can't extract more latent variables than the data supports. Asking for an
	# absurd nlv should silently clamp to min(n, p), not error or return garbage.
	Random.seed!(2)
	y1 = randn(20)
	m = BigRiverEssence.plskern(randn(20, 8), reshape(y1, :, 1); nlv = 50)    # ask for 50, p=8
	@test size(m.R, 2) == 8                            # clamped to p (the limiting dim)
	y2 = randn(6)
	m2 = BigRiverEssence.plskern(randn(6, 40), reshape(y2, :, 1); nlv = 30)   # ask for 30, n=6
	@test size(m2.R, 2) == 6                           # clamped to n this time
end

@testset "scores T are orthogonal (PLS property)" begin
	# A defining property of PLS: the latent scores are mutually orthogonal, so TᵀT
	# is diagonal. We check the off-diagonal entries vanish and the diagonal (the
	# score variances) is strictly positive.
	Random.seed!(3)
	X = randn(100, 25);
	y = randn(100)
	m = BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = 10)
	G = m.T' * m.T
	offdiag = maximum(abs(G[i, j]) for i in 1:10 for j in 1:10 if i != j)
	@test offdiag < tol_ord                            # off-diagonals ≈ 0 ⇒ orthogonal
	@test all(diag(G) .> 0)                            # each score direction has real variance
end

@testset "T = Xc·R (scores are linear in deflated X)" begin
	# The scores aren't a black box: they're exactly the centered/scaled data times
	# the projection matrix R. This identity lets plskerntransform reproduce scores
	# on new data, so it must hold exactly on the training data.
	Random.seed!(4)
	X = randn(60, 20);
	Y = randn(60, 3)
	m = BigRiverEssence.plskern(X, Y; nlv = 6)
	Xc = (X .- m.xmeans') ./ m.xscales'                # rebuild Xc from X (which the fit left intact)
	@test m.T ≈ Xc * m.R
end

@testset "algo1 and algo2 give identical results" begin
	# plskern offers two algebraically-equivalent kernel formulations. They're
	# different computation paths to the same model, so every output must agree —
	# and the regression coefficients B, which are uniquely determined, must match.
	Random.seed!(5)
	X = randn(120, 40);
	Y = randn(120, 2)
	m1 = BigRiverEssence.plskern(X, Y; nlv = 12, method = :algo1)
	m2 = BigRiverEssence.plskern(X, Y; nlv = 12, method = :algo2)      # same X, Y reused safely — neither fit mutates them
	@test isapprox(m1.R, m2.R; rtol = tol_ord)
	@test isapprox(m1.T, m2.T; rtol = tol_ord)
	@test isapprox(m1.Q, m2.Q; rtol = tol_ord)
	B1, i1 = BigRiverEssence.plskerncoef(m1);
	B2, i2 = BigRiverEssence.plskerncoef(m2)
	@test isapprox(B1, B2; rtol = tol_ord)             # B has no sign/rotation freedom ⇒ identical
	@test isapprox(i1, i2; rtol = tol_ord)
end

@testset "full-rank PLS equals OLS (theorem anchor)" begin
	# The anchoring theorem: when nlv = p and X is well-conditioned, the PLS latent
	# space spans X's full column space, so PLS regression collapses to ordinary least
	# squares. The prediction must match OLS to machine precision.
	Random.seed!(6)
	n, p = 100, 20
	X = randn(n, p);
	y = randn(n)
	m = BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = p)
	ŷ = vec(BigRiverEssence.plskernpredict(m, X))
	Xc = X .- mean(X, dims = 1)
	B_ols = Xc \ (y .- mean(y))                            # OLS on the same y the fit used (it stays intact)
	ŷ_ols = mean(y) .+ Xc * B_ols
	@test isapprox(ŷ, ŷ_ols; rtol = tol_ord)
	# Same theorem for a multi-response Y — full-rank PLS = multivariate OLS.
	Y = randn(n, 3)
	mY = plskern(X, Y; nlv = p)
	Yc = Y .- mean(Y, dims = 1)
	B_olsY = Xc \ Yc
	@test isapprox(BigRiverEssence.plskernpredict(mY, X), (mean(Y, dims = 1) .+ Xc * B_olsY); rtol = tol_ord)
end

@testset "plskerncoef: B and intercept shapes & reconstruction" begin
	# plskerncoef collapses the latent model into a plain linear predictor (B, intercept).
	# Two things to verify: the shapes, and that predicting via B reproduces
	# plskernpredict exactly — i.e. coef-then-apply ≡ the model's own predict.
	Random.seed!(7)
	n, p, q = 70, 15, 3
	X = randn(n, p);
	Y = randn(n, q)
	m = BigRiverEssence.plskern(X, Y; nlv = 8)
	B, intercept = BigRiverEssence.plskerncoef(m)
	@test size(B) == (p, q)                # one coefficient per (feature, response)
	@test size(intercept) == (1, q)        # one intercept per response
	@test BigRiverEssence.plskernpredict(m, X) ≈ intercept .+ X * B    # the two prediction routes agree
	# Nested property: asking coef for fewer components must equal a model actually
	# fit at that smaller nlv — truncation and refitting give the same B.
	B5, _ = BigRiverEssence.plskerncoef(m; nlv = 5)
	m5 = BigRiverEssence.plskern(X, Y; nlv = 5)
	B5b, _ = BigRiverEssence.plskerncoef(m5)
	@test isapprox(B5, B5b; rtol = tol_ord)
end

@testset "plskerntransform: scores on training data == m.T" begin
	# Transforming the original training data should reproduce the stored scores m.T
	# exactly (it's the same T = Xc·R computation). Also check the nlv keyword returns
	# just the first nlv score columns.
	Random.seed!(8)
	X = randn(50, 12);
	y = randn(50)
	m = BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = 6)
	@test isapprox(BigRiverEssence.plskerntransform(m, X), m.T; rtol = tol_ord)
	@test isapprox(BigRiverEssence.plskerntransform(m, X; nlv = 3), m.T[:, 1:3]; rtol = tol_ord)
end

@testset "standardize=true scales X and Y" begin
	# With standardize=true the model should record each column's standard deviation
	# as its scale; with standardize=false those scales stay at 1. Columns are put on
	# deliberately different scales so a missing standardization would be obvious.
	Random.seed!(9)
	X = randn(60, 10) .* (1:10)';
	Y = randn(60, 2)     # column j scaled by j
	m = BigRiverEssence.plskern(X, Y; nlv = 5, standardize = true)
	@test m.xscales ≈ vec(std(X, dims = 1))            # recorded scales = the real column SDs (X intact)
	@test m.yscales ≈ vec(std(Y, dims = 1))
	md = BigRiverEssence.plskern(X, Y; nlv = 5, standardize = false)
	@test all(md.xscales .== 1.0)                      # off ⇒ scales are all 1
end

@testset "Y must be a matrix (vector responses are rejected)" begin
	# plskern's signature takes Y::Matrix{Float64}, so a bare vector doesn't match any
	# method and must throw. The n×1 matrix form (reshape) is the supported way to fit
	# a single response.
	Random.seed!(10)
	X = randn(40, 8);
	y = randn(40)
	@test_throws MethodError BigRiverEssence.plskern(X, y; nlv = 4)              # vector Y ⇒ no matching method
	mm = BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = 4)                  # the correct single-response form
	@test size(mm.Q) == (1, 4)                                 # one response column
end

@testset "argument validation" begin
	# An unknown method symbol must be rejected with ArgumentError.
	Random.seed!(0)
	X = randn(30, 6);
	y = randn(30)
	@test_throws ArgumentError BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = 2, method = :bogus)
end

@testset "matches Jchemo.plskern (live, if available)" begin
	# Cross-implementation check against Jchemo (a live Julia PLS package, no R needed).
	# Coefficients and predictions have no sign freedom, so they must match directly;
	# the latent scores carry per-column sign ambiguity, so those compare via |dot|.
	if !HAS_JCHEMO
		@test_skip "Jchemo not installed"
	else
		Random.seed!(1234)
		n, p, nlv = 400, 50, 12
		X = randn(n, p);
		y = randn(n)

		m_mine    = BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = nlv, method = :algo1)
		B_mine, _ = BigRiverEssence.plskerncoef(m_mine)

		mod = Jchemo.plskern(; nlv = nlv)               # Jchemo scal=false ↔ our standardize=false
		Jchemo.fit!(mod, X, y)                          # X, y are the same arrays our fit used —
		B_jc = Jchemo.coef(mod).B                       # safe, since plskern doesn't modify them

		@test maximum(abs.(B_mine .- B_jc)) < tol_julia # B is sign-unambiguous ⇒ direct compare
		ŷ_mine = vec(BigRiverEssence.plskernpredict(m_mine, X))
		ŷ_jc   = vec(Jchemo.predict(mod, X).pred)
		@test maximum(abs.(ŷ_mine .- ŷ_jc)) < tol_julia # predictions also have no sign freedom
		# Scores match only up to per-column sign — latent factors are sign-ambiguous.
		T_jc = Jchemo.transf(mod, X)
		for a in 1:nlv
			@test abs(dot(m_mine.T[:, a] ./ norm(m_mine.T[:, a]),
				T_jc[:, a] ./ norm(T_jc[:, a]))) > 1 - tol_julia
		end
		# The algo2 path must agree with Jchemo too (both algorithms, same answer).
		m2 = BigRiverEssence.plskern(X, reshape(y, :, 1); nlv = nlv, method = :algo2)
		B2, _ = BigRiverEssence.plskerncoef(m2)
		@test maximum(abs.(B2 .- B_jc)) < tol_julia
		# And the multi-response case cross-checks against Jchemo as well.
		Y = randn(n, 3)
		mYmine = BigRiverEssence.plskern(X, Y; nlv = nlv);
		BYmine, _ = BigRiverEssence.plskerncoef(mYmine)
		modY = Jchemo.plskern(; nlv = nlv);
		Jchemo.fit!(modY, X, Y)
		@test maximum(abs.(BYmine .- Jchemo.coef(modY).B)) < tol_julia
	end
end
