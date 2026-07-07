# Test/cca_test.jl — tests for cca (canonical correlation analysis, after Weenink
# 2003), cca_transform, and the internal solvers (_cca_svd_opt, _cca_cov_opt,
# _qnormalize!). Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl.
#
# ORIENTATION: cca takes variables in ROWS, observations in COLUMNS (X is dx×n) —
# the OPPOSITE of pca/spc. So means are over dims=2, and the canonical variates
# come out as (components × observations). Watch this throughout.
#
# Reference is MultivariateStats.CCA (pure Julia — no fixtures, no R). CCA is a
# deterministic eigenproblem, so results agree bit-identically up to per-column
# sign: the MVS check uses the tight tol_ord, with projections compared
# sign-invariantly (column-wise |·|) and correlations directly (they're sign-free).








# Largest per-column abs-difference between two projection matrices. Canonical
# directions are eigenvectors, so their per-column sign is arbitrary — comparing
# abs.() columns avoids a sign flip reading as a failure.
projdiff(A, B) = maximum(norm(abs.(@view A[:, j]) .- abs.(@view B[:, j])) for j in 1:size(A, 2))

@testset "output structure & invariants" begin
	# Basic contract: type, shapes, recorded means (over observations = dims=2), and
	# the property canonical correlations must satisfy — in [0,1] and descending.
	Random.seed!(1)
	dx, dy, n = 6, 5, 400
	X = randn(dx, n);
	Y = randn(dy, n)         # variables × observations
	p = min(dx, dy)                            # at most min(dx,dy) canonical pairs
	M = BigRiverEssence.cca(X, Y; method = :svd)

	@test M isa BigRiverEssence.ccaStructure
	@test length(M.xmean) == dx && length(M.ymean) == dy   # one mean per variable
	@test size(M.xproj) == (dx, p)             # dx variables × p canonical directions
	@test size(M.yproj) == (dy, p)
	@test length(M.corrs) == p
	@test M.xmean ≈ vec(mean(X, dims = 2))     # means over COLUMNS (observations)
	@test M.ymean ≈ vec(mean(Y, dims = 2))
	# Canonical correlations are genuine correlations (≤ 1) and come out largest-first.
	@test all(0 .<= M.corrs .<= 1 + tol_ord)
	@test issorted(M.corrs; rev = true)
end

@testset "outdim controls number of components" begin
	# outdim caps how many canonical pairs are returned (default min(dx,dy)).
	Random.seed!(2)
	X = randn(6, 300);
	Y = randn(5, 300)
	M = BigRiverEssence.cca(X, Y; method = :svd, outdim = 3)
	@test length(M.corrs) == 3
	@test size(M.xproj, 2) == 3 && size(M.yproj, 2) == 3
end

@testset "ground truth: recovers a planted shared latent" begin
	# Build X and Y to share ONE latent factor z (each is a random loading of z plus
	# small noise). CCA should find that shared dimension: the leading canonical
	# correlation ≈ 1, and the rest clearly smaller since nothing else is shared.
	Random.seed!(42)
	dx, dy, n = 5, 4, 300
	z = randn(1, n)                                       # the single shared latent
	X = randn(dx, 1) * z .+ 0.05 .* randn(dx, n)         # X = (loading)·z + noise
	Y = randn(dy, 1) * z .+ 0.05 .* randn(dy, n)         # Y shares the same z
	M = BigRiverEssence.cca(X, Y; method = :svd)
	@test M.corrs[1] > 0.99                       # leading correlation ≈ 1 (the shared z)
	@test M.corrs[2] < 0.5                        # nothing else is shared ⇒ rest small
	# And the leading canonical variates actually realize that correlation when we
	# project the data and correlate the two sides' first variate.
	Zx = BigRiverEssence.cca_transform(M, X, :x)
	Zy = BigRiverEssence.cca_transform(M, Y, :y)
	@test isapprox(abs(cor(Zx[1, :], Zy[1, :])), M.corrs[1]; atol = tol_ord)
end

@testset "cca_transform: canonical variates have the right correlations" begin
	# The meaning of the canonical correlations: projecting each side onto its
	# canonical directions yields variates whose j-th pair correlates at exactly
	# corrs[j]. This ties the reported numbers to an observable quantity.
	Random.seed!(3)
	dx, dy, n = 6, 5, 400
	X = randn(dx, n);
	Y = randn(dy, n)
	sh = randn(2, n);
	X[1:2, :] .+= sh;
	Y[1:2, :] .+= sh   # plant 2 shared dimensions
	M = BigRiverEssence.cca(X, Y; method = :svd)
	Zx = BigRiverEssence.cca_transform(M, X, :x)
	Zy = BigRiverEssence.cca_transform(M, Y, :y)
	@test size(Zx) == (length(M.corrs), n)        # variates: components × observations
	# Each canonical-variate pair correlates at the reported canonical correlation.
	for j in 1:length(M.corrs)
		@test isapprox(abs(cor(Zx[j, :], Zy[j, :])), M.corrs[j]; atol = tol_ord)
	end
	@test_throws ArgumentError BigRiverEssence.cca_transform(M, X, :z)   # component must be :x or :y
end

@testset ":svd and :cov agree (internal consistency)" begin
	# The two solvers reach the same canonical correlations by different routes —
	# SVD of the centered data vs the covariance generalized eigenproblem. They must
	# agree (compared sorted, since either could order ties differently).
	Random.seed!(7)
	dx, dy, n = 6, 5, 400
	X = randn(dx, n);
	Y = randn(dy, n)
	sh = randn(2, n);
	X[1:2, :] .+= sh;
	Y[1:2, :] .+= sh
	Ms = BigRiverEssence.cca(X, Y; method = :svd)
	Mc = BigRiverEssence.cca(X, Y; method = :cov)
	@test norm(sort(Ms.corrs) .- sort(Mc.corrs)) < tol_ord
end

@testset "argument validation" begin
	# The inconsistent inputs must throw: mismatched observation counts (a dimension
	# error), out-of-range outdim, and an unknown method.
	Random.seed!(0)
	X = randn(6, 100);
	Y = randn(5, 100)
	@test_throws DimensionMismatch BigRiverEssence.cca(X, randn(5, 80))       # X has 100 cols, Y has 80
	@test_throws ArgumentError BigRiverEssence.cca(X, Y; outdim = 0)          # outdim must be ≥ 1
	@test_throws ArgumentError BigRiverEssence.cca(X, Y; outdim = 6)          # outdim > min(dx,dy)=5
	@test_throws ArgumentError BigRiverEssence.cca(X, Y; method = :bogus)     # method must be :svd or :cov
end

# ----------------------------------------------------------------------------
# Internal helpers — tested directly so a regression in a solver surfaces here.
# ----------------------------------------------------------------------------

@testset "internal: _qnormalize! (C-normalize columns)" begin
	# CCA directions are normalized under a COVARIANCE metric, not the plain L2 norm:
	# each canonical variate should have unit variance, i.e. pⱼᵀ C pⱼ = 1. _qnormalize!
	# rescales each column to satisfy that. (This is why CCA needs its own normalizer
	# rather than a plain unit-norm.)
	Random.seed!(5)
	d, p = 6, 3
	Craw = randn(d, d);
	C = Craw * Craw' + I        # a symmetric positive-definite metric
	P = randn(d, p)
	BigRiverEssence._qnormalize!(P, C)
	for j in 1:p
		@test isapprox(dot(@view(P[:, j]), C * @view(P[:, j])), 1.0; atol = tol_ord)   # pⱼᵀ C pⱼ = 1
	end
end

@testset "internal: _cca_svd_opt (SVD solver)" begin
	# The stable SVD-based solver (Weenink's recommended route — no covariance formed).
	# On centered data it must return valid, sorted correlations and record nobs (the
	# SVD path threads n through, unlike the covariance path which stores −1).
	Random.seed!(11)
	dx, dy, n = 6, 5, 400
	X = randn(dx, n);
	Y = randn(dy, n)
	sh = randn(2, n);
	X[1:2, :] .+= sh;
	Y[1:2, :] .+= sh
	xm = vec(mean(X, dims = 2));
	ym = vec(mean(Y, dims = 2))
	M = BigRiverEssence._cca_svd_opt(copy(X) .- xm, copy(Y) .- ym, xm, ym, min(dx, dy))   # pass pre-centered data
	@test M isa BigRiverEssence.ccaStructure
	@test issorted(M.corrs; rev = true)
	@test all(0 .<= M.corrs .<= 1 + tol_ord)
	@test M.nobs == n                               # the SVD path records the sample size
end

@testset "internal: _cca_cov_opt (covariance solver, both dx≤dy and dx>dy)" begin
	# The covariance generalized-eigenproblem solver reduces through the SMALLER side
	# for efficiency, so it has two branches (dx≤dy and dx>dy). We exercise BOTH by
	# swapping the dimensions, since a bug could hide in just one branch.
	Random.seed!(12)
	for (dx, dy) in ((5, 8), (8, 5))                # (5,8) hits dx≤dy; (8,5) hits dx>dy
		n = 400
		X = randn(dx, n);
		Y = randn(dy, n)
		k = min(dx, dy)
		sh = randn(2, n);
		X[1:2, :] .+= sh;
		Y[1:2, :] .+= sh
		xm = vec(mean(X, dims = 2));
		ym = vec(mean(Y, dims = 2))
		Zx = X .- xm;
		Zy = Y .- ym
		Cxx = (Zx * Zx') ./ (n - 1)                 # build the covariances the solver expects
		Cyy = (Zy * Zy') ./ (n - 1)
		Cxy = (Zx * Zy') ./ (n - 1)
		M = BigRiverEssence._cca_cov_opt(Cxx, Cyy, Cxy, xm, ym, k)
		@test length(M.corrs) == k
		@test all(0 .<= M.corrs .<= 1 + tol_ord)
		@test issorted(M.corrs; rev = true)
		# The recovered X-directions must be C-normalized: PₓᵀCxxPₓ has unit diagonal
		# (each canonical variate has unit variance) — the covariance-metric property.
		D = M.xproj' * Cxx * M.xproj
		@test all(isapprox.(diag(D), 1.0; atol = tol_ord))
	end
end

# ----------------------------------------------------------------------------
# Cross-check against MultivariateStats.CCA (pure-Julia reference, both solvers).
# ----------------------------------------------------------------------------

@testset "matches MultivariateStats.CCA" begin
	# CCA is a deterministic eigenproblem, so against MVS we get machine-precision
	# agreement (the tight tol_ord), not the loose cross-language bar — there's no R
	# here. Run two sizes and BOTH our solvers against the same MVS reference.
	Random.seed!(7)
	for (dx, dy, n) in ((6, 5, 400), (50, 40, 2000))
		X = randn(dx, n);
		Y = randn(dy, n)
		nsh = min(dx, dy, 5)
		sh = randn(nsh, n);
		X[1:nsh, :] .+= sh;
		Y[1:nsh, :] .+= sh   # plant shared structure

		ref = MultivariateStats.fit(MultivariateStats.CCA, X, Y; method = :svd)
		rc  = MultivariateStats.correlations(ref)
		rPx = MultivariateStats.xprojection(ref);
		rPy = MultivariateStats.yprojection(ref)

		for meth in (:svd, :cov)
			M = BigRiverEssence.cca(X, Y; method = meth)
			# Correlations are sign-free and it's a pure-Julia eigenproblem, so they
			# match to machine precision (sorted, since tie ordering can differ).
			@test norm(sort(M.corrs) .- sort(rc)) < tol_ord
			# Projections match up to per-column sign (eigenvectors carry sign freedom).
			@test projdiff(M.xproj, rPx) < tol_ord
			@test projdiff(M.yproj, rPy) < tol_ord
		end
	end
end
