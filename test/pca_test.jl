# Tests for pca and its helpers (pca_transform, pca_invtransform).
# Tolerances (tol_ord / tol_r) come from runtests.jl

# Largest per-column difference between two loading matrices, comparing up to sign.
# SVD/eigen directions are only defined up to a sign flip, so we compare abs.()
# columns — otherwise two correct answers that differ only in sign would look wrong.
loaddiff(A, B) = maximum(norm(abs.(A[:, j]) .- abs.(B[:, j])) for j in 1:size(A, 2))

@testset "output structure & invariants" begin
	# The basic contract: right type, right shapes, and the mathematical properties
	# a PCA result must always satisfy regardless of the data.
	Random.seed!(1)
	n, p, k = 200, 30, 5
	X = randn(n, p)
	m = pca(X; k = k, method = :svd)

	@test m isa pcaStructure
	@test length(m.mean) == p          # one mean per feature
	@test length(m.scale) == p          # one scale per feature
	@test size(m.loadings) == (p, k)     # p features × k components
	@test length(m.variances) == k
	@test length(m.propOFvar) == k

	# Each principal direction is a unit vector — loadings are normalized.
	for j in 1:k
		@test isapprox(norm(m.loadings[:, j]), 1.0; atol = tol_ord)
	end
	# Variances can't be negative (they're squared singular values), and the
	# components come out largest-first.
	@test all(m.variances .>= -tol_ord)     # allow a hair below 0 for roundoff
	@test issorted(m.variances; rev = true)
	# Each proportion is a fraction of total variance, so it lives in [0,1].
	@test all(0 .<= m.propOFvar .<= 1 + tol_ord)

	# The stored mean should be the actual column means, and with no standardization
	# the scales should all be exactly 1 (we only centered, didn't divide).
	@test isapprox(m.mean, vec(mean(X, dims = 1)); atol = tol_ord)
	@test all(m.scale .== 1.0)
	# With standardize=true, the scales should be the column standard deviations.
	ms = pca(X; k = k, standardize = true)
	@test isapprox(ms.scale, vec(std(X, dims = 1)); atol = tol_ord)
end

@testset "loadings are orthonormal" begin
	# Beyond being unit-norm individually, the principal directions must be mutually
	# orthogonal — together they form an orthonormal basis, so Vᵀ V is the identity.
	# We check both decomposition paths, since each should produce an orthonormal V.
	Random.seed!(21)
	n, p, k = 300, 15, 6
	X = randn(n, p)
	for method in (:svd, :cov)
		m = pca(X; k = k, method = method)
		@test isapprox(m.loadings' * m.loadings, I(k); atol = tol_ord)
	end
end

@testset "ground-truth: recovers a planted direction" begin
	# If the data really lies along one known direction w (plus tiny noise), PCA had
	# better find it: the first component should point along ±w and soak up almost
	# all the variance. This is the "does it actually do PCA" sanity check, with a
	# known answer rather than a reference implementation.
	Random.seed!(7)
	n, p = 500, 10
	w = normalize(randn(p))               # the true direction the data lies along
	scores = randn(n) .* 5.0              # big spread along w
	X = scores * w' .+ 0.01 .* randn(n, p)   # rank-1 signal + small isotropic noise

	for method in (:svd, :cov)
		m = pca(X; k = 2, method = method)
		# PC1 aligns with w (|dot| because the sign of a loading is arbitrary).
		@test abs(dot(m.loadings[:, 1], w)) > 1 - tol_r
		# and PC1 explains essentially all the variance, since the signal is rank-1.
		@test m.propOFvar[1] > 1 - tol_r
	end
end

@testset "ground-truth: variances equal the data spectrum" begin
	# The component variances aren't arbitrary: for centered data they must equal
	# the squared singular values divided by (n-1). This pins down the actual
	# *values*, not just that they're sorted — and needs no external reference,
	# just the definition of PCA variance.
	Random.seed!(31)
	n, p, k = 400, 12, 12
	X = randn(n, p)
	Xc = X .- mean(X, dims = 1)
	truevars = (svdvals(Xc) .^ 2) ./ (n - 1)   # the variances PCA should report
	for method in (:svd, :cov)
		m = pca(X; k = k, method = method)
		@test isapprox(m.variances, truevars[1:k]; rtol = tol_ord)
	end
end

@testset ":svd and :cov agree" begin
	# The two methods are different routes to the same answer — SVD of the data vs
	# eigendecomposition of the covariance. They must land in the same place. This
	# also guards the :cov path's XᵀX − nμμᵀ scatter trick: if that introduced any
	# numerical drift, the loadings here would stop matching :svd.
	Random.seed!(3)
	n, p, k = 300, 20, 8
	X = randn(n, p)
	msvd = pca(X; k = k, method = :svd)
	mcov = pca(X; k = k, method = :cov)
	@test isapprox(msvd.variances, mcov.variances; rtol = tol_ord)
	@test isapprox(msvd.propOFvar, mcov.propOFvar; rtol = tol_ord)
	@test loaddiff(msvd.loadings, mcov.loadings) < tol_ord   # same directions (up to sign)
end

@testset "proportions sum correctly" begin
	# When we keep every component, no variance is left out, so the proportions of
	# variance explained must add up to exactly 1.
	Random.seed!(17)
	n, p = 200, 10
	X = randn(n, p)
	m = pca(X; k = p, method = :svd)
	@test isapprox(sum(m.propOFvar), 1.0; atol = tol_ord)
end

@testset "sign consistency" begin
	# Loading signs are arbitrary, so we canonicalize them: SignConsistency_opt!
	# flips each column so its largest-magnitude entry is positive. This makes
	# results reproducible run-to-run and comparable across implementations.
	Random.seed!(5)
	X = randn(100, 8)
	for method in (:svd, :cov)
		m = pca(X; k = 4, method = method)
		for j in 1:4
			c = m.loadings[:, j]
			@test c[argmax(abs.(c))] > 0       # the pivot (biggest) entry is positive
		end
	end
end

@testset "pca :svd wide-data branch (p > n)" begin
	Random.seed!(12311)
	# Wide data: more features than observations forces the p > n path inside :svd.
	n, p = 20, 50
	X = randn(n, p)
	k = 10

	# Force :svd so we exercise the SVD branch (not :cov); p > n routes us into the
	# transpose-SVD sub-branch specifically.
	m_svd = pca(X; k = k, method = :svd)
	# :cov on the same wide data is an independent route to the same PCA.
	m_cov = pca(X; k = k, method = :cov)

	# variances: same eigenvalues computed two ways, same machine ⇒ exact-level
	@test isapprox(m_svd.variances, m_cov.variances; rtol = tol_ord)
	# loadings agree up to per-column sign (Julia-vs-Julia cross-method)
	@test isapprox(abs.(m_svd.loadings), abs.(m_cov.loadings); rtol = tol_julia)

	# structural properties
	@test size(m_svd.loadings) == (p, k)
	@test length(m_svd.variances) == k
	@test isapprox(m_svd.loadings' * m_svd.loadings, I(k); atol = tol_ord)   # orthonormal columns
	@test all(diff(m_svd.variances) .<= tol_ord)                            # descending
	@test all(m_svd.variances .> 0)                                         # positive
	@test all(m_svd.propOFvar .> 0)
	@test sum(m_svd.propOFvar) <= 1 + tol_ord                               # ≤ 1 for truncated PCA
end

@testset "pca :svd wide-data branch with standardize" begin
	Random.seed!(12654)
	# Same wide branch, now with standardize=true to cover that sub-path.
	n, p = 15, 40
	X = randn(n, p)
	k = 8

	m_svd = pca(X; k = k, method = :svd, standardize = true)
	m_cov = pca(X; k = k, method = :cov, standardize = true)

	@test isapprox(m_svd.variances, m_cov.variances; rtol = tol_julia)
	@test isapprox(abs.(m_svd.loadings), abs.(m_cov.loadings); rtol = tol_julia)
	@test size(m_svd.loadings) == (p, k)
	@test all(m_svd.variances .> 0)
end

@testset "transform / inverse round-trip" begin
	# Projecting data into PC space and back should be lossless when we keep all
	# components — no information is discarded, so the reconstruction returns X.
	Random.seed!(9)
	n, p = 150, 12
	X = randn(n, p)
	m = pca(X; k = p)
	scores = pca_transform(m, X)
	@test size(scores) == (n, p)
	# Full-rank reconstruction recovers the original data exactly.
	Xrec = pca_invtransform(m, scores)
	@test isapprox(Xrec, X; atol = tol_ord)
	# And the scores are exactly the centered data projected onto the loadings —
	# the definition of the transform, checked directly.
	Xc = X .- mean(X, dims = 1)
	@test isapprox(scores, Xc * m.loadings; atol = tol_ord)
end

@testset "standardize round-trip" begin
	# Same round-trip, but with standardize=true and columns deliberately put on
	# wildly different scales. The inverse transform has to undo both the scaling
	# and the centering to land back on X — this catches a missing scale step.
	Random.seed!(13)
	X      = randn(120, 10) .* (1:10)' .+ 5           # column j scaled by j, then shifted
	m      = pca(X; k = 10, standardize = true)
	scores = pca_transform(m, X)
	Xrec   = pca_invtransform(m, scores)
	@test isapprox(Xrec, X; atol = tol_ord)
end

@testset "determinism" begin
	# pca is a pure deterministic computation — no RNG inside — so fitting the same
	# data twice must give bit-identical results (== , not isapprox).
	Random.seed!(99)
	X = randn(180, 14)
	a = pca(X; k = 6, method = :svd)
	b = pca(X; k = 6, method = :svd)
	@test a.loadings == b.loadings
	@test a.variances == b.variances
end

@testset "argument validation" begin
	# Bad arguments should be rejected loudly with ArgumentError, not silently
	# produce garbage or hit some downstream error.
	Random.seed!(0)
	X = randn(50, 8)
	@test_throws ArgumentError pca(X; k = 0)                     # k must be ≥ 1
	@test_throws ArgumentError pca(X; k = 9)                     # k > min(n,p)=8
	@test_throws ArgumentError pca(X; method = :IWontMention)  # method not :auto/:cov/:svd
end
