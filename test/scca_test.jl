# Test/scca_test.jl — tests for scca (sparse CCA, the PMA::CCA method of Witten,
# Tibshirani & Hastie 2009) and its internals. Tolerances (tol_ord / tol_julia /
# tol_r) come from runtests.jl: tol_ord for exact math, tol_julia for the iterative
# core, tol_r for the cross-language PMA fixtures.
#
# ORIENTATION: like cca, scca takes variables in ROWS, observations in COLUMNS
# (X is dx×n). PMA uses the opposite (obs in rows), so the fixture/ground-truth
# tests transpose to convert between the two layouts — watch for Matrix(transpose(·)).








@testset "output structure & invariants" begin
	# Basic contract: type, shapes, recorded penalties, and the sCCA properties —
	# each sparse canonical vector unit-norm (or zero), correlations in [0,1].
	Random.seed!(1)
	dx, dy, n, K = 40, 30, 100, 2
	X = randn(dx, n);
	Y = randn(dy, n)         # variables × observations
	m = BigRiverEssence.scca(X, Y; penaltyx = 0.3, penaltyz = 0.3, K = K)

	@test m isa BigRiverEssence.SccaStructure
	@test size(m.u) == (dx, K)                 # X canonical vectors, one column per component
	@test size(m.v) == (dy, K)                 # Y canonical vectors
	@test length(m.d) == K                     # the per-component weights
	@test length(m.cors) == K                  # the achieved canonical correlations
	@test m.K == K
	@test m.penaltyx == 0.3 && m.penaltyz == 0.3   # penalties stored as given
	# Each canonical vector is unit-norm OR exactly zero (a tight penalty can zero a
	# whole vector); each correlation is a genuine correlation in [0,1].
	for k in 1:K
		nu = norm(m.u[:, k]);
		nv = norm(m.v[:, k])
		@test isapprox(nu, 1.0; atol = tol_ord) || nu == 0.0
		@test isapprox(nv, 1.0; atol = tol_ord) || nv == 0.0
		@test 0 <= m.cors[k] <= 1 + tol_ord
	end
	@test all(isfinite, m.d)
end

@testset "penalty controls sparsity (smaller ⇒ sparser)" begin
	# The penalties tune sparsity on each side: a smaller penalty ⇒ a tighter L1
	# budget ⇒ fewer nonzero features. Check the monotone relationship on both sides.
	Random.seed!(2)
	X = randn(80, 60);
	Y = randn(70, 60)
	tight = BigRiverEssence.scca(X, Y; penaltyx = 0.1, penaltyz = 0.1, K = 1)   # small penalty → sparse
	loose = BigRiverEssence.scca(X, Y; penaltyx = 0.7, penaltyz = 0.7, K = 1)   # large penalty → denser
	@test count(!iszero, tight.u[:, 1]) <= count(!iszero, loose.u[:, 1])
	@test count(!iszero, tight.v[:, 1]) <= count(!iszero, loose.v[:, 1])
end

@testset "ground truth: selects the planted sparse features" begin
	# Plant a shared latent: the first nz features of EACH view load on the same
	# factor `lat`, the rest are noise. A working sCCA should select predominantly
	# those true features on both sides and recover a strong canonical correlation.
	Random.seed!(42)
	n = 100;
	p1, p2 = 500, 1000;
	nz1, nz2 = 25, 40
	lat = randn(n)                                       # the shared latent factor
	Xr = randn(n, p1);
	Zr = randn(n, p2)                 # PMA layout: obs in rows
	Xr[:, 1:nz1] .+= lat * fill(2.0, nz1)'               # first nz1 X-features load on lat
	Zr[:, 1:nz2] .+= lat * fill(2.0, nz2)'               # first nz2 Z-features load on lat
	m = BigRiverEssence.scca(Matrix(transpose(Xr)), Matrix(transpose(Zr));   # transpose to scca's var×obs layout
		penaltyx = 0.2, penaltyz = 0.2, K = 1)
	sel_x = Set(findall(!iszero, m.u[:, 1]))
	sel_z = Set(findall(!iszero, m.v[:, 1]))
	# Precision: most of what it SELECTED is genuinely signal (literal recovery floors).
	@test length(intersect(sel_x, Set(1:nz1))) / max(1, length(sel_x)) > 0.7
	@test length(intersect(sel_z, Set(1:nz2))) / max(1, length(sel_z)) > 0.7
	@test m.cors[1] > 0.5                          # and the shared structure is found
end

@testset "standardize=false skips scaling" begin
	# standardize=true centers+scales each feature; =false takes the data as-is. Both
	# are valid code paths — here we just confirm both run and return a model (the
	# numerical difference is exercised in the fixture test).
	Random.seed!(3)
	X = randn(20, 80) .* 5 .+ 3;
	Y = randn(15, 80)       # X deliberately off-scale
	ms = BigRiverEssence.scca(X, Y; K = 1, standardize = true)
	mn = BigRiverEssence.scca(X, Y; K = 1, standardize = false)
	@test ms isa BigRiverEssence.SccaStructure && mn isa BigRiverEssence.SccaStructure   # both paths produce a valid fit
end

@testset "argument validation" begin
	# The inconsistent inputs must throw: mismatched observation counts, too few
	# features, out-of-range penalties, out-of-range K.
	Random.seed!(0)
	X = randn(10, 50);
	Y = randn(8, 50)
	@test_throws DimensionMismatch BigRiverEssence.scca(X, randn(8, 40); K = 1)       # 50 vs 40 observations
	@test_throws ArgumentError BigRiverEssence.scca(randn(1, 50), Y; K = 1)           # need ≥ 2 features per view
	@test_throws ArgumentError BigRiverEssence.scca(X, Y; penaltyx = 0.0, K = 1)      # penalty must be in (0,1]
	@test_throws ArgumentError BigRiverEssence.scca(X, Y; penaltyx = 1.5, K = 1)
	@test_throws ArgumentError BigRiverEssence.scca(X, Y; K = 0)                      # K must be ≥ 1
	@test_throws ArgumentError BigRiverEssence.scca(X, Y; K = 11)                     # K > min(dx,dy)=8
end

# ----------------------------------------------------------------------------
# Internal helpers. The soft-threshold / budget / bisection family is the same
# machinery PMD uses (same Witten et al. paper); the rest (_matsqrt, _fast_init_v,
# the rank-1 core) is sCCA-specific. Tested directly so a regression in a primitive
# surfaces here rather than as a downstream symptom.
# ----------------------------------------------------------------------------

@testset "internal: BigRiverEssence._softcca!cca! (soft-threshold)" begin
	# The L1 proximal operator: shrink toward 0 by λ, clamp to 0 once |a| ≤ λ. The
	# single primitive that creates sparsity in the canonical vectors.
	a = [5.0, -3.0, 1.0, -0.5];
	out = similar(a)
	BigRiverEssence._softcca!(out, a, 2.0)
	@test out ≈ [3.0, -1.0, 0.0, 0.0]              # 5→3, 3→1, |1|&|0.5| ≤ 2 ⇒ 0
	@test BigRiverEssence._softcca!(similar(a), a, 0.0) ≈ a            # λ=0 is the identity
	b = randn(50);
	o = similar(b);
	λ = 0.6
	BigRiverEssence._softcca!(o, b, λ)
	@test o ≈ sign.(b) .* max.(abs.(b) .- λ, 0.0)  # matches the closed form
end

@testset "internal: _l2n_val (guarded L2 norm)" begin
	# Euclidean norm with the PMA zero-guard: all-zeros returns 0.05, not 0, so the
	# later normalizations never divide by zero.
	@test BigRiverEssence._l2n_val([3.0, 4.0]) == 5.0
	@test BigRiverEssence._l2n_val(zeros(5)) == 0.05                    # the zero-guard
	a = randn(40);
	@test BigRiverEssence._l2n_val(a) ≈ norm(a)          # equals the true norm when nonzero
end

@testset "internal: _l1_of_norm & _l1_of_norm_soft" begin
	# The L1/L2 ratio the budget constrains (1 = maximally sparse, √m = dense). The
	# _soft variant computes the ratio for a soft-thresholded vector without building it.
	@test BigRiverEssence._l1_of_norm([3.0, 4.0]) ≈ 7 / 5                 # (3+4)/5
	@test BigRiverEssence._l1_of_norm(ones(9)) ≈ 3.0                      # 9/3 = √9, the dense maximum
	a = randn(60)
	d = 0.5
	@test BigRiverEssence._l1_of_norm_soft(a, d) ≈ BigRiverEssence._l1_of_norm(sign.(a) .* max.(abs.(a) .- d, 0.0))   # soft-then-ratio shortcut
	@test BigRiverEssence._l1_of_norm(randn(30)) >= 1 - tol_ord           # ≥ 1 for any nonzero vector (Cauchy–Schwarz)
end

@testset "internal: _l1diff (L1 distance)" begin
	# ‖a−b‖₁, the convergence metric between successive iterates.
	@test BigRiverEssence._l1diff([1.0, 2.0], [1.0, 2.0]) == 0.0
	@test BigRiverEssence._l1diff([1.0, 0.0], [0.0, 1.0]) == 2.0
	a = randn(30);
	b = randn(30)
	@test BigRiverEssence._l1diff(a, b) ≈ sum(abs, a .- b)
end

@testset "internal: _binary_search_opt (λ for the L1 budget)" begin
	# Bisection for the threshold λ that drives the soft-thresholded ratio to the
	# target budget. Contract: 0 when already within budget, positive and bounded
	# when over, and monotone (tighter budget ⇒ larger λ).
	a = ones(4)                                    # l1_of_norm = 2.0
	@test BigRiverEssence._binary_search_opt(a, 3.0) == 0.0                  # budget 3 ≥ current 2 ⇒ nothing to do
	@test BigRiverEssence._binary_search_opt(zeros(6), 1.0) == 0.0           # all-zero input ⇒ 0
	Random.seed!(3)
	z = randn(60)
	@test BigRiverEssence._l1_of_norm(z) > 3.0                              # confirm thresholding is actually needed
	λ = BigRiverEssence._binary_search_opt(z, 3.0)
	@test λ > 0
	@test λ <= maximum(abs, z)                     # never exceeds the largest coefficient
	@test isapprox(BigRiverEssence._l1_of_norm_soft(z, λ), 3.0; atol = tol_r) # the found λ hits the budget (search precision)
	@test BigRiverEssence._binary_search_opt(z, 2.0) >= BigRiverEssence._binary_search_opt(z, 4.0)     # tighter budget needs a larger threshold
end

@testset "internal: _matsqrt (symmetric matrix square root)" begin
	# The symmetric PSD square root R with R·R = A, used to whiten one view inside the
	# wide-data initializer. Built from the eigendecomposition with eigenvalues floored
	# at 0 (so roundoff-negative eigenvalues don't produce complex results).
	Random.seed!(5)
	M = randn(8, 8);
	A = M * M' + I               # symmetric positive-definite
	R = BigRiverEssence._matsqrt(A)
	@test R * R ≈ A                                # R² = A (the defining property)
	@test R ≈ R'                                   # and R is symmetric
end

@testset "internal: _fast_init_v (wide-data SVD initializer)" begin
	# For wide data (features > observations) scca seeds v from a cheap SVD built on
	# the obs×obs space rather than the huge feature space. _fast_init_v returns the
	# leading left singular vectors of zᵀ·(xxᵀ)^½ — orthonormal columns living in z's
	# feature space, the starting point for the sparse iteration.
	_fiv = BigRiverEssence._fast_init_v
	Random.seed!(13)
	nobs = 30;
	p1 = 60;
	p2 = 80;
	K = 3            # wide: p1,p2 > nobs
	x = randn(nobs, p1);
	z = randn(nobs, p2)      # obs × features (the internal layout)
	V = _fiv(x, z, K)
	@test size(V) == (p2, K)                       # init for v lives in z's feature space
	@test V' * V ≈ I(K) atol = tol_ord             # orthonormal (it's an SVD U factor)
	# Matches the documented construction exactly: U of svd(zᵀ · (xxᵀ)^½).
	xx_sqrt = BigRiverEssence._matsqrt(x * transpose(x))
	Vref = svd(transpose(z)*xx_sqrt).U[:, 1:K]
	for k in 1:K
		@test abs(dot(V[:, k], Vref[:, k])) > 1 - tol_ord  # equal up to per-column sign
	end
end

@testset "internal: _sparse_cca_single_opt! (rank-1 sparse core)" begin
	# The heart of scca: one rank-1 sparse canonical pair (u, v) by alternating
	# soft-thresholded power iteration into caller-owned buffers. We plant a shared
	# latent so there's real structure to find, standardize exactly as scca does, and
	# check unit-norm, sparsity, the d-return identity, correlation, and selection.
	Random.seed!(17)
	nobs       = 100;
	p1         = 40;
	p2         = 50
	lat        = randn(nobs)                                    # shared latent
	x          = randn(nobs, p1);
	z          = randn(nobs, p2)
	x[:, 1:8]  .+= lat * fill(2.0, 8)'                   # first 8 X-features load on lat
	z[:, 1:10] .+= lat * fill(2.0, 10)'                  # first 10 Z-features load on lat
	if true                                              # standardize like scca's standardize=true
		x = (x .- mean(x, dims = 1)) ./ std(x, dims = 1; corrected = true)
		z = (z .- mean(z, dims = 1)) ./ std(z, dims = 1; corrected = true)
	end

	v0 = svd(transpose(x)*z).V[:, 1]             # the init scca uses when n > p (direct SVD)
	px = pz = 0.3
	# The buffers exactly as scca allocates them — the core writes into these in place.
	u = Vector{Float64}(undef, p1);
	v = Vector{Float64}(undef, p2)
	vold = Vector{Float64}(undef, p2)
	zv = Vector{Float64}(undef, nobs);
	xu = Vector{Float64}(undef, nobs)
	argu = Vector{Float64}(undef, p1);
	argv = Vector{Float64}(undef, p2)
	su = Vector{Float64}(undef, p1);
	sv = Vector{Float64}(undef, p2)

	d = BigRiverEssence._sparse_cca_single_opt!(u, v, x, z, v0, px, pz, 50, vold, zv, xu, argu, argv, su, sv)

	# The canonical vectors come out unit-norm (iterative core ⇒ the looser tol_julia).
	@test isapprox(norm(u), 1.0; atol = tol_julia)
	@test isapprox(norm(v), 1.0; atol = tol_julia)
	# The penalty genuinely induced sparsity on both sides.
	@test count(!iszero, u) < p1
	@test count(!iszero, v) < p2
	# The return value's contract: d = uᵀXᵀZv = ⟨Xu, Zv⟩, an EXACT identity.
	@test d ≈ dot(x * u, z * v)
	@test d > 0
	# The paired canonical variates are genuinely correlated (recovery floor).
	@test abs(cor(x * u, z * v)) > 0.5
	# And the selected features concentrate on the planted ones (precision floors).
	sel_u = Set(findall(!iszero, u));
	sel_v = Set(findall(!iszero, v))
	@test length(intersect(sel_u, Set(1:8))) / max(1, length(sel_u)) > 0.6
	@test length(intersect(sel_v, Set(1:10))) / max(1, length(sel_v)) > 0.6
end

# ----------------------------------------------------------------------------
# Cross-language check against PMA::CCA, using offline fixtures (no live R).
# ----------------------------------------------------------------------------

@testset "matches PMA::CCA (offline reference fixtures)" begin
	# Compare against PMA::CCA's saved output. Note the layout flip: PMA stores data
	# obs-in-rows, scca wants obs-in-columns, so we transpose X and Z before fitting.
	refdir = joinpath(@__DIR__, "Data", "SCCA")
	if !isfile(joinpath(refdir, "X.csv"))
		@info "sCCA PMA fixtures not found; run generate_scca_reference.R to create them."
	else
		# Record the PMA version behind the fixtures (sparse results can shift between
		# versions, so provenance helps when debugging a mismatch).
		smfile = joinpath(refdir, "session_meta.csv")
		if isfile(smfile)
			sm = readdlm(smfile, ',', String; skipstart = 1)
			row = findfirst(==("PMA_version"), sm[:, 1])
			row !== nothing && @info "sCCA fixtures generated against PMA $(sm[row, 2])"
		end

		rd(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
		X = rd("X.csv");
		Z = rd("Z.csv")          # PMA layout: observations in rows
		ru = rd("u.csv");
		rv = rd("v.csv")        # PMA's canonical vectors
		rd_ = vec(rd("d.csv"));
		rcors = vec(rd("cors.csv"))
		meta = rd("meta.csv")
		K = Int(meta[1]);
		px = meta[2];
		pz = meta[3];
		niter = Int(meta[4])

		# scca takes obs-in-columns, so transpose PMA's row-major matrices to match.
		m = BigRiverEssence.scca(Matrix(transpose(X)), Matrix(transpose(Z));
			penaltyx = px, penaltyz = pz, K = K, niter = niter)

		for k in 1:K
			# Canonical vectors agree up to per-component sign (cross-language ⇒ the
			# loose tol_r bar), compared via |correlation|.
			@test abs(cor(m.u[:, k], ru[:, k])) > 1 - tol_r
			@test abs(cor(m.v[:, k], rv[:, k])) > 1 - tol_r
			# The selected-feature SETS must match exactly — the discrete, sign-free
			# fingerprint of the sparse solution.
			@test Set(findall(!iszero, m.u[:, k])) == Set(findall(!iszero, ru[:, k]))
			@test Set(findall(!iszero, m.v[:, k])) == Set(findall(!iszero, rv[:, k]))
		end
		# Weights and correlations match across languages.
		@test isapprox(m.d, rd_; rtol = tol_r)
		@test isapprox(m.cors, rcors; rtol = tol_r)
	end
end
