# Test/pmd_test.jl — tests for the two-sided PMD(L1,L1): penalized matrix
# decomposition with an L1 penalty on BOTH the left (u) and right (v) factors,
# after Witten, Tibshirani & Hastie (2009). Tolerances (tol_ord / tol_julia /
# tol_r) come from runtests.jl: tol_ord for exact math, tol_julia for the
# iterative power-iteration results, tol_r for the cross-language R fixtures.

@testset "output structure & invariants" begin
	# Basic contract: right type and shapes, and the properties a PMD result must
	# always hold — unit-norm factors, finite weights, budgets in range.
	Random.seed!(1)
	n, p, K = 80, 60, 3
	X = randn(n, p)
	m = pmd(X; sumabs = 0.4, K = K)

	@test m isa pmdStructure
	@test size(m.u) == (n, K)          # left factors: one length-n column per component
	@test size(m.v) == (p, K)          # right factors: one length-p column per component
	@test length(m.d) == K
	@test m.K == K

	# Each factor column is unit-norm — OR exactly zero, which happens when a tight
	# penalty zeros out an entire factor (a legitimate sparse outcome, not a bug).
	for k in 1:K
		nu = norm(m.u[:, k]);
		nv = norm(m.v[:, k])
		@test isapprox(nu, 1.0; atol = tol_ord) || nu == 0.0
		@test isapprox(nv, 1.0; atol = tol_ord) || nv == 0.0
	end
	# The weights dₖ are real scalars and must be finite (no NaN/Inf leaking through).
	@test all(isfinite, m.d)
	# The fractional sumabs budget gets resolved into an absolute L1 budget; it must
	# land in the valid range [1, √dim] for each side.
	@test 1 <= m.sumabsu <= sqrt(n) + tol_ord
	@test 1 <= m.sumabsv <= sqrt(p) + tol_ord
	# When centering, the grand mean that was subtracted is recorded in the model.
	@test isapprox(m.meanx, mean(X); atol = tol_ord)
end

@testset "max budget reduces to rank-1 SVD (theorem anchor)" begin
	# The key theoretical check: when both budgets are maxed (sumabsu=√n, sumabsv=√p),
	# neither L1 penalty binds, so no soft-thresholding happens. PMD then degenerates
	# to plain power iteration, whose fixed point is the rank-1 SVD of the centered
	# data. So at max budget, PMD must reproduce the leading singular triple.
	Random.seed!(7)
	n, p = 60, 40
	X = randn(n, p)
	Xc = X .- mean(X)
	m = pmd(X; sumabsu = sqrt(n), sumabsv = sqrt(p), K = 1)
	F = svd(Xc)
	@test abs(dot(m.u[:, 1], F.U[:, 1])) > 1 - tol_julia   # u converges to U₁ (|dot|: sign-free)
	@test abs(dot(m.v[:, 1], F.V[:, 1])) > 1 - tol_julia   # v converges to V₁
	# With no penalty binding, nothing is zeroed — both factors are fully dense.
	@test count(!iszero, m.u[:, 1]) == n
	@test count(!iszero, m.v[:, 1]) == p
end

@testset "two-sided sparsity contract" begin
	# The defining behavior of PMD(L1,L1): tightening the budget makes BOTH factors
	# sparser, and a max budget leaves them dense. We check the monotone relationship
	# — smaller sumabs never increases the number of nonzeros, on either side.
	Random.seed!(11)
	n, p  = 80, 60
	X     = randn(n, p)
	dense = pmd(X; sumabsu = sqrt(n), sumabsv = sqrt(p), K = 1)   # no penalty → dense
	mid   = pmd(X; sumabs = 0.5, K = 1)
	tight = pmd(X; sumabs = 0.25, K = 1)                        # strong penalty → sparse
	@test count(!iszero, dense.u[:, 1]) == n
	@test count(!iszero, dense.v[:, 1]) == p
	# Tighter budget ⇒ no more nonzeros than a looser one (monotone), on both sides.
	@test count(!iszero, tight.u[:, 1]) <= count(!iszero, mid.u[:, 1]) <= n
	@test count(!iszero, tight.v[:, 1]) <= count(!iszero, mid.v[:, 1]) <= p
	@test count(!iszero, tight.v[:, 1]) < p           # and the tight budget really does zero some
end

@testset "ground-truth: recovers planted sparse factors" begin
	# Plant two well-separated rank-1 components, each with a sparse u AND a sparse v,
	# then add noise. With a strong signal and the SVD init, the optimum is unique,
	# so PMD should recover both the directions and which features are nonzero.
	Random.seed!(42)
	n, p = 80, 60
	# component 1: u nonzero in rows 1:20, v nonzero in cols 1:15
	u1 = [randn(20); zeros(60)];
	v1 = [randn(15); zeros(45)]
	# component 2: disjoint support from component 1, so they don't interfere
	u2 = [zeros(40); randn(20); zeros(20)];
	v2 = [zeros(30); randn(15); zeros(15)]
	X = 6.0*(u1*v1') .+ 4.0*(u2*v2') .+ 0.3 .* randn(n, p)   # two signals + small noise
	m = pmd(X; sumabsu = 0.45*sqrt(n), sumabsv = 0.45*sqrt(p), K = 2, center = true)

	for (k, (ut, vt)) in enumerate(((u1, v1), (u2, v2)))
		# Direction recovery: the fitted factor aligns with the planted one. These are
		# deliberately LITERAL floors (0.9, 0.8) — looser than 1-tol_r — because the
		# data is noisy planted ground truth, not an exact reference.
		@test abs(dot(m.u[:, k] ./ norm(m.u[:, k]), ut ./ norm(ut))) > 0.9
		@test abs(dot(m.v[:, k] ./ norm(m.v[:, k]), vt ./ norm(vt))) > 0.9
		# Support recovery: most of the truly-nonzero features are picked up.
		selu = Set(findall(!iszero, m.u[:, k]));
		trueu = Set(findall(!iszero, ut))
		selv = Set(findall(!iszero, m.v[:, k]));
		truev = Set(findall(!iszero, vt))
		@test length(intersect(selu, trueu)) / length(trueu) > 0.8
		@test length(intersect(selv, truev)) / length(truev) > 0.8
	end
end

@testset "multiple components (deflation)" begin
	# With K > 1, PMD deflates after each component and extracts the next. Check the
	# output shapes are right and every component is sparse at this binding budget.
	Random.seed!(5)
	n, p, K = 60, 50, 4
	X = randn(n, p)
	m = pmd(X; sumabs = 0.4, K = K)
	@test size(m.u) == (n, K)
	@test size(m.v) == (p, K)
	@test length(m.d) == K
	for k in 1:K
		@test count(!iszero, m.v[:, k]) < p     # each deflated component is still penalized
	end
end

@testset "sumabsu/sumabsv override sumabs" begin
	# If you pass explicit per-side budgets, they take precedence over the scalar
	# sumabs — the model should record exactly what you asked for, ignoring sumabs.
	Random.seed!(15)
	n, p = 70, 50
	X = randn(n, p)
	su = 0.3*sqrt(n);
	sv = 0.6*sqrt(p)
	m = pmd(X; sumabs = 0.9, sumabsu = su, sumabsv = sv, K = 1)   # sumabs=0.9 should be ignored
	@test isapprox(m.sumabsu, su; atol = tol_ord)
	@test isapprox(m.sumabsv, sv; atol = tol_ord)
end

@testset "center=false skips centering" begin
	# With center=false the data isn't mean-subtracted, so no grand mean is recorded
	# — the meanx field is left as NaN to signal "no centering was done".
	Random.seed!(9)
	n, p = 50, 40
	X = randn(n, p) .+ 5.0                # deliberately nonzero mean
	m = pmd(X; sumabs = 0.5, K = 1, center = false)
	@test isnan(m.meanx)                  # NaN sentinel = "didn't center"
end

@testset "determinism (SVD init, no random dependence)" begin
	# PMD's inner loop primes `vold` with randn, but the SVD-based initialization
	# fixes the starting point — so the primer only forces the first iteration to
	# run, it doesn't move the converged answer. Repeat calls must therefore agree.
	Random.seed!(99)
	n, p = 60, 40
	X = randn(n, p)
	a = pmd(X; sumabs = 0.4, K = 2)
	b = pmd(X; sumabs = 0.4, K = 2)
	for k in 1:2
		@test abs(dot(a.v[:, k] ./ norm(a.v[:, k]), b.v[:, k] ./ norm(b.v[:, k]))) > 1 - tol_ord
		@test isapprox(a.d[k], b.d[k]; rtol = tol_ord)
	end
end

@testset "argument validation" begin
	# Out-of-range arguments must raise ArgumentError rather than silently misbehave.
	Random.seed!(0)
	n, p = 100, 16
	X = randn(n, p)
	# K must be in 1:min(n,p)
	@test_throws ArgumentError pmd(X; K = 0)
	@test_throws ArgumentError pmd(X; K = min(n, p)+1)
	# the fractional sumabs must be in (0,1]
	@test_throws ArgumentError pmd(X; sumabs = 0.0)
	@test_throws ArgumentError pmd(X; sumabs = 1.5)
	# explicit absolute budgets must each be in [1, √dim]
	@test_throws ArgumentError pmd(X; sumabsu = 0.5, sumabsv = 2.0)        # sumabsu < 1
	@test_throws ArgumentError pmd(X; sumabsu = 2.0, sumabsv = sqrt(p)+1)  # sumabsv > √p
end

# ---------------------------------------------------------------------------
# Internal-helper tests. These exercise the building blocks of the PMD iteration
# directly (via BigRiverEssence._name) so a regression in a primitive is caught
# at the source, not just as a downstream symptom in the full fit.
# ---------------------------------------------------------------------------

@testset "internal: _pmd_soft (soft-threshold operator)" begin
	# The L1 proximal operator S(a,λ) = sign(a)·max(|a|−λ, 0): shrink toward zero by
	# λ, snap to zero once |a| ≤ λ. This is the single primitive that creates sparsity.
	soft = BigRiverEssence._pmd_soft
	@test soft(5.0, 2.0) == 3.0          # shrink by λ
	@test soft(-5.0, 2.0) == -3.0         # sign preserved through the shrink
	@test soft(1.0, 2.0) == 0.0          # |a| ≤ λ ⇒ snapped to exactly 0
	@test soft(-1.0, 2.0) == 0.0
	@test soft(3.0, 0.0) == 3.0          # λ=0 is the identity (no shrink)
	@test soft(0.0, 1.0) == 0.0          # sign(0)=0 ⇒ 0, and crucially no NaN
	# Must match the closed form elementwise — not just on the hand-picked cases.
	a = randn(100);
	λ = 0.7
	@test soft.(a, λ) ≈ sign.(a) .* max.(abs.(a) .- λ, 0.0)
end

@testset "internal: _pmd_l2n (guarded L2 norm)" begin
	# Euclidean norm, but with PMA's zero-guard: an all-zero vector returns 0.05
	# instead of 0, so the later normalization steps never divide by zero.
	l2n = BigRiverEssence._pmd_l2n
	@test l2n([3.0, 4.0]) == 5.0
	@test l2n(zeros(5)) == 0.05         # the guard: 0 would blow up a downstream /‖·‖
	a = randn(50)
	@test l2n(a) ≈ norm(a)                # identical to the true norm whenever nonzero
end

@testset "internal: _pmd_l1_norm & _pmd_l1_norm_soft (L1/L2 ratio)" begin
	# The quantity the L1 budget actually constrains is the ratio ‖a‖₁/‖a‖₂, which
	# ranges from 1 (one nonzero) to √m (all equal). l1_norm computes it; the _soft
	# variant computes it for a soft-thresholded vector without materializing it.
	l1n  = BigRiverEssence._pmd_l1_norm
	l1ns = BigRiverEssence._pmd_l1_norm_soft
	soft = BigRiverEssence._pmd_soft
	@test l1n([3.0, 4.0]) ≈ 7 / 5         # (3+4) / 5
	@test l1n(ones(9)) ≈ 3.0           # 9 / 3 = √9, the dense maximum
	@test l1n(zeros(4)) == 0.0          # 0 / 0.05 guard ⇒ 0 (no NaN)
	# The soft-then-ratio shortcut must equal applying soft first, then the ratio.
	a = randn(80);
	λ = 0.5
	@test l1ns(a, λ) ≈ l1n(soft.(a, λ))
	# The ratio is ≥ 1 for any nonzero vector (Cauchy–Schwarz lower bound).
	@test l1n(randn(30)) >= 1 - tol_ord
end

@testset "internal: _pmd_l1diff (L1 distance)" begin
	# ‖a−b‖₁, the convergence metric between successive iterates.
	l1diff = BigRiverEssence._pmd_l1diff
	@test l1diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0     # identical ⇒ 0
	@test l1diff([1.0, 0.0], [0.0, 1.0]) == 2.0               # |1-0|+|0-1|
	a = randn(40);
	b = randn(40)
	@test l1diff(a, b) ≈ sum(abs, a .- b)
end

@testset "internal: _pmd_binary_search (λ solving the L1 budget)" begin
	# Given a target L1 budget, bisection finds the threshold λ such that the
	# soft-thresholded vector hits that budget. The contract: 0 when already within
	# budget, positive and bounded when over, and monotone in the budget.
	bs   = BigRiverEssence._pmd_binary_search
	l1n  = BigRiverEssence._pmd_l1_norm
	l1ns = BigRiverEssence._pmd_l1_norm_soft
	# Already within budget ⇒ nothing to threshold ⇒ λ = 0.
	a = ones(4)                            # l1_norm = 2.0
	@test l1n(a) ≈ 2.0
	@test bs(a, 3.0) == 0.0                # target 3 ≥ current 2 ⇒ no thresholding
	@test bs(zeros(6), 1.0) == 0.0         # all-zero input ⇒ 0 (nothing to do)
	# Over budget ⇒ a positive λ that drives the soft ratio down to the target.
	Random.seed!(3)
	z = randn(60)
	@test l1n(z) > 3.0                     # confirm thresholding is genuinely needed
	λ = bs(z, 3.0)
	@test λ > 0
	@test λ <= maximum(abs, z)             # λ never exceeds the largest |coefficient|
	@test isapprox(l1ns(z, λ), 3.0; atol = tol_r)   # the found λ actually hits the budget (to search precision)
	# Monotone: a tighter (smaller) budget requires a larger threshold.
	@test bs(z, 2.0) >= bs(z, 4.0)
end

@testset "internal: _pmd_soft_normalize! (soft-threshold then unit-normalize)" begin
	# The fused update used each iteration: soft-threshold into a buffer, then scale
	# to unit L2 norm. Done in place to avoid allocating per iteration.
	sn   = BigRiverEssence._pmd_soft_normalize!
	soft = BigRiverEssence._pmd_soft
	arg  = [3.0, -4.0, 0.5];
	out  = similar(arg)
	sn(out, arg, 1.0)                      # soft([3,-4,0.5], 1) = [2,-3,0]; then /‖·‖ = /√13
	@test out ≈ [2.0, -3.0, 0.0] ./ sqrt(13)
	@test isapprox(norm(out), 1.0; atol = tol_ord)  # result is unit-norm when anything survives
	@test sign.(out) == sign.([2.0, -3.0, 0.0])   # signs carry through unchanged
	# Must match the explicit soft-then-normalize on a random vector.
	a = randn(50);
	o = similar(a);
	λ = 0.6
	sn(o, a, λ)
	s = soft.(a, λ)
	@test o ≈ s ./ norm(s)
	# If λ thresholds EVERYTHING away, the guard (0.05) prevents 0/0 — output is all
	# zeros, no NaN. This is the degenerate case the zero-guard exists for.
	o2 = similar(a);
	sn(o2, a, maximum(abs, a) + 1.0)
	@test all(iszero, o2)
end

@testset "internal: _pmd_check_v (SVD-based initialization)" begin
	# The iteration is seeded from the leading right singular vectors. _pmd_check_v
	# computes them via the cheaper of two routes depending on shape, and must return
	# unit columns equal (up to sign) to svd's V on both tall and wide inputs.
	cv = BigRiverEssence._pmd_check_v
	# Tall (p ≤ n): goes through the svd(xᵀx) branch (small p×p problem).
	Random.seed!(21)
	Xt = randn(60, 40);
	K = 3
	Vt = cv(Xt, K)
	@test size(Vt) == (40, K)
	Ft = svd(Xt)
	for k in 1:K
		@test isapprox(norm(@view Vt[:, k]), 1.0; atol = tol_ord)     # unit columns
		@test abs(dot(Vt[:, k], Ft.V[:, k])) > 1 - tol_ord           # = right singular vec (up to sign)
	end
	# Wide (p > n): goes through the svd(xxᵀ) branch, then back-projects to p-space.
	Xw = randn(30, 50)
	Vw = cv(Xw, K)
	@test size(Vw) == (50, K)
	Fw = svd(Xw)
	for k in 1:K
		@test isapprox(norm(@view Vw[:, k]), 1.0; atol = tol_ord)
		@test abs(dot(Vw[:, k], Fw.V[:, k])) > 1 - tol_ord
	end
end

@testset "internal: _pmd_smd! (rank-1 sparse core)" begin
	# The heart of PMD: one rank-1 penalized factor pair, computed by alternating
	# soft-thresholded power iteration into preallocated buffers. Two checks — at max
	# budget it must equal the rank-1 SVD; at a binding budget it must be sparse.
	smd = BigRiverEssence._pmd_smd!
	cv  = BigRiverEssence._pmd_check_v
	Random.seed!(31)
	n, p = 50, 40
	X = randn(n, p);
	Xc = X .- mean(X)
	v0 = cv(Xc, 1)[:, 1]                   # SVD-based starting v
	u = Vector{Float64}(undef, n);
	v = Vector{Float64}(undef, p)
	vold = Vector{Float64}(undef, p)
	argu = Vector{Float64}(undef, n);
	argv = Vector{Float64}(undef, p)
	# Max budget (√n, √p): no penalty binds ⇒ pure power iteration ⇒ rank-1 SVD.
	d = smd(Xc, v0, sqrt(n), sqrt(p), 50, u, v, vold, argu, argv)
	F = svd(Xc)
	@test isapprox(norm(u), 1.0; atol = tol_julia)
	@test isapprox(norm(v), 1.0; atol = tol_julia)
	@test abs(dot(u, F.U[:, 1])) > 1 - tol_julia      # u → U₁
	@test abs(dot(v, F.V[:, 1])) > 1 - tol_julia      # v → V₁
	@test isapprox(d, F.S[1]; rtol = tol_julia)       # the weight d → σ₁
	@test d > 0
	# A binding budget (0.4) makes both factors genuinely sparse. Fresh buffers so the
	# previous call's results don't leak in.
	u2 = similar(u);
	v2 = similar(v);
	vo2 = similar(vold)
	au2 = similar(argu);
	av2 = similar(argv)
	smd(Xc, v0, 0.4 * sqrt(n), 0.4 * sqrt(p), 50, u2, v2, vo2, au2, av2)
	@test count(!iszero, v2) < p
	@test count(!iszero, u2) < n
end

@testset "matches R PMA::PMD (offline reference fixtures)" begin
	# Cross-language check against R's PMA::PMD, using precomputed fixtures in
	# Test/Data/PMD/ (generated by pmd.R). Offline so the suite doesn't need a live R;
	# if the fixtures aren't present, the test skips with a note rather than failing.
	refdir = joinpath(@__DIR__, "Data", "PMD")
	if !isfile(joinpath(refdir, "X.csv"))
		@info "PMD R-reference fixtures not found; skipping. Run pmd.R to create them."
	else
		# Record which PMA version produced the fixtures — sparse-method results can
		# shift slightly between package versions, so provenance matters for debugging.
		smfile = joinpath(refdir, "session_meta.csv")
		if isfile(smfile)
			sm = readdlm(smfile, ',', String; skipstart = 1)
			row = findfirst(==("PMA_version"), sm[:, 1])
			row !== nothing && @info "PMD fixtures generated against PMA $(sm[row, 2])"
		end

		X = readdlm(joinpath(refdir, "X.csv"), ',', Float64; skipstart = 1)
		u_r = readdlm(joinpath(refdir, "u_pmd.csv"), ',', Float64; skipstart = 1)
		v_r = readdlm(joinpath(refdir, "v_pmd.csv"), ',', Float64; skipstart = 1)
		d_r = vec(readdlm(joinpath(refdir, "d_pmd.csv"), ',', Float64; skipstart = 1))
		meta = readdlm(joinpath(refdir, "meta.csv"), ',', Float64; skipstart = 1)
		K = Int(meta[3]);
		su = meta[4];
		sv = meta[5]

		# X.csv was already grand-mean-centered inside R, so we fit with center=false
		# to avoid double-centering — otherwise the comparison would be apples-to-oranges.
		m = pmd(X; sumabsu = su, sumabsv = sv, K = K, center = false)

		for k in 1:K
			# Factors align up to sign; cross-language numerics ⇒ the looser tol_r bar.
			@test abs(dot(m.u[:, k] ./ norm(m.u[:, k]), u_r[:, k] ./ norm(u_r[:, k]))) > 1 - tol_r
			@test abs(dot(m.v[:, k] ./ norm(m.v[:, k]), v_r[:, k] ./ norm(v_r[:, k]))) > 1 - tol_r
			# The exact same features must be zeroed on both sides — support recovery
			# is the discrete fingerprint of the sparse solution, and it should match exactly.
			@test Set(findall(!iszero, m.u[:, k])) == Set(findall(!iszero, u_r[:, k]))
			@test Set(findall(!iszero, m.v[:, k])) == Set(findall(!iszero, v_r[:, k]))
		end
		# Singular values match across languages (continuous values ⇒ tol_r).
		@test isapprox(m.d, d_r; rtol = tol_r)
	end
end
