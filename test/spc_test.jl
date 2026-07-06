# Test/spc_test.jl — tests for SPC (Witten, Tibshirani & Hastie 2009 sparse PCA):
# the deflation variant spc and the orthogonal-score variant spc_orth, plus their
# internals (finding_v!, init_rsv, prop_var_explained, spca_component!/spca_component_orth!).
# Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl: tol_ord for exact
# math, tol_julia for iterative power-iteration results, tol_r for cross-language R.

@testset "output structure & invariants" begin
	# Basic contract: type, shapes, recorded centering stats, and the SPC properties
	# — unit-norm (or zero) loadings, finite variances, and a cumulative proportion
	# of variance explained that's nondecreasing and within [0,1].
	Random.seed!(1)
	n, p, k = 80, 60, 3
	X = randn(n, p)
	m = spc(X; k = k, c = 0.6*sqrt(p))

	@test m isa spcStructure
	@test size(m.loadings) == (p, k)
	@test length(m.variances) == k
	@test length(m.propOFvar) == k
	@test length(m.mean) == p
	@test length(m.scale) == p

	# Each sparse loading is unit-norm — OR exactly zero, if a tight budget zeroed
	# the whole component (a legitimate sparse outcome).
	for j in 1:k
		nv = norm(m.loadings[:, j])
		@test isapprox(nv, 1.0; atol = tol_ord) || nv == 0.0
	end
	@test all(isfinite, m.variances)
	@test all(isfinite, m.propOFvar)
	# propOFvar is CUMULATIVE proportion of variance explained: in [0,1] and rising.
	@test all(0 .<= m.propOFvar .<= 1 + tol_ord)
	@test all(diff(m.propOFvar) .>= -tol_ord)              # nondecreasing
	# Default standardize=false ⇒ only centered, so scales stay at 1.
	@test isapprox(m.mean, vec(mean(X, dims = 1)); atol = tol_ord)
	@test all(isapprox.(m.scale, 1.0; atol = tol_ord))
end

@testset "max budget reduces to ordinary PCA (theorem anchor)" begin
	# The anchoring theorem: at c = √p the L1 penalty can never bind (no unit vector
	# has L1 norm above √p), so no soft-thresholding happens and SPC degenerates to
	# ordinary PCA — the loading is the dense top principal component of centered X.
	Random.seed!(7)
	n, p = 60, 40
	X = randn(n, p)
	Xc = X .- mean(X, dims = 1)
	m = spc(X; k = 1, c = sqrt(p))
	F = svd(Xc)
	@test abs(dot(m.loadings[:, 1], F.V[:, 1])) > 1 - tol_julia   # = top PC (|dot|: sign-free, iterative)
	@test count(!iszero, m.loadings[:, 1]) == p          # and fully dense (no penalty bound)
end

@testset "sparsity contract (c controls sparsity)" begin
	# The budget c tunes sparsity: smaller c ⇒ fewer nonzeros (monotone), c = √p ⇒ dense.
	Random.seed!(11)
	n, p  = 80, 60
	X     = randn(n, p)
	dense = spc(X; k = 1, c = sqrt(p))         # no penalty → dense
	mid   = spc(X; k = 1, c = 0.5*sqrt(p))
	tight = spc(X; k = 1, c = 1.5)             # tight penalty → sparse
	@test count(!iszero, dense.loadings[:, 1]) == p
	@test count(!iszero, tight.loadings[:, 1]) <= count(!iszero, mid.loadings[:, 1]) <= p   # monotone
	@test count(!iszero, tight.loadings[:, 1]) < p        # tight really does zero some
end

@testset "ground-truth: recovers planted sparse loading" begin
	# Plant a sparse rank-1 signal (v nonzero in cols 1:75). A TIGHT budget gives high
	# PRECISION — the few variables it selects are genuinely signal — but recall is
	# capped by the budget, so we assert on precision + direction here, and check that
	# LOOSENING the budget raises recall. Floors are literal recovery bars (noisy data).
	Random.seed!(42)
	n, p = 200, 300
	u_true = [randn(50); zeros(n-50)]
	v_true = [randn(75); zeros(p-75)]
	X = u_true * v_true' .+ randn(n, p)                  # rank-1 sparse signal + noise
	m = spc(X; k = 1, c = 4.0)                               # tight budget
	vh = m.loadings[:, 1]
	est = Set(findall(!iszero, vh));
	truth = Set(1:75)
	prec = length(intersect(est, truth)) / length(est)
	@test prec > 0.8                                    # most selected vars are truly signal
	@test abs(dot(vh ./ norm(vh), v_true ./ norm(v_true))) > 0.6   # direction aligns
	# Loosening the budget lets more true variables in ⇒ recall climbs.
	m2 = spc(X; k = 1, c = 8.0)
	rec2 = length(intersect(Set(findall(!iszero, m2.loadings[:, 1])), truth)) / 75
	@test rec2 > 0.8                                    # looser budget recovers most of the support
end

@testset "spc_orth: scores are orthonormal" begin
	# The difference between the two variants is in the SCORES. Deflation (spc) leaves
	# score directions correlated; the orthogonal variant (spc_orth) forces them
	# orthogonal. spcStructure doesn't store the scores, so we recompute T = Xc·V and
	# compare off-diagonal score correlations — orth should be ≤ deflation.
	Random.seed!(5)
	n, p, k = 100, 50, 4
	X = randn(n, p)
	Xc = X .- mean(X, dims = 1)
	mo = spc_orth(X; k = k, c = 2.0)
	md = spc(X; k = k, c = 2.0)
	offdiag(C) = maximum(abs(C[i, j]) for i in 1:size(C, 1) for j in 1:size(C, 2) if i != j)
	Co = cor(Xc * mo.loadings)
	Cd = cor(Xc * md.loadings)
	@test offdiag(Co) <= offdiag(Cd) + tol_ord          # orth scores less correlated than deflation
end

@testset "multiple components, shapes" begin
	# With k > 1, shapes are right and every component is sparse at this binding budget.
	Random.seed!(5)
	n, p, k = 60, 50, 4
	X = randn(n, p)
	m = spc(X; k = k, c = 2.0)
	@test size(m.loadings) == (p, k)
	@test length(m.variances) == k
	for j in 1:k
		@test count(!iszero, m.loadings[:, j]) < p        # each component penalized
	end
end

@testset "standardize=true scales columns" begin
	# standardize=true records each column's SD as its scale; false leaves scales at 1.
	# Columns are put on wildly unequal scales so a missing standardization is obvious.
	Random.seed!(9)
	n, p = 60, 40
	X = randn(n, p) .* (1:p)'            # column j scaled by j
	m = spc(X; k = 1, c = 0.5*sqrt(p), standardize = true)
	@test isapprox(m.scale, vec(std(X, dims = 1)); atol = tol_ord)
	md = spc(X; k = 1, c = 0.5*sqrt(p), standardize = false)
	@test all(isapprox.(md.scale, 1.0; atol = tol_ord))
end

@testset "determinism (SVD init, no random dependence)" begin
	# SPC is seeded from the SVD, so despite any internal iteration it's deterministic
	# — repeat calls on the same data must give the same loadings and variances.
	Random.seed!(99)
	n, p = 60, 40
	X = randn(n, p)
	a = spc(X; k = 2, c = 0.5*sqrt(p))
	b = spc(X; k = 2, c = 0.5*sqrt(p))
	for j in 1:2
		@test abs(dot(a.loadings[:, j] ./ norm(a.loadings[:, j]),
			b.loadings[:, j] ./ norm(b.loadings[:, j]))) > 1 - tol_ord
		@test isapprox(a.variances[j], b.variances[j]; rtol = tol_ord)
	end
end

@testset "argument validation" begin
	# The budget c must lie in [1, √p] for both variants — outside that range throws.
	Random.seed!(0)
	n, p = 100, 16
	X = randn(n, p)
	@test_throws ArgumentError spc(X; k = 1, c = 0.5)            # c < 1
	@test_throws ArgumentError spc(X; k = 1, c = sqrt(p)+1)      # c > √p
	@test_throws ArgumentError spc_orth(X; k = 1, c = 0.5)
	@test_throws ArgumentError spc_orth(X; k = 1, c = sqrt(p)+1)
end

# ----------------------------------------------------------------------------
# Internal helpers — tested directly so a regression in a primitive surfaces here,
# not as a confusing symptom in the full sparse-PCA fit.
# ----------------------------------------------------------------------------

@testset "internal: l1_diff (L1 distance)" begin
	# ‖a−b‖₁, the convergence metric between successive loading iterates.
	l1d = BigRiverEssence.l1_diff
	@test l1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0       # identical ⇒ 0
	@test l1d([1.0, 0.0], [0.0, 1.0]) == 2.0
	a = randn(40);
	b = randn(40)
	@test l1d(a, b) ≈ sum(abs, a .- b)
end

@testset "internal: finding_v! (soft-threshold to the L1 budget)" begin
	# The core sparsifier: given a raw direction z and a budget c, find the threshold
	# so the soft-thresholded, then L2-normalized, vector has L1 norm = c. The contract:
	# unit L2, L1 hits c (when binding), signs inherited from z, small entries zeroed.
	fv = BigRiverEssence.finding_v!
	p = 60
	Random.seed!(2)
	z = randn(p);
	v = similar(z);
	s = similar(z)              # s is preallocated scratch

	# (1) Slack budget: c = √p is the max possible L1 of a unit vector, so the penalty
	# can't bind — z is just normalized and returned, no thresholding.
	fv(v, s, z, sqrt(p))
	@test v ≈ z ./ norm(z)
	@test isapprox(norm(v), 1.0; atol = tol_ord)

	# (2) Binding budgets: the result is unit-L2 and its L1 norm lands on the target c
	# (to search precision), with signs taken from z.
	for c in (2.0, 4.0, 6.0)
		fv(v, s, z, c)
		@test isapprox(norm(v), 1.0; atol = tol_ord)         # unit L2
		@test isapprox(sum(abs, v), c; atol = tol_r)         # L1 = c (bisection precision)
		for i in eachindex(v)
			v[i] != 0 && @test sign(v[i]) == sign(z[i])      # surviving signs match z
		end
	end

	# Sparsity only arises for a TIGHT budget (c well below √p ≈ 7.75).
	fv(v, s, z, 2.0)
	@test count(!iszero, v) < p                              # tight ⇒ some entries zeroed

	# (3) Monotone: a tighter budget is at least as sparse as a looser one.
	fv(v, s, z, 2.0);
	n_tight = count(!iszero, v)
	fv(v, s, z, 5.0);
	n_loose = count(!iszero, v)
	@test n_tight <= n_loose

	# (4) Cross-check against the explicit construction: independently re-solve the
	# threshold δ by bisection here, build soft(z,δ)/‖·‖ by hand, and confirm
	# finding_v! lands on the same direction. Two routes to the same budget-solve.
	c = 3.0;
	fv(v, s, z, c)
	δ = let lo = 0.0, hi = maximum(abs, z)          # bracket δ in [0, max|z|]
		for _ in 1:200
			mid = (lo + hi) / 2
			sm = sign.(z) .* max.(abs.(z) .- mid, 0.0)       # soft-threshold at mid
			(sum(abs, sm) / (norm(sm) + eps()) < c) ? (hi = mid) : (lo = mid)   # L1/L2 vs target
		end
		(lo + hi) / 2
	end
	ref = sign.(z) .* max.(abs.(z) .- δ, 0.0);
	ref ./= norm(ref)
	@test abs(dot(v, ref)) > 1 - tol_julia          # same answer, computed two ways
end

@testset "internal: init_rsv (top-k right singular vectors, both branches)" begin
	# The SVD-based initialization. init_rsv computes the top-k right singular vectors
	# via the cheaper route for the shape — eigen(XᵀX) when tall, eigen(XXᵀ) + back-
	# projection when wide — and must equal svd's V (up to sign) on both.
	irsv = BigRiverEssence.init_rsv
	# Tall (p ≤ n): the eigen(XᵀX) branch (small p×p problem).
	Random.seed!(21)
	Xt = randn(60, 40);
	k = 3
	Vt = irsv(Xt, k)
	@test size(Vt) == (40, k)
	Ft = svd(Xt)
	for j in 1:k
		@test isapprox(norm(@view Vt[:, j]), 1.0; atol = tol_ord)    # unit columns
		@test abs(dot(Vt[:, j], Ft.V[:, j])) > 1 - tol_ord          # = right singular vector (up to sign)
	end
	# Wide (p > n): the eigen(XXᵀ) branch, then back-project to p-space.
	Xw = randn(30, 50)
	Vw = irsv(Xw, k)
	@test size(Vw) == (50, k)
	Fw = svd(Xw)
	for j in 1:k
		@test isapprox(norm(@view Vw[:, j]), 1.0; atol = tol_ord)
		@test abs(dot(Vw[:, j], Fw.V[:, j])) > 1 - tol_ord
	end
end

@testset "internal: prop_var_explained (trace identity vs explicit projection)" begin
	# prop_var_explained computes cumulative variance explained by the first k sparse
	# loadings. Sparse loadings are NOT orthonormal, so the projection onto their span
	# needs the (VᵀV)⁻¹ term: ‖Xc·Vk·(VkᵀVk)⁻¹·Vkᵀ‖²_F / ‖Xc‖²_F. The implementation
	# uses a cheaper trace rewrite of this; this test proves the rewrite is exact by
	# comparing to the explicit projection form, using a deliberately NON-orthonormal V.
	pve = BigRiverEssence.prop_var_explained
	Random.seed!(33)
	n, p, K = 50, 40, 4
	Xc = randn(n, p) .- mean(randn(n, p), dims = 1)
	V = randn(p, K);
	V[abs.(V) .< 0.3] .= 0.0      # sparse, non-orthonormal ⇒ stresses (VᵀV)⁻¹
	got = pve(Xc, V)

	# Reference: the explicit displaced (oblique) projection form the rewrite replaces.
	totsq = sum(abs2, Xc)
	ref = [
		let Vk = V[:, 1:k]
			sum(abs2, Xc * Vk * inv(Vk' * Vk) * Vk') / totsq
		end for k in 1:K
	]
	@test got ≈ ref                                # the trace rewrite is algebraically exact
	@test all(0 .<= got .<= 1 + tol_ord)           # valid proportions
	@test all(diff(got) .>= -tol_ord)              # cumulative ⇒ nondecreasing

	# Independent anchor: for ORTHONORMAL V (the SVD basis), the (VᵀV)⁻¹ term is I and
	# the formula must collapse to the familiar Σσ₁..ₖ² / Σσ² ratio.
	Vo = svd(Xc).V[:, 1:K]
	S  = svd(Xc).S
	@test pve(Xc, Vo) ≈ cumsum(S[1:K] .^ 2) ./ sum(abs2, S)
end

@testset "internal: spca_component! (rank-1 sparse core, deflation)" begin
	# The deflation core: one rank-1 sparse loading by soft-thresholded power iteration
	# into preallocated buffers. Two checks — max budget must give the rank-1 SVD, a
	# binding budget must give a sparse loading.
	sc = BigRiverEssence.spca_component!
	irsv = BigRiverEssence.init_rsv
	Random.seed!(31)
	n, p = 50, 40
	X = randn(n, p);
	Xc = X .- mean(X, dims = 1)
	v0 = irsv(Xc, 1)[:, 1]                          # SVD-based starting direction
	u = Vector{Float64}(undef, n);
	Xv = Vector{Float64}(undef, n)
	Xtu = Vector{Float64}(undef, p);
	s = Vector{Float64}(undef, p)
	vold = Vector{Float64}(undef, p);
	v = copy(v0)

	# Max budget (c = √p): no penalty binds ⇒ pure power iteration ⇒ rank-1 SVD of Xc.
	d = sc(v, Xc, sqrt(p), u, Xv, Xtu, s, vold; niter = 100)
	F = svd(Xc)
	@test isapprox(norm(u), 1.0; atol = tol_julia)
	@test isapprox(norm(v), 1.0; atol = tol_julia)
	@test abs(dot(v, F.V[:, 1])) > 1 - tol_julia      # v → V₁
	@test abs(dot(u, F.U[:, 1])) > 1 - tol_julia      # u → U₁
	@test isapprox(d, F.S[1]; rtol = tol_julia)       # the weight d → σ₁
	@test d > 0

	# A binding budget makes v genuinely sparse.
	v2 = copy(v0)
	sc(v2, Xc, 2.0, u, Xv, Xtu, s, vold; niter = 100)
	@test count(!iszero, v2) < p
end

@testset "internal: spca_component_orth! (orthogonal-score core)" begin
	# The orthogonal-score core: like spca_component! but it deflates each new score
	# against the previously-found ones (U_prev) so the scores come out orthogonal.
	# Check that with no prior scores it matches the ordinary core, and with one prior
	# score the new one is orthogonal to it — the defining property of the orth variant.
	sco = BigRiverEssence.spca_component_orth!
	irsv = BigRiverEssence.init_rsv
	Random.seed!(37)
	n, p = 60, 40
	X = randn(n, p);
	Xc = X .- mean(X, dims = 1)
	Vinit = irsv(Xc, 2)
	u = Vector{Float64}(undef, n);
	uold = Vector{Float64}(undef, n)
	Xv = Vector{Float64}(undef, n);
	Xtu = Vector{Float64}(undef, p)
	s = Vector{Float64}(undef, p);
	vold = Vector{Float64}(undef, p)
	proj = Vector{Float64}(undef, 2)

	# Component 1: U_prev is empty (zero columns) ⇒ no orthogonalization ⇒ reduces to
	# the ordinary core. Capture the resulting score u₁ for the next step.
	U = Matrix{Float64}(undef, n, 2)
	v1 = Vinit[:, 1]
	sco(v1, Xc, sqrt(p), @view(U[:, 1:0]), u, uold, Xv, Xtu, s, vold, proj; niter = 100)
	U[:, 1] .= u
	F = svd(Xc)
	@test abs(dot(U[:, 1], F.U[:, 1])) > 1 - tol_julia    # u₁ → U₁ at max budget

	# Component 2: now U_prev = u₁, so the returned u₂ must be orthogonal to u₁.
	v2 = Vinit[:, 2]
	sco(v2, Xc, sqrt(p), @view(U[:, 1:1]), u, uold, Xv, Xtu, s, vold, proj; niter = 100)
	@test isapprox(norm(u), 1.0; atol = tol_julia)
	@test abs(dot(u, U[:, 1])) < tol_ord              # u₂ ⊥ u₁ — the orth property
end

@testset "matches R PMA::SPC (offline reference fixtures)" begin
	# Cross-language check against R's PMA::SPC, for BOTH variants (orth=FALSE and
	# orth=TRUE), using saved fixtures. Skips with a note if they're absent.
	refdir = joinpath(@__DIR__, "Data", "SPC")
	if !isfile(joinpath(refdir, "X.csv"))
		@info "SPC R-reference fixtures not found; skipping. Run spc.R to create them."
	else
		# Record the PMA version that produced the fixtures (results can shift across
		# versions, so provenance helps when chasing a mismatch).
		smfile = joinpath(refdir, "session_meta.csv")
		if isfile(smfile)
			sm = readdlm(smfile, ',', String; skipstart = 1)
			row = findfirst(==("PMA_version"), sm[:, 1])
			row !== nothing && @info "SPC fixtures generated against PMA $(sm[row, 2])"
		end

		X    = readdlm(joinpath(refdir, "X.csv"), ',', Float64; skipstart = 1)
		v_r  = readdlm(joinpath(refdir, "v_spc.csv"), ',', Float64; skipstart = 1)
		d_r  = vec(readdlm(joinpath(refdir, "d_spc.csv"), ',', Float64; skipstart = 1))
		vo_r = readdlm(joinpath(refdir, "v_spc_orth.csv"), ',', Float64; skipstart = 1)
		do_r = vec(readdlm(joinpath(refdir, "d_spc_orth.csv"), ',', Float64; skipstart = 1))
		meta = readdlm(joinpath(refdir, "meta.csv"), ',', Float64; skipstart = 1)
		n    = Int(meta[1]);
		K    = Int(meta[3]);
		sv   = meta[4]

		# X.csv is RAW (uncentered) — spc/spc_orth center internally, matching what R did.
		m  = spc(X; k = K, c = sv)
		mo = spc_orth(X; k = K, c = sv)
		# spcStructure stores variances, not PMA's d; recover d = √(variance·(n−1)) to compare.
		d_jl  = sqrt.(max.(m.variances, 0) .* (n - 1))
		do_jl = sqrt.(max.(mo.variances, 0) .* (n - 1))

		for k in 1:K
			# orth=FALSE: loadings align up to sign (cross-language ⇒ tol_r), and the
			# selected-variable SET must match exactly.
			@test abs(dot(m.loadings[:, k] ./ norm(m.loadings[:, k]),
				v_r[:, k] ./ norm(v_r[:, k]))) > 1 - tol_r
			@test Set(findall(!iszero, m.loadings[:, k])) == Set(findall(!iszero, v_r[:, k]))
			# orth=TRUE: same two checks for the orthogonal variant.
			@test abs(dot(mo.loadings[:, k] ./ norm(mo.loadings[:, k]),
				vo_r[:, k] ./ norm(vo_r[:, k]))) > 1 - tol_r
			@test Set(findall(!iszero, mo.loadings[:, k])) == Set(findall(!iszero, vo_r[:, k]))
		end
		@test isapprox(d_jl, d_r; rtol = tol_r)            # singular values match (cross-language)
		@test isapprox(do_jl, do_r; rtol = tol_r)
	end
end
