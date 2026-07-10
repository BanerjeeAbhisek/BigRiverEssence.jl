@testset "internal: _sign_consistency_opt! (largest-|entry| made positive)" begin

	# column whose largest-magnitude entry is negative ⇒ whole column flips
	V = reshape([1.0, -3.0, 2.0], 3, 1)
	BigRiverEssence._sign_consistency_opt!(V)
	@test V[:, 1] == [-1.0, 3.0, -2.0]            # the −3 (largest |·|) becomes +3
	@test V[argmax(abs.(V[:, 1])), 1] > 0         # leading entry now positive

	# column whose largest-magnitude entry is already positive ⇒ unchanged
	V2 = reshape([1.0, 4.0, -2.0], 3, 1)
	BigRiverEssence._sign_consistency_opt!(V2)
	@test V2[:, 1] == [1.0, 4.0, -2.0]

	# per-column independence: each column flipped on its own pivot
	V3 = [-5.0  2.0
		 1.0 -7.0
		 3.0  4.0]
	BigRiverEssence._sign_consistency_opt!(V3)
	@test V3[:, 1] == [5.0, -1.0, -3.0]           # col1 pivot −5 → flip
	@test V3[:, 2] == [-2.0, 7.0, -4.0]           # col2 pivot −7 → flip
	for j in 1:2
		@test V3[argmax(abs.(V3[:, j])), j] > 0   # each column's pivot is positive
	end

	# returns the same array it mutated (in-place)
	V4 = reshape([2.0, -9.0, 1.0], 3, 1)
	@test BigRiverEssence._sign_consistency_opt!(V4) === V4

	# idempotent: applying twice == applying once (pivot already positive 2nd time)
	V5    = randn(8, 4)
	once  = BigRiverEssence._sign_consistency_opt!(copy(V5))
	twice = BigRiverEssence._sign_consistency_opt!(BigRiverEssence._sign_consistency_opt!(copy(V5)))
	@test once == twice

	# the operation only changes signs, never magnitudes
	V6 = randn(6, 3);
	before = abs.(copy(V6))
	BigRiverEssence._sign_consistency_opt!(V6)
	@test abs.(V6) == before

	# all-zero column ⇒ sign(0)=0 guard leaves it untouched (no NaN/Inf)
	V7 = reshape([0.0, 0.0, 0.0], 3, 1)
	BigRiverEssence._sign_consistency_opt!(V7)
	@test all(iszero, V7)
	@test all(isfinite, V7)
end

@testset "internal: vip (Variable Importance in Projection)" begin
	# VIP scores summarize each variable's contribution across components. The defining
	# properties: correct shape, the normalization (mean squared VIP = 1), nonnegativity,
	# and that it works on both plsda and splsda fits.

	Random.seed!(1)
	n, p, ncomp = 60, 40, 3
	y = repeat(["A", "B", "C"], inner = 20)
	X = randn(n, p)

	# --- on a plsda (dense) fit ---
	m = BigRiverEssence.plsda(X, y, ncomp)
	V = BigRiverEssence.vip(m)

	@test size(V) == (p, ncomp)                       # one row per variable, one col per component

	# The core normalization: the mean of the SQUARED VIP scores (last column) is 1.
	# This is what makes VIP > 1 the "above-average importance" threshold.
	@test isapprox(mean(V[:, end] .^ 2), 1.0; atol = tol_ord)

	@test all(V .>= 0)                                # VIP scores are nonnegative (sqrt of nonneg)
	@test all(isfinite, V)                            # no NaN/Inf

	# --- works on an splsda (sparse) fit too (same fields, same computation) ---
	ms = BigRiverEssence.splsda(X, y, ncomp, [15, 15, 15])
	Vs = BigRiverEssence.vip(ms)
	@test size(Vs) == (p, ncomp)
	@test isapprox(mean(Vs[:, end] .^ 2), 1.0; atol = tol_ord)   # normalization holds for sparse too
	@test all(Vs .>= 0)

	# --- signal variables score higher than noise (VIP actually ranks importance) ---
	# Plant class-specific means in the first 5 variables; the rest are noise.
	Random.seed!(2)
	classes = ["A", "B", "C"]; n_per = 20
	yy = repeat(classes, inner = n_per); nn = length(yy); pp = 60
	XX = randn(nn, pp) .* 0.5
	for (ci, cls) in enumerate(classes)
		XX[findall(==(cls), yy), 1:5] .+= ci * 2.0
	end
	msig = BigRiverEssence.plsda(XX, yy, 2)
	Vsig = BigRiverEssence.vip(msig)[:, end]
	@test mean(Vsig[1:5]) > mean(Vsig[6:end])         # signal vars have higher VIP than noise

	# --- single-component case: VIP[:,1] = sqrt(p)·|loading| (the h=1 branch) ---
	m1 = BigRiverEssence.plsda(X, y, 1)
	V1 = BigRiverEssence.vip(m1)
	@test size(V1) == (p, 1)
	@test isapprox(mean(V1[:, 1] .^ 2), 1.0; atol = tol_ord)   # normalization still holds at ncomp=1

	# --- the number of above-threshold variables is sensible (not all, not none) ---
	nabove = count(>(1.0), V[:, end])
	@test 0 < nabove < p                              # some variables clear the bar, not all
end