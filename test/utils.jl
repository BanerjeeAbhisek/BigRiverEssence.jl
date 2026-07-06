@testset "internal: SignConsistency_opt! (largest-|entry| made positive)" begin
	_signfix = BigRiverEssence.SignConsistency_opt!

	# column whose largest-magnitude entry is negative ⇒ whole column flips
	V = reshape([1.0, -3.0, 2.0], 3, 1)
	_signfix(V)
	@test V[:, 1] == [-1.0, 3.0, -2.0]            # the −3 (largest |·|) becomes +3
	@test V[argmax(abs.(V[:, 1])), 1] > 0         # leading entry now positive

	# column whose largest-magnitude entry is already positive ⇒ unchanged
	V2 = reshape([1.0, 4.0, -2.0], 3, 1)
	_signfix(V2)
	@test V2[:, 1] == [1.0, 4.0, -2.0]

	# per-column independence: each column flipped on its own pivot
	V3 = [-5.0  2.0
		 1.0 -7.0
		 3.0  4.0]
	_signfix(V3)
	@test V3[:, 1] == [5.0, -1.0, -3.0]           # col1 pivot −5 → flip
	@test V3[:, 2] == [-2.0, 7.0, -4.0]           # col2 pivot −7 → flip
	for j in 1:2
		@test V3[argmax(abs.(V3[:, j])), j] > 0   # each column's pivot is positive
	end

	# returns the same array it mutated (in-place)
	V4 = reshape([2.0, -9.0, 1.0], 3, 1)
	@test _signfix(V4) === V4

	# idempotent: applying twice == applying once (pivot already positive 2nd time)
	V5    = randn(8, 4)
	once  = _signfix(copy(V5))
	twice = _signfix(_signfix(copy(V5)))
	@test once == twice

	# the operation only changes signs, never magnitudes
	V6 = randn(6, 3);
	before = abs.(copy(V6))
	_signfix(V6)
	@test abs.(V6) == before

	# all-zero column ⇒ sign(0)=0 guard leaves it untouched (no NaN/Inf)
	V7 = reshape([0.0, 0.0, 0.0], 3, 1)
	_signfix(V7)
	@test all(iszero, V7)
	@test all(isfinite, V7)
end
