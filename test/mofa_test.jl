# Tests for MOFA preprocessing, initialization, moments, and Gaussian weights.
# Individual weight sweeps are tested; there is no complete fitting loop yet.
# Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl.

@testset "sample groups preserve row order" begin
	indices, labels, rows = BigRiverEssence._mofa_groups(4, nothing)
	@test indices == [1, 1, 1, 1]
	@test labels == [1]
	@test rows == [[1, 2, 3, 4]]
	groups = [:study_b, :study_a, :study_b, :study_a]
	before = copy(groups)
	indices, labels, rows = BigRiverEssence._mofa_groups(4, groups)
	@test indices == [1, 2, 1, 2]
	@test labels == [:study_b, :study_a]
	@test rows == [[1, 3], [2, 4]]
	@test groups == before
	@test BigRiverEssence._mofa_groups(3, [20, 10, 20])[2] == [20, 10]
	@test BigRiverEssence._mofa_groups(3, ["b", "a", "b"])[3] == [[1, 3], [2]]
	@test_throws ArgumentError BigRiverEssence._mofa_groups(0, nothing)
	@test_throws DimensionMismatch BigRiverEssence._mofa_groups(3, [:a, :b])
	for labels in ([missing, :a], [nothing, :a], [NaN, 1.0], [Inf, 1.0])
		@test_throws ArgumentError BigRiverEssence._mofa_groups(2, labels)
	end
end

@testset "Gaussian moments retain uncertainty and reuse storage" begin
	mu = [0.0 -2.0; 3.0 0.5]
	variance = [1.0 0.25; 2.0 4.0]
	q = BigRiverEssence.MofaGaussian(copy(mu), copy(variance), zeros(2, 2))
	buffer = q.second_moment
	@test BigRiverEssence._mofa_gaussian_moments!(q) === q
	@test q.mean == mu
	@test q.variance == variance
	@test q.second_moment == [1.0 4.25; 11.0 4.25]
	@test q.second_moment === buffer
	q.mean[1, 1] = 2.0
	q.variance[2, 2] = 3.0
	BigRiverEssence._mofa_gaussian_moments!(q)
	@test q.second_moment[1, 1] == 5.0
	@test q.second_moment[2, 2] == 3.25
	@test q.second_moment === buffer
	bad = BigRiverEssence.MofaGaussian(zeros(2, 1), ones(1, 1), zeros(2, 1))
	@test_throws DimensionMismatch BigRiverEssence._mofa_gaussian_moments!(bad)
	bad = BigRiverEssence.MofaGaussian(zeros(2, 1), ones(2, 1), zeros(1, 1))
	@test_throws DimensionMismatch BigRiverEssence._mofa_gaussian_moments!(bad)
	for value in (0.0, -1.0, Inf, NaN)
		bad = BigRiverEssence.MofaGaussian(zeros(1, 1), fill(value, 1, 1), zeros(1, 1))
		@test_throws ArgumentError BigRiverEssence._mofa_gaussian_moments!(bad)
	end
	for value in (Inf, NaN, floatmax(Float64))
		bad = BigRiverEssence.MofaGaussian(fill(value, 1, 1), ones(1, 1), zeros(1, 1))
		@test_throws ArgumentError BigRiverEssence._mofa_gaussian_moments!(bad)
	end
end

@testset "Gamma expectation uses rates, not scales" begin
	shape = [1.0 2.0; 3.0 4.0]
	rate = [2.0 4.0; 0.5 1.0]
	shape_before, rate_before = copy(shape), copy(rate)
	expected = zeros(2, 2)
	@test BigRiverEssence._mofa_gamma_mean!(expected, shape, rate) === expected
	@test expected == [0.5 0.5; 6.0 4.0]
	@test shape == shape_before
	@test rate == rate_before
	@test_throws DimensionMismatch BigRiverEssence._mofa_gamma_mean!(zeros(1, 1), shape, rate)
	@test_throws DimensionMismatch BigRiverEssence._mofa_gamma_mean!(expected, shape, ones(1, 1))
	for value in (0.0, -1.0, Inf, NaN)
		@test_throws ArgumentError BigRiverEssence._mofa_gamma_mean!(zeros(1, 1), fill(value, 1, 1), ones(1, 1))
		@test_throws ArgumentError BigRiverEssence._mofa_gamma_mean!(zeros(1, 1), ones(1, 1), fill(value, 1, 1))
	end
	@test_throws ArgumentError BigRiverEssence._mofa_gamma_mean!(zeros(1, 1), fill(floatmax(Float64), 1, 1), fill(eps(Float64), 1, 1))
	@test_throws ArgumentError BigRiverEssence._mofa_gamma_mean!(zeros(1, 1), fill(floatmin(Float64), 1, 1), fill(floatmax(Float64), 1, 1))
end

@testset "initial factor rescaling" begin
	Z = [1.0 10.0; 3.0 30.0; 5.0 20.0]
	@test BigRiverEssence._mofa_rescale_factors!(Z) === Z
	@test Z == [-1.0 -1.0; 0.0 1.0; 1.0 0.0]
	for Z in (zeros(0, 1), zeros(2, 0), ones(2, 1),
		reshape([NaN, 1.0], 2, 1), reshape([Inf, 1.0], 2, 1),
		reshape([-floatmax(Float64), floatmax(Float64)], 2, 1))
		@test_throws ArgumentError BigRiverEssence._mofa_rescale_factors!(Z)
	end
end

@testset "initial state, Python defaults & compact workspace" begin
	X = [1.0 2.0; 3.0 1.0; 2.0 7.0; 5.0 4.0; 6.0 8.0; 4.0 3.0]
	Y = reshape([3.0, 2.0, 6.0, 1.0, 8.0, 5.0], 6, 1)
	d = BigRiverEssence._mofa_preprocess([X, Y]; groups = [:a, :b, :a, :b, :a, :b])
	before = deepcopy(d)
	s, workspace = BigRiverEssence._mofa_initialize(d, 2)
	@test s isa BigRiverEssence.MofaState{Float64}
	@test workspace isa BigRiverEssence.MofaWorkspace{Float64}
	@test size(s.z.mean) == (6, 2)
	@test s.z.variance == ones(6, 2)
	@test s.z.second_moment ≈ s.z.mean .^ 2 .+ 1 atol = tol_ord
	@test vec(minimum(s.z.mean, dims = 1)) == [-1.0, -1.0]
	@test vec(maximum(s.z.mean, dims = 1)) == [1.0, 1.0]
	@test length(s.w) == length(s.tau_mean) == 2
	for m in 1:2
		p = size(d.data[m], 2)
		@test s.w[m].mean == zeros(p, 2)
		@test s.w[m].variance == ones(p, 2)
		@test s.w[m].second_moment == ones(p, 2)
		@test s.tau_shape[m] == ones(2, p)
		@test s.tau_rate[m] == ones(2, p)
		@test s.tau_mean[m] == ones(2, p)
		@test d.data[m] == before.data[m]
		@test d.missing_mask[m] == before.missing_mask[m]
	end
	@test s.tau_prior_shape == s.tau_prior_rate == 1e-3
	@test workspace.sample_buffer == zeros(6)
	@test workspace.feature_buffer == zeros(2)
	@test workspace.factor_buffer == zeros(2)
	# Noise state is G×p per view, not an expanded n×p precision matrix.
	@test size(s.tau_mean[1]) == (2, 2)
	# Every independently updated quantity has its own storage.
	@test s.z.mean !== s.z.variance !== s.z.second_moment
	s.w[1].variance[1, 1] = 9.0
	s.tau_shape[1][1, 1] = 4.0
	workspace.sample_buffer[1] = 99.0
	@test s.w[1].second_moment[1, 1] == 1.0
	@test s.w[2].variance[1, 1] == 1.0
	@test s.tau_rate[1][1, 1] == s.tau_mean[1][1, 1] == 1.0
	@test d.data[1] == before.data[1]
	@test s.z.mean[1, 1] != 99.0
end

@testset "PCA initialization: whitening, imputation & input preservation" begin
	X = [1.0 NaN; 3.0 6.0; 8.0 2.0; 2.0 9.0; 10.0 5.0; 4.0 8.0]
	Y = reshape([2.0, 1.0, 8.0, 3.0, 7.0, 4.0], 6, 1)
	for center in (false, true)
		d = BigRiverEssence._mofa_preprocess([X, Y]; center_groups = center)
		before = deepcopy(d)
		Z = BigRiverEssence._mofa_pca_factors(d, 2)
		# Independent reference: explicitly impute and whiten the PCA scores,
		# then apply the Python dense-factor min/max transformation.
		imputed = hcat(d.data...)
		imputed[1, 2] = mean(d.data[1][2:end, 2])
		centered = imputed .- mean(imputed, dims = 1)
		F = svd(centered; full = false)
		scores = centered * F.V[:, 1:2]
		for j in 1:2
			pivot = argmax(abs.(F.V[:, j]))
			F.V[pivot, j] < 0 && (scores[:, j] .*= -1)
			scores[:, j] ./= F.S[j] / sqrt(size(centered, 1) - 1)
		end
		expected = 2 .* (scores .- minimum(scores, dims = 1)) ./
			(maximum(scores, dims = 1) .- minimum(scores, dims = 1)) .- 1
		@test Z ≈ expected atol = tol_ord
		for m in eachindex(d.data)
			@test d.data[m] == before.data[m]
			@test d.missing_mask[m] == before.missing_mask[m]
		end
		@test d.data[1][1, 2] == 0.0
		@test d.missing_mask[1][1, 2]
		s, _ = BigRiverEssence._mofa_initialize(d, 2)
		@test s.z.mean ≈ expected atol = tol_ord
	end
	# Both tall and wide concatenated matrices use the same PCA convention.
	Random.seed!(52)
	for X in (randn(12, 4), randn(4, 12))
		d = BigRiverEssence._mofa_preprocess([X])
		Z = BigRiverEssence._mofa_pca_factors(d, 2)
		@test size(Z) == (size(X, 1), 2)
		@test all(isfinite, Z)
		@test minimum(Z) ≈ -1.0 atol = tol_ord
		@test maximum(Z) ≈ 1.0 atol = tol_ord
	end
end

@testset "random initialization & RNG behavior" begin
	d = BigRiverEssence._mofa_preprocess([reshape(collect(1.0:6.0), 6, 1)])
	Random.seed!(12)
	draws = randn(6, 2)
	expected = 2 .* (draws .- minimum(draws, dims = 1)) ./
		(maximum(draws, dims = 1) .- minimum(draws, dims = 1)) .- 1
	Random.seed!(12)
	s1, _ = BigRiverEssence._mofa_initialize(d, 2; init = :random)
	@test s1.z.mean ≈ expected atol = tol_ord
	Random.seed!(12)
	s2, _ = BigRiverEssence._mofa_initialize(d, 2; init = :random)
	@test s1.z.mean == s2.z.mean
	@test s1.z.second_moment == s2.z.second_moment
	s3, _ = BigRiverEssence._mofa_initialize(d, 2; init = :random)
	@test s3.z.mean != s2.z.mean
	# An explicitly supplied RNG must not consume the global RNG sequence.
	Random.seed!(21)
	next_global = rand()
	Random.seed!(21)
	s4, _ = BigRiverEssence._mofa_initialize(d, 2; init = :random, rng = MersenneTwister(12))
	@test rand() == next_global
	@test size(s4.z.mean) == (6, 2)
	# PCA initialization does not draw random numbers.
	Random.seed!(21)
	BigRiverEssence._mofa_initialize(d, 1)
	@test rand() == next_global
end

@testset "custom initialization copies means without rescaling" begin
	d = BigRiverEssence._mofa_preprocess([ones(3, 2), ones(3, 1)])
	Z = [2.0 -3.0; 4.0 8.0; -1.0 0.0]
	W = [[1.0 2.0; -3.0 4.0], reshape([5.0, 6.0], 1, 2)]
	s, workspace = BigRiverEssence._mofa_initialize(d, 2; init = :custom, z_init = Z, w_init = W)
	@test s.z.mean == Z
	@test s.z.second_moment == Z .^ 2 .+ 1
	for m in eachindex(W)
		@test s.w[m].mean == W[m]
		@test s.w[m].second_moment == W[m] .^ 2 .+ 1
	end
	s.z.mean[1, 1] = 100.0
	s.w[1].mean[1, 1] = 200.0
	@test Z[1, 1] == 2.0
	@test W[1][1, 1] == 1.0
	@test workspace.sample_buffer == zeros(3)
	s, _ = BigRiverEssence._mofa_initialize(d, 2; init = :custom, z_init = Float32.(Z))
	@test s.z.mean isa Matrix{Float64}
	@test s.z.mean == Z
	@test all(W -> all(iszero, W.mean), s.w)
	# One sample is valid for custom factors; PCA/random cannot rescale it.
	one = BigRiverEssence._mofa_preprocess([ones(1, 1)])
	s, _ = BigRiverEssence._mofa_initialize(one, 1; init = :custom, z_init = fill(2.0, 1, 1))
	@test s.z.mean == fill(2.0, 1, 1)
	@test s.z.second_moment == fill(5.0, 1, 1)
end

@testset "initialization argument validation & degenerate PCA" begin
	d = BigRiverEssence._mofa_preprocess([[1.0 2.0; 3.0 1.0; 2.0 5.0]])
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 0)
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, -1)
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 1; init = :other)
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 1; init = :custom)
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 1; z_init = ones(3, 1))
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 1; w_init = [ones(2, 1)])
	@test_throws DimensionMismatch BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(2, 1))
	for Z in (fill(Inf, 3, 1), fill(NaN, 3, 1), fill("x", 3, 1), fill(1im, 3, 1))
		@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = Z)
	end
	for W in ([], [ones(2, 1), ones(2, 1)], ones(2, 1), ["x"], [fill(Inf, 2, 1)])
		@test_throws ArgumentError BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(3, 1), w_init = W)
	end
	@test_throws DimensionMismatch BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = ones(3, 1), w_init = [ones(3, 1)])
	for k in (0, -1, 3)
		@test_throws ArgumentError BigRiverEssence._mofa_pca_factors(d, k)
	end
	rank_one = BigRiverEssence._mofa_preprocess([[1.0 2.0; 2.0 4.0; 3.0 6.0]])
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(rank_one, 2)
	constant = BigRiverEssence._mofa_preprocess([ones(3, 2)])
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(constant, 1)
	one = BigRiverEssence._mofa_preprocess([ones(1, 1)])
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(one, 1)
	@test_throws ArgumentError BigRiverEssence._mofa_initialize(one, 1; init = :random)
	extreme = BigRiverEssence._mofa_preprocess([reshape([floatmax(Float64), -floatmax(Float64)], 2, 1)]; center_groups = false)
	@test_throws ArgumentError BigRiverEssence._mofa_pca_factors(extreme, 1)
end

@testset "Gaussian weights: hand-calculated one-factor update" begin
	X = reshape([1.0, 3.0, 5.0], 3, 1)
	d = BigRiverEssence._mofa_preprocess([X]; groups = [:b, :a, :b], center_groups = false)
	Z = reshape([1.0, 2.0, -1.0], 3, 1)
	# A huge old value must not cause cancellation when excluding this coordinate.
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = Z, w_init = [fill(1e20, 1, 1)])
	s.z.variance[:, 1] .= [0.5, 2.0, 1.0]
	BigRiverEssence._mofa_gaussian_moments!(s.z)
	s.tau_mean[1][:, 1] .= [2.0, 4.0]
	before = deepcopy(s)
	data_before = deepcopy(d)
	mean_buffer, variance_buffer, second_buffer = s.w[1].mean, s.w[1].variance, s.w[1].second_moment
	workspace.feature_buffer .= 123.0
	@test BigRiverEssence._mofa_update_weights!(d, s, workspace) === s
	# Precision = 1 + 2*1.5 + 4*6 + 2*2 = 32.
	# Numerator = 2*1*1 + 4*2*3 + 2*(-1)*5 = 16.
	@test s.w[1].variance[1, 1] ≈ 1 / 32 atol = tol_ord
	@test s.w[1].mean[1, 1] ≈ 0.5 atol = tol_ord
	@test s.w[1].second_moment[1, 1] ≈ 0.25 + 1 / 32 atol = tol_ord
	@test s.w[1].variance[1, 1] != 1 / 21  # using squared means would be wrong
	@test s.w[1].mean === mean_buffer
	@test s.w[1].variance === variance_buffer
	@test s.w[1].second_moment === second_buffer
	@test s.z.mean == before.z.mean
	@test s.z.variance == before.z.variance
	@test s.z.second_moment == before.z.second_moment
	@test s.tau_mean == before.tau_mean
	@test s.tau_shape == before.tau_shape
	@test s.tau_rate == before.tau_rate
	@test d.data == data_before.data
	@test d.missing_mask == data_before.missing_mask
	@test workspace.feature_buffer == fill(123.0, 1)
end

@testset "Gaussian weights: multiview sequential scalar reference" begin
	Random.seed!(31)
	n, k = 7, 3
	views = [randn(n, 4), randn(n, 2)]
	views[1][2, 3] = NaN
	views[1][5, 1] = NaN
	views[2][1, :] .= NaN
	d = BigRiverEssence._mofa_preprocess(views; groups = [:b, :a, :b, :a, :b, :a, :b])
	s, workspace = BigRiverEssence._mofa_initialize(d, k; init = :custom,
		z_init = randn(n, k), w_init = [randn(4, k), randn(2, k)])
	s.z.variance .= 0.1 .+ rand(n, k)
	BigRiverEssence._mofa_gaussian_moments!(s.z)
	for m in eachindex(views)
		s.tau_mean[m] .= 0.2 .+ 3 .* rand(size(s.tau_mean[m])...)
	end
	expected = deepcopy(s)
	frozen = deepcopy(s)
	for sweep in 1:3
		# Deliberately simple reference: component-major, explicit exclusion sums,
		# and no reused prediction buffer. Later components see earlier updates.
		for m in eachindex(d.data), component in 1:k, j in axes(d.data[m], 2)
			precision, numerator = 1.0, 0.0
			for i in 1:n
				d.missing_mask[m][i, j] && continue
				prediction = sum(expected.z.mean[i, other] * expected.w[m].mean[j, other]
					for other in 1:k if other != component)
				noise = expected.tau_mean[m][d.group_indices[i], j]
				precision += noise * expected.z.second_moment[i, component]
				numerator += noise * expected.z.mean[i, component] * (d.data[m][i, j] - prediction)
			end
			expected.w[m].variance[j, component] = 1 / precision
			expected.w[m].mean[j, component] = numerator / precision
		end
		BigRiverEssence._mofa_update_weights!(d, s, workspace)
		for m in eachindex(d.data)
			@test s.w[m].mean ≈ expected.w[m].mean atol = tol_ord
			@test s.w[m].variance ≈ expected.w[m].variance atol = tol_ord
			@test s.w[m].second_moment ≈ s.w[m].mean .^ 2 .+ s.w[m].variance atol = tol_ord
			@test all(x -> 0 < x <= 1, s.w[m].variance)
		end
	end
	@test s.z.mean == frozen.z.mean
	@test s.z.second_moment == frozen.z.second_moment
	@test s.tau_mean == frozen.tau_mean
end

@testset "Gaussian weights use updated coordinates, not simultaneous updates" begin
	d = BigRiverEssence._mofa_preprocess([reshape([1.0, 2.0], 2, 1)]; center_groups = false)
	Z = [1.0 1.0; 1.0 2.0]
	s, workspace = BigRiverEssence._mofa_initialize(d, 2; init = :custom, z_init = Z)
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	@test s.w[1].mean[1, 1] ≈ 3 / 5 atol = tol_ord
	@test s.w[1].variance[1, 1] ≈ 1 / 5 atol = tol_ord
	@test s.w[1].mean[1, 2] ≈ (5 - 3 * (3 / 5)) / 8 atol = tol_ord
	@test s.w[1].variance[1, 2] ≈ 1 / 8 atol = tol_ord
	@test !isapprox(s.w[1].mean[1, 2], 5 / 8; atol = tol_ord)
end

@testset "Gaussian weights exclude missing entries and unobserved groups" begin
	X = reshape([1.0, NaN, 3.0, NaN], 4, 1)
	d = BigRiverEssence._mofa_preprocess([X]; groups = [:a, :b, :a, :b], center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(4, 1))
	s.tau_mean[1][:, 1] .= [2.0, 1e100]
	# These placeholders are deliberately invalid: the mask, not their value,
	# must keep them out of the likelihood and the update.
	d.data[1][2, 1] = Inf
	d.data[1][4, 1] = NaN
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	@test s.w[1].variance[1, 1] ≈ 1 / 9 atol = tol_ord
	@test s.w[1].mean[1, 1] ≈ 8 / 9 atol = tol_ord
	# With no observations at all, the conditional posterior equals the prior.
	d.missing_mask[1] .= true
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	@test s.w[1].mean == zeros(1, 1)
	@test s.w[1].variance == ones(1, 1)
	@test s.w[1].second_moment == ones(1, 1)
	# Zero factor means do not remove factor uncertainty from the denominator.
	d = BigRiverEssence._mofa_preprocess([ones(3, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = zeros(3, 1))
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	@test s.w[1].mean == zeros(1, 1)
	@test s.w[1].variance == fill(1 / 4, 1, 1)
end

@testset "Gaussian weight update validation" begin
	d = BigRiverEssence._mofa_preprocess([[1.0 2.0; 3.0 4.0], ones(2, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(2, 1))
	for field in (:sample_buffer, :factor_buffer)
		bad_workspace = deepcopy(workspace)
		empty!(getproperty(bad_workspace, field))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(d, s, bad_workspace)
	end
	for field in (:w, :tau_mean)
		bad = deepcopy(s)
		pop!(getproperty(bad, field))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(d, bad, workspace)
	end
	bad_data = deepcopy(d)
	pop!(bad_data.missing_mask)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(bad_data, s, workspace)
	bad_data = deepcopy(d)
	pop!(bad_data.group_indices)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(bad_data, s, workspace)
	for group in (0, 2)
		bad_data = deepcopy(d)
		bad_data.group_indices[1] = group
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(bad_data, s, workspace)
	end
	for field in (:mean, :variance, :second_moment)
		bad = deepcopy(s)
		bad.w[2] = BigRiverEssence.MofaGaussian(
			field === :mean ? zeros(2, 1) : zeros(1, 1),
			field === :variance ? ones(2, 1) : ones(1, 1),
			field === :second_moment ? ones(2, 1) : ones(1, 1))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(d, bad, workspace)
		@test bad.w[1].mean == s.w[1].mean  # validate all views before updating
	end
	bad = deepcopy(s)
	bad.tau_mean[2] = ones(2, 1)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(d, bad, workspace)
	bad_data = deepcopy(d)
	bad_data.data[2] = ones(3, 1)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(bad_data, s, workspace)
	bad_data = deepcopy(d)
	bad_data.missing_mask[2] = falses(2, 2)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(bad_data, s, workspace)
	bad_z = BigRiverEssence.MofaGaussian(ones(2, 1), ones(1, 1), ones(2, 1))
	bad = BigRiverEssence.MofaState(bad_z, s.w, s.tau_shape, s.tau_rate, s.tau_mean, 1e-3, 1e-3)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(d, bad, workspace)
	for value in (0.0, -1.0, Inf, NaN)
		bad = deepcopy(s)
		bad.tau_mean[1][1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, bad, workspace)
		bad = deepcopy(s)
		bad.z.second_moment[1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, bad, workspace)
	end
	for value in (Inf, NaN)
		bad = deepcopy(s)
		bad.z.mean[1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, bad, workspace)
		bad = deepcopy(s)
		bad.w[1].mean[1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, bad, workspace)
		bad_data = deepcopy(d)
		bad_data.data[1][1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(bad_data, deepcopy(s), workspace)
	end
	bad = deepcopy(s)
	bad.tau_mean[1][1, 1] = floatmax(Float64)
	@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, bad, workspace)
	bad_data = deepcopy(d)
	bad_data.data[1][1, 1] = 1e200
	@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(bad_data, deepcopy(s), workspace)
	for shape in ((0, 1), (2, 0))
		bad_z = BigRiverEssence.MofaGaussian(zeros(shape...), ones(shape...), ones(shape...))
		bad = BigRiverEssence.MofaState(bad_z, s.w, s.tau_shape, s.tau_rate, s.tau_mean, 1e-3, 1e-3)
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, bad, workspace)
	end
	for field in (:data, :group_labels)
		bad_data = deepcopy(d)
		empty!(getproperty(bad_data, field))
		@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(bad_data, s, workspace)
	end
end

@testset "output structure & default centering" begin
	X = [1.0 2.0; 3.0 6.0; 5.0 10.0]
	Y = reshape([2.0, 4.0, 9.0], 3, 1)
	views = [X, Y]
	before = deepcopy(views)
	d = BigRiverEssence._mofa_preprocess(views)
	@test d isa BigRiverEssence.MofaData{Float64,Int}
	@test length(d.data) == length(d.missing_mask) == 2
	@test size(d.data[1]) == (3, 2)
	@test size(d.data[2]) == (3, 1)
	@test d.group_indices == [1, 1, 1]
	@test d.group_labels == [1]
	@test d.group_rows == [[1, 2, 3]]
	@test d.scales == ones(1, 2)
	for m in eachindex(views)
		@test d.data[m] ≈ views[m] .- mean(views[m], dims = 1) atol = tol_ord
		@test d.means[m] ≈ mean(views[m], dims = 1) atol = tol_ord
		@test d.observed_counts[m] == fill(3, 1, size(views[m], 2))
		@test d.missing_mask[m] isa BitMatrix
		@test !any(d.missing_mask[m])
		@test d.data[m] !== views[m]
	end
	@test views == before
end

@testset "group-wise centering with interleaved rows" begin
	X = [1.0 10.0; 20.0 100.0; 5.0 18.0; 28.0 116.0]
	d = BigRiverEssence._mofa_preprocess([X]; groups = [:b, :a, :b, :a])
	@test d.group_labels == [:b, :a]
	@test d.group_indices == [1, 2, 1, 2]
	@test d.means[1] == [3.0 14.0; 24.0 108.0]
	@test d.data[1] == [-2.0 -4.0; -4.0 -8.0; 2.0 4.0; 4.0 8.0]
	@test d.observed_counts[1] == [2 2; 2 2]
end

@testset "missing values are masked, not imputed observations" begin
	X = Union{Missing,Float64}[1.0 missing; 3.0 NaN; 5.0 8.0; missing 12.0]
	before = copy(X)
	d = BigRiverEssence._mofa_preprocess([X]; groups = [1, 1, 2, 2])
	@test d.missing_mask[1] == Bool[0 1; 0 1; 0 0; 1 0]
	@test d.observed_counts[1] == [2 0; 1 2]
	@test d.means[1] == [2.0 0.0; 5.0 10.0]
	@test d.data[1] == [-1.0 0.0; 1.0 0.0; 0.0 -2.0; 0.0 2.0]
	@test all(isfinite, d.data[1])
	@test all(iszero, d.data[1][d.missing_mask[1]])
	@test isequal(X, before)
	# A sample can have an entire missing view; other views retain its row.
	X = [NaN NaN; 1.0 2.0; 3.0 4.0]
	Y = reshape([5.0, 6.0, 7.0], 3, 1)
	d = BigRiverEssence._mofa_preprocess([X, Y]; groups = [:a, :b, :b])
	@test all(d.missing_mask[1][1, :])
	@test d.means[1][1, :] == [0.0, 0.0]
	@test d.observed_counts[1][1, :] == [0, 0]
	@test d.data[1][1, :] == [0.0, 0.0]
	@test size(d.data[2], 1) == 3
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess([X, Y];
		groups = [:a, :b, :b], scale_groups = true)
end

@testset "population scaling helper" begin
	Y = [1.0 3.0; 5.0 0.0; 9.0 11.0]
	mask = falses(3, 2)
	mask[2, 2] = true
	original = copy(Y)
	mask_before = copy(mask)
	values = [1.0, 5.0, 3.0]
	scale = BigRiverEssence._mofa_scale!(Y, mask, [1, 2])
	@test scale ≈ std(values; corrected = false) atol = tol_ord
	@test Y[1:2, :] ≈ original[1:2, :] ./ scale atol = tol_ord
	@test Y[3, :] == original[3, :]
	@test Y[2, 2] == 0.0
	@test mask == mask_before
	@test_throws ArgumentError BigRiverEssence._mofa_scale!(zeros(2, 2), trues(2, 2), [1, 2])
	@test_throws ArgumentError BigRiverEssence._mofa_scale!(ones(2, 2), falses(2, 2), [1, 2])
	@test_throws ArgumentError BigRiverEssence._mofa_scale!(fill(Inf, 2, 2), falses(2, 2), [1, 2])
end

@testset "all preprocessing combinations & original-unit reconstruction" begin
	X = [1.0 8.0; 4.0 NaN; 7.0 20.0; 11.0 28.0; 14.0 44.0]
	Y = reshape([3.0, NaN, 9.0, 12.0, 18.0], 5, 1)
	views = [X, Y]
	before = deepcopy(views)
	groups = [:a, :b, :a, :b, :b]
	for center in (false, true), view_scale in (false, true), group_scale in (false, true)
		d = BigRiverEssence._mofa_preprocess(views; groups = groups,
			center_groups = center, scale_views = view_scale, scale_groups = group_scale)
		for m in eachindex(views)
			# Independent, deliberately simple reference: retain NaNs, use explicit
			# slices, and compute population std with Statistics.std.
			expected = copy(views[m])
			expected_means = zeros(2, size(expected, 2))
			if center
				for (g, rows) in enumerate(d.group_rows), j in axes(expected, 2)
					values = filter(!isnan, expected[rows, j])
					if !isempty(values)
						expected_means[g, j] = mean(values)
						expected[rows, j] .-= expected_means[g, j]
					end
				end
			end
			expected_scales = ones(2)
			if view_scale
				s = std(filter(!isnan, vec(expected)); corrected = false)
				expected ./= s
				expected_scales .*= s
			end
			if group_scale
				for (g, rows) in enumerate(d.group_rows)
					s = std(filter(!isnan, vec(expected[rows, :])); corrected = false)
					expected[rows, :] ./= s
					expected_scales[g] *= s
				end
			end
			observed = .!d.missing_mask[m]
			@test d.data[m][observed] ≈ expected[observed] atol = tol_ord
			@test d.means[m] ≈ expected_means atol = tol_ord
			@test d.scales[:, m] ≈ expected_scales atol = tol_ord
			@test all(iszero, d.data[m][d.missing_mask[m]])
			for j in axes(views[m], 2), i in axes(views[m], 1)
				observed[i, j] || continue
				g = d.group_indices[i]
				@test d.data[m][i, j] * d.scales[g, m] + d.means[m][g, j] ≈
					views[m][i, j] atol = tol_ord
			end
		end
	end
	@test isequal(views, before)
end

@testset "scaling is block-wise, not per-feature" begin
	X = [1.0 10.0; 3.0 30.0; 5.0 50.0]
	d = BigRiverEssence._mofa_preprocess([X]; scale_views = true)
	@test std(vec(d.data[1]); corrected = false) ≈ 1.0 atol = tol_ord
	@test std(d.data[1][:, 2]) / std(d.data[1][:, 1]) ≈ 10.0 atol = tol_ord
	d = BigRiverEssence._mofa_preprocess([X, 10 .* X]; scale_views = true)
	@test d.data[1] ≈ d.data[2] atol = tol_ord
	@test d.scales[1, 2] ≈ 10 * d.scales[1, 1] atol = tol_ord
end

@testset "input conversion, constants & independent copies" begin
	for X in ([1 2; 3 4], Float32[1 2; 3 4], view([1.0 2.0; 3.0 4.0], :, :))
		d = BigRiverEssence._mofa_preprocess([X]; center_groups = false)
		@test d.data[1] isa Matrix{Float64}
		@test d.data[1] == X
		@test d.means[1] == zeros(1, 2)
	end
	X = [1.0 2.0; 3.0 4.0]
	d = BigRiverEssence._mofa_preprocess([X, X]; center_groups = false)
	d.data[1][1, 1] = 100.0
	@test d.data[2][1, 1] == X[1, 1] == 1.0
	constant = BigRiverEssence._mofa_preprocess([ones(3, 2)])
	@test all(iszero, constant.data[1])
	@test constant.means[1] == ones(1, 2)
	@test constant.observed_counts[1] == fill(3, 1, 2)
	single = BigRiverEssence._mofa_preprocess([reshape([4.0], 1, 1)])
	@test single.data[1] == zeros(1, 1)
	@test single.means[1] == fill(4.0, 1, 1)
end

@testset "argument validation & failure does not mutate input" begin
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess(Matrix{Float64}[])
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess([[1.0, 2.0]])
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess([zeros(0, 2)])
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess([zeros(2, 0)])
	@test_throws DimensionMismatch BigRiverEssence._mofa_preprocess([zeros(2, 2), zeros(3, 1)])
	@test_throws DimensionMismatch BigRiverEssence._mofa_preprocess([zeros(2, 2)]; groups = [1])
	for X in (fill(Inf, 2, 2), fill(-Inf, 2, 2), fill("x", 2, 2),
		fill(1 + 2im, 2, 2), fill(missing, 2, 2), fill(NaN, 2, 2),
		[1.0 NaN; 2.0 NaN], fill(big"1e400", 2, 2))
		@test_throws ArgumentError BigRiverEssence._mofa_preprocess([X])
	end
	for center in (false, true)
		@test_throws ArgumentError BigRiverEssence._mofa_preprocess([ones(3, 2)];
			center_groups = center, scale_views = true)
		@test_throws ArgumentError BigRiverEssence._mofa_preprocess([ones(3, 2)];
			center_groups = center, scale_groups = true)
	end
	# Finite inputs can still overflow during arithmetic; reject unusable output.
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess([reshape([floatmax(Float64), -floatmax(Float64)], 2, 1)])
	views = [[1.0 2.0; 3.0 4.0], fill(Inf, 2, 1)]
	before = deepcopy(views)
	@test_throws ArgumentError BigRiverEssence._mofa_preprocess(views)
	@test views == before
end
