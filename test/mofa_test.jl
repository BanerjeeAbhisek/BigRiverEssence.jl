# Tests for MOFA preprocessing, initialization, Gaussian/sparse updates,
# complete fitting, diagnostics, and Python reference agreement.
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
	@test workspace.coordinate_sums === nothing
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

@testset "prediction kernel: small-rank SIMD and larger-rank BLAS" begin
	Random.seed!(73)
	for n in (1, 7, 60), k in (1, 2, 16, 17, 25)
		basis, coefficients = randn(n, k), randn(k)
		k > 1 && (coefficients[2] = 0.0)
		before_basis, before_coefficients = copy(basis), copy(coefficients)
		expected = basis * coefficients
		storage = fill(NaN, n + 2)
		prediction = @view storage[2:n + 1]
		@test BigRiverEssence._mofa_predict!(prediction, basis, coefficients) === prediction
		@test prediction ≈ expected atol = tol_ord rtol = tol_ord
		@test isnan(storage[1]) && isnan(storage[end])
		@test basis == before_basis && coefficients == before_coefficients
		fill!(coefficients, 0.0)
		@test BigRiverEssence._mofa_predict!(prediction, basis, coefficients) == zeros(n)
	end
	# Exclusion must preserve small signals without subtract/add-back cancellation.
	basis, coefficients = [1e20 1.0; -1e20 2.0], [0.0, 1.0]
	@test BigRiverEssence._mofa_predict!(zeros(2), basis, coefficients) == [1.0, 2.0]
	@test BigRiverEssence._mofa_predict!(ones(3), zeros(3, 0), Float64[]) == zeros(3)
	@test BigRiverEssence._mofa_predict!(Float64[], zeros(0, 2), ones(2)) == Float64[]
	prediction = fill(123.0, 3)
	@test_throws DimensionMismatch BigRiverEssence._mofa_predict!(prediction, ones(2, 2), ones(2))
	@test prediction == fill(123.0, 3)
	@test_throws DimensionMismatch BigRiverEssence._mofa_predict!(prediction, ones(3, 2), ones(1))
	@test prediction == fill(123.0, 3)
end

@testset "coordinate suffix sums and offset storage" begin
	Random.seed!(74)
	for n in (1, 5), k in (1, 3, 8, 17)
		basis, coefficients = randn(n, k), randn(k)
		before_basis, before_coefficients = copy(basis), copy(coefficients)
		sums = fill(123.0, n + 4, k + 1)
		@test BigRiverEssence._mofa_coordinate_sums!(sums, basis, coefficients, 2) === sums
		@test sums[3:n + 2, 1] == zeros(n)
		@test sums[3:n + 2, end] == zeros(n)
		for component in 1:k - 1
			expected = basis[:, component + 1:end] * coefficients[component + 1:end]
			@test sums[3:n + 2, component + 1] ≈ expected atol = tol_ord rtol = tol_ord
		end
		@test sums[1:2, :] == fill(123.0, 2, k + 1)
		@test sums[n + 3:end, :] == fill(123.0, 2, k + 1)
		@test basis == before_basis && coefficients == before_coefficients
	end
	# A huge original first coordinate must never enter its exclusion prediction.
	sums = zeros(2, 4)
	BigRiverEssence._mofa_coordinate_sums!(sums, ones(2, 3), [1e20, 1.0, 2.0])
	@test sums[:, 2] == [3.0, 3.0]
	@test_throws ArgumentError BigRiverEssence._mofa_coordinate_sums!(zeros(2, 3), ones(2, 2), ones(2), -1)
	@test_throws DimensionMismatch BigRiverEssence._mofa_coordinate_sums!(zeros(1, 3), ones(2, 2), ones(2))
	@test_throws DimensionMismatch BigRiverEssence._mofa_coordinate_sums!(zeros(2, 2), ones(2, 2), ones(2))
	@test_throws DimensionMismatch BigRiverEssence._mofa_coordinate_sums!(zeros(2, 3), ones(2, 2), ones(1))
end

@testset "cached and direct sweeps agree across all ARD/sparsity combinations" begin
	Random.seed!(75)
	views = [randn(9, 6), randn(9, 5)]
	views[1][2, 3] = NaN
	views[2][4, :] .= NaN
	for k in (8, 17), aw in (false, true), az in (false, true), sw in (false, true)
		data = BigRiverEssence._mofa_preprocess(views; groups = [:a, :b, :a, :b, :a, :b, :a, :b, :a])
		cached, scratch = BigRiverEssence._mofa_initialize(data, k; init = :custom,
			z_init = randn(9, k), w_init = [randn(6, k), randn(5, k)],
			ard_weights = aw, ard_factors = az, spikeslab_weights = sw)
		direct = deepcopy(cached)
		uncached = BigRiverEssence.MofaWorkspace(copy(scratch.sample_buffer),
			copy(scratch.feature_buffer), copy(scratch.factor_buffer), nothing)
		@test size(scratch.coordinate_sums) == (11, k + 1)
		@test scratch.coordinate_sums == zeros(11, k + 1)
		for sweep in 1:3
			for (state, workspace) in ((cached, scratch), (direct, uncached))
				BigRiverEssence._mofa_update_weights!(data, state, workspace)
				BigRiverEssence._mofa_update_ard!(data, state; weights = true, factors = false)
				BigRiverEssence._mofa_update_inclusions!(state)
				BigRiverEssence._mofa_update_factors!(data, state, workspace)
				BigRiverEssence._mofa_update_ard!(data, state; weights = false, factors = true)
				BigRiverEssence._mofa_update_noise!(data, state, workspace)
			end
			for field in (:mean, :variance, :second_moment)
				@test getfield(cached.z, field) ≈ getfield(direct.z, field) atol = tol_ord rtol = tol_ord
				for m in 1:2
					@test getfield(cached.w[m], field) ≈ getfield(direct.w[m], field) atol = tol_ord rtol = tol_ord
				end
			end
			for m in 1:2
				@test cached.tau_mean[m] ≈ direct.tau_mean[m] atol = tol_ord rtol = tol_ord
			end
			@test BigRiverEssence._mofa_elbo(data, cached, scratch).total ≈
				BigRiverEssence._mofa_elbo(data, direct, uncached).total atol = tol_ord rtol = tol_ord
		end
		bad = BigRiverEssence.MofaWorkspace(copy(scratch.sample_buffer),
			copy(scratch.feature_buffer), copy(scratch.factor_buffer), zeros(8, k))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(data, cached, bad)
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(data, cached, bad)
	end
	workspace = BigRiverEssence.MofaWorkspace(zeros(9), zeros(11), zeros(8))
	@test size(workspace.coordinate_sums) == (11, 9)
end

@testset "parallel loading/factor sweeps preserve serial results" begin
	Random.seed!(76)
	views = [randn(11, 5), randn(11, 7), randn(11, 3)]
	views[1][2, 3] = NaN
	views[2][4, :] .= NaN
	for X in views
		X[6, :] .= NaN
	end
	groups = repeat([:a, :b], 6)[1:11]
	for aw in (false, true), az in (false, true), sw in (false, true)
		data = BigRiverEssence._mofa_preprocess(views; groups)
		serial, scratch = BigRiverEssence._mofa_initialize(data, 8; init = :custom,
			z_init = randn(11, 8), w_init = [randn(size(X, 2), 8) for X in views],
			ard_weights = aw, ard_factors = az, spikeslab_weights = sw)
		parallel, workers = deepcopy(serial), deepcopy(scratch)
		@test isempty(scratch.parallel_scratch)
		push!(workers.parallel_scratch, (workers.coordinate_sums, workers.factor_buffer))
		for _ in 2:4
			push!(workers.parallel_scratch, (similar(workers.coordinate_sums), zeros(8)))
		end
		@test BigRiverEssence._mofa_check_parallel_scratch(workers, 15, 8) === nothing
		for sweep in 1:3
			for (state, workspace) in ((serial, scratch), (parallel, workers))
				workspace.feature_buffer .= 123.0
				@test BigRiverEssence._mofa_update_weights!(data, state, workspace) === state
				@test workspace.feature_buffer == fill(123.0, 7)
				BigRiverEssence._mofa_update_ard!(data, state; weights = true, factors = false)
				BigRiverEssence._mofa_update_inclusions!(state)
				workspace.sample_buffer .= 123.0
				@test BigRiverEssence._mofa_update_factors!(data, state, workspace) === state
				@test workspace.sample_buffer == fill(123.0, 11)
				BigRiverEssence._mofa_update_ard!(data, state; weights = false, factors = true)
				BigRiverEssence._mofa_update_noise!(data, state, workspace)
			end
			# Worker partitioning does not change any within-row reduction order.
			for field in (:mean, :variance, :second_moment)
				@test getfield(serial.z, field) == getfield(parallel.z, field)
				for m in eachindex(views)
					@test getfield(serial.w[m], field) == getfield(parallel.w[m], field)
				end
			end
			for field in (:tau_shape, :tau_rate, :tau_mean, :tau_logmean)
				@test getfield(serial, field) == getfield(parallel, field)
			end
			for field in (:alpha_w, :alpha_z, :theta_w, :spike_w)
				a, b = getfield(serial, field), getfield(parallel, field)
				if a !== nothing
					for (left, right) in (a isa AbstractVector ? zip(a, b) : ((a, b),))
						for parameter in fieldnames(typeof(left))
							@test getfield(left, parameter) == getfield(right, parameter)
						end
					end
				end
			end
			@test BigRiverEssence._mofa_elbo(data, serial, scratch) ==
				BigRiverEssence._mofa_elbo(data, parallel, workers)
		end
		before = deepcopy(serial)
		@test BigRiverEssence._mofa_weight_sweep!(data, serial, scratch.sample_buffer,
			scratch.factor_buffer, scratch.coordinate_sums, 1:0) === serial
		@test BigRiverEssence._mofa_factor_sweep!(data, serial, scratch.feature_buffer,
			scratch.factor_buffer, scratch.coordinate_sums, 1:0) === serial
		@test serial.z.mean == before.z.mean
		@test all(serial.w[m].mean == before.w[m].mean for m in eachindex(views))
	end
end

@testset "parallel scratch ownership, validation, and automatic threshold" begin
	Random.seed!(77)
	data = BigRiverEssence._mofa_preprocess([randn(9, 6), randn(9, 5)])
	state, workspace = BigRiverEssence._mofa_initialize(data, 8; init = :custom, z_init = randn(9, 8))
	@test BigRiverEssence._mofa_check_parallel_scratch(workspace, 11, 8) === nothing
	for bad_pair in ((zeros(10, 9), zeros(8)), (zeros(11, 8), zeros(8)), (zeros(11, 9), zeros(7)))
		bad = deepcopy(workspace)
		push!(bad.parallel_scratch, bad_pair)
		@test_throws DimensionMismatch BigRiverEssence._mofa_check_parallel_scratch(bad, 11, 8)
		before = deepcopy(state)
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_weights!(data, state, bad)
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(data, state, bad)
		@test state.z.mean == before.z.mean && all(state.w[m].mean == before.w[m].mean for m in 1:2)
	end
	for shared in (:sums, :factors)
		bad = deepcopy(workspace)
		storage = zeros(99)
		pair = (reshape(storage, 11, 9), zeros(8))
		push!(bad.parallel_scratch, pair)
		if shared === :sums
			push!(bad.parallel_scratch, (pair[1], zeros(8)))
		else
			push!(bad.parallel_scratch, (zeros(11, 9), pair[2]))
		end
		@test_throws ArgumentError BigRiverEssence._mofa_check_parallel_scratch(bad, 11, 8)
	end
	large_data = BigRiverEssence._mofa_preprocess([randn(500, 250), randn(500, 250)])
	_, large_scratch = BigRiverEssence._mofa_initialize(large_data, 40; init = :custom, z_init = randn(500, 40))
	@test length(large_scratch.parallel_scratch) == (Threads.nthreads() > 1 ? min(Threads.nthreads(), 8) : 0)
	if !isempty(large_scratch.parallel_scratch)
		@test first(large_scratch.parallel_scratch)[1] === large_scratch.coordinate_sums
		@test first(large_scratch.parallel_scratch)[2] === large_scratch.factor_buffer
		@test BigRiverEssence._mofa_check_parallel_scratch(large_scratch, 500, 40) === nothing
	end
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

@testset "Gaussian factors: hand-calculated multiview update" begin
	X = [1.0 3.0; 5.0 7.0]
	Y = reshape([2.0, 4.0], 2, 1)
	d = BigRiverEssence._mofa_preprocess([X, Y]; groups = [:b, :a], center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = fill(1e20, 2, 1), w_init = [reshape([1.0, 2.0], 2, 1), fill(-1.0, 1, 1)])
	s.w[1].variance[:, 1] .= [0.5, 1.0]
	s.w[2].variance[1, 1] = 2.0
	for W in s.w
		BigRiverEssence._mofa_gaussian_moments!(W)
	end
	s.tau_mean[1] .= [2.0 4.0; 1.0 3.0]
	s.tau_mean[2][:, 1] .= [3.0, 2.0]
	before, data_before = deepcopy(s), deepcopy(d)
	mean_buffer, variance_buffer, second_buffer = s.z.mean, s.z.variance, s.z.second_moment
	workspace.sample_buffer .= 123.0
	@test BigRiverEssence._mofa_update_factors!(d, s, workspace) === s
	# Sample 1: precision = 1 + 2*1.5 + 4*5 + 3*3 = 33;
	# numerator = 2*1*1 + 4*2*3 + 3*(-1)*2 = 20.
	# Sample 2: precision = 1 + 1*1.5 + 3*5 + 2*3 = 47/2;
	# numerator = 1*1*5 + 3*2*7 + 2*(-1)*4 = 39.
	@test s.z.mean[:, 1] ≈ [20 / 33, 78 / 47] atol = tol_ord
	@test s.z.variance[:, 1] ≈ [1 / 33, 2 / 47] atol = tol_ord
	@test s.z.second_moment ≈ s.z.mean .^ 2 .+ s.z.variance atol = tol_ord
	@test s.z.variance[1, 1] != 1 / 22  # squared loading means omit uncertainty
	@test s.z.mean === mean_buffer
	@test s.z.variance === variance_buffer
	@test s.z.second_moment === second_buffer
	for m in eachindex(s.w)
		@test s.w[m].mean == before.w[m].mean
		@test s.w[m].variance == before.w[m].variance
		@test s.w[m].second_moment == before.w[m].second_moment
	end
	@test s.tau_mean == before.tau_mean
	@test s.tau_shape == before.tau_shape
	@test s.tau_rate == before.tau_rate
	@test d.data == data_before.data
	@test d.missing_mask == data_before.missing_mask
	@test workspace.sample_buffer == fill(123.0, 2)
end

@testset "Gaussian factors: multiview sequential scalar reference" begin
	Random.seed!(41)
	n, k = 7, 3
	views = [randn(n, 4), randn(n, 2)]
	views[1][2, 3] = NaN
	views[1][5, 1] = NaN
	views[2][1, :] .= NaN
	d = BigRiverEssence._mofa_preprocess(views; groups = [:b, :a, :b, :a, :b, :a, :b])
	s, workspace = BigRiverEssence._mofa_initialize(d, k; init = :custom,
		z_init = randn(n, k), w_init = [randn(4, k), randn(2, k)])
	for m in eachindex(views)
		s.w[m].variance .= 0.1 .+ rand(size(s.w[m].variance)...)
		BigRiverEssence._mofa_gaussian_moments!(s.w[m])
		s.tau_mean[m] .= 0.2 .+ 3 .* rand(size(s.tau_mean[m])...)
	end
	expected, frozen = deepcopy(s), deepcopy(s)
	for sweep in 1:3
		# Independent scalar calculation, component-major like the Python kernel.
		for component in 1:k, i in 1:n
			precision, numerator = 1.0, 0.0
			for m in eachindex(d.data), j in axes(d.data[m], 2)
				d.missing_mask[m][i, j] && continue
				prediction = sum(expected.z.mean[i, other] * expected.w[m].mean[j, other]
					for other in 1:k if other != component)
				noise = expected.tau_mean[m][d.group_indices[i], j]
				precision += noise * expected.w[m].second_moment[j, component]
				numerator += noise * expected.w[m].mean[j, component] * (d.data[m][i, j] - prediction)
			end
			expected.z.variance[i, component] = 1 / precision
			expected.z.mean[i, component] = numerator / precision
		end
		BigRiverEssence._mofa_update_factors!(d, s, workspace)
		@test s.z.mean ≈ expected.z.mean atol = tol_ord
		@test s.z.variance ≈ expected.z.variance atol = tol_ord
		@test s.z.second_moment ≈ s.z.mean .^ 2 .+ s.z.variance atol = tol_ord
		@test all(x -> 0 < x <= 1, s.z.variance)
	end
	for m in eachindex(s.w)
		@test s.w[m].mean == frozen.w[m].mean
		@test s.w[m].variance == frozen.w[m].variance
		@test s.w[m].second_moment == frozen.w[m].second_moment
	end
	@test s.tau_mean == frozen.tau_mean
end

@testset "Gaussian factors use updated coordinates and all views" begin
	d = BigRiverEssence._mofa_preprocess([[1.0 2.0]]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 2; init = :custom,
		z_init = zeros(1, 2), w_init = [[1.0 1.0; 1.0 2.0]])
	BigRiverEssence._mofa_update_factors!(d, s, workspace)
	@test s.z.mean[1, 1] ≈ 3 / 5 atol = tol_ord
	@test s.z.variance[1, 1] ≈ 1 / 5 atol = tol_ord
	@test s.z.mean[1, 2] ≈ (5 - 3 * (3 / 5)) / 8 atol = tol_ord
	@test s.z.variance[1, 2] ≈ 1 / 8 atol = tol_ord
	@test !isapprox(s.z.mean[1, 2], 5 / 8; atol = tol_ord)
	# Splitting the features into views must preserve an equally weighted update.
	split = BigRiverEssence._mofa_preprocess([fill(1.0, 1, 1), fill(2.0, 1, 1)]; center_groups = false)
	t, scratch = BigRiverEssence._mofa_initialize(split, 2; init = :custom,
		z_init = zeros(1, 2), w_init = [[1.0 1.0], [1.0 2.0]])
	BigRiverEssence._mofa_update_factors!(split, t, scratch)
	@test t.z.mean ≈ s.z.mean atol = tol_ord
	@test t.z.variance ≈ s.z.variance atol = tol_ord
end

@testset "Gaussian factors exclude missing entries and retain uncertainty" begin
	X = reshape([NaN, 3.0, NaN], 3, 1)
	Y = reshape([2.0, NaN, NaN], 3, 1)
	d = BigRiverEssence._mofa_preprocess([X, Y]; groups = [:a, :b, :a], center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = ones(3, 1), w_init = [ones(1, 1), fill(2.0, 1, 1)])
	s.tau_mean[1][:, 1] .= [1e100, 2.0]
	s.tau_mean[2][:, 1] .= [3.0, 1e100]
	d.data[1][1, 1] = Inf
	d.data[1][3, 1] = NaN
	d.data[2][2, 1] = NaN
	d.data[2][3, 1] = Inf
	BigRiverEssence._mofa_update_factors!(d, s, workspace)
	@test s.z.mean[:, 1] ≈ [12 / 16, 6 / 5, 0.0] atol = tol_ord
	@test s.z.variance[:, 1] ≈ [1 / 16, 1 / 5, 1.0] atol = tol_ord
	@test s.z.second_moment[3, 1] == 1.0  # completely missing sample: prior
	d = BigRiverEssence._mofa_preprocess([ones(3, 2)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(3, 1))
	BigRiverEssence._mofa_update_factors!(d, s, workspace)
	@test s.z.mean == zeros(3, 1)
	@test s.z.variance == fill(1 / 3, 3, 1)
end

@testset "Gaussian factor/weight transpose agreement" begin
	Random.seed!(42)
	X = randn(4, 3)
	X[2, 1] = NaN
	d = BigRiverEssence._mofa_preprocess([X]; center_groups = false)
	Z, W = randn(4, 2), randn(3, 2)
	s, workspace = BigRiverEssence._mofa_initialize(d, 2; init = :custom, z_init = Z, w_init = [W])
	s.w[1].variance .= 0.1 .+ rand(3, 2)
	BigRiverEssence._mofa_gaussian_moments!(s.w[1])
	s.tau_mean[1] .= 2.0
	dt = BigRiverEssence._mofa_preprocess([permutedims(X)]; center_groups = false)
	t, scratch = BigRiverEssence._mofa_initialize(dt, 2; init = :custom, z_init = W, w_init = [Z])
	t.z.variance .= s.w[1].variance
	BigRiverEssence._mofa_gaussian_moments!(t.z)
	t.tau_mean[1] .= 2.0
	BigRiverEssence._mofa_update_factors!(d, s, workspace)
	BigRiverEssence._mofa_update_weights!(dt, t, scratch)
	@test s.z.mean ≈ t.w[1].mean atol = tol_ord
	@test s.z.variance ≈ t.w[1].variance atol = tol_ord
	@test s.z.second_moment ≈ t.w[1].second_moment atol = tol_ord
end

@testset "Gaussian factor update validation" begin
	d = BigRiverEssence._mofa_preprocess([[1.0 2.0; 3.0 4.0], ones(2, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = ones(2, 1), w_init = [ones(2, 1), ones(1, 1)])
	for field in (:feature_buffer, :factor_buffer)
		bad_workspace = deepcopy(workspace)
		empty!(getproperty(bad_workspace, field))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(d, s, bad_workspace)
	end
	for field in (:w, :tau_mean)
		bad = deepcopy(s)
		pop!(getproperty(bad, field))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(d, bad, workspace)
	end
	bad_data = deepcopy(d)
	pop!(bad_data.missing_mask)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(bad_data, s, workspace)
	bad_data = deepcopy(d)
	pop!(bad_data.group_indices)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(bad_data, s, workspace)
	for group in (0, 2)
		bad_data = deepcopy(d)
		bad_data.group_indices[1] = group
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(bad_data, s, workspace)
	end
	for field in (:mean, :variance, :second_moment)
		bad = deepcopy(s)
		bad.w[2] = BigRiverEssence.MofaGaussian(
			field === :mean ? zeros(2, 1) : zeros(1, 1),
			field === :variance ? ones(2, 1) : ones(1, 1),
			field === :second_moment ? ones(2, 1) : ones(1, 1))
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(d, bad, workspace)
		@test bad.z.mean == s.z.mean
	end
	bad = deepcopy(s)
	bad.tau_mean[2] = ones(2, 1)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(d, bad, workspace)
	for field in (:data, :missing_mask)
		bad_data = deepcopy(d)
		if field === :data
			bad_data.data[2] = ones(3, 1)
		else
			bad_data.missing_mask[2] = falses(2, 2)
		end
		@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(bad_data, s, workspace)
	end
	bad_z = BigRiverEssence.MofaGaussian(ones(2, 1), ones(1, 1), ones(2, 1))
	bad = BigRiverEssence.MofaState(bad_z, s.w, s.tau_shape, s.tau_rate, s.tau_mean, 1e-3, 1e-3)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_factors!(d, bad, workspace)
	for value in (0.0, -1.0, Inf, NaN)
		bad = deepcopy(s)
		bad.tau_mean[1][1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, bad, workspace)
		bad = deepcopy(s)
		bad.w[1].second_moment[1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, bad, workspace)
	end
	for value in (Inf, NaN)
		bad = deepcopy(s)
		bad.z.mean[1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, bad, workspace)
		bad = deepcopy(s)
		bad.w[1].mean[1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, bad, workspace)
		bad_data = deepcopy(d)
		bad_data.data[1][1, 1] = value
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(bad_data, deepcopy(s), workspace)
	end
	bad = deepcopy(s)
	bad.tau_mean[1][1, 1] = floatmax(Float64)
	@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, bad, workspace)
	bad_data = deepcopy(d)
	bad_data.data[1][1, 1] = 1e200
	@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(bad_data, deepcopy(s), workspace)
	for shape in ((0, 1), (2, 0))
		bad_z = BigRiverEssence.MofaGaussian(zeros(shape...), ones(shape...), ones(shape...))
		bad = BigRiverEssence.MofaState(bad_z, s.w, s.tau_shape, s.tau_rate, s.tau_mean, 1e-3, 1e-3)
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, bad, workspace)
	end
	for field in (:data, :group_labels)
		bad_data = deepcopy(d)
		empty!(getproperty(bad_data, field))
		@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(bad_data, s, workspace)
	end
end

@testset "Gamma and Beta hyperparameter expectations" begin
	q = BigRiverEssence.MofaGamma(ones(1, 2), ones(1, 2), zeros(1, 2), zeros(1, 2), 1e-14, 1e-14)
	@test BigRiverEssence._mofa_gamma_moments!(q) === q
	@test q.mean == ones(1, 2)
	@test q.logmean ≈ fill(-0.5772156649015329, 1, 2) atol = tol_ord
	q.shape .= 2.0
	q.rate .= 4.0
	BigRiverEssence._mofa_gamma_moments!(q)
	@test q.mean == fill(0.5, 1, 2)
	@test q.logmean ≈ fill(1 - 0.5772156649015329 - log(4), 1, 2) atol = tol_ord
	@test BigRiverEssence._mofa_check_gamma(q, (1, 2)) === nothing
	@test BigRiverEssence._mofa_check_gamma(nothing, (1, 2)) === nothing
	@test_throws DimensionMismatch BigRiverEssence._mofa_check_gamma(q, (2, 1))
	@test_throws DimensionMismatch BigRiverEssence._mofa_gamma_moments!(zeros(1, 2), zeros(1, 1), q.shape, q.rate)
	q.mean[1] = NaN
	@test_throws ArgumentError BigRiverEssence._mofa_check_gamma(q, (1, 2))
	q.shape[1] = -1.0
	@test_throws ArgumentError BigRiverEssence._mofa_gamma_moments!(q)
	bad = BigRiverEssence.MofaGamma(ones(1, 1), ones(1, 1), ones(1, 1), zeros(1, 1), 0.0, 1.0)
	@test_throws ArgumentError BigRiverEssence._mofa_gamma_moments!(bad)
	b = BigRiverEssence.MofaBeta(ones(1, 2), ones(1, 2), zeros(1, 2), zeros(1, 2), zeros(1, 2), 1.0, 1.0)
	@test BigRiverEssence._mofa_beta_moments!(b) === b
	@test b.mean == fill(0.5, 1, 2)
	@test b.logmean ≈ fill(-1.0, 1, 2) atol = tol_ord
	@test b.logcomplement ≈ fill(-1.0, 1, 2) atol = tol_ord
	b.a .= 2.0
	b.b .= 3.0
	BigRiverEssence._mofa_beta_moments!(b)
	@test b.mean ≈ fill(2 / 5, 1, 2) atol = tol_ord
	@test b.logmean ≈ fill(-13 / 12, 1, 2) atol = tol_ord
	@test b.logcomplement ≈ fill(-7 / 12, 1, 2) atol = tol_ord
	@test BigRiverEssence._mofa_check_beta(b, (1, 2)) === nothing
	@test_throws DimensionMismatch BigRiverEssence._mofa_check_beta(b, (2, 1))
	b.logmean[1] = Inf
	@test_throws ArgumentError BigRiverEssence._mofa_check_beta(b, (1, 2))
	b.a[1] = 0.0
	@test_throws ArgumentError BigRiverEssence._mofa_beta_moments!(b)
	bad_beta = BigRiverEssence.MofaBeta(ones(1, 1), ones(1, 2), ones(1, 1), zeros(1, 1), zeros(1, 1), 1.0, 1.0)
	@test_throws DimensionMismatch BigRiverEssence._mofa_beta_moments!(bad_beta)
end

@testset "sparse-loading moments and stable inclusion probabilities" begin
	q = BigRiverEssence.MofaSpikeSlab([2.0 -3.0; 0.0 1.0], [0.5 2.0; 1.0 0.1], [4.0, 5.0], [0.0 0.25; 1.0 1.0])
	w = BigRiverEssence.MofaGaussian(zeros(2, 2), zeros(2, 2), zeros(2, 2))
	@test BigRiverEssence._mofa_spike_moments!(w, q) === w
	@test w.mean == [0.0 -0.75; 0.0 1.0]
	@test w.second_moment ≈ [0.0 2.75; 1.0 1.1] atol = tol_ord
	@test w.variance ≈ [0.0 2.1875; 1.0 0.1] atol = tol_ord
	@test BigRiverEssence._mofa_check_spike(q, (2, 2)) === nothing
	@test_throws DimensionMismatch BigRiverEssence._mofa_check_spike(q, (1, 2))
	q.inclusion[1] = 1.1
	@test_throws ArgumentError BigRiverEssence._mofa_spike_moments!(w, q)
	for x in (-1000.0, -10.0, 0.0, 10.0, 1000.0)
		@test 0 <= BigRiverEssence._mofa_logistic(x) <= 1
	end
	@test BigRiverEssence._mofa_logistic(-Inf) == 0.0
	@test BigRiverEssence._mofa_logistic(Inf) == 1.0
	@test BigRiverEssence._mofa_logistic(0.0) == 0.5
	@test_throws ArgumentError BigRiverEssence._mofa_logistic(NaN)
end

@testset "optional ARD and sparse initialization" begin
	d = BigRiverEssence._mofa_preprocess([[1.0 3.0; 2.0 1.0; 4.0 2.0], ones(3, 1)]; groups = [:b, :a, :b])
	for aw in (false, true), az in (false, true), sw in (false, true)
		s, _ = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(3, 1),
			ard_weights = aw, ard_factors = az, spikeslab_weights = sw)
		@test (s.alpha_w !== nothing) == aw
		@test (s.alpha_z !== nothing) == az
		@test (s.spike_w !== nothing) == sw
		@test (s.theta_w !== nothing) == sw
		if aw
			@test size(s.alpha_w.mean) == (2, 1)
			@test s.alpha_w.mean == ones(2, 1)
			@test s.alpha_w.prior_shape == s.alpha_w.prior_rate == 1e-14
		end
		if az
			@test size(s.alpha_z.mean) == (2, 1)
			@test s.alpha_z.mean == ones(2, 1)
		end
		if sw
			@test s.theta_w.mean == fill(0.5, 2, 1)
			@test all(q -> all(==(1.0), q.inclusion), s.spike_w)
			@test s.spike_w[1].mean !== s.w[1].mean
			@test length(s.spike_w[1].inactive_variance) == 1
		end
		@test s.tau_logmean[1] ≈ fill(-0.5772156649015329, 2, 2) atol = tol_ord
	end
end

@testset "noise posterior: counts, uncertainty, missing groups, and stability" begin
	d = BigRiverEssence._mofa_preprocess([reshape([5.0, NaN, 2.0], 3, 1)]; groups = [:a, :b, :a], center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = reshape([1.0, 4.0, 2.0], 3, 1), w_init = [ones(1, 1)])
	s.z.variance[:, 1] .= [0.5, 0.25, 1.0]
	s.w[1].variance[1] = 2.0
	BigRiverEssence._mofa_gaussian_moments!(s.z)
	BigRiverEssence._mofa_gaussian_moments!(s.w[1])
	before = deepcopy(s)
	d.data[1][2] = Inf  # masked: must be ignored
	@test BigRiverEssence._mofa_check_moments(d, s) === nothing
	@test BigRiverEssence._mofa_update_noise!(d, s, workspace) === s
	@test s.tau_shape[1][:, 1] ≈ [1.001, 0.001] atol = tol_ord
	@test s.tau_rate[1][:, 1] ≈ [15.251, 0.001] atol = tol_ord
	@test s.tau_mean[1][:, 1] ≈ [1.001 / 15.251, 1.0] atol = tol_ord
	@test all(isfinite, s.tau_logmean[1])
	@test s.tau_logmean[1][2] < -900  # Gamma(.001,.001), not initial Gamma(1,1)
	@test s.tau_logmean[1][1] != log(s.tau_mean[1][1])
	@test s.z.mean == before.z.mean
	@test s.w[1].mean == before.w[1].mean
	@test s.z.second_moment == before.z.second_moment
	# Nearly exact large reconstruction: the expanded Gram identity cancels,
	# but explicit variances still contribute 1e4 + 1e4 to expected error.
	d = BigRiverEssence._mofa_preprocess([fill(1e16, 1, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = fill(1e8, 1, 1), w_init = [fill(1e8, 1, 1)])
	s.z.variance .= 1e-12
	s.w[1].variance .= 1e-12
	BigRiverEssence._mofa_gaussian_moments!(s.z)
	BigRiverEssence._mofa_gaussian_moments!(s.w[1])
	BigRiverEssence._mofa_update_noise!(d, s, workspace)
	@test s.tau_rate[1][1] ≈ 1e4 + 0.001 atol = tol_ord
	@test s.tau_shape[1][1] == 0.501
end

@testset "dense ARD updates and prior precision in coordinate updates" begin
	d = BigRiverEssence._mofa_preprocess([ones(3, 2)]; groups = [:b, :a, :b], center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = reshape([1.0, 2.0, 3.0], 3, 1), w_init = [reshape([2.0, -1.0], 2, 1)],
		ard_weights = true, ard_factors = true)
	s.w[1].variance[:, 1] .= [0.5, 3.0]
	BigRiverEssence._mofa_gaussian_moments!(s.w[1])
	@test BigRiverEssence._mofa_update_ard!(d, s; factors = false) === s
	@test s.alpha_w.shape[1] ≈ 1 + 1e-14 atol = tol_ord
	@test s.alpha_w.rate[1] ≈ 4.25 + 1e-14 atol = tol_ord
	@test s.alpha_z.shape == ones(2, 1)
	frozen = deepcopy(s.alpha_w)
	BigRiverEssence._mofa_update_ard!(d, s; weights = false)
	@test s.alpha_z.shape[:, 1] ≈ [1 + 1e-14, 0.5 + 1e-14] atol = tol_ord
	@test s.alpha_z.rate[:, 1] ≈ [6 + 1e-14, 2.5 + 1e-14] atol = tol_ord
	@test s.alpha_w.mean == frozen.mean
	d = BigRiverEssence._mofa_preprocess([fill(2.0, 1, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(1, 1),
		ard_weights = true, ard_factors = true)
	s.alpha_w.shape .= 4.0
	s.alpha_z.shape .= 3.0
	BigRiverEssence._mofa_gamma_moments!(s.alpha_w)
	BigRiverEssence._mofa_gamma_moments!(s.alpha_z)
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	@test s.w[1].variance[1] ≈ 1 / 6 atol = tol_ord
	@test s.w[1].mean[1] ≈ 1 / 3 atol = tol_ord
	BigRiverEssence._mofa_update_factors!(d, s, workspace)
	@test s.z.variance[1] ≈ 18 / 59 atol = tol_ord
	@test s.z.mean[1] ≈ 12 / 59 atol = tol_ord
end

@testset "sparse weights, underlying ARD moment, and Beta updates" begin
	d = BigRiverEssence._mofa_preprocess([reshape([3.0, 1.0], 2, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom,
		z_init = reshape([1.0, 2.0], 2, 1), ard_weights = true, spikeslab_weights = true)
	before_theta = deepcopy(s.theta_w)
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	g = inv(1 + exp(-(0.5 * log(1 / 8) + 25 / 16)))
	@test s.spike_w[1].mean[1] ≈ 5 / 8 atol = tol_ord
	@test s.spike_w[1].variance[1] ≈ 1 / 8 atol = tol_ord
	@test s.spike_w[1].inclusion[1] ≈ g atol = tol_ord
	@test s.w[1].mean[1] ≈ g * 5 / 8 atol = tol_ord
	@test s.w[1].second_moment[1] ≈ g * 33 / 64 atol = tol_ord
	@test s.theta_w.logmean == before_theta.logmean  # no implicit hyperparameter update
	underlying = g * 33 / 64 + (1 - g)
	BigRiverEssence._mofa_update_ard!(d, s; factors = false)
	@test s.alpha_w.rate[1] ≈ 1e-14 + underlying / 2 atol = tol_ord
	@test s.alpha_w.rate[1] > s.w[1].second_moment[1] / 2
	@test s.spike_w[1].inactive_variance == [1.0]  # not retroactively replaced
	@test BigRiverEssence._mofa_update_inclusions!(s) === s
	@test s.theta_w.a[1] ≈ 1 + g atol = tol_ord
	@test s.theta_w.b[1] ≈ 2 - g atol = tol_ord
	@test s.theta_w.mean[1] ≈ (1 + g) / 3 atol = tol_ord
	# Exact zero effective moments are legitimate for sparse weights.
	s.spike_w[1].inclusion .= 0.0
	BigRiverEssence._mofa_spike_moments!(s.w[1], s.spike_w[1])
	BigRiverEssence._mofa_update_factors!(d, s, workspace)
	@test s.z.mean == zeros(2, 1)
	@test s.z.variance == ones(2, 1)
	BigRiverEssence._mofa_update_noise!(d, s, workspace)
	@test all(isfinite, s.tau_mean[1])
	# Extreme Beta shapes produce saturated inclusions without exponential overflow.
	s.theta_w.a .= 1e-300
	s.theta_w.b .= 1.0
	BigRiverEssence._mofa_beta_moments!(s.theta_w)
	BigRiverEssence._mofa_update_weights!(d, s, workspace)
	@test s.spike_w[1].inclusion == zeros(1, 1)
end

@testset "noise and regularization validation/no-op paths" begin
	d = BigRiverEssence._mofa_preprocess([ones(2, 1)]; center_groups = false)
	s, workspace = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(2, 1))
	@test BigRiverEssence._mofa_update_ard!(d, s) === s
	@test BigRiverEssence._mofa_update_inclusions!(s) === s
	bad_workspace = deepcopy(workspace)
	empty!(bad_workspace.sample_buffer)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_noise!(d, s, bad_workspace)
	bad = deepcopy(s)
	bad.tau_logmean[1] = zeros(2, 1)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_noise!(d, bad, workspace)
	bad = deepcopy(s)
	pop!(bad.tau_rate)
	@test_throws DimensionMismatch BigRiverEssence._mofa_update_noise!(d, bad, workspace)
	bad = deepcopy(s)
	bad.z.variance[1] = -1.0
	@test_throws ArgumentError BigRiverEssence._mofa_check_moments(d, bad)
	bad_data = deepcopy(d)
	bad_data.group_indices[1] = 0
	@test_throws ArgumentError BigRiverEssence._mofa_check_moments(bad_data, s)
	bad_data = deepcopy(d)
	bad_data.missing_mask[1] = falses(1, 1)
	@test_throws DimensionMismatch BigRiverEssence._mofa_check_moments(bad_data, s)
	bad_data = deepcopy(d)
	bad_data.data[1][1] = Inf
	@test_throws ArgumentError BigRiverEssence._mofa_update_noise!(bad_data, deepcopy(s), workspace)
	t, scratch = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(2, 1),
		ard_weights = true, ard_factors = true, spikeslab_weights = true)
	t.alpha_w.mean[1] = 0.0
	@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, t, scratch)
	@test_throws ArgumentError BigRiverEssence._mofa_update_ard!(d, t)
	t.alpha_w.mean[1] = 1.0
	t.alpha_z.mean[1] = NaN
	@test_throws ArgumentError BigRiverEssence._mofa_update_factors!(d, t, scratch)
	t.alpha_z.mean[1] = 1.0
	t.spike_w[1].inclusion[1] = -1.0
	@test_throws ArgumentError BigRiverEssence._mofa_update_inclusions!(t)
	@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, t, scratch)
	t.spike_w[1].inclusion[1] = 1.0
	t.alpha_w.shape .= 0.5
	t.alpha_w.rate .= floatmax(Float64)
	BigRiverEssence._mofa_gamma_moments!(t.alpha_w)
	@test_throws ArgumentError BigRiverEssence._mofa_update_weights!(d, t, scratch)
end

@testset "ELBO distributions, hand calculation, and validation" begin
	q = BigRiverEssence.MofaBeta(fill(2.0, 1, 1), ones(1, 1), zeros(1, 1), zeros(1, 1), zeros(1, 1), 1.0, 1.0)
	BigRiverEssence._mofa_beta_moments!(q)
	@test BigRiverEssence._mofa_beta_elbo(q) ≈ 0.5 - log(2) atol = tol_ord
	@test BigRiverEssence._mofa_gamma_elbo(ones(1, 1), ones(1, 1), ones(1, 1), fill(-Base.MathConstants.eulergamma, 1, 1), 1.0, 1.0) ≈ 0 atol = tol_ord
	@test_throws DimensionMismatch BigRiverEssence._mofa_gamma_elbo(ones(1, 1), ones(2, 1), ones(1, 1), zeros(1, 1), 1.0, 1.0)
	@test_throws ArgumentError BigRiverEssence._mofa_gamma_elbo(ones(1, 1), ones(1, 1), ones(1, 1), zeros(1, 1), 0.0, 1.0)
	@test_throws ArgumentError BigRiverEssence._mofa_gamma_elbo(zeros(1, 1), ones(1, 1), ones(1, 1), zeros(1, 1), 1.0, 1.0)
	d = BigRiverEssence._mofa_preprocess([fill(2.0, 1, 1)]; center_groups = false)
	s, b = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(1, 1))
	before = deepcopy(s)
	lb = BigRiverEssence._mofa_elbo(d, s, b)
	@test lb.y ≈ -0.5 * (log(2pi) + Base.MathConstants.eulergamma + 6) atol = tol_ord
	@test lb.w ≈ 0 atol = tol_ord
	@test lb.z ≈ -0.5 atol = tol_ord
	@test lb.alpha_w == lb.alpha_z == lb.theta_w == 0
	@test lb.total ≈ sum(values(lb)[1:7]) atol = tol_ord
	@test s.z.mean == before.z.mean && s.tau_rate == before.tau_rate
	BigRiverEssence._mofa_update_noise!(d, s, b)
	@test collect(values(BigRiverEssence._mofa_elbo(d, s, b))) ≈ collect(values(BigRiverEssence._mofa_elbo(d, s, b; noise_updated = true))) atol = tol_ord
	bad_workspace = BigRiverEssence.MofaWorkspace(zeros(2), zeros(1), zeros(1))
	@test_throws DimensionMismatch BigRiverEssence._mofa_elbo(d, s, bad_workspace)
	bad = deepcopy(s)
	bad.z.variance .= 0
	@test_throws ArgumentError BigRiverEssence._mofa_elbo(d, bad, b)
	bad = deepcopy(s)
	bad.tau_shape[1][1] = NaN
	@test_throws ArgumentError BigRiverEssence._mofa_elbo(d, bad, b)
	# Sparse inclusion probabilities exactly zero and one must remain finite.
	for inclusion in (0.0, 1.0)
		ss, bb = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(1, 1), spikeslab_weights = true)
		ss.spike_w[1].inclusion .= inclusion
		BigRiverEssence._mofa_spike_moments!(ss.w[1], ss.spike_w[1])
		@test isfinite(BigRiverEssence._mofa_elbo(d, ss, bb).total)
	end
end

@testset "ELBO convergence confirmations and gates" begin
	for mode in (:fast, :medium, :slow)
		@test BigRiverEssence._mofa_convergence(0.0, -100.0, 1, mode) == (2, false)
		@test BigRiverEssence._mofa_convergence(0.0, -100.0, 2, mode) == (3, true)
		@test BigRiverEssence._mofa_convergence(1.0, -100.0, 2, mode) == (1, false)
	end
	@test BigRiverEssence._mofa_convergence(1e-4, -100.0, 1, :fast) == (2, false)
	@test BigRiverEssence._mofa_convergence(1e-4, -100.0, 1, :medium) == (1, false)
	@test BigRiverEssence._mofa_convergence(0.0, 0.0, 1, :slow) == (2, false)
	@test_throws ArgumentError BigRiverEssence._mofa_convergence(NaN, -1.0, 1, :fast)
	@test_throws ArgumentError BigRiverEssence._mofa_convergence(0.0, Inf, 1, :fast)
	@test_throws ArgumentError BigRiverEssence._mofa_convergence(0.0, -1.0, 3, :fast)
	@test_throws ArgumentError BigRiverEssence._mofa_convergence(0.0, -1.0, 1, :invalid)
	# Zero factor means stay zero, but variances/noise still need to settle.
	# Check the actual ELBO confirmations rather than assuming immediate convergence.
	kwargs = (k = 1, init = :custom, z_init = zeros(4, 1), ard_weights = false,
		spikeslab_weights = false, freq_elbo = 1, center_groups = false)
	X = [ones(4, 1)]
	m = mofa(X; kwargs..., maxiter = 20)
	@test m.converged && 3 <= m.niter < 20
	@test m.elbo_iterations == collect(0:m.niter)
	@test all(100 .* abs.(diff(m.elbo)[end-1:end]) ./ abs(m.elbo[1]) .< 5e-4)
	m = mofa(X; kwargs..., maxiter = 20, miniter = 5)
	@test m.converged && m.niter == 7
	m = mofa(X; kwargs..., maxiter = 20, start_elbo = 5)
	@test m.converged && m.niter == 7
	m = mofa(X; kwargs..., maxiter = 4, forceiter = true)
	@test !m.converged && m.niter == 4
end

@testset "full fits and ELBO components against mofapy2" begin
	folder = joinpath(@__DIR__, "Data", "MOFA")
	X = [readdlm(joinpath(folder, "X$m.csv"), ',', Float64) for m in 1:2]
	groups = vec(Int.(readdlm(joinpath(folder, "groups.csv"), ',', Float64)))
	Z = readdlm(joinpath(folder, "Z_init.csv"), ',', Float64)
	for aw in (false, true), az in (false, true), sw in (false, true)
		case = joinpath(folder, "aw$(Int(aw))_az$(Int(az))_sw$(Int(sw))")
		ref = readdlm(joinpath(case, "elbo.csv"), ',', Float64)
		d = BigRiverEssence._mofa_preprocess(X; groups)
		s, b = BigRiverEssence._mofa_initialize(d, 2; init = :custom, z_init = Z,
			ard_weights = aw, ard_factors = az, spikeslab_weights = sw)
		@test collect(values(BigRiverEssence._mofa_elbo(d, s, b))) ≈ vec(ref[1, 2:9]) atol = tol_julia rtol = tol_ord
		for iteration in 1:30
			BigRiverEssence._mofa_update_weights!(d, s, b)
			BigRiverEssence._mofa_update_ard!(d, s; weights = true, factors = false)
			iteration >= 3 && BigRiverEssence._mofa_update_inclusions!(s)
			BigRiverEssence._mofa_update_factors!(d, s, b)
			BigRiverEssence._mofa_update_ard!(d, s; weights = false, factors = true)
			BigRiverEssence._mofa_update_noise!(d, s, b)
			explicit = collect(values(BigRiverEssence._mofa_elbo(d, s, b)))
			fast = collect(values(BigRiverEssence._mofa_elbo(d, s, b; noise_updated = true)))
			@test explicit ≈ vec(ref[iteration + 1, 2:9]) atol = tol_julia rtol = tol_ord
			@test fast ≈ explicit atol = tol_ord rtol = tol_ord
		end
		m = mofa(X; k = 2, groups, init = :custom, z_init = Z,
			ard_weights = aw, ard_factors = az, spikeslab_weights = sw,
			start_sparsity = 3, freq_elbo = 1, maxiter = 30, forceiter = true)
		@test m.elbo ≈ ref[:, 9] atol = tol_julia rtol = tol_ord
		@test m.z ≈ s.z.mean atol = tol_ord
		@test m.z === m.posterior.z.mean && m.w[1] === m.posterior.w[1].mean
		@test m.elbo_iterations == collect(0:30)
		@test m.niter == 30 && !m.converged
		arrays = Any[m.z, m.posterior.z.variance]
		for view in 1:2
			append!(arrays, [m.w[view], m.posterior.w[view].second_moment,
				m.posterior.tau_shape[view], m.posterior.tau_rate[view]])
		end
		if aw
			for view in 1:2
				append!(arrays, [m.posterior.alpha_w.shape[view, :], m.posterior.alpha_w.rate[view, :]])
			end
		end
		az && append!(arrays, [m.posterior.alpha_z.shape, m.posterior.alpha_z.rate])
		if sw
			for view in 1:2
				append!(arrays, [m.posterior.spike_w[view].inclusion,
					m.posterior.theta_w.a[view, :], m.posterior.theta_w.b[view, :]])
			end
		end
		@test vcat(vec.(arrays)...) ≈ vec(readdlm(joinpath(case, "posterior.csv"), ',', Float64)) atol = tol_julia rtol = tol_ord
		@test m.r2 ≈ readdlm(joinpath(case, "r2.csv"), ',', Float64) atol = tol_ord
		@test vec(m.factor_r2) ≈ vec(readdlm(joinpath(case, "factor_r2.csv"), ',', Float64)) atol = tol_ord
		stopping = vec(readdlm(joinpath(case, "stopping.csv"), ',', Float64))
		stopped = mofa(X; k = 2, groups, init = :custom, z_init = Z,
			ard_weights = aw, ard_factors = az, spikeslab_weights = sw,
			start_sparsity = 3, freq_elbo = 5, maxiter = 1000)
		@test stopped.niter == Int(stopping[1])
		@test stopped.converged == Bool(stopping[2])
		@test stopped.elbo[end] ≈ stopping[3] atol = tol_julia rtol = tol_ord
	end
end

@testset "public fit options, reconstruction, and input preservation" begin
	Random.seed!(160)
	Z = randn(30, 2)
	X = [Z * randn(2, 7) + 0.02 * randn(30, 7), Z * randn(2, 5) + 0.02 * randn(30, 5)]
	original = deepcopy(X)
	# Dense model should recover the observed low-rank structure.
	m = mofa(X; k = 2, ard_weights = false, spikeslab_weights = false,
		maxiter = 100, forceiter = true)
	@test minimum(m.r2) > 0.99
	@test all(diff(m.elbo) .>= -tol_julia)
	@test X == original
	@test !hasfield(typeof(m), :data) && !hasfield(typeof(m), :workspace)
	@test size(m.factor_r2) == (1, 2, 2)
	@test all(q -> all(>(0), q), m.posterior.tau_mean)
	# Restore the original measurement units, including interleaved groups.
	groups = repeat(["b", "a"], 15)
	m = mofa(X; k = 2, groups, scale_views = true, scale_groups = true,
		ard_weights = false, spikeslab_weights = false, maxiter = 100, forceiter = true)
	for view in 1:2
		processed = m.z * transpose(m.w[view])
		restored = [processed[i, j] * m.scales[m.group_indices[i], view] +
			m.means[view][m.group_indices[i], j] for i in axes(processed, 1), j in axes(processed, 2)]
		@test norm(restored - X[view]) / norm(X[view]) < 0.05
	end
	@test m.group_labels == ["b", "a"]
	# Default model flags, PCA, random initialization, and reproducible RNG.
	m = mofa(X; k = 2, maxiter = 2, forceiter = true)
	@test m.options.ard_weights && m.options.spikeslab_weights && !m.options.ard_factors
	@test m.posterior.alpha_w !== nothing && m.posterior.spike_w !== nothing
	random_a = mofa(X; k = 2, init = :random, rng = MersenneTwister(160), maxiter = 3, forceiter = true)
	random_b = mofa(X; k = 2, init = :random, rng = MersenneTwister(160), maxiter = 3, forceiter = true)
	@test random_a.z == random_b.z && random_a.elbo == random_b.elbo
	@test random_a.z != mofa(X; k = 2, init = :random, rng = MersenneTwister(161), maxiter = 3, forceiter = true).z
	# Unscheduled final objective is retained once, but is not an early-stop check.
	m = mofa(X; k = 2, maxiter = 4, freq_elbo = 5, forceiter = true)
	@test m.elbo_iterations == [0, 1, 4]
	m = mofa(X; k = 2, maxiter = 2, start_elbo = 5, forceiter = true)
	@test m.elbo_iterations == [0, 2]
	@test_logs (:warn, r"MOFA reached maxiter") mofa(X; k = 2, maxiter = 1)
	@test_logs (:info, r"MOFA ELBO") mofa(X; k = 2, maxiter = 1, forceiter = true, verbose = true)
	for kwargs in ((maxiter = 0,), (start_elbo = 0,), (freq_elbo = 0,),
		(start_sparsity = 0,), (miniter = -1,), (convergence_mode = :invalid,), (k = 0,))
		@test_throws ArgumentError mofa(X; kwargs...)
	end
	@test_throws ArgumentError mofa(X; init = :custom, k = 2)
	@test_throws DimensionMismatch mofa(X; init = :custom, k = 2, z_init = ones(2, 2))
	# R² is undefined for zero-energy and entirely missing group/view blocks.
	d = BigRiverEssence._mofa_preprocess([zeros(3, 1), reshape([1.0, 2.0, NaN], 3, 1)]; groups = [1, 1, 2])
	s, b = BigRiverEssence._mofa_initialize(d, 1; init = :custom, z_init = ones(3, 1))
	r2, factor_r2 = BigRiverEssence._mofa_variance_explained(d, s, b)
	@test all(isnan, r2[:, 1]) && isnan(r2[2, 2])
	@test all(isnan, factor_r2[:, 1, :]) && isnan(factor_r2[2, 2, 1])
	@test r2[1, 2] == 0
	@test_throws DimensionMismatch BigRiverEssence._mofa_variance_explained(d, s, BigRiverEssence.MofaWorkspace(zeros(1), zeros(1), zeros(1)))
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
