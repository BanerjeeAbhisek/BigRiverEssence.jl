# Tests for MOFA input validation and Gaussian preprocessing.
# No model is fitted in this first implementation step.
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
