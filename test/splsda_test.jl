# Test/splsda_test.jl — tests for splsda (sparse PLS discriminant analysis, after
# Lê Cao, Boitard & Besse 2011) and its internals: _center_scale, _unmap (one-hot
# encoder), _soft_threshold_L1! (the keepX sparsifier), _sqdiff.
# Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl: tol_ord for exact
# identities, tol_r for the cross-language mixOmics fixtures.





@testset "output structure & invariants" begin
	# Basic contract: right type and shapes (note loadings_Y has one row per CLASS,
	# not per feature — the labels are dummy-encoded into K class columns), plus the
	# defining sparsity property: each X-loading keeps exactly keepX nonzeros.
	Random.seed!(1)
	n, p, ncomp = 60, 100, 2
	y = repeat(["A", "B", "C"], inner = 20)
	X = randn(n, p)
	m = BigRiverEssence.splsda(X, y, ncomp, [10, 10])

	@test m isa BigRiverEssence.Splsda
	@test size(m.variates_X) == (n, ncomp)        # X scores (sample coordinates)
	@test size(m.variates_Y) == (n, ncomp)        # Y scores
	@test size(m.loadings_X) == (p, ncomp)        # sparse X loadings
	@test size(m.loadings_Y) == (3, ncomp)        # K=3 classes ⇒ 3 rows, not p
	@test m.ncomp == ncomp
	@test m.keepX == [10, 10]
	@test m.classes == ["A", "B", "C"]            # default ordering is sorted-unique
	@test size(m.Y_dummy) == (n, 3)               # one-hot label matrix
	# Each component's X-loading keeps EXACTLY keepX nonzeros and is unit-norm —
	# this exact count is the whole point of the keepX sparsity budget.
	for c in 1:ncomp
		@test count(!iszero, m.loadings_X[:, c]) == 10
		@test isapprox(norm(m.loadings_X[:, c]), 1.0; atol = tol_ord)
	end
end

@testset "keepX controls sparsity exactly" begin
	# keepX is a hard variable count, not a soft penalty: ask for kx variables and the
	# loading has exactly kx nonzeros, across a range of kx. This is what distinguishes
	# the keepX parameterization from a λ that only indirectly controls sparsity.
	Random.seed!(2)
	y = repeat(["A", "B"], inner = 25)
	X = randn(50, 80)
	for kx in (5, 20, 50)
		m = BigRiverEssence.splsda(X, y, 1, [kx])
		@test count(!iszero, m.loadings_X[:, 1]) == kx
	end
end

@testset "Y_dummy is a valid one-hot encoding" begin
	# The class labels are turned into an indicator matrix (the PLS-DA trick: regress
	# X onto the dummy-coded classes). Verify it's a proper one-hot: exactly one 1 per
	# row, only 0/1 entries, and column c really is the indicator for class c.
	Random.seed!(3)
	y = repeat(["A", "B", "C"], inner = 15)
	m = BigRiverEssence.splsda(randn(45, 30), y, 1, [5])
	@test all(sum(m.Y_dummy, dims = 2) .== 1.0)   # each row has exactly one 1 (one class)
	@test all(x -> x == 0.0 || x == 1.0, m.Y_dummy)
	for (c, cls) in enumerate(m.classes)
		@test m.Y_dummy[:, c] == Float64.(y .== cls)   # column c ⇔ membership in class c
	end
end

@testset "ground truth: selects discriminative variables & separates classes" begin
	# Plant signal: the first 10 variables carry class-specific means; variables 11:500
	# are pure noise. A working sPLS-DA must (a) SELECT the signal variables and (b)
	# produce scores that SEPARATE the classes. Both are checked with known ground truth.
	Random.seed!(123)
	classes = ["A", "B", "C"];
	n_per = 20
	y = repeat(classes, inner = n_per);
	n = length(y);
	p = 500
	X = randn(n, p) .* 0.5
	for (ci, cls) in enumerate(classes)
		X[findall(==(cls), y), 1:10] .+= ci * 2.0     # each class shifts vars 1:10 by a class-specific amount
	end
	m = BigRiverEssence.splsda(X, y, 2, [10, 10])

	# (a) Variable selection: most of the 10 true signal variables are picked up.
	sel = findall(!iszero, m.loadings_X[:, 1])
	@test length(intersect(sel, 1:10)) >= 8       # literal recovery floor (noisy ground truth)
	# (b) Class separation: component-1 scores cluster tightly within class and spread
	# widely between classes. We quantify with the between/within variance ratio — a
	# one-way-ANOVA F-style statistic; ≫ 1 means clean separation.
	sc      = m.variates_X[:, 1];
	grand   = mean(sc)
	between = sum(n_per * (mean(sc[findall(==(c), y)]) - grand)^2 for c in classes)
	within  = sum(sum((sc[i] - mean(sc[findall(==(y[i]), y)]))^2
	for i in findall(==(c), y)) for c in classes)
	@test between / within > 10                   # separation floor
end

@testset "levels controls class ordering" begin
	# By default classes are sorted; passing `levels` overrides that ordering, which
	# fixes the column order of both the class loadings and the dummy matrix. Useful
	# for matching an external convention (e.g. lining up with mixOmics' factor levels).
	Random.seed!(4)
	y = repeat(["A", "B", "C"], inner = 10)
	m1 = BigRiverEssence.splsda(randn(30, 20), y, 1, [5])
	m2 = BigRiverEssence.splsda(randn(30, 20), y, 1, [5]; levels = ["C", "B", "A"])
	@test m1.classes == ["A", "B", "C"]           # default: sorted
	@test m2.classes == ["C", "B", "A"]           # custom: as supplied
	@test m2.Y_dummy[:, 1] == Float64.(y .== "C") # column 1 now tracks "C", per the levels order
end

@testset "argument validation" begin
	# The arguments that can be inconsistent must throw: keepX must have one entry per
	# component, and `levels` must list every class exactly (right count, no unknowns).
	Random.seed!(0)
	y = repeat(["A", "B"], inner = 10);
	X = randn(20, 15)
	@test_throws ArgumentError BigRiverEssence.splsda(X, y, 2, [5])          # keepX length 1 ≠ ncomp 2
	@test_throws ArgumentError BigRiverEssence.splsda(X, y, 1, [5]; levels = ["A", "B", "C"])  # 3 levels, 2 classes
	@test_throws ArgumentError BigRiverEssence.splsda(X, y, 1, [5]; levels = ["A", "Z"])       # "Z" isn't a class
end

# ----------------------------------------------------------------------------
# Internal helpers — tested directly so a regression in a building block surfaces
# here, not as a puzzling failure in the full fit.
# ----------------------------------------------------------------------------

@testset "internal: _center_scale" begin
	# Column centering, optionally to unit variance. Two things to nail: with scale=true
	# columns end up mean-0 and SD-1 (using the n−1 corrected SD, matching mixOmics);
	# with scale=false they're only centered. Plus the constant-column guard.
	Random.seed!(5)
	M = randn(40, 8) .* (1:8)' .+ (1:8)'                        # columns with varied scale + offset
	Cs = BigRiverEssence._center_scale(M; scale = true)
	@test all(abs.(vec(mean(Cs, dims = 1))) .< tol_ord)         # every column centered
	@test all(isapprox.(vec(std(Cs, dims = 1; corrected = true)), 1.0; atol = tol_ord))  # and unit-SD
	Cc = BigRiverEssence._center_scale(M; scale = false)
	@test all(abs.(vec(mean(Cc, dims = 1))) .< tol_ord)         # centered…
	@test !all(isapprox.(vec(std(Cc, dims = 1)), 1.0; atol = tol_ord))   # …but NOT scaled
	# A constant column has zero SD; dividing by it would give NaN/Inf. The guard sets
	# such a column to all-zeros instead — a constant carries no discriminative signal.
	Mz = hcat(randn(20), fill(3.0, 20))
	Csz = BigRiverEssence._center_scale(Mz; scale = true)
	@test all(isfinite, Csz)                                    # no NaN/Inf from the zero SD
	@test all(Csz[:, 2] .== 0.0)                                # the constant column zeroed out
end

@testset "internal: _unmap (one-hot encoder)" begin
	# The label → indicator-matrix encoder. Default classes are sorted; the rows encode
	# membership; `levels` reorders the columns; bad levels throw.
	y = ["B", "A", "C", "A", "B"]
	Yd, classes = BigRiverEssence._unmap(y)
	@test classes == ["A", "B", "C"]                            # sorted unique labels
	@test Yd == [0 1 0; 1 0 0; 0 0 1; 1 0 0; 0 1 0]             # each row one-hot for its label
	# Custom levels permute the columns (so "B" lands in column 2 of THIS ordering).
	Yd2, classes2 = BigRiverEssence._unmap(y; levels = ["C", "B", "A"])
	@test classes2 == ["C", "B", "A"]
	@test Yd2[1, :] == [0, 1, 0]                                # row 1 is "B" → middle column
	@test_throws ArgumentError BigRiverEssence._unmap(y; levels = ["A", "B"])       # too few levels (3 classes)
	@test_throws ArgumentError BigRiverEssence._unmap(y; levels = ["A", "B", "Z"])  # "Z" present, "C" missing
end

@testset "internal: _soft_threshold_L1! (keeps p−nx largest, zeros nx smallest)" begin
	# The sparsifier behind keepX, reparameterized by nx = number to DROP (= p − keepX).
	# It zeros the nx smallest-magnitude entries and soft-thresholds the survivors by
	# λ = the largest dropped magnitude. Here nx=2, so the two smallest |x| are zeroed.
	x = [5.0, -3.0, 1.0, -0.5, 2.0]              # magnitudes: 5, 3, 1, 0.5, 2
	out = similar(x)
	absx = similar(x);
	ord = similar(x, Int);
	ranks = similar(x, Int)   # preallocated scratch
	BigRiverEssence._soft_threshold_L1!(out, x, 2, absx, ord, ranks)           # drop the 2 smallest (|−0.5|, |1|)
	@test out[4] == 0.0 && out[3] == 0.0          # those two are zeroed
	@test all(out[[1, 2, 5]] .!= 0.0)             # the three largest survive
	@test sign.(out[[1, 2, 5]]) == sign.(x[[1, 2, 5]])   # survivors keep their sign
	# Survivors are shrunk by λ = the largest DROPPED magnitude, which is |1.0| = 1.
	@test out[1] ≈ 5.0 - 1.0
	@test out[5] ≈ 2.0 - 1.0
	@test out[2] ≈ -(3.0 - 1.0)                   # sign carried through the shrink
	# nx ≤ 0 means "drop nothing" ⇒ the input copies through unchanged.
	o2 = similar(x);
	BigRiverEssence._soft_threshold_L1!(o2, x, 0, absx, ord, ranks)
	@test o2 == x
end

@testset "internal: _sqdiff (squared L2 distance)" begin
	# Σ(aᵢ−bᵢ)², the convergence metric between successive loading iterates.
	@test BigRiverEssence._sqdiff([1.0, 2.0], [1.0, 2.0]) == 0.0           # identical ⇒ 0
	@test BigRiverEssence._sqdiff([0.0, 0.0], [3.0, 4.0]) == 25.0          # 3²+4²
	a = randn(30);
	b = randn(30)
	@test BigRiverEssence._sqdiff(a, b) ≈ sum(abs2, a .- b)
end

# ----------------------------------------------------------------------------
# Cross-language check against mixOmics::splsda, using offline fixtures (no live R).
# ----------------------------------------------------------------------------

@testset "matches mixOmics::splsda (offline reference fixtures)" begin
	# Compare against mixOmics' saved output. We pass `levels` so our class ordering
	# matches R's factor levels — otherwise the components could line up but the class
	# columns wouldn't, and the loadings_Y comparison would spuriously fail.
	refdir = joinpath(@__DIR__, "Data", "SPLSDA")
	if !isfile(joinpath(refdir, "X.csv"))
		@info "sPLS-DA mixOmics fixtures not found; run generate_splsda_reference.R to create them."
	else
		# Record the mixOmics version that produced the fixtures (results can drift
		# across versions, so provenance helps when chasing a mismatch).
		smfile = joinpath(refdir, "session_meta.csv")
		if isfile(smfile)
			sm = readdlm(smfile, ',', String; skipstart = 1)
			row = findfirst(==("mixOmics_version"), sm[:, 1])
			row !== nothing && @info "sPLS-DA fixtures generated against mixOmics $(sm[row, 2])"
		end

		rdf(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
		rds(f) = vec(readdlm(joinpath(refdir, f), ',', String; skipstart = 1))
		X = rdf("X.csv")
		y = rds("Y.csv")
		lx = rdf("lx.csv");
		ly = rdf("ly.csv")    # mixOmics X- and Y-loadings
		vx = rdf("vx.csv");
		vy = rdf("vy.csv")    # mixOmics X- and Y-variates
		levs = rds("levels.csv")                     # R's factor level order
		meta = rdf("meta.csv")
		ncomp = Int(meta[1]);
		keepX = [Int(meta[2]), Int(meta[3])]

		m = BigRiverEssence.splsda(X, y, ncomp, keepX; levels = levs)   # levels = levs aligns our class order with R's

		for c in 1:ncomp
			# Loadings/variates match up to per-component sign (each component's SVD
			# sign is arbitrary), so compare via |correlation| ≈ 1, not raw difference.
			@test abs(cor(m.loadings_X[:, c], lx[:, c])) > 1 - tol_r
			@test abs(cor(m.variates_X[:, c], vx[:, c])) > 1 - tol_r
			@test abs(cor(m.loadings_Y[:, c], ly[:, c])) > 1 - tol_r
			@test abs(cor(m.variates_Y[:, c], vy[:, c])) > 1 - tol_r
			# The SET of selected variables must match EXACTLY — support recovery is the
			# discrete fingerprint of the sparse fit, and is sign-independent.
			@test Set(findall(!iszero, m.loadings_X[:, c])) == Set(findall(!iszero, lx[:, c]))
		end
	end
end
