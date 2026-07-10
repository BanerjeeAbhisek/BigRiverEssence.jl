# Test/plsda_test.jl — tests for plsda (PLS discriminant analysis), the non-sparse
# counterpart of splsda. Reuses the same internals (_center_scale, _unmap, _sqdiff),
# so those are tested in splsda_test.jl and not repeated here.
# Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl.

@testset "output structure & invariants" begin
	# Basic contract: right type and shapes (loadings_Y has one row per CLASS, not per
	# feature). Unlike splsda, loadings are DENSE — every variable contributes, so each
	# X-loading column has (essentially) all-nonzero entries and is unit-norm.
	Random.seed!(1)
	n, p, ncomp = 60, 100, 2
	y = repeat(["A", "B", "C"], inner = 20)
	X = randn(n, p)
	m = BigRiverEssence.plsda(X, y, ncomp)

	@test m isa BigRiverEssence.Plsda
	@test size(m.variates_X) == (n, ncomp)        # X scores
	@test size(m.variates_Y) == (n, ncomp)        # Y scores
	@test size(m.loadings_X) == (p, ncomp)        # dense X loadings
	@test size(m.loadings_Y) == (3, ncomp)        # K=3 classes ⇒ 3 rows
	@test m.ncomp == ncomp
	@test m.classes == ["A", "B", "C"]            # default: sorted-unique
	@test size(m.Y_dummy) == (n, 3)               # one-hot label matrix
	# Loadings are dense (no keepX sparsity) and unit-norm.
	for c in 1:ncomp
		@test count(!iszero, m.loadings_X[:, c]) == p      # ALL variables retained
		@test isapprox(norm(m.loadings_X[:, c]), 1.0; atol = tol_ord)
	end
end

@testset "Y_dummy is a valid one-hot encoding" begin
	# The class labels become an indicator matrix (regress X onto dummy-coded classes).
	# Verify proper one-hot: exactly one 1 per row, only 0/1 entries, column c ⇔ class c.
	Random.seed!(3)
	y = repeat(["A", "B", "C"], inner = 15)
	m = BigRiverEssence.plsda(randn(45, 30), y, 1)
	@test all(sum(m.Y_dummy, dims = 2) .== 1.0)
	@test all(x -> x == 0.0 || x == 1.0, m.Y_dummy)
	for (c, cls) in enumerate(m.classes)
		@test m.Y_dummy[:, c] == Float64.(y .== cls)
	end
end

@testset "ground truth: separates classes" begin
	# Plant signal: the first 10 variables carry class-specific means; 11:500 are noise.
	# Unlike splsda, plsda does NOT select variables (dense loadings), but its scores
	# must still SEPARATE the classes — that's the discriminant part.
	Random.seed!(123)
	classes = ["A", "B", "C"];
	n_per = 20
	y = repeat(classes, inner = n_per);
	n = length(y);
	p = 500
	X = randn(n, p) .* 0.5
	for (ci, cls) in enumerate(classes)
		X[findall(==(cls), y), 1:10] .+= ci * 2.0
	end
	m = BigRiverEssence.plsda(X, y, 2)

	# Class separation on component 1: between/within variance ratio (ANOVA-F style).
	sc      = m.variates_X[:, 1];
	grand   = mean(sc)
	between = sum(n_per * (mean(sc[findall(==(c), y)]) - grand)^2 for c in classes)
	within  = sum(sum((sc[i] - mean(sc[findall(==(y[i]), y)]))^2
	for i in findall(==(c), y)) for c in classes)
	@test between / within > 10                   # separation floor
	# The signal variables should carry the largest loadings (even though none are zeroed).
	load1 = abs.(m.loadings_X[:, 1])
	@test mean(load1[1:10]) > mean(load1[11:end])  # signal vars load more than noise vars
end

@testset "levels controls class ordering" begin
	# Passing `levels` overrides the default sorted ordering, fixing the column order of
	# the class loadings and the dummy matrix (e.g. to match mixOmics' factor levels).
	Random.seed!(4)
	y = repeat(["A", "B", "C"], inner = 10)
	m1 = BigRiverEssence.plsda(randn(30, 20), y, 1)
	m2 = BigRiverEssence.plsda(randn(30, 20), y, 1; levels = ["C", "B", "A"])
	@test m1.classes == ["A", "B", "C"]
	@test m2.classes == ["C", "B", "A"]
	@test m2.Y_dummy[:, 1] == Float64.(y .== "C")
end

@testset "argument validation" begin
	# `levels` must list every class exactly (right count, no unknowns).
	Random.seed!(0)
	y = repeat(["A", "B"], inner = 10);
	X = randn(20, 15)
	@test_throws ArgumentError BigRiverEssence.plsda(X, y, 1; levels = ["A", "B", "C"])  # 3 levels, 2 classes
	@test_throws ArgumentError BigRiverEssence.plsda(X, y, 1; levels = ["A", "Z"])       # "Z" isn't a class
end

# ----------------------------------------------------------------------------
# Cross-language check against mixOmics::plsda, using offline fixtures (no live R).
# ----------------------------------------------------------------------------

@testset "matches mixOmics::plsda (offline reference fixtures)" begin
	# Compare against mixOmics' saved output. We pass `levels` so our class ordering
	# matches R's factor levels — otherwise components could align but class columns
	# wouldn't, and the loadings_Y comparison would spuriously fail.
	refdir = joinpath(@__DIR__, "Data", "PLSDA")
	if !isfile(joinpath(refdir, "X.csv"))
		@info "PLS-DA mixOmics fixtures not found; run generate_plsda_reference.R to create them."
	else
		smfile = joinpath(refdir, "session_meta.csv")
		if isfile(smfile)
			sm = readdlm(smfile, ',', String; skipstart = 1)
			row = findfirst(==("mixOmics_version"), sm[:, 1])
			row !== nothing && @info "PLS-DA fixtures generated against mixOmics $(sm[row, 2])"
		end

		rdf(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
		rds(f) = vec(readdlm(joinpath(refdir, f), ',', String; skipstart = 1))
		X = rdf("X.csv")
		y = rds("Y.csv")
		lx = rdf("lx.csv");
		ly = rdf("ly.csv")     # mixOmics X- and Y-loadings
		vx = rdf("vx.csv");
		vy = rdf("vy.csv")     # mixOmics X- and Y-variates
		levs = rds("levels.csv")
		meta = rdf("meta.csv")
		ncomp = Int(meta[1])

		m = BigRiverEssence.plsda(X, y, ncomp; levels = levs)

		for c in 1:ncomp
			# Loadings/variates match up to per-component sign, so compare via
			# |correlation| ≈ 1 rather than raw difference.
			@test abs(cor(m.loadings_X[:, c], lx[:, c])) > 1 - tol_r
			@test abs(cor(m.variates_X[:, c], vx[:, c])) > 1 - tol_r
			@test abs(cor(m.loadings_Y[:, c], ly[:, c])) > 1 - tol_r
			@test abs(cor(m.variates_Y[:, c], vy[:, c])) > 1 - tol_r
		end
	end
end