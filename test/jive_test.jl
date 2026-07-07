# Test/jive_test.jl — tests for jive (JIVE, matching r.jive's orthIndiv variant) and
# its internals: _jive_rjive_core_opt2 (given-ranks core), _jive_perm_ranks_opt (the
# permutation rank estimator), and the safe_svd* SVD wrappers.
# Tolerances (tol_ord / tol_julia / tol_r) come from runtests.jl. JIVE adds a fourth,
# STRUCTURAL kind of tolerance: orthogonality floors (< 1e-4) — see the notes below
# on why those are deliberately looser than tol_ord.








# Mirror jive's own preprocessing: row-center each block, then apply r.jive's
# Frobenius scaling (divide by ‖block‖·√(total elements)). Reproduced here so the
# internal-helper tests can feed the cores the SAME preprocessed input jive uses,
# and so we can check the preprocessing in isolation.
function _preprocess(Xs; center = true, scale = true)
	k = length(Xs)
	sum_n = sum(size(X, 1) * size(X, 2) for X in Xs)
	Xc = Vector{Matrix{Float64}}(undef, k)
	for i in 1:k
		Xi = center ? Float64.(Xs[i]) .- mean(Xs[i], dims = 2) : Float64.(Xs[i])
		scale && (Xi ./= (norm(Xi) * sqrt(sum_n)))
		Xc[i] = Matrix{Float64}(Xi)
	end
	return Xc
end

# Canonical correlations between the column spaces of B1 and B2, using their first
# `d` directions. We orthonormalize each (via QR) then take the singular values of
# Q1ᵀQ2 — these are the cosines of the principal angles between the subspaces.
# `d` is the SUBSPACE dimension (e.g. the joint rank), NOT B's column count: a
# stacked joint basis has many columns but spans only a d-dimensional subspace, so
# comparing raw matrices would be wrong — we compare what they SPAN.
function canon(B1, B2, d)
	Q1 = Matrix(qr(B1).Q)[:, 1:d]
	Q2 = Matrix(qr(B2).Q)[:, 1:d]
	svdvals(Q1' * Q2)
end

# Simulated data with KNOWN structure (matches jive.R's generator): a shared joint
# signal S (rank rT) plus block-specific individual signals (ranks r1T, r2T), plus
# noise. Used by the given-ranks tests.
function make_data(; seed = 2024, noise = 0.3)
	Random.seed!(seed)
	n = 80;
	rT, r1T, r2T = 2, 3, 3;
	p1, p2 = 60, 50
	S = randn(rT, n)                          # the shared joint scores both blocks see
	U1 = randn(p1, rT);
	U2 = randn(p2, rT)     # block-specific joint loadings
	S1 = randn(r1T, n);
	W1 = randn(p1, r1T)    # block 1's individual signal
	S2 = randn(r2T, n);
	W2 = randn(p2, r2T)    # block 2's individual signal
	X1 = U1*S + W1*S1 .+ noise .* randn(p1, n)
	X2 = U2*S + W2*S2 .+ noise .* randn(p2, n)
	(; X1, X2, S, n, rT, r1T, r2T, p1, p2)
end

# Strongly-structured data for the auto-rank tests: a clear rank-2 joint and rank-3
# individuals, well-separated from noise, so the permutation test reliably recovers
# the planted ranks (2, [3,3]).
function make_struct(; seed = 42)
	Random.seed!(seed)
	n = 80;
	p1, p2 = 60, 50
	Sj = randn(2, n)
	X1 = randn(p1, 2)*Sj + randn(p1, 3)*randn(3, n)*0.5 + randn(p1, n)*0.1
	X2 = randn(p2, 2)*Sj + randn(p2, 3)*randn(3, n)*0.5 + randn(p2, n)*0.1
	(; X1, X2, n)
end


@testset "output structure & invariants" begin
	# Basic contract: right type, one J and one A block per input, correct shapes,
	# and the joint scores S have orthonormal ROWS (S Sᵀ = I) — S is the shared basis.
	d = make_data()
	m = BigRiverEssence.jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])

	@test m isa BigRiverEssence.jiveStructure
	@test length(m.J) == 2 && length(m.A) == 2
	@test size(m.J[1]) == (d.p1, d.n) && size(m.J[2]) == (d.p2, d.n)
	@test size(m.A[1]) == (d.p1, d.n) && size(m.A[2]) == (d.p2, d.n)
	@test size(m.S) == (d.rT, d.n)
	@test m.r == d.rT
	@test m.ri == [d.r1T, d.r2T]
	@test all(isfinite, m.S)
	@test m.S * m.S' ≈ I(d.rT) atol = tol_ord     # joint scores: orthonormal rows
end


@testset "joint structure has rank r; individual has rank ri" begin
	# JIVE's defining decomposition: the stacked joint must be EXACTLY rank r, and
	# each individual block EXACTLY rank ri. We check the first dropped singular value
	# is negligible relative to the largest — i.e. the rank cliff is sharp.
	d = make_data()
	m = BigRiverEssence.jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])
	sJ = svdvals(reduce(vcat, m.J))
	@test sJ[d.rT+1] / sJ[1] < tol_ord          # σ_{r+1}/σ₁ ≈ 0 ⇒ stacked joint is rank r
	for (i, ri) in enumerate([d.r1T, d.r2T])
		sA = svdvals(m.A[i])
		@test sA[ri+1] / sA[1] < tol_ord        # likewise each individual block is rank ri
	end
end

@testset "joint ⊥ individual (orthogonality constraint)" begin
	# The whole point of BigRiverEssence.jive: joint and individual structure are ORTHOGONAL, so
	# variation isn't double-counted. We check both the row-space orthogonality
	# (Jstack ⊥ each A) and the score-space orthogonality (S ⊥ each Si).
	# The < 1e-4 floor (NOT tol_ord) is structural: r.BigRiverEssence.jive's orthIndiv enforces this
	# orthogonality only APPROXIMATELY via alternating projection, so it converges to
	# ~1e-4, never machine zero. Tightening this would fail on a correct fit.
	d = make_data()
	m = BigRiverEssence.jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])
	Jstack = reduce(vcat, m.J)
	for i in 1:2
		@test norm(Jstack * m.A[i]') / (norm(Jstack) * norm(m.A[i])) < 1e-4   # joint ⊥ individual
	end
	for i in 1:2
		@test norm(m.S * m.Si[i]') / (norm(m.S) * norm(m.Si[i])) < 1e-4       # joint scores ⊥ indiv scores
	end
end

@testset "ground truth: recovers joint subspace (noiseless)" begin
	# IMPORTANT subtlety: r.jive's orthIndiv variant does NOT reconstruct X exactly,
	# even with zero noise — the orthogonality constraint deliberately trades perfect
	# reconstruction for clean joint/individual separation. So the right ground-truth
	# check is SUBSPACE recovery (does the fitted joint span the true joint space?)
	# plus orthogonality — NOT ‖X − (J+A)‖ ≈ 0, which would (correctly) fail.
	Random.seed!(2024)
	n, rT, r1T, r2T, p1, p2 = 80, 2, 3, 3, 60, 50
	S = randn(rT, n);
	U1 = randn(p1, rT);
	U2 = randn(p2, rT)
	S1 = randn(r1T, n);
	W1 = randn(p1, r1T);
	S2 = randn(r2T, n);
	W2 = randn(p2, r2T)
	X1c = U1*S + W1*S1;
	X2c = U2*S + W2*S2                # clean, noiseless data

	m = BigRiverEssence.jive([X1c, X2c], rT, [r1T, r2T]; scale = false)
	Jstack = reduce(vcat, m.J)
	for i in 1:2
		@test norm(Jstack * m.A[i]') / (norm(Jstack) * norm(m.A[i])) < 1e-4   # still orthogonal
	end
	# The fitted joint scores must span the same subspace as the true (centered) S.
	# Compared via canonical correlation, so a different-but-equivalent basis still passes.
	Strue = S .- mean(S, dims = 2)
	@test minimum(canon(Strue', m.S', rT)) > 0.99        # principal angles ≈ 0 ⇒ same subspace
end

@testset "auto ranks: permutation finds the planted ranks" begin
	# With ranks NOT supplied, jive estimates them by permutation test. On the
	# well-separated structured data it must recover the planted (2, [3,3]).
	s = make_struct()
	Random.seed!(999)                                    # fix RNG: the permutation test is stochastic
	m = BigRiverEssence.jive([s.X1, s.X2]; nperm = 100)
	@test m.r == 2                                        # joint rank
	@test m.ri == [3, 3]                                 # individual ranks
end

@testset "positional and keyword signatures agree" begin
	# jive(Xs, r, ri) and jive(Xs; r=, ri=) are two spellings of the same call — the
	# positional form just forwards to the keyword one, so results must be identical.
	d = make_data()
	a = BigRiverEssence.jive([d.X1, d.X2], d.rT, [d.r1T, d.r2T])
	b = BigRiverEssence.jive([d.X1, d.X2]; r = d.rT, ri = [d.r1T, d.r2T])
	for i in 1:2
		@test a.J[i] ≈ b.J[i]
		@test a.A[i] ≈ b.A[i]
	end
end

@testset "preprocessing: center + r.jive scaling" begin
	# Verify the two preprocessing steps in isolation: every row is mean-zero after
	# centering, and the Frobenius scaling shrinks the block's norm (it divides by a
	# quantity > 1), so the scaled block has smaller norm than the merely-centered one.
	d = make_data()
	Xc = _preprocess([d.X1, d.X2])
	@test all(abs(sum(@view Xc[1][r, :])) < tol_ord for r in 1:d.p1)   # each row sums to ≈ 0
	@test norm(Xc[1]) < norm(d.X1 .- mean(d.X1, dims = 2))             # scaling reduced the norm
end

@testset "argument validation" begin
	# All blocks must share the same number of columns (observations). Mismatched
	# column counts can't be jointly decomposed, so it must throw.
	Random.seed!(0)
	X1 = randn(20, 50);
	X2 = randn(15, 40)               # 50 vs 40 columns — incompatible
	@test_throws ArgumentError BigRiverEssence.jive([X1, X2], 2, [2, 2])
end

# ----------------------------------------------------------------------------
# Internal helpers — exercised directly so a regression in a primitive is caught
# at the source rather than as a confusing symptom in the full decomposition.
# ----------------------------------------------------------------------------

@testset "internal: safe_svd / safe_svdvals / safe_svd!" begin
	# The safe_svd* wrappers fall back to a robust algorithm on LAPACK convergence
	# failures, but on normal input they must behave exactly like Base's svd: reproduce
	# the factorization, and return the same singular values. safe_svd! mutates input.
	Random.seed!(7)
	A = randn(40, 25)
	F = BigRiverEssence.safe_svd(A)
	@test F.U * Diagonal(F.S) * F.Vt ≈ A                  # factorization reconstructs A
	@test BigRiverEssence.safe_svdvals(A) ≈ svdvals(A)                          # singular values match Base
	@test F.S ≈ svdvals(A)
	Acopy = copy(A)                                       # safe_svd! overwrites its argument,
	F2 = BigRiverEssence.safe_svd!(Acopy)                                    # so pass a copy to keep A intact
	@test F2.U * Diagonal(F2.S) * F2.Vt ≈ A
end

@testset "internal: _jive_rjive_core_opt2 (given-ranks core)" begin
	# The alternating joint/individual core, called on preprocessed data with the
	# ranks fixed. Must return a valid jiveStructure with the right ranks, an exactly
	# rank-r joint, and the joint ⊥ individual orthogonality (same 1e-4 floor as above).
	d = make_data()
	Xc = _preprocess([d.X1, d.X2])
	conv = 1e-6 * norm(reduce(vcat, Xc))                  # same convergence threshold jive uses
	m = BigRiverEssence._jive_rjive_core_opt2(Xc, d.n, d.rT, [d.r1T, d.r2T]; conv = conv, maxiter = 1000)
	@test m isa BigRiverEssence.jiveStructure
	@test m.r == d.rT && m.ri == [d.r1T, d.r2T]
	sJ = svdvals(reduce(vcat, m.J))
	@test sJ[d.rT+1] / sJ[1] < tol_ord                  # joint is exactly rank r
	Jstack = reduce(vcat, m.J)
	@test norm(Jstack * m.A[1]') / (norm(Jstack) * norm(m.A[1])) < 1e-4   # orthogonality holds
end

@testset "internal: _jive_perm_ranks_opt (rank estimator)" begin
	# The permutation rank estimator in isolation: on the structured data it must
	# return the planted joint and individual ranks. RNG fixed since it permutes.
	s = make_struct()
	Xc = _preprocess([s.X1, s.X2])
	conv = 1e-6 * norm(reduce(vcat, Xc))
	Random.seed!(999)
	rJ, rA = BigRiverEssence._jive_perm_ranks_opt(Xc, s.n; nperm = 100, alpha = 0.05, conv = conv, maxiter = 1000)
	@test rJ == 2
	@test rA == [3, 3]
end

# ----------------------------------------------------------------------------
# Cross-language checks against r.jive, using offline fixtures (no live R needed).
# ----------------------------------------------------------------------------

@testset "matches r.jive (offline reference fixtures, given ranks)" begin
	# With ranks GIVEN, JIVE is deterministic — no permutation, no RNG — so the
	# decomposition should match r.jive's saved output pointwise (to cross-language
	# tol_r). Fixtures from jive.R; skip with a note if they're absent.
	refdir = joinpath(@__DIR__, "Data", "JIVE")
	if !isfile(joinpath(refdir, "X1.csv"))
		@info "JIVE r.jive fixtures not found; skipping. Run jive.R to create them."
	else
		# Record the r.jive version that produced the fixtures (results can shift
		# between package versions, so provenance helps when debugging a mismatch).
		smfile = joinpath(refdir, "session_meta.csv")
		if isfile(smfile)
			sm = readdlm(smfile, ',', String; skipstart = 1)
			row = findfirst(==("r.jive_version"), sm[:, 1])
			row !== nothing && @info "JIVE fixtures generated against r.jive $(sm[row, 2])"
		end

		rd(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
		X1 = rd("X1.csv");
		X2 = rd("X2.csv")
		D1 = rd("D1.csv");
		D2 = rd("D2.csv")              # r.jive's preprocessed (scaled) blocks
		J1 = rd("J1.csv");
		J2 = rd("J2.csv")              # r.jive's joint structure
		A1 = rd("A1.csv");
		A2 = rd("A2.csv")              # r.jive's individual structure
		meta = rd("meta.csv")
		rT = Int(meta[2]);
		r1T = Int(meta[3]);
		r2T = Int(meta[4])

		m = BigRiverEssence.jive([X1, X2], rT, [r1T, r2T])

		# First confirm OUR preprocessing reproduces r.jive's scaled input — if this
		# diverged, every downstream comparison would fail for the wrong reason.
		Xc = _preprocess([X1, X2])
		@test norm(Xc[1] .- D1) < tol_r
		@test norm(Xc[2] .- D2) < tol_r

		# Given-ranks JIVE is deterministic, so the decomposition matches pointwise.
		@test norm(m.J[1] .- J1) < tol_r
		@test norm(m.J[2] .- J2) < tol_r
		@test norm(m.A[1] .- A1) < tol_r
		@test norm(m.A[2] .- A2) < tol_r
	end
end

@testset "matches r.jive — auto ranks (offline fixtures)" begin
	# The auto-rank path against r.jive. Two-part check: (1) both implementations
	# recover the SAME ranks, then (2) — because the decompositions can't be
	# bit-identical (R and Julia draw different permutation RNG streams) — compare
	# the joint SUBSPACE via canonical correlation rather than pointwise.
	refdir = joinpath(@__DIR__, "Data", "JIVE")
	if !isfile(joinpath(refdir, "X1s.csv"))
		@info "JIVE auto-rank fixtures not found; re-run jive.R to create them."
	else
		rd(f) = readdlm(joinpath(refdir, f), ',', Float64; skipstart = 1)
		X1s = rd("X1s.csv");
		X2s = rd("X2s.csv")
		J1p = rd("J1p.csv");
		J2p = rd("J2p.csv")
		ranks = vec(rd("ranks_perm.csv"))
		rJ_r = Int(ranks[1]);
		rA_r = Int.(ranks[2:end])

		Random.seed!(999)
		m = BigRiverEssence.jive([X1s, X2s]; nperm = 100)

		# Print both sides' ranks — handy when eyeballing a run, and harmless if they match.
		println("  Julia : joint=$(m.r)  indiv=$(m.ri)")
		println("  r.jive: joint=$rJ_r  indiv=$rA_r")
		println("  match : ", m.r == rJ_r && m.ri == rA_r)

		# (1) Same ranks recovered on this well-separated data.
		@test m.r == rJ_r
		@test m.ri == rA_r
		# (2) Joint subspaces agree. Guarded on matching rank, since canon needs a
		# common dimension — if the ranks differed the subspace comparison is undefined.
		if m.r == rJ_r
			cc = canon(reduce(vcat, m.J)', vcat(J1p, J2p)', m.r)
			@test minimum(cc) > 1 - tol_r        # principal angles ≈ 0 across languages
		end
	end
end
