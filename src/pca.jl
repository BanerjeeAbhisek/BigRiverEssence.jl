"""
	PcaStructure{T}

Container for a fitted PCA model, as returned by `pca`
# Fields
- `mean::Vector{T}`: The column means of the training data (length p), removed
  during centering
- `scale::Vector{T}`: The column scales (length p) — the column standard
  deviations when `standardize=true`, otherwise ones
- `loadings::Matrix{T}`: The p×k principal directions (right singular vectors of
  the centered data / leading eigenvectors of the covariance matrix), unit-norm
  columns ordered by decreasing explained variance
- `variances::Vector{T}`: The variance explained by each of the k components
  (the eigenvalues, σᵢ² / (n-1))
- `propOFvar::Vector{T}`: The proportion of total variance explained by each
  component (variances ./ total variance)
"""
struct PcaStructure{T}
	mean::Vector{T}
	scale::Vector{T}
	loadings::Matrix{T}
	variances::Vector{T}
	propOFvar::Vector{T}
end

"""
	pca_transform(m::PcaStructure, X::Matrix{Float64})

Project data onto the principal directions of a fitted PCA model
# Arguments
- `m::PcaStructure`: A fitted PCA model, as returned by `pca`
- `X::Matrix{Float64}`: 2d array of floats; the observations (rows) by features
  (columns) to project, with the same features as the training data
# Value
2d array of floats; the n×k matrix of principal-component scores
"""
function pca_transform(m::PcaStructure, X::Matrix{Float64})
	# Apply the SAME centering and scaling the model learned, then project. Using
	# the stored stats (not X's own mean/std) is what makes this valid for new data.
	Xcentered = (X .- m.mean') ./ m.scale'
	return Xcentered * m.loadings
end

"""
	pca_invtransform(m::PcaStructure, scores::Matrix{Float64})

Reconstruct data in the original feature space from principal-component scores
# Arguments
- `m::PcaStructure`: A fitted PCA model, as returned by `pca`
- `scores::Matrix{Float64}`: 2d array of floats; the n×k matrix of
  principal-component scores to invert, with k matching the number of components
  retained in `m`
# Value
2d array of floats; the n×p reconstruction in the original units. Exact only
when all components are retained (k = p); otherwise a low-rank approximation
"""
function pca_invtransform(m::PcaStructure, scores::Matrix{Float64})
	# Reverse the forward transform in reverse order: undo the projection first
	# (scores → centered feature space), then undo scaling, then undo centering.
	Xcentered = scores * m.loadings'
	return Xcentered .* m.scale' .+ m.mean'
end

"""
	pca(X::Matrix{Float64}; k::Int = minimum(size(X)),
		standardize::Bool = false, method::Symbol = :auto)

Fit a principal component analysis (PCA) model
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; observations (rows) by features (columns)
- `k::Int`: The number of components to retain; 1 ≤ k ≤ min(n,p). Defaults to min(n,p)
- `standardize::Bool`: Whether to scale columns to unit variance in addition to
  centering. Defaults to false (center only)
- `method::Symbol`: `:auto` (pick by shape — `:cov` when n ≥ p, `:svd` when p > n),
  `:cov` (eigendecomposition of the p×p covariance), or `:svd` (SVD of the centered
  data). Defaults to `:auto`
# Value
A `PcaStructure` holding the column means, scales, k loadings, component variances,
and proportion of variance explained
"""
function pca(X::Matrix{Float64}; k::Int = minimum(size(X)),
	standardize::Bool = false, method::Symbol = :auto)
	n, p = size(X)
	1 <= k <= min(n, p) || throw(ArgumentError("k=$k must be in 1:min(n,p)=$(min(n,p))"))

	# Pick the cheaper decomposition by data shape. A tall matrix (more observations
	# than features) gives a small p×p covariance to eigendecompose; a wide matrix
	# is cheaper to attack with an SVD of the data itself. Resolve :auto first so the
	# validation below sees a concrete method.
	if method === :auto
		method = n >= p ? :cov : :svd
	end
	method in (:cov, :svd) || throw(ArgumentError("method must be :auto, :cov, or :svd, got :$method"))

	colmeans = vec(mean(X, dims = 1))                          # column means (length p)
	colstds  = standardize ? vec(std(X, dims = 1)) : Float64[] # empty vector = "not standardizing"

	if method === :cov
		# Build the p×p scatter matrix S = Xcᵀ Xc WITHOUT ever forming a centered
		# copy of X. The identity Xcᵀ Xc = Xᵀ X − n·(μ μᵀ) lets us start from the raw
		# Gram matrix and apply a rank-1 correction — saving an n×p allocation.
		scatter = mul!(Matrix{Float64}(undef, p, p), transpose(X), X)   # raw Gram, Xᵀ X
		@inbounds for j in 1:p, i in 1:p
			scatter[i, j] -= n * colmeans[i] * colmeans[j]    # subtract n·μμᵀ ⇒ centered scatter
		end
		if standardize                                        # fold column scaling in: S ← D⁻¹ S D⁻¹
			@inbounds for j in 1:p, i in 1:p
				scatter[i, j] /= (colstds[i] * colstds[j])
			end
		end
		scatter_sym = Symmetric(scatter)

		# Total variance is the trace of the (scaled) covariance — read it straight
		# off the scatter's diagonal, no extra pass over the data.
		total = tr(scatter_sym) / (n - 1)
		# eigen with a range asks LAPACK for only the top-k eigenpairs (cheaper than
		# a full decomposition). They come back ascending, so reverse to descending.
		topk = eigen(scatter_sym, (p-k+1):p)
		vars = reverse(topk.values) ./ (n - 1)
		loadings = topk.vectors[:, k:-1:1]                    # reorder columns to descending variance
	else  # :svd — center first, then SVD in whichever orientation LAPACK handles faster
		Xcentered = standardize ? (X .- colmeans') ./ vec(std(X, dims = 1))' : X .- colmeans'
		if p > n
			# Wide data: SVD the TALL transpose Xcᵀ (p×n) because LAPACK's SVD runs
			# faster on tall matrices. The loadings we want — the right singular
			# vectors of Xc — are the LEFT singular vectors of Xcᵀ, so we read F.U.
			F = svd!(permutedims(Xcentered))                  # Xcᵀ is p×n (tall)
			svals = @view F.S[1:k]
			vars = collect(svals .^ 2 ./ (n - 1))
			loadings = Matrix(@view F.U[:, 1:k])              # U of the transpose = the loadings
			total = sum(abs2, F.S) / (n - 1)
		else
			# Tall (or square) data: SVD Xc directly; the loadings are its right
			# singular vectors F.V.
			F = svd!(Xcentered)
			svals = @view F.S[1:k]
			vars = collect(svals .^ 2 ./ (n - 1))
			loadings = Matrix(@view F.V[:, 1:k])
			total = sum(abs2, F.S) / (n - 1)
		end
	end

	_sign_consistency_opt!(loadings)                            # pin each PC's arbitrary sign for reproducibility
	scale_out = standardize ? colstds : ones(Float64, p)      # store ones when not standardizing
	return PcaStructure{Float64}(colmeans, scale_out, loadings, vars, vars ./ total)
end
