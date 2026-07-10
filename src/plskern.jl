"""
	Plskern{T}

Container for a fitted kernel PLS regression model, as returned by `plskern`.
Holds the standard PLS factor matrices and the centering/scaling statistics
# Fields
- `W::Matrix{T}`: The p×nlv weight matrix (unit-norm columns)
- `P::Matrix{T}`: The p×nlv X-loadings
- `Q::Matrix{T}`: The q×nlv Y-loadings
- `R::Matrix{T}`: The p×nlv weight matrix in the deflated basis, so that scores
  satisfy T = Xc·R (the SIMPLS-style weights)
- `T::Matrix{T}`: The n×nlv X-scores (orthogonal columns)
- `xmeans::Vector{T}`: The column means of X removed during centering
- `xscales::Vector{T}`: The column scales of X (standard deviations if
  `standardize=true`, otherwise ones)
- `ymeans::Vector{T}`: The column means of Y removed during centering
- `yscales::Vector{T}`: The column scales of Y
"""
struct Plskern{T}
	W::Matrix{T}
	P::Matrix{T}
	Q::Matrix{T}
	R::Matrix{T}
	T::Matrix{T}      # X-scores (named Tt inside plskern to avoid clashing with the type param)
	xmeans::Vector{T}
	xscales::Vector{T}
	ymeans::Vector{T}
	yscales::Vector{T}
end

"""
	plskern(X::Matrix{Float64}, Y::Matrix{Float64}; nlv::Int = 2,
			standardize::Bool = false, method::Symbol = :algo1)

Fit a kernel partial least squares (PLS) regression using the improved algorithms
of Dayal & MacGregor (1997). The caller's `X` and `Y` are left untouched; centering
and scaling are done on internal copies
# Arguments
- `X::Matrix{Float64}`: 2d array of floats; the n×p predictor matrix
- `Y::Matrix{Float64}`: 2d array of floats; the n×q response matrix
- `nlv::Int`: The number of latent variables; clamped to min(nlv, n, p). Defaults to 2
- `standardize::Bool`: Whether to scale each column to unit standard deviation in
  addition to centering. Defaults to false (center only)
- `method::Symbol`: `:algo1` (kernel via XᵀY) or `:algo2` (kernel via XᵀX,
  faster when p ≪ n). Both give identical results. Defaults to `:algo1`
# Value
A `Plskern` holding the weight (W, R), loading (P, Q), and score (T)
matrices together with the centering means and scales
"""
function plskern(X::Matrix{Float64}, Y::Matrix{Float64}; nlv = 2,
	standardize = false, method = :algo1)
	n, p = size(X)                        # n samples, p predictors
	q    = size(Y, 2)                     # q responses (1 for a single y)
	nlv  = min(nlv, n, p)                 # you can't ask for more components than the data supports
	method in (:algo1, :algo2) || throw(ArgumentError("method must be :algo1 or :algo2, you entered :$method"))
	T_ = Float64

	# Centering/scaling stats from the supplied data. Standardizing scales each
	# column to unit SD; otherwise the "scale" is just 1, so dividing by it is a no-op.
	xmeans  = vec(mean(X, dims = 1))
	ymeans  = vec(mean(Y, dims = 1))
	xscales = standardize ? vec(std(X, dims = 1)) : ones(T_, p)
	yscales = standardize ? vec(std(Y, dims = 1)) : ones(T_, q)

	# Build centered/scaled COPIES so the caller's X and Y are never modified. The
	# rest of the routine works against these copies (Xc, Yc), not the originals.
	Xc = Matrix{T_}(undef, n, p)
	Yc = Matrix{T_}(undef, n, q)
	@inbounds for j in 1:p, i in 1:n
		Xc[i, j] = (X[i, j] - xmeans[j]) / xscales[j]
	end
	@inbounds for j in 1:q, i in 1:n
		Yc[i, j] = (Y[i, j] - ymeans[j]) / yscales[j]
	end

	# PLS works off the cross-covariance between X and Y. XtY is that p×q summary —
	# how each predictor relates to each response. The algorithm chips away at it
	# one component at a time (the deflation step at the bottom of the loop).
	XtY = Xc' * Yc
	# algo2 also precomputes the p×p predictor covariance XtX up front. That lets it
	# find each loading without re-touching the big data matrix every iteration — a
	# win when there are few predictors (p small), wasted effort when p is large.
	XtX = method === :algo2 ? Xc' * Xc : nothing

	# Output slots: one column per component.
	W = zeros(T_, p, nlv);
	P = zeros(T_, p, nlv);
	Q = zeros(T_, q, nlv)
	R = zeros(T_, p, nlv);
	Tt = zeros(T_, n, nlv)

	# Scratch vectors refilled each component instead of allocated fresh — keeps the
	# per-component work allocation-free.
	w = Vector{T_}(undef, p);
	r = Vector{T_}(undef, p);
	t = Vector{T_}(undef, n)
	pbuf = Vector{T_}(undef, p);
	qbuf = Vector{T_}(undef, q)
	Xt = transpose(Xc)                    # transposed view of the centered X, handy for algo1

	for a in 1:nlv
		# Step 1: this component's direction in X-space (the weight w) — the direction
		# of strongest X–Y covariance. With one response that's just the single column
		# of XtY; with several it's the top singular vector of XtY.
		if q == 1
			@views w .= XtY[:, 1]
		else
			w .= svd(XtY).U[:, 1]
		end
		w ./= norm(w)                     # unit-length direction

		# Step 2: orthogonalize w against earlier components. r is w with the influence
		# of every previous component subtracted out, so the new scores don't just
		# rediscover structure we've already peeled off.
		copyto!(r, w)
		for j in 1:(a-1)
			r .-= dot(@view(P[:, j]), w) .* @view(R[:, j])
		end

		# Step 3: project the samples onto r to get the scores t and the X-loading p.
		# tt is the squared length of the scores, which normalizes everything below.
		if method === :algo1
			mul!(t, Xc, r);
			tt = dot(t, t)             # t = Xc·r, then ‖t‖²
			mul!(pbuf, Xt, t);
			pbuf ./= tt            # loading p = how each predictor projects onto t
			@views Tt[:, a] .= t
		else
			mul!(pbuf, XtX, r);
			tt = dot(r, pbuf);
			pbuf ./= tt   # same p and ‖t‖², but via XtX (no full Xc·r needed for the loading)
			mul!(t, Xc, r);
			@views Tt[:, a] .= t                  # still form the scores t themselves to store
		end

		# Step 4: the matching loading on the Y side.
		mul!(qbuf, transpose(XtY), r);
		qbuf ./= tt

		# Step 5: deflate — remove what this component explained from XtY, so the next
		# iteration searches only what's left. This is the heart of how PLS peels off
		# components one at a time.
		BLAS.ger!(-tt, pbuf, qbuf, XtY)               # XtY ← XtY − tt·(p·qᵀ)

		# stash this component's vectors and move on
		@views W[:, a] .= w;
		@views P[:, a] .= pbuf
		@views Q[:, a] .= qbuf;
		@views R[:, a] .= r
	end

	return Plskern{T_}(W, P, Q, R, Tt, xmeans, xscales, ymeans, yscales)
end

"""
	plskern_coef(m::Plskern; nlv::Int = size(m.R, 2))

Assemble the regression coefficient matrix and intercept from a fitted PLS model
# Arguments
- `m::Plskern`: A fitted PLS model, as returned by `plskern`
- `nlv::Int`: The number of latent variables to include; clamped to the number
  fitted. Defaults to all fitted components
# Value
A tuple `(B, intercept)` where `B` is the p×q coefficient matrix and `intercept`
is the 1×q intercept, such that a prediction is `intercept .+ Xnew * B`. The
scales are folded in so B maps raw (uncentered, unscaled) X to Y
"""
function plskern_coef(m::Plskern; nlv = size(m.R, 2))
	nlv = min(nlv, size(m.R, 2))
	# B in the original (raw) units: undo the X-scaling on R, recombine with Q, reapply the Y-scaling.
	B = (m.R[:, 1:nlv] ./ m.xscales) * (m.Q[:, 1:nlv]') .* m.yscales'
	intercept = m.ymeans' .- m.xmeans' * B            # absorb the centering into the intercept
	return B, intercept
end

"""
	plskern_predict(m::Plskern, Xnew::Matrix{Float64}; nlv::Int = size(m.R, 2))

Predict responses for new observations from a fitted PLS model
# Arguments
- `m::Plskern`: A fitted PLS model, as returned by `plskern`
- `Xnew::Matrix{Float64}`: 2d array of floats; the new observations (rows) by
  predictors (columns), with the same p predictors as the training data
- `nlv::Int`: The number of latent variables to use; clamped to the number
  fitted. Defaults to all fitted components
# Value
2d array of floats; the predicted n×q responses, `intercept .+ Xnew * B`
"""
function plskern_predict(m::Plskern, Xnew; nlv = size(m.R, 2))
	Xnew = Matrix{Float64}(Xnew)
	B, intercept = plskern_coef(m; nlv = nlv)
	return intercept .+ Xnew * B                       # apply the coefficient model
end

"""
	plskern_transform(m::Plskern, Xnew::Matrix{Float64}; nlv::Int = size(m.R, 2))

Project new observations onto the PLS latent space (compute their X-scores)
# Arguments
- `m::Plskern`: A fitted PLS model, as returned by `plskern`
- `Xnew::Matrix{Float64}`: 2d array of floats; the new observations (rows) by
  predictors (columns), with the same p predictors as the training data
- `nlv::Int`: The number of latent variables to project onto; clamped to the
  number fitted. Defaults to all fitted components
# Value
2d array of floats; the n×nlv matrix of X-scores. For the training data this
reproduces the stored scores `m.T`, since scores are linear: T = Xc·R
"""
function plskern_transform(m::Plskern, Xnew; nlv = size(m.R, 2))
	Xnew = Matrix{Float64}(Xnew)
	nlv  = min(nlv, size(m.R, 2))
	Xc   = (Xnew .- m.xmeans') ./ m.xscales'           # center and scale with the stored stats
	return Xc * m.R[:, 1:nlv]                          # scores = Xc·R
end