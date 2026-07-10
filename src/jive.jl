"""
	JiveStructure{T}

Container for a fitted JIVE (Joint and Individual Variation Explained)
decomposition, as returned by `jive`
# Fields
- `J::Vector{Matrix{T}}`: The joint structure, one pᵢ×n matrix per data block;
  all blocks share the same n-dimensional joint row space
- `A::Vector{Matrix{T}}`: The individual structure, one pᵢ×n matrix per block,
  orthogonal to the joint structure
- `S::Matrix{T}`: The r×n joint scores (orthonormal rows), the basis of the
  shared joint row space
- `U::Vector{Matrix{T}}`: The joint loadings, one pᵢ×r block, so that J[i] = U[i]·S
- `Si::Vector{Matrix{T}}`: The individual scores, one rᵢ×n matrix per block
- `Wi::Vector{Matrix{T}}`: The individual loadings, one pᵢ×rᵢ matrix per block,
  so that A[i] = Wi[i]·Si[i]
- `r::Int`: The joint rank (dimension of the shared structure)
- `ri::Vector{Int}`: The individual ranks, one per block
"""
struct JiveStructure{T}
	J::Vector{Matrix{T}}
	A::Vector{Matrix{T}}
	S::Matrix{T}
	U::Vector{Matrix{T}}
	Si::Vector{Matrix{T}}
	Wi::Vector{Matrix{T}}
	r::Int
	ri::Vector{Int}
end

"""
	_safe_svd(A)

Compute an SVD, falling back to a slower but more robust algorithm if the
default one fails to converge
# Arguments
- `A`: 2d array of floats; the matrix to decompose
# Value
An `SVD` factorization object. Tries the default divide-and-conquer SVD first;
if it raises a `LAPACKException` (a convergence failure on certain ill-conditioned
inputs), retries with the QR-iteration algorithm, which is slower but more stable.
Any other error is rethrown
"""
function _safe_svd(A)
	try
		return svd(A)
	catch e
		e isa LinearAlgebra.LAPACKException || rethrow()   # only catch convergence failures
		return svd(A; alg = LinearAlgebra.QRIteration())   # robust fallback
	end
end

"""
	_safe_svdvals(A)

Compute singular values, falling back to a more robust algorithm if the default
one fails to converge
# Arguments
- `A`: 2d array of floats; the matrix whose singular values are wanted
# Value
A vector of singular values. Like `_safe_svd`, retries with QR iteration on a
`LAPACKException` and rethrows anything else
"""
function _safe_svdvals(A)
	try
		return svdvals(A)
	catch e
		e isa LinearAlgebra.LAPACKException || rethrow()
		return svdvals(A; alg = LinearAlgebra.QRIteration())
	end
end

"""
	_safe_svd!(A)

In-place variant of `_safe_svd`: compute an SVD, overwriting `A`, with a robust
fallback on convergence failure
# Arguments
- `A`: 2d array of floats; the matrix to decompose, OVERWRITTEN in place by the
  default path
# Value
An `SVD` factorization object. Tries the in-place `svd!` first; on a
`LAPACKException` retries with the (non-mutating) QR-iteration SVD. Used in the
inner JIVE iterations where the input is scratch that can be destroyed
"""
function _safe_svd!(A)
	try
		return svd!(A)
	catch e
		e isa LinearAlgebra.LAPACKException || rethrow()
		return svd(A; alg = LinearAlgebra.QRIteration())
	end
end

"""
	_jive_rjive_core_opt2(Xc::Vector{Matrix{Float64}}, n::Int, r::Int,
						  ri::Vector{Int}; conv::Float64, maxiter::Int)

Compute the JIVE decomposition for GIVEN joint and individual ranks, by
alternating between estimating the joint and individual structure
# Arguments
- `Xc::Vector{Matrix{Float64}}`: The preprocessed (centered/scaled) data blocks,
  each pᵢ×n with a shared column dimension n
- `n::Int`: The number of columns (observations) shared across blocks
- `r::Int`: The joint rank
- `ri::Vector{Int}`: The individual ranks, one per block
- `conv::Float64`: The convergence threshold on the change in J and A between
  iterations
- `maxiter::Int`: The maximum number of outer iterations
# Value
A `JiveStructure` with the joint (J, S, U) and individual (A, Si, Wi) parts.
Implements the r.jive "orthIndiv" algorithm of Lock et al. (2013): wide blocks
are first compressed to their score space (via SVD) for efficiency, then the
joint structure is the rank-r SVD of the stacked residual (data minus individual),
and each block's individual structure is the rank-rᵢ SVD of its residual
projected off the joint row space and off the other blocks' individual spaces.
The two estimates are alternated to convergence, then mapped back to the original
row space
"""
function _jive_rjive_core_opt2(Xc::Vector{Matrix{Float64}}, n::Int, r::Int, ri::Vector{Int};
	conv::Float64, maxiter::Int)
	T_ = Float64
	k = length(Xc)

	# compress tall blocks (pᵢ > n) to their n-dim score space so the iteration
	# works on small matrices; Ubig[i] maps the compressed result back to pᵢ rows
	Ubig = Vector{Matrix{T_}}(undef, k)
	Xr   = Vector{Matrix{T_}}(undef, k)
	for i in 1:k
		if size(Xc[i], 1) > size(Xc[i], 2)
			F = _safe_svd(Xc[i]);
			nc = size(Xc[i], 2)
			Xr[i] = Diagonal(F.S[1:nc]) * F.Vt[1:nc, :]    # compressed block (nc×n)
			Ubig[i] = F.U[:, 1:nc]                          # back-projection to original rows
		else
			Xr[i] = Xc[i]
			Ubig[i] = Matrix{T_}(I, size(Xc[i], 1), size(Xc[i], 1))
		end
	end

	pis = [size(X, 1) for X in Xr]
	rowranges = (
		let rr = Vector{UnitRange{Int}}(undef, k);
			idx=1
			for i in 1:k
				;
				rr[i]=idx:(idx+pis[i]-1);
				idx+=pis[i];
			end;
			rr
		end
	)
	ptot = sum(pis)

	A    = [zeros(T_, pis[i], n) for i in 1:k]      # individual structure per block
	J    = [zeros(T_, pis[i], n) for i in 1:k]      # joint structure per block
	Vind = [zeros(T_, n, ri[i]) for i in 1:k]       # each block's individual row space
	Xtot = reduce(vcat, Xr)                          # stacked compressed data

	# scratch, reused across iterations
	Jtot  = fill(-1.0, ptot, n)
	Atot  = fill(-1.0, ptot, n)
	Jlast = similar(Jtot)
	Alast = similar(Atot)
	tmpJ  = Matrix{T_}(undef, ptot, n)
	V     = Matrix{T_}(undef, n, r)                 # joint row space
	USj   = Matrix{T_}(undef, ptot, r)
	tmpi  = [Matrix{T_}(undef, pis[i], n) for i in 1:k]
	projr = [Matrix{T_}(undef, pis[i], r) for i in 1:k]

	nrun = 0;
	converged = false
	while nrun < maxiter && !converged
		copyto!(Jlast, Jtot);
		copyto!(Alast, Atot)

		#  joint update: rank-r SVD of (stacked data − individual) 
		if r > 0
			@. tmpJ = Xtot - Atot
			s = _safe_svd!(tmpJ)
			@views mul!(USj, s.U[:, 1:r], Diagonal(s.S[1:r]))
			@views mul!(Jtot, USj, s.Vt[1:r, :])             # joint = rank-r truncation
			@views copyto!(V, transpose(s.Vt[1:r, :]))       # the joint row space
		else
			fill!(Jtot, 0.0)
		end
		for i in 1:k
			@views J[i] .= Jtot[rowranges[i], :]
		end

		#  individual update: per block, rank-rᵢ SVD of the block residual,
		#     projected off the joint row space AND off other blocks' individual spaces 
		for i in 1:k
			if ri[i] > 0
				tmp = tmpi[i]
				@. tmp = Xr[i] - J[i]                        # residual after removing joint
				if r > 0
					mul!(projr[i], tmp, V)
					mul!(tmp, projr[i], transpose(V), -1.0, 1.0)   # remove joint row-space component
				end
				if nrun > 0
					for j in 1:k
						j == i && continue
						Vj = Vind[j]
						pj = tmp * Vj
						mul!(tmp, pj, transpose(Vj), -1.0, 1.0)    # remove other blocks' individual spaces
					end
				end
				s = _safe_svd!(tmp)
				@views copyto!(Vind[i], transpose(s.Vt[1:ri[i], :]))
				@views mul!(A[i], s.U[:, 1:ri[i]] * Diagonal(s.S[1:ri[i]]), s.Vt[1:ri[i], :])
			else
				fill!(A[i], 0)
			end
		end

		# on the very first pass, seed the individual row spaces by mutually
		# orthogonalizing the blocks, then re-SVD to get clean Vind for next round
		if nrun == 0
			for i in 1:k, j in 1:k
				j == i && continue
				Vj = Vind[j]
				pj = A[i] * Vj
				mul!(A[i], pj, transpose(Vj), -1.0, 1.0)
			end
			for i in 1:k
				if ri[i] > 0
					s = _safe_svd(A[i])
					@views copyto!(Vind[i], transpose(s.Vt[1:ri[i], :]))
				end
			end
		end

		for i in 1:k
			@views Atot[rowranges[i], :] .= A[i]
		end

		# converged when both J and A stop changing
		if norm(Jtot .- Jlast) <= conv && norm(Atot .- Alast) <= conv
			converged = true
		end
		nrun += 1
	end

	# map the compressed J, A back to the original row space and factor each part
	Jfull = [Ubig[i] * J[i] for i in 1:k]
	Afull = [Ubig[i] * A[i] for i in 1:k]
	Fj = _safe_svd(reduce(vcat, Jfull))
	S = Matrix(@view Fj.Vt[1:r, :])                  # joint scores (orthonormal rows)
	pis_full = [size(Ji, 1) for Ji in Jfull]
	Ufull = Fj.U[:, 1:r] * Diagonal(Fj.S[1:r])
	U = Matrix{T_}[];
	idx=1
	for p in pis_full
		;
		push!(U, Ufull[idx:(idx+p-1), :]);
		idx+=p;
	end    # split joint loadings by block
	Si = Matrix{T_}[];
	Wi = Matrix{T_}[]
	for i in 1:k
		Fi = _safe_svd(Afull[i])
		push!(Si, Matrix(@view Fi.Vt[1:ri[i], :]))                   # individual scores
		push!(Wi, Fi.U[:, 1:ri[i]] * Diagonal(Fi.S[1:ri[i]]))         # individual loadings
	end
	return JiveStructure{T_}(Jfull, Afull, S, U, Si, Wi, r, ri)
end

"""
	_jive_perm_ranks_opt(Xc::Vector{Matrix{Float64}}, n::Int; nperm::Int,
						 alpha::Float64, conv::Float64, maxiter::Int,
						 maxrounds::Int = 10)

Estimate the joint and individual ranks by a permutation test
# Arguments
- `Xc::Vector{Matrix{Float64}}`: The preprocessed data blocks, each pᵢ×n
- `n::Int`: The shared number of columns (observations)
- `nperm::Int`: The number of permutations used to build each null distribution
- `alpha::Float64`: The significance level; a singular value counts as signal if
  it exceeds the (1−alpha) quantile of the permuted null
- `conv::Float64`: The convergence threshold passed to the core fit
- `maxiter::Int`: The maximum core-fit iterations
- `maxrounds::Int`: The maximum number of estimate-then-refit rounds. Defaults to 10
# Value
A tuple `(rJ, rA)` of the estimated joint rank and the vector of individual
ranks. The joint rank is the number of leading singular values of the stacked
(data − individual) residual that exceed the column-permuted null; each
individual rank is found the same way on each block's (data − joint) residual.
Ranks and the fit are alternated until the rank estimates stabilize, following
the r.jive permutation procedure of Lock et al. (2013)
"""
function _jive_perm_ranks_opt(Xc::Vector{Matrix{Float64}}, n::Int;
	nperm::Int, alpha::Float64, conv::Float64,
	maxiter::Int, maxrounds::Int = 10)
	k = length(Xc)
	Jperp = [zeros(size(Xc[i])) for i in 1:k]       # current joint estimate (for the A-residual)
	Aperp = [zeros(size(Xc[i])) for i in 1:k]       # current individual estimate (for the J-residual)
	last = fill(-2, k+1);
	current = fill(-1, k+1)
	rJ = 0;
	rA = zeros(Int, k);
	nrun = 0

	ptot      = sum(size(X, 1) for X in Xc)
	fullstack = Matrix{Float64}(undef, ptot, n)     # reused permutation buffer (joint test)
	permcols  = Vector{Int}(undef, n)

	while last != current && nrun < maxrounds
		last = copy(current)

		#  joint rank: compare actual SVs of the stacked (data − individual)
		#     against a column-permuted null (permuting breaks shared structure) 
		full = [Xc[i] .- Aperp[i] for i in 1:k]
		actual = __safe_svdvals(reduce(vcat, full))
		nsv = min(n, ptot)
		perms = zeros(nperm, nsv)
		rowr = (
			let rr=Vector{UnitRange{Int}}(undef, k);
				idx=1
				for i in 1:k
					;
					rr[i]=idx:(idx+size(full[i], 1)-1);
					idx+=size(full[i], 1);
				end;
				rr
			end
		)
		for p in 1:nperm
			for i in 1:k
				randperm!(permcols)                          # permute each block's columns independently
				@views fullstack[rowr[i], :] .= full[i][:, permcols]
			end
			sv = __safe_svdvals(fullstack)
			m = min(length(sv), nsv)
			@views perms[p, 1:m] .= sv[1:m]
		end
		rJ = 0
		for i in 1:nsv
			actual[i] > quantile(@view(perms[:, i]), 1-alpha) ? (rJ += 1) : break   # count SVs above the null
		end
		rJ = max(rJ, last[1])                                # ranks only grow across rounds

		#  individual ranks: same test per block on its (data − joint) residual 
		for i in 1:k
			ind = Xc[i] .- Jperp[i]
			pi_ = size(ind, 1)
			actual_i = __safe_svdvals(ind)
			nsv_i = min(n, pi_)
			perms_i = zeros(nperm, nsv_i)
			permbuf = Matrix{Float64}(undef, pi_, n)
			for p in 1:nperm
				for row in 1:pi_
					randperm!(permcols)                      # permute within each row
					@views permbuf[row, :] .= ind[row, permcols]
				end
				sv = __safe_svdvals(permbuf)
				m = min(length(sv), nsv_i)
				@views perms_i[p, 1:m] .= sv[1:m]
			end
			ra = 0
			for j in 1:nsv_i
				actual_i[j] > quantile(@view(perms_i[:, j]), 1-alpha) ? (ra += 1) : break
			end
			rA[i] = ra
		end

		current = vcat(rJ, rA)

		# refit at the new ranks so the next round's residuals reflect them
		if last != current && rJ > 0
			fit = _jive_rjive_core_opt2(Xc, n, rJ, rA; conv = conv, maxiter = maxiter)
			Jperp = fit.J;
			Aperp = fit.A
		end
		nrun += 1
	end
	return rJ, rA
end

"""
	jive(Xs::Vector{Matrix{Float64}}; r = nothing, ri = nothing, scale = true,
		 center = true, tol = nothing, maxiter = 1000, nperm = 100, alpha = 0.05)

Fit a JIVE (Joint and Individual Variation Explained) decomposition of several
data blocks sharing a common set of observations
# Arguments
- `Xs::Vector{Matrix{Float64}}`: The data blocks, each pᵢ×n with the SAME number
  of columns n (observations); rows (features) may differ per block
- `r`: The joint rank; if nothing, estimated by permutation test. Defaults to nothing
- `ri`: The vector of individual ranks; if nothing, estimated by permutation test.
  Defaults to nothing
- `scale::Bool`: Whether to scale each block by its Frobenius norm (and √(total
  elements)) so blocks contribute comparably. Defaults to true
- `center::Bool`: Whether to row-center each block. Defaults to true
- `tol`: The convergence threshold; if nothing, set to 1e-6·‖stacked data‖.
  Defaults to nothing
- `maxiter::Int`: The maximum core-fit iterations. Defaults to 1000
- `nperm::Int`: The permutations per rank test (when ranks are estimated).
  Defaults to 100
- `alpha::Float64`: The significance level for the rank test. Defaults to 0.05
# Value
A `JiveStructure` with the joint and individual decompositions. Each block is
first row-centered and Frobenius-scaled so no block dominates by sheer magnitude.
If the joint and individual ranks are not supplied, they are estimated by a
permutation test; the decomposition is then computed by alternating between
estimating the shared joint structure and each block's individual structure until
they converge. The result splits each block into joint structure (a common row
space across all blocks) and block-specific individual structure orthogonal to
it, after Lock et al. (2013)
"""
function jive(Xs::Vector{Matrix{Float64}};
	r = nothing, ri = nothing,
	scale = true, center = true, tol = nothing,
	maxiter = 1000, nperm = 100, alpha = 0.05)
	k = length(Xs)
	n = size(Xs[1], 2)
	all(size(X, 2) == n for X in Xs) || throw(ArgumentError("all datasets need the same number of columns"))

	# preprocess: row-center, then Frobenius-scale so no block dominates by sheer magnitude
	nel = [size(X, 1)*size(X, 2) for X in Xs];
	sum_n = sum(nel)
	Xc = Vector{Matrix{Float64}}(undef, k)
	for i in 1:k
		Xi = center ? Xs[i] .- mean(Xs[i], dims = 2) : copy(Xs[i])   # copy keeps the caller's block intact
		scale && (Xi ./= (norm(Xi) * sqrt(sum_n)))
		Xc[i] = Xi
	end
	conv = tol === nothing ? 1e-6 * norm(reduce(vcat, Xc)) : tol

	# estimate ranks if the caller didn't supply them
	if r === nothing || ri === nothing
		println("Estimating ranks via permutation test...")
		r, ri = _jive_perm_ranks_opt(Xc, n; nperm = nperm, alpha = alpha, conv = conv, maxiter = maxiter)
		println("Estimated joint rank: $r, individual ranks: $ri")
	end

	return _jive_rjive_core_opt2(Xc, n, r, ri; conv = conv, maxiter = maxiter)
end

"""
	jive(Xs::Vector{Matrix{Float64}}, r::Int, ri::Vector{Int}; kwargs...)

Positional-argument form of `jive` for supplying the joint rank `r` and
individual ranks `ri` directly
# Arguments
- `Xs::Vector{Matrix{Float64}}`: The data blocks, each pᵢ×n
- `r::Int`: The joint rank
- `ri::Vector{Int}`: The individual ranks, one per block
- `kwargs...`: Any keyword arguments accepted by the keyword form of `jive`
# Value
A `JiveStructure`; identical to calling `jive(Xs; r = r, ri = ri, kwargs...)`
"""
jive(Xs::Vector{Matrix{Float64}}, r::Int, ri::Vector{Int}; kwargs...) =
	jive(Xs; r = r, ri = ri, kwargs...)
