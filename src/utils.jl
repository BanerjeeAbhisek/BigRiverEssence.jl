
"""
	_sign_consistency_opt!(V)

Fix the sign of each column of a loading matrix so the largest-magnitude entry is
positive, in place
# Arguments
- `V`: 2d array of floats; the loading/direction matrix whose columns are sign-fixed
  in place
# Value
The matrix `V`, with each column multiplied by ±1 so that its entry of largest
absolute value is positive. SVD- and eigen-derived directions are only determined
up to a sign, so this canonicalizes the choice to make results reproducible across
runs and comparable across implementations. An all-zero column (whose largest
entry has sign 0) is left untouched
"""
function _sign_consistency_opt!(V)
	@inbounds for c in eachcol(V)
		mi = 1;
		mv = abs(c[1])
		for i in 2:length(c)                 # find the index of the largest-magnitude entry
			a = abs(c[i])
			if a > mv
				;
				mv = a;
				mi = i;
			end
		end
		s = sign(c[mi])
		s != 0 && (c .*= s)                  # flip the whole column if that entry is negative
	end
	return V
end


"""
	vip(m)

Compute Variable Importance in Projection (VIP) scores for a fitted PLS
discriminant model
# Arguments
- `m`: a fitted model (`Plsda` or `Splsda`) exposing the X-loadings
  (`loadings_X`), the X-variates (`variates_X`), the dummy-encoded response
  (`Y_dummy`), and the number of components (`ncomp`)
# Value
A p×ncomp matrix of VIP scores, where p is the number of variables (rows of the
X-loadings). Entry `[j, h]` is the importance of variable j using the first h
components; the last column is the cross-component VIP that summarizes each
variable's contribution over all components. The scores are normalized so that the
mean squared VIP equals 1, so a value above 1 marks an above-average-importance
variable — the conventional selection threshold. This follows the VIP definition of
Wold et al. as implemented in `mixOmics::vip`: each component's contribution is
weighted by the redundancy (the squared correlation between the dummy response and
that component's variate) it explains, so components that better separate the
classes count more toward the score
"""
function vip(m)
	W = m.loadings_X
	ncomp = m.ncomp
	Y = m.Y_dummy
	p = size(W, 1)
	VIP = zeros(p, ncomp)
	cor2 = reshape(cor(Y, m.variates_X) .^ 2, size(Y, 2), ncomp)
	VIP[:, 1] .= W[:, 1] .^ 2
	for h in 2:ncomp
		Rd = vec(sum(cor2[:, 1:h], dims = 1))
		VIP[:, h] = (W[:, 1:h] .^ 2 * Rd) ./ sum(Rd)
	end
	return sqrt.(p .* VIP)
end