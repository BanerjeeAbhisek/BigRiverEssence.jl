
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


