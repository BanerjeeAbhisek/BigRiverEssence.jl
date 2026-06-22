module BigRiverSchneider


using LinearAlgebra, Statistics, Random

include("pca.jl")
export pca, pcaStructure, pca_transform, pca_invtransform, pca_opt

include("pmd.jl") 
export  pmd, pmd_orth

include("plskern.jl") 
export  plskern, Plsr, plskerncoef, plskernpredict, plskerntransform

include("jive.jl") 
export  JiveResult, jive, jive_fast, jive_rjive

include("splsda.jl")
export SplsdaResult, splsda

include("cca.jl")
export CcaResult, cca, cca_transform





# Optimized functions
include("pca_opt.jl")
export pca_opt

include("pmd_opt.jl")
export pmd_opt, pmd_orth_opt

include("plskern_opt.jl")
export plskern_opt

include("jive_opt.jl")
export jive_rjive_opt

include("splsda_opt.jl")
export splsda_opt

include("cca_opt.jl")
export cca_opt


end 
