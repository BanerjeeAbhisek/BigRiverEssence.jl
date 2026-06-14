module BigRiverSchneider


using LinearAlgebra, Statistics, Random

include("pca.jl")
export pca, pcaStructure, pca_transform, pca_invtransform

include("pmd.jl") 
export  pmd, pmd_orth

include("plskern.jl") 
export  plskern, Plsr, plskerncoef, plskernpredict, plskerntransform

include("jive.jl") 
export  JiveResult, jive, jive_fast, jive_similar, jive_rjive





end 
