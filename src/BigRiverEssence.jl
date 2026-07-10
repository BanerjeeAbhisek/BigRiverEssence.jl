module BigRiverEssence


using LinearAlgebra, Statistics, Random


include("utils.jl")

include("pca.jl")
export PcaStructure, pca, pca_transform, pca_invtransform

include("pmd.jl")
export PmdStructure, pmd

include("spc.jl")
export SpcStructure, spc, spc_orth

include("plskern.jl")
export PlskernStructure, plskern, plskern_coef, plskern_predict, plskern_transform

include("jive.jl")
export JiveStructure, jive

include("splsda.jl")
export SplsdaStructure, splsda

include("cca.jl")
export CcaStructure, cca, cca_transform

include("scca.jl")
export SccaStructure, scca




end
