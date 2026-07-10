module BigRiverEssence


using LinearAlgebra, Statistics, Random


include("utils.jl")

include("pca.jl")
export Pca, pca, pca_transform, pca_invtransform

include("pmd.jl")
export Pmd, pmd

include("spc.jl")
export Spc, spc, spc_orth

include("plskern.jl")
export Plskern, plskern, plskern_coef, plskern_predict, plskern_transform

include("jive.jl")
export Jive, jive

include("plsda.jl")
export Plsda, plsda

include("splsda.jl")
export Splsda, splsda

include("cca.jl")
export Cca, cca, cca_transform

include("scca.jl")
export Scca, scca




end
