module BigRiverSchneider


using LinearAlgebra, Statistics, Random


include("utils.jl")

include("pca.jl")
export pcaStructure, pca, pca_transform, pca_invtransform

include("pmd.jl")
export pmdStructure, pmd

include("spc.jl")
export spcStructure, spc, spc_orth

include("plskern.jl")
export plskernStructure, plskern, plskerncoef, plskernpredict, plskerntransform

include("jive.jl")
export jiveStructure, jive

include("plsda.jl")
export PlsdaStructure, plsda


include("splsda.jl")
export splsdaStructure, splsda

include("cca.jl")
export ccaStructure, cca, cca_transform

include("scca.jl")
export sccaStructure, scca




end
