using BigRiverEssence
using Documenter

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

makedocs(; modules=[BigRiverEssence], sitename="BigRiverEssence.jl", pages=[
        "Home" => "index.md",
        "Principal Component Analysis (PCA)" => "pca_tutorial.md",
        "Penalized Matrix Decomposition (PMD)" => "pmd_tutorial.md",
        "Sparse Principal Component Analysis (SPC)" => "spc_tutorial.md",
        "Partial Least Squares kernel Regression (PLSkern)" => "plskern_tutorial.md",
        "Partial Least Squares Discriminant Analysis (PLSDA)" => "plsda_tutorial.md",
        "Sparse Partial Least Squares Discriminant Analysis (SPLSDA)" => "splsda_tutorial.md",
        "Joint and Individual Variation Explained (JIVE)" => "jive_tutorial.md",
        "Canonical Correlation Analysis (CCA)" => "cca_tutorial.md",
        "Sparse Canonical Correlation Analysis (SCCA)" => "scca_tutorial.md",
        "API Reference" => "api.md", 
        # "Example: MLM for ordinal predictors" => "example_ordinal_data.md",
        # "Types and Functions" => "functions.md",
    ]
)

deploydocs(;
    repo = "github.com/senresearch/BigRiverEssence.jl.git",
    devbranch= "main",
    devurl = "dev"
)
