using BigRiverEssence
using Documenter

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

makedocs(; modules=[BigRiverEssence], sitename="BigRiverEssence.jl", pages=[
        "Home" => "index.md",
        "Principal Component Analysis" => "pca_tutorial.md",
        "Penalized Matrix Decomposition" => "pmd_tutorial.md",
        "Sparse Principal Component Analysis" => "spc_tutorial.md",
        "Partial Least Squares kernel Regression" => "plskern_tutorial.md",
        "Sparse Partial Least Squares Discriminant Analysis" => "splsda_tutorial.md",
        "Joint and Individual Variation Explained" => "jive_tutorial.md",
        "Canonical Correlation Analysis" => "cca_tutorial.md",
        "Sparse Canonical Correlation Analysis" => "scca_tutorial.md",
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
