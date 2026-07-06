# benchmark_struct_specialization.jl
# Demonstrates that a parametric struct {T} and a Float64-hardcoded struct compile
# to equivalent code and run at the same speed when used with Float64.

using LinearAlgebra, Statistics, BenchmarkTools, Random

# ---------------------------------------------------------------------------
# Version 1: parametric struct {T}
# ---------------------------------------------------------------------------
struct str1{T}
    mean::Vector{T}
    scale::Vector{T}
    loadings::Matrix{T}
    variances::Vector{T}
    propOFvar::Vector{T}
end

# ---------------------------------------------------------------------------
# Version 2: identical fields, T replaced by concrete Float64
# ---------------------------------------------------------------------------
struct str2
    mean::Vector{Float64}
    scale::Vector{Float64}
    loadings::Matrix{Float64}
    variances::Vector{Float64}
    propOFvar::Vector{Float64}
end

# Standalone sign-fixing helper so the script runs without the package
function sign_fix!(V)
    @inbounds for j in 1:size(V, 2)
        col = @view V[:, j]
        idx = argmax(abs.(col))
        col[idx] < 0 && (col .*= -1)
    end
    return V
end

# ---------------------------------------------------------------------------
# The two fit functions: IDENTICAL body, differing ONLY in the struct returned
# ---------------------------------------------------------------------------
function cca1(X::Matrix{Float64}; k::Int = minimum(size(X)), standardize::Bool = false)
    n, p = size(X)
    colmeans = vec(mean(X, dims = 1))
    colstds  = standardize ? vec(std(X, dims = 1)) : Float64[]

    scatter = mul!(Matrix{Float64}(undef, p, p), transpose(X), X)
    @inbounds for j in 1:p, i in 1:p
        scatter[i, j] -= n * colmeans[i] * colmeans[j]
    end
    if standardize
        @inbounds for j in 1:p, i in 1:p
            scatter[i, j] /= (colstds[i] * colstds[j])
        end
    end
    scatter_sym = Symmetric(scatter)
    total = tr(scatter_sym) / (n - 1)
    topk = eigen(scatter_sym, (p-k+1):p)
    vars = reverse(topk.values) ./ (n - 1)
    loadings = topk.vectors[:, k:-1:1]
    sign_fix!(loadings)
    scale_out = standardize ? colstds : ones(Float64, p)

    return str1{Float64}(colmeans, scale_out, loadings, vars, vars ./ total)   # ← parametric
end

function cca2(X::Matrix{Float64}; k::Int = minimum(size(X)), standardize::Bool = false)
    n, p = size(X)
    colmeans = vec(mean(X, dims = 1))
    colstds  = standardize ? vec(std(X, dims = 1)) : Float64[]

    scatter = mul!(Matrix{Float64}(undef, p, p), transpose(X), X)
    @inbounds for j in 1:p, i in 1:p
        scatter[i, j] -= n * colmeans[i] * colmeans[j]
    end
    if standardize
        @inbounds for j in 1:p, i in 1:p
            scatter[i, j] /= (colstds[i] * colstds[j])
        end
    end
    scatter_sym = Symmetric(scatter)
    total = tr(scatter_sym) / (n - 1)
    topk = eigen(scatter_sym, (p-k+1):p)
    vars = reverse(topk.values) ./ (n - 1)
    loadings = topk.vectors[:, k:-1:1]
    sign_fix!(loadings)
    scale_out = standardize ? colstds : ones(Float64, p)

    return str2(colmeans, scale_out, loadings, vars, vars ./ total)            # ← concrete
end

# ---------------------------------------------------------------------------
# Correctness check: both must produce identical results
# ---------------------------------------------------------------------------
Random.seed!(1)
X = randn(500, 50)
k = 10

r1 = cca1(X; k = k)
r2 = cca2(X; k = k)
println("Max loadings diff : ", maximum(abs.(r1.loadings .- r2.loadings)))
println("Max variances diff: ", maximum(abs.(r1.variances .- r2.variances)))
println("Identical results : ",
    r1.loadings == r2.loadings && r1.variances == r2.variances)
println()

# ---------------------------------------------------------------------------
# The benchmark with @btime
# ---------------------------------------------------------------------------
println("="^55)
println("Benchmark: parametric str1{T} vs concrete str2")
println("="^55)

print("cca1 (parametric {T})   : ")
@btime cca1($X; k = $k);

print("cca2 (concrete Float64) : ")
@btime cca2($X; k = $k);

println("="^55)

# ---------------------------------------------------------------------------
# For a fair ratio, capture full samples too (median, not the @btime minimum)
# ---------------------------------------------------------------------------
b1 = @benchmark cca1($X; k = $k)
b2 = @benchmark cca2($X; k = $k)
t1 = median(b1).time; t2 = median(b2).time
println("Median cca1 : ", round(t1/1e3, digits = 2), " μs")
println("Median cca2 : ", round(t2/1e3, digits = 2), " μs")
println("Ratio (cca1/cca2): ", round(t1/t2, digits = 4),
        "   (", round(100 * abs(t1 - t2)/min(t1,t2), digits = 2), "% apart)")
println("="^55)
