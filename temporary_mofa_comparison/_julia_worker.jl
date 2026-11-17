using BigRiverEssence, BenchmarkTools, DelimitedFiles, LinearAlgebra, Statistics, TOML

"""
    _save_array(directory, name, values)

Save one numerical comparison array in Julia's column-major order
# Arguments
- `directory`: Destination folder, created when necessary
- `name`: Array identifier without an extension
- `values`: Numerical array; original dimensions are described by its identifier
# Value
Nothing; writes a single-column CSV outside all timed/profiled expressions
"""
function _save_array(directory, name, values)
    mkpath(directory)
    writedlm(joinpath(directory, name * ".csv"), vec(values), ',')
    open(joinpath(directory, "shapes.toml"), "a") do io
        TOML.print(io, Dict(name => collect(size(values))))
    end
    return nothing
end

"""
    _export_result(model::Mofa, directory)

Save posterior parameters, reconstructions, and training diagnostics
# Arguments
- `model::Mofa`: Fitted model to compare with mofapy2
- `directory`: Destination folder for named, flattened numerical arrays
# Value
Nothing. Reads fitted arrays without changing them; excludes all file I/O
and reconstruction multiplication from the timing experiment
"""
function _export_result(model::Mofa, directory)
    s = model.posterior
    for (name, array) in (("z_mean", model.z), ("z_variance", s.z.variance),
        ("z_second_moment", s.z.second_moment), ("elbo", model.elbo),
        ("elbo_iterations", model.elbo_iterations), ("r2", model.r2),
        ("factor_r2", model.factor_r2))
        _save_array(directory, name, array)
    end
    for m in eachindex(model.w)
        for (name, array) in (("w_mean", s.w[m].mean), ("w_variance", s.w[m].variance),
            ("w_second_moment", s.w[m].second_moment), ("tau_shape", s.tau_shape[m]),
            ("tau_rate", s.tau_rate[m]), ("tau_mean", s.tau_mean[m]),
            ("tau_logmean", s.tau_logmean[m]),
            ("reconstruction", model.z * transpose(model.w[m])))
            _save_array(directory, "$(name)_$m", array)
        end
        if s.spike_w !== nothing
            q = s.spike_w[m]
            for (name, array) in (("inclusion", q.inclusion), ("slab_mean", q.mean),
                ("slab_variance", q.variance), ("inactive_variance", q.inactive_variance))
                _save_array(directory, "$(name)_$m", array)
            end
        end
    end
    for (name, q) in (("alpha_w", s.alpha_w), ("alpha_z", s.alpha_z))
        q === nothing && continue
        for field in (:shape, :rate, :mean, :logmean)
            _save_array(directory, "$(name)_$field", getfield(q, field))
        end
    end
    if s.theta_w !== nothing
        for field in (:a, :b, :mean, :logmean, :logcomplement)
            _save_array(directory, "theta_w_$field", getfield(s.theta_w, field))
        end
    end
    return nothing
end

"""
    _wait_for_file(path; timeout=1800.0)

Wait for the external memory sampler's synchronization signal
# Arguments
- `path`: Exact file that the supervising process will create
- `timeout`: Maximum waiting time in seconds
# Value
Nothing. Sleeps briefly between checks and throws an error on timeout
"""
function _wait_for_file(path; timeout = 1800.0)
    started = time()
    while !isfile(path)
        time() - started < timeout || error("Timed out waiting for $path")
        sleep(0.001)
    end
    return nothing
end

"""
    _main(config_path, job_directory, mode)

Run one warmed Julia timing or externally sampled memory experiment
# Arguments
- `config_path`: TOML configuration naming shared inputs and model settings
- `job_directory`: Isolated directory for this child process's results
- `mode`: `bench` for BenchmarkTools or `rss` for one post-warm-up fit
# Value
Nothing; writes metrics.toml and, for bench mode, numerical output arrays.
Calls the public mofa function directly inside @benchmarkable, not a fitting wrapper
"""
function _main(config_path, job_directory, mode)
    config = TOML.parsefile(config_path)
    realpath(pkgdir(BigRiverEssence)) == realpath(config["repository"]) ||
        error("Julia is not using the requested local checkout. Run Pkg.develop as described in README.md.")
    BLAS.set_num_threads(config["threads"])
    base_options = (center_groups = true, scale_views = false, scale_groups = false,
        ard_weights = config["ard_weights"], ard_factors = config["ard_factors"],
        spikeslab_weights = config["spikeslab_weights"], start_elbo = 1,
        freq_elbo = config["freq_elbo"], start_sparsity = config["start_sparsity"],
        forceiter = true, verbose = false, init = :custom)
    if mode == "rss"
        # Warm a separate tiny problem before loading the target, as in the
        # benchmark repository's isolated NMF memory worker.
        warm_views = [Float64[sin(i + 2j) for i in 1:8, j in 1:p] for p in (6, 4)]
        warm_z = Float64[sin(i + j) for i in 1:8, j in 1:2]
        warm_options = merge(base_options, (start_sparsity = 1, freq_elbo = 1))
        mofa(warm_views; warm_options..., k = 2, groups = repeat([0, 1]; inner = 4),
            z_init = warm_z, maxiter = 2)
        warm_views = nothing
        warm_z = nothing
        GC.gc()
    end
    data_directory = config["data_directory"]
    views = [readdlm(joinpath(data_directory, "X$m.csv"), ',', Float64) for m in 1:config["views"]]
    groups = vec(Int.(readdlm(joinpath(data_directory, "groups.csv"), ',', Float64)))
    z_init = readdlm(joinpath(data_directory, "Z_init.csv"), ',', Float64)
    options = (; base_options..., k = config["factors"], groups = groups,
        z_init = z_init, maxiter = config["sweeps"])
    if mode == "bench"
        mofa(views; options...)
    end
    _wait_for_file(config_path)
    sleep(0.001)
    GC.gc()
    metrics = Dict{String,Any}("language" => "Julia", "version" => string(VERSION),
        "package_version" => string(pkgversion(BigRiverEssence)),
        "benchmarktools_version" => string(pkgversion(BenchmarkTools)),
        "package_path" => pkgdir(BigRiverEssence), "julia_threads" => Threads.nthreads(),
        "blas" => string(BLAS.get_config()), "blas_threads" => BLAS.get_num_threads(),
        "sweeps" => config["sweeps"])
    if mode == "bench"
        samples = config["samples"]
        # @btime uses the same engine and reports minimum(trial). @benchmark
        # retains all samples so the CSV can also show medians and sample counts.
        benchmark = @benchmarkable mofa($views; $options...) evals=1 samples=samples seconds=1.0e8
        trial = run(benchmark; warmup = false, capture_result = false)
        best = minimum(trial)
        metrics["time_seconds"] = best.time / 1e9
        metrics["time_min_s"] = best.time / 1e9
        metrics["time_median_s"] = median(trial.times) / 1e9
        metrics["allocations"] = best.allocs
        metrics["allocated_memory_kb"] = best.memory / 1024
        metrics["allocation_backend"] = "BenchmarkTools: Julia-managed allocations"
        metrics["samples"] = length(trial.times)
        writedlm(joinpath(job_directory, "timing_samples_s.csv"), trial.times ./ 1e9, ',')
        # Independent numerical fit outside the timed expression.
        model = mofa(views; options...)
        metrics["final_elbo"] = model.elbo[end]
        metrics["converged"] = model.converged
        _export_result(model, joinpath(job_directory, "arrays"))
    elseif mode == "rss"
        touch(joinpath(job_directory, "ready"))
        _wait_for_file(joinpath(job_directory, "go"))
        model = mofa(views; options...)
        sleep(0.05)
        touch(joinpath(job_directory, "done"))
        _wait_for_file(joinpath(job_directory, "release"))
        # Keep the entire fitted result live until the sampler releases us.
        metrics["final_elbo"] = model.elbo[end]
    else
        error("Unknown worker mode: $mode")
    end
    open(joinpath(job_directory, "metrics.toml"), "w") do io
        TOML.print(io, metrics)
    end
    return nothing
end

_main(ARGS...)
