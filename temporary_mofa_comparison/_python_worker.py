"""Original mofapy2 training engine; no reimplementation of its updates."""

import contextlib
import gc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import tempfile
import time
import timeit
try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # Backport for Python 3.9/3.10
    except ModuleNotFoundError:
        raise SystemExit("Python < 3.11 requires tomli. Run: python -m pip install tomli") from None
import memray
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from mofapy2.build_model.init_model import initModel
from mofapy2.build_model.build_model import buildBiofam
from mofapy2.build_model.utils import process_data
from mofapy2.core.BayesNet import BayesNet


def _fit(views, groups, z_init, config):
    """Preprocess, initialize, train, and compute the same diagnostics as mofa."""
    n, k = z_init.shape
    count = len(views)
    data_options = dict(center_groups=True, scale_views=False, scale_groups=False,
                        groups_names=list(dict.fromkeys(groups)), samples_groups=groups)
    data = process_data([x.copy() for x in views], ["gaussian"] * count, data_options, groups)
    dimensions = dict(N=n, K=k, M=count, D=[x.shape[1] for x in views], G=len(np.unique(groups)))
    model_options = dict(ard_weights=config["ard_weights"], ard_factors=config["ard_factors"],
                         spikeslab_weights=config["spikeslab_weights"], spikeslab_factors=False,
                         likelihoods=["gaussian"] * count)
    init = initModel(dimensions, data, ["gaussian"] * count, groups, seed=config["seed"])
    init.initZ(qmean=z_init.copy())
    if config["spikeslab_weights"]:
        init.initSW(qmean_S1=0)
        init.initThetaW()
    else:
        init.initW(qmean=0)
    if config["ard_weights"]:
        init.initAlphaW()
    if config["ard_factors"]:
        init.initAlphaZ()
    init.initTau()
    init.initY()
    builder = buildBiofam(data, dimensions, data_options, model_options, {})
    builder.init_model = init
    builder.createMarkovBlankets()
    nodes = init.getNodes()
    schedule = ["Y", "W"] + (["AlphaW"] if config["ard_weights"] else []) + \
        (["ThetaW"] if config["spikeslab_weights"] else []) + ["Z"] + \
        (["AlphaZ"] if config["ard_factors"] else []) + ["Tau"]
    options = dict(maxiter=config["sweeps"] + 1, start_drop=1, freq_drop=1,
                   drop=dict(min_r2=None), verbose=False, quiet=True,
                   convergence_mode="fast", forceiter=True, schedule=schedule,
                   start_sparsity=config["start_sparsity"], gpu_mode=False,
                   start_elbo=1, freqELBO=config["freq_elbo"], min_iter=0,
                   stochastic=False, weight_views=False)
    net = BayesNet(dimensions, nodes)
    net.setTrainOptions(options)
    net.iterate()
    history = net.getTrainingStats()["elbo"][:config["sweeps"] + 1]
    iterations = np.flatnonzero(np.isfinite(history))
    elbo = history[iterations]
    if iterations[-1] != config["sweeps"]:
        iterations = np.append(iterations, config["sweeps"])
        elbo = np.append(elbo, net.calculateELBO()["total"])
    return net, np.asarray(net.calculate_variance_explained(total=True)), \
        np.asarray(net.calculate_variance_explained()), iterations, elbo


def _export_result(result, directory):
    """Save flattened column-major arrays after timing has finished."""
    directory.mkdir()
    net, r2, factor_r2, iterations, elbo = result
    nodes = net.getNodes()

    def save(name, values):
        array = np.asarray(values)
        np.savetxt(directory / (name + ".csv"), array.ravel(order="F"), delimiter=",")
        with (directory / "shapes.toml").open("a") as stream:
            stream.write(f"{name} = {list(array.shape)}\n")

    z = nodes["Z"].getExpectations()
    save("z_mean", z["E"])
    save("z_variance", nodes["Z"].Q.getParameters()["var"])
    save("z_second_moment", z["E2"])
    for name, array in (("r2", r2), ("factor_r2", factor_r2),
                        ("elbo_iterations", iterations), ("elbo", elbo)):
        save(name, array)
    for m, node in enumerate(nodes["W"].getNodes()):
        index = m + 1
        w = node.getExpectations()
        variance = w["E2"] - w["E"] ** 2 if "EB" in w else node.Q.getParameters()["var"]
        save(f"w_mean_{index}", w["E"])
        save(f"w_variance_{index}", variance)
        save(f"w_second_moment_{index}", w["E2"])
        save(f"reconstruction_{index}", z["E"] @ w["E"].T)
        tau_node = nodes["Tau"].getNodes()[m]
        tau_params = tau_node.Q.getParameters()
        tau = tau_node.getExpectations(expand=False)
        for name, array in (("shape", tau_params["a"]), ("rate", tau_params["b"]),
                            ("mean", tau["E"]), ("logmean", tau["lnE"])):
            save(f"tau_{name}_{index}", array)
        if "EB" in w:
            params = node.Q.getParameters()
            save(f"inclusion_{index}", w["EB"])
            save(f"slab_mean_{index}", params["mean_B1"])
            save(f"slab_variance_{index}", params["var_B1"])
            save(f"inactive_variance_{index}", params["var_B0"][0, :])
    for name, key in (("alpha_w", "AlphaW"), ("alpha_z", "AlphaZ")):
        if key not in nodes:
            continue
        qs = nodes[key].getNodes() if key == "AlphaW" else [nodes[key]]
        for field, parameter in (("shape", "a"), ("rate", "b"), ("mean", "E"), ("logmean", "lnE")):
            values = [q.Q.getParameters()[parameter] if field in ("shape", "rate") else
                      q.Q.getExpectations()[parameter] for q in qs]
            save(f"{name}_{field}", np.vstack(values))
    if "ThetaW" in nodes:
        qs = nodes["ThetaW"].getNodes()
        for field, parameter in (("a", "a"), ("b", "b"), ("mean", "E"),
                                 ("logmean", "lnE"), ("logcomplement", "lnEInv")):
            values = [q.Q.getParameters()[parameter] if field in ("a", "b") else
                      q.Q.getExpectations()[parameter] for q in qs]
            save(f"theta_w_{field}", np.vstack(values))


def _wait_for_file(path, timeout=1800):
    """Wait for the external sampler without doing any fitting work."""
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.001)


def _main(config_path, job_directory, mode):
    config = tomllib.loads(Path(config_path).read_text())
    job = Path(job_directory)
    data_directory = Path(config["data_directory"])
    version = importlib.metadata.version("mofapy2")
    if version != "0.7.5":
        raise RuntimeError(f"Expected mofapy2 0.7.5; found {version}. Follow README.md setup.")
    metrics = dict(language="Python", version=platform.python_version(), package_version=version,
                   sweeps=config["sweeps"], numpy_version=np.__version__,
                   scipy_version=importlib.metadata.version("scipy"),
                   pandas_version=importlib.metadata.version("pandas"),
                   memray_version=importlib.metadata.version("memray"),
                   python_executable=sys.executable)
    source = importlib.metadata.distribution("mofapy2").read_text("direct_url.json")
    metrics["mofapy2_install_source"] = json.loads(source) if source else None
    with threadpool_limits(limits=config["threads"], user_api="blas"), open(os.devnull, "w") as sink, \
            contextlib.redirect_stdout(sink):
        if mode == "rss":
            # Same protocol as the NMF benchmark: compile/load on a small,
            # separate problem, not a full target fit that pre-fills allocators.
            # Exercise ELBO and sparse-Beta update paths during warm-up too.
            warm_config = dict(config, sweeps=2, start_sparsity=1, freq_elbo=1)
            rows = np.arange(1.0, 9.0)[:, None]
            warm_views = [np.sin(rows + 2 * np.arange(1.0, p + 1)[None, :])
                          for p in (6, 4)]
            warm_z = np.sin(rows + np.arange(1.0, 3.0)[None, :])
            _fit(warm_views, np.repeat([0, 1], 4), warm_z, warm_config)
            del warm_views, warm_z, rows, warm_config
            gc.collect()
        views = [np.loadtxt(data_directory / f"X{m}.csv", delimiter=",", ndmin=2)
                 for m in range(1, config["views"] + 1)]
        groups = np.loadtxt(data_directory / "groups.csv", delimiter=",", dtype=int)
        z_init = np.loadtxt(data_directory / "Z_init.csv", delimiter=",", ndmin=2)
        if mode == "bench":
            _fit(views, groups, z_init, config)
        gc.collect()
        metrics["threadpools"] = threadpool_info()
        if mode == "bench":
            times = timeit.Timer(lambda: _fit(views, groups, z_init, config)).repeat(
                repeat=config["samples"], number=1)
            metrics.update(time_seconds=min(times), time_min_s=min(times), time_median_s=float(np.median(times)),
                           samples=len(times), converged=False)
            np.savetxt(job / "timing_samples_s.csv", times, delimiter=",")
            gc.collect()
            # Exact allocation-profiling settings used by nmf_py_bench.py.
            # Trace is separate from timing and deleted after saving its stats.
            with tempfile.TemporaryDirectory(prefix="mofa-memray-") as scratch:
                capture = Path(scratch) / "capture.bin"
                statistics_path = job / "memray_statistics.json"
                with memray.Tracker(capture, native_traces=True, trace_python_allocators=True):
                    result = _fit(views, groups, z_init, config)
                completed = subprocess.run(
                    [sys.executable, "-m", "memray", "stats", "--json", "-o",
                     str(statistics_path), str(capture)], capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f"Memray failed: {completed.stderr.strip()}")
                statistics = json.loads(statistics_path.read_text())
            metrics.update(allocations=int(statistics["total_num_allocations"]),
                           allocated_memory_kb=int(statistics["total_bytes_allocated"]) / 1024,
                           allocation_backend="Memray: Python and native allocation events",
                           final_elbo=float(result[-1][-1]))
            _export_result(result, job / "arrays")
        elif mode == "rss":
            (job / "ready").touch()
            _wait_for_file(job / "go")
            result = _fit(views, groups, z_init, config)
            time.sleep(0.05)  # hold the complete result live for the RSS sampler
            (job / "done").touch()
            _wait_for_file(job / "release")
            metrics["final_elbo"] = float(result[-1][-1])
        else:
            raise ValueError(f"Unknown worker mode: {mode}")
    (job / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _main(*sys.argv[1:])
