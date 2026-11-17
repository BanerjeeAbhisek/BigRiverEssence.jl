#!/usr/bin/env python3
"""Compare local BigRiverEssence MOFA with the original mofapy2 engine."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # Backport for Python 3.9/3.10
    except ModuleNotFoundError:
        raise SystemExit("Python < 3.11 requires tomli. Run: python -m pip install tomli") from None

import numpy as np
import psutil


HERE = Path(__file__).resolve().parent
CASES = {"tiny": (60, [12, 8], 2), "small": (200, [100, 60], 5),
         "medium": (500, [300, 200], 10), "large": (1000, [600, 400], 20)}
MODELS = {"dense": (False, False, False), "ard": (True, True, False),
          "sparse": (True, True, True)}


def _write_csv(path, rows):
    """Replace only this run's report; never edit another run's results."""
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _write_toml(path, config):
    """Write flat settings readable without adding Julia JSON dependencies."""
    lines = []
    for key, value in config.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, str):
            value = json.dumps(value)
        else:
            value = str(value)
        lines.append(f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n")


def _machine():
    """Describe the machine without assuming a platform-specific CPU name."""
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return dict(os={"Darwin": "Mac"}.get(platform.system(), platform.system()),
                processor=cpu, physical_cores=psutil.cpu_count(logical=False),
                logical_cores=psutil.cpu_count(logical=True),
                ram_gb=round(psutil.virtual_memory().total / 1024**3, 2), platform=platform.platform())


def _generate(directory, n, features, factors, groups, missing, seed):
    """Generate shared observations and one shared custom initialization once."""
    directory.mkdir()
    rng = np.random.RandomState(seed)
    labels = np.arange(n) * groups // n  # contiguous, first appearance agrees with np.unique
    z = rng.normal(size=(n, factors))
    for m, p in enumerate(features, 1):
        x = z @ rng.normal(size=(factors, p)) + 0.2 * rng.normal(size=(n, p))
        x += labels[:, None] * (m * 0.5)
        mask = rng.uniform(size=x.shape) < missing
        mask[0, :] = False  # every feature has at least one observed value
        x[mask] = np.nan
        if missing > 0:
            x[4, :] = np.nan  # one sample missing in every view
        np.savetxt(directory / f"X{m}.csv", x, delimiter=",")
    np.savetxt(directory / "groups.csv", labels, delimiter=",", fmt="%d")
    np.savetxt(directory / "Z_init.csv", rng.normal(size=(n, factors)), delimiter=",")


def _tree_rss(process):
    """Sum resident bytes over a worker and its children, as in the benchmark suite."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def _run_worker(language, config, directory, mode, args, environment):
    """Run a child; externally sample the fit's RSS only after its ready signal."""
    directory.mkdir(parents=True)
    if language == "Julia":
        command = [args.julia, "--startup-file=no", f"--project={args.julia_project}"] + args.julia_arg + \
            [str(HERE / "_julia_worker.jl"), str(config), str(directory), mode]
    else:
        command = [sys.executable, str(HERE / "_python_worker.py"), str(config), str(directory), mode]
    started = time.monotonic()
    rss = {}
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            monitored = psutil.Process(process.pid)
            sampling = False
            while process.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(f"{language} {mode} exceeded {args.timeout}s. See {directory / 'worker.log'}")
                if mode == "rss":
                    if not sampling and (directory / "ready").exists():
                        baseline = _tree_rss(monitored)
                        peak, observations = baseline, 1
                        sampling = True
                        (directory / "go").touch()
                    if sampling:
                        peak = max(peak, _tree_rss(monitored))
                        observations += 1
                        if (directory / "done").exists():
                            rss = dict(baseline_rss_kb=baseline / 1024,
                                       peak_rss_kb=peak / 1024,
                                       incremental_peak_rss_kb=max(peak - baseline, 0) / 1024,
                                       memory_kb=max(peak - baseline, 0) / 1024,
                                       rss_samples=observations,
                                       rss_interval_ms=args.rss_interval_ms)
                            (directory / "release").touch()
                            sampling = False
                            # The ready file remains; do not start another interval.
                            mode = "released"
                time.sleep(args.rss_interval_ms / 1000 if sampling else 0.01)
            if process.returncode:
                raise RuntimeError(f"{language} worker failed (exit {process.returncode}). See {directory / 'worker.log'}")
            if directory.name.endswith("rss") and not rss:
                raise RuntimeError(f"{language} worker did not complete the memory-sampling handshake")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    metrics = directory / ("metrics.toml" if language == "Julia" else "metrics.json")
    parsed = tomllib.loads(metrics.read_text()) if language == "Julia" else json.loads(metrics.read_text())
    return parsed | rss


def _compare(julia_directory, python_directory, case, model, args):
    """Compare raw matched-initialization posteriors as well as identifiable predictions."""
    julia_files = {p.name for p in julia_directory.glob("*.csv")}
    python_files = {p.name for p in python_directory.glob("*.csv")}
    if not julia_files or julia_files != python_files:
        raise RuntimeError(f"Missing numerical arrays: {julia_files ^ python_files}")
    julia_shapes = tomllib.loads((julia_directory / "shapes.toml").read_text())
    python_shapes = tomllib.loads((python_directory / "shapes.toml").read_text())
    if julia_shapes != python_shapes:
        raise RuntimeError(f"Original numerical array dimensions disagree: {julia_shapes} vs {python_shapes}")
    paired_directory = julia_directory.parent.parent / "paired_answers"
    paired_directory.mkdir()
    rows = []
    readable = [f"# Actual numerical answers: {case}/{model}", "",
                "Values are written with 17 significant digits, sufficient to round-trip Float64. "
                "Indices are one-based, in Julia column-major order. All entries appear in the paired CSVs.", "",
                "| Quantity | Index | Julia | Python | Julia − Python |",
                "|---|---|---:|---:|---:|"]
    for name in sorted(julia_files):
        a = np.atleast_1d(np.loadtxt(julia_directory / name, delimiter=","))
        b = np.atleast_1d(np.loadtxt(python_directory / name, delimiter=","))
        if a.shape != b.shape:
            raise RuntimeError(f"Array shape mismatch for {name}: {a.shape} versus {b.shape}")
        finite = np.isfinite(a) & np.isfinite(b)
        difference = a[finite] - b[finite]
        allow_nan = name in ("r2.csv", "factor_r2.csv")
        valid = bool(np.all(finite | (np.isnan(a) & np.isnan(b)))) if allow_nan else bool(np.all(finite))
        element_close = np.isclose(a, b, rtol=args.rtol, atol=args.atol, equal_nan=allow_nan)
        element_close &= finite | (allow_nan & np.isnan(a) & np.isnan(b))
        close = valid and bool(np.all(element_close))
        quantity = Path(name).stem
        shape = tuple(julia_shapes[quantity])
        if np.prod(shape) != a.size:
            raise RuntimeError(f"Invalid exported dimensions for {quantity}")
        # Stream all entries rather than building a large in-memory report.
        # repr(float) round-trips the exact saved binary64 value; .17g in the
        # readable report avoids rounding away small Julia/Python differences.
        with (paired_directory / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case", "model", "quantity", "index_1based", "julia_value",
                             "python_value", "difference", "within_tolerance"])
            for offset, (julia, python) in enumerate(zip(a, b)):
                index = ",".join(str(int(i) + 1) for i in np.unravel_index(offset, shape, order="F"))
                equal = bool(element_close[offset])
                writer.writerow([case, model, quantity, index, float(julia), float(python),
                                 float(julia - python), equal])
                if quantity in ("elbo", "r2") or offset < 5:
                    readable.append(f"| {quantity} | ({index}) | {julia:.17g} | {python:.17g} | {julia-python:.17g} |")
        rows.append(dict(case=case, model=model, quantity=quantity, elements=a.size,
                         shape=" × ".join(map(str, shape)),
                         paired_answers=str((paired_directory / name).relative_to(paired_directory.parent)),
                         max_abs_diff=float(np.max(np.abs(difference))) if difference.size else 0.0,
                         relative_frobenius_diff=float(np.linalg.norm(difference) /
                                                      max(np.linalg.norm(b[finite]), np.finfo(float).eps)),
                         within_tolerance=close, rtol=args.rtol, atol=args.atol))
    (paired_directory.parent / "numerical_answers.md").write_text("\n".join(readable) + "\n")
    return rows


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--case", choices=[*CASES, "all"], default="small")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--n", type=int, help="Override number of samples")
    parser.add_argument("--features", help="Override view sizes, e.g. 100,60")
    parser.add_argument("--factors", type=int, help="Override latent dimension")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--missing", type=float, default=0.02)
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=160)
    parser.add_argument("--freq-elbo", type=int, default=5)
    parser.add_argument("--start-sparsity", type=int, default=3)
    parser.add_argument("--rss-interval-ms", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=1800, help="Per-worker timeout in seconds")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--julia", default=shutil.which("julia") or "julia")
    parser.add_argument("--julia-arg", action="append", default=[], help="Extra Julia flag, e.g. --julia-arg=--compiled-modules=no")
    parser.add_argument("--julia-project", type=Path, help="Reuse an already configured benchmarking environment instead of creating a disposable one")
    parser.add_argument("--offline", action="store_true", help="Set up the disposable Julia environment using only cached packages")
    parser.add_argument("--repo", type=Path, default=HERE.parent, help="Local BigRiverEssence checkout")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if min(args.sweeps, args.samples, args.threads, args.groups, args.freq_elbo, args.start_sparsity) < 1:
        parser.error("Counts and schedule intervals must be positive")
    if not 0 <= args.missing < 1 or min(args.rss_interval_ms, args.timeout) <= 0 or min(args.rtol, args.atol) < 0:
        parser.error("Invalid missingness, sampling interval, timeout, or tolerance")
    return args


def _prepare_julia_environment(args, scratch, output, environment):
    """Put benchmarking dependencies outside the checkout, never in its Project.toml."""
    if args.julia_project is not None:
        args.julia_project = args.julia_project.resolve()
        return
    args.julia_project = scratch
    print("Preparing a disposable Julia environment outside the repository ...", flush=True)
    code = '''using Pkg
Pkg.offline(ARGS[3] == "true")
Pkg.activate(ARGS[1])
Pkg.develop(path=ARGS[2])
Pkg.add([PackageSpec(name="BenchmarkTools", version="1.8"),
         PackageSpec(name="DelimitedFiles"), PackageSpec(name="Statistics")])
Pkg.instantiate()
'''
    command = [args.julia, "--startup-file=no"] + args.julia_arg + ["-e", code,
               str(scratch), str(args.repo.resolve()), str(args.offline).lower()]
    try:
        with (output / "julia_environment_setup.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"Julia environment setup timed out. See {output / 'julia_environment_setup.log'}") from error
    if result.returncode:
        raise RuntimeError(f"Disposable Julia setup failed. See {output / 'julia_environment_setup.log'}")


def _run(args, scratch):
    repo = args.repo.resolve()
    if not (repo / "src" / "mofa.jl").is_file():
        raise RuntimeError(f"Not a BigRiverEssence checkout: {repo}")
    machine = _machine()
    parent = args.output.resolve() / machine["os"]
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=stamp + "_", dir=parent))
    print(f"Output: {output}", flush=True)
    environment = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"):
        environment[key] = str(args.threads)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe = ("import importlib.metadata as m; import threadpoolctl; import memray; "
             "from mofapy2.build_model.init_model import initModel; "
             "from mofapy2.core.BayesNet import BayesNet; "
             "assert m.version('mofapy2') == '0.7.5', 'Expected mofapy2 0.7.5'")
    with (output / "python_environment_check.log").open("w") as log:
        checked = subprocess.run([sys.executable, "-c", probe], stdout=log,
                                 stderr=subprocess.STDOUT, env=environment, timeout=args.timeout)
    if checked.returncode:
        raise RuntimeError(f"Python dependencies are unavailable or incompatible. Follow the temporary folder's README setup; see {output / 'python_environment_check.log'}")
    _prepare_julia_environment(args, scratch, output, environment)
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, git_status = "unavailable", "unavailable"
    metadata = dict(machine=machine, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    utc_started=stamp, git_commit=commit, git_status=git_status,
                    mofa_source_sha256=hashlib.sha256((repo / "src" / "mofa.jl").read_bytes()).hexdigest(),
                    memory_units="bytes / 1024 (same _kb convention as benchmark repository)",
                    measurement_protocol=dict(julia_timing="BenchmarkTools minimum, evals=1, default GC policy",
                        python_timing="timeit.repeat, number=1, default cyclic-GC-disabled policy",
                        python_allocations="Memray native_traces=True, trace_python_allocators=True",
                        memory="fresh-process incremental peak process-tree RSS after separate tiny warm-up"),
                    workers=[], input_sha256={})
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print("Time excludes startup/JIT/I/O; one warmed full fit per timing sample.", flush=True)
    print("Metrics: time_seconds, memory_kb (incremental peak RSS), allocations.", flush=True)
    print("Allocation backends: Julia BenchmarkTools; Python/native Memray. Memory units: bytes/1024.\n", flush=True)
    benchmark_rows, comparison_rows, summaries = [], [], []
    cases = list(CASES) if args.case == "all" else [args.case]
    models = list(MODELS) if args.model == "all" else [args.model]
    for case in cases:
        n, features, factors = CASES[case]
        n = args.n if args.n is not None else n
        factors = args.factors if args.factors is not None else factors
        features = [int(x) for x in args.features.split(",")] if args.features else features
        if n < 6 or not features or min(features) < 1 or not 1 <= args.groups <= n or not 1 <= factors <= min(n - 1, sum(features)):
            raise ValueError("Require n>=6, positive view sizes, 1<=groups<=n, and a valid factor count")
        case_directory = output / case
        case_directory.mkdir()
        data_directory = case_directory / "inputs"
        _generate(data_directory, n, features, factors, args.groups, args.missing, args.seed)
        for p in data_directory.iterdir():
            metadata["input_sha256"][str(p.relative_to(output))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for model in models:
            aw, az, sw = MODELS[model]
            directory = case_directory / model
            directory.mkdir()
            config = dict(data_directory=str(data_directory), repository=str(repo),
                          views=len(features), factors=factors, threads=args.threads, seed=args.seed,
                          sweeps=args.sweeps, samples=args.samples, freq_elbo=args.freq_elbo,
                          start_sparsity=args.start_sparsity, ard_weights=aw, ard_factors=az,
                          spikeslab_weights=sw)
            config_path = directory / "config.toml"
            _write_toml(config_path, config)
            metadata_path.write_text(json.dumps(metadata, indent=2))
            paired = {}
            for language in ("Julia", "Python"):
                print(f"{case}/{model}: {language} timing and memory ...", flush=True)
                label = language.lower()
                timing = _run_worker(language, config_path, directory / (label + "_bench"), "bench", args, environment)
                memory = _run_worker(language, config_path, directory / (label + "_rss"), "rss", args, environment)
                metadata["workers"].append(dict(case=case, model=model, language=language, timing=timing, memory=memory))
                metadata_path.write_text(json.dumps(metadata, indent=2))
                # Numerical answers must come from the exported numerical fit,
                # not a second fit in the independently sampled RSS worker.
                metrics = memory | timing
                row = dict(case=case, model=model, language=language, n=n, features="+".join(map(str, features)),
                           factors=factors, groups=args.groups, sweeps=args.sweeps, threads=args.threads,
                           os=machine["os"], processor=machine["processor"],
                           physical_cores=machine["physical_cores"], logical_cores=machine["logical_cores"],
                           ram_gb=machine["ram_gb"])
                for field in ("time_seconds", "memory_kb", "allocations", "allocated_memory_kb", "allocation_backend",
                              "samples", "time_min_s", "time_median_s", "baseline_rss_kb", "peak_rss_kb",
                              "incremental_peak_rss_kb", "rss_samples", "rss_interval_ms",
                              "final_elbo"):
                    row[field] = metrics.get(field, "")
                benchmark_rows.append(row)
                paired[language] = row
                _write_csv(output / "benchmarks.csv", benchmark_rows)
            numerical = _compare(directory / "julia_bench" / "arrays", directory / "python_bench" / "arrays", case, model, args)
            comparison_rows.extend(numerical)
            _write_csv(output / "numerical_comparison.csv", comparison_rows)
            passed = all(row["within_tolerance"] for row in numerical)
            speedup = paired["Python"]["time_min_s"] / paired["Julia"]["time_min_s"]
            summaries.append(dict(case=case, model=model, arrays_compared=len(numerical),
                                  all_arrays_within_tolerance=passed, python_time_over_julia=speedup,
                                  julia_time_min_s=paired["Julia"]["time_min_s"],
                                  python_time_min_s=paired["Python"]["time_min_s"],
                                  julia_memory_kb=paired["Julia"]["memory_kb"],
                                  python_memory_kb=paired["Python"]["memory_kb"],
                                  julia_allocations=paired["Julia"]["allocations"],
                                  python_allocations=paired["Python"]["allocations"],
                                  julia_final_elbo=paired["Julia"]["final_elbo"],
                                  python_final_elbo=paired["Python"]["final_elbo"]))
            _write_csv(output / "comparison_summary.csv", summaries)
            print(f"  Numerical match: {passed} ({len(numerical)} arrays); Python/Julia time: {speedup:.3f}x", flush=True)
            for language in ("Julia", "Python"):
                row = paired[language]
                print(f"  {language}: {row['time_seconds']:.8g} s; {row['memory_kb']:.2f} kB incremental RSS; "
                      f"{row['allocations']} allocations; {row['allocated_memory_kb']:.2f} kB allocated", flush=True)
                print(f"    Final ELBO: {row['final_elbo']:.17g}", flush=True)
                print(f"  {language} RSS: baseline {row['baseline_rss_kb']:.2f}, peak {row['peak_rss_kb']:.2f}, "
                      f"increment {row['incremental_peak_rss_kb']:.2f} kB", flush=True)
            julia_r2 = np.atleast_1d(np.loadtxt(directory / "julia_bench/arrays/r2.csv", delimiter=","))
            python_r2 = np.atleast_1d(np.loadtxt(directory / "python_bench/arrays/r2.csv", delimiter=","))
            for index, (julia, python) in enumerate(zip(julia_r2, python_r2), 1):
                print(f"  R² [{index}, column-major]: Julia {julia:.17g}; Python {python:.17g}", flush=True)
            print(f"  Every numerical entry: {directory / 'paired_answers'}\n", flush=True)
    report = ["# MOFA comparison", "", f"Machine: {machine['processor']}; {machine['logical_cores']} logical cores; {machine['ram_gb']:.2f} GB RAM.", "",
              "| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        report.append(f"| {row['case']} | {row['model']} | {row['julia_time_min_s']:.6g} | {row['python_time_min_s']:.6g} | "
                      f"{row['julia_memory_kb']:.2f} | {row['python_memory_kb']:.2f} | "
                      f"{row['julia_allocations']} | {row['python_allocations']} | "
                      f"{row['python_time_over_julia']:.3f}× | {row['all_arrays_within_tolerance']} |")
    report += ["", "Python/Julia > 1 means Julia was faster in this run. Check numerical agreement before interpreting speed.", "",
               "Memory = incremental peak RSS; see benchmarks.csv for baseline/peak RSS and cumulative allocated_memory_kb as well.",
               "RSS includes native memory but is sampled and can miss brief peaks. Warm allocators may produce zero incremental RSS; this is not zero allocation.",
               "Allocation counts use BenchmarkTools (Julia-managed) and Memray (Python and native events), as in the benchmark suite; instrumentation scopes still differ.", "",
               "## Actual numerical answers", "",
               "| Case | Model | Final ELBO: Julia | Final ELBO: Python | Full answers |",
               "|---|---|---:|---:|---|"]
    for row in summaries:
        location = f"{row['case']}/{row['model']}"
        report.append(f"| {row['case']} | {row['model']} | {row['julia_final_elbo']:.17g} | "
                      f"{row['python_final_elbo']:.17g} | [Values]({location}/numerical_answers.md), "
                      f"[every entry]({location}/paired_answers/) |")
    (output / "summary.md").write_text("\n".join(report) + "\n")
    print(f"Reports saved in {output}", flush=True)
    return 0 if all(row["all_arrays_within_tolerance"] for row in summaries) else 2


def main():
    args = _arguments()
    # Only this disposable directory is removed. Reports and repository files
    # are outside it; Julia's shared package cache is not removed.
    with tempfile.TemporaryDirectory(prefix="mofa-comparison-env-") as scratch:
        return _run(args, Path(scratch))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError, psutil.Error, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
