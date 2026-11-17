# MOFA: Julia versus Python comparison

These are temporary scripts: delete the `temporary_mofa_comparison` folder
when finished. Run `compare_mofa.py` to compare the **local BigRiverEssence**
Gaussian MOFA implementation with the original **mofapy2 0.7.5** engine.
**No changes are made to the repository's Project.toml or Manifest.toml.**

## Setup

Use Julia **1.10 or newer** and preferably Python **3.11 or newer** in your
chosen Python/conda environment. Python 3.9/3.10 is also supported with the
`tomli` backport. If the Python dependencies are not already available, install
them in the same environment/interpreter used to launch the runner:

```sh
python -m pip install "mofapy2 @ https://github.com/bioFAM/mofapy2/archive/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7.zip" psutil threadpoolctl memray
python -m pip install "tomli; python_version < '3.11'"
```

This installs the same pinned mofapy2 source used to validate our implementation,
plus process-memory and thread-control tools. Prefer a separate Python environment
if you already depend on another mofapy2 version.
Memray is required for allocation profiling: there is no silent `tracemalloc`
fallback. Memray currently supports Linux and macOS; the full three-metric
experiment requires one of those platforms, even though result routing also
recognizes Windows.

If your prompt says `(base)`, do not assume it is the benchmarking environment.
For example, use `conda activate bigriveressence-benchmarks` and check
`python --version` before installing or running. A missing `tomllib` means
Python is older than 3.11; the scripts automatically use `tomli` in that case.

The runner creates a **disposable Julia environment in the system temporary
directory**, adds BenchmarkTools and the local package there, and removes only
that disposable environment when it finishes. Nothing is added to the package's
Project.toml or Manifest.toml, and no Project.toml is added to this script folder.
Julia may download/cache packages on the first run. Use `--offline` if the
dependencies are already cached. `--julia-project /path/to/existing/environment`
can reuse a separately configured benchmarking environment without modifying it.
Workers check that they load your local checkout, not a registry release.

## Run

A quick first check:

```sh
python temporary_mofa_comparison/compare_mofa.py --case tiny --sweeps 10 --samples 3
```

The standard comparison uses the small dataset and all three model settings:

```sh
python temporary_mofa_comparison/compare_mofa.py
```

Other examples:

```sh
# Dense Gaussian model only
python temporary_mofa_comparison/compare_mofa.py --case small --model dense --sweeps 100 --samples 5

# All dataset sizes and all model settings; can take substantially longer
python temporary_mofa_comparison/compare_mofa.py --case all --model all

# Only the large dataset, with all three model settings
python temporary_mofa_comparison/compare_mofa.py --case large --model all --samples 5

# Choose your own dimensions and threading
python temporary_mofa_comparison/compare_mofa.py --case small --n 300 --features 150,80,40 --factors 5 --threads 1

# Use the default MOFA sparsity warm-up rather than this comparison's shorter warm-up
python temporary_mofa_comparison/compare_mofa.py --start-sparsity 50

# Inspect every available option
python temporary_mofa_comparison/compare_mofa.py --help
```

Use `--julia /absolute/path/to/julia` if Julia is not on PATH. The Python
workers always use the same interpreter that launches `compare_mofa.py`.
The runner works independently of the current directory when called with
an absolute script path. `--output` selects another results parent folder.

| Case | Samples | Features per view | Factors |
|---|---:|---|---:|
| tiny | 60 | 12, 8 | 2 |
| small | 200 | 100, 60 | 5 |
| medium | 500 | 300, 200 | 10 |
| large | 1,000 | 600, 400 | 20 |

`--case all --model all` runs all **12 configurations**: four dataset sizes
times three model settings. The large case uses the same default 50 sweeps,
five timing samples, four threads per runtime, memory/allocation profilers, and
full-precision numerical comparisons. It generates its shared inputs
automatically; no separate data-generation command is required. Large runs
take longer and save larger paired-answer reports, but do not overwrite any
previous run.

| Model | Loading ARD | Factor ARD | Sparse loadings |
|---|---|---|---|
| dense | off | off | off |
| ard | on | on | off |
| sparse | on | on | on |

Default settings: two sample groups, 2% missing entries and one sample missing
across views, Float64 data, seed 160, 50 update sweeps, five timing samples,
four Julia worker threads and a four-thread BLAS limit in each language,
group centering, no view/group scaling, equal view weights,
ELBO checks at sweep 1 and every five thereafter, Beta updates from sweep 3,
and no early stopping. The model settings are explicit in every saved config.

## What is matched?

Both languages read the **same saved matrices**, group labels, and initial
factor matrix. Both start with zero loading means and the same posterior
variances/hyperparameters. A custom initialization avoids RNG and PCA
sign differences and tests the fitting algorithm rather than two PCA solvers.

Each complete fitted-call measurement includes copying/preprocessing the
inputs, constructing posterior state, all updates, ELBO checks, and final
explained-variation diagnostics. Data generation/loading, imports, startup,
Julia compilation, file writing, and the comparison calculations are excluded.
The first full fit is a warm-up. Workers run sequentially, not concurrently.

For large problems, BigRiverEssence uses those Julia threads to update independent
loading/sample rows with private scratch. It keeps the coordinates within each
row sequential and uses no cross-worker floating-point reduction. Small cases
and `--threads 1` use the serial path. Thread counts are not increased to obtain
a speedup: both languages retain the requested limit. Extra Julia workers
require additional scratch and task allocations, which remain included in the
reported metrics. Noise and diagnostic calculations remain serial.

Python uses the **original** `process_data`, `initModel`, `buildBiofam` Markov
blankets, and `BayesNet.iterate` routines. The small adapter only wires those
APIs together to accept the shared custom factors; it does not implement any
MOFA update. Python entry-point PCA initialization and HDF5 saving are not timed.
Python `maxiter = sweeps + 1` matches Julia `maxiter = sweeps`, since Python's
count includes initialization. A final unscheduled ELBO is calculated for both.

## Three metrics: same measurement procedures as the benchmark repository

This follows `BigRiverEssence_benchmarks/scripts/nmf/` and its external RSS
monitor. In `benchmarks.csv`, each language has these three main metrics:

| Metric | Julia | Python |
|---|---|---|
| `time_seconds` | Minimum BenchmarkTools time for one complete `mofa` call | Minimum `timeit` time for one complete original-engine fit |
| `memory_kb` | Incremental peak process-tree RSS | Incremental peak process-tree RSS |
| `allocations` | BenchmarkTools allocation count | Memray total allocation events, including tracked native allocations |

`allocated_memory_kb` is also retained: cumulative allocation bytes from each
profiler, not peak resident memory. The profilers have different scopes:
BenchmarkTools counts Julia-managed allocations; Memray traces Python and
supported native allocator events. Identical column names do not make their
underlying runtime allocation events identical. The common RSS experiment
is the like-measured resident-memory comparison.

### Speed

Julia calls the **public `mofa` function directly** in `BenchmarkTools.@benchmarkable`,
with interpolated inputs and `evals=1`. This is the same measurement engine as
`@btime`; retaining the trial lets the report show both minimum and median times
and all raw samples. As in `nmf_jl_bench.jl`, it uses
`run(benchmark; warmup=false, capture_result=false)`, with BenchmarkTools'
default GC settings after an explicit warm-up. Python uses
`timeit.Timer(...).repeat(repeat=samples, number=1)`, as in `nmf_py_bench.py`.
This uses `timeit`'s default cyclic-GC-disabled policy during each measurement;
it does not claim identical GC behavior between runtimes. Memory tracing is
not enabled during these timing samples.

`python_time_over_julia > 1` means Julia was faster in that particular run.
Check numerical agreement before interpreting a speedup. Matching iterations
does not prove that the selected number of iterations is sufficient for a
scientific analysis; early stopping is deliberately disabled to compare equal work.

### Memory and allocations

**Common cross-language memory metric:** an external `psutil` sampler measures
the child process tree's resident set size (RSS), including resident native-library
memory, during a separate full fit. In each new memory worker a **separate tiny
problem** is fitted first, then collected; target inputs are loaded, and garbage
collection runs before the ready signal. The target problem is not fitted before
measurement, avoiding the previous full-target warm-up's allocator reuse.
The tiny warm-up runs two sweeps with ELBO and sparsity updates enabled from
the first sweep, exercising those code paths before the measurement.
The sampler records baseline RSS immediately
before the fit, sampled peak RSS, and incremental peak `max(peak - baseline, 0)`.
Baseline includes the runtime, imports, shared inputs, and allocator state after
the tiny warm-up. Startup/imports, warm-up compilation, and output serialization are outside
the sampling interval. The fitted result stays live until sampling finishes.
The workers use file handshakes for synchronization; a 50 ms post-fit hold
keeps the result live while sampling, matching the benchmark suite's hold.
Any compilation specific to target dimensions or a different view count can
still contribute in a memory worker; RSS is not a pure algorithm-workspace measure.

RSS is **sampled**, not an exact allocation total: the default 0.5 ms interval may
miss brief peaks. Allocator reuse can yield a zero increment despite substantial
allocation. Report baseline and absolute peak alongside the increment; runtime
baselines differ between languages. RSS is summed over the worker and its
children, not over unrelated processes. Shared resident pages can consequently
be counted more than once if children exist; it is not PSS or total bytes allocated.
The tiny case is primarily a numerical/setup check; use larger cases for a
more informative memory experiment, without assuming the increment must be nonzero.

**Separate language-specific diagnostics:**

- Julia: `allocations` and `allocated_memory_kb` are BenchmarkTools'
  cumulative Julia-managed allocation estimates for one fit. They do not capture
  arbitrary native-library allocation.
- Python: `allocations` and `allocated_memory_kb` come from a separate untimed
  Memray fit with `native_traces=True, trace_python_allocators=True`, using
  `total_num_allocations` and `total_bytes_allocated` from `memray stats --json`.
  This replaces the old retained-block/peak-traced `tracemalloc` diagnostics.
  Each job retains `memray_statistics.json`; the potentially large binary trace
  is kept only in a disposable system-temporary folder and removed afterwards.

All memory columns use **bytes / 1024**, matching the benchmark repository's
`_kb` columns (technically KiB). RAM uses bytes / 1024³. All timing columns
are in seconds. Requested BLAS thread counts default to four in both languages;
library identities, versions, and actual thread limits are recorded in metadata.

## Results

Each invocation creates a new timestamped directory under
`temporary_mofa_comparison/results/Linux/`, `Mac/`, or `Windows/`. No previous
run is overwritten. Keep the folder untracked and delete it when finished;
no repository .gitignore rules are changed. Move any results you want to keep
before deleting the temporary folder.

- `summary.md`: paired table with all three metrics for both languages, plus
  the actual final ELBO values and links to the numerical answers.
- `comparison_summary.csv`: speed ratios and numerical pass/fail flags.
- `benchmarks.csv`: min/median seconds, both RSS metrics, allocation diagnostics,
  dataset dimensions, processor, core counts, RAM, and threading.
- `numerical_comparison.csv`: maximum absolute and relative Frobenius differences
  for sample/loading means and variances, second moments, all noise and ARD
  parameters/moments, sparse inclusions and slab parameters, ELBO histories,
  explained variation, and view reconstructions.
- `metadata.json`: complete settings, versions, BLAS libraries, Git commit/status,
  source hash, input hashes, and worker details.
- Per-case folders: the shared input data, TOML configs, raw timing samples,
  numerical outputs, profiling metrics, and worker logs.
- `<case>/<model>/numerical_answers.md`: readable, 17-significant-digit paired
  values: the complete ELBO history, all group/view R² values, and a preview of
  each other output array.
- `<case>/<model>/paired_answers/<quantity>.csv`: **every element** of each
  Julia and Python output, side by side, with one-based array indices, the
  signed difference, and its tolerance check. No numerical answers are replaced
  by booleans: raw Float64 values are preserved in round-trippable decimal form.
  This includes factors, loadings, variances, all ARD/noise/sparsity parameters,
  reconstructions, ELBO histories, and explained variation. Original array
  dimensions are retained in each worker's `arrays/shapes.toml`.

Default numerical tolerances are `rtol=1e-5`, `atol=1e-7`; change them with
`--rtol` and `--atol`. Raw factors are compared because their initialization is
matched. Long runs can amplify floating-point differences in non-identifiable
factors; inspect reconstructions and ELBOs, not just factor entries. Paired NaNs
are accepted only for undefined R² diagnostics. Nonfinite posterior values or
ELBOs fail the check. Exit code 2 means numerical checks failed; code 1 means a
worker/setup error. Failures retain logs and completed reports instead of
being silently dropped or converted to zero memory/time.

References: [mofapy2 source](https://github.com/bioFAM/mofapy2/tree/90418e5021b3ae735ebf1fea5d7e07cae71c8bb7),
[BenchmarkTools](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/),
[timeit](https://docs.python.org/3/library/timeit.html),
[Memray](https://bloomberg.github.io/memray/),
[psutil](https://psutil.readthedocs.io/).
