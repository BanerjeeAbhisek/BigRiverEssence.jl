# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| large | dense | 7.57618 | 7.53916 | 1872.00 | 110080.00 | 151 | 348500 | 0.995× | True |
| large | ard | 7.67641 | 7.65843 | 1056.00 | 113456.00 | 169 | 364248 | 0.998× | True |
| large | sparse | 7.88908 | 7.0703 | 4928.00 | 113072.00 | 204 | 476581 | 0.896× | True |

Python/Julia > 1 means Julia was faster in this run. Check numerical agreement before interpreting speed.

Memory = incremental peak RSS; see benchmarks.csv for baseline/peak RSS and cumulative allocated_memory_kb as well.
RSS includes native memory but is sampled and can miss brief peaks. Warm allocators may produce zero incremental RSS; this is not zero allocation.
Allocation counts use BenchmarkTools (Julia-managed) and Memray (Python and native events), as in the benchmark suite; instrumentation scopes still differ.

## Actual numerical answers

| Case | Model | Final ELBO: Julia | Final ELBO: Python | Full answers |
|---|---|---:|---:|---|
| large | dense | -209074.33153872579 | -209074.33153872588 | [Values](large/dense/numerical_answers.md), [every entry](large/dense/paired_answers/) |
| large | ard | -44107.990901512596 | -44107.990901513083 | [Values](large/ard/numerical_answers.md), [every entry](large/ard/paired_answers/) |
| large | sparse | -44318.03768717598 | -44318.037687172968 | [Values](large/sparse/numerical_answers.md), [every entry](large/sparse/paired_answers/) |
