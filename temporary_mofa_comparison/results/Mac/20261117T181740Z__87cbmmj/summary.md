# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | dense | 0.000981 | 0.0136811 | 0.00 | 64.00 | 116 | 78493 | 13.946× | True |
| tiny | ard | 0.00101958 | 0.0161825 | 0.00 | 32.00 | 134 | 93725 | 15.872× | True |
| tiny | sparse | 0.00124075 | 0.0203522 | 0.00 | 48.00 | 163 | 115901 | 16.403× | True |
| small | dense | 0.0492388 | 0.0723445 | 0.00 | 1552.00 | 128 | 123615 | 1.469× | True |
| small | ard | 0.0497154 | 0.0741274 | 0.00 | 2048.00 | 146 | 139059 | 1.491× | True |
| small | sparse | 0.051856 | 0.0809682 | 16.00 | 2576.00 | 181 | 174224 | 1.561× | True |
| medium | dense | 0.788361 | 1.03014 | 0.00 | 34688.00 | 151 | 203408 | 1.307× | True |
| medium | ard | 0.821142 | 1.08011 | 1216.00 | 29968.00 | 169 | 218992 | 1.315× | True |
| medium | sparse | 0.832837 | 1.07457 | 0.00 | 42464.00 | 204 | 284264 | 1.290× | True |

Python/Julia > 1 means Julia was faster in this run. Check numerical agreement before interpreting speed.

Memory = incremental peak RSS; see benchmarks.csv for baseline/peak RSS and cumulative allocated_memory_kb as well.
RSS includes native memory but is sampled and can miss brief peaks. Warm allocators may produce zero incremental RSS; this is not zero allocation.
Allocation counts use BenchmarkTools (Julia-managed) and Memray (Python and native events), as in the benchmark suite; instrumentation scopes still differ.

## Actual numerical answers

| Case | Model | Final ELBO: Julia | Final ELBO: Python | Full answers |
|---|---|---:|---:|---|
| tiny | dense | -626.70773571056952 | -626.70773571056873 | [Values](tiny/dense/numerical_answers.md), [every entry](tiny/dense/paired_answers/) |
| tiny | ard | -862.72979711325377 | -862.72979711325399 | [Values](tiny/ard/numerical_answers.md), [every entry](tiny/ard/paired_answers/) |
| tiny | sparse | -867.63329184513827 | -867.63329184513805 | [Values](tiny/sparse/numerical_answers.md), [every entry](tiny/sparse/paired_answers/) |
| small | dense | -7482.8126944827982 | -7482.8126944828073 | [Values](small/dense/numerical_answers.md), [every entry](small/dense/paired_answers/) |
| small | ard | -4846.2095303641181 | -4846.2095303641036 | [Values](small/ard/numerical_answers.md), [every entry](small/ard/paired_answers/) |
| small | sparse | -4883.0986274630386 | -4883.0986274630541 | [Values](small/sparse/numerical_answers.md), [every entry](small/sparse/paired_answers/) |
| medium | dense | -50028.396954164855 | -50028.396954164906 | [Values](medium/dense/numerical_answers.md), [every entry](medium/dense/paired_answers/) |
| medium | ard | -11978.331750931789 | -11978.331750931837 | [Values](medium/ard/numerical_answers.md), [every entry](medium/ard/paired_answers/) |
| medium | sparse | -12088.113965444549 | -12088.11396544442 | [Values](medium/sparse/numerical_answers.md), [every entry](medium/sparse/paired_answers/) |
