# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | dense | 0.000997042 | 0.0151326 | 0.00 | 80.00 | 117 | 78493 | 15.177× | True |
| tiny | ard | 0.00108263 | 0.0163167 | 0.00 | 48.00 | 135 | 93725 | 15.071× | True |
| tiny | sparse | 0.00118121 | 0.0202957 | 0.00 | 48.00 | 164 | 115901 | 17.182× | True |
| small | dense | 0.0500185 | 0.0711776 | 0.00 | 960.00 | 129 | 123627 | 1.423× | True |
| small | ard | 0.0514357 | 0.0751439 | 32.00 | 2176.00 | 147 | 139059 | 1.461× | True |
| small | sparse | 0.052908 | 0.0804047 | 0.00 | 1664.00 | 182 | 174224 | 1.520× | True |
| medium | dense | 0.728201 | 1.03008 | 1184.00 | 34256.00 | 152 | 203396 | 1.415× | True |
| medium | ard | 0.735011 | 1.01321 | 128.00 | 34752.00 | 170 | 218992 | 1.378× | True |
| medium | sparse | 0.751 | 1.04902 | 0.00 | 30768.00 | 205 | 284264 | 1.397× | True |
| large | dense | 4.07814 | 8.26393 | 2352.00 | 104160.00 | 2369 | 348500 | 2.026× | True |
| large | ard | 4.20586 | 7.7049 | 0.00 | 110512.00 | 2387 | 364248 | 1.832× | True |
| large | sparse | 3.84865 | 7.05575 | 3968.00 | 117504.00 | 2422 | 476581 | 1.833× | True |

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
| large | dense | -209074.33153872579 | -209074.33153872588 | [Values](large/dense/numerical_answers.md), [every entry](large/dense/paired_answers/) |
| large | ard | -44107.990901512596 | -44107.990901513083 | [Values](large/ard/numerical_answers.md), [every entry](large/ard/paired_answers/) |
| large | sparse | -44318.03768717598 | -44318.037687172968 | [Values](large/sparse/numerical_answers.md), [every entry](large/sparse/paired_answers/) |
