# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | dense | 0.00100517 | 0.013768 | 0.00 | 80.00 | 117 | 78493 | 13.697× | True |
| tiny | ard | 0.00318221 | 0.0169169 | 0.00 | 64.00 | 135 | 93725 | 5.316× | True |
| tiny | sparse | 0.00120546 | 0.0200923 | 0.00 | 32.00 | 164 | 115901 | 16.668× | True |
| small | dense | 0.0498589 | 0.0704139 | 32.00 | 512.00 | 129 | 123615 | 1.412× | True |
| small | ard | 0.0501875 | 0.0741661 | 0.00 | 3056.00 | 147 | 139059 | 1.478× | True |
| small | sparse | 0.0530532 | 0.0814692 | 0.00 | 1088.00 | 182 | 174224 | 1.536× | True |
| medium | dense | 0.712699 | 0.990432 | 0.00 | 35056.00 | 152 | 203408 | 1.390× | True |
| medium | ard | 0.715604 | 1.00062 | 80.00 | 34336.00 | 170 | 218992 | 1.398× | True |
| medium | sparse | 0.731532 | 0.981497 | 496.00 | 34480.00 | 205 | 284264 | 1.342× | True |
| large | dense | 3.09146 | 7.1992 | 2256.00 | 99616.00 | 2369 | 348500 | 2.329× | True |
| large | ard | 3.18005 | 7.32296 | 2720.00 | 97664.00 | 2387 | 364248 | 2.303× | True |
| large | sparse | 3.0994 | 6.70531 | 2128.00 | 101888.00 | 2422 | 476581 | 2.163× | True |

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
