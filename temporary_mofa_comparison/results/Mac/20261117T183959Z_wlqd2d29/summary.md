# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | dense | 0.000970083 | 0.0135686 | 0.00 | 48.00 | 116 | 78493 | 13.987× | True |
| tiny | ard | 0.00106121 | 0.015919 | 0.00 | 48.00 | 134 | 93725 | 15.001× | True |
| tiny | sparse | 0.00119592 | 0.0205322 | 0.00 | 64.00 | 163 | 115901 | 17.169× | True |
| small | dense | 0.0490891 | 0.0710343 | 0.00 | 1584.00 | 128 | 123615 | 1.447× | True |
| small | ard | 0.0498041 | 0.0735064 | 0.00 | 1088.00 | 146 | 139059 | 1.476× | True |
| small | sparse | 0.0521016 | 0.0803327 | 0.00 | 1376.00 | 181 | 174224 | 1.542× | True |
| medium | dense | 0.789161 | 1.02332 | 784.00 | 32192.00 | 151 | 203396 | 1.297× | True |
| medium | ard | 0.794868 | 1.02006 | 288.00 | 33248.00 | 169 | 218992 | 1.283× | True |
| medium | sparse | 0.810703 | 1.00743 | 0.00 | 36736.00 | 204 | 284264 | 1.243× | True |

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
