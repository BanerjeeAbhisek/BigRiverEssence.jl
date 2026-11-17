# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | dense | 0.00100746 | 0.0138281 | 0.00 | 32.00 | 116 | 78493 | 13.726× | True |
| tiny | ard | 0.00108162 | 0.0188381 | 0.00 | 64.00 | 134 | 93725 | 17.417× | True |
| tiny | sparse | 0.00120629 | 0.0199497 | 0.00 | 48.00 | 163 | 115901 | 16.538× | True |
| small | dense | 0.050844 | 0.0695939 | 96.00 | 1952.00 | 128 | 123615 | 1.369× | True |
| small | ard | 0.0499926 | 0.0721587 | 0.00 | 848.00 | 146 | 139059 | 1.443× | True |
| small | sparse | 0.0523956 | 0.0803593 | 48.00 | 2064.00 | 181 | 174224 | 1.534× | True |
| medium | dense | 0.893029 | 0.975014 | 0.00 | 39072.00 | 148 | 203396 | 1.092× | True |
| medium | ard | 0.911766 | 0.998063 | 1232.00 | 35280.00 | 166 | 218992 | 1.095× | True |
| medium | sparse | 0.915037 | 0.962326 | 64.00 | 29888.00 | 201 | 284264 | 1.052× | True |

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
| small | dense | -7482.8126944828036 | -7482.8126944828073 | [Values](small/dense/numerical_answers.md), [every entry](small/dense/paired_answers/) |
| small | ard | -4846.209530364119 | -4846.2095303641036 | [Values](small/ard/numerical_answers.md), [every entry](small/ard/paired_answers/) |
| small | sparse | -4883.0986274630413 | -4883.0986274630541 | [Values](small/sparse/numerical_answers.md), [every entry](small/sparse/paired_answers/) |
| medium | dense | -50028.396954164862 | -50028.396954164906 | [Values](medium/dense/numerical_answers.md), [every entry](medium/dense/paired_answers/) |
| medium | ard | -11978.331750931773 | -11978.331750931837 | [Values](medium/ard/numerical_answers.md), [every entry](medium/ard/paired_answers/) |
| medium | sparse | -12088.113965444551 | -12088.11396544442 | [Values](medium/sparse/numerical_answers.md), [every entry](medium/sparse/paired_answers/) |
