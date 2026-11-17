# MOFA comparison

Machine: Apple M1; 8 logical cores; 16.00 GB RAM.

| Case | Model | Julia (s) | Python (s) | Julia memory (kB) | Python memory (kB) | Julia allocations | Python allocations | Python/Julia time | Numerical match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | dense | 0.00197067 | 0.0141923 | 0.00 | 48.00 | 116 | 78493 | 7.202× | True |
| tiny | ard | 0.00192538 | 0.0192734 | 0.00 | 48.00 | 134 | 93725 | 10.010× | True |
| tiny | sparse | 0.00226292 | 0.0256897 | 0.00 | 80.00 | 163 | 115901 | 11.352× | True |
| small | dense | 0.0927613 | 0.0791146 | 0.00 | 1568.00 | 128 | 123615 | 0.853× | True |
| small | ard | 0.0916856 | 0.0837875 | 0.00 | 2272.00 | 146 | 139059 | 0.914× | True |
| small | sparse | 0.0964758 | 0.0939293 | 0.00 | 2928.00 | 181 | 174224 | 0.974× | True |
| medium | dense | 1.60297 | 1.31723 | 80.00 | 43024.00 | 148 | 203396 | 0.822× | True |
| medium | ard | 1.5709 | 1.47676 | 128.00 | 35008.00 | 166 | 218992 | 0.940× | True |
| medium | sparse | 1.65835 | 1.51019 | 160.00 | 43536.00 | 201 | 284264 | 0.911× | True |

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
| small | dense | -7482.8126944828009 | -7482.8126944828073 | [Values](small/dense/numerical_answers.md), [every entry](small/dense/paired_answers/) |
| small | ard | -4846.2095303641145 | -4846.2095303641036 | [Values](small/ard/numerical_answers.md), [every entry](small/ard/paired_answers/) |
| small | sparse | -4883.0986274630422 | -4883.0986274630541 | [Values](small/sparse/numerical_answers.md), [every entry](small/sparse/paired_answers/) |
| medium | dense | -50028.396954164906 | -50028.396954164906 | [Values](medium/dense/numerical_answers.md), [every entry](medium/dense/paired_answers/) |
| medium | ard | -11978.331750931806 | -11978.331750931837 | [Values](medium/ard/numerical_answers.md), [every entry](medium/ard/paired_answers/) |
| medium | sparse | -12088.113965444551 | -12088.11396544442 | [Values](medium/sparse/numerical_answers.md), [every entry](medium/sparse/paired_answers/) |
