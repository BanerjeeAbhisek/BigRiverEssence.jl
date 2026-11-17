# MOFA comparison

Machine: Apple M1; 8 logical cores; 17.18 GB RAM.

| Case | Model | Julia min (s) | Python min (s) | Python/Julia | Numerical match |
|---|---|---:|---:|---:|---|
| tiny | dense | 0.000428417 | 0.00396375 | 9.252× | True |
| tiny | ard | 0.000449542 | 0.00484021 | 10.767× | True |
| tiny | sparse | 0.000482667 | 0.00552142 | 11.439× | True |

Python/Julia > 1 means Julia was faster in this run. Check numerical agreement before interpreting speed.

See benchmarks.csv for min/median times, baseline/peak/incremental RSS, and language-specific allocation diagnostics.
RSS includes native memory but is sampled and can miss brief peaks. Warm allocators may produce zero incremental RSS; this is not zero allocation.
Julia cumulative allocations and Python retained traced blocks/peak traced bytes are different quantities and must not be ranked against each other.
