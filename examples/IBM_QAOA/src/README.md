# IBM_QAOA `src`

This folder contains lightweight utilities used by the IBM QAOA example notebooks (in particular, `notebooks/Analysis.ipynb`).

## Files

- `__init__.py`
  - Package marker for the IBM_QAOA example modules.

- `Processing.py`
  - Loads and parses IBM Quantum hardware result data and QAOA parameter-training data.
  - Provides helpers to locate data/instance files and normalize them into Python objects used for downstream analysis.

- `approx_ratio_calc.py`
  - MaxCut-specific helpers for approximation ratio calculations.
  - Locates per-instance min/max cut reference data, extracts needed parameters, and computes approximation ratios.

- `utils.py`
  - General helper utilities used by the notebooks:
    - DataFrame helpers (e.g., expanding IBM counts into sample rows)
    - Plotting helpers used in the analysis notebook
    - Small statistical/label helpers used by plots (e.g., `sem`, `title_from_instance_names`)
    - Factory helper `make_asof_per_file` to build the groupby-apply function used when merging cumulative training duration.
