# IBM_QAOA Notebooks

This folder contains notebooks used to analyze IBM Quantum QAOA experiments (hardware runs + parameter-training runs) and to integrate the resulting measurements into the `stochastic-benchmark` style performance/resource analysis.

Upstream repository: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting

## Files

- `Analysis.ipynb`
  - End-to-end analysis notebook:
    - Loads IBM hardware data and training data (via `../src/Processing.py`)
    - Computes MaxCut approximation ratios (via `../src/approx_ratio_calc.py`)
    - Builds derived resource-cost columns (e.g., total duration = QPU time + estimated training time)
    - Produces performance plots and training-cost breakdown plots

## Notes

- The notebook expects local data directories/instance directories to be available (paths are currently set inside the notebook).
- Several helper functions used by the notebook live in `../src/utils.py`.
