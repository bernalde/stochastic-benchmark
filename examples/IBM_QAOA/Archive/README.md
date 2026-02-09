# IBM QAOA Data Processing

This directory contains tools for processing IBM quantum hardware QAOA (Quantum Approximate Optimization Algorithm) experimental data for MaxCut problems.

## Directory Structure

The processing pipeline requires N-based subdirectory organization:

```
IBM_QAOA/
├── instances/
│   ├── random_regular/N{n}/*.json    # R3R problem instances
│   └── heavy_hex/N{n}/*.json         # Heavy-hex problem instances
├── R3R/
│   └── minmax_cuts/N{n}/*.json       # Minmax cuts for R3R
├── heavy_hex/
│   └── minmax_cuts/N{n}/*.json       # Minmax cuts for heavy-hex  
├── R3R_results_hardware/              # Hardware results (R3R topology)
├── heavy_hex_results_hardware/        # Hardware results (heavy-hex topology)
└── ibm_qaoa_processing.py            # Main processing script
```

**Note**: All data must be organized in N-based subdirectories (e.g., `N40/`, `N144/`). Flat structures are not supported.

## Key Terminology

### QAOA Parameters
- **p** (QAOA layers): Number of parameter layers in QAOA circuit
  - Each layer consists of a pair of parameters (γ, β)
  - Common values: p = 2, 4, 5, 6, 10
  - ⚠️ **Not** circuit depth (which includes all gates)

### Qubit Terminology
- **Program qubits**: Qubits as defined in the problem Hamiltonian (pre-mapping)
- **Physical qubits**: Actual hardware qubits after SAT mapping
- **Edge map**: Dictionary mapping program qubit indices to physical qubit indices

### Energy and Cut Values
- **eval_energy**: QAOA expectation value `<H>` from hardware (can be positive or negative)
- **cut_value**: Sum of weights for edges crossing the partition (always positive)
- **Approximation ratio**: Performance metric in range [0, 1]

## Energy Convention

QAOA for MaxCut uses the Hamiltonian:

```
H = -0.5 * Σ w_ij (1 - Z_i Z_j) for all edges (i,j)
```

Key conversions:
```python
# Transform QAOA energy to cut value
cut_val = eval_energy + 0.5 * sum_weights

# Calculate approximation ratio
approx_ratio = (cut_val - min_cut) / (max_cut - min_cut)
```

## Processing Pipeline

### 1. Load Instance Data
```python
edges = load_instance_file(instance_id, n_nodes, topology='R3R')
# Returns: Dict[(i,j), weight] for program qubits
```

### 2. Load Minmax Cuts
```python
minmax_data = load_minmax_cuts(minmax_dir='R3R/minmax_cuts', n_nodes=40)
# Returns: Dict with min_cut, max_cut, sum_weights for each instance
```

### 3. Process Hardware Trials
```python
result = parse_hardware_trial(trial_data, instance_id, depth, minmax_data, topology)
# Returns: QAOAResult with approximation_ratio, bitstring_cut_values, etc.
```

### 4. Convert to DataFrame
```python
df = results_to_dataframe(results, parameter_names)
# Returns: DataFrame with columns: instance, p, Approximation_Ratio, MeanTime, etc.
```

## Quick Start

```python
import ibm_qaoa_processing as ibm

# Process hardware data
sb, agg_df = ibm.process_qaoa_data(
    json_pattern='heavy_hex_results_hardware/N144/*.json',
    output_dir='heavy_hex_results_hardware',
    n_nodes=144,
    topology='heavy_hex',
    minmax_dir='heavy_hex/minmax_cuts'
)

# View results
print(agg_df[['instance', 'p', 'Approximation_Ratio', 'method']].head())
```

## Data Files

### Instance Files
Format: `{instance_id}_{params}_weighted.json`
```json
{
  "edge list": [
    {"nodes": [0, 1], "weight": 1.0},
    {"nodes": [1, 2], "weight": 1.0}
  ],
  "Description": "..."
}
```

### Minmax Cuts Files
Format: `{instance_id}_{params}_maxmin_cut.json`
```json
{
  "min_cut": 20.0,
  "max_cut": 80.0,
  "sum_of_weights": 100.0,
  "instance": "instances/random_regular/000_40nodes_random3regular.json"
}
```

### Hardware Results Files
Format: `{method}_{instance_id}_p{p}_{hash}.json`
```json
{
  "trial_data": {
    "counts": {"0101...": 1, "1010...": 1}
  },
  "circuit_metadata": {
    "eval_energy": 10.5,
    "params": [0.5, 1.2, ...],
    "trainer": "FA_PP_opt_10"
  },
  "sat_mapper": {
    "edge_map": {"0": 0, "1": 5, ...},
    "duration": 12.3
  }
}
```

## Topologies

### R3R (Random 3-Regular)
- **Structure**: Random 3-regular graphs
- **Node counts**: 10, 20, 40, 50, 60, 70 (All nodes not available for all methods)
- **SAT mapping**: Non-identity (requires bitstring remapping)
- **QAOA Parameter training methods tested**: F, FA, I, TQA, TS, RTS

### Heavy-Hex
- **Structure**: IBM native heavy-hex topology  
- **Node counts**: 12, 39, 105, 106, 144 (All nodes not available for all methods)
- **SAT mapping**: Identity mapping (no remapping needed)
- **Methods tested**: F, I, TQA

## Common Issues

### 1. Approximation Ratios > 1
**Solution**: Fixed by applying the canonical energy-to-cut transformation: `cut_val = energy + 0.5 * sum_weights`. The `evaluate_bitstring_cut_value()` function now returns cut values directly.

### 2. Missing N-based Subdirectory
**Error**: `ValueError: N-based subdirectory not found`  
**Solution**: Organize files in N-based subdirectories (e.g., `minmax_cuts/N40/`)

### 3. Bitstring Length Mismatch
**Warning**: `Bitstring length != edge_map size`  
**Cause**: Hardware measurement on wrong number of qubits  
**Impact**: Mismatched bitstrings are skipped

## Analysis Notebook

See `ibm_qaoa_analysis_hardware.ipynb` and - `ibm_qaoa_analysis.ipynb` for:
- Data loading and exploration
- Approximation ratio analysis by method and p
- Performance comparisons across topologies
- Bootstrap analysis for confidence intervals

## Files

- `ibm_qaoa_processing.py` - Main processing pipeline
- `ibm_qaoa_analysis.ipynb` - Simulation results analysis notebook
- `ibm_qaoa_analysis_hardware.ipynb` - Hardware results analysis notebook
- `transfer_simulation_data.py` - Transfer simulation experiment data from source repository
- `transfer_hardware_data.py` - Transfer hardware experiment data from source repository
- `test_approx_ratio_fix.py` - Validation tests for approximation ratio fix
- `demo_realistic_fix.py` - Demonstration with realistic values

## References

- **QAOA-Parameter-Setting repository**: https://github.com/uantum-Working-Groups QAOA-Parameter-Settin

- **Stochastic Benchmark framework**: Performance and statistical analysis tools

- **IBM Quantum**: Hardware execution platform

## Contact

For questions about the data or processing pipeline, refer to the QAOA-Parameter-Setting repository.
