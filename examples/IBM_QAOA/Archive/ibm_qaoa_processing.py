"""
IBM QAOA Data Processing Pipeline

This module processes IBM quantum hardware and simulation QAOA (Quantum Approximate Optimization Algorithm) 
results for MaxCut problems. It handles data ingestion, cut value calculations, 
approximation ratio computation and uses the stochastic benchmark framework to generate window
stickers (performance plots) to determine best parameter setting strategies.

Directory Structure:
===================
Requires N-based subdirectory organization:
- instances/random_regular/N{n}/*.json (problem instance files)
- minmax_cuts/N{n}/*.json (minmax cut files)
- Results organized by N value for efficient loading

Terminology:
===========
- p (QAOA layers): Number of QAOA parameter layers, not circuit depth
- Program qubits: Qubits in the problem Hamiltonian (before hardware mapping)
- Physical qubits: Actual hardware qubits (after SAT mapping)
- Cut value: Sum of weights for edges crossing the partition

Time Tracking:
=============
The module tracks multiple time components for comprehensive resource analysis:

1. **TrainTime** (Parameter Training):
   - Source: Simulation JSON files, nested keys "0", "1", "2", etc.
   - Path: data["2"][str(p)]["train_duration"] for F/I methods
   - Meaning: Time spent optimizing QAOA parameters using classical optimizer
   - Both R3R and heavy_hex topologies

2. **SATMappingTime** (Pre-processing):
   - Source: Hardware JSON files, 'sat mapper' -> 'duration'
   - Meaning: Time to map logical problem qubits to physical hardware qubits
   - R3R: Non-zero (actual SAT mapping required)
   - heavy_hex: ~0.0 (identity mapping, native topology)

3. **TotalTime** (Complete Resource Cost):
   - Formula: TotalTime = TrainTime + SATMappingTime
   - Recommended for end-to-end performance ratio analysis
   - Captures full algorithm resource consumption

4. **MeanTime** (Deprecated):
   - Kept for backwards compatibility
   - Now equals TotalTime

Resource Column Selection:
-------------------------
Use the `resource_col` parameter in `setup_bootstrap_parameters()` to select which
time component to analyze:
- 'TotalTime': Complete resource cost (default, recommended)
- 'TrainTime': Parameter training only
- 'SATMappingTime': Pre-processing only

Example:
    >>> bs_params = setup_bootstrap_parameters(resource_col='TotalTime')
    >>> # For comparative analysis:
    >>> results = run_comparative_time_analysis(...)

IMPORTANT - Energy Convention:
==============================
QAOA for MaxCut uses the Hamiltonian: H = -0.5 * Σ w_ij (1 - Z_i Z_j) for edges (i,j)

Key conventions:
- eval_energy (from hardware JSON): QAOA expectation value <H> (can be positive or negative)
- cut_value: Sum of weights for cut edges (always positive)
- Transformation: cut_val = energy + 0.5 * sum_weights (from graph_utils.py)
- Pre-factor: -0.5 for MaxCut QAOA
- Maximization: QAOA maximizes <H>, so better solutions have larger (less negative) energies

Example:
  If eval_energy = 10.0 and sum_weights = 100.0:
    cut_val = 10.0 + 0.5 * 100.0 = 60.0
  Then: approx_ratio = (cut_val - min_cut) / (max_cut - min_cut)

Reference: graph_utils.py in QAOA-Parameter-Setting repository (canonical implementation)
"""

import os
import json
import re
import glob
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Add stochastic-benchmark src to path
sys.path.append('../../src')

# Stochastic Benchmark imports
import stochastic_benchmark
import bootstrap
import interpolate
import stats
import success_metrics
import names

# Setup logging
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class ProcessingConfig:
    """Configuration for IBM QAOA data processing."""
    persist_raw: bool = True                     # Write per-file pickles during ingestion
    interpolate_diversity_threshold: int = 6     # Skip interpolation if instances*depths < threshold
    fabricate_single_trial: bool = True          # Create synthetic bootstrap for single trials
    seed: int = 42                               # Random seed for train/test split
    log_progress_interval: int = 50              # Log progress every N files

@dataclass
class QAOAResult:
    """Data structure to hold parsed QAOA result for a single trial.
    
    Time Fields:
    -----------
    train_duration: SAT mapping pre-processing time from hardware files (seconds)
                   - R3R topology: Non-zero (actual SAT mapping required)
                   - heavy_hex topology: 0.0 (identity mapping, native topology)
                   - Source: hardware JSON 'sat mapper' -> 'duration'
    
    sim_train_duration: QAOA parameter training time from simulation files (seconds)
                       - Both topologies: Parameter optimization time using classical optimizer
                       - Source: simulation JSON nested keys "0", "1", "2", etc.
                       - Example path: data["2"][str(p)]["train_duration"] for F/I methods
    
    Derived Columns (in DataFrame):
    ------------------------------
    TrainTime: Parameter training time (sim_train_duration)
    SATMappingTime: Pre-processing time (train_duration)
    TotalTime: Sum of TrainTime + SATMappingTime (complete resource cost)
    """
    trial_id: int
    instance_id: str
    depth: int  # Number of QAOA layers (p parameter)
    energy: float
    approximation_ratio: float
    success: bool
    train_duration: float = np.nan  # SAT mapping duration (R3R: actual time, heavy_hex: ~0.0)
    trainer_name: str = ""
    evaluator: Optional[str] = None  # Datatype could be str or None
    sim_train_duration: float = np.nan  # Parameter training duration from simulation files
    bitstring_energies: Optional[List[float]] = None  # Individual bitstring cut values for bootstrap analysis
    optimized_params: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)

def load_minmax_cuts(minmax_dir: str = 'R3R/minmax_cuts', n_nodes: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Load minmax cuts data from JSON files in N-based subdirectory structure.
    
    Args:
        minmax_dir: Base directory containing N-based subdirectories (e.g., 'R3R/minmax_cuts')
        n_nodes: Number of nodes/qubits (required). Must match subdirectory N{n_nodes}/
        
    Returns:
        Dictionary mapping instance_id to minmax cuts data
        Format: {instance_id: {'min_cut': float, 'max_cut': float, 'sum_weights': float, 'n_nodes': int}}
        
    Raises:
        ValueError: If n_nodes is None or if N-based subdirectory doesn't exist
    """
    minmax_data = {}
    
    if not os.path.exists(minmax_dir):
        raise ValueError(f"Minmax cuts directory not found: {minmax_dir}")
    
    if n_nodes is None:
        raise ValueError("n_nodes parameter is required. Specify number of qubits (e.g., n_nodes=40 for N40)")
    
    # Load from N-based subdirectory only
    n_subdir = os.path.join(minmax_dir, f'N{n_nodes}')
    if not os.path.exists(n_subdir):
        raise ValueError(f"N-based subdirectory not found: {n_subdir}. "
                        f"Please organize minmax cuts in N-based subdirectories (e.g., {minmax_dir}/N{n_nodes}/)")
    
    json_files = glob.glob(os.path.join(n_subdir, '*.json'))
    logger.info(f"Loading {len(json_files)} minmax cut files from {n_subdir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Primary: Extract instance ID from 'instance' field path
            instance_path = data.get('instance', '')
            instance_id = None
            file_n_nodes = None
            
            if instance_path:
                # Parse instance path to extract instance_id
                # Examples:
                #   "instances/random_regular/000_40nodes_random3regular.json"
                #   "instances/heavy_hex/000_7_3_heavyhex_144nodes_weighted.json"
                path_match = re.search(r'/(\d+)_.*?(\d+)nodes', instance_path)
                if path_match:
                    instance_id = path_match.group(1)
                    file_n_nodes = int(path_match.group(2))
            
            # Fallback: Extract from filename if instance field not found
            if instance_id is None:
                filename = os.path.basename(json_file)
                # Pattern 1 (R3R): 000_40nodes_random3regular_maxmin_cut.json
                # Pattern 2 (heavy_hex): 000_7_3_heavyhex_144nodes_weighted_maxmin_cut.json
                match = re.match(r'(\d+)_.*?(\d+)nodes_', filename)
                if match:
                    instance_id = match.group(1)
                    file_n_nodes = int(match.group(2))
            
            # Skip if we couldn't extract instance_id
            if instance_id is None:
                logger.warning(f"Could not extract instance_id from {json_file}")
                continue
            
            # Filter by node count if specified
            if n_nodes is not None and file_n_nodes != n_nodes:
                continue
            
            minmax_data[instance_id] = {
                'min_cut': data.get('min_cut', 0.0),
                'max_cut': data.get('max_cut', 0.0),
                'sum_of_weights': data.get('sum_of_weights', 0.0),  # Keep original key name
                'n_nodes': file_n_nodes,
                'instance_path': instance_path  # Store for reference
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading minmax cuts file {json_file}: {e}")
    
    logger.info(f"Loaded minmax cuts data for {len(minmax_data)} instances")
    return minmax_data

def load_instance_file(instance_id: str, node_count: int, topology: str = 'R3R', 
                       instance_dir: str = 'instances') -> Dict[Tuple[int, int], float]:
    """
    Load instance file directly and extract edges.
    
    Instance JSON format:
    {
        "edge list": [
            {"nodes": [i, j], "weight": w},
            ...
        ],
        "Description": "..."
    }
    
    Args:
        instance_id: Instance ID (e.g., "000")
        node_count: Number of nodes (e.g., 40)
        topology: 'R3R' or 'heavy_hex'
        instance_dir: Base directory for instance files
        
    Returns:
        Dictionary mapping (node_i, node_j) tuples to edge weights
        Same format as parse_cost_operator_to_edges() for compatibility
        
    Raises:
        FileNotFoundError: If instance file not found
        ValueError: If instance file has invalid format
    """
    if topology == 'R3R':
        # R3R pattern: 000_40nodes_random3regular.json (N-based subdirectory)
        pattern = f"{instance_dir}/random_regular/N{node_count}/{instance_id}_{node_count}nodes_random3regular.json"
        matching_files = glob.glob(pattern)
    else:  # heavy_hex
        # Heavy hex pattern: 000_*_*_heavyhex_144nodes_weighted.json (N-based subdirectory)
        # Layer dimensions vary, so use glob
        pattern = f"{instance_dir}/heavy_hex/N{node_count}/{instance_id}_*_heavyhex_{node_count}nodes_weighted.json"
        matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No instance file found matching pattern: {pattern}")
    
    if len(matching_files) > 1:
        logger.warning(f"Multiple instance files match pattern {pattern}, using first: {matching_files[0]}")
    
    instance_file = matching_files[0]
    logger.debug(f"Loading instance from: {instance_file}")
    
    try:
        with open(instance_file, 'r') as f:
            instance_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Error loading instance file {instance_file}: {e}")
    
    # Extract edges from "edge list" key
    edge_list_data = instance_data.get('edge list', [])
    if not edge_list_data:
        raise ValueError(f"Instance file {instance_file} has empty or missing 'edge list'")
    
    # Convert to dictionary format: (node_i, node_j) -> weight
    edges = {}
    for edge_data in edge_list_data:
        nodes = edge_data.get('nodes', [])
        weight = edge_data.get('weight', 1.0)
        if len(nodes) != 2:
            logger.warning(f"Invalid edge format in {instance_file}: {edge_data}")
            continue
        # Normalize edge order (always store as (min, max))
        edge_tuple = tuple(sorted([nodes[0], nodes[1]]))
        edges[edge_tuple] = abs(float(weight))
    
    logger.debug(f"Loaded {len(edges)} edges from instance {instance_id} (N={node_count}, topology={topology})")
    return edges

@lru_cache(maxsize=256)
def load_simulation_training_duration(instance_id: str, method: str, evaluator: str, 
                                      p: int, topology: str, nodes: int) -> float:
    """
    Load training duration from simulation file for heavy-hex topology.
    
    For heavy-hex, the SAT mapping is identity (native topology), so SAT mapping
    duration is 0. Instead, we need the QAOA parameter training time from simulation files.
    
    Args:
        instance_id: Instance ID (e.g., "000")
        method: Method name (e.g., "F", "I", "TQA")
        evaluator: "MPS" or "PP"
        p: QAOA depth/layers
        topology: Topology name (e.g., "heavy_hex", "R3R")
        nodes: Number of nodes (e.g., 144)
    
    Returns:
        Training duration in seconds, or NaN if not found
    """
    # Only load for heavy_hex topology (R3R uses SAT mapping duration)
    if topology != "heavy_hex":
        return np.nan
    
    # Build pattern to find simulation file
    # Example: heavy_hex/F/N144/20251020_155631_000N144HH73_MC_F_PP_optMW6_10.json
    pattern = f"{topology}/{method}/N{nodes}/*_{instance_id}N{nodes}HH*_MC_{method}_{evaluator}_*_{p}.json"
    
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        logger.warning(f"No simulation file found for training duration: "
                      f"instance={instance_id}, method={method}, evaluator={evaluator}, p={p}, N={nodes}")
        return np.nan
    
    try:
        with open(matching_files[0], 'r') as f:
            sim_data = json.load(f)
        
        # Extract train_duration for this p value
        # Two file structures exist:
        # 1. F/I methods: Recursive structure with top-level "0","1","2" and nested p values
        #    - p<=2: data[str(p)]["train_duration"]
        #    - p>2:  data["2"][str(p)]["train_duration"]
        # 2. TQA methods: Simple structure with top-level "0","1" only
        #    - p=5:  data["0"]["train_duration"]
        #    - p=10: data["1"]["train_duration"]
        
        p_key = str(p)
        
        # Try direct key mapping first (for p <= 2, or exact match)
        if p_key in sim_data and isinstance(sim_data[p_key], dict) and "train_duration" in sim_data[p_key]:
            train_duration = sim_data[p_key]["train_duration"]
            logger.debug(f"Loaded training duration {train_duration:.2f}s for "
                       f"{instance_id}/{method}/{evaluator}/p={p} (direct key)")
            return float(train_duration)
        
        # Try nested structure for F/I methods with p > 2
        if "2" in sim_data and isinstance(sim_data["2"], dict):
            if p_key in sim_data["2"] and isinstance(sim_data["2"][p_key], dict):
                if "train_duration" in sim_data["2"][p_key]:
                    train_duration = sim_data["2"][p_key]["train_duration"]
                    logger.debug(f"Loaded training duration {train_duration:.2f}s for "
                               f"{instance_id}/{method}/{evaluator}/p={p} (nested recursive)")
                    return float(train_duration)
        
        # TQA special case: p=5 → key "0", p=10 → key "1"
        # Check if this is a TQA file by looking for characteristic keys
        if "args" in sim_data or len([k for k in sim_data.keys() if k.isdigit()]) <= 2:
            # Map p values to TQA keys
            tqa_key_map = {5: "0", 10: "1"}
            if p in tqa_key_map:
                tqa_key = tqa_key_map[p]
                if tqa_key in sim_data and isinstance(sim_data[tqa_key], dict):
                    if "train_duration" in sim_data[tqa_key]:
                        train_duration = sim_data[tqa_key]["train_duration"]
                        logger.debug(f"Loaded training duration {train_duration:.2f}s for "
                                   f"{instance_id}/{method}/{evaluator}/p={p} (TQA mapping)")
                        return float(train_duration)
        
        logger.warning(f"No train_duration found for p={p} in {matching_files[0]}")
        return np.nan
        
    except (json.JSONDecodeError, IOError, KeyError) as e:
        logger.warning(f"Error loading training duration from {matching_files[0]}: {e}")
        return np.nan

def maxcut_approximation_ratio(energy: float, min_cut: float, max_cut: float, sum_weights: float) -> float:
    """
    Calculate MaxCut approximation ratio using the formula:
    cut_val = energy + 0.5 * sum_weights
    approximation_ratio = (cut_val - min_cut) / (max_cut - min_cut)
    
    Args:
        energy: Energy value from QAOA result
        min_cut: Minimum cut value for the instance
        max_cut: Maximum cut value for the instance
        sum_weights: Sum of all edge weights
        
    Returns:
        Approximation ratio (0.0 to 1.0), or NaN if calculation fails
    """
    if np.isnan(energy):
        return np.nan
    
    # Calculate cut value
    cut_val = energy + 0.5 * sum_weights
    
    # Calculate approximation ratio
    denominator = max_cut - min_cut
    if denominator == 0:
        logger.warning("Max cut equals min cut - cannot calculate approximation ratio")
        return np.nan
    
    approx_ratio = (cut_val - min_cut) / denominator
    return approx_ratio

def parse_qaoa_trial(trial_data: Dict[str, Any], trial_id: int, instance_id: str, depth: int) -> QAOAResult:
    """
    Parse a single trial dictionary into a QAOAResult object.
    """
    # Extract basic metrics
    energy = trial_data.get('energy', np.nan)
    approx_ratio = trial_data.get('approximation ratio', np.nan)
    duration = trial_data.get('train_duration', 0.0)
    
    # Extract metadata - handle trainer dict
    trainer_info = trial_data.get('trainer', {})
    if isinstance(trainer_info, dict):
        trainer = trainer_info.get('trainer_name', 'Unknown')
        evaluator = trainer_info.get('evaluator', None)
    else:
        trainer = str(trainer_info) if trainer_info else 'Unknown'
        evaluator = None
    
    success = trial_data.get('success', False)
    
    # Extract parameters (gamma, beta, etc.)
    params = trial_data.get('optimal_params', [])
    
    # Extract history if available
    history = trial_data.get('history', [])
    
    return QAOAResult(
        trial_id=trial_id,
        instance_id=instance_id,
        depth=depth,
        energy=energy,
        approximation_ratio=approx_ratio,
        train_duration=duration,
        trainer_name=trainer,
        evaluator=evaluator,
        success=success,
        optimized_params=params,
        energy_history=history
    )

def load_qaoa_results(json_data: Dict[str, Any]) -> List[QAOAResult]:
    """
    Load and parse a list of QAOA trial dictionaries.
    """
    results = []
    # The JSON structure is a dict with trial IDs as keys
    if isinstance(json_data, dict):
        trial_keys = [key for key in json_data if key.isdigit()]
        for trial_id in trial_keys:
            trial_data = json_data[trial_id]
            # We pass placeholders for instance_id and depth as they are added later
            res = parse_qaoa_trial(trial_data, int(trial_id), "0", 0)
            results.append(res)
    return results

def convert_to_dataframe(qaoa_results: List[QAOAResult], instance_id: str, p: int, optimized: Optional[str] = None, minmax_data: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """
    Convert a list of QAOA result objects to a DataFrame.
    
    Parameters:
        qaoa_results: List of QAOA result objects
        instance_id: Instance identifier
        p: Circuit depth
        optimized: Optimization flag ('opt', 'noOpt', or None if not specified)
        minmax_data: Dictionary of minmax cuts data for calculating approximation ratios
    """
    data = []
    for res in qaoa_results:
        # Extract parameters
        params = res.optimized_params
        
        # Calculate approximation ratio using minmax cuts data if available
        approx_ratio = res.approximation_ratio
        # Check if energy is numeric and not NaN
        energy_is_valid = isinstance(res.energy, (int, float)) and not np.isnan(res.energy)
        if minmax_data and instance_id in minmax_data and energy_is_valid:
            cuts = minmax_data[instance_id]
            calculated_ratio = maxcut_approximation_ratio(
                res.energy,
                cuts['min_cut'],
                cuts['max_cut'],
                cuts['sum_of_weights']
            )
            # Use calculated ratio if original is NaN, otherwise prefer calculated
            approx_is_valid = isinstance(approx_ratio, (int, float)) and not np.isnan(approx_ratio)
            if not approx_is_valid or not np.isnan(calculated_ratio):
                approx_ratio = calculated_ratio
        
        # Create separate time columns for flexible resource analysis
        # TrainTime: Parameter training from simulation (both topologies)
        # SATMappingTime: Pre-processing from hardware SAT mapper (R3R: actual time, heavy_hex: ~0.0)
        # TotalTime: Sum of both components (complete resource cost)
        
        # Handle train_duration (SAT mapping time)
        # Check if it's a valid numeric value, otherwise use 0.0
        if isinstance(res.train_duration, (int, float)) and not np.isnan(res.train_duration):
            sat_mapping_time = res.train_duration
        else:
            sat_mapping_time = 0.0
        
        # Handle sim_train_duration (parameter training time)
        # Check if it's a valid numeric value, otherwise use 0.0
        if isinstance(res.sim_train_duration, (int, float)) and not np.isnan(res.sim_train_duration):
            train_time = res.sim_train_duration
        else:
            train_time = 0.0
        
        total_time = train_time + sat_mapping_time
        
        row = {
            'trial_id': res.trial_id,
            'instance': str(instance_id),  # Keep as string to preserve leading zeros
            'p': p,
            'Energy': res.energy,
            'Approximation_Ratio': approx_ratio,
            'TrainTime': train_time,
            'SATMappingTime': sat_mapping_time,
            'TotalTime': total_time,
            'MeanTime': total_time,  # Deprecated: kept for backwards compatibility, use TotalTime instead
            'trainer': res.trainer_name,
            'evaluator': res.evaluator,
            'success': res.success,
            'n_iterations': len(res.energy_history),
            'optimized': optimized,
            'bitstring_energies': res.bitstring_energies  # Cache for bootstrap
        }
        
        # Add parameters
        if params:
            for k, val in enumerate(params):
                row[f'param_{k}'] = val
                
        data.append(row)
        
    df = pd.DataFrame(data)
    return df

def prepare_stochastic_benchmark_data(df: pd.DataFrame, instance_id: str, p: int, output_dir: str) -> str:
    """
    Save the DataFrame to a pickle file formatted for stochastic benchmark.
    
    Note: Files are saved to output_dir/exp_raw/ to match StochasticBenchmark expectations.
    """
    # StochasticBenchmark expects raw data in exp_raw subdirectory
    raw_data_dir = os.path.join(output_dir, "exp_raw")
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        
    # Construct filename
    # Format: raw_results_inst=<instance_id>_depth=<p>.pkl
    filename = f"raw_results_inst={instance_id}_depth={p}.pkl"
    filepath = os.path.join(raw_data_dir, filename)
    
    # Save
    df.to_pickle(filepath)
    return filepath

def setup_qaoa_benchmark(here: str = 'exp_raw') -> Tuple[stochastic_benchmark.stochastic_benchmark, List[str]]:
    """
    Initialize and configure the stochastic benchmark object.
    """
    parameter_names = ['gamma', 'beta', 'p']
    response_key = 'PerfRatio'
    
    sb = stochastic_benchmark.stochastic_benchmark(
        parameter_names=parameter_names,
        response_key=response_key,
        here=here,
        instance_cols=['instance'],
        smooth=True
    )
    
    return sb, parameter_names

def setup_bootstrap_parameters(resource_col: str = 'TotalTime') -> bootstrap.BSParams_range_iter:
    """
    Configure bootstrap parameters with configurable resource column.
    
    Args:
        resource_col: Time column to use for resource analysis. Options:
                     - 'TotalTime': Sum of training + SAT mapping (default, complete resource cost)
                     - 'TrainTime': Parameter training time only
                     - 'SATMappingTime': Pre-processing time only
                     - 'MeanTime': Deprecated, same as TotalTime for backwards compatibility
    """
    shared_args = {
        'response_col': 'Approximation_Ratio',
        'resource_col': resource_col,
        'response_dir': 1,  # Maximize
        'confidence_level': 68,
        'random_value': 0.0
    }

    metric_args = {}
    metric_args['Response'] = {'opt_sense': -1} 
    metric_args['SuccessProb'] = {'gap': 0.1, 'response_dir': 1}
    metric_args['RTT'] = {
        'fail_value': np.nan,
        'RTT_factor': 1.0,
        'gap': 0.1,
        's': 0.99
    }

    def update_rules(self, df):
        GTMinEnergy = df['GTMinEnergy'].iloc[0]
        self.shared_args['best_value'] = GTMinEnergy
        # Use the configured resource column for RTT factor
        self.metric_args['RTT']['RTT_factor'] = df[resource_col].iloc[0]

    sms = [
        success_metrics.Response,
        success_metrics.PerfRatio,
        success_metrics.SuccessProb,
        success_metrics.Resource
    ]

    boots_range = range(10, 51, 10)

    bsParams = bootstrap.BootstrapParameters(
        shared_args=shared_args,
        update_rule=update_rules,
        agg='count',
        metric_args=metric_args,
        success_metrics=sms,
        keep_cols=[]
    )

    bs_iter_class = bootstrap.BSParams_range_iter()
    bsparams_iter = bs_iter_class(bsParams, boots_range)
    
    return bsparams_iter

# ============================================================================
# HARDWARE-SPECIFIC FUNCTIONS
# ============================================================================

def organize_hardware_files(hardware_dir: str = "R3R/Hardware") -> Dict[Tuple[int, str], List[str]]:
    """
    Organize hardware files by node count and instance ID.
    Physically moves files into subdirectories: N40/, N50/, etc.
    
    Args:
        hardware_dir: Directory containing hardware JSON files
        
    Returns:
        Dictionary mapping (node_count, instance_id) tuples to lists of file paths
    """
    if not os.path.exists(hardware_dir):
        logger.error(f"Hardware directory not found: {hardware_dir}")
        return {}
    
    hardware_files = glob.glob(os.path.join(hardware_dir, "*.json"))
    logger.info(f"Found {len(hardware_files)} hardware files in {hardware_dir}")
    
    organized = {}
    for filepath in hardware_files:
        filename = os.path.basename(filepath)
        # Pattern 1 (R3R): ###N##R3R_depth_hash.json (e.g., 000N40R3R_2_d498d5hlag1s73biv770.json)
        # Pattern 2 (heavy_hex): ###N##HH##_depth_hash.json (e.g., 000N105HH72_10_hash.json)
        
        # Try R3R pattern first
        match = re.match(r'(\d+)N(\d+)R3R_(\d+)_([a-z0-9]+)\.json', filename)
        if match:
            instance_id = match.group(1)
            node_count = int(match.group(2))
            depth = int(match.group(3))
            
            key = (node_count, instance_id)
            if key not in organized:
                organized[key] = []
            organized[key].append(filepath)
        else:
            # Try heavy_hex pattern
            match = re.match(r'(\d+)N(\d+)HH\d+_(\d+)_([a-z0-9]+)\.json', filename)
            if match:
                instance_id = match.group(1)
                node_count = int(match.group(2))
                depth = int(match.group(3))
                
                key = (node_count, instance_id)
                if key not in organized:
                    organized[key] = []
                organized[key].append(filepath)
    
    logger.info(f"Organized hardware files into {len(organized)} groups")
    return organized

# Hardware processing loads edges directly from instances/ directory using load_instance_file()

def validate_edge_map_program_qubits(edge_map: Dict[str, int], expected_program_qubits: int) -> bool:
    """
    Validate that edge_map is complete and consistent.
    
    Args:
        edge_map: Dictionary mapping program qubit indices to physical indices
        expected_program_qubits: Expected number of program qubits
        
    Returns:
        True if valid, False otherwise
    """
    if not edge_map:
        logger.error("Edge map is empty")
        return False
    
    # Check all program indices 0 to N-1 are present
    program_indices = set(int(k) for k in edge_map.keys())
    expected_indices = set(range(expected_program_qubits))
    
    if program_indices != expected_indices:
        missing = expected_indices - program_indices
        extra = program_indices - expected_indices
        if missing:
            logger.error(f"Edge map missing program qubits: {missing}")
        if extra:
            logger.warning(f"Edge map has extra program qubits: {extra}")
        return False
    
    # Check no duplicate physical qubit mappings
    physical_indices = list(edge_map.values())
    if len(physical_indices) != len(set(physical_indices)):
        logger.error("Edge map has duplicate physical qubit assignments")
        return False
    
    return True

def remap_bitstring_to_program_order(bitstring_physical: str, edge_map: Dict[str, int]) -> str:
    """
    Remap bitstring from physical qubit order to program qubit order.
    
    Args:
        bitstring_physical: Bitstring in physical qubit order (as measured on hardware)
        edge_map: Dictionary mapping program indices to physical indices
        
    Returns:
        Bitstring in program qubit order (for Hamiltonian evaluation)
    """
    n_qubits = len(edge_map)
    
    # Note: Length check moved to caller to avoid excessive logging
    if len(bitstring_physical) != n_qubits:
        return bitstring_physical  # Return as-is if mismatch
    
    # Create program bitstring by reordering according to edge_map
    # program_bit[i] = physical_bit[edge_map[str(i)]]
    program_bits = ['0'] * n_qubits
    for program_idx in range(n_qubits):
        physical_idx = edge_map[str(program_idx)]
        program_bits[program_idx] = bitstring_physical[physical_idx]
    
    return ''.join(program_bits)

def evaluate_bitstring_cut_value(bitstring_program: str, edges: Dict[Tuple[int, int], float]) -> float:
    """
    Evaluate MaxCut cut value for a single bitstring.
    
    Args:
        bitstring_program: Bitstring in program qubit order (string of '0' and '1')
        edges: Dictionary of edges (i, j) -> weight
        
    Returns:
        Cut value: Sum of weights for edges crossing the partition (always positive)
        Note: This is the direct cut value, not the QAOA energy H = -0.5 * cut_value
    """
    cut_value = 0.0
    
    for (i, j), weight in edges.items():
        # XOR: edge is cut if bits differ
        if bitstring_program[i] != bitstring_program[j]:
            cut_value += weight
    
    return cut_value

def parse_method_name(method_str: str) -> Dict[str, Any]:
    """
    Parse hardware method string into standardized components.
    
    Hardware methods combine: trainer_evaluator_opt_reps
    Examples:
      - "FA_PP_opt_2" -> trainer='FA', evaluator='PP', optimized='opt', reps=2
      - "I_MPS_10" -> trainer='I', evaluator='MPS', optimized=None, reps=10
      - "F_PP_10" -> trainer='F', evaluator='PP', optimized=None, reps=10
    
    Args:
        method_str: Raw method string from hardware metadata
        
    Returns:
        Dictionary with keys: trainer, evaluator, optimized, reps, full_method
    """
    parts = method_str.split('_')
    
    result = {
        'trainer': parts[0] if len(parts) > 0 else 'Unknown',
        'evaluator': parts[1] if len(parts) > 1 else None,
        'optimized': None,
        'reps': None,
        'full_method': method_str
    }
    
    # Check for optimization flag and reps
    if len(parts) >= 3:
        # Pattern: trainer_evaluator_opt_reps or trainer_evaluator_reps
        if parts[2] in ['opt', 'noOpt']:
            result['optimized'] = parts[2]
            if len(parts) >= 4:
                try:
                    result['reps'] = int(parts[3])
                except ValueError:
                    pass
        else:
            # No opt flag, just reps
            try:
                result['reps'] = int(parts[2])
            except ValueError:
                pass
    
    return result

def parse_hardware_trial(trial_data: Dict[str, Any],
                        minmax_data: Dict[str, Dict[str, float]], 
                        instance_id: str, depth: int, topology: str = 'R3R') -> QAOAResult:
    """
    Parse a single hardware trial into QAOAResult object.
    
    Args:
        trial_data: Trial data from hardware JSON
        minmax_data: Min/max cuts for approximation ratio calculation
        instance_id: Instance identifier
        depth: Circuit depth
        topology: 'R3R' or 'heavy_hex'
        
    Returns:
        QAOAResult object with hardware-specific fields populated
    """
    # Extract method and standardize
    circuit_metadata = trial_data.get('metadata', {}).get('circuit_metadata', {})
    method_str = circuit_metadata.get('method', 'Unknown')
    method_info = parse_method_name(method_str)
    
    # Extract edge map and validate
    sat_mapper = circuit_metadata.get('sat mapper', {})
    edge_map = sat_mapper.get('edge_map', {})
    
    # Convert edge_map keys to strings if they aren't already
    edge_map = {str(k): int(v) for k, v in edge_map.items()}
    
    # Get node count from minmax_data (more reliable than inferring)
    node_count = minmax_data.get(instance_id, {}).get('n_nodes', 0)
    
    # If edge_map is empty (heavy_hex topology), create identity mapping
    # This means bitstrings are already in logical node order
    if len(edge_map) == 0:
        if node_count > 0:
            # Create identity mapping: logical[i] = physical[i]
            edge_map = {str(i): i for i in range(node_count)}
            logger.debug(f"Empty edge_map detected, using identity mapping for {node_count} qubits (heavy_hex)")
        else:
            logger.error(f"Cannot determine node count for instance {instance_id}")
            node_count = 0
    else:
        # R3R uses non-identity SAT mapping
        node_count = len(edge_map)
        if not validate_edge_map(edge_map, node_count):
            logger.warning(f"Invalid edge map for method {method_str}, instance {instance_id}")
    
    # Load edges directly from instance file (canonical source)
    try:
        edges = load_instance_file(instance_id, node_count, topology)
        logger.debug(f"Loaded {len(edges)} edges from instance file for {instance_id}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load instance file for {instance_id}: {e}")
        raise RuntimeError(f"Cannot process hardware trial without instance file for {instance_id}")
    
    # Use eval_energy from hardware JSON (this is the expected energy)
    mean_energy = circuit_metadata.get('eval_energy', np.nan)
    
    # Extract bitstring counts and compute individual cut values for bootstrapping
    counts_dict = trial_data.get('counts', {})
    
    bitstring_cut_values = []
    mismatch_logged = False  # Only log mismatch once per trial
    for bitstring_physical, count in counts_dict.items():
        # Check length before remapping to avoid excessive logging
        if len(bitstring_physical) != len(edge_map):
            if not mismatch_logged:
                logger.warning(f"Bitstring length {len(bitstring_physical)} != edge_map size {len(edge_map)} for instance {instance_id}, p={depth}. Skipping mismatched bitstrings.")
                mismatch_logged = True
            continue  # Skip this bitstring
        
        bitstring_program = remap_bitstring_to_program_order(bitstring_physical, edge_map)
        cut_value = evaluate_bitstring_cut_value(bitstring_program, edges)
        # Store cut value 'count' times (though count is always 1 in your data)
        bitstring_cut_values.extend([cut_value] * count)
    
    # Calculate approximation ratio using canonical formula from graph_utils.py
    # eval_energy from hardware JSON is QAOA expectation value <H> (can be positive or negative)
    # Transform to cut value using: cut_val = energy + 0.5 * sum_weights
    # Then: approx_ratio = (cut_val - min_cut) / (max_cut - min_cut)
    approx_ratio = np.nan
    if instance_id in minmax_data and not np.isnan(mean_energy):
        cuts = minmax_data[instance_id]
        sum_weights = cuts['sum_of_weights']
        
        # Canonical transformation: energy to cut value
        # From graph_utils.py: cut_val = energy + 0.5 * sum_weights
        cut_val = mean_energy + 0.5 * sum_weights
        
        # Calculate approximation ratio
        denominator = cuts['max_cut'] - cuts['min_cut']
        if denominator != 0:
            approx_ratio = (cut_val - cuts['min_cut']) / denominator
        else:
            logger.warning(f"Max cut equals min cut for instance {instance_id} - cannot calculate approximation ratio")
            approx_ratio = np.nan
        
        # Validate approximation ratio is in reasonable bounds [0, 1]
        # Allow small tolerance for numerical precision
        if not np.isnan(approx_ratio) and (approx_ratio < -0.01 or approx_ratio > 1.01):
            logger.warning(f"Instance {instance_id}: approximation ratio {approx_ratio:.4f} outside [0,1] range. "
                         f"eval_energy={mean_energy:.4f}, cut_val={cut_val:.4f}, "
                         f"min_cut={cuts['min_cut']:.4f}, max_cut={cuts['max_cut']:.4f}")
    
    # Extract other metadata from circuit_metadata
    params = circuit_metadata.get('params', [])
    duration = sat_mapper.get('duration', 0.0)  # SAT mapper duration
    trainer = circuit_metadata.get('trainer', 'Unknown')
    
    # Parse method string to extract components for training duration lookup
    # Format: "F_MPS_10", "I_PP_10", "TQA_MPS_opt_10", etc.
    # Extract: method_base (F/I/TQA), evaluator (MPS/PP), p_value
    method_match = re.match(r'([A-Z]+)_(MPS|PP)(?:_opt)?_(\d+)', method_str)
    sim_train_duration = np.nan
    
    # Load training duration for BOTH topologies (previously only heavy_hex)
    # This ensures TotalTime includes both SAT mapping and parameter training
    if method_match:
        method_base = method_match.group(1)  # F, I, TQA, etc.
        evaluator = method_match.group(2)     # MPS or PP
        p_from_method = int(method_match.group(3))  # p value from method string
        
        # Load training duration from simulation file
        sim_train_duration = load_simulation_training_duration(
            instance_id=instance_id,
            method=method_base,
            evaluator=evaluator,
            p=p_from_method,
            topology=topology,
            nodes=node_count
        )
    
    # Trial ID: meaningful identifier combining instance, p, and method
    # Format: inst{instance_id}_p{depth}_{method}
    trial_id = f"inst{instance_id}_p{depth}_{method_str}"
    
    return QAOAResult(
        trial_id=trial_id,
        instance_id=instance_id,
        depth=depth,
        energy=mean_energy,
        approximation_ratio=approx_ratio,
        train_duration=duration,
        trainer_name=method_info['trainer'],
        evaluator=method_info['evaluator'],
        sim_train_duration=sim_train_duration,  # Training duration from simulation
        success=True,  # Hardware runs are considered "successful" measurements
        optimized_params=params,
        bitstring_energies=bitstring_cut_values  # Store cut values for bootstrap sampling
    )

# ============================================================================
# END HARDWARE-SPECIFIC FUNCTIONS
# ============================================================================

def group_name_fcn(raw_filename):
    """Extract group name from filename.
    
    IBM-SPECIFIC: Assumes filename format raw_results_inst=<id>_depth=<p>.pkl
    """
    raw_filename = os.path.basename(raw_filename)
    try:
        start_idx = raw_filename.index('inst')
        end_idx = raw_filename.index('.')
        return raw_filename[start_idx:end_idx]
    except ValueError:
        return raw_filename

def process_qaoa_data(json_pattern: str = "R3R/*.json", output_dir: str = "exp_raw", 
                     config: Optional[ProcessingConfig] = None, minmax_dir: str = "R3R/minmax_cuts",
                     hardware_mode: bool = False) -> Tuple[stochastic_benchmark.stochastic_benchmark, pd.DataFrame]:
    """
    Main processing function for both simulation and hardware data.
    
    IBM-SPECIFIC: Default json_pattern targets IBM QAOA R3R experimental data directory.
    
    Args:
        json_pattern: Glob pattern for JSON files
        output_dir: Directory for output artifacts
        config: Processing configuration (uses defaults if None)
        minmax_dir: Directory containing minmax cuts JSON files
        hardware_mode: If True, process hardware files with bitstring energy evaluation
    """
    if config is None:
        config = ProcessingConfig()
    
    # ========================================================================
    # HARDWARE MODE: Process quantum device measurement data
    # ========================================================================
    if hardware_mode:
        logger.info("Processing in HARDWARE MODE - analyzing quantum device measurements")
        
        # Find hardware JSON files
        json_files = glob.glob(json_pattern)
        logger.info(f"Found {len(json_files)} hardware JSON files.")
        
        # Extract N value from first hardware file to load matching minmax cuts
        n_nodes = None
        if json_files:
            first_file = os.path.basename(json_files[0])
            # Match both R3R pattern (N40R3R) and heavy-hex pattern (N144HH73)
            n_match = re.search(r'N(\d+)(?:R3R|HH)', first_file)
            if n_match:
                n_nodes = int(n_match.group(1))
                logger.info(f"Detected N={n_nodes} from hardware filename, loading matching minmax cuts")
        
        # Load minmax cuts data for approximation ratio calculation (filtered by N if detected)
        minmax_data = load_minmax_cuts(minmax_dir, n_nodes=n_nodes)
        logger.info(f"Minmax cuts loaded: {len(minmax_data)} instances available for N={n_nodes}")
        
        all_qaoa_df = []
        data_files = []
        
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    hardware_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping malformed file {json_file}: {e}")
                continue
            
            # Parse filename: ###N##R3R_depth_hash.json or ###N##HH##_depth_hash.json
            filename = os.path.basename(json_file)
            
            # Try R3R pattern first
            match = re.match(r'(\d+)N(\d+)R3R_(\d+)_([a-z0-9]+)\.json', filename)
            topology = 'R3R'
            if not match:
                # Try heavy_hex pattern
                match = re.match(r'(\d+)N(\d+)HH\d+_(\d+)_([a-z0-9]+)\.json', filename)
                topology = 'heavy_hex'
            
            if not match:
                logger.warning(f"Filename doesn't match hardware pattern: {filename}")
                continue
            
            instance_id = match.group(1)
            node_count = int(match.group(2))
            depth = int(match.group(3))
            
            logger.debug(f"Processing {filename}: instance={instance_id}, N={node_count}, p={depth}, topology={topology}")
            
            # Hardware files contain list of trials (8 methods per file)
            if not isinstance(hardware_data, list):
                logger.warning(f"Hardware file {filename} is not a list of trials")
                continue
            
            # Validate edge_map consistency across trials
            edge_maps = []
            for trial in hardware_data:
                circuit_metadata = trial.get('metadata', {}).get('circuit_metadata', {})
                edge_map = circuit_metadata.get('sat mapper', {}).get('edge_map', {})
                edge_maps.append(edge_map)
            
            # Check if all edge_maps are identical
            if len(edge_maps) > 1:
                first_map = json.dumps(edge_maps[0], sort_keys=True)
                for idx, emap in enumerate(edge_maps[1:], 1):
                    if json.dumps(emap, sort_keys=True) != first_map:
                        logger.warning(f"Edge map differs for trial {idx} in {filename}")
            
            # Process each trial (8 methods)
            qaoa_results = []
            for trial_data in hardware_data:
                try:
                    result = parse_hardware_trial(trial_data, minmax_data, 
                                                 instance_id, depth, topology=topology)
                    qaoa_results.append(result)
                except Exception as e:
                    logger.warning(f"Error parsing trial in {filename}: {e}")
                    continue
            
            # Convert to DataFrame with standardized columns
            if qaoa_results:
                df = convert_to_dataframe(qaoa_results, instance_id, depth, 
                                         optimized=None, minmax_data=minmax_data)
                
                # Add method-specific columns from hardware
                for idx, result in enumerate(qaoa_results):
                    circuit_metadata = hardware_data[idx].get('metadata', {}).get('circuit_metadata', {})
                    method_str = circuit_metadata.get('method', 'Unknown')
                    method_info = parse_method_name(method_str)
                    
                    # Remove p value from method name (it's redundant with the 'p' column)
                    # Method names like "F_MPS_10" → "F_MPS"
                    # This allows same method to appear at different p values
                    method_clean = re.sub(r'_\d+$', '', method_info['full_method'])
                    
                    if idx < len(df):
                        df.loc[idx, 'method'] = method_clean
                        df.loc[idx, 'optimized'] = method_info['optimized']
                        df.loc[idx, 'reps'] = method_info['reps']
                
                # Add GTMinEnergy proxy
                if len(df) > 0 and 'GTMinEnergy' not in df.columns:
                    df['GTMinEnergy'] = df['Energy'].min()
                
                all_qaoa_df.append(df)
                
                # Save raw pickle if requested
                if config.persist_raw:
                    data_file = prepare_stochastic_benchmark_data(df, instance_id, depth, output_dir)
                    data_files.append(data_file)
            
            # Progress logging (more frequent for hardware processing)
            if (i + 1) % min(10, config.log_progress_interval) == 0 or (i + 1) == len(json_files):
                logger.info(f"Processed {i+1}/{len(json_files)} hardware files...")
        
        # Aggregate hardware results
        if all_qaoa_df:
            agg_df = pd.concat(all_qaoa_df, ignore_index=True)
            agg_df['instance'] = agg_df['instance'].astype(int)
            agg_df = agg_df.sort_values(by=['instance', 'p']).reset_index(drop=True)
            logger.info(f"Hardware processing complete: {len(agg_df)} total trials")
        else:
            agg_df = pd.DataFrame()
            logger.warning("No hardware data processed")
            return None, agg_df
    
    # ========================================================================
    # SIMULATION MODE: Process statevector/simulation data (original logic)
    # ========================================================================
    else:
        # 1. Load and Parse
        # IBM-SPECIFIC: File discovery via glob pattern
        json_files = glob.glob(json_pattern)
        logger.info(f"Found {len(json_files)} JSON files.")
        
        # Handle case where no files match the pattern
        if len(json_files) == 0:
            logger.warning(f"No files found matching pattern: {json_pattern}")
            logger.warning("Returning empty results - this topology/method/N combination has no data")
            return None, pd.DataFrame()
        
        # Extract N value from first simulation file to load matching minmax cuts
        n_nodes = None
        if json_files:
            first_file = os.path.basename(json_files[0])
            # Match both R3R pattern (N10R3R) and heavy-hex pattern (N12HH11, N144HH73, etc.)
            n_match = re.search(r'N(\d+)(?:R3R|HH)', first_file)
            if n_match:
                n_nodes = int(n_match.group(1))
                logger.info(f"Detected N={n_nodes} from simulation filename, loading matching minmax cuts")
        
        # Load minmax cuts data for approximation ratio calculation (filtered by N if detected)
        minmax_data = load_minmax_cuts(minmax_dir, n_nodes=n_nodes)
        
        all_qaoa_df = []
        data_files = []
        
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    qaoa_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping malformed file {json_file}: {e}")
                continue
                
            qaoa_results = load_qaoa_results(qaoa_data)
            
        # IBM-SPECIFIC: Extract instance_id, depth, and optimization flag from IBM filename format
        # Expected format: YYYYMMDD_HHMMSS_###N##R3R_MC_FA_SV_noOpt_#.json or ..._opt_#.json
        # where ### is instance, final # is depth p
        filename = os.path.basename(json_file)
        parts = filename.split('_')
        
        # Instance ID
        try:
            instance_str = parts[2]
            match = re.match(r'(\d+)', instance_str)
            instance_id = match.group(1) if match else '0'
        except IndexError:
            instance_id = '0'
        
        # Optimization flag (check if filename contains _opt, _noOpt, or neither)
        if '_noOpt_' in filename:
            optimized = 'noOpt'
        elif '_opt_' in filename:
            optimized = 'opt'
        else:
            optimized = None  # No optimization flag in filename
            
        # Depth (p)
        try:
            depth_str = parts[-1].replace('.json', '')
            p = int(depth_str)
        except (ValueError, IndexError):
            p = None
            
        # Convert to DataFrame with minmax cuts data
        df = convert_to_dataframe(qaoa_results, instance_id, p, optimized, minmax_data)
        
        # IBM-SPECIFIC: Add GTMinEnergy proxy when ground truth unavailable
        # Uses minimum observed energy as best-known value for bootstrap update_rules
        if len(df) > 0 and 'GTMinEnergy' not in df.columns:
             # Use min energy found in this file as proxy
             df['GTMinEnergy'] = df['Energy'].min()

        if len(df) > 0:  # Only add non-empty DataFrames
            all_qaoa_df.append(df)
            
            # Optionally save raw pickle (legacy behavior)
            if config.persist_raw:
                data_file = prepare_stochastic_benchmark_data(df, instance_id, p, output_dir)
                data_files.append(data_file)
        
            # Progress logging
            if (i + 1) % config.log_progress_interval == 0:
                logger.info(f"Processed {i+1}/{len(json_files)} files...")
        
        # Aggregate
        if all_qaoa_df:
            agg_df = pd.concat(all_qaoa_df, ignore_index=True)
            agg_df['instance'] = agg_df['instance'].astype(int)
            agg_df = agg_df.sort_values(by=['instance', 'p']).reset_index(drop=True)
        else:
            agg_df = pd.DataFrame()
            return None, agg_df

    # ========================================================================
    # BOOTSTRAP AND INTERPOLATION (Common to both modes)
    # ========================================================================
    
    # 2. Setup Benchmark
    # When persist_raw=True, StochasticBenchmark expects files directly in output_dir
    # When persist_raw=False, we'll generate temp pickles in output_dir
    sb, param_names = setup_qaoa_benchmark(here=output_dir)
    
    # 3. Bootstrap REMOVED - deferred to notebook for bitstring-level resampling
    # Bootstrap will be performed in notebook after visualization cells using
    # cached bitstring energies for proper uncertainty quantification
    
    # Initialize empty bootstrap results
    sb.bs_results = None
    
    # 4. Interpolation SKIPPED - not applicable without bootstrap
    logger.info("Skipping interpolation (bootstrap deferred to notebook)")
    sb.interp_results = None

    return sb, agg_df

def save_bitstring_energies(agg_df: pd.DataFrame, output_dir: str = "bitstring_cache") -> str:
    """
    Save cached bitstring energies from all trials to pickle file for bootstrap sampling.
    
    Args:
        agg_df: DataFrame with trial data including 'bitstring_energies' column
        output_dir: Directory to save cache file
        
    Returns:
        Path to saved cache file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract bitstring energies for each trial
    cache = {}
    for idx, row in agg_df.iterrows():
        key = (row['instance'], row['p'], row['method'])
        if 'bitstring_energies' in row and row['bitstring_energies'] is not None:
            cache[key] = {
                'energies': row['bitstring_energies'],
                'instance': row['instance'],
                'depth': row['p'],
                'method': row['method'],
                'trial_id': row['trial_id'],
                'mean_energy': row['Energy'],
                'approx_ratio': row['Approximation_Ratio']
            }
    
    cache_file = os.path.join(output_dir, "bitstring_energies_cache.pkl")
    pd.to_pickle(cache, cache_file)
    logger.info(f"Saved bitstring energies for {len(cache)} trials to {cache_file}")
    return cache_file

def load_bitstring_energies(cache_file: str = "bitstring_cache/bitstring_energies_cache.pkl") -> Dict:
    """
    Load cached bitstring energies for bootstrap sampling.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        Dictionary mapping (instance, depth, method) -> energy data
    """
    cache = pd.read_pickle(cache_file)
    logger.info(f"Loaded bitstring energies for {len(cache)} trials from {cache_file}")
    return cache

def get_bitstring_energies_for_trial(cache: Dict, instance: int, depth: int, method: str) -> Optional[List[float]]:
    """
    Retrieve bitstring energies for a specific trial for bootstrap resampling.
    
    Args:
        cache: Dictionary from load_bitstring_energies()
        instance: Instance ID
        depth: Circuit depth (p)
        method: Method name
        
    Returns:
        List of individual bitstring energies, or None if not found
    """
    key = (instance, depth, method)
    if key in cache:
        return cache[key]['energies']
    return None

def run_comparative_time_analysis(json_pattern: str, output_dir: str, 
                                  minmax_dir: str, hardware_mode: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run bootstrap analysis with different time columns for comparison.
    
    This function demonstrates how to compare performance using different resource metrics:
    - TotalTime: Complete resource cost (training + SAT mapping)
    - TrainTime: Parameter training time only
    - SATMappingTime: Pre-processing time only
    
    Args:
        json_pattern: Glob pattern for JSON files
        output_dir: Directory for output artifacts
        minmax_dir: Directory containing minmax cuts
        hardware_mode: If True, process hardware files
        
    Returns:
        Dictionary mapping resource_col name to bootstrap results DataFrame
        
    Example:
        >>> results = run_comparative_time_analysis(
        ...     json_pattern='heavy_hex/*.json',
        ...     output_dir='heavy_hex_results',
        ...     minmax_dir='heavy_hex/minmax_cuts',
        ...     hardware_mode=True
        ... )
        >>> # Compare TotalTime vs TrainTime performance
        >>> total_time_results = results['TotalTime']
        >>> train_time_results = results['TrainTime']
    """
    time_columns = ['TotalTime', 'TrainTime', 'SATMappingTime']
    comparative_results = {}
    
    for resource_col in time_columns:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Running analysis with resource_col='{resource_col}'")
        logger.info(f"{'='*60}\\n")
        
        # Process data (only needs to be done once, but repeated for clarity)
        sb, agg_df = process_qaoa_data(
            json_pattern=json_pattern,
            output_dir=output_dir,
            minmax_dir=minmax_dir,
            hardware_mode=hardware_mode
        )
        
        if agg_df.empty:
            logger.warning(f"No data processed for {resource_col}")
            continue
        
        # Setup bootstrap with specific resource column
        bs_params = setup_bootstrap_parameters(resource_col=resource_col)
        
        # Run bootstrap analysis (if you want to run it here)
        # Note: In the current workflow, bootstrap is deferred to notebooks
        # This is just a demonstration of how to configure it
        
        # Store results
        comparative_results[resource_col] = agg_df.copy()
        
        logger.info(f"Completed analysis for {resource_col}")
        logger.info(f"Mean {resource_col}: {agg_df[resource_col].mean():.4f} seconds")
        logger.info(f"Std {resource_col}: {agg_df[resource_col].std():.4f} seconds\\n")
    
    return comparative_results

if __name__ == "__main__":
    process_qaoa_data()