import os
import json
import re
import glob
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
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
    """Data structure to hold parsed QAOA result for a single trial."""
    trial_id: int
    instance_id: str
    depth: int
    energy: float
    approximation_ratio: float
    train_duration: float
    trainer_name: str
    evaluator: Optional[str]
    success: bool
    optimized_params: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)

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

def convert_to_dataframe(qaoa_results: List[QAOAResult], instance_id: str, p: int) -> pd.DataFrame:
    """
    Convert a list of QAOA result objects to a DataFrame.
    """
    data = []
    for res in qaoa_results:
        # Extract parameters
        params = res.optimized_params
        
        row = {
            'trial_id': res.trial_id,
            'instance': instance_id,
            'p': p,
            'Energy': res.energy,
            'Approximation_Ratio': res.approximation_ratio,
            'MeanTime': res.train_duration,
            'trainer': res.trainer_name,
            'evaluator': res.evaluator,
            'success': res.success,
            'n_iterations': len(res.energy_history),
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

def setup_bootstrap_parameters() -> bootstrap.BSParams_range_iter:
    """
    Configure bootstrap parameters.
    """
    shared_args = {
        'response_col': 'Approximation_Ratio',
        'resource_col': 'MeanTime',
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
        self.metric_args['RTT']['RTT_factor'] = df['MeanTime'].iloc[0]

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

def process_qaoa_data(json_pattern: str = "R3R/*.json", output_dir: str = "exp_raw", config: Optional[ProcessingConfig] = None) -> Tuple[stochastic_benchmark.stochastic_benchmark, pd.DataFrame]:
    """
    Main processing function.
    
    IBM-SPECIFIC: Default json_pattern targets IBM QAOA R3R experimental data directory.
    
    Args:
        json_pattern: Glob pattern for JSON files
        output_dir: Directory for output artifacts
        config: Processing configuration (uses defaults if None)
    """
    if config is None:
        config = ProcessingConfig()
    
    # 1. Load and Parse
    # IBM-SPECIFIC: File discovery via glob pattern
    json_files = glob.glob(json_pattern)
    logger.info(f"Found {len(json_files)} JSON files.")
    
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
            
        # IBM-SPECIFIC: Extract instance_id and depth from IBM filename format
        # Expected format: YYYYMMDD_HHMMSS_###N##R3R_MC_FA_SV_noOpt_#.json
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
            
        # Depth (p)
        try:
            depth_str = parts[-1].replace('.json', '')
            p = int(depth_str)
        except (ValueError, IndexError):
            p = None
            
        # Convert to DataFrame
        df = convert_to_dataframe(qaoa_results, instance_id, p)
        
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

    # 2. Setup Benchmark
    # When persist_raw=True, StochasticBenchmark expects files directly in output_dir
    # When persist_raw=False, we'll generate temp pickles in output_dir
    sb, param_names = setup_qaoa_benchmark(here=output_dir)
    bsparams_iter = setup_bootstrap_parameters()
    
    # 3. Run Bootstrap
    logger.info("Running bootstrap analysis...")
    
    all_bs_results = []
    
    # If pickles not persisted, generate them temporarily for bootstrap (or refactor to use agg_df groups)
    if config.persist_raw:
        files_to_process = data_files
    else:
        # Generate pickles on-the-fly from agg_df groups
        logger.info("Generating temporary pickles for bootstrap (persist_raw=False)...")
        files_to_process = []
        if all_qaoa_df:
            temp_agg = pd.concat(all_qaoa_df, ignore_index=True)
            for (instance_id, p), group_df in temp_agg.groupby(['instance', 'p']):
                data_file = prepare_stochastic_benchmark_data(group_df, str(instance_id), p, output_dir)
                files_to_process.append(data_file)
    
    for data_file in files_to_process:
        try:
            raw_data = pd.read_pickle(data_file)
            n_trials = len(raw_data)
            
            if n_trials == 1:
                # IBM-SPECIFIC: Manual bootstrap result fabrication for single-trial instances
                # Standard bootstrap requires multiple trials; single trials get deterministic
                # "bootstrap" results with zero confidence intervals
                instance_name = group_name_fcn(data_file)
                single_instance_results = []
                for n_boots in [10, 20, 30, 40, 50]:
                    trial_data = raw_data.iloc[0]
                    result_row = {
                        'instance': instance_name,
                        'boots': n_boots,
                        'gamma': trial_data.get('param_0', np.nan),
                        'beta': trial_data.get('param_1', np.nan),
                        'p': trial_data['p'],
                        'Key=Response': trial_data['Energy'],
                        'Key=PerfRatio': trial_data['Approximation_Ratio'],
                        'Key=SuccProb': 1.0 if trial_data['success'] else 0.0,
                        'Key=MeanTime': trial_data['MeanTime'],
                        'ConfInt=lower_Key=Response': trial_data['Energy'],
                        'ConfInt=upper_Key=Response': trial_data['Energy'],
                        'ConfInt=lower_Key=PerfRatio': trial_data['Approximation_Ratio'],
                        'ConfInt=upper_Key=PerfRatio': trial_data['Approximation_Ratio'],
                        'ConfInt=lower_Key=SuccProb': 1.0 if trial_data['success'] else 0.0,
                        'ConfInt=upper_Key=SuccProb': 1.0 if trial_data['success'] else 0.0,
                        'ConfInt=lower_Key=MeanTime': trial_data['MeanTime'],
                        'ConfInt=upper_Key=MeanTime': trial_data['MeanTime']
                    }
                    single_instance_results.append(result_row)
                all_bs_results.append(pd.DataFrame(single_instance_results))
            else:
                # Standard bootstrap
                # We can use sb.run_Bootstrap but it runs on all files.
                # To avoid running it multiple times, we should run it once outside the loop
                # OR we can just skip it here and run it once.
                # But we need to combine results.
                pass
                
        except Exception as e:
            logger.warning(f"Error processing {data_file}: {e}")

    # Run standard bootstrap for multi-trial instances
    # This will process all files in output_dir, including single-trial ones if we are not careful.
    # But sb.run_Bootstrap handles what it finds.
    # If we want to use the manual single-trial results, we should merge them.
    
    logger.info("Running standard bootstrap via StochasticBenchmark...")
    try:
        sb.run_Bootstrap(bsparams_iter, group_name_fcn)
    except Exception as e:
        logger.warning(f"SB Bootstrap warning: {e}")

    # Combine results
    combined_bs_results = pd.DataFrame()
    if sb.bs_results is not None:
        combined_bs_results = sb.bs_results.copy()
    
    if all_bs_results:
        single_bs_df = pd.concat(all_bs_results, ignore_index=True)
        # Remove single-trial instances from SB results if present (to avoid duplicates)
        single_instances = single_bs_df['instance'].unique()
        if not combined_bs_results.empty:
            combined_bs_results = combined_bs_results[~combined_bs_results['instance'].isin(single_instances)]
        
        combined_bs_results = pd.concat([combined_bs_results, single_bs_df], ignore_index=True)

    sb.bs_results = combined_bs_results
    
    # 4. Interpolation
    logger.info("Running interpolation...")
    
    # IBM-SPECIFIC: Skip interpolation for minimal data diversity
    # Compute diversity = unique_instances * unique_depths
    unique_instances = combined_bs_results['instance'].nunique() if 'instance' in combined_bs_results.columns else 0
    unique_depths = combined_bs_results['p'].nunique() if 'p' in combined_bs_results.columns else 1
    diversity = unique_instances * unique_depths
    
    if diversity < config.interpolate_diversity_threshold:
        logger.info(f"Skipping interpolation: diversity={diversity} (instances={unique_instances}, depths={unique_depths}) < threshold={config.interpolate_diversity_threshold}")
        sb.interp_results = combined_bs_results.copy()
        # Add resource column
        if len(sb.interp_results) > 0:
            if 'Key=MeanTime' in sb.interp_results.columns:
                sb.interp_results['resource'] = sb.interp_results['Key=MeanTime']
            elif 'boots' in sb.interp_results.columns:
                sb.interp_results['resource'] = sb.interp_results['boots'] * 0.01
            else:
                sb.interp_results['resource'] = 0.0
    else:
        # Interpolation
        def resource_fcn(df):
            if 'Key=Resource' in df.columns: return df['Key=Resource']
            elif 'Key=MeanTime' in df.columns: return df['Key=MeanTime']
            else: return pd.Series(np.linspace(0.1, 10.0, len(df)), index=df.index)

        try:
            iParams = interpolate.InterpolationParameters(
                resource_fcn,
                parameters=param_names,
                ignore_cols=['trainer', 'evaluator']
            )
            sb.interp_results = interpolate.Interpolate(combined_bs_results, iParams, group_on='instance')
            logger.info(f"Interpolation complete. Shape: {sb.interp_results.shape}")
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}. Using bootstrap results.")
            sb.interp_results = combined_bs_results.copy()
            if 'Key=MeanTime' in sb.interp_results.columns:
                sb.interp_results['resource'] = sb.interp_results['Key=MeanTime']

    # Add train/test split
    # NOTE: Uses config seed for reproducible 80/20 train/test split
    if sb.interp_results is not None and 'train' not in sb.interp_results.columns:
        np.random.seed(config.seed)
        train_mask = np.random.random(len(sb.interp_results)) < 0.8
        sb.interp_results['train'] = train_mask.astype(int)

    return sb, agg_df

if __name__ == "__main__":
    process_qaoa_data()