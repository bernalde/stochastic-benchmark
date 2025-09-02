"""
IBM QAOA data processing script for stochastic benchmark
Converts IBM QAOA JSON results to format compatible with stochastic-benchmark
"""

import json
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

# Add stochastic-benchmark src to path
sys.path.append('../../src')
import bootstrap
import interpolate
import names
import random_exploration
import sequential_exploration
import stats
import stochastic_benchmark
import success_metrics
from utils_ws import *


@dataclass
class QAOAResult:
    """Data class to hold QAOA optimization results"""
    trial_id: int
    optimized_params: List[float]
    train_duration: float
    energy: float
    trainer_name: str
    method: str = None
    success: bool = False
    energy_history: List[float] = None
    parameter_history: List[List[float]] = None
    x0: List[float] = None


def load_qaoa_json(json_file: str) -> List[QAOAResult]:
    """
    Load QAOA results from JSON file and convert to QAOAResult objects
    
    Parameters
    ----------
    json_file : str
        Path to JSON file containing QAOA results
        
    Returns
    -------
    List[QAOAResult]
        List of QAOA optimization results
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for trial_id, trial_data in data.items():
        # Skip if this is just a random point (trainer_name == "RandomPoint")
        if trial_data.get('trainer', {}).get('trainer_name') == 'RandomPoint':
            continue
            
        # Extract energy - handle "NA" values
        energy = trial_data.get('energy', np.nan)
        if energy == "NA":
            energy = np.nan
            
        # Extract success - handle string boolean
        success_str = trial_data.get('success', 'False')
        success = success_str.lower() == 'true' if isinstance(success_str, str) else bool(success_str)
        
        trainer_info = trial_data.get('trainer', {})
        method = None
        if isinstance(trainer_info, dict):
            method = trainer_info.get('method', None)
        
        result = QAOAResult(
            trial_id=int(trial_id),
            optimized_params=trial_data.get('optimized_params', []),
            train_duration=trial_data.get('train_duration', 0.0),
            energy=energy,
            trainer_name=trainer_info.get('trainer_name', 'Unknown') if isinstance(trainer_info, dict) else str(trainer_info),
            method=method,
            success=success,
            energy_history=trial_data.get('energy_history', []),
            parameter_history=trial_data.get('parameter_history', []),
            x0=trial_data.get('x0', [])
        )
        results.append(result)
    
    return results


def qaoa_results_to_dataframe(results: List[QAOAResult], instance_id: int = 1) -> pd.DataFrame:
    """
    Convert QAOA results to DataFrame format compatible with stochastic-benchmark
    
    Parameters
    ----------
    results : List[QAOAResult]
        List of QAOA optimization results
    instance_id : int, default=1
        Instance identifier for this problem
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns expected by stochastic-benchmark
    """
    data_rows = []
    
    for result in results:
        # Create parameter columns - for QAOA typically gamma and beta
        params = result.optimized_params
        param_dict = {}
        if len(params) >= 1:
            param_dict['gamma'] = params[0]
        if len(params) >= 2:
            param_dict['beta'] = params[1]
        # Add more parameters if needed
        for i, param in enumerate(params[2:], start=2):
            param_dict[f'param_{i}'] = param
            
        row = {
            'trial_id': result.trial_id,
            'instance': instance_id,
            'Energy': result.energy if not np.isnan(result.energy) else -999,  # Use placeholder for missing energy
            'MeanTime': result.train_duration,
            'trainer': result.trainer_name,
            'method': result.method or 'Unknown',
            'success': result.success,
            'n_iterations': len(result.energy_history) if result.energy_history else 0,
            'count': 1,  # Each trial represents one evaluation
            **param_dict
        }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Add GTMinEnergy (ground truth minimum energy) - placeholder for now
    # In practice, this should be the known optimal energy for the problem
    df['GTMinEnergy'] = df['Energy'].min()  # Use best found energy as placeholder
    
    return df


def prepare_qaoa_raw_data(json_files: List[str], output_dir: str = 'exp_raw'):
    """
    Prepare raw QAOA data in format expected by stochastic-benchmark
    
    Parameters
    ----------
    json_files : List[str]
        List of paths to JSON files containing QAOA results
    output_dir : str, default='exp_raw'
        Directory to save processed raw data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, json_file in enumerate(json_files):
        instance_id = i + 1
        results = load_qaoa_json(json_file)
        df = qaoa_results_to_dataframe(results, instance_id)
        
        output_file = os.path.join(output_dir, f'raw_results_inst={instance_id}.pkl')
        df.to_pickle(output_file)
        print(f"Processed {json_file} -> {output_file} ({len(df)} trials)")


def postprocess_linear(recipe):
    """Linear postprocessing for QAOA parameters"""
    x_range = (0.01, 10.0)  # Reasonable range for QAOA training time
    post_recipe_dict = {}
    
    parameter_names = ['gamma', 'beta']
    x = np.array(recipe['resource'])
    range_idx = np.where((x >= x_range[0]) & (x <= x_range[1]))
    x = x[range_idx]
    post_recipe_dict['resource'] = x
    x = x.reshape((-1, 1))
    
    for param in parameter_names:
        if param in recipe.columns:
            param_vals = np.array(recipe[param])
            param_vals = param_vals[range_idx]
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(x, param_vals)
            param_pred = model.predict(x)
            post_recipe_dict[param] = param_pred
    
    pred_recipe = pd.DataFrame.from_dict(post_recipe_dict)
    return pred_recipe


def postprocess_random(meta_params):
    """Postprocessing for random search meta-parameters"""
    x_range = (0.01, 10.0)
    post_recipe_dict = {}
    
    parameter_names = ['ExploreFrac', 'tau']
    x = np.array(meta_params['TotalBudget'])
    range_idx = np.where((x >= x_range[0]) & (x <= x_range[1]))
    x = x[range_idx]
    post_recipe_dict['TotalBudget'] = x
    x = x.reshape((-1, 1))
    
    for param in parameter_names:
        if param in meta_params.columns:
            param_vals = np.array(meta_params[param])
            param_vals = param_vals[range_idx]
            if param == 'ExploreFrac':
                post_recipe_dict[param] = np.mean(param_vals) * np.ones_like(param_vals)
            elif param == 'tau':
                param_pred = param_vals.copy()
                range_idx_1 = np.where(x <= 1.0)[0]
                param_pred[range_idx_1] = 0.1
                range_idx_2 = np.where(x > 1.0)[0]
                param_pred[range_idx_2] = 1.0
                post_recipe_dict[param] = param_pred
    
    pred_recipe = pd.DataFrame.from_dict(post_recipe_dict)
    if 'ExploreFrac' in pred_recipe.columns and 'TotalBudget' in pred_recipe.columns:
        pred_recipe['ExplorationBudget'] = pred_recipe['TotalBudget'] * pred_recipe['ExploreFrac']
    return pred_recipe


def setup_qaoa_stochastic_benchmark():
    """
    Set up stochastic benchmark for QAOA data analysis
    
    Returns
    -------
    stochastic_benchmark.stochastic_benchmark
        Configured stochastic benchmark object
    """
    # Basic configuration
    here = os.getcwd()
    parameter_names = ['gamma', 'beta']  # QAOA parameters
    instance_cols = ['instance']
    
    # Response information
    response_key = 'PerfRatio'  # Will be computed from Energy
    response_dir = 1  # Maximize performance ratio
    
    # Optimization settings
    recover = True
    reduce_mem = True
    smooth = True
    
    sb = stochastic_benchmark.stochastic_benchmark(
        parameter_names, here, instance_cols, response_key, response_dir, recover, reduce_mem, smooth
    )
    
    # Bootstrap parameters
    shared_args = {
        'response_col': 'Energy',
        'resource_col': 'MeanTime',
        'response_dir': -1,  # Minimize energy
        'confidence_level': 68,
        'random_value': 0.0
    }
    
    metric_args = {}
    metric_args['Response'] = {'opt_sense': -1}
    metric_args['SuccessProb'] = {'gap': 0.1, 'response_dir': -1}  # Success within 10% of optimum
    metric_args['RTT'] = {
        'fail_value': np.nan, 
        'RTT_factor': 1.0,
        'gap': 0.1, 
        's': 0.99
    }
    
    def update_rules(self, df):
        """Update bootstrap parameters for each group"""
        GTMinEnergy = df['GTMinEnergy'].iloc[0]
        self.shared_args['best_value'] = GTMinEnergy
        self.metric_args['RTT']['RTT_factor'] = df['MeanTime'].iloc[0]
    
    # Success metrics
    sms = [
        success_metrics.Response,
        success_metrics.PerfRatio,
        success_metrics.InvPerfRatio,
        success_metrics.SuccessProb,
        success_metrics.Resource,
        success_metrics.RTT
    ]
    
    boots_range = range(10, 101, 10)  # Smaller range for QAOA data
    
    bsParams = bootstrap.BootstrapParameters(
        shared_args=shared_args,
        update_rule=update_rules,
        agg='count',
        metric_args=metric_args,
        success_metrics=sms,
        keep_cols=['trainer', 'method']
    )
    
    bs_iter_class = bootstrap.BSParams_range_iter()
    bsparams_iter = bs_iter_class(bsParams, boots_range)
    
    # Group name function
    def group_name_fcn(raw_filename):
        raw_filename = os.path.basename(raw_filename)
        start_idx = raw_filename.index('inst')
        end_idx = raw_filename.index('.')
        return raw_filename[start_idx:end_idx]
    
    # Run bootstrap
    sb.run_Bootstrap(bsparams_iter, group_name_fcn)
    
    # Interpolation
    def resource_fcn(df):
        return df['MeanTime'] * df['boots']  # Resource is time * bootstrap iterations
    
    iParams = interpolate.InterpolationParameters(
        resource_fcn,
        parameters=parameter_names,
        ignore_cols=['trainer', 'method']
    )
    
    sb.run_Interpolate(iParams)
    
    # Statistics
    train_test_split = 0.8
    metrics = ['Response', 'RTT', 'PerfRatio', 'SuccProb', 'MeanTime', 'InvPerfRatio']
    stParams = stats.StatsParameters(metrics=metrics, stats_measures=[stats.Median()])
    
    sb.run_Stats(stParams, train_test_split)
    
    return sb


def main():
    """Main execution function"""
    # Example usage - modify paths as needed
    json_files = ['20250721_171511_example.json']  # Add more files as needed
    
    # Prepare raw data
    print("Processing QAOA JSON files...")
    prepare_qaoa_raw_data(json_files)
    
    # Set up and run stochastic benchmark
    print("Setting up stochastic benchmark...")
    sb = setup_qaoa_stochastic_benchmark()
    
    # Run baseline
    print("Running baseline analysis...")
    sb.run_baseline()
    sb.run_ProjectionExperiment('TrainingStats', postprocess_linear, 'linear')
    sb.run_ProjectionExperiment('TrainingResults', postprocess_linear, 'linear')
    
    # Set up search experiments
    recipes, _ = sb.baseline.evaluate()
    recipes.reset_index(inplace=True)
    resource_values = list(recipes['resource'])
    budgets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Time budgets for QAOA
    budgets = np.unique([take_closest(resource_values, b) for b in budgets])
    
    key = names.param2filename({'Key': 'PerfRatio', 'Metric': 'median'}, '')
    
    rsParams = random_exploration.RandomSearchParameters(
        budgets=budgets,
        parameter_names=parameter_names,
        key=key
    )
    
    sb.run_RandomSearchExperiment(rsParams, postprocess=postprocess_random, postprocess_name='custom')
    
    print("QAOA stochastic benchmark analysis complete!")
    return sb


if __name__ == '__main__':
    main()
