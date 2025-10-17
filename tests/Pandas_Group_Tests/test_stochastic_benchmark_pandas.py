import pandas as pd
import pytest
from stochastic_benchmark import stochastic_benchmark, VirtualBestBaseline, ProjectionExperiment
from stats import StatsParameters, Median

@pytest.fixture
def stochastic_benchmark_instance():
    sb = stochastic_benchmark(
        interp_results=pd.DataFrame({
            'instance': [1, 1, 2, 2],
            'resource': [1, 2, 1, 2],
            'response': [0.5, 0.6, 0.7, 0.8],
            'param1': [10, 20, 30, 40],
            'train': [0, 1, 0, 1]
        }),
        parameter_names=['param1'],
        instance_cols=['instance'],
        response_key='response',
        smooth=False
    )
    sb.stat_params = StatsParameters(metrics=['response'], stats_measures=[Median()])
    return sb

# -------------------------------
# stochastic_benchmark tests
# -------------------------------
def test_groupby_apply_returns_expected_shape(stochastic_benchmark_instance):
    df = stochastic_benchmark_instance.interp_results
    grouped = df.groupby(['instance', 'resource']).apply(lambda x: x.mean())
    assert isinstance(grouped.index, pd.MultiIndex)
    assert 'response' in grouped.columns

# -------------------------------
# VirtualBestBaseline tests
# -------------------------------
def test_virtual_best_baseline_populate(stochastic_benchmark_instance):
    sb = stochastic_benchmark_instance
    sb.here = type('obj', (), {'checkpoints': '/tmp'})()  # dummy path
    vb = VirtualBestBaseline(sb)
    assert hasattr(vb, 'rec_params')
    assert all(col in vb.rec_params.columns for col in ['resource', 'param1'])

def test_virtual_best_baseline_evaluate(stochastic_benchmark_instance):
    vb = VirtualBestBaseline(stochastic_benchmark_instance)
    params_df, eval_df = vb.evaluate()
    assert 'resource' in params_df.columns
    assert 'response' in eval_df.columns
    assert 'response_lower' in eval_df.columns
    assert 'response_upper' in eval_df.columns

def test_virtual_best_baseline_recalibrate(stochastic_benchmark_instance):
    vb = VirtualBestBaseline(stochastic_benchmark_instance)
    new_df = vb.rec_params.sample(2)
    vb.recalibrate(new_df)
    expected_cols = ['resource'] + stochastic_benchmark_instance.parameter_names
    for col in expected_cols:
        assert col in vb.rec_params.columns

# -------------------------------
# ProjectionExperiment tests
# -------------------------------
def test_projection_experiment_evaluate(stochastic_benchmark_instance):
    pe = ProjectionExperiment(stochastic_benchmark_instance, "TrainingStats")
    params_df, eval_df = pe.evaluate()
    assert 'resource' in params_df.columns
    assert 'response' in eval_df.columns
    assert 'response_lower' in eval_df.columns
    assert 'response_upper' in eval_df.columns

def test_projection_experiment_evaluate_monotone(stochastic_benchmark_instance):
    pe = ProjectionExperiment(stochastic_benchmark_instance, "TrainingStats")
    params_df, eval_df = pe.evaluate_monotone()
    diffs = eval_df['response'].diff().fillna(0)
    assert all(diffs >= 0) or all(diffs <= 0)

def test_projection_experiment_recipe_path(stochastic_benchmark_instance):
    pe = ProjectionExperiment(stochastic_benchmark_instance, "TrainingStats")
    pe.set_rec_path()
    assert pe.rec_path.endswith(".pkl")
