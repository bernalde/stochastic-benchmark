import pandas as pd
import numpy as np
import pytest
from training import split_train_test, virtual_best, evaluate

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'instance': ['a', 'b', 'a', 'b'],
        'resource': [1, 1, 2, 2],
        'param1': [0.1, 0.2, 0.3, 0.4],
        'response': [10, 20, 30, 40]
    })

def test_split_train_test_runs(sample_df):
    df = sample_df.copy()
    df_split = split_train_test(df.copy(), split_on=['instance'], ptrain=0.75)
    assert 'train' in df_split.columns
    assert set(df_split['train'].unique()).issubset({0,1})
    assert len(df_split) == len(df)

def test_virtual_best_runs(sample_df):
    df = sample_df.copy()
    param_names = ['param1']
    response_col = 'response'
    vb_df = virtual_best(
        df,
        parameter_names=param_names,
        response_col=response_col,
        response_dir=1,
        groupby=['instance'],
        resource_col='resource',
        additional_cols=[],
        smooth=False
    )
    assert isinstance(vb_df, pd.DataFrame)
    expected_cols = ['instance', 'resource'] + param_names
    for col in expected_cols:
        assert col in vb_df.columns
    assert len(vb_df) == df.groupby(['instance', 'resource']).ngroups

def test_evaluate_runs(sample_df):
    df = sample_df.copy()
    recipes = pd.DataFrame({'resource': [1,2], 'param1': [0.15, 0.35]})
    
    def distance_fcn(df1, df2, resource):
        return pd.Series(np.zeros(len(df1)), index=df1.index)
    
    df_eval = evaluate(df, recipes, distance_fcn, parameter_names=['param1'], resource_col='resource', group_on=[])
    assert isinstance(df_eval, pd.DataFrame)
    assert 'param1' in df_eval.columns
    
    df_eval_grouped = evaluate(df, recipes, distance_fcn, parameter_names=['param1'], resource_col='resource', group_on=['instance'])
    assert isinstance(df_eval_grouped, pd.DataFrame)
    assert 'param1' in df_eval_grouped.columns
    assert len(df_eval_grouped) == len(df)
