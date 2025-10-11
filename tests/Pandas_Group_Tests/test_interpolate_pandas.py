import pandas as pd
import pytest
from interpolate import Interpolate, Interpolate_reduce_mem, InterpolationParameters
import numpy as np

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'instance': ['x','x','y','y'],
        'resource': [1,2,1,2],  # ensure >1 resource per instance
        'response': [1.0, 2.0, 3.0, 4.0]  # lowercase 'response'
    })

@pytest.fixture
def interp_params():
    return InterpolationParameters(
        resource_fcn=lambda df: df['resource'], 
        resource_value_type='manual',
        resource_values=[1.0, 1.5, 2.0]
    )

# -------------------------------
# Test Interpolate
# -------------------------------
def test_interpolate_runs(sample_df, interp_params):
    df = sample_df.copy()
    df_interp = Interpolate(df, interp_params, group_on='instance').reset_index()
    
    assert isinstance(df_interp, pd.DataFrame)
    for col in ['instance', 'resource', 'response']:
        assert col in df_interp.columns
    # Each of the 2 instances should have 3 interpolated points
    assert len(df_interp) == 2 * len(interp_params.resource_values)

# -------------------------------
# Test Interpolate_reduce_mem
# -------------------------------
def test_interpolate_reduce_mem_runs(tmp_path, sample_df, interp_params):
    df_list = []
    for i in range(2):
        file_path = tmp_path / f"df_{i}.pkl"
        sample_df.to_pickle(file_path)
        df_list.append(str(file_path))
    
    df_interp = Interpolate_reduce_mem(df_list, interp_params, group_on='instance')
    
    assert isinstance(df_interp, pd.DataFrame)
    for col in ['instance', 'resource', 'response']:
        assert col in df_interp.columns
    # After the fix in Interpolate_reduce_mem, it should not have a multi-index
    # Each of the 2 instances from each of the 2 files should have 3 points
    assert len(df_interp) == len(df_list) * 2 * len(interp_params.resource_values)
