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
    return InterpolationParameters(resource_fcn=lambda df: np.ones(len(df)))

# -------------------------------
# Test Interpolate
# -------------------------------
def test_interpolate_runs(sample_df, interp_params):
    df = sample_df.copy()
    df_interp = Interpolate(df, interp_params, group_on='instance')
    
    assert isinstance(df_interp, pd.DataFrame)
    for col in ['instance', 'resource', 'response']:
        assert col in df_interp.columns
    if isinstance(df_interp.index, pd.MultiIndex):
        df_interp = df_interp.reset_index(drop=True)
    assert len(df_interp) >= len(df)

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
    if isinstance(df_interp.index, pd.MultiIndex):
        df_interp = df_interp.reset_index(drop=True)
    assert len(df_interp) >= len(sample_df) * len(df_list)
