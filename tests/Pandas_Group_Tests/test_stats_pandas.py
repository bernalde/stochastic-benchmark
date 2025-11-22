import pandas as pd
import pytest
from stats import Stats, StatsParameters, Median

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'instance': ['x','y','x','y'],
        'response': [1,2,3,4]  # lowercase 'response'
    })

@pytest.fixture
def stats_params():
    return StatsParameters(metrics=['response'], stats_measures=[Median()])

def test_stats_runs(sample_df, stats_params):
    df = sample_df.copy()
    df_stats = Stats(df, stats_params, group_on=['instance'])
    
    assert isinstance(df_stats, pd.DataFrame)
    expected_cols = ['response', 'response_lower', 'response_upper', 'count']
    for col in expected_cols:
        assert col in df_stats.columns
    if isinstance(df_stats.index, pd.MultiIndex):
        df_stats = df_stats.reset_index(drop=True)
    assert len(df_stats) == df['instance'].nunique()
