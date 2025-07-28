import pytest
import pandas as pd
import numpy as np
import sys
import os

# Monkey patch pandas DataFrame iteritems for pandas>=2
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

# Add src directory to path for imports
TESTS_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(TESTS_DIR, os.pardir, 'src'))
sys.path.insert(0, SRC_PATH)

import names
import random_exploration
import sequential_exploration


def create_test_df():
    """Create a simple dataframe for exploration tests."""
    key = names.param2filename({"Key": "PerfRatio"}, "")
    CIlower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
    CIupper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
    df = pd.DataFrame({
        'resource': [5, 10],
        'sweep': [0, 0],
        'replica': [0, 0],
        'order': [1, 2],
        key: [0.5, 0.8],
        CIlower: [0.4, 0.7],
        CIupper: [0.6, 0.9]
    })
    return df, key


class TestRandomSingle:
    """Tests for random_exploration.single_experiment."""

    def test_single_experiment_basic(self):
        df, key = create_test_df()
        params = random_exploration.RandomSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            Nexperiments=1,
            taus=[5],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        result = random_exploration.single_experiment(df, params, 10, 0.5, 5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 1 in result['exploit'].values
        assert result['CummResource'].iloc[-1] <= 10

    def test_single_experiment_insufficient_budget(self):
        df, key = create_test_df()
        params = random_exploration.RandomSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            Nexperiments=1,
            taus=[5],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        # explore_budget < tau should return None
        result = random_exploration.single_experiment(df, params, 4, 0.5, 5)
        assert result is None


class TestSequentialSingle:
    """Tests for sequential_exploration.SequentialExplorationSingle."""

    def test_sequential_single_basic(self):
        df, key = create_test_df()
        params = sequential_exploration.SequentialSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            taus=[5],
            order_cols=['order'],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        result = sequential_exploration.SequentialExplorationSingle(df, params, 0, 10, 0.5, 5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result['exploit'].sum() >= 1

    def test_sequential_single_edge_cases(self):
        df, key = create_test_df()
        params = sequential_exploration.SequentialSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            taus=[5],
            order_cols=['order'],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        # insufficient budget
        result = sequential_exploration.SequentialExplorationSingle(df, params, 0, 4, 0.5, 5)
        assert result is None
        # Data with NaNs should return None after dropna
        df_nan = df.copy()
        df_nan.loc[0, 'order'] = np.nan
        result = sequential_exploration.SequentialExplorationSingle(df_nan, params, 0, 10, 0.5, 5)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
