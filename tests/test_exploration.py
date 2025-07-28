import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'src')

if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

from random_exploration import RandomSearchParameters, prepare_search as rs_prepare, summarize_experiments as rs_summary
from sequential_exploration import SequentialSearchParameters, prepare_search as ss_prepare, summarize_experiments as ss_summary
import stats


class TestRandomExploration:
    def test_prepare_search(self):
        df = pd.DataFrame({'resource':[10,20,30]})
        params = RandomSearchParameters(taus=[5,15,25])
        rs_prepare(df, params)
        assert list(params.taus) == [10,20]

    def test_summarize_experiments(self):
        df = pd.DataFrame({
            'TotalBudget':[100,100],
            'ExplorationBudget':[50,50],
            'tau':[10,10],
            'PerfRatio':[0.8,0.9]
        })
        params = RandomSearchParameters()
        params.key = 'PerfRatio'
        params.stat_measure = stats.Mean()
        params.optimization_dir = 1
        best, exp = rs_summary(df, params)
        assert len(best) == 1
        assert len(exp) >= 1


class TestSequentialExploration:
    def test_prepare_search(self):
        df = pd.DataFrame({'resource':[10,20,30]})
        params = SequentialSearchParameters(taus=[8,18,25])
        ss_prepare(df, params)
        assert list(params.taus) == [10,20]

    def test_summarize_experiments(self):
        df = pd.DataFrame({
            'TotalBudget':[100,100],
            'ExplorationBudget':[50,50],
            'tau':[10,10],
            'PerfRatio':[0.8,0.7]
        })
        params = SequentialSearchParameters()
        params.key = 'PerfRatio'
        params.stat_measure = stats.Mean()
        params.optimization_dir = -1
        best, exp = ss_summary(df, params)
        assert len(best) == 1
        assert len(exp) >= 1


