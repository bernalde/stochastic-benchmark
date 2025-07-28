import pandas as pd
import numpy as np
from unittest.mock import patch
import os

if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sequential_exploration import SequentialSearchParameters, prepare_search, summarize_experiments

class TestPrepareSearch:
    def test_calls_take_closest(self):
        df = pd.DataFrame({'resource':[1,5,10]})
        params = SequentialSearchParameters(taus=[2,7])
        with patch('sequential_exploration.take_closest', side_effect=lambda arr, v: v) as mock_tc:
            prepare_search(df, params)
            assert mock_tc.call_count == 2
            assert np.all(params.taus == np.unique([2,7]))

class TestSummarizeExperiments:
    def test_basic(self):
        data = pd.DataFrame({
            'TotalBudget':[10,10,20,20],
            'ExplorationBudget':[2,2,4,4],
            'tau':[1,1,1,1],
            'PerfRatio':[0.5,0.6,0.7,0.8]
        })
        params = SequentialSearchParameters()
        best, exp = summarize_experiments(data, params)
        assert len(best) == 2
        assert best.iloc[0]['TotalBudget'] == 20
        assert exp[exp["TotalBudget"]==20]["PerfRatio"].max() == 0.8

