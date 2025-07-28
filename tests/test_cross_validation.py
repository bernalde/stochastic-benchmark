import pytest
from scipy.special import erfinv
import pandas as pd
import numpy as np
import os

# compatibility for pandas >=2
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cross_validation import baseline_evaluate, all_ci, propagate_ci
import stats

class TestBaselineEvaluate:
    def test_basic(self):
        df = pd.DataFrame({
            'resource': [1, 1, 2, 2],
            'param': [10, 12, 20, 22],
            'metric': [0.1, 0.2, 0.3, 0.4]
        })
        params, eval_df = baseline_evaluate(df, ['param'], 'metric')
        assert list(params.columns) == ['resource', 'param']
        assert params.loc[0, 'param'] == 11
        assert eval_df.loc[1, 'metric'] == 0.35

class TestAllCI:
    def test_compute(self):
        df = pd.DataFrame({'resource': [1,2,3], 'p':[1.0,2.0,3.0]})
        result = all_ci(df, 'p', confidence_level=68)
        mean = df['p'].mean()
        std = np.nanstd(df["p"].values)
        fact = erfinv(68/100.0)*np.sqrt(2.0)
        np.testing.assert_allclose(result['mean'], [mean])
        np.testing.assert_allclose(result['CI_l'], [mean - fact*std])
        np.testing.assert_allclose(result['CI_u'], [mean + fact*std])

class TestPropagateCI:
    def test_mean(self):
        df = pd.DataFrame({
            'response':[10,20,30],
            'response_lower':[9,19,29],
            'response_upper':[11,21,31]
        })
        res = propagate_ci(df, 'mean')
        sm = stats.Mean()
        cent, cl, cu = sm.ConfInts(df['response'], df['response_lower'], df['response_upper'])
        np.testing.assert_allclose(res['mean'], [cent])
        np.testing.assert_allclose(res['CI_l'], [cl])
        np.testing.assert_allclose(res['CI_u'], [cu])

    def test_invalid_measure(self):
        df = pd.DataFrame({'response':[1], 'response_lower':[0], 'response_upper':[2]})
        with pytest.raises(ValueError):
            propagate_ci(df, 'invalid')

