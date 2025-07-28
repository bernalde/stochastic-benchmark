import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'src')

if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

from cross_validation import (
    baseline_evaluate,
    all_ci,
    propagate_ci,
    random_exp_evaluate,
    seq_search_evaluate,
)


class TestBaselineEvaluate:
    def test_basic(self):
        df = pd.DataFrame({
            'resource': [1, 1, 2, 2],
            'param1': [0.1, 0.2, 0.3, 0.4],
            'response': [10, 20, 30, 40]
        })
        params_df, eval_df = baseline_evaluate(df, ['param1'], 'response')
        assert list(params_df.columns) == ['resource', 'param1']
        assert list(eval_df.columns) == ['resource', 'response']
        assert len(params_df) == 2
        assert len(eval_df) == 2


class TestAllCI:
    def test_all_ci_basic(self):
        df = pd.DataFrame({'resource': [1,2,3], 'p': [1.0, 2.0, 3.0]})
        res = all_ci(df, 'p', confidence_level=68)
        assert 'mean' in res.columns
        assert 'CI_l' in res.columns
        assert 'CI_u' in res.columns
        assert res['mean'].iloc[0] == pytest.approx(2.0)


class TestPropagateCI:
    def test_propagate_ci_mean(self):
        df = pd.DataFrame({
            'response': [1.0, 2.0, 3.0],
            'response_lower': [0.5, 1.5, 2.5],
            'response_upper': [1.5, 2.5, 3.5]
        })
        res = propagate_ci(df, 'mean')
        assert {'mean','CI_l','CI_u'} <= set(res.columns)

    def test_invalid_measure(self):
        df = pd.DataFrame({'response':[1],'response_lower':[0],'response_upper':[2]})
        with pytest.raises(ValueError):
            propagate_ci(df, 'other')


class TestEvaluateFuncs:
    def test_random_exp_evaluate(self):
        df = pd.DataFrame({
            'TotalBudget':[10,10,20,20],
            'resource':[1,2,1,2],
            'param1':[0.1,0.2,0.3,0.4],
            'Resp':[10,20,30,40],
            'ConfInt=lower_Resp':[9,18,28,38],
            'ConfInt=upper_Resp':[11,22,32,42]
        })
        params_df, eval_df = random_exp_evaluate(df, ['param1'], 'Resp')
        assert 'resource' in params_df.columns
        assert 'response' in eval_df.columns
        assert len(params_df) == 2

    def test_seq_search_evaluate(self):
        df = pd.DataFrame({
            'TotalBudget':[10,20],
            'param1':[0.1,0.2],
            'resource':[5,5],
            'Resp':[11,12],
            'ConfInt=upper_Resp':[12,13],
            'ConfInt=lower_Resp':[10,11]
        })
        params_df, eval_df = seq_search_evaluate(df, ['param1'], 'Resp')
        assert 'resource' in params_df.columns
        assert len(eval_df) == 2


