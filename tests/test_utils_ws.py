import pandas as pd
import numpy as np

# Ensure compatibility with pandas >=2
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils_ws import gen_log_space, take_closest, interp

class TestGenLogSpace:
    def test_basic_generation(self):
        vals = gen_log_space(1, 100, 5)
        assert len(vals) == 5
        assert vals[0] == 1
        assert vals[-1] <= 100
        assert np.all(np.diff(vals) > 0)

class TestTakeClosest:
    def test_various_cases(self):
        arr = [1, 3, 5, 7]
        assert take_closest(arr, 6) == 5
        assert take_closest(arr, 3) == 3
        assert take_closest(arr, 0) == 1
        assert take_closest(arr, 10) == 7

class TestInterp:
    def test_numeric_and_object_columns(self):
        df = pd.DataFrame({'num': [1, 2, 3], 'cat': ['x', 'y', 'z']}, index=[0, 5, 10])
        result = interp(df, [0, 2, 5, 7, 10])
        assert list(result.index) == [0, 2, 5, 7, 10]
        np.testing.assert_almost_equal(result['num'].tolist(), [1, 1.4, 2, 2.4, 3])
        assert result.loc[0, 'cat'] == 'x'
        assert result.loc[5, 'cat'] == 'y'
        assert pd.isna(result.loc[2, 'cat']) and pd.isna(result.loc[7, 'cat'])
=======
import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'src')

# Compatibility for pandas 2.x
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

from utils_ws import gen_log_space, interp, take_closest


class TestGenLogSpace:
    def test_basic_generation(self):
        result = gen_log_space(1, 10, 3)
        assert isinstance(result, np.ndarray)
        assert list(result) == [1, 3, 9]
        assert len(result) == 3


class TestTakeClosest:
    def test_take_closest_middle(self):
        lst = [1, 3, 5, 7]
        value = 4
        assert take_closest(lst, value) == 3

    def test_take_closest_edges(self):
        lst = [10, 20, 30]
        assert take_closest(lst, 5) == 10
        assert take_closest(lst, 35) == 30


class TestInterp:
    def test_interp_numeric(self):
        df = pd.DataFrame({'a': [1, 3], 'b': [10, 20]}, index=[0, 1])
        new_idx = [0, 0.5, 1]
        out = interp(df, new_idx)
        assert list(out.index) == new_idx
        np.testing.assert_array_almost_equal(out['a'].values, [1, 2, 3])
        np.testing.assert_array_almost_equal(out['b'].values, [10, 15, 20])


