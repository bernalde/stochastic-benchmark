import pandas as pd
import numpy as np

from stats import Quantile


def test_quantile_binomial_no_error():
    q = Quantile(q=50, nboots=10, style="binomial")
    base = pd.DataFrame({"x": np.arange(1, 101)})
    lower = base - 0.1
    upper = base + 0.1
    cent, lb, ub = q.ConfInts(base, lower, upper)
    assert isinstance(lb, (float, int, np.floating, np.integer))
    assert isinstance(ub, (float, int, np.floating, np.integer))
    assert lb < cent["x"] < ub
