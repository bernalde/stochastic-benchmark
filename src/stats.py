from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy.stats
import os
from tqdm import tqdm
from typing import List, Tuple, Union, DefaultDict
import warnings

import names

EPSILON = 1e-10

tqdm.pandas()

mean_median_method = "simple_average"  # or 'inverse_variance_weighing'


@dataclass
class StatsParameters:
    """
    Parameters for stats computation

    Attributes
    ----------
    metrics : list
        List of metrics to compute
    lower_bounds : DefaultDict[str, dict]
        Dictionary of lower bounds for each metric
    upper_bounds : DefaultDict[str, dict]
        Dictionary of upper bounds for each metric
    stats_measures : list
        List of statistics to compute

    Methods
    -------
    __post_init__()
        Sets default values for lower and upper bounds
    """

    metrics: list = field(
        default_factory=lambda: [
            "MinEnergy",
            "RTT",
            "PerfRatio",
            "SuccProb",
            "MeanTime",
            "InvPerfRatio",
        ]
    )
    lower_bounds: DefaultDict[str, dict] = field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    upper_bounds: DefaultDict[str, dict] = field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    # PyLance cries as follows: Subscript for class "defaultdict" will generate runtime exception; enclose type annotation in quotes
    # We need to also include the mean computation here
    stats_measures: List = field(default_factory=lambda: [Median()])

    def __post_init__(self):
        self.lower_bounds["SuccProb"] = 0.0
        self.lower_bounds["MeanTime"] = 0.0
        self.lower_bounds["InvPerfRatio"] = EPSILON

        self.upper_bounds["SuccProb"] = 1.0
        self.upper_bounds["PerfRatio"] = 1.0


class StatsMeasure:
    """
    General class for stats measures with a center and confidence intervals

    Methods
    -------
    __init__()
        Initializes the name of the stats measure
    __call__()
        Computes the center of the stats measure
    center()
        Computes the center of the stats measure
    ConfIntlower()
        Computes the lower confidence interval of the stats measure
    ConfIntupper()
        Computes the upper confidence interval of the stats measure
    ConfInts()
        Computes the confidence intervals of the stats measure
    """

    def __init__(self):
        self.name = None

    def __call__(self, base, lower, upper):
        raise NotImplementedError(
            "Call should be overriden by a subclass of StatsMeasure"
        )

    def center(self, base, lower, upper):
        raise NotImplementedError(
            "Center should be overriden by a subclass of StatsMeasure"
        )

    def ConfIntlower(self, base, lower, upper):
        raise NotImplementedError(
            "ConfIntlower should be overriden by a subclass of StatsMeasure"
        )

    def ConfIntupper(self, base, lower, upper):
        raise NotImplementedError(
            "ConfIntupper should be overriden by a subclass of StatsMeasure"
        )

    def ConfInts(self, base, lower, upper):
        raise NotImplementedError(
            "ConfInts should be overriden by a subclass of StatsMeasure"
        )


class Mean(StatsMeasure):
    """
    Mean stat measure

    Methods
    -------
    __init__()
        Initializes the name of the stats measure Mean
    __call__()
        Computes the mean of the stats measure Mean
    compute_weights()
        Computes the weights for the mean computation
    center()
        Computes the center of the stats measure Mean
    ConfInts()
        Computes the confidence intervals of the stats measure Mean
    """

    def __init__(self):
        self.name = "mean"

    def __call__(self, base: pd.DataFrame):
        return base.mean()

    def compute_weights(self, upper, lower):
        if mean_median_method == "simple_average":
            self.weights = np.array([1 for _ in range(len(upper))])
        elif mean_median_method == "inverse_variance_weighing":
            self.weights = np.array(4 / (upper - lower) ** 2)
        else:
            raise ValueError(
                "mean_method can only be 'simple_average' or 'inverse_variance_weighing'"
            )

    def center(self, base: pd.Series, lower: pd.Series, upper: pd.Series):
        if not hasattr(self, "weights"):
            self.compute_weights(upper, lower)

        return sum(self.weights * base) / sum(self.weights)

    def ConfInts(self, base: pd.Series, lower: pd.Series, upper: pd.Series):
        """Compute confidence intervals
        If mean_median_method=="inverse_variance_weighing", then use inverse variance weighing to propagate mean and variance
        Else, if mean_median_method=="simple_average", do the former, but with weights set to 1.
        See the Context Section of https://en.wikipedia.org/wiki/Inverse-variance_weighting
        """
        self.compute_weights(upper, lower)
        cent = self.center(base, lower, upper)

        deviations = (
            upper - lower
        ) / 2  # Series object with values of σ * factor. For example, for 68% CIs, this will just be a series object with values of σ for different instances
        combined_deviation = np.sqrt(
            sum(self.weights**2 * deviations**2) / sum(self.weights) ** 2
        )
        CIlower = cent - combined_deviation
        CIupper = cent + combined_deviation
        return cent, CIlower, CIupper


class Median(StatsMeasure):
    """
    Median stat measure

    Methods
    -------
    __init__()
        Initializes the name of the stats measure Median
    __call__()
        Computes the median of the stats measure Median
    ConfInts()
        Computes the confidence intervals of the stats measure Median
    """

    def __init__(self):
        self.name = "median"

    def __call__(self, base: pd.DataFrame):
        return base.median()
    
    def center(self, base: pd.Series):
        return base.median()

    def ConfInts(self, base: pd.Series, lower: pd.Series, upper: pd.Series):
        """
        The center value and confidence intervals for the median are obtained by modifying the corresponding values obtained from methods in the Mean class.
        Specifically:
            deviation for Median = √(π/2) deviation for Mean
        Reference: https://mathworld.wolfram.com/StatisticalMedian.html
        """
        mean_sm = Mean()
        cent_mean, _, CIupper_mean = mean_sm.ConfInts(base, lower, upper)
        deviation_for_mean = CIupper_mean - cent_mean
        deviation_for_median = deviation_for_mean * np.sqrt(np.pi / 2)
        cent = self.center(base)
        CIlower = cent - deviation_for_median
        CIupper = cent + deviation_for_median
        return cent, CIlower, CIupper


class Percentile(StatsMeasure):
    """
    Percentile stat measure

    Methods
    -------
    __init__()
        Initializes the name of the stats measure Percentile
    __call__()
        Computes the percentile of the stats measure Percentile
    center()
        Computes the center of the stats measure Percentile
    ConfInts()
        Computes the confidence intervals of the stats measure Percentile
    """

    def __init__(self, q, nboots, confidence_level: float = 68):
        self.q = q
        self.name = "{}Percentile".format(q)
        self.nboots = int(nboots)
        self.confidence_level = confidence_level

    def __call__(self, base: pd.DataFrame):
        return base.quantile(self.q / 100.0)

    def center(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        """
        Find the center of the percentile by taking the mean of the upper and lower bounds

        Parameters
        ----------
        base: pd.DataFrame
            Dataframe with the metric to be analyzed
        lower: pd.DataFrame
            Dataframe with the lower bounds of the metric
        upper: pd.DataFrame
            Dataframe with the upper bounds of the metric

        Returns
        -------
        pd.DataFrame
            Dataframe with the center of the percentile
        """
        return base.quantile(self.q / 100.0)

    def ConfInts(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        """
        Find the confidence intervals of the percentile by bootstrapping

        Parameters
        ----------
        base: pd.DataFrame
            Dataframe with the metric to be analyzed
        lower: pd.DataFrame
            Dataframe with the lower bounds of the metric
        upper: pd.DataFrame
            Dataframe with the upper bounds of the metric

        Returns
        -------
        pd.DataFrame
            Dataframe with the confidence intervals of the percentile
        
        """
        cent = base.quantile(self.q / 100.0)
        boot_dist = []
        for i in range(self.nboots):
            resampler = np.random.randint(
                0, len(base), len(base), dtype=np.intp
            )  # intp is indexing dtype
            # Check the following, in original code sample_ci_lower = x[key_string + '_conf_interval_upper'].values.take(resampler, axis=0)
            # but that doesn't make sense
            sample = base.values.take(resampler, axis=0)
            sample_ci_upper = upper.values.take(resampler, axis=0)
            sample_ci_lower = lower.values.take(resampler, axis=0)
            sample_std = (sample_ci_upper - sample_ci_lower) / 2.0
            sample_error = np.random.normal(0, sample_std, len(sample))
            boot_dist.append(pd.Series(sample + sample_error).quantile(self.q / 100.0))
        p = 0.50 - self.confidence_level / (2 * 100.0), 0.50 + self.confidence_level / (
            2.0 * 100.0
        )
        (CIlower, CIupper) = pd.Series(boot_dist).quantile(p)
        return cent, CIlower, CIupper


class Quantile(StatsMeasure):
    """
    Percentile stat measure modified with proper confidence intervals
    and options of different intervals w/o need to boostrap for standard error

    Methods
    -------
    __init__()
        Initializes the name of the stats measure Quantile
    __call__()
        Computes the quantile of the stats measure Quantile
    center()
        Computes the center of the stats measure Quantile
    ConfInts()
        Computes the confidence intervals of the stats measure Quantile
    """

    def __init__(self, q, nboots, confidence_level: float = 95, style="MJ"):
        self.q = q
        self.name = "{}Quantile".format(q)
        self.nboots = int(nboots)
        self.confidence_level = confidence_level
        self.style = style
        self.alpha = 1 - (confidence_level / 100.0)

    def __call__(self, base: pd.DataFrame):
        return base.quantile(self.q / 100.0)

    def center(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        return base.quantile(self.q / 100.0)

    def ConfInts(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        import scipy.stats

        cent = base.quantile(self.q / 100.0)

        qt = self.q / 100.0
        n = len(base)

        df = np.array(sorted(base.values))

        if self.style == "MJ":
            a = qt * (n + 1)
            b = (1 - qt) * (n + 1)
            cdfs = scipy.stats.beta.cdf(np.array([i / n for i in range(n + 1)]), a, b)

            W = cdfs[1:] - cdfs[:-1]
            c1 = np.sum(W * df)
            c2 = np.sum(W * (df**2))
            se = np.sqrt(c2 - (c1**2))
            est = c1
            margin = se * scipy.stats.t.ppf(q=(1 - self.alpha / 2), df=n - 1)
            ub = est + margin
            lb = est - margin

            return cent, lb, ub

        elif self.style == "HD":
            h = scipy.stats.mstats.hdquantiles(df, prob=qt, var=True)
            est = h.data[0][0]
            se = np.sqrt(h.data[1][0])
            distval = scipy.stats.t.ppf(q=(1 - self.alpha / 2), df=n - 1)
            margin = distval * se

            ub = est + margin
            lb = est - margin

            return cent, lb, ub

        elif self.style == "kernel":
            q25 = np.quantile(df, 0.25)
            q75 = np.quantile(df, 0.75)
            q_int = np.quantile(df, qt)
            h = 1.2 * (q75 - q25) / (n**0.2)
            nint = len(df[(df > (q_int - h)) & (df < (q_int + h))])
            fhat = nint / (2 * h)
            se = 1 / (2 * np.sqrt(n) * fhat)
            distval = scipy.stats.norm.ppf(1 - self.alpha / 2)

            ub = q_int + distval * se
            lb = q_int - distval * se

            return cent, lb, ub

        elif self.style == "binomial":
            search = 2
            search_range = np.arange(-search, search + 1, 1)
            u = (
                scipy.stats.binom.ppf(q=1 - self.alpha / 2, n=n, p=qt)
                + search_range
                + 1
            )
            l = scipy.stats.binom.ppf(q=self.alpha / 2, n=n, p=qt) + search_range
            u[u > n] = np.inf
            l[l < 0] = -np.inf

            a = scipy.stats.binom.cdf(u - 1, n, qt)
            b = scipy.stats.binom.cdf(l - 1, n, qt)

            coverage = (a[:, None] - b).T

            if np.max(coverage) < 1 - self.alpha:
                i = np.unravel_index(coverage.argmax(), coverage.shape)
            else:
                minval = min(coverage[coverage >= 1 - self.alpha])
                i = np.argwhere(coverage == minval)[-1][0]
                j = len(search_range) * i
                u = int(np.repeat(u, len(search_range))[j])
                l = int(np.repeat(l, len(search_range))[j])

            ub, lb = df[u], df[l]
            return cent, lb[0], ub[0]

        elif self.style == "normal_binomial":
            distval = scipy.stats.norm.ppf(1 - self.alpha / 2)
            l = qt - distval * np.sqrt((qt * (1 - qt)) / n)
            u = qt + distval * np.sqrt((qt * (1 - qt)) / n)
            ub = np.quantile(df, u)
            lb = np.quantile(df, l)

            return cent, lb, ub

        else:
            return "Type of interval not found!"


def StatsSingle(df_single: pd.DataFrame, stat_params: StatsParameters):
    """
    Compute statistics for a single column

    Parameters
    ----------
    df_single: pd.DataFrame
        Dataframe with the metric to be analyzed
    stat_params: StatsParameters
        Parameters for the statistics

    Returns
    -------
    pd.DataFrame
        Dataframe with the statistics
    """
    if len(df_single) == 1:
        return pd.DataFrame()
    df_dict = {}
    for sm in stat_params.stats_measures:
        for key in stat_params.metrics:
            pre_base = names.param2filename({"Key": key}, "")
            pre_CIlower = names.param2filename({"Key": key, "ConfInt": "lower"}, "")
            pre_CIupper = names.param2filename({"Key": key, "ConfInt": "upper"}, "")

            metric_basename = names.param2filename({"Key": key, "Metric": sm.name}, "")
            metric_CIlower_name = names.param2filename(
                {"Key": key, "Metric": sm.name, "ConfInt": "lower"}, ""
            )
            metric_CIupper_name = names.param2filename(
                {"Key": key, "Metric": sm.name, "ConfInt": "upper"}, ""
            )

            base, CIlower, CIupper = sm.ConfInts(
                df_single[pre_base], df_single[pre_CIlower], df_single[pre_CIupper]
            )

            df_dict[metric_basename] = [base]
            df_dict[metric_CIlower_name] = [CIlower]
            df_dict[metric_CIupper_name] = [CIupper]
            df_dict["count"] = len(df_single[pre_base])

    df_stats_single = pd.DataFrame.from_dict(df_dict)
    return df_stats_single


def applyBounds(df: pd.DataFrame, stat_params: StatsParameters):
    """
    Apply the bounds to the dataframe.
    The bounds are applied to the corresponding confidence intervals to be clipped.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the metric to be analyzed
    stat_params: StatsParameters
        Parameters for the statistics

    Returns
    -------
    pd.DataFrame
        Dataframe with the statistics
    """
    for sm in stat_params.stats_measures:
        for key, value in stat_params.lower_bounds.items():
            center_name = names.param2filename(
                {"Key": key, "Metric": sm.name}, ""
            )
            if center_name in df.columns:
                df_copy = df.loc[:, (center_name)].copy()
                df_copy.clip(lower=value, inplace=True)
                df.loc[:, (center_name)] = df_copy
            lower_name = names.param2filename(
                {"Key": key, "Metric": sm.name, "ConfInt": "lower"}, ""
            )
            if lower_name in df.columns:
                df_copy = df.loc[:, (lower_name)].copy()
                df_copy.clip(lower=value, inplace=True)
                df.loc[:, (lower_name)] = df_copy

        for key, value in stat_params.upper_bounds.items():
            center_name = names.param2filename(
                {"Key": key, "Metric": sm.name}, ""
            )
            if center_name in df.columns:
                df_copy = df.loc[:, (center_name)].copy()
                df_copy.clip(upper=value, inplace=True)
                df.loc[:, (center_name)] = df_copy
            upper_name = names.param2filename(
                {"Key": key, "Metric": sm.name, "ConfInt": "upper"}, ""
            )
            if upper_name in df.columns:
                df_copy = df.loc[:, (upper_name)].copy()
                df_copy.clip(upper=value, inplace=True)
                df.loc[:, (upper_name)] = df_copy
    return


def Stats(df: pd.DataFrame, stats_params: StatsParameters, group_on):
    """
    Compute statistics for a dataframe

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the metric to be analyzed
    stats_params: StatsParameters
        Parameters for the statistics
    group_on: list
        List of columns to group on

    Returns
    -------
    pd.DataFrame
        Dataframe with the statistics
    """

    def dfSS(df):
        return StatsSingle(df, stats_params)
    # df_stats = df.groupby(group_on).progress_apply(dfSS, include_groups=False).reset_index()
    df_stats = df.groupby(group_on).progress_apply(dfSS).reset_index()
    df_stats.drop("level_{}".format(len(group_on)), axis=1, inplace=True)
    applyBounds(df_stats, stats_params)

    return df_stats
