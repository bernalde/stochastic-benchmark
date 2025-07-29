from collections import namedtuple
import copy
import glob
from math import floor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
from random import choice
import seaborn.objects as so
import seaborn as sns
import warnings
import logging


import bootstrap
import df_utils
import interpolate
from plotting import *
import random_exploration
import sequential_exploration
import stats
import success_metrics
import training

logger = logging.getLogger(__name__)
import names
import utils_ws

median = False


def default_bootstrap(
    nboots=1000,
    response_col=names.param2filename({"Key": "MinEnergy"}, ""),
    resource_col=names.param2filename({"Key": "MeanTime"}, ""),
):
    """
    Default bootstrapping parameters

    Parameters
    ----------
    nboots : int, optional
        Number of bootstrap iterations, by default 1000
    response_col : str, optional
        Column name of response, by default names.param2filename({"Key": "MinEnergy"}, "")
    resource_col : str, optional
        Column name of resource, by default names.param2filename({"Key": "MeanTime"}, "")

    Returns
    -------
    bsparams_iter : bootstrap.BSParams_iter
        Iterator of bootstrap parameters
    """
    shared_args = {
        "response_col": response_col,
        "resource_col": resource_col,
        "response_dir": -1,
        "confidence_level": 68,
        "random_value": 0.0,
    }

    metric_args = {}
    metric_args["Response"] = {"opt_sense": -1}
    metric_args["SuccessProb"] = {"gap": 1.0, "response_dir": -1}
    metric_args["RTT"] = {
        "fail_value": np.nan,
        "RTT_factor": 1.0,
        "gap": 1.0,
        "s": 0.99,
    }
    sms = [
        success_metrics.Response,
        success_metrics.PerfRatio,
        success_metrics.InvPerfRatio,
        success_metrics.SuccessProb,
        success_metrics.Resource,
        success_metrics.RTT,
    ]
    bsParams = bootstrap.BootstrapParameters(shared_args, metric_args, sms)

    bs_iter_class = bootstrap.BSParams_iter()
    bsparams_iter = bs_iter_class(bsParams, nboots)
    return bsparams_iter


def sweep_boots_resource(df):
    """
    Default resource computation - resource = sweeps * boots

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to compute resource for

    Returns
    -------
    pd.Series
        Series of resource values
    """
    return df["sweep"] * df["boots"]


class Experiment:
    """
    Base class for experiments

    Attributes
    ----------
    parent : Experiment
        Parent experiment
    name : str
        Name of experiment

    Methods
    -------
    __init__(parent, name)
        Initializes experiment
    evaluate()
        Evaluates experiment
    evaluate_monotone()
        Monotonizes the response and parameters from evaluate
    """

    def __init__(self):
        return

    def evaluate(self):
        raise NotImplementedError(
            "Evaluate should be overriden by a subclass of Experiment"
        )

    def evaluate_monotone(self):
        """
        Monotonizes the response and parameters from evaluate

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        res = self.evaluate()
        if len(res) == 2:
            params_df, eval_df = res
        elif len(res) == 3:
            params_df, eval_df, preproc_params = res
        joint = params_df.merge(eval_df, on="resource")
        joint = df_utils.monotone_df(joint, "resource", "response", 1)
        params_df = joint.loc[:, ["resource"] + self.parent.parameter_names]
        eval_df = joint.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]

        if len(res) == 2:
            return params_df, eval_df
        elif len(res) == 3:
            return params_df, eval_df, preproc_params


class ProjectionExperiment(Experiment):
    """
    Holds information needed for projection experiments.
    Used for evaluating performance of a recipe on the test set if the user cannot re-run experiments.
    Recipes can be post-processed by a user-defined function (e.g., smoothed fit) and queried for running
    evaluatation experiments.

    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    project_from : str
        'TrainingStats' or 'TrainingResults'
    recipe : pd.DataFrame
        Recommended parameters for each resource (can be postprocessed). This is not projected
    rec_params : pd.DataFrame
        Projected recommended parameters for each resource
    rec_path : str
        Path to recipe
    postprocess : function
        Function to postprocess recipe
    postprocess_name : str
        Name of postprocessing function

    Methods
    -------
    __init__(parent, project_from, postprocess=None, postprocess_name=None)
        Initializes projection experiment
    set_rec_path()
        Sets rec_path
    get_TrainingStats_recipe()
        Gets recipe from TrainingStats
    get_TrainingResults_recipe()
        Gets recipe from TrainingResults
    evaluate()
        Evaluates experiment
    """

    def __init__(self, parent, project_from, postprocess=None, postprocess_name=None):
        self.parent = parent
        self.name = "Projection from {}".format(project_from)
        self.project_from = project_from
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

    def populate(self):
        """
        Adds recipe depending on source. Currently only projection from the best recommended from the training stats or results are available.
        Any addition recipe specifications should be implemented here
        """
        # Set rec_path, i.e. the path where the recipe is/will be stored
        self.set_rec_path()

        # Prepare the recipes
        if self.project_from == "TrainingStats":
            self.get_TrainingStats_recipe()
        elif self.project_from == "TrainingResults":
            self.get_TrainingResults_recipe()
        else:
            raise NotImplementedError(
                "Projection from {} has not been implemented".format(self.project_from)
            )

        # Run the projections
        if os.path.exists(self.rec_path):
            self.rec_params = pd.read_pickle(self.rec_path)
        else:
            logger.info("Evaluating recommended parameters on testing results")
            testing_results = self.parent.interp_results[
                self.parent.interp_results["train"] == 0
            ].copy()
            self.rec_params = training.evaluate(
                testing_results,
                self.recipe,
                training.scaled_distance,
                parameter_names=self.parent.parameter_names,
                group_on=self.parent.instance_cols,
            )
            self.rec_params.to_pickle(self.rec_path)

    def get_TrainingResults_recipe(self):
        """
        If TrainingResults recipe is already stored in a pkl file, load it. Otherwise, create and store it by obtaining the best parameters from training_stats (and post_processing, if requested)
        """
        vb_train_path = os.path.join(
            self.parent.here.checkpoints, "VirtualBest_train.pkl"
        )

        if os.path.exists(vb_train_path):
            self.vb_train = pd.read_pickle(vb_train_path)
        else:
            response_col = names.param2filename({"Key": self.parent.response_key}, "")
            training_results = self.parent.interp_results[
                self.parent.interp_results["train"] == 1
            ].copy()
            self.vb_train = training.virtual_best(
                training_results,
                parameter_names=self.parent.parameter_names,
                response_col=response_col,
                response_dir=1,
                groupby=self.parent.instance_cols,
                resource_col="resource",
                smooth=self.parent.smooth,
            )
            self.vb_train.to_pickle(vb_train_path)

        self.recipe = training.best_recommended(
            self.vb_train.copy(),
            parameter_names=self.parent.parameter_names,
            resource_col="resource",
            additional_cols=["boots"],
        ).reset_index()

        if self.postprocess is not None:
            self.preproc_recipe = self.recipe.copy()
            self.recipe = self.postprocess(self.recipe)

    def get_TrainingStats_recipe(self):
        """
        If TrainingStats recipe is already stored in a pkl file, load it. Otherwise, create and store it by obtaining the best parameters from training_stats (and post_processing, if requested)
        """

        best_rec_train_path = os.path.join(
            self.parent.here.checkpoints, "BestRecommended_train.pkl"
        )
        if os.path.exists(best_rec_train_path):
            # If the recipe was already stored in a pkl file, simply load it
            self.recipe = pd.read_pickle(best_rec_train_path)
        else:
            # If not, create the recipe dataframe, and store it in a pkl file
            # Get the name of the response column
            response_col = names.param2filename(
                {
                    "Key": self.parent.response_key,
                    "Metric": self.parent.stat_params.stats_measures[0].name,
                },
                "",
            )

            # Obtain the recipe, before the postprocessing step
            self.recipe = training.best_parameters(
                self.parent.training_stats.copy(),
                parameter_names=self.parent.parameter_names,
                response_col=response_col,
                response_dir=1,
                resource_col="resource",
                additional_cols=["boots"],
                smooth=self.parent.smooth,
            )

            self.recipe.to_pickle(best_rec_train_path)

        if self.postprocess is not None:
            best_rec_train_path_post = os.path.join(
                self.parent.here.checkpoints,
                "BestRecommended_train_postprocess={}.pkl".format(
                    self.postprocess_name
                ),
            )
            # Copy the recipe to preproc_recipe before postprocessing
            self.preproc_recipe = self.recipe.copy()
            # Implement post-processing
            self.recipe = self.postprocess(self.recipe)
            self.recipe.to_pickle(best_rec_train_path_post)

    def evaluate(self, monotone=False):
        """
        Evaluates the recommended parameters on the testing results, and returns the recommended parameters and the responses

        Parameters
        ----------
        monotone : bool, optional
            If True, the recommended parameters are evaluated on the monotone testing results, by default False

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        # params_df = self.rec_params.loc[:, ['resource'] + self.parent.parameter_names].copy()

        # base = names.param2filename({'Key': self.parent.response_key}, '')
        # CIlower = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'lower'}, '')
        # CIupper = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'upper'}, '')
        # eval_df = self.rec_params.copy()
        # eval_df.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )

        # joint = self.rec_params.copy()
        # base = names.param2filename({'Key': self.parent.response_key}, '')
        # CIlower = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'lower'}, '')
        # CIupper = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'upper'}, '')
        # joint.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )
        # joint = joint.loc[:, ['resource'] + self.parent.parameter_names +
        # ['response', 'response_lower', 'response_upper'] + self.parent.instance_cols]

        # extrapolate_from = self.parent.interp_results.loc[self.parent.interp_results['train'] == 0].copy()
        # extrapolate_from.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )

        # def mono(df):
        #     # res = df_utils.monotone_df(joint, 'resource', 'response', 1,
        #     # extrapolate_from=extrapolate_from, match_on = self.parent.parameter_names + self.parent.instance_cols)
        #     res = df_utils.monotone_df(joint, 'resource', 'response', 1)
        #     return res

        # joint = joint.groupby(self.parent.instance_cols, include_groups=False).apply(mono)

        # params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        # eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        # params_df = params_df.groupby('resource').mean()
        # params_df.reset_index(inplace=True)

        # eval_df = eval_df.groupby('resource').median()
        # eval_df = eval_df.groupby('resource').median()
        # eval_df.reset_index(inplace=True)

        params_df = self.rec_params.loc[
            :, ["resource"] + self.parent.parameter_names
        ].copy()
        params_df = params_df.groupby("resource").mean()
        params_df.reset_index(inplace=True)

        base = names.param2filename({"Key": self.parent.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "upper"}, ""
        )
        eval_df = self.rec_params.copy()
        eval_df.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )
        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        if median:
            eval_df = eval_df.groupby("resource").median()
        else:
            eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df

    def evaluate_monotone(self):
        """
        Monotonizes the response and parameters from evaluate

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df, eval_df = self.evaluate()

        joint = params_df.merge(eval_df, on="resource")
        extrapolate_from = self.parent.testing_stats.copy()
        base = names.param2filename(
            {
                "Key": self.parent.response_key,
                "Metric": self.parent.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIlower = names.param2filename(
            {
                "Key": self.parent.response_key,
                "ConfInt": "lower",
                "Metric": self.parent.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIupper = names.param2filename(
            {
                "Key": self.parent.response_key,
                "ConfInt": "upper",
                "Metric": self.parent.stat_params.stats_measures[0].name,
            },
            "",
        )
        extrapolate_from.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        # joint = df_utils.monotone_df(joint, 'resource', 'response', 1,
        #     extrapolate_from=extrapolate_from, match_on = self.parent.parameter_names)
        joint = df_utils.monotone_df(joint, "resource", "response", 1)
        params_df = joint.loc[:, ["resource"] + self.parent.parameter_names]
        eval_df = joint.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        return params_df, eval_df

    def set_rec_path(self):
        """
        Define the path where the recipe is to be stored
        """
        if self.postprocess is not None:
            self.rec_path = os.path.join(
                self.parent.here.checkpoints,
                "Projection_from={}_postprocess={}.pkl".format(
                    self.project_from, self.postprocess_name
                ),
            )
        else:
            self.rec_path = os.path.join(
                self.parent.here.checkpoints,
                "Projection_from={}.pkl".format(self.project_from),
            )


class StaticRecommendationExperiment(Experiment):
    """
    Holds parameters for fixed recommendation experiments

    Attributes
    ----------
    parent : stochastic_benchmark
    name : str
        name for pretty printing
    rec_params : pd.DataFrame
        Recommended parameters for evaluation
    preproc_rec_params : pd.DataFrame
        Recommended parameters before processing

    Methods
    -------
    __init__(parent, init_from)
        Initialize the class
    list_runs()
        Returns a list of experiments evaluate
    evaluate()
        Returns the recommended parameters and responses
    evaluate_monotone()
        Monotonizes the response and parameters from evaluate
    set_rec_path()
        Define the path where the recipe is to be stored
    """

    def __init__(self, parent, init_from):
        self.parent = parent
        self.name = "FixedRecommendation"

        if type(init_from) == ProjectionExperiment:
            self.rec_params = init_from.recipe
            if init_from.postprocess is not None:
                self.preproc_rec_params = init_from.preproc_recipe.copy()

        elif type(init_from) == pd.DataFrame:
            self.rec_params = init_from
        else:
            warn_str = (
                "init_from type is not supported. No recommended parameters are set."
            )
            warnings.warn(warn_str)

    def list_runs(self):
        """
        Returns a list of experiments evaluate.

        Returns
        -------
        runs : list
            List of named tuples of parameters
        """
        parameter_names = "resource " + " ".join(self.parent.parameter_names)
        Parameter = namedtuple("Parameter", parameter_names)
        runs = []
        for _, row in self.rec_params.iterrows():
            runs.append(
                Parameter(
                    row["resource"], *[row[k] for k in self.parent.parameter_names]
                )
            )
        return runs

    def attach_runs(self, df, process=True):
        """
        Attaches reruns of experiment to the experiment object

        Parameters
        ----------
        df : pd.DataFrame or str
            Dataframe of responses
        process : bool
            Whether to process the dataframe

        Returns
        -------
        None
        """
        if type(df) == str:
            df = pd.read_pickle(df)
        if process:
            group_on = self.parent.instance_cols + ["resource"]
            self.eval_df = self.parent.evaluate_without_bootstrap(df, group_on)
        else:
            self.eval_df = df
        self.parent.baseline.recalibrate(self.eval_df)

    def evaluate(self):
        """
        Evaluates the recommended parameters

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.rec_params.loc[
            :, ["resource"] + self.parent.parameter_names
        ].copy()
        preproc_params = self.preproc_rec_params.loc[
            :, ["resource"] + self.parent.parameter_names
        ].copy()
        # params_df = params_df.groupby('resource').mean()
        # params_df.reset_index(inplace=True)

        base = names.param2filename({"Key": self.parent.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "upper"}, ""
        )
        eval_df = self.eval_df.copy()
        eval_df.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )
        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df, preproc_params


class RandomSearchExperiment(Experiment):
    """
    Holds parameters needed for random search experiment

    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    meta_params : pd.DataFrame
        Best metaparameters (Exploration budget and Tau)
    eval_train : pd.DataFrame
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set
    rsParams : dict
        Dictionary of parameters for random search
    postprocess : function
        Function to postprocess the results
    postprocess_name : str
        Name of the postprocessing function

    Methods
    -------
    __init__(parent, rsParams, postprocess=None, postprocess_name=None)
        Initialize the class
    populate()
        Populates meta_params, eval_train, eval_test
    """

    def __init__(self, parent, rsParams, postprocess=None, postprocess_name=None):
        self.parent = parent
        self.name = "RandomSearch"
        self.rsParams = rsParams
        self.meta_parameter_names = ["ExploreFrac", "tau"]
        self.resource = "TotalBudget"
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

    def populate(self):
        """
        Populates meta_params, eval_train, eval_test
        """
        meta_params_path = os.path.join(
            self.parent.here.checkpoints, "RandomSearch_meta_params.pkl"
        )
        eval_train_path = os.path.join(
            self.parent.here.checkpoints, "RandomSearch_evalTrain.pkl"
        )

        if self.postprocess is None:
            eval_test_path = os.path.join(
                self.parent.here.checkpoints, "RandomSearch_evalTest.pkl"
            )
        else:
            eval_test_path = os.path.join(
                self.parent.here.checkpoints,
                "RandomSearch_evalTest_postprocess={}.pkl".format(
                    self.postprocess_name
                ),
            )

        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
        else:
            self.meta_params, self.eval_train, _ = random_exploration.RandomExploration(
                self.parent.training_stats, self.rsParams
            )
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params["ExploreFrac"] = (
            self.meta_params["ExplorationBudget"] / self.meta_params["TotalBudget"]
        )

        if self.postprocess is not None:
            self.preproc_meta_params = self.meta_params.copy()
            self.meta_params = self.postprocess(self.meta_params)

        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            logger.info("\t Evaluating random search on test")
            self.eval_test = random_exploration.apply_allocations(
                self.parent.testing_stats.copy(), self.rsParams, self.meta_params
            )
            self.eval_test.to_pickle(eval_test_path)

    def evaluate(self):
        """
        Evaluates the random search

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.eval_test.loc[:, ["TotalBudget"] + self.parent.parameter_names]
        params_df = params_df.groupby("TotalBudget").mean()
        params_df.reset_index(inplace=True)
        params_df.rename(columns={"TotalBudget": "resource"}, inplace=True)

        base = names.param2filename(
            {
                "Key": self.parent.response_key,
                "Metric": self.parent.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIlower = names.param2filename(
            {
                "Key": self.parent.response_key,
                "ConfInt": "lower",
                "Metric": self.parent.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIupper = names.param2filename(
            {
                "Key": self.parent.response_key,
                "ConfInt": "upper",
                "Metric": self.parent.stat_params.stats_measures[0].name,
            },
            "",
        )
        eval_df = self.eval_test.copy()
        eval_df.drop("resource", axis=1, inplace=True)
        eval_df.rename(
            columns={
                "TotalBudget": "resource",
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        if median:
            eval_df = eval_df.groupby("resource").median()
        else:
            eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df


class SequentialSearchExperiment(Experiment):
    """
    Holds parameters needed for sequential search experiment

    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    meta_params : pd.DataFrame
        Best metaparameters (Exploration budget and Tau)
    eval_train : pd.DataFrame4
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set
    ssParams : SequentialSearchParameters
        Parameters for sequential search
    id_name : str
        Name of experiment
    postprocess : function
        Function to postprocess meta_params
    postprocess_name : str
        Name of postprocess function

    Methods
    -------
    populate()
        Populates meta_params, eval_train, eval_test
    evaluate()
        Evaluates the sequential search
    """

    def __init__(
        self, parent, ssParams, id_name=None, postprocess=None, postprocess_name=None
    ):
        self.parent = parent
        if id_name is None:
            self.name = "SequentialSearch"
        else:
            self.name = "SequentialSearch_{}".format(id_name)
        self.ssParams = ssParams
        self.id_name = id_name
        self.meta_parameter_names = ["ExploreFrac", "tau"]
        self.resource = "TotalBudget"
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

    def populate(self):
        if self.id_name is None:
            meta_params_path = os.path.join(
                self.parent.here.checkpoints, "SequentialSearch_meta_params.pkl"
            )
            eval_train_path = os.path.join(
                self.parent.here.checkpoints, "SequentialSearch_evalTrain.pkl"
            )
            if self.postprocess is None:
                eval_test_path = os.path.join(
                    self.parent.here.checkpoints, "SequentialSearch_evalTest.pkl"
                )
            else:
                eval_test_path = os.path.join(
                    self.parent.here.checkpoints,
                    "SequentialSearch_evalTest_postprocess={}.pkl".format(
                        self.postprocess_name
                    ),
                )
        else:
            meta_params_path = os.path.join(
                self.parent.here.checkpoints,
                "SequentialSearch_meta_params_id={}.pkl".format(self.id_name),
            )
            eval_train_path = os.path.join(
                self.parent.here.checkpoints,
                "SequentialSearch_evalTrain_id={}.pkl".format(self.id_name),
            )
            if self.postprocess is None:
                eval_test_path = os.path.join(
                    self.parent.here.checkpoints,
                    "SequentialSearch_evalTest_id={}.pkl".format(self.id_name),
                )
            else:
                eval_test_path = os.path.join(
                    self.parent.here.checkpoints,
                    "SequentialSearch_evalTest_id={}_postprocess={}.pkl".format(
                        self.id_name, self.postprocess_name
                    ),
                )

        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
            self.eval_train = pd.read_pickle(eval_train_path)
        else:
            training_results = self.parent.interp_results[
                self.parent.interp_results["train"] == 1
            ].copy()
            (
                self.meta_params,
                self.eval_train,
                _,
            ) = sequential_exploration.SequentialExploration(
                training_results, self.ssParams, group_on=self.parent.instance_cols
            )
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params["ExploreFrac"] = (
            self.meta_params["ExplorationBudget"] / self.meta_params["TotalBudget"]
        )
        if self.postprocess is not None:
            self.preproc_meta_params = self.meta_params.copy()
            self.meta_params = self.postprocess(self.meta_params)
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            # try:
            logger.info("\t Evaluating sequential search on test")
            testing_results = self.parent.interp_results[
                self.parent.interp_results["train"] == 0
            ].copy()
            self.eval_test = sequential_exploration.apply_allocations(
                testing_results,
                self.ssParams,
                self.meta_params,
                self.parent.instance_cols,
            )
            self.eval_test.to_pickle(eval_test_path)
            # except:
            #     print('Not enough test data for sequential search. Evaluating on train.')

    def evaluate(self):
        """
        Evaluates the sequential search

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        if hasattr(self, "eval_test"):
            params_df = self.eval_test.loc[
                :, ["TotalBudget"] + self.parent.parameter_names
            ]
            eval_df = self.eval_test.copy()
        else:
            params_df = self.eval_train.loc[
                :, ["TotalBudget"] + self.parent.parameter_names
            ]
            eval_df = self.eval_train.copy()

        for col in params_df.columns:
            if params_df[col].dtype == "object":
                params_df.loc[:, col] = params_df.loc[:, col].astype(float)

        temp = params_df.groupby("TotalBudget").mean()
        params_df.reset_index(inplace=True)
        params_df.rename(columns={"TotalBudget": "resource"}, inplace=True)
        base = names.param2filename({"Key": self.parent.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "upper"}, ""
        )

        eval_df.drop("resource", axis=1, inplace=True)
        eval_df.rename(
            columns={
                "TotalBudget": "resource",
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        if median:
            eval_df = eval_df.groupby("resource").median()
        else:
            eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df


class VirtualBestBaseline:
    """
    Calculates virtual best on an instance by instance basis

    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    rec_params : pd.DataFrame
        Dataframe of best paremeters per instance and resource level

    Methods
    -------
    savename()
        Returns the path to save the results
    populate()
        Calculates the virtual best
    evaluate()
        Evaluates the virtual best
    recalibrate()
        Recalibrates the response of virtual best based on best found value
    """

    def __init__(self, parent):
        self.parent = parent
        self.name = "VirtualBest"
        self.populate()

    def savename(self):
        return os.path.join(self.parent.here.checkpoints, "VirtualBest_test.pkl")

    def populate(self):
        if os.path.exists(self.savename()):
            self.rec_params = pd.read_pickle(self.savename())
        else:
            response_col = names.param2filename({"Key": self.parent.response_key}, "")
            testing_results = self.parent.interp_results[
                self.parent.interp_results["train"] == 0
            ].copy()
            self.rec_params = training.virtual_best(
                testing_results,
                parameter_names=self.parent.parameter_names,
                response_col=response_col,
                response_dir=self.parent.response_dir,
                groupby=self.parent.instance_cols,
                resource_col="resource",
                smooth=self.parent.smooth,
                additional_cols=[
                    "ConfInt=lower_" + response_col,
                    "ConfInt=upper_" + response_col,
                ],
            )
            self.rec_params.to_pickle(self.savename())

    def recalibrate(self, new_df):
        """
        Parameters
        ----------
        new_df : pd.DataFrame
            pandas dataframe with the new data. Should only have columns
            ['resource'. *(parameters_names), response, response_lower, response_upper]
            response cols should match name of results columns
        Updates params and evaluation to take in new data
        """
        base = names.param2filename({"Key": self.parent.response_key}, "")
        joint_cols = (
            ["resource", base] + self.parent.parameter_names + self.parent.instance_cols
        )
        new_df = new_df.loc[:, joint_cols]
        joint = pd.concat(
            [self.rec_params.loc[:, joint_cols], new_df], ignore_index=True
        )

        self.rec_params = training.virtual_best(
            joint,
            parameter_names=self.parent.parameter_names,
            response_col=base,
            response_dir=self.parent.response_dir,
            groupby=self.parent.instance_cols,
            resource_col="resource",
            additional_cols=[],
            smooth=self.parent.smooth,
        )

    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.rec_params.loc[:, ["resource"] + self.parent.parameter_names]
        params_df = params_df.groupby("resource").mean()

        base = names.param2filename({"Key": self.parent.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent.response_key, "ConfInt": "upper"}, ""
        )
        eval_df = self.rec_params.copy()
        eval_df.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]

        def StatsSingle(df_single: pd.DataFrame, stat_params: stats.StatsParameters):
            """
            Function for computing the stat (such as mean) and confidence intervals of the response

            Parameters
            ----------
            df_single : pd.DataFrame
                Dataframe of a single resource level
            stat_params : stats.StatsParameters
                Only one stats_measure will be used.

            Returns
            -------
            pd.Dataframe
            """
            df_dict = {}
            sm = stat_params.stats_measures[0]  # Ignore the rest if they exist
            base, CIlower, CIupper = sm.ConfInts(
                df_single["response"],
                df_single["response_lower"],
                df_single["response_upper"],
            )

            df_dict["response"] = [base]
            df_dict["response_lower"] = [CIlower]
            df_dict["response_upper"] = [CIupper]
            df_dict["count"] = len(df_single["response"])

            df_stats_single = pd.DataFrame.from_dict(df_dict)
            return df_stats_single

        def applyBounds(df: pd.DataFrame, stat_params: stats.StatsParameters):
            """
            Trim the response values obtained from statsSingle to be between 0 and 1

            Parameters
            ----------
            df : pd.DataFrame
                Dataframe of a single resource level
            stat_params : stats.StatsParameters
                Only one stats_measure will be used.
            """
            df_copy = df.loc[:, ("response_lower")].copy()
            df_copy.clip(lower=0.0, inplace=True)
            df.loc[:, ("response_lower")] = df_copy

            df_copy = df.loc[:, ("response_upper")].copy()
            df_copy.clip(upper=1.0, inplace=True)
            df.loc[:, ("response_upper")] = df_copy
            return

        def Stats(
            df: pd.DataFrame, stats_params: stats.StatsParameters, group_on=["resource"]
        ):
            """
            Compute a stat(eg. mean) of the response along with CIs for it, for each value of resource for the virtual best

            Parameters
            ----------
            df : pd.DataFrame
                Dataframe of a single resource level with columns 'resource', 'response', 'response_lower' and 'response_upper'
            stats_params : stats.StatsParameters
                Only one statsMeasure will be used.
            group_on : list[str]
                Confidence interval propagation will be done for all rows of dataframe having the same values for groupon

            Returns
            -------
            pd.DataFrame
                Dataframe with columns 'resource', 'response', 'response_lower' and 'response_upper'
            """

            def dfSS(df):
                return StatsSingle(df, stats_params)

            df_stats = (
                df.groupby(group_on)
                .progress_apply(dfSS, include_groups=False)
                .reset_index()
            )
            df_stats.drop("level_{}".format(len(group_on)), axis=1, inplace=True)
            applyBounds(df_stats, stats_params)

            return df_stats

        if median:
            stParams = stats.StatsParameters(
                metrics=["response"], stats_measures=[stats.Median()]
            )
            eval_df = Stats(eval_df, stParams, ["resource"])
        else:
            stParams = stats.StatsParameters(
                metrics=["response"], stats_measures=[stats.Mean()]
            )
            eval_df = Stats(eval_df, stParams, ["resource"])
        eval_df.reset_index(inplace=True)
        return params_df, eval_df


class stochastic_benchmark:
    """
    Attributes
    ----------
    parameter_names : list[str]
        list of parameter names
    here : str
        path to parent directory of data
    instance_cols : list[str]
        Columns that define an instance. i.e., datapoints that match on all of these cols are the same instance
    bsParams_iter :
        Iterator that yields bootstrap parameters
    iParams :
        Interpolation parameters
    stat_params : stats.StatsParameters
        Parameters for computing stats dataframes
    resource_fcn : callable(pd.DataFrame)
        Function that writes a 'resource' function depending on dataframe parameters
    response_key : str
        Column that we want to optimize
    train_test_split : float
        Fraction of instances that should be training data
    recover : bool
        Whether dataframes should be recovered where available or generated from scratch
    reduce_mem : bool
        Whether to reduce memory usage by converting columns to appropriate types
    smooth : bool
        Whether to smooth the response values
    bs_results : pd.DataFrame
        Dataframe of bootstrap results
    interp_results : pd.DataFrame
        Dataframe of interpolation results
    training_stats : pd.DataFrame
        Dataframe of training stats
    testing_stats : pd.DataFrame
        Dataframe of testing stats
    experiments : list[Experiment]
        List of experiments

    Methods
    -------
    initAll(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Initialize all dataframes
    run_Bootstrap(bsParams_iter, group_name_fcn)
        Run bootstrap
    set_Bootstrap(bs_results)
        Set bootstrap results
    run_Interpolate(iParams)
        Run interpolation
    run_Stats(stat_params, train_test_split)
        Run stats
    populate_training_stats()
        Populate training stats
    populate_testing_stats()
        Populate testing stats
    populate_interp_results()
        Populate interpolation results
    evaluate_without_bootstrap(df, group_on)
        Evaluate a dataframe without bootstrap
    run_Baseline(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run baseline
    run_ProjectionExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run projection experiment
    run_RandomSearchExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run random search experiment
    run_SequentialSearchExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run sequential search experiment
    run_StaticRecommendationExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run static recommendation experiment
    init_Plotting()
        Initialize plotting
    """

    def __init__(
        self,
        parameter_names,
        here=os.getcwd(),
        instance_cols=["instance"],
        response_key="PerfRatio",
        response_dir=1,
        recover=True,
        reduce_mem=True,
        smooth=True,
    ):
        # Needed at initialization (for everything)
        self.here = names.paths(here)
        self.parameter_names = parameter_names
        self.instance_cols = instance_cols

        self.recover = recover
        self.reduce_mem = reduce_mem
        self.smooth = smooth

        self.response_key = response_key
        self.response_dir = response_dir

        ## Dataframes needed for experiments and baselines
        self.bs_results = None
        self.interp_results = None
        self.training_stats = None
        self.testing_stats = None

        self.experiments = []

    def initAll(
        self,
        bsParams_iter=default_bootstrap(),
        iParams=None,
        stat_params=stats.StatsParameters(stats_measures=[stats.Median()]),
        resource_fcn=sweep_boots_resource,
        train_test_split=0.5,
        group_name_fcn=None,
    ):
        # Bootstrapping
        self.bsParams_iter = bsParams_iter
        self.group_name_fcn = group_name_fcn  # and stats

        # Interpolation
        self.resource_fcn = resource_fcn

        if iParams is None:
            self.iParams = interpolate.InterpolationParameters(
                self.resource_fcn, parameters=self.parameter_names
            )
        else:
            self.iParams = iParams

        # Stats
        self.stat_params = stat_params
        self.train_test_split = train_test_split

        # Recursive file recovery
        while any(
            [
                v is None
                for v in [self.interp_results, self.training_stats, self.testing_stats]
            ]
        ):
            self.populate_training_stats()
            self.populate_testing_stats()
            self.populate_interp_results()
            # self.populate_bs_results()

    def run_Bootstrap(self, bsParams_iter, group_name_fcn=None):
        if self.bs_results is not None:
            logger.info("Bootstrapped results is already populated: doing nothing.")
            return

        if self.reduce_mem:
            self.raw_data = glob.glob(os.path.join(self.here.raw_data, "*.pkl"))

            if len(self.raw_data) == 0:
                found_bs_results = glob.glob(
                    os.path.join(self.here.checkpoints, "bootstrapped_results*.pkl")
                )
                if len(found_bs_results) >= 1:
                    logger.info(
                        "Found %s bootstrapped results files and no raw data: reading results.",
                        len(found_bs_results),
                    )
                    self.bs_results = found_bs_results
                else:
                    raise Exception(
                        "No raw data found at: {} \n No bootstrapped results found at: {}".format(
                            self.here.raw_data, self.here.checkpoints
                        )
                    )
            else:
                if group_name_fcn is None:
                    raise Exception(
                        "group_name_fcn should be provided for reduced memory version."
                    )

                def raw2bs_names(raw_filename):
                    group_name = group_name_fcn(raw_filename)
                    bs_filename = os.path.join(
                        self.here.checkpoints,
                        "bootstrapped_results_{}.pkl".format(group_name),
                    )
                    return bs_filename

                bs_names = [raw2bs_names(raw_file) for raw_file in self.raw_data]

                if (
                    all([os.path.exists(bs_name) for bs_name in bs_names])
                    and len(bs_names) > 1
                    and self.recover
                ):
                    logger.info(
                        "All bootstrapped results are already found in checkpoints: reading results."
                    )
                    self.bs_results = bs_names
                    return

                group_on = self.parameter_names + self.instance_cols
                self.bs_results = bootstrap.Bootstrap_reduce_mem(
                    self.raw_data,
                    group_on,
                    bsParams_iter,
                    self.here.checkpoints,
                    group_name_fcn,
                )
        else:
            if os.path.exists(self.here.bootstrap) and self.recover:
                logger.info(
                    "All bootstrapped results are already found in checkpoints: reading results."
                )
                self.bs_results = pd.read_pickle(self.here.bootstrap)
                return

            logger.info("Running bootstrapped results")
            group_on = self.parameter_names + self.instance_cols
            if not hasattr(self, "raw_data"):
                self.raw_data = df_utils.read_exp_raw(self.here.raw_data)

            progress_dir = os.path.join(self.here.progress, "bootstrap/")
            if not os.path.exists(progress_dir):
                os.makedirs(progress_dir)

            self.bs_results = bootstrap.Bootstrap(
                self.raw_data, group_on, bsParams_iter, progress_dir
            )
            self.bs_results.to_pickle(self.here.bootstrap)

    def set_Bootstrap(self, bs_results):
        """
        Sets bootstrap results without doing anything
        """
        if type(bs_results) == str:
            self.bs_results = pd.read_pickle(bs_results)
        elif type(bs_results) == pd.DataFrame:
            self.bs_results = bs_results
        elif type(bs_results) == list:
            if type(bs_results[0]) == pd.DataFrame:
                self.bs_results = pd.concat(bs_results, ignore_index=True)
            elif type(bs_results[0]) == str:
                self.bs_results = bs_results

    def run_Interpolate(self, iParams):
        if self.interp_results is not None:
            logger.info("Interpolated results is already populated: doing nothing.")
            return

        if os.path.exists(self.here.interpolate) and self.recover:
            logger.info(
                "Interpolated results are found in checkpoints: reading results."
            )
            self.interp_results = pd.read_pickle(self.here.interpolate)
            return

        if self.bs_results is None:
            raise Exception(
                "Bootstrapped results needs to be populated before interpolation."
            )

        if self.reduce_mem:
            logger.info("Interpolating results with parameters: %s", iParams)
            self.interp_results = interpolate.Interpolate_reduce_mem(
                self.bs_results, iParams, self.parameter_names + self.instance_cols
            )
        else:
            logger.info("Interpolating results with parameters: %s", iParams)
            self.interp_results = interpolate.Interpolate(
                self.bs_results, iParams, self.parameter_names + self.instance_cols
            )

        base = names.param2filename({"Key": self.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.response_key, "ConfInt": "upper"}, ""
        )
        self.interp_results.dropna(subset=[base, CIlower, CIupper], inplace=True)

        # self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, self.train_test_split)
        self.interp_results.to_pickle(self.here.interpolate)
        self.bs_results = None

    def run_Stats(self, stat_params, train_test_split=0.5):
        self.stat_params = stat_params
        if self.interp_results is None:
            raise Exception(
                "Interpolated results needs to be populated before computing stats."
            )

        if "train" not in self.interp_results.columns:
            self.interp_results = training.split_train_test(
                self.interp_results, self.instance_cols, train_test_split
            )
            self.interp_results.to_pickle(self.here.interpolate)

        if self.training_stats is None:
            if os.path.exists(self.here.training_stats) and self.recover:
                logger.info("Training stats found in checkpoints: reading results.")
                self.training_stats = pd.read_pickle(self.here.training_stats)

            else:
                training_results = self.interp_results[
                    self.interp_results["train"] == 1
                ]
                logger.info("Computing training stats")
                self.training_stats = stats.Stats(
                    training_results,
                    stat_params,
                    self.parameter_names + ["boots", "resource"],
                )
                self.training_stats.to_pickle(self.here.training_stats)

        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                logger.info("Testing stats found in checkpoints: reading results.")
                self.testing_stats = pd.read_pickle(self.here.testing_stats)

            else:
                testing_results = self.interp_results[self.interp_results["train"] == 0]
                if len(testing_results) == 0:
                    warnings.warn("There are no testing sets")
                    # raise Exception('No instances assigned to test set. Reassign train/test split')

                else:
                    self.testing_stats = stats.Stats(
                        testing_results,
                        stat_params,
                        self.parameter_names + ["boots", "resource"],
                    )
                    self.testing_stats.to_pickle(self.here.testing_stats)

    def populate_training_stats(self):
        """
        Tries to recover or computes training stats
        """
        if self.training_stats is None:
            if os.path.exists(self.here.training_stats) and self.recover:
                self.training_stats = pd.read_pickle(self.here.training_stats)
            elif self.interp_results is not None:
                training_results = self.interp_results[
                    self.interp_results["train"] == 1
                ]
                logger.info("Computing training stats")
                self.training_stats = stats.Stats(
                    training_results,
                    self.stat_params,
                    self.parameter_names + ["boots", "resource"],
                )
                self.training_stats.to_pickle(self.here.training_stats)

    def populate_testing_stats(self):
        """
        Tries to recover or computes testing stats
        """
        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                self.testing_stats = pd.read_pickle(self.here.testing_stats)

            elif self.interp_results is not None:
                testing_results = self.interp_results[self.interp_results["train"] == 0]
                logger.info("Computing testing stats")
                if len(testing_results) == 0:
                    self.testing_stats = pd.DataFrame()

                else:
                    self.testing_stats = stats.Stats(
                        testing_results,
                        self.stat_params,
                        self.parameter_names + ["boots", "resource"],
                    )
                    self.testing_stats.to_pickle(self.here.testing_stats)

    def populate_interp_results(self):
        """
        Tries to recover or computes interpolated results
        """
        if self.interp_results is None:
            if os.path.exists(self.here.interpolate) and self.recover:
                self.interp_results = pd.read_pickle(self.here.interpolate)
                if "train" not in self.interp_results.columns:
                    self.interp_results = training.split_train_test(
                        self.interp_results, self.instance_cols, self.train_test_split
                    )
                    self.interp_results.to_pickle(self.here.interpolate)

            elif self.bs_results is not None:
                # print(self.bs_results)
                if self.reduce_mem:
                    logger.info(
                        "Interpolating results with parameters: %s", self.iParams
                    )
                    self.interp_results = interpolate.Interpolate_reduce_mem(
                        self.bs_results,
                        self.iParams,
                        self.parameter_names + self.instance_cols,
                    )
                else:
                    logger.info(
                        "Interpolating results with parameters: %s", self.iParams
                    )
                    self.interp_results = interpolate.Interpolate(
                        self.bs_results,
                        self.iParams,
                        self.parameter_names + self.instance_cols,
                    )

                base = names.param2filename({"Key": self.response_key}, "")
                CIlower = names.param2filename(
                    {"Key": self.response_key, "ConfInt": "lower"}, ""
                )
                CIupper = names.param2filename(
                    {"Key": self.response_key, "ConfInt": "upper"}, ""
                )
                self.interp_results.dropna(
                    subset=[base, CIlower, CIupper], inplace=True
                )

                self.interp_results = training.split_train_test(
                    self.interp_results, self.instance_cols, self.train_test_split
                )
                self.interp_results.to_pickle(self.here.interpolate)
                self.bs_results = None
            else:
                self.populate_bs_results(self.bsParams_iter, self.group_name_fcn)

    # def populate_bs_results(self, group_name_fcn=None):
    #     """
    #     Tries to recover or computes bootstrapped results
    #     """

    #     if self.bs_results is None:
    #         if self.reduce_mem:
    #             def raw2bs_names(raw_filename):
    #                 group_name = group_name_fcn(raw_filename)
    #                 bs_filename = os.path.join(self.here.checkpoints, 'bootstrapped_results_{}.pkl'.format(group_name))
    #                 return bs_filename

    #             self.raw_data = glob.glob(os.path.join(self.here.raw_data, '*.pkl'))
    #             bs_names = [raw2bs_names(raw_file) for raw_file in self.raw_data]

    #             if all([os.path.exists(bs_name) for bs_name in bs_names]) and len(bs_names) > 1 and self.recover:
    #                 print('Reading bootstrapped results')
    #                 self.bs_results = bs_names
    #             else:
    #                 group_on = self.parameter_names + self.instance_cols
    #                 if not hasattr(self, 'raw_data'):
    #                     print('Running bootstrapped results')
    #                     self.raw_data = glob.glob(os.path.join(self.here.raw_data, '*.pkl'))
    #                 self.bs_results = bootstrap.Bootstrap_reduce_mem(self.raw_data, group_on, self.bsParams_iter, self.here.checkpoints, group_name_fcn)

    #         else:
    #             if os.path.exists(self.here.bootstrap) and self.recover:
    #                 print('Reading bootstrapped results')
    #                 self.bs_results = pd.read_pickle(self.here.bootstrap)
    #             else:
    #                 print('Running bootstrapped results')
    #                 group_on = self.parameter_names + self.instance_cols
    #                 if not hasattr(self, 'raw_data'):
    #                     self.raw_data = df_utils.read_exp_raw(self.here.raw_data)

    #                 progress_dir = os.path.join(self.here.progress, 'bootstrap/')
    #                 if not os.path.exists(progress_dir):
    #                     os.makedirs(progress_dir)

    #                 self.bs_results = bootstrap.Bootstrap(self.raw_data, group_on, self.bsParams_iter, progress_dir)
    #                 self.bs_results.to_pickle(self.here.bootstrap)

    def evaluate_without_bootstrap(self, df, group_on):
        """ "
        Runs same computations evaluations as bootstrap without bootstrapping

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe of results
        group_on : list[str]
            Columns to group on

        Returns
        -------
        pd.DataFrame
            Dataframe of results
        """
        bs_params = next(self.bsParams_iter)
        resource_col = bs_params.shared_args["resource_col"]
        response_col = bs_params.shared_args["response_col"]
        agg = bs_params.agg

        def evaluate_single(df_single):
            bs_params.update_rule(bs_params, df_single)
            resources = df_single[resource_col].values
            responses = df_single[response_col].values
            resources = np.repeat(resources, df_single[agg])
            responses = np.repeat(responses, df_single[agg])

            bs_df = pd.DataFrame()
            for metric_ref in bs_params.success_metrics:
                metric = metric_ref(
                    bs_params.shared_args, bs_params.metric_args[metric_ref.__name__]
                )
                metric.evaluate(bs_df, responses, resources)
            for col in bs_params.keep_cols:
                if col in df_single.columns:
                    val = df_single[col].iloc[0]
                    bs_df[col] = val

            return bs_df

        full_eval = (
            df.groupby(group_on)
            .apply(lambda df: evaluate_single(df), include_groups=False)
            .reset_index()
        )
        full_eval.drop(columns=["level_{}".format(len(group_on))], inplace=True)
        return full_eval

    def run_baseline(self):
        """
        Adds virtual best baseline
        """
        logger.info("Runnng baseline")
        self.baseline = VirtualBestBaseline(self)

    def run_ProjectionExperiment(
        self, project_from, postprocess=None, postprocess_name=None
    ):
        """
        Runs projections experiments

        Parameters
        ----------
        project_from : str
            Name of experiment to project from
        postprocess : callable(pd.DataFrame)
            Function to postprocess results
        postprocess_name : str
            Name of postprocess

        Returns
        -------
        ProjectionExperiment
            Experiment object
        """
        logger.info("Running projection experiment")
        self.experiments.append(
            ProjectionExperiment(self, project_from, postprocess, postprocess_name)
        )

    def run_RandomSearchExperiment(
        self, rsParams, postprocess=None, postprocess_name=None
    ):
        """
        Runs random search experiments
        """
        logger.info("Running random search experiment")
        self.experiments.append(
            RandomSearchExperiment(
                self,
                rsParams,
                postprocess=postprocess,
                postprocess_name=postprocess_name,
            )
        )

    def run_SequentialSearchExperiment(
        self, ssParams, id_name=None, postprocess=None, postprocess_name=None
    ):
        """
        Runs sequential search experiments

        Parameters
        ----------
        ssParams : SequentialSearchParameters
            Parameters for sequential search
        id_name : str
            Name of experiment
        postprocess : callable(pd.DataFrame)
            Function to postprocess results
        postprocess_name : str
            Name of postprocess

        Returns
        -------
        SequentialSearchExperiment
            Experiment object
        """
        logger.info("Running sequential search experiment")
        self.experiments.append(
            SequentialSearchExperiment(
                self,
                ssParams,
                id_name,
                postprocess=postprocess,
                postprocess_name=postprocess_name,
            )
        )

    def run_StaticRecommendationExperiment(self, init_from):
        """
        Runs static recommendation experiments

        Parameters
        ----------
        init_from : str
            Name of experiment to initialize from

        Returns
        -------
        StaticRecommendationExperiment
            Experiment object
        """
        logger.info("Running static recommendation experiment")
        self.experiments.append(StaticRecommendationExperiment(self, init_from))

    def initPlotting(self):
        """
        Sets up plotting - this should be run after all experiments are run
        """
        self.plots = Plotting(self.here.checkpoints)

    def export_plot_csvs(self, monotone=False):
        """Save csv files required for plotting without loading pickle files."""

        params_dir = os.path.join(self.here.checkpoints, "params_plotting")
        perf_dir = os.path.join(self.here.checkpoints, "performance_plotting")
        meta_dir = os.path.join(self.here.checkpoints, "meta_params_plotting")
        for d in [params_dir, perf_dir, meta_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Baseline parameters and performance
        params_df, eval_df = self.baseline.evaluate()
        params_df.to_csv(os.path.join(params_dir, "baseline.csv"))
        eval_df.to_csv(os.path.join(perf_dir, "baseline.csv"))

        # Experiments
        for experiment in self.experiments:
            if monotone and hasattr(experiment, "evaluate_monotone"):
                res = experiment.evaluate_monotone()
            else:
                res = experiment.evaluate()

            params_df = res[0]
            eval_df = res[1]

            params_df.to_csv(os.path.join(params_dir, f"{experiment.name}.csv"))
            if len(res) == 3:
                preproc_params = res[2]
                preproc_params.to_csv(
                    os.path.join(params_dir, f"{experiment.name}params.csv")
                )

            eval_df.to_csv(os.path.join(perf_dir, f"{experiment.name}.csv"))

            if hasattr(experiment, "meta_params"):
                mp_df = experiment.meta_params.copy()
                sort_col = getattr(experiment, "resource", "TotalBudget")
                if sort_col in mp_df.columns:
                    mp_df.sort_values(by=sort_col, inplace=True)
                mp_df.to_csv(os.path.join(meta_dir, f"{experiment.name}.csv"))

            if hasattr(experiment, "preproc_meta_params"):
                mp_df = experiment.preproc_meta_params.copy()
                sort_col = getattr(experiment, "resource", "TotalBudget")
                if sort_col in mp_df.columns:
                    mp_df.sort_values(by=sort_col, inplace=True)
                mp_df.to_csv(os.path.join(meta_dir, f"{experiment.name}_preproc.csv"))
