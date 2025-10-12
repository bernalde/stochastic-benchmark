import df_utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import training
import os
import numpy as np
import logging
from matplotlib.collections import LineCollection

monotone = False
plot_vb_CI = True
dir_path = os.path.dirname(os.path.realpath(__file__))
ws_style = os.path.join(dir_path, "ws.mplstyle")

plt.style.use(ws_style)

logger = logging.getLogger(__name__)


class Plotting:
    """
    Plotting helpers for coordinating plots using csv files

    Attributes
    ----------
    checkpoints_dir : str
        Directory where plotting csv files are stored
    colors : list[str]
        Color palette for experiments. Baseline will always be black
    xcale : str
        Scale for shared x axis
    xlims : tuple
        Limits for shared x axis

    Methods
    -------
    __init__(checkpoints_dir)
        Initialize plotting object
    set_colors(cp)
        Sets color palette and reassigns colors to experiments
    set_xlims(xlims)
        Sets limits for shared x
    make_legend(ax, baseline_bool, experiment_bools)
        Makes legend for each experiment
    apply_shared(p, baseline_bool=True, experiment_bools=None)
        Apply shared plot components (xscale, xlim, legends)
    plot_parameters_together()
        Plot the parameters together
    plot_parameters_separate()
        Plot the parameters separately
    plot_parameters_distance()
        Plot the parameters distance to the virtual best
    plot_performance()
        Plot the performance
    plot_meta_parameters()
        Plot the meta parameters
    """

    def __init__(self, checkpoints_dir):
        self.checkpoints_dir = checkpoints_dir
        self.params_dir = os.path.join(checkpoints_dir, "params_plotting")
        self.perf_dir = os.path.join(checkpoints_dir, "performance_plotting")
        self.meta_dir = os.path.join(checkpoints_dir, "meta_params_plotting")

        # Determine experiments and parameters from stored csv files
        baseline_csv = os.path.join(self.params_dir, "baseline.csv")
        if os.path.exists(baseline_csv):
            df = pd.read_csv(baseline_csv)
            self.parameter_names = [c for c in df.columns if c != "resource"]
        else:
            self.parameter_names = []

        self.experiment_names = []
        if os.path.exists(self.params_dir):
            for f in os.listdir(self.params_dir):
                if not f.endswith(".csv"):
                    continue
                name = os.path.splitext(f)[0]
                if name == "baseline" or name.endswith("params"):
                    continue
                self.experiment_names.append(name)

        self.colors = sns.color_palette("tab10", len(self.experiment_names))
        self.baseline_color = "black"
        self.experiment_colors = {
            n: self.colors[i] for i, n in enumerate(self.experiment_names)
        }
        self.xscale = "log"

    def set_colors(self, cp):
        """
        Sets color palette and reassigns colors to experiments

        Parameters
        ----------
        cp : list[str]
            Color palette for experiments. Baseline will always be black
        """
        self.colors = cp
        self.experiment_colors = {
            n: self.colors[i] for i, n in enumerate(self.experiment_names)
        }

    def set_xlims(self, xlims):
        """
        Sets limits for shared x

        Parameters
        ----------
        xlims : tuple
            limits for shared x axis
        """
        self.xlims = xlims

    def make_legend(self, ax, baseline_bool, experiment_bools):
        """
        Makes legend for each experiment

        Parameters
        ----------
        ax : matplotlib.axes
            axes to plot on
        baseline_bool : bool
            whether to include baseline in legend
        experiment_bools : list[bool]
            whether to include each experiment in legend
        """
        if baseline_bool:
            color_patches = [
                mpatches.Patch(color=self.baseline_color, label="baseline")
            ]
        else:
            color_patches = []

        for idx, name in enumerate(self.experiment_names):
            if experiment_bools[idx]:
                color_patches.append(
                    mpatches.Patch(color=self.experiment_colors[name], label=name)
                )

        ax.legend(handles=color_patches)

    def apply_shared(self, p, baseline_bool=True, experiment_bools=None):
        """
        Apply shared plot components (xscale, xlim, legends)

        Parameters
        ----------
        p : seaborn object
            plot to apply shared components to
        baseline_bool : bool
            whether to include baseline in legend
        experiment_bools : list[bool]
            whether to include each experiment in legend
        """
        if experiment_bools is None:
            experiment_bools = [True] * len(self.experiment_names)

        if type(p) is dict:
            for k, v in p.items():
                p[k] = self.apply_shared(v, baseline_bool, experiment_bools)
            return p

        p = p.scale(x=self.xscale)
        if hasattr(self, "xlims"):
            p = p.limit(x=self.xlims)

        fig = plt.figure()
        p = p.on(fig).plot()
        ax = fig.axes[0]
        self.make_legend(ax, baseline_bool, experiment_bools)

        return fig

    def plot_parameters_together(self):
        """Plot the parameters (Virtual Best and projection experiments)
        Create a single figure with a subfigure corresponding to each parameter

        Returns
        -------
        fig : matplotlib.figure
            Figure handle
        axes : dict
            Dictionary of axis handles. The keys are the parameter names
        """

        fig, axes_list = plt.subplots(len(self.parameter_names), 1)

        # Convert axes_list to a dictionary
        axes = dict()
        for ind, param in enumerate(self.parameter_names):
            axes[param] = axes_list[ind]

        # Get the best parameters from the Virtual Baseline
        params_df = pd.read_csv(os.path.join(self.params_dir, "baseline.csv"))
        eval_df = pd.read_csv(os.path.join(self.perf_dir, "baseline.csv"))
        eval_df = df_utils.monotone_df(eval_df, "resource", "response", 1)

        # plot the virtual baseline parameters
        for param in self.parameter_names:
            points = np.array(
                [params_df.index.values, params_df[param].values]
            ).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(eval_df["response"].min(), eval_df["response"].max())
            lc = LineCollection(segments, cmap="Spectral", norm=norm)
            # Set the values used for colormapping
            lc.set_array(eval_df["response"])
            lc.set_label("baseline")
            lc.set_linewidth(8)
            lc.set_alpha(0.75)
            line = axes[param].add_collection(lc)
            _ = axes[param].plot(
                params_df.index, params_df[param], "o", ms=2, mec="k", alpha=0.25
            )

        cbar = fig.colorbar(line, ax=axes_list.ravel().tolist())
        cbar.ax.tick_params()
        cbar.set_label("response")

        # Plot parameters from experiments
        for exp in self.experiment_names:
            params_df = pd.read_csv(os.path.join(self.params_dir, f"{exp}.csv"))
            preproc_file = os.path.join(self.params_dir, f"{exp}params.csv")
            has_meta = os.path.exists(os.path.join(self.meta_dir, f"{exp}.csv"))

            for param in self.parameter_names:
                if not has_meta:
                    _ = axes[param].plot(
                        params_df["resource"],
                        params_df[param],
                        "o-",
                        ms=2,
                        lw=1.5,
                        color=self.experiment_colors[exp],
                        label=exp,
                    )
                if os.path.exists(preproc_file):
                    preproc_params = pd.read_csv(preproc_file)
                    axes[param].plot(
                        preproc_params["resource"],
                        preproc_params[param],
                        color=self.experiment_colors[exp],
                        marker="x",
                        linestyle=":",
                        ms=2,
                        lw=1.5,
                    )

        # Finally, add more properties such as labels, legend, etc.
        for param in self.parameter_names:
            axes[param].grid(axis="y")
            axes[param].set_ylabel(param)
            axes[param].set_xscale(self.xscale)
            axes[param].set_xlabel("Resource")
            if hasattr(self, "xlims"):
                axes[param].set_xlim(self.xlims)
            # axes[param].legend()
        handles, labels = axes_list[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=[0.5, 0], loc="upper center")
        # plt.legend()
        # fig.tight_layout()

        return fig, axes

    def plot_parameters_separate(self):
        """Plot the parameters (Virtual Best and projection experiments)
        Create a separate figure for each parameter

        Returns:
            figs: Dictionary of figure handles. The keys are the parameter names
            axes: Dictionary of axis handles. The keys are the parameter names
        """
        # For each resource value, obtain the best parameter value from VirtualBestBaseline

        figs = dict()
        axes = dict()

        for param in self.parameter_names:
            # Create one figure for each parameter
            figs[param], axes[param] = plt.subplots(1, 1)

        # Get the best parameters from the Virtual Baseline
        params_df = pd.read_csv(os.path.join(self.params_dir, "baseline.csv"))
        eval_df = pd.read_csv(os.path.join(self.perf_dir, "baseline.csv"))
        eval_df = df_utils.monotone_df(eval_df, "resource", "response", 1)

        # plot the virtual baseline parameters
        for param in self.parameter_names:
            points = np.array(
                [params_df.index.values, params_df[param].values]
            ).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(eval_df["response"].min(), eval_df["response"].max())
            lc = LineCollection(segments, cmap="Spectral", norm=norm)
            # Set the values used for colormapping
            lc.set_array(eval_df["response"])
            lc.set_label("baseline")
            lc.set_linewidth(8)
            lc.set_alpha(0.75)
            line = axes[param].add_collection(lc)
            cbar = figs[param].colorbar(line, ax=axes[param])
            cbar.ax.tick_params()
            cbar.set_label("response")

            _ = axes[param].plot(
                params_df.index, params_df[param], "o", ms=2, mec="k", alpha=0.25
            )

        # Plot parameters from experiments
        for exp in self.experiment_names:
            params_df = pd.read_csv(os.path.join(self.params_dir, f"{exp}.csv"))
            preproc_file = os.path.join(self.params_dir, f"{exp}params.csv")
            has_meta = os.path.exists(os.path.join(self.meta_dir, f"{exp}.csv"))

            for param in self.parameter_names:
                if not has_meta:
                    _ = axes[param].plot(
                        params_df["resource"],
                        params_df[param],
                        "o-",
                        ms=2,
                        lw=1.5,
                        color=self.experiment_colors[exp],
                        label=exp,
                    )
                if os.path.exists(preproc_file):
                    preproc_params = pd.read_csv(preproc_file)
                    axes[param].plot(
                        preproc_params["resource"],
                        preproc_params[param],
                        color=self.experiment_colors[exp],
                        marker="x",
                        linestyle=":",
                        ms=2,
                        lw=1.5,
                    )

        # Finally, add more properties such as labels, legend, etc.
        for param in self.parameter_names:
            axes[param].grid(axis="y")
            axes[param].set_ylabel(param)
            axes[param].set_xscale(self.xscale)
            axes[param].set_xlabel("Resource")
            if hasattr(self, "xlims"):
                axes[param].set_xlim(self.xlims)
            axes[param].legend()
            figs[param].tight_layout()

        return figs, axes

    def plot_parameters_distance(self):
        """
        Plots the scaled distance between the recommended parameters and virtual best
        """
        recipes = pd.read_csv(os.path.join(self.params_dir, "baseline.csv"))

        all_params_list = []
        count = 0
        for exp in self.experiment_names:
            params_df = pd.read_csv(os.path.join(self.params_dir, f"{exp}.csv"))
            params_df["exp_idx"] = count
            all_params_list.append(params_df)
            count += 1

        all_params = pd.concat(all_params_list, ignore_index=True)
        dist_params_list = []

        for _, recipe in recipes.reset_index().iterrows():
            res_df = all_params[all_params["resource"] == recipe["resource"]]
            temp_df_eval = training.scaled_distance(
                res_df, recipe, self.parameter_names
            )
            temp_df_eval.loc[:, "resource"] = recipe["resource"]
            dist_params_list.append(temp_df_eval)
        all_params = pd.concat(dist_params_list, ignore_index=True)

        fig, axs = plt.subplots(1, 1)
        axs.plot(all_params["resource"], all_params["distance_scaled"])

        for idx, exp in enumerate(self.experiment_names):
            metaflag = os.path.exists(os.path.join(self.meta_dir, f"{exp}.csv"))
            params_df = all_params[all_params["exp_idx"] == idx]
            if metaflag:
                axs.plot(
                    params_df["resource"],
                    params_df["distance_scaled"],
                    marker="x",
                    linestyle=":",
                    color=self.experiment_colors[exp],
                    label=exp,
                )
            else:
                axs.plot(
                    params_df["resource"],
                    params_df["distance_scaled"],
                    marker="o",
                    color=self.experiment_colors[exp],
                    label=exp,
                )

        axs.grid(axis="y")
        axs.set_ylabel("distance_scaled")
        axs.set_xscale(self.xscale)
        axs.set_xlabel("Resource")
        axs.legend(loc="best")
        fig.tight_layout()

        return fig, axs

    def plot_performance(self):
        """
        Plots the monotonized performance for each experiment (with the baseline)
        """
        # If saved data for virtual best exists, simply load it. Otherwise, compute the curve from baseline.
        save_loc = self.perf_dir
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        save_file = os.path.join(save_loc, "baseline.csv")
        eval_df = pd.read_csv(save_file)

        fig, axs = plt.subplots(1, 1)
        if plot_vb_CI:
            axs.fill_between(
                eval_df["resource"],
                eval_df["response_lower"],
                eval_df["response_upper"],
                alpha=0.25,
                color="k",
                lw=0,
            )
        _ = axs.plot(
            eval_df["resource"],
            eval_df["response"],
            "o-",
            ms=5,
            lw=1,
            color=self.baseline_color,
            label="baseline",
        )

        for exp in self.experiment_names:
            try:
                save_file = os.path.join(save_loc, f"{exp}.csv")
                eval_df = pd.read_csv(save_file)

                # Add confidence intervals
                axs.fill_between(
                    eval_df["resource"],
                    eval_df["response_lower"],
                    eval_df["response_upper"],
                    alpha=0.25,
                    color=self.experiment_colors[exp],
                    lw=0,
                )  # , label="CI Mean"+legend_str) # color='b',
                # Add mean/median line
                _ = axs.plot(
                    eval_df["resource"],
                    eval_df["response"],
                    "o-",
                    ms=5,
                    lw=1,
                    color=self.experiment_colors[exp],
                    label=exp,
                )
            except:
                continue

        axs.grid(axis="y")
        axs.set_ylabel("response")
        axs.set_xscale(self.xscale)
        axs.set_xlabel("Resource")
        if hasattr(self, "xlims"):
            axs.set_xlim(self.xlims)
        axs.legend(loc="lower right")
        fig.tight_layout()
        return fig, axs

    def plot_meta_parameters(self):
        """
        Plots meta parameters for experiments that have them (random search and sequential search)
        """
        figs = dict()
        axes = dict()

        # Location where meta parameter csv files are saved or are to be saved
        save_loc = self.meta_dir
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        for idx, exp in enumerate(self.experiment_names):
            exp_figs = {}
            exp_axes = {}
            save_file = os.path.join(save_loc, f"{exp}.csv")
            if not os.path.exists(save_file):
                continue
            metaparams_df = pd.read_csv(save_file)

            preproc_file = os.path.join(save_loc, f"{exp}_preproc.csv")
            metaparams_preproc_df = None
            if os.path.exists(preproc_file):
                metaparams_preproc_df = pd.read_csv(preproc_file)

            meta_params_cols = [c for c in metaparams_df.columns if c != "TotalBudget"]
            for param in meta_params_cols:
                fig, axs = plt.subplots(1, 1)
                axs.plot(
                    metaparams_df["TotalBudget"],
                    metaparams_df[param],
                    color=self.experiment_colors[exp],
                    marker="o",
                    label=exp,
                )
                if (
                    metaparams_preproc_df is not None
                    and param in metaparams_preproc_df.columns
                ):
                    axs.plot(
                        metaparams_preproc_df["TotalBudget"],
                        metaparams_preproc_df[param],
                        color=self.experiment_colors[exp],
                        marker="x",
                        linestyle="--",
                    )
                axs.grid(axis="y")
                axs.set_ylabel(param)
                axs.set_xscale(self.xscale)
                axs.set_xlabel("TotalBudget")
                axs.legend(loc="best")
                fig.tight_layout()
                exp_figs[param] = fig
                exp_axes[param] = axs

            figs[exp] = exp_figs
            axes[exp] = exp_axes
        return figs, axes
