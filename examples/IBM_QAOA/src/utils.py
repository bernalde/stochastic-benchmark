import re
import math
from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


def is_empty_nested_list(x):
    """Check whether `x` is a non-empty list of empty lists.

    Parameters
    ----------
    x : object
        Value to check.

    Returns
    -------
    bool
        True if `x` is a list with at least one element and every element is an empty list.
    """
    return isinstance(x, list) and len(x) > 0 and all(isinstance(i, list) and len(i) == 0 for i in x)

def counts_to_samples_df(df_hardware: pd.DataFrame) -> pd.DataFrame:
    """
    Expand df_hardware rows into one row per *shot* (or per unique bitstring w/ weight).
    Returns a really long DataFrame with columns:
    - instance_name, training_method, job_p, training_p, bitstring, count, prob
    """
    rows = []
    for _, r in df_hardware.iterrows():
        counts = r["counts"]
        if not isinstance(counts, dict) or len(counts) == 0:
            continue

        total = sum(counts.values())
        for b, c in counts.items():
            rows.append({
                "instance_name": r["instance_name"],
                "training_method": r["training_method"],
                "job_p": r["job_p"],
                "train_p": r["training_p"],
                "bitstring": b,
                "count": c,
                "prob": c / total if total else np.nan,
            })

    return pd.DataFrame(rows)

def plot_ar_hist_by_training_method_with_points(
    df_samples,
    instance_name: str,
    job_p: int,
    bins=70,
    width_scale=0.42,
    symmetric=True,
    normalize="global"):

    # ---------- Filter ----------
    df = df_samples[(df_samples["instance_name"] == instance_name) &
                    (df_samples["job_p"] == job_p)].copy()
    if df.empty:
        raise ValueError("No rows found for that (instance_name, job_p).")

    # ---------- Top 1% probability mass per method ----------
    df_top = (
        df.sort_values(["training_method", "approximation_ratio"], ascending=[True, False])
          .groupby("training_method", group_keys=False)
          .apply(lambda g: g.head(1) if g["prob"].iloc[0] > 0.01
                 else g[g["prob"].cumsum() <= 0.01])
    )

    def _plot_one(dfin, title_suffix):
        methods = sorted(dfin["training_method"].unique())
        x_pos = np.arange(len(methods))
        x_map = {m: x_pos[i] for i, m in enumerate(methods)}

        best_df = (dfin.groupby("training_method", as_index=False)["approximation_ratio"]
                       .max().rename(columns={"approximation_ratio": "best_AR"}))

        y = dfin["approximation_ratio"].to_numpy()
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        bin_edges = np.linspace(y_min, y_max, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_h = (bin_edges[1] - bin_edges[0]) * 0.9

        # ---------- Weighted hist per method ----------
        H = []
        for m in methods:
            d = dfin[dfin["training_method"] == m]
            h, _ = np.histogram(
                d["approximation_ratio"].to_numpy(),
                bins=bin_edges,
                weights=d["prob"].to_numpy(),
            )
            H.append(h)
        H = np.array(H)

        if normalize == "per_method":
            denom = np.maximum(H.max(axis=1, keepdims=True), 1e-12)
        else:
            denom = max(H.max(), 1e-12)

        W = (H / denom) * width_scale

        # ---------- Plot ----------
        fig, ax = plt.subplots(figsize=(1.25 * len(methods) + 3, 5))
        for i, m in enumerate(methods):
            widths = W[i]
            if symmetric:
                ax.barh(bin_centers, 2 * widths, left=x_pos[i] - widths,
                        height=bin_h, alpha=0.85)
            else:
                ax.barh(bin_centers, widths, left=x_pos[i],
                        height=bin_h, alpha=0.85)

        ax.scatter(best_df["training_method"].map(x_map),
                   best_df["best_AR"],
                   s=45, edgecolors="k", linewidths=0.6,
                   label="Best AR", zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel("Approximation Ratio")
        ax.set_xlabel("Training_method")
        ax.set_xlim(-0.7, len(methods) - 0.3)
        ax.set_ylim(0.45, 1.00)
        ax.set_title(f"{instance_name} | p={job_p} | {title_suffix}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # ---------- Plots ----------
    _plot_one(df, "All samples")
    _plot_one(df_top, "Top 1%")

def plot_training_bricks(agg, step_cols):

    def lighten(color, amount=0.5):
        c = np.array(mcolors.to_rgb(color))
        return tuple(c + (1 - c) * amount)

    methods = sorted(agg["method_base"].unique())
    depths  = sorted(agg["job_p"].dropna().unique())

    EDGE_LW = 0.9

    fig, ax = plt.subplots(figsize=(14,5))

    cmap = plt.get_cmap("tab10")
    method_color = {m: cmap(i % 10) for i,m in enumerate(methods)}

    x = np.arange(len(depths))
    width = 0.8 / max(1,len(methods))

    for i,m in enumerate(methods):
        sub = agg[agg["method_base"]==m].set_index("job_p").reindex(depths)

        xpos = x - 0.4 + width/2 + i*width
        bottom = np.zeros(len(depths))

        base = method_color[m]

        outer_vals = sub["outer_init"].to_numpy()
        ax.bar(xpos, outer_vals, width, bottom=bottom,
            color=base, edgecolor="black", linewidth=EDGE_LW)
        bottom += outer_vals

        for s_idx,c in enumerate(step_cols, start=1):
            vals = sub[c].to_numpy()
            col = lighten(base, amount=min(0.85, 0.18+0.06*s_idx))
            ax.bar(xpos, vals, width, bottom=bottom,
                color=col, edgecolor="black", linewidth=EDGE_LW)
            bottom += vals

        ax.errorbar(
            xpos,
            sub["brick_total"].to_numpy(),
            yerr=sub["sem_total"].to_numpy(),
            fmt="none", ecolor="black", capsize=4, lw=1.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Mean training duration (s)")
    ax.set_title("Mean training duration across instances with depth wise breakdown")
    ax.set_yscale("log")

    handles=[plt.Rectangle((0,0),1,1,facecolor=method_color[m],edgecolor="black") for m in methods]
    ax.legend(handles,methods,bbox_to_anchor=(1.02,1),loc="upper left",frameon=False)

    plt.tight_layout()
    plt.show()


def sem(s: pd.Series) -> float:
    """Compute the standard error of the mean (SEM) of a Series.

    Parameters
    ----------
    s : pandas.Series
        Input values.

    Returns
    -------
    float
        Standard error of the mean. Returns 0.0 when fewer than two non-null
        observations are available.
    """
    n = int(s.count())
    return 0.0 if n <= 1 else float(s.std(ddof=1) / math.sqrt(n))


def title_from_instance_names(d: pd.DataFrame, p_val: float) -> str:
    """Build a plot title from instance names and a QAOA depth.

    Parameters
    ----------
    d : pandas.DataFrame
        DataFrame containing an optional ``instance_name`` column.
    p_val : float
        QAOA depth value to include in the title.

    Returns
    -------
    str
        Title string. If ``instance_name`` is not present or contains no valid
        values, the title will only include ``p``.
    """
    p_txt = int(p_val) if float(p_val).is_integer() else p_val
    names = d["instance_name"].dropna().astype(str).unique().tolist() if "instance_name" in d.columns else []
    if not names:
        return f"p = {p_txt}"
    cores = sorted({n[3:] for n in names if len(n) > 3})
    return f"{cores[0]} | p = {p_txt}" if len(cores) == 1 else f"p = {p_txt}"


def make_asof_per_file(inner: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create a per-file as-of merge function for accumulating inner durations.

    This factory exists to mirror the notebook-local ``asof_per_file`` helper,
    while keeping ``inner`` as an explicit dependency.

    Parameters
    ----------
    inner : pandas.DataFrame
        Precomputed DataFrame with cumulative inner durations.
        Must contain the columns ``file_name``, ``depth_step``, and
        ``inner_cum``.

    Returns
    -------
    Callable[[pandas.DataFrame], pandas.DataFrame]
        A function suitable for use with ``df.groupby('file_name').apply(...)``.
        The group DataFrame is expected to contain a numeric ``job_p`` column.

    Notes
    -----
    The returned function:
    - Drops rows where ``job_p`` is NaN (matching the notebook behavior).
    - Adds an ``inner_duration_sum`` column representing the cumulative sum of
      inner durations up to (and including) the largest ``depth_step`` not
      exceeding ``job_p``.
    """

    def asof_per_file(g: pd.DataFrame) -> pd.DataFrame:
        fn = g.name
        rhs = inner[inner["file_name"].eq(fn)].sort_values("depth_step")
        g2 = g.dropna(subset=["job_p"]).sort_values("job_p")
        if rhs.empty:
            g2["inner_duration_sum"] = 0.0
            return g2
        out = pd.merge_asof(
            g2,
            rhs[["depth_step", "inner_cum"]],
            left_on="job_p",
            right_on="depth_step",
            direction="backward",
        )
        out["inner_duration_sum"] = out["inner_cum"].fillna(0.0)
        return out.drop(columns=["depth_step", "inner_cum"], errors="ignore")

    return asof_per_file