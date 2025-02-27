import json
import os
from pathlib import Path
import sys
from typing import cast

import pandas as pd
from plot2 import load_merged_df, default_exclude_datasets
from plot import AllOutput, AllOutputHandler
from common import default_compression_method_order, default_omit_methods

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import operator
import math
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import networkx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns


def Friedman_Nemenyi(alpha=0.05, df_perf=None):
    df_counts = pd.DataFrame(
        {"count": df_perf.groupby(["classifier_name"]).size()}
    ).reset_index()
    # Record the maximum number of datasets
    max_nb_datasets = df_counts["count"].max()
    # Create a list of classifiers
    classifiers = list(
        df_counts.loc[df_counts["count"] == max_nb_datasets]["classifier_name"]
    )

    # print('classifiers: ', classifiers)

    """
    Expected input format for friedmanchisquare is:
                Dataset1        Dataset2        Dataset3        Dataset4        Dataset5
    classifer1
    classifer2
    classifer3 
    """

    # Compute friedman p-value
    friedman_p_value = friedmanchisquare(
        *(
            np.array(df_perf.loc[df_perf["classifier_name"] == c]["accuracy"])
            for c in classifiers
        )
    )[1]

    # Decide whether to reject the null hypothesis
    # If p-value >= alpha: we cannot reject the null hypothesis. No statistical difference.
    if friedman_p_value >= alpha:
        print("No statistical difference...")
        return None, None, None
    # Friedman test OK
    # Prepare input for Nemenyi test
    data = []
    for c in classifiers:
        data.append(df_perf.loc[df_perf["classifier_name"] == c]["accuracy"])
    data = np.array(data, dtype=np.float64)
    # Conduct the Nemenyi post-hoc test
    # print(classifiers)
    # Order is classifiers' order
    nemenyi = posthoc_nemenyi_friedman(data.T)

    # print(nemenyi)

    # Original code: p_values.append((classifier_1, classifier_2, p_value, False)), True: represents there exists statistical difference
    p_values = []

    # Comparing p-values with the alpha value
    for nemenyi_indx in nemenyi.index:
        for nemenyi_columns in nemenyi.columns:
            if nemenyi_indx < nemenyi_columns:
                if nemenyi.loc[nemenyi_indx, nemenyi_columns] < alpha:
                    p_values.append(
                        (
                            classifiers[nemenyi_indx],
                            classifiers[nemenyi_columns],
                            nemenyi.loc[nemenyi_indx, nemenyi_columns],
                            True,
                        )
                    )
                else:
                    p_values.append(
                        (
                            classifiers[nemenyi_indx],
                            classifiers[nemenyi_columns],
                            nemenyi.loc[nemenyi_indx, nemenyi_columns],
                            False,
                        )
                    )
            else:
                continue

    # Nemenyi test OK

    m = len(classifiers)

    # Sort by classifier name then by dataset name
    sorted_df_perf = df_perf.loc[
        df_perf["classifier_name"].isin(classifiers)
    ].sort_values(["classifier_name", "dataset_name"])

    rank_data = np.array(sorted_df_perf["accuracy"]).reshape(m, max_nb_datasets)

    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(classifiers),
        columns=np.unique(sorted_df_perf["dataset_name"]),
    )

    dfff = df_ranks.rank(ascending=False)
    # compute average rank
    average_ranks = (
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    )

    return p_values, average_ranks, max_nb_datasets


def graph_ranks(
    avranks,
    names,
    p_values,
    cd=None,
    cdmethod=None,
    lowv=None,
    highv=None,
    width=200,
    textspace=1,
    reverse=False,
    filename=None,
    **kwargs,
):
    width = width
    textspace = float(textspace)
    """l is an array of array 
        [[......]
         [......]
         [......]]; 
    n is an integer"""

    # n th column
    def nth(l, n):
        n = lloc(l, n)
        # Return n th column
        return [a[n] for a in l]

    """l is an array of array 
        [[......]
         [......]
         [......]]; 
    n is an integer"""

    # return an integer, count from front or from back.
    def lloc(l, n):
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    # lr is an array of integers
    # Maximum range start from all zeros. Returns an iterable element of tuple.
    def mxrange(lr):
        # If nothing in the array
        if not len(lr):
            yield ()
        else:
            index = lr[0]
            # Check whether index is an integer.
            if isinstance(index, int):
                index = [index]
            # *index: index must be an iterable []
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    # Form a tuple, and generate an iterable value
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums
    # lowv: low value
    if lowv is None:
        """int(math.floor(min(ssums))): select the minimum value in ssums and take floor.
           Then compare with 1 to see which one is the minimum."""
        lowv = min(1, int(math.floor(min(ssums))))
    # highv: high value
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4
    # how many algorithms
    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    # Position of rank
    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        # Set up the format
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # set up the formats
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant + 2

    # matplotlib figure format setup
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1.0 / height
    wf = 1.0 / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Line plots
    def line(l, color="k", **kwargs):
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    # Add text to the plot
    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None

    # [lowv, highv], step size is 0.5
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        # If a is an integer
        if a == int(a):
            tick = bigtick
        # Plot a line
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    # Add text to the plot, only for integer value
    for a in range(lowv, highv + 1):
        text(
            rankpos(a),
            cline - tick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    # Format for the first half of algorithms
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=linewidth,
        )

        color = "k"
        text(
            textspace - 0.2,
            chei,
            filter_names(nnames[i]),
            color=color,
            ha="right",
            va="center",
            size=16,
        )
        # text(textspace - 0.2, chei, filter_names(name_mapping[nnames[i]] if nnames[i] in name_mapping.keys() else nnames[i]), color=color, ha="right", va="center", size=16)

    # Format for the second half of algorithms
    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=linewidth,
        )

        color = "k"
        text(
            textspace + scalewidth + 0.2,
            chei,
            filter_names(nnames[i]),
            color=color,
            ha="left",
            va="center",
            size=16,
        )
        # text(textspace + scalewidth + 0.2, chei, filter_names(name_mapping[nnames[i]] if nnames[i] in name_mapping.keys() else nnames[i]), color=color, ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line(
                [(rankpos(ssums[l]) - side, start), (rankpos(ssums[r]) + side, start)],
                linewidth=linewidth_sign,
            )
            start += height

    start = cline + 0.2
    side = -0.02
    height = 0.1

    # Generate cliques and plot a line to connect elements in cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    # Plot a line to connect elements in cliques
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        # Test
        # print("ssums[min_idx]: {}; ssums[max_idx]: {}".format(ssums[min_idx], ssums[max_idx]))
        line(
            [
                (rankpos(ssums[min_idx]) - side, start),
                (rankpos(ssums[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
        )
        start += height

    return fig, ax


def form_cliques(p_values, nnames):
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1
    g = networkx.Graph(g_data)

    # Test
    # print("p_values in form_cliques:\n{}".format(p_values))
    # print("g_data:\n{}".format(g_data))

    # Returns all maximal cliques in an undirected graph.
    return networkx.find_cliques(g)


def reshape_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Reshapes the DataFrame by setting 'Dataset' as the row index, 'Method' as columns,
    and the specified metric as values.

    Parameters:
    - df: pd.DataFrame (input data)
    - metric: str (column name of the desired metric)

    Returns:
    - reshaped_df: pd.DataFrame (reshaped data)
    """
    df = df.reset_index(inplace=False)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in the dataframe.")

    reshaped_df = df.pivot(index="Dataset", columns="Method", values=metric)
    return reshaped_df


def plot_cd_diagram(
    df: pd.DataFrame, metric: str, out_dir: Path, kind: str = "Overall"
) -> pd.DataFrame:
    eval_list = []
    if kind == "TS":
        df = df[df["Time Series"]]
        df = reshape_dataframe(df, metric)
    elif kind == "non-TS":
        df = df[~df["Time Series"]]
        df = reshape_dataframe(df, metric)
    else:
        df = reshape_dataframe(df, metric)
    if metric == "Comp Ratio":
        metric = "Comp Ratio"
        df = 1 / df
    for index, row in df.iterrows():
        for method in default_compression_method_order():
            dataset_name = row.name
            eval_list.append([method, dataset_name, row[method]])
    eval_df = pd.DataFrame(
        eval_list, columns=["classifier_name", "dataset_name", "accuracy"]
    )
    p_values, average_ranks, _ = Friedman_Nemenyi(df_perf=eval_df, alpha=0.05)
    ranking = average_ranks.keys().to_list()[::-1]
    print(f"{kind}_{metric}\n", average_ranks)

    fig, ax = graph_ranks(
        average_ranks.values,
        average_ranks.keys(),
        p_values,
        cd=None,
        reverse=True,
        width=5,
        textspace=1.5,
    )
    fig: plt.Figure
    ax: plt.Axes
    # ax.set_title("Critical Diagram ({}=0.05)".format(r'$\alpha$'),fontsize=20)
    if kind == "Overall":
        plt.title(
            "Critical Diagram ({}=0.05)\n on {}".format(r"$\alpha$", metric),
            fontsize=20,
        )
    else:
        plt.title(
            "Critical Diagram ({}=0.05)\n on {}/{}".format(r"$\alpha$", kind, metric),
            fontsize=20,
        )
    print(f"titles: {ax.title}")
    plt.tight_layout()
    plt.savefig(
        out_dir.joinpath(f"CD_{kind.replace(' ', '_')}_{metric.replace(' ', '_')}.png"),
        bbox_inches="tight",
    )
    # plt.close(fig)

    return average_ranks


def get_rank_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Computes the ranking of methods for a given metric across different datasets.

    Parameters:
    - df: pd.DataFrame - The input DataFrame containing the performance metrics.
    - metric: str - The metric to rank the methods on.

    Returns:
    - pd.DataFrame - A DataFrame where each row represents a dataset and columns are methods,
                     containing the ranking of each method for the given metric.
    """
    # Reshape the DataFrame to have 'Dataset' as index and 'Method' as columns
    df_pivot = df.pivot(index="Dataset", columns="Method", values=metric)

    # Determine the ranking direction (lower is better for "Comp Ratio", otherwise higher is better)
    ascending = metric == "Comp Ratio"

    # Compute rankings
    rank_df = df_pivot.rank(axis=1, ascending=ascending, method="average")

    return rank_df


def average_rankings(rank_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Averages multiple ranking DataFrames along the metric dimension.

    Parameters:
    - rank_dfs: list of pd.DataFrame - A list of ranking DataFrames obtained from `get_rank_df()`.

    Returns:
    - pd.DataFrame - A DataFrame where each row represents a dataset, columns are methods,
                     and values are the averaged rankings across different metrics.
    """
    # Concatenate along the new axis and compute the mean across metrics
    averaged_rank_df = pd.concat(rank_dfs, axis=0).groupby(level=0).mean()

    return averaged_rank_df


def average_rankings_across_datasets(averaged_rank_df: pd.DataFrame) -> pd.Series:
    """
    Averages the rankings across datasets to obtain a final ranking for each method.

    Parameters:
    - averaged_rank_df: pd.DataFrame - The DataFrame containing averaged rankings across metrics.

    Returns:
    - pd.Series - A Series where the index represents methods and values represent the final averaged ranking.
    """
    return averaged_rank_df.mean(axis=0)  # Compute the mean across datasets


def plot_rank_boxplot(rank_df: pd.DataFrame, output_path: Path, y_label: str = "Rank"):
    """
    Plots a boxplot of rankings for each method without averaging over datasets.

    Parameters:
    - rank_df: pd.DataFrame - DataFrame containing ranking values for each method across datasets.
    - output_path: Path - The file path where the plot will be saved.

    Returns:
    - None (Saves the plot to the specified output path)
    """
    # sort rank_df by the mean ranking
    rank_df = rank_df.reindex(rank_df.mean().sort_values().index, axis=1)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.reset_orig()

    # Create the boxplot
    sns.boxplot(
        data=rank_df,
        showfliers=False,  # Hide outliers
        meanprops=dict(color="k", linestyle="--"),
        showmeans=True,
        meanline=True,
        ax=ax,
    )

    # Customize labels
    ax.set_xticks(range(len(rank_df.columns)))
    ax.set_xticklabels(rank_df.columns, rotation=45, fontsize=12)
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(y_label, fontsize=12)
    fig.tight_layout()

    # Save the figure
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_cd.py <stats.json>")
        sys.exit(1)
    json_file_path = sys.argv[1]
    out_dir = Path(json_file_path).parent.joinpath(Path(json_file_path).stem)
    cd_out_dir = out_dir.joinpath("cd_diagrams")
    rank_boxplot_out_dir = out_dir.joinpath("rank_boxplots")
    out_dir = None
    os.makedirs(cd_out_dir, exist_ok=True)
    os.makedirs(rank_boxplot_out_dir, exist_ok=True)
    with open(json_file_path, "r") as f:
        all_output = cast(AllOutput, json.load(f))
    all_output: AllOutputHandler = AllOutputHandler(all_output)

    merged_df = load_merged_df("stats.json", all_output.averaged().map_to_stats())
    merged_df = merged_df.reset_index(inplace=False)
    merged_df = merged_df[~merged_df["Method"].isin(default_omit_methods())]
    merged_df = merged_df[~merged_df["Dataset"].isin(default_exclude_datasets)]

    metrics = ["Comp Ratio", "Comp Throughput", "DeComp Throughput"]
    filter_metrics = [
        *[f"Filter {q} Throughput" for q in ["Gt 90", "Gt 50", "Gt 10", "Eq", "Ne"]]
    ]

    for metric in metrics:
        plot_cd_diagram(merged_df, metric, cd_out_dir)
    for metric in filter_metrics:
        plot_cd_diagram(merged_df, metric, cd_out_dir)

    metrics_ranking_df = {}
    filter_metrics_ranking_df = {}

    for metric in metrics:
        rank_df = get_rank_df(merged_df, metric)
        metrics_ranking_df[metric] = rank_df
        plot_rank_boxplot(
            rank_df,
            rank_boxplot_out_dir.joinpath(
                f"{metric.replace(' ', '_')}_rank_boxplot.png"
            ),
        )
    for metric in filter_metrics:
        rank_df = get_rank_df(merged_df, metric)
        filter_metrics_ranking_df[metric] = rank_df
        plot_rank_boxplot(
            rank_df,
            rank_boxplot_out_dir.joinpath(
                f"{metric.replace(' ', '_')}_rank_boxplot.png"
            ),
        )

    compression_performance_average_ranking = average_rankings(
        metrics_ranking_df.values()
    )
    plot_rank_boxplot(
        compression_performance_average_ranking,
        rank_boxplot_out_dir.joinpath(
            "compression_performance_average_rank_boxplot.png"
        ),
    )

    filter_average_ranking = average_rankings(filter_metrics_ranking_df.values())
    averaged_ranking = average_rankings(
        [*metrics_ranking_df.values(), filter_average_ranking]
    )

    plot_rank_boxplot(
        averaged_ranking, rank_boxplot_out_dir.joinpath("average_rank_boxplot.png")
    )


if __name__ == "__main__":
    main()
