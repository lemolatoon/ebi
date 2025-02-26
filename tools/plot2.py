import copy
import math
from statistics import fmean
import sys
import os
import json
from typing import Dict, List, Optional, TypedDict, cast
from pathlib import Path
import pandas as pd

from plot import (
    AllOutput,
    AllOutputHandler,
    AveragedAllOutput,
    AveragedAllOutputHandler,
    AveragedAllOutputInner,
    AveragedStats,
    get_color,
    get_color_exe,
    segment_mapping,
    mapped_processing_types,
    compression_methods,
)

from common import default_compression_method_order, default_omit_methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


def init_plt_font():
    # del matplotlib.font_manager.weight_dict['roman']
    # matplotlib.font_manager._rebuild()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.titlesize": 20,
        }
    )


class Stats(TypedDict):
    # Decimal precision stats
    decimal_precision_max: int
    decimal_precision_min: int
    decimal_precision_avg: float
    decimal_precision_std: float

    # Value stats
    value_max: float
    value_min: float
    value_avg: float
    value_std: float

    # Average leading/trailing zeros in XOR of consecutive values
    xor_leading_zeros_avg: float
    xor_trailing_zeros_avg: float

    num_values: int


file_to_dataset = {
    "blockchain_output_total_usd": "Blockchain-tr",
    "influxdb_bird_migration_lat_lon": "Bird-migration",
    "inforce_stocks_de": "Stocks-DE",
    "inforce_stocks_uk": "Stocks-UK",
    "inforce_stocks_usa": "Stocks-USA",
    "kaggle_btc_usd": "Bitcoin-price",
    "kaggle_poi_lat": "POI-lat",
    "kaggle_poi_lon": "POI-lon",
    "kaggle_ssd_hdd_capacity": "SD-bench",
    "meteoblue_temperature": "Basel-temp",
    "meteoblue_wind_speed": "Basel-wind",
    "neon_2d_wind_speed_and_direction": "Wind-dir",
    "neon_barometric_pressure": "Air-Pressure",
    "neon_dust_and_particulate": "PM10-dust",
    "neon_ir_biological_temperature": "IR-bio-temp",
    "neon_relative_humidity_on_buoy": "Dew-Point-Temp",
    "pbi_arade_4": "Arade/4",
    "pbi_cms_1": "CMS/1",
    "pbi_cms_25": "CMS/25",
    "pbi_cms_9": "CMS/9",
    "pbi_gov_10": "Gov/10",
    "pbi_gov_26": "Gov/26",
    "pbi_gov_30": "Gov/30",
    "pbi_gov_31": "Gov/31",
    "pbi_gov_40": "Gov/40",
    "pbi_medicare3_1": "Medicare/1",
    "pbi_medicare3_9": "Medicare/9",
    "pbi_nyc_29": "NYC/29",
    "udayton_city_temperature": "City-Temp",
    "wfp_food_price": "Food-prices",
}

column_to_renamed_column = {
    "decimal_precision_max": "Max DP",
    "decimal_precision_min": "Min DP",
    "decimal_precision_avg": "Avg DP",
    "decimal_precision_std": "Std DP",
    "value_max": "Max Value",
    "value_min": "Min Value",
    "value_avg": "Avg Value",
    "value_std": "Std Value",
    "xor_leading_zeros_avg": "Avg Leading Zeros",
    "xor_trailing_zeros_avg": "Avg Trailing Zeros",
    "num_values": "Num Values",
    "exponent_max": "Max Exponent",
    "exponent_min": "Min Exponent",
    "exponent_avg": "Avg Exponent",
    "exponent_std": "Std Exponent",
    "non_unique_value_ratio": "Non-Unique Value Ratio",
}

dataset_to_semantics = {
    "Air-Pressure": "Barometric Pressure (kPa)",
    "Basel-temp": "Temperature (C°)",
    "Basel-wind": "Wind Speed (Km/h)",
    "Bird-migration": "Coordinates (lat, lon)",
    "Bitcoin-price": "Exchange Rate (BTC-USD)",
    "City-Temp": "Temperature (F°)",
    "Dew-Point-Temp": "Temperature (C°)",
    "IR-bio-temp": "Temperature (C°)",
    "PM10-dust": "Dust content in air (mg/m3)",
    "Stocks-DE": "Monetary (Stocks)",
    "Stocks-UK": "Monetary (Stocks)",
    "Stocks-USA": "Monetary (Stocks)",
    "Wind-dir": "Angle Degree (0°-360°)",
    "Arade/4": "Energy",
    "Blockchain-tr": "Monetary (BTC)",
    "CMS/1": "Monetary Avg. (USD)",
    "CMS/25": "Monetary Std. Dev. (USD)",
    "CMS/9": "Discrete Count",
    "Food-prices": "Monetary (USD)",
    "Gov/10": "Monetary (USD)",
    "Gov/26": "Monetary (USD)",
    "Gov/30": "Monetary (USD)",
    "Gov/31": "Monetary (USD)",
    "Gov/40": "Monetary (USD)",
    "Medicare/1": "Monetary Avg. (USD)",
    "Medicare/9": "Discrete Count",
    "NYC/29": "Coordinates (lon)",
    "POI-lat": "Coordinates (lat, in radians)",
    "POI-lon": "Coordinates (lon, in radians)",
    "SD-bench": "Storage Capacity (GB)",
}

dataset_to_sources = {
    "Air-Pressure": "NEON",
    "Basel-temp": "meteoblue",
    "Basel-wind": "meteoblue",
    "Bird-migration": "InfluxDB",
    "Bitcoin-price": "Kaggle",
    "City-Temp": "Udayton",
    "Dew-Point-Temp": "NEON",
    "IR-bio-temp": "NEON",
    "PM10-dust": "NEON",
    "Stocks-DE": "Inforce",
    "Stocks-UK": "Inforce",
    "Stocks-USA": "Inforce",
    "Wind-dir": "NEON",
    "Arade/4": "PBI Bench.",
    "Blockchain-tr": "Blockchain",
    "CMS/1": "PBI Bench.",
    "CMS/25": "PBI Bench.",
    "CMS/9": "PBI Bench.",
    "Food-prices": "WFP",
    "Gov/10": "PBI Bench.",
    "Gov/26": "PBI Bench.",
    "Gov/30": "PBI Bench.",
    "Gov/31": "PBI Bench.",
    "Gov/40": "PBI Bench.",
    "Medicare/1": "PBI Bench.",
    "Medicare/9": "PBI Bench.",
    "NYC/29": "PBI Bench.",
    "POI-lat": "Kaggle",
    "POI-lon": "Kaggle",
    "SD-bench": "Kaggle",
}

dataset_to_tex_name = {
    "Air-Pressure": "neon2021airpressure",
    "Basel-temp": "meteoblue2025",
    "Basel-wind": "meteoblue2025",
    "Bird-migration": "influxdata2025sampledata",
    "Bitcoin-price": "kaggle2020bitcoin",
    "City-Temp": "kaggle2019citytemp",
    "Dew-Point-Temp": "neon2021dewpointtemp",
    "IR-bio-temp": "neon2021irbiotemp",
    "PM10-dust": "neon2021dustinair",
    "Stocks-DE": "spring2020inforce",
    "Stocks-UK": "spring2020inforce",
    "Stocks-USA": "spring2020inforce",
    "Wind-dir": "neon2025winddir",
    "Arade/4": "pbibench",
    "Blockchain-tr": "blockchair_bitcoin_2025",
    "CMS/1": "pbibench",
    "CMS/25": "pbibench",
    "CMS/9": "pbibench",
    "Food-prices": "wfp2021foodprices",
    "Gov/10": "pbibench",
    "Gov/26": "pbibench",
    "Gov/30": "pbibench",
    "Gov/31": "pbibench",
    "Gov/40": "pbibench",
    "Medicare/1": "pbibench",
    "Medicare/9": "pbibench",
    "NYC/29": "pbibench",
    "POI-lat": "kaggle2019poi",
    "POI-lon": "kaggle2019poi",
    "SD-bench": "kaggle2022ssdhdd",
}

dataset_is_time_series = {
    "Air-Pressure": True,
    "Basel-temp": True,
    "Basel-wind": True,
    "Bird-migration": True,
    "Bitcoin-price": True,
    "City-Temp": True,
    "Dew-Point-Temp": True,
    "IR-bio-temp": True,
    "PM10-dust": True,
    "Stocks-DE": True,
    "Stocks-UK": True,
    "Stocks-USA": True,
    "Wind-dir": True,
    "Arade/4": False,
    "Blockchain-tr": False,
    "CMS/1": False,
    "CMS/25": False,
    "CMS/9": False,
    "Food-prices": False,
    "Gov/10": False,
    "Gov/26": False,
    "Gov/30": False,
    "Gov/31": False,
    "Gov/40": False,
    "Medicare/1": False,
    "Medicare/9": False,
    "NYC/29": False,
    "POI-lat": False,
    "POI-lon": False,
    "SD-bench": False,
}

datasets_inorder = [
    "Air-Pressure",
    "Basel-temp",
    "Basel-wind",
    "Bird-migration",
    "Bitcoin-price",
    "City-Temp",
    "Dew-Point-Temp",
    "IR-bio-temp",
    "PM10-dust",
    "Stocks-DE",
    "Stocks-UK",
    "Stocks-USA",
    "Wind-dir",
    "Arade/4",
    "Blockchain-tr",
    "CMS/1",
    "CMS/25",
    "CMS/9",
    "Food-prices",
    "Gov/10",
    "Gov/26",
    "Gov/30",
    "Gov/31",
    "Gov/40",
    "Medicare/1",
    "Medicare/9",
    "NYC/29",
    "POI-lat",
    "POI-lon",
    "SD-bench",
]


def load_stats(stats_path: Path) -> Stats:
    with open(stats_path, "r") as f:
        return cast(Dict[str, Stats], json.load(f))


def load_stats_df(stats_path: Path) -> pd.DataFrame:
    stats = load_stats(stats_path)
    df = pd.DataFrame(stats).T
    # round fp values at 2 decimal places
    df = df.round(2)
    # rename rows to dataset names
    df = df.rename(index=file_to_dataset)
    df = df.rename(columns=column_to_renamed_column)
    # cast Max DP, Min DP, Num Values to int
    df["Max DP"] = df["Max DP"].astype(int)
    df["Min DP"] = df["Min DP"].astype(int)
    df["Num Values"] = df["Num Values"].astype(int)
    # add semantics column
    df["Semantics"] = df.index.map(lambda x: dataset_to_semantics[x])
    # add Dataset column
    df["Dataset"] = df.index
    # add Source column
    df["Source"] = df.index.map(lambda x: dataset_to_sources[x])
    # add tex name column
    df["Tex Name"] = df.index.map(lambda x: dataset_to_tex_name[x])
    # add Time Series column
    df["Time Series"] = df.index.map(lambda x: dataset_is_time_series[x])
    return df


def create_data_existence_table(
    averaged_all_output: AveragedAllOutputHandler,
) -> pd.DataFrame:
    methods = averaged_all_output.methods()
    datasets_set = set()
    for method in methods:
        for ds in averaged_all_output.datasets_in_method(method):
            datasets_set.add(ds)
    datasets = sorted(list(datasets_set))

    table = []
    for ds in datasets:
        row = {}
        for method in methods:
            method_dict = averaged_all_output.data[method]
            if method_dict is not None and ds in method_dict:
                val = method_dict[ds]
                row[method] = True if val is not None else False
            else:
                row[method] = False
        table.append(row)

    df = pd.DataFrame(table, index=datasets, columns=methods)
    # rename dataset names
    df = df.rename(index=file_to_dataset)
    return df


def create_average_stats_table(
    averaged_all_output: AveragedAllOutputHandler,
    skip_none_containing_dataset=False,
    skip_none_containing_method=False,
) -> pd.DataFrame:
    df = pd.DataFrame(
        averaged_all_output.compute_average_across_dataset(
            skip_none_containing_dataset=skip_none_containing_dataset,
            skip_none_containing_method=skip_none_containing_method,
        )
    )
    df = df.T
    # rename columns
    column_name_mapping = {
        "compression_ratio": "Comp Ratio",
        "compression_throughput": "Comp Throughput",
        "decompression_throughput": "DeComp Throughput",
        "filter_gt_10th_percentile_throughput": "Filter Gt 10 Throughput",
        "filter_gt_50th_percentile_throughput": "Filter Gt 50 Throughput",
        "filter_gt_90th_percentile_throughput": "Filter Gt 90 Throughput",
        "filter_eq_throughput": "Filter Eq Throughput",
        "filter_ne_throughput": "Filter Ne Throughput",
        "max_throughput": "Max Throughput",
        "sum_throughput": "Sum Throughput",
        "compression_exec_time_ratios": "Comp Exec Time Ratios",
        "decompression_exec_time_ratios": "DeComp Exec Time Ratios",
        "filter_gt_10th_percentile_exec_time_ratios": "Filter Gt 10 Exec Time Ratios",
        "filter_gt_50th_percentile_exec_time_ratios": "Filter Gt 50 Exec Time Ratios",
        "filter_gt_90th_percentile_exec_time_ratios": "Filter Gt 90 Exec Time Ratios",
        "compression_execution_time": "Comp Exec Time",
        "decompression_execution_time": "DeComp Exec Time",
        "filter_gt_10th_percentile_execution_time": "Filter Gt 10 Exec Time",
        "filter_gt_50th_percentile_execution_time": "Filter Gt 50 Exec Time",
        "filter_gt_90th_percentile_execution_time": "Filter Gt 90 Exec Time",
    }
    df = df.rename(columns=column_name_mapping)
    return df


def create_normalized_exec_time_df(
    averaged_all_output: AveragedAllOutputHandler,
    metric: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        averaged_all_output.compute_normalized_segmented_execution_times(
            metric=metric,
        )
    )
    df.rename(index=file_to_dataset, inplace=True)
    df = df.T
    return df


default_exclude_datasets = ["CMS/25", "NYC/29", "POI-lat", "POI-lon"]


def create_normalized_exec_time_df_all(
    averaged_all_output: AveragedAllOutputHandler,
) -> pd.DataFrame:
    dfs = []
    metrics = ["compress", "materialize", "filter_gt_90th_percentile"]
    for metric in metrics:
        df = create_normalized_exec_time_df(averaged_all_output, metric)
        # drop columns based on 'default_exclude_datasets'
        df = df.drop(default_exclude_datasets, axis=1)
        # pick up the sample cell
        execution_times_key = df.iloc[0]["City-Temp"].keys()

        # compute average across datasets
        def applyf(row):
            return {k: fmean([v[k] for v in row]) for k in execution_times_key}

        dfs.append(df.apply(applyf, axis=1))
    df = pd.concat(
        dfs,
        axis=1,
        keys=["Comp Exec Time", "DeComp Exec Time", "Filter Gt 90 Exec Time"],
    )
    return df


def create_normalized_exec_time_df_all_for_dataset(
    averaged_all_output: AveragedAllOutputHandler,
    dataset: str,
) -> pd.DataFrame:
    dfs = []
    metrics = ["compress", "materialize", "filter_gt_90th_percentile"]
    for metric in metrics:
        df = create_normalized_exec_time_df(averaged_all_output, metric)
        # drop columns based on 'default_exclude_datasets'
        df = df.drop(default_exclude_datasets, axis=1)
        # compute average across datasets
        dfs.append(df[dataset])
    df = pd.concat(
        dfs,
        axis=1,
        keys=["Comp Exec Time", "DeComp Exec Time", "Filter Gt 90 Exec Time"],
    )
    return df


def create_all_df(
    average_stats_all: Dict[str, Dict[str, Optional[AveragedStats]]],
) -> pd.DataFrame:
    # flatten the dictionary
    def flatten_averaged_stats(
        stats: AveragedStats, method: str, dataset: str
    ) -> Dict[str, Optional[float]]:
        return {
            "Method": method,
            "Dataset": dataset,
            "Comp Ratio": stats["compression_ratio"],
            "Comp Throughput": stats["compression_throughput"],
            "DeComp Throughput": stats["decompression_throughput"],
            "Filter Gt 10 Throughput": stats["filter_gt_10th_percentile_throughput"],
            "Filter Gt 50 Throughput": stats["filter_gt_50th_percentile_throughput"],
            "Filter Gt 90 Throughput": stats["filter_gt_90th_percentile_throughput"],
            "Filter Eq Throughput": stats["filter_eq_throughput"],
            "Filter Ne Throughput": stats["filter_ne_throughput"],
            "Max Throughput": stats["max_throughput"],
            "Sum Throughput": stats["sum_throughput"],
        }

    rows = []
    for method, ds_map in average_stats_all.items():
        if ds_map is None:
            continue
        for dataset, stats in ds_map.items():
            if stats is not None:
                row = flatten_averaged_stats(
                    stats, method, file_to_dataset.get(dataset, dataset)
                )
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index(["Method", "Dataset"], inplace=True)
    return df


def load_merged_df(stats_path: Path, average_stats_all) -> pd.DataFrame:
    stats_df = load_stats_df(stats_path)
    all_df = create_all_df(average_stats_all)
    all_df.reset_index(inplace=True)
    all_df.set_index("Dataset", inplace=True)
    all_df = all_df.copy()[
        [
            "Method",
            "Comp Ratio",
            *[
                f"Filter {i} Throughput"
                for i in ["Gt 10", "Gt 50", "Gt 90", "Eq", "Ne"]
            ],
            *[f"{n} Throughput" for n in ["Comp", "DeComp", "Max", "Sum"]],
        ]
    ]
    merged = pd.merge(all_df, stats_df, on="Dataset", how="inner")

    merged.set_index(["Method", "Dataset"], inplace=True)
    return merged


def plot_scatter(df_merged: pd.DataFrame, output_path: Path):
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.titlesize": 20,
        }
    )
    # Drop index
    df_merged = df_merged.reset_index(inplace=False)
    df_all_methods = df_merged[~df_merged["Method"].isin(default_omit_methods())]
    # Calculate the average Comp Ratio for each method
    method_avg_comp_ratio = df_all_methods.groupby("Method")["Comp Ratio"].mean()
    # Compute the inverse of Comp Ratio and format it as "x.x x"
    method_avg_inv_comp_ratio = 1 / method_avg_comp_ratio
    formatted_labels = {
        method: f"{method} ({method_avg_inv_comp_ratio[method]:.1f}x)"
        for method in method_avg_inv_comp_ratio.index
    }

    # Sort methods by the inverse Comp Ratio in descending order
    sorted_methods = method_avg_inv_comp_ratio.sort_values(ascending=False).index

    # Recreate the scatter plot with sorted legend and formatted labels
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    scatter = sns.scatterplot(
        data=df_all_methods,
        x="Comp Throughput",
        y="DeComp Throughput",
        hue="Method",
        palette="tab20",
        alpha=0.7,
        edgecolor="k",
        s=100,  # Increase marker size
        marker="o",  # Use circular markers
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Avg. Comp Throughput (GB/s) (Log Scale)")
    plt.ylabel("Avg. DeComp Throughput (GB/s) (Log Scale)")
    # plt.title("Scatter Plot of Comp Throughput vs DeComp Throughput (Log Scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Modify legend with sorted labels and formatted inverse Comp Ratio values
    handles, labels = scatter.get_legend_handles_labels()
    sorted_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=8, color=get_color(method)
        )
        for i, method in enumerate(sorted_methods)
    ]
    sorted_labels = [formatted_labels[method] for method in sorted_methods]

    plt.legend(
        sorted_handles,
        sorted_labels,
        title="Method (Inv Comp Ratio)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
    )

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()


default_metrics = [
    "Comp Ratio",
    "Filter Gt 10 Throughput",
    "Filter Gt 50 Throughput",
    "Filter Gt 90 Throughput",
    "Filter Eq Throughput",
    "Filter Ne Throughput",
    "Comp Throughput",
    "DeComp Throughput",
    "Sum Throughput",
    "Max Throughput",
]


def plot_radar_chart(
    df: pd.DataFrame, output_path: Path, metrics: List[str] = default_metrics
):
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.titlesize": 20,
        }
    )
    base_fontsize = 15
    # Drop index
    df = df.reset_index(inplace=False)
    df_filtered = df[~df["Method"].isin(default_omit_methods())]

    grouped_df = df_filtered.groupby("Method")[metrics].mean()

    if "Comp Ratio" in metrics:
        grouped_df["Comp Ratio"] = 1 / grouped_df["Comp Ratio"]

    normalized_data = grouped_df.div(grouped_df.max())

    formatted_metrics = [
        m.replace("Throughput", "\nThroughput").replace(
            "Comp Ratio", "    Inv \n    Comp Ratio"
        )
        for m in metrics
    ]

    method_labels = default_compression_method_order()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for method in method_labels:
        values = normalized_data.loc[method].tolist()
        values += values[:1]
        ax.plot(angles, values, label=method, linewidth=1.5, color=get_color(method))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        formatted_metrics, fontsize=base_fontsize, fontweight="bold", ha="center"
    )

    # Adjust the legend to not be cut off
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        fontsize=base_fontsize,
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()


def plot_bar_chart(
    df: pd.DataFrame,
    ax: plt.Axes,
    metric: str,
    xor_series_df: pd.DataFrame | None = None,
    ucr2018_series_df: pd.DataFrame | None = None,
    embeddings_series_df: pd.DataFrame | None = None,
):
    """
    Generate a bar chart and plot it on the given axis.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the necessary columns.
    ax : matplotlib.axes.Axes
        The axis on which to plot the bar chart.
    metric : str
        The metric to visualize.
    """
    default_fontsize = 12
    print(
        f"Plotting {metric} bar chart: xor_series_df={xor_series_df}, ucr2018_series_df={ucr2018_series_df}, embeddings_series_df={embeddings_series_df}"
    )

    # Compute mean values for Time-Series, Non-Time-Series, and Overall
    time_series_df = df[df["DatasetType"] == "Time-Series"]
    non_time_series_df = df[df["DatasetType"] == "Non-Time-Series"]

    time_series_mean = (
        time_series_df.groupby("Method")[metric].mean().rename("Time-Series")
    )
    non_time_series_mean = (
        non_time_series_df.groupby("Method")[metric].mean().rename("Non-Time-Series")
    )
    overall_mean = df.groupby("Method")[metric].mean().rename("TS+Non-TS")

    xor_mean = None
    ucr2018_mean = None
    embeddings_mean = None
    if xor_series_df is not None:
        xor_mean = (
            xor_series_df.groupby("Method")[metric].mean().rename("XOR Synthetic")
        )
    if ucr2018_series_df is not None:
        ucr2018_mean = (
            ucr2018_series_df.groupby("Method")[metric].mean().rename("UCR2018")
        )
    if embeddings_series_df is not None:
        embeddings_mean = (
            embeddings_series_df.groupby("Method")[metric].mean().rename("Embeddings")
        )

    # Sort by Method at the order of `default_compression_method_order`
    dfs = []
    for df in [
        time_series_mean,
        non_time_series_mean,
        overall_mean,
        xor_mean,
        ucr2018_mean,
        embeddings_mean,
    ]:
        if df is None:
            continue
        df: pd.Series
        df.index = pd.Categorical(
            df.index, categories=default_compression_method_order(), ordered=True
        )
        df.sort_index(inplace=True)
        dfs.append(df)

    # Combine results into a single DataFrame
    plot_df = pd.concat(dfs, axis=1)

    small_default = 10

    # Plot bar chart
    plot_df.plot(
        kind="bar",
        ax=ax,
        color=plt.get_cmap("tab10").colors,
        fontsize=small_default,
        legend=False,
    )

    # Set labels and legend
    ax.set_xlabel("Method", fontsize=small_default)

    # Modify y-axis label to indicate GB/s for throughput metrics
    # if "Comp Throughput" == metric:
    #    ax.set_ylabel("Compression Throughput (GB/s)", fontsize=default_fontsize)
    # elif "DeComp Throughput" == metric:
    #    ax.set_ylabel("Decompression Throughput (GB/s)", fontsize=default_fontsize)
    # elif "Filter Gt 90 Throughput" == metric:
    #    ax.set_ylabel("Filter Greater than 90th Percentile Throughput (GB/s)", fontsize=default_fontsize)
    if "Throughput" in metric:
        ax.set_ylabel(f"{metric} (GB/s)", fontsize=default_fontsize)
    elif "Comp Ratio" == metric:
        ax.set_ylabel("Comp Ratio (Smaller is Better)", fontsize=default_fontsize)
    else:
        ax.set_ylabel(metric, fontsize=default_fontsize)
    # goes to the bottom a bit
    ax.yaxis.set_label_coords(-0.075, 0.45)

    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=small_default
    )

    # ax.legend(fontsize=small_default, frameon=True)
    ax.tick_params(axis="both", labelsize=small_default)


def make_subplots_all(
    df: pd.DataFrame,
    output_path: Path,
    metrics: List[str],
    excluded_datasets: List[str] = default_exclude_datasets,
    xor_series_df: pd.DataFrame | None = None,
    ucr2018_series_df: pd.DataFrame | None = None,
    embeddings_series_df: pd.DataFrame | None = None,
):
    """
    Create subplots for all metrics arranged in 3 columns and display as a single figure.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the necessary columns.
    metrics : list
        List of metrics to visualize.
    """
    separated_dir = output_path.parent.joinpath(output_path.stem)
    os.makedirs(separated_dir, exist_ok=True)

    # Modify df to add DatasetType, exclude unnecessary methods, and remove specific datasets
    df = df.copy()
    df.reset_index(inplace=True)
    df["DatasetType"] = df["Time Series"].map(
        {True: "Time-Series", False: "Non-Time-Series"}
    )
    df = df[~df["Method"].isin(default_omit_methods())]
    df = df[~df["Dataset"].isin(excluded_datasets)]

    num_metrics = len(metrics)
    num_cols = 3  # Set the number of columns
    num_rows = math.ceil(num_metrics / num_cols)  # Compute required rows

    axes: np.ndarray[plt.Axes]
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()  # Convert to 1D array for easy iteration

    ax_for_legend: plt.Axes = None
    for ax, metric in zip(axes, metrics):
        ax: plt.Axes
        if metric in ["Comp Ratio", "DeComp Throughput", "Comp Throughput"]:
            plot_bar_chart(
                df, ax, metric, xor_series_df, ucr2018_series_df, embeddings_series_df
            )
            ax_for_legend = ax
        else:
            plot_bar_chart(df, ax, metric)

        # Save each subplot separately
        metric_filename = separated_dir / f"{metric.replace(' ', '_')}.png"
        single_fig, single_ax = plt.subplots(1, 1, figsize=(7, 3))
        if metric in ["Comp Ratio", "DeComp Throughput", "Comp Throughput"]:
            plot_bar_chart(
                df,
                single_ax,
                metric,
                xor_series_df,
                ucr2018_series_df,
                embeddings_series_df,
            )
        else:
            plot_bar_chart(df, single_ax, metric)

        single_fig.tight_layout()
        single_fig.savefig(metric_filename, format="png", dpi=300)

    handles, labels = ax_for_legend.get_legend_handles_labels()

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    # Add a single legend for the entire figure
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3
        + len(
            list(
                filter(
                    lambda x: x is not None,
                    [xor_series_df, ucr2018_series_df, embeddings_series_df],
                )
            )
        ),
        fontsize=12,
        frameon=True,
    )

    # save only the legend
    # Add a single legend for the entire figure
    n_col = 3 + len(
        list(
            filter(
                lambda x: x is not None,
                [xor_series_df, ucr2018_series_df, embeddings_series_df],
            )
        )
    )
    legend_fig, legend_ax = plt.subplots(figsize=(1.8 * n_col, 1))
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=n_col,
        fontsize=12,
        frameon=True,
    )
    legend_ax.axis("off")  # Hide axes for legend-only figure
    legend_filename = separated_dir / "legend.png"
    legend_fig.savefig(legend_filename, format="png", dpi=300)
    plt.close(legend_fig)  # Close the legend figure

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, format="png", dpi=300)


def default_processing_types_order():
    pts = list(mapped_processing_types)
    pts.remove("IO Read")
    pts.remove("IO Write")
    pts.remove("Other Processing")
    # push IO Read and IO Write to the front
    pts.insert(0, "IO Read")
    pts.insert(1, "IO Write")
    # push Other Processing to the end
    pts.append("Other Processing")
    return pts


hatch_map_exe: dict[str, str] = {}


def get_hatch_exe(label: str) -> str:
    global hatch_map_exe
    plt.rcParams["hatch.linewidth"] = 0.3
    hatches = [
        "O.",
        "//",
        "++",
        "//",
        "\\\\",
        "||",
        "--",
        "++",
        "xx",
        "oo",
        "OO",
        "..",
        "**",
    ]
    hatches.reverse()

    if label in hatch_map_exe:
        return hatch_map_exe[label]

    next_index = len(hatch_map_exe) % len(hatches)
    hatch_map_exe[label] = hatches[next_index]

    return hatch_map_exe[label]


def plot_absolute_stacked_execution_times_for_methods(
    ax: plt.Axes,
    df: pd.DataFrame,
    throughput_column: str,
    ratio_column: str,
    y_label: str,
    segment_mapping: Dict[str, Dict[str, List[str]]],
    processing_types_order: List[str] = default_processing_types_order(),
    compression_methods_order: List[str] = default_compression_method_order(),
):
    """
    Plots a stacked bar chart of absolute execution times per compression method.

    Parameters:
    ax (plt.Axes): The matplotlib axis to draw the plot on.
    df (pd.DataFrame): The dataframe containing execution times and execution time ratios.
    time_column (str): The column name containing absolute execution times.
    ratio_column (str): The column name containing execution time ratios as dictionaries.
    segment_mapping (dict): A dictionary mapping compression methods to their execution time components.
    y_label (str): Label for the Y-axis.
    """
    default_fontsize = 12

    patched_segment_mapping = copy.deepcopy(segment_mapping)

    # Compute absolute execution times based on ratios
    times_by_method = {}

    for method, row in df.iterrows():
        throughput = row[throughput_column]
        normalized_exec_time = row[ratio_column]  # Convert string dict to actual dict

        method_times = {}
        for key, ratio in normalized_exec_time.items():
            if ratio > 0:
                mapped_keys = patched_segment_mapping.get(method, {}).get(key, [key])
                label = "+".join(mapped_keys)
                abs_time = ratio / throughput
                if abs_time > 0:
                    method_times[label] = method_times.get(label, 0) + abs_time

        if method_times:
            times_by_method[method] = method_times

    # Identify unique processing types for consistent ordering
    valid_processing_types = []
    # for method_data in times_by_method.values():
    #     valid_processing_types.append(method_data.keys())
    for processing_type in processing_types_order:
        if any(
            processing_type in times_by_method[method]
            for method in compression_methods_order
        ):
            valid_processing_types.append(processing_type)

    # Stacked bar chart setup
    bar_width = 0.5
    index = np.arange(len(times_by_method))
    bottom = np.zeros(len(times_by_method))

    # Plot each processing type with consistent colors
    for processing_type in valid_processing_types:
        values = [
            times_by_method[method].get(processing_type, 0)
            for method in compression_methods_order
        ]
        ax.bar(
            index,
            values,
            bar_width,
            label=processing_type,
            bottom=bottom,
            color=get_color_exe(processing_type),  # Ensure color consistency
            hatch=get_hatch_exe(processing_type),
            alpha=0.8,
        )
        bottom += np.array(values)

    # Configure axis labels
    ax.set_ylabel(y_label, fontsize=default_fontsize)
    ax.set_xticks(index)
    ax.set_xticklabels(compression_methods_order, rotation=45, ha="right")
    # ax.set_title(f"Execution Times for {dataset_name} by Compression Method")


def make_all_plots_for_stacked(all_df: pd.DataFrame, output_path: Path):
    separated_dir = output_path.parent.joinpath(output_path.stem)
    os.makedirs(separated_dir, exist_ok=True)

    all_df = all_df[~all_df.index.isin(default_omit_methods())]
    columns = [
        ("Comp Exec Time", "Comp Execution Time (s/GB)"),
        ("DeComp Exec Time", "DeComp Execution Time (s/GB)"),
        ("Filter Gt 90 Exec Time", "Filter Execution Time (s/GB)"),
    ]
    columns = [
        ("Comp Throughput", "Comp Exec Time Ratios", "Comp Execution Time (s/GB)"),
        (
            "DeComp Throughput",
            "DeComp Exec Time Ratios",
            "DeComp Execution Time (s/GB)",
        ),
        (
            "Filter Gt 90 Throughput",
            "Filter Gt 90 Exec Time Ratios",
            "Filter Execution Time (s/GB)",
        ),
    ]

    # Create figure and axes (3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Generate plots and remove individual legends
    for ax, (throughput_column, ratio_col, dataset_name) in zip(axes, columns):
        plot_absolute_stacked_execution_times_for_methods(
            ax, all_df, throughput_column, ratio_col, dataset_name, segment_mapping
        )
        ax: plt.Axes
        ax.legend().set_visible(False)  # Remove individual legends

        # Add a single legend for all subplots below the figure
        single_fig, single_ax = plt.subplots(1, 1, figsize=(6, 4))
        single_filename = separated_dir / f"{ratio_col.replace(' ', '_')}.png"
        plot_absolute_stacked_execution_times_for_methods(
            single_ax,
            all_df,
            throughput_column,
            ratio_col,
            dataset_name,
            segment_mapping,
        )
        single_fig.tight_layout()
        single_fig.savefig(single_filename, format="png", dpi=300)

    # Collect all legend handles and labels from subplots
    lines_labels = [cast(plt.Axes, ax).get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # Remove duplicate legend labels while keeping order
    unique_labels = {}
    for handle, label in zip(lines, labels):
        if label not in unique_labels:
            unique_labels[label] = handle

    def label_priority(item):
        label, handle = item
        if label == "IO Read":
            return 0
        if label == "IO Write":
            return 1
        if label == "Other Processing":
            return 100
        if "+" in label:
            return 3
        return 2

    unique_labels_sorted = sorted(unique_labels.items(), key=label_priority)
    handles = [handle for label, handle in unique_labels_sorted]
    labels = [label for label, handle in unique_labels_sorted]
    # Add a single legend for all subplots below the figure
    # fig.legend(unique_labels.values(), unique_labels.keys(), title="Processing Types", loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.15), frameon=True)
    fig.legend(
        handles,
        labels,
        title="Processing Types",
        loc="lower center",
        ncol=5,
        frameon=True,
    )

    # save only the legend
    legend_fig, legend_ax = plt.subplots(figsize=(3.5 * 5, 2))
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=5,
        # fontsize=12,
        frameon=True,
    )
    legend_ax.axis("off")  # Hide axes for legend-only figure
    legend_filename = separated_dir / "legend.png"
    legend_fig.savefig(legend_filename, format="png", dpi=300)
    plt.close(legend_fig)  # Close the legend figure

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.2, 1, 1])  # Adjust layout to fit legend
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_table(
    df_merged: pd.DataFrame, column_name: str, factor=1, small_is_better=True
):
    df_merged = df_merged.reset_index(inplace=False)
    df_merged[column_name] = df_merged[column_name] * factor
    df = df_merged[~df_merged["Method"].isin(default_omit_methods())]
    df_filtered = df[["Method", "Dataset", column_name, "Time Series"]]

    table = df_filtered.pivot(index="Dataset", columns="Method", values=column_name)

    time_series_avg = (
        df_filtered[df_filtered["Time Series"]].groupby("Method")[column_name].mean()
    )
    non_time_series_avg = (
        df_filtered[~df_filtered["Time Series"]].groupby("Method")[column_name].mean()
    )
    overall_avg = df_filtered.groupby("Method")[column_name].mean()

    table.loc["Time Series AVG"] = time_series_avg
    table.loc["Non-Time Series AVG"] = non_time_series_avg
    table.loc["Overall AVG"] = overall_avg

    best_values = table.min(axis=1)
    if not small_is_better:
        best_values = table.max(axis=1)

    latex_table = "\\begin{tabular}{l" + "c" * len(table.columns) + "}\n\\toprule\n"
    latex_table += "Dataset & " + " & ".join(table.columns) + " \\\\\n\\midrule\n"

    time_series_datasets = df_filtered[df_filtered["Time Series"]]["Dataset"].unique()
    non_time_series_datasets = df_filtered[~df_filtered["Time Series"]][
        "Dataset"
    ].unique()

    time_series_avg_inserted = False
    for dataset in datasets_inorder:
        if dataset in time_series_datasets and dataset not in non_time_series_datasets:
            row_type = "TimeSeries"
        elif dataset in non_time_series_datasets:
            row_type = "NonTimeSeries"
        else:
            row_type = "Unknown"

        # Time Series AVG
        if row_type == "NonTimeSeries" and not time_series_avg_inserted:
            best_value_ts_avg = table.loc["Time Series AVG"].min()
            if not small_is_better:
                best_value_ts_avg = table.loc["Time Series AVG"].max()
            latex_table += (
                "\\midrule\n\\textbf{Time Series AVG} & "
                + " & ".join(
                    [
                        f"\\textbf{{{val:.2f}}}"
                        if val == best_value_ts_avg
                        else f"{val:.2f}"
                        for val in time_series_avg
                    ]
                )
                + " \\\\\n\\midrule\n"
            )
            time_series_avg_inserted = True

        row = [dataset]
        for method in table.columns:
            value = table.loc[dataset, method]
            if value == best_values[dataset]:  # 最小値を太字にする
                row.append(f"\\textbf{{{value:.2f}}}")
            else:
                row.append(f"{value:.2f}")

        latex_table += " & ".join(row) + " \\\\\n"

    # Non-Time Series AVG
    best_value_non_ts_avg = table.loc["Non-Time Series AVG"].min()
    if not small_is_better:
        best_value_non_ts_avg = table.loc["Non-Time Series AVG"].max()
    latex_table += (
        "\\midrule\n\\textbf{Non-Time Series AVG} & "
        + " & ".join(
            [
                f"\\textbf{{{val:.2f}}}"
                if val == best_value_non_ts_avg
                else f"{val:.2f}"
                for val in non_time_series_avg
            ]
        )
        + " \\\\\n\\midrule\n"
    )

    # Overall AVG
    best_value_overall_avg = table.loc["Overall AVG"].min()
    if not small_is_better:
        best_value_overall_avg = table.loc["Overall AVG"].max()
    latex_table += (
        "\\textbf{Overall AVG} & "
        + " & ".join(
            [
                f"\\textbf{{{val:.2f}}}"
                if val == best_value_overall_avg
                else f"{val:.2f}"
                for val in overall_avg
            ]
        )
        + " \\\\\n\\bottomrule\n\\end{tabular}"
    )

    print(latex_table)


def print_method_inorder_by(
    df: pd.DataFrame,
    column_name: str,
    small_is_better=True,
    exclude_datasets: List[str] = default_exclude_datasets,
):
    df = df.reset_index(inplace=False)
    df_filtered = df[~df["Method"].isin(default_omit_methods())]
    df_filtered = df_filtered[~df_filtered["Dataset"].isin(exclude_datasets)]
    df_filtered = df_filtered[["Method", "Dataset", column_name, "Time Series"]]
    # Mean values for Time-Series, Non-Time-Series, and Overall
    time_series_avg = (
        df_filtered[df_filtered["Time Series"]].groupby("Method")[column_name].mean()
    )
    non_time_series_avg = (
        df_filtered[~df_filtered["Time Series"]].groupby("Method")[column_name].mean()
    )
    overall_avg = df_filtered.groupby("Method")[column_name].mean()

    # print method name inorder
    time_series_avg.sort_values(ascending=small_is_better, inplace=True)
    print(f"Time Series AVG of {column_name}")
    print(time_series_avg)
    non_time_series_avg.sort_values(ascending=small_is_better, inplace=True)
    print(f"Non-Time Series AVG {column_name}")
    print(non_time_series_avg)
    overall_avg.sort_values(ascending=small_is_better, inplace=True)
    print(f"Overall AVG of {column_name}")
    print(overall_avg)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for dataset overview."""
    latex_table = """
\\begin{table*}[ht]
    \\centering
    \\caption{Dataset Overview}
    \\label{tab:dataset_overview}
    \\begin{tabular}{l l l c}
        \\toprule
        Dataset & Semantics & Source & Time Series \\\\
        \\midrule
"""

    for _, row in df.iterrows():
        latex_table += f"        {row['Dataset']}~\\cite{{{row['Tex Name']}}} & {row['Semantics']} & {row['Source']} & {'Yes' if row['Time Series'] else 'No'} \\\\\n"

    latex_table += """
        \\bottomrule
    \\end{tabular}
\\end{table*}
"""

    return latex_table


def generate_statistics_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for dataset statistics."""
    stats_table = """
\\begin{table*}[ht]
    \\centering
    \\caption{Dataset Statistics}
    \\label{tab:dataset_statistics}
    \\begin{tabular}{l r r r r r r}
        \\toprule
        Dataset & Max DP & Min DP & Avg DP & Std DP & Max Value & Min Value \\\\
        \\midrule
"""

    for _, row in df.iterrows():
        stats_table += f"        {row['Dataset']}~\\cite{{{row['Tex Name']}}} & {row['Max DP']} & {row['Min DP']} & {row['Avg DP']} & {row['Std DP']} & {row['Max Value']} & {row['Min Value']} \\\\\n"

    stats_table += """
        \\bottomrule
    \\end{tabular}
\\end{table*}
"""

    return stats_table


def generate_combined_latex_table(df: pd.DataFrame) -> str:
    """Generate a combined LaTeX table with dataset overview and statistics."""

    n_columns = 5
    loc_specifier = "c" + "|l" * 3 + "|c" * (n_columns - 5) + "|r"
    assert n_columns == len(loc_specifier.lower().replace("|", ""))
    latex_table = f"""
    \\begin{{tabular}}{{{loc_specifier}}}
        \\toprule"""

    def make_multicol(column: str, n: int, content: str):
        return f"\\multicolumn{{{n}}}{{c|}}{{\\makecell{{{column} \\\\ {content}}}}}"

    headings = [
        "",
        "Dataset",
        "Semantics",
        "Source",
        "\# of Records",
    ]
    latex_table += "        " + " & ".join(headings) + " \\\\\n"
    latex_table += """
        \\midrule
"""

    n_time_series = df["Time Series"].sum()
    n_non_time_series = len(df) - n_time_series
    latex_table += (
        f"        \\multirow{{{n_time_series}}}{{*}}"
        + "{\\rotatebox{90}{\\textbf{Time Series}}}\n"
    )

    inserted_midrule = False
    for dataset in datasets_inorder:
        row = df.loc[df["Dataset"] == dataset].iloc[0]
        number_of_records_with_commas = "{:,}".format(row["Num Values"])

        row_values = [
            f"{row['Dataset']}~\\cite{{{row['Tex Name']}}}",
            row["Semantics"],
            row["Source"],
            number_of_records_with_commas,
        ]
        assert len(row_values) + 1 == n_columns

        if not inserted_midrule and not row["Time Series"]:
            latex_table += "        \\midrule\n"
            latex_table += (
                f"        \\multirow{{{n_non_time_series}}}{{*}}"
                + "{\\rotatebox{90}{\\textbf{Non-Time Series}}}\n"
            )
            inserted_midrule = True

        latex_table += "        & " + " & ".join(row_values) + " \\\\\n"

    latex_table += """
        \\bottomrule
    \\end{tabular}
"""

    return latex_table


def generate_combined_latex_table2(df: pd.DataFrame) -> str:
    """Generate a combined LaTeX table with dataset overview and statistics."""

    n_columns = 13
    loc_specifier = "c" + "|l" + "|c" * (n_columns - 2)
    assert n_columns == len(loc_specifier.lower().replace("|", ""))
    latex_table = f"""
    \\begin{{tabular}}{{{loc_specifier}}}
        \\toprule
"""

    def make_multicol(column: str, n: int, content: str):
        return f"\\multicolumn{{{n}}}{{c|}}{{\\makecell{{{column} \\\\ {content}}}}}"

    headings = [
        "",
        "Dataset",
        make_multicol("Values", 3, "Non-Unique \\% | Avg. | Std. Dev."),
        make_multicol("Decimal Precision", 4, "Max|Min|Avg|Std.Dev."),
        make_multicol("IEEE754 Exponent", 2, "Avg|Std.Dev."),
        make_multicol(
            "Avg. XOR Zero Count \\\\ with the Previous Value", 2, "Leading | Trailing"
        ),
    ]
    latex_table += "        " + " & ".join(headings) + " \\\\\n"
    latex_table += """
        \\midrule
"""

    n_time_series = df["Time Series"].sum()
    n_non_time_series = len(df) - n_time_series
    latex_table += (
        f"        \\multirow{{{n_time_series}}}{{*}}"
        + "{\\rotatebox{90}{\\textbf{Time Series}}}\n"
    )

    inserted_midrule = False
    for dataset in datasets_inorder:
        row = df.loc[df["Dataset"] == dataset].iloc[0]

        row_values = [
            # f"{row['Dataset']}~\\cite{{{row['Tex Name']}}}",
            f"{row['Dataset']}",
            f"{row['Non-Unique Value Ratio'] * 100:.1f}\\%",
            f"{row['Avg Value']:.1f}",
            f"{row['Std Value']:.1f}",
            str(row["Max DP"]),
            str(row["Min DP"]),
            f"{row['Avg DP']:.1f}",
            f"{row['Std DP']:.1f}",
            f"{row['Avg Exponent']:.1f}",
            f"{row['Std Exponent']:.1f}",
            f"{row['Avg Leading Zeros']:.1f}",
            f"{row['Avg Trailing Zeros']:.1f}",
        ]
        assert len(row_values) + 1 == n_columns

        if not inserted_midrule and not row["Time Series"]:
            latex_table += "        \\midrule\n"
            latex_table += (
                f"        \\multirow{{{n_non_time_series}}}{{*}}"
                + "{\\rotatebox{90}{\\textbf{Non-Time Series}}}\n"
            )
            inserted_midrule = True

        latex_table += "        & " + " & ".join(row_values) + " \\\\\n"

    latex_table += """
        \\bottomrule
    \\end{tabular}
"""

    return latex_table


def generate_combined(df: pd.DataFrame) -> str:
    latex_table = """
\\begin{table*}[ht]
    \\centering
    \\caption{Dataset Overview}
    \\label{tab:dataset_overview}
"""
    latex_table += generate_combined_latex_table(df)
    latex_table += """
\\end{table*}
"""
    latex_table += """
\\begin{table*}[ht]
    \\centering
    \\caption{Dataset Statistics}
    \\label{tab:dataset_statistics}
"""
    latex_table += generate_combined_latex_table2(df)
    latex_table += """
\\end{table*}
"""
    return latex_table


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python plot2.py <json_file> [<xor_json_file> <ucr2018_json_file> <embeddings_json_file>]"
        )
        sys.exit(1)
    if len(sys.argv) > 2 and len(sys.argv) < 5:
        print(
            "Usage: python plot2.py <json_file> [<xor_json_file> <ucr2018_json_file> <embeddings_json_file>]"
        )
        sys.exit(1)

    json_file_path = sys.argv[1]
    out_dir = Path(json_file_path).parent.joinpath(Path(json_file_path).stem)
    os.makedirs(out_dir, exist_ok=True)
    with open(json_file_path, "r") as f:
        all_output = cast(AllOutput, json.load(f))
    all_output: AllOutputHandler = AllOutputHandler(all_output)

    if len(sys.argv) == 5:
        import embedding_plot, ucr2018_plot, plot_xor

        xor_json_file_path = sys.argv[2]
        xor_df = plot_xor.create_df(xor_json_file_path)

        ucr2018_json_file_path = sys.argv[3]
        ucr2018_df = ucr2018_plot.create_df(ucr2018_json_file_path)

        embeddings_json_file_path = sys.argv[4]
        embeddings_df = embedding_plot.create_df(embeddings_json_file_path)
        print(xor_df)
        print(ucr2018_df)
        print(embeddings_df)
    else:
        xor_df = None
        ucr2018_df = None
        embeddings_df = None

    # averaged() を呼び出し、AveragedAllOutput へ変換
    averaged_all_output = all_output.averaged()

    data_existence_table = create_data_existence_table(averaged_all_output)
    print("===== Data Existence Table =====")
    print(data_existence_table)

    average_stats_all: Dict[str, Dict[str, Optional[AveragedStats]]] = (
        averaged_all_output.map_to_stats()
    )
    print("===== Average Stats All =====")
    df_all = create_all_df(average_stats_all)
    stats_df = load_stats_df("stats.json")
    stats_df.to_csv("dataset_stats.csv")
    print(stats_df)
    print(df_all)
    merged = load_merged_df("stats.json", average_stats_all)
    merged.to_csv("merged.csv")
    plot_scatter(merged, out_dir.joinpath("scatter.png"))
    # TODO: omit Comp Ratio from radar chart and add comp ratios plot within TS datasets and non-TS datasets.
    plot_radar_chart(
        merged,
        out_dir.joinpath("radar.png"),
        metrics=[
            *[f"{n} Throughput" for n in ["Comp", "DeComp", "Max", "Sum"]],
            *[
                f"Filter {i} Throughput"
                for i in ["Gt 10", "Gt 50", "Gt 90", "Eq", "Ne"]
            ],
        ],
    )
    plot_radar_chart(
        merged,
        out_dir.joinpath("radar_all.png"),
        metrics=[
            "Comp Ratio",
            *[f"{n} Throughput" for n in ["Comp", "DeComp", "Max", "Sum"]],
            *[
                f"Filter {i} Throughput"
                for i in ["Gt 10", "Gt 50", "Gt 90", "Eq", "Ne"]
            ],
        ],
    )
    plot_radar_chart(
        merged,
        out_dir.joinpath("radar_reduced.png"),
        metrics=[
            "Comp Ratio",
            *[f"{n} Throughput" for n in ["Comp", "DeComp", "Max", "Sum"]],
            *[f"Filter {i} Throughput" for i in ["Gt 90"]],
        ],
    )
    make_subplots_all(
        merged,
        out_dir.joinpath("bars.png"),
        [
            "Comp Ratio",
            "Comp Throughput",
            "DeComp Throughput",
            "Filter Gt 90 Throughput",
            "Max Throughput",
            "Sum Throughput",
        ],
    )
    make_subplots_all(
        merged,
        out_dir.joinpath("bars_with_other_datasets.png"),
        [
            "Comp Ratio",
            "Comp Throughput",
            "DeComp Throughput",
            "Filter Gt 90 Throughput",
            "Max Throughput",
            "Sum Throughput",
        ],
        xor_series_df=xor_df,
        ucr2018_series_df=ucr2018_df,
        embeddings_series_df=embeddings_df,
    )
    print("===== Comp Ratio Table =====")
    print(merged.groupby(level="Method")["Comp Throughput"].mean())
    print(merged.groupby(level="Method")["DeComp Throughput"].mean())
    print("Normalized:")
    print(
        (
            merged.groupby(level="Method")["Comp Throughput"].mean()
            / merged.groupby(level="Method")["Comp Throughput"].mean().loc["FFIAlp"]
        )
        ** (-1)
    )
    print(
        (
            merged.groupby(level="Method")["DeComp Throughput"].mean()
            / merged.groupby(level="Method")["DeComp Throughput"].mean().loc["FFIAlp"]
        )
        ** (-1)
    )
    print_table(merged, "Comp Ratio")
    print_method_inorder_by(merged, "Comp Ratio")
    averaged_stats_all_df = create_average_stats_table(
        averaged_all_output, skip_none_containing_dataset=True
    )
    print(averaged_stats_all_df.columns)
    print(
        averaged_stats_all_df[
            [
                "Comp Throughput",
                "DeComp Throughput",
                "Filter Gt 90 Exec Time",
                "Comp Exec Time Ratios",
                "DeComp Exec Time Ratios",
                "Filter Gt 90 Exec Time Ratios",
            ]
        ]
    )
    averaged_stats_all_df[
        [
            "Comp Throughput",
            "DeComp Throughput",
            "Filter Gt 90 Exec Time",
            "Comp Exec Time Ratios",
            "DeComp Exec Time Ratios",
            "Filter Gt 90 Exec Time Ratios",
        ]
    ].to_csv("exec_time_ratios.csv")
    make_all_plots_for_stacked(averaged_stats_all_df, out_dir.joinpath("stacked.png"))
    return
    print("===== Filter Gt 90 Throughput Table =====")
    print_table(merged, "Filter Gt 90 Throughput", factor=1, small_is_better=False)
    print(generate_combined(stats_df))
    return
    clustered_df = cluster_compression_analysis(stats_df, df_all)
    print(clustered_df)

    # print where Method = "BUFF" and Dataset = "City-Temp"
    print("===== BUFF City-Temp =====")
    print(df_all.loc[(slice(None), "City-Temp"), "Comp Ratio"] * 64)
    print(df_all.loc[("BUFF", slice(None)), "Comp Ratio"])
    partial_datasets_df = data_existence_table.all(axis=1)
    valid_datasets = partial_datasets_df[partial_datasets_df].index

    df_filtered = df_all[df_all.index.get_level_values("Dataset").isin(valid_datasets)]
    # print(df_all.loc[(slice(None), "NYC/29"), "Comp Ratio"])
    # print(df_filtered.loc[(slice(None), "NYC/29"), "Comp Ratio"])

    print(df_filtered.groupby(level="Method")["Comp Ratio"].mean() * 64)
    print(df_filtered.groupby(level="Method")["Comp Throughput"].mean())
    print(df_filtered.groupby(level="Method")["DeComp Throughput"].mean())
    print(df_all.loc[("BUFF", slice(None)), "DeComp Throughput"])
    print(df_filtered.groupby(level="Method")["Filter Gt 90 Throughput"].mean())

    average_stats_table = create_average_stats_table(
        averaged_all_output, skip_none_containing_dataset=True
    )
    print("===== Average Stats Table =====")
    print(average_stats_table)
    print("===== Compression Stats Table =====")
    # print only "compression_ratio", "compression_throughput", "decompression_throughput" from average_stats_table


if __name__ == "__main__":
    init_plt_font()
    main()
