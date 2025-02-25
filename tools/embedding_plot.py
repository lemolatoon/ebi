from typing_extensions import TypedDict
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from typing import TypedDict, List, Dict, Optional
from datetime import datetime
import pandas as pd

from tqdm import tqdm
from plot import (
    CompressionConfig,
    ExecutionTimes,
    ExecutionTimesWithOthers,
    SegmentLabelMapping,
    CompressionMethodKeys,
    compression_methods,
    plot_absolute_stacked_execution_times_for_methods,
    plot_boxplot,
    execution_times_keys,
    plot_comparison,
)
import polars as pl

from common import default_compression_method_order, default_omit_methods


class CompressStatistics(TypedDict):
    compression_elapsed_time_nano_secs: int
    uncompressed_size: int
    compressed_size: int
    compressed_size_chunk_only: int
    compression_ratio: float
    compression_ratio_chunk_only: float


class SimpleCompressionPerformanceForOneDataset(TypedDict):
    n: int
    dataset_name: str

    compression_config: CompressionConfig
    compression_elapsed_time_nanos: List[int]
    compression_segmented_execution_times: List[ExecutionTimes]
    compression_statistics: List[CompressStatistics]
    compression_result_string: List[str]

    decompression_elapsed_time_nanos: List[int]
    decompression_segmented_execution_times: List[ExecutionTimes]
    decompression_result_string: List[str]
    precision: int


def load_json(
    file_path: str,
) -> Dict[str, Dict[str, SimpleCompressionPerformanceForOneDataset]]:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def create_df(dir: str) -> pd.DataFrame:
    assert os.path.exists(dir)
    data = load_json(dir)

    dataset_names = list(data.keys())
    dataset_names.remove("command_type")

    # Initialize storage for each method
    compression_metrics = {
        method_name: {"Comp Ratio": [], "Comp Throughput": [], "DeComp Throughput": []}
        for method_name in compression_methods
    }

    # Compute metrics for each dataset and method
    for dataset_name in dataset_names:
        methods = data[dataset_name]
        for method_name, performance in methods.items():
            if method_name not in compression_methods:
                continue

            # Compute averages
            ratio = fmean(
                compress["compressed_size"] / compress["uncompressed_size"]
                for compress in performance["compression_statistics"]
            )
            compression_throughput = fmean(
                compress["uncompressed_size"]
                / compress["compression_elapsed_time_nano_secs"]
                for compress in performance["compression_statistics"]
            )
            decompression_throughput = fmean(
                compress["uncompressed_size"] / decompress
                for compress, decompress in zip(
                    performance["compression_statistics"],
                    performance["decompression_elapsed_time_nanos"],
                )
            )

            compression_metrics[method_name]["Comp Ratio"].append(ratio)
            compression_metrics[method_name]["Comp Throughput"].append(
                compression_throughput
            )
            compression_metrics[method_name]["DeComp Throughput"].append(
                decompression_throughput
            )

    # Convert to pandas DataFrame with method as index
    df_dict = {
        method_name: {
            "Comp Ratio": fmean(values["Comp Ratio"]),
            "Comp Throughput": fmean(values["Comp Throughput"]),
            "DeComp Throughput": fmean(values["DeComp Throughput"]),
        }
        for method_name, values in compression_metrics.items()
    }

    df = pd.DataFrame.from_dict(df_dict, orient="index")
    df.index.name = "Method"

    df.reset_index(inplace=True)
    df = df[~df["Method"].isin(default_omit_methods())]
    df.set_index("Method", inplace=True)

    # Sort the DataFrame by the order of the compression methods
    df = df.reindex(default_compression_method_order())

    return df


def plot_graphs(data: Dict[str, Dict[str, SimpleCompressionPerformanceForOneDataset]]):
    out_dir = "results"
    out_dir = os.path.join(out_dir, "embedding")
    os.makedirs(out_dir, exist_ok=True)

    dataset_names = list(data.keys())
    dataset_names.remove("command_type")
    compression_ratios_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    compression_throughput_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    decompression_throughput_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    # Plot absolute stacked execution times for methods
    for i, dataset_name in enumerate(dataset_names):
        methods = data[dataset_name]
        for method_name, performance in methods.items():
            if method_name not in compression_methods:
                continue
            ratio = fmean(
                compress["compressed_size"] / compress["uncompressed_size"]
                for compress in performance["compression_statistics"]
            )
            compression_throughput = fmean(
                compress["uncompressed_size"]
                / compress["compression_elapsed_time_nano_secs"]
                for compress in performance["compression_statistics"]
            )
            decompression_throughput = fmean(
                compress["uncompressed_size"] / decompress
                for compress, decompress in zip(
                    performance["compression_statistics"],
                    performance["decompression_elapsed_time_nanos"],
                )
            )

            compression_ratios_data[method_name][i] = ratio
            compression_throughput_data[method_name][i] = compression_throughput
            decompression_throughput_data[method_name][i] = decompression_throughput

    compression_ratios_df = pl.DataFrame(compression_ratios_data)
    compression_throughput_df = pl.DataFrame(compression_throughput_data)
    decompression_throughput_df = pl.DataFrame(decompression_throughput_data)

    barchart_dir = os.path.join(out_dir, "barchart")
    boxplot_dir = os.path.join(out_dir, "boxplot")
    os.makedirs(barchart_dir, exist_ok=True)
    os.makedirs(boxplot_dir, exist_ok=True)
    plot_comparison(
        compression_ratios_df.columns,
        [
            compression_ratios_df[column].mean()
            for column in compression_ratios_df.columns
        ],
        "Average Compression Ratio",
        "Compression Ratio",
        os.path.join(barchart_dir, "average_compression_ratios.png"),
        max_value=1.1,
    )
    plot_comparison(
        compression_throughput_df.columns,
        [
            compression_throughput_df[column].mean()
            for column in compression_throughput_df.columns
        ],
        "Average Compression Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_compression_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )
    plot_comparison(
        decompression_throughput_df.columns,
        [
            decompression_throughput_df[column].mean()
            for column in decompression_throughput_df.columns
        ],
        "Average Decompression Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_decompression_throughput.png"),
        note_str="*ALP,Buff utilizes SIMD instructions",
    )

    plot_boxplot(
        compression_ratios_df,
        "Boxplot for Average Compression Ratios (smaller, better)",
        "Compression Ratio",
        os.path.join(boxplot_dir, "boxplot_compression_ratios.png"),
    )
    plot_boxplot(
        compression_throughput_df,
        "Boxplot for Average Compression Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "boxplot_compression_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    plot_boxplot(
        decompression_throughput_df,
        "Boxplot for Average Decompression(Materialize) Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "boxplot_decompression_throughput.png"),
        note_str="*ALP,Buff utilize SIMD instructions",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python plot.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    data = load_json(json_file_path)
    plot_graphs(data)
