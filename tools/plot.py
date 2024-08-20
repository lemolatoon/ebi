from collections import defaultdict
import json
import os
import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from statistics import fmean
from tqdm import tqdm

from typing import Any, Generic, Mapping, TypeVar, TypedDict, List, Optional, Dict
from datetime import datetime


class CompressionConfig(TypedDict):
    chunk_option: (
        dict  # Replace with the correct type if you have a more specific definition
    )
    # Replace with the correct type if you have a more specific definition
    compressor_config: dict


class FilterConfig(TypedDict):
    predicate: (
        dict  # Replace with the correct type if you have a more specific definition
    )
    chunk_id: Optional[int]
    bitmask: Optional[List[int]]


class FilterMaterializeConfig(TypedDict):
    predicate: (
        dict  # Replace with the correct type if you have a more specific definition
    )
    chunk_id: Optional[int]
    bitmask: Optional[List[int]]


class MaterializeConfig(TypedDict):
    chunk_id: Optional[int]
    bitmask: Optional[List[int]]


class CompressStatistics(TypedDict):
    compression_elapsed_time_nano_secs: int
    uncompressed_size: int
    compressed_size: int
    compressed_size_chunk_only: int
    compression_ratio: float
    compression_ratio_chunk_only: float


T = TypeVar("T")


class OutputWrapper(TypedDict, Generic[T]):
    version: str
    config_path: str
    compression_config: CompressionConfig
    command_specific: T
    elapsed_time_nanos: int
    input_filename: str
    datetime: datetime
    result_string: str


class FilterFamilyOutput(TypedDict):
    filter: List[OutputWrapper[FilterConfig]]
    filter_materialize: List[OutputWrapper[FilterMaterializeConfig]]


class AllOutputInner(TypedDict):
    compress: List[OutputWrapper[CompressStatistics]]
    filters: Mapping[str, FilterFamilyOutput]
    materialize: List[OutputWrapper[MaterializeConfig]]


class AllOutput(TypedDict):
    __root__: Dict[str, Dict[str, AllOutputInner]]


def plot_comparison(
    labels: List[str],
    data: List[float | Any] | pl.Series | tuple,
    title,
    y_label,
    output_path,
    max_value=None,
):
    # Remove None values
    filtered_labels_data = [
        (label, value) for label, value in zip(labels, data) if value is not None
    ]

    # Unzip the filtered list back into labels and data
    filtered_labels, filtered_data = zip(*filtered_labels_data)

    plt.figure(figsize=(12, 8))
    plt.bar(x=filtered_labels, height=filtered_data, color="blue")
    plt.ylim(0, max_value)
    plt.title(title)
    plt.xlabel("Compression Method")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplot(data, title, y_label, output_path):
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    data.boxplot(grid=False)

    # Customize the plot
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


compression_methods = [
    "Uncompressed",
    "RLE",
    "Gorilla",
    "Chimp",
    "Chimp128",
    "ElfOnChimp",
    "Elf",
    "BUFF",
    "DeltaSprintz",
    "Zstd",
    "Gzip",
    "Snappy",
    "FFIAlp",
]


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        sys.exit(1)

    path = sys.argv[1]

    with open(path, "r") as file:
        all_output: AllOutput = json.load(file)

    dataset_names = list(all_output.keys())
    dataset_names.remove("command_type")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    compression_ratios_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    compression_throughput_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    decompression_throughput_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    filter_throughput_data: dict[str, dict[str, List[Optional[float]]]] = {
        key: {
            method_name: [None] * len(dataset_names)
            for method_name in compression_methods
        }
        for key in ["eq", "ne", "greater"]
    }
    filter_materialize_throughput_data: dict[str, dict[str, List[Optional[float]]]] = {
        key: {
            method_name: [None] * len(dataset_names)
            for method_name in compression_methods
        }
        for key in ["eq", "ne", "greater"]
    }

    for dataset_index, dataset_name in enumerate(tqdm(dataset_names)):
        methods: Dict[str, AllOutputInner] = all_output[dataset_name]
        if dataset_name == "command_type":
            continue
        for method_name in tqdm(methods, leave=False):
            output: AllOutputInner = methods[method_name]
            ratio = fmean(
                compress["command_specific"]["compression_ratio"]
                for compress in output["compress"]
            )
            uncompressed_size = output["compress"][0]["command_specific"][
                "uncompressed_size"
            ]
            compression_throughput = uncompressed_size / fmean(
                compress["command_specific"]["compression_elapsed_time_nano_secs"]
                for compress in output["compress"]
            )
            decompression_throughput = uncompressed_size / fmean(
                materialize["elapsed_time_nanos"]
                for materialize in output["materialize"]
            )

            compression_ratios_data[method_name][dataset_index] = ratio
            compression_throughput_data[method_name][dataset_index] = (
                compression_throughput
            )
            decompression_throughput_data[method_name][dataset_index] = (
                decompression_throughput
            )

            for filter_name in ["eq", "ne", "greater"]:
                uncompressed_size = output["compress"][0]["command_specific"][
                    "uncompressed_size"
                ]
                filter_throughput = uncompressed_size / fmean(
                    f["elapsed_time_nanos"]
                    for f in output["filters"][filter_name]["filter"]
                )
                filter_throughput_data[filter_name][method_name][dataset_index] = (
                    filter_throughput
                )

                filter_materialize_throughput = uncompressed_size / fmean(
                    f["elapsed_time_nanos"]
                    for f in output["filters"][filter_name]["filter_materialize"]
                )
                filter_materialize_throughput_data[filter_name][method_name][
                    dataset_index
                ] = filter_materialize_throughput

    # Convert to DataFrame
    compression_ratios_df = pl.DataFrame(compression_ratios_data)
    compression_throughput_df = pl.DataFrame(compression_throughput_data)
    decompression_throughput_df = pl.DataFrame(decompression_throughput_data)
    filter_throughput_dfs = {
        key: pl.DataFrame(filter_throughput_data[key])
        for key in ["eq", "ne", "greater"]
    }
    filter_materialize_throughput_dfs = {
        key: pl.DataFrame(filter_materialize_throughput_data[key])
        for key in ["eq", "ne", "greater"]
    }

    # dataset-wise plot_comparison

    for dataset_index, dataset_name in enumerate(tqdm(dataset_names)):
        dataset_out_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        plot_comparison(
            compression_ratios_df.columns,
            compression_ratios_df.row(dataset_index),
            f"{dataset_name}: Compression Ratio",
            "Compression Ratio (smaller, better)",
            os.path.join(
                dataset_out_dir,
                f"compression_ratios.png",
            ),
            max_value=1.1,
        )
        plot_comparison(
            compression_throughput_df.columns,
            compression_throughput_df.row(dataset_index),
            f"{dataset_name}: Compression Throughput",
            "Throughput (GB/s)",
            os.path.join(
                dataset_out_dir,
                f"compression_throughputs.png",
            ),
        )
        dataset_filter_dir = os.path.join(dataset_out_dir, "filter")
        os.makedirs(dataset_filter_dir, exist_ok=True)
        for filter_name in ["eq", "ne", "greater"]:
            plot_comparison(
                filter_throughput_dfs[filter_name].columns,
                filter_throughput_dfs[filter_name].row(dataset_index),
                f"{dataset_name}: {
                    filter_name} Filter Throughput (bigger, better)",
                "Throughput (GB/s)",
                os.path.join(
                    dataset_filter_dir,
                    f"{filter_name}_filter_elapsed_seconds.png",
                ),
            )
            plot_comparison(
                filter_materialize_throughput_dfs[filter_name].columns,
                filter_materialize_throughput_dfs[filter_name].row(dataset_index),
                f"{dataset_name}: {
                    filter_name} Filter Materialize Throughput (bigger, better)",
                "Throughput (GB/s)",
                os.path.join(
                    dataset_filter_dir,
                    f"{
                        filter_name}_filter_materialize_elapsed_seconds.png",
                ),
            )

        plot_comparison(
            decompression_throughput_df.columns,
            decompression_throughput_df.row(dataset_index),
            f"{dataset_name}: Decompression(Materialize) Throughput (bigger, better)",
            "Throughput (GB/s)",
            os.path.join(
                dataset_out_dir,
                f"materialize_elapsed_seconds.png",
            ),
        )

    # filter for all rows making sure there are no null values
    # as preparing the average plots
    compression_ratios_df = compression_ratios_df.filter(
        pl.all_horizontal(pl.col("*").is_not_null())
    )
    compression_throughput_df = compression_throughput_df.filter(
        pl.all_horizontal(pl.col("*").is_not_null())
    )
    decompression_throughput_df = decompression_throughput_df.filter(
        pl.all_horizontal(pl.col("*").is_not_null())
    )
    for filter_name in ["eq", "ne", "greater"]:
        filter_throughput_dfs[filter_name] = filter_throughput_dfs[filter_name].filter(
            pl.all_horizontal(pl.col("*").is_not_null())
        )
        filter_materialize_throughput_dfs[filter_name] = (
            filter_materialize_throughput_dfs[
                filter_name
            ].filter(pl.all_horizontal(pl.col("*").is_not_null()))
        )

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
        "Compression Ratio (smaller, better)",
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
    )

    os.makedirs(os.path.join(barchart_dir, "filter"), exist_ok=True)
    os.makedirs(os.path.join(boxplot_dir, "filter"), exist_ok=True)
    for filter_name in ["eq", "ne", "greater"]:
        plot_comparison(
            filter_throughput_dfs[filter_name].columns,
            [
                filter_throughput_dfs[filter_name][column].mean()
                for column in filter_throughput_dfs[filter_name].columns
            ],
            f"Average {filter_name.upper()} Filter Throughput (bigger, better)",
            "Throughput (GB/s)",
            os.path.join(
                os.path.join(barchart_dir, "filter"),
                f"average_{filter_name}_filter_throughput.png",
            ),
        )
        plot_comparison(
            filter_materialize_throughput_dfs[filter_name].columns,
            [
                filter_materialize_throughput_dfs[filter_name][column].mean()
                for column in filter_materialize_throughput_dfs[filter_name].columns
            ],
            f"Average {
                filter_name.upper()} Filter Materialize Throughput (bigger, better)",
            "Throughput (GB/s)",
            os.path.join(
                os.path.join(barchart_dir, "filter"),
                f"average_{filter_name}_filter_materialize_throughput.png",
            ),
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
    )

    plot_boxplot(
        decompression_throughput_df,
        "Boxplot for Average Decompression(Materialize) Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "boxplot_decompression_throughput.png"),
    )

    for filter_name in ["eq", "ne", "greater"]:
        plot_boxplot(
            filter_throughput_dfs[filter_name],
            f"Boxplot for Average {
                filter_name.upper()} Filter Throughput (bigger, better)",
            f"Throughput (GB/s)",
            os.path.join(
                os.path.join(boxplot_dir, "filter"),
                f"boxplot_{
                    filter_name}_filter_throughput.png",
            ),
        )
        plot_boxplot(
            filter_materialize_throughput_dfs[filter_name],
            f"Boxplot for Average {
                filter_name.upper()} Filter Materialize Throughput (bigger, better)",
            f"Throughput (GB/s, bigger, better)",
            os.path.join(
                os.path.join(boxplot_dir, "filter"),
                f"boxplot_{
                    filter_name}_filter_materialize_throughput.png",
            ),
        )


if __name__ == "__main__":
    main()
