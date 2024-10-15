import copy
import json
import os
from pathlib import Path
import sys
from typing import Callable
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from statistics import fmean
from tqdm import tqdm
import itertools

from typing_extensions import (
    Any,
    Generic,
    Literal,
    Mapping,
    TypeVar,
    TypedDict,
    List,
    Optional,
    Dict,
)
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


class MaxConfig(TypedDict):
    chunk_id: Optional[int]
    bitmask: Optional[List[int]]


class SumConfig(TypedDict):
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


class ExecutionTimes(TypedDict):
    io_read_nanos: int
    io_write_nanos: int
    xor_nanos: int
    delta_nanos: int
    quantization_nanos: int
    bit_packing_nanos: int
    compare_insert_nanos: int
    sum_nanos: int
    decompression_nanos: int


execution_times_keys: List[str] = [
    "io_read_nanos",
    "io_write_nanos",
    "xor_nanos",
    "delta_nanos",
    "quantization_nanos",
    "bit_packing_nanos",
    "compare_insert_nanos",
    "sum_nanos",
    "decompression_nanos",
]


class ExecutionTimesWithOthers(TypedDict):
    io_read_nanos: int
    io_write_nanos: int
    xor_nanos: int
    delta_nanos: int
    quantization_nanos: int
    bit_packing_nanos: int
    compare_insert_nanos: int
    sum_nanos: int
    decompression_nanos: int
    others: int


class OutputWrapper(TypedDict, Generic[T]):
    version: str
    config_path: str
    compression_config: CompressionConfig
    command_specific: T
    elapsed_time_nanos: int
    execution_times: ExecutionTimes
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
    max: List[OutputWrapper[MaxConfig]]
    sum: List[OutputWrapper[SumConfig]]


class AllOutput(TypedDict):
    __root__: Dict[str, Dict[str, AllOutputInner]]


CompressionMethodKeys = Literal[
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


class SegmentLabelMapping(TypedDict):
    io_read_nanos: List[str]
    io_write_nanos: List[str]
    xor_nanos: List[str]
    others: List[str]
    bit_packing_nanos: List[str]
    decompression_nanos: List[str]
    compare_insert_nanos: List[str]
    delta_nanos: List[str]
    quantization_nanos: List[str]
    sum_nanos: List[str]


default_mapping: SegmentLabelMapping = {
    "io_read_nanos": ["IO Read"],
    "io_write_nanos": ["IO Write"],
    "xor_nanos": ["XOR"],
    "others": ["Others"],
    "bit_packing_nanos": ["Bit Packing"],
    "decompression_nanos": ["Decompression"],
    "compare_insert_nanos": ["Compare"],
    "delta_nanos": ["Delta"],
    "bit_packing_nanos": ["Bit Packing"],
    "quantization_nanos": ["Quantization"],
    "sum_nanos": ["Sum"],
}
xor_patch = {
    "xor_nanos": ["XOR", "Bit Packing"],
}
segment_mapping: Dict[CompressionMethodKeys, SegmentLabelMapping] = {
    "Uncompressed": {
        **default_mapping,
    },
    "RLE": {
        **default_mapping,
    },
    "Gorilla": {
        **default_mapping,
        **xor_patch,
    },
    "Chimp": {
        **default_mapping,
        **xor_patch,
    },
    "Chimp128": {
        **default_mapping,
        **xor_patch,
    },
    "ElfOnChimp": {
        **default_mapping,
        **xor_patch,
    },
    "Elf": {
        **default_mapping,
        **xor_patch,
    },
    "BUFF": {
        **default_mapping,
        "bit_packing_nanos": ["Bit Packing", "Quantization", "Delta"],
        "compare_insert_nanos": ["Compare", "Bit Packing", "Quantization", "Delta"],
        "sum_nanos": ["Sum", "Bit Packing", "Quantization", "Delta"],
    },
    "DeltaSprintz": {
        **default_mapping,
        "decompression_nanos": ["Bit Packing", "ZigZag", "Quantization", "Delta"],
        "compare_insert_nanos": [
            "Compare",
            "Bit Packing",
            "ZigZag",
            "Quantization",
            "Delta",
        ],
    },
    "Zstd": {
        **default_mapping,
    },
    "Gzip": {
        **default_mapping,
    },
    "Snappy": {
        **default_mapping,
    },
    "FFIAlp": {
        **default_mapping,
    },
}  # type: ignore
compression_methods: List[CompressionMethodKeys] = [
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
skip_methods = set(["RLE"])
for method in skip_methods:
    compression_methods.remove(method)
method_mapping = {
    **{method: method for method in compression_methods},
    **{f"BUFF_{i}": f"Buff_{i}" for i in [1, 3, 5, 8]},
    "BUFF": "Buff",
}
def map_method(methods: List[str]) -> List[str]:
    return [method_mapping[method] for method in methods]
mapped_processing_types = set(
    "+".join(segment_mapping[method][mapping_key])
    for method in compression_methods
    for mapping_key in segment_mapping[method]
)

fontsize = 15

# Function to calculate the ratios for each process
def calculate_ratios(
    method: str, data: ExecutionTimesWithOthers, total_time: float
) -> Dict[str, float]:
    ratios = {key: 0.0 for key in mapped_processing_types}
    for key in data.keys():
        value = data[key]
        key_mapped = "+".join(segment_mapping[method][key])
        ratios[key_mapped] = (value / total_time) * 100  # Express as a percentage
    return ratios


color_map_exe: dict[str, tuple[float, float, float, float]] = {}


def get_color_exe(label: str) -> tuple[float, float, float, float]:
    global color_map_exe
    if label in color_map_exe:
        return color_map_exe[label]
    next_index = len(color_map_exe)
    color_map_exe[label] = matplotlib.colormaps["tab20"](next_index)

    return color_map_exe[label]


def add_notes_to_plot(
    fig: Figure,
    note_str: str | None,
    x_pos: float = 0.99,
    y_pos: float = 0.01,
    fontsize: int = 12,
):
    if note_str is not None:
        fig.text(x_pos, y_pos, note_str, ha="right", va="bottom", fontsize=fontsize)

omit_title = True

def plot_stacked_execution_time_ratios_for_methods(
    # methods to dataset to execution times
    data: Dict[str, List[ExecutionTimesWithOthers]],
    dataset_index: int,
    dataset_name: str,
    y_label: str,
    output_path: str,
    note_str: str | None = None,
):
    plot_relative_stacked_execution_time_ratios_for_methods(
        data, dataset_index, dataset_name, output_path, note_str=note_str
    )
    plot_absolute_stacked_execution_times_for_methods(
        data,
        dataset_index,
        dataset_name,
        output_path.replace(".png", "_absolute.png"),
        note_str=note_str,
    )


# Function to plot a stacked bar chart for execution time ratios by compression method
def plot_relative_stacked_execution_time_ratios_for_methods(
    # methods to dataset to execution times
    data: Dict[str, List[ExecutionTimesWithOthers]],
    dataset_index: int,
    dataset_name: str,
    output_path: str,
    note_str: str | None = None,
):
    methods = list(data.keys())
    processing_types = list(data[methods[0]][0].keys())

    # Calculate the ratio for each compression method
    ratios_by_method = {}
    for method in methods:
        total_time = sum(data[method][dataset_index].values())
        ratios_by_method[method] = calculate_ratios(
            method, data[method][dataset_index], total_time
        )

    # Identify processing types that are non-zero across all methods
    valid_processing_types = []
    for processing_type in mapped_processing_types:
        if any(ratios_by_method[method][processing_type] > 0 for method in methods):
            valid_processing_types.append(processing_type)

    # Prepare to plot the chart
    bar_width = 0.5
    index = np.arange(len(methods))

    fig = plt.figure(figsize=(12, 8))
    add_notes_to_plot(fig, note_str, fontsize=fontsize)

    # Plot the stacked bar chart for each processing type
    bottom = np.zeros(len(methods))
    for i, processing_type in enumerate(valid_processing_types):
        values = [ratios_by_method[method][processing_type] for method in methods]
        plt.bar(
            index,
            values,
            bar_width,
            label=processing_type,
            bottom=bottom,
            color=get_color_exe(processing_type),
        )
        bottom += values

    # Add chart decorations
    plt.xlabel("Compression Methods", fontsize=fontsize)
    plt.ylabel("Execution Time Percentage (%)", fontsize=fontsize)
    if not omit_title:
        plt.title(f"Execution Time Ratios for {dataset_name} by Compression Method", fontsize=fontsize)
    plt.xticks(index, methods, rotation=45, fontsize=fontsize)
    plt.legend(title="Processing Types", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_absolute_stacked_execution_times_for_methods(
    # methods to dataset to execution times
    data: Dict[str, List[ExecutionTimesWithOthers]],
    dataset_index: int,
    dataset_name: str,
    output_path: str,
    note_str: str | None = None,
    patch_label_mapping: Callable[
        [Dict[CompressionMethodKeys, SegmentLabelMapping]], Any
    ]
    | None = None,
    y_label: str = "Execution Time (ns)",
):
    methods: List[CompressionMethodKeys] = list(data.keys())  # type: ignore
    processing_types = list(data[methods[0]][0].keys())
    patched_segment_mapping = copy.deepcopy(segment_mapping)
    if patch_label_mapping is not None:
        patch_label_mapping(patched_segment_mapping)

    # Calculate the absolute time for each compression method
    times_by_method = {}
    for method in methods:
        times_by_method[method] = {key: 0.0 for key in mapped_processing_types}
        for processing_type in processing_types:
            labels = patched_segment_mapping[method][processing_type]
            mapped_key = "+".join(labels)
            times_by_method[method][mapped_key] = data[method][dataset_index][
                processing_type
            ]

    # Identify processing types that are non-zero across all methods
    valid_processing_types = []
    for processing_type in mapped_processing_types:
        if any(times_by_method[method][processing_type] > 0 for method in methods):
            valid_processing_types.append(processing_type)

    # Prepare to plot the chart
    bar_width = 0.5
    index = np.arange(len(methods))

    fig = plt.figure(figsize=(12, 8))
    add_notes_to_plot(fig, note_str, fontsize=fontsize)

    # Plot the stacked bar chart for each processing type
    bottom = np.zeros(len(methods))
    for i, processing_type in enumerate(valid_processing_types):
        values = [times_by_method[method][processing_type] for method in methods]
        plt.bar(
            index,
            values,
            bar_width,
            label=processing_type,
            bottom=bottom,
            color=get_color_exe(processing_type),
        )
        bottom += values

    # Add chart decorations
    plt.xlabel("Compression Methods", fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)  # y-axis now reflects absolute time
    if not omit_title:
        plt.title(f"Execution Times for {dataset_name} by Compression Method", fontsize=fontsize)
    plt.xticks(index, map_method(methods), rotation=45, fontsize=fontsize)
    plt.legend(title="Processing Types", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_comparison(
    labels: List[str],
    data: List[float | Any] | pl.Series | tuple,
    title,
    y_label,
    output_path,
    max_value=None,
    note_str: str | None = None,
):
    # Remove None values
    filtered_labels_data = [
        (label, value) for label, value in zip(labels, data) if value is not None
    ]

    # Unzip the filtered list back into labels and data
    filtered_labels, filtered_data = zip(*filtered_labels_data)

    fig = plt.figure(figsize=(12, 8))
    add_notes_to_plot(fig, note_str, fontsize=fontsize)
    plt.bar(x=map_method(filtered_labels), height=filtered_data, color="blue")
    plt.ylim(0, max_value)
    if not omit_title:
        plt.title(title, fontsize=fontsize)
    plt.xlabel("Compression Method", fontsize=fontsize)   
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(rotation=45, ha="right", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplot(data, title, y_label, output_path, note_str: str | None = None):
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    data = data.rename(columns=method_mapping)
    # Plot the boxplot
    fig = plt.figure(figsize=(10, 6))
    add_notes_to_plot(fig, note_str, fontsize=fontsize)
    data.boxplot(grid=False, fontsize=fontsize)

    # Customize the plot
    if not omit_title:
        plt.title(title, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(rotation=45, ha="right", fontsize=fontsize)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


color_map: dict[str, tuple[float, float, float, float]] = {}


def get_color(label: str) -> tuple[float, float, float, float]:
    global color_map
    if label in color_map:
        return color_map[label]
    next_index = len(color_map)
    color_map[label] = matplotlib.colormaps["tab20"](next_index)

    return color_map[label]


def plot_combined_radar_chart(data_dict, title, labels, output_path):
    metrics = list(data_dict.keys())
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize each metric's data by dividing by the maximum value for that metric
    normalized_data_dict = {}
    for metric, values in data_dict.items():
        max_value = max(values)
        normalized_data_dict[metric] = [v / max_value for v in values]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Plot each label's data
    for i, label in enumerate(labels):
        values = [normalized_data_dict[metric][i] for metric in metrics]
        values += values[:1]  # Repeat the first value to close the polygon
        ax.plot(angles, values, color=get_color(method_mapping[label]), label=method_mapping[label], linewidth=2, fontsize=fontsize)

    ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(metrics)
    ax.set_xticklabels(
        metrics, fontsize=12, fontweight="bold", horizontalalignment="center"
    )

    if not omit_title:
        plt.title(title, size=15, color="black", y=1.1, fontsize=fontsize)

    # Adjust legend to be split into two rows if there are many labels
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=fontsize)

    # Add padding around the chart to prevent clipping
    # plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.3)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    plt.savefig(output_path, format="png", dpi=300)
    plt.close()


filter_methods = ["eq", "ne", *[f"greater_{i}th_percentile" for i in [10, 50, 90]]]


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
    out_dir = os.path.join(out_dir, Path(path).stem)
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
    max_throughput_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    sum_throughput_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    filter_throughput_data: dict[str, dict[str, List[Optional[float]]]] = {
        key: {
            method_name: [None] * len(dataset_names)
            for method_name in compression_methods
        }
        for key in filter_methods
    }
    filter_materialize_throughput_data: dict[str, dict[str, List[Optional[float]]]] = {
        key: {
            method_name: [None] * len(dataset_names)
            for method_name in compression_methods
        }
        for key in filter_methods
    }

    # execution times
    compression_execution_times_data: dict[
        str, List[Optional[ExecutionTimesWithOthers]]
    ] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    decompression_execution_times_data: dict[
        str, List[Optional[ExecutionTimesWithOthers]]
    ] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    decompression_execution_times_ratio_data: Dict[str, List[Dict[str, float]]] = {
        method_name: [None] * len(dataset_names) for method_name in compression_methods
    }
    filter_execution_times_data: dict[
        str, dict[str, List[Optional[ExecutionTimesWithOthers]]]
    ] = {
        key: {
            method_name: [None] * len(dataset_names)
            for method_name in compression_methods
        }
        for key in filter_methods
    }

    filter_materialize_execution_times_data: dict[
        str, dict[str, List[Optional[ExecutionTimesWithOthers]]]
    ] = {
        key: {
            method_name: [None] * len(dataset_names)
            for method_name in compression_methods
        }
        for key in filter_methods
    }

    for dataset_index, dataset_name in enumerate(tqdm(dataset_names)):
        methods: Dict[str, AllOutputInner] = all_output[dataset_name]
        if dataset_name == "command_type":
            continue
        for method_name in tqdm(methods, leave=False):
            if  method_name in skip_methods:
                continue
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
            max_throughput = uncompressed_size / fmean(
                max["elapsed_time_nanos"] for max in output["max"]
            )
            sum_throughput = uncompressed_size / fmean(
                sum["elapsed_time_nanos"] for sum in output["sum"]
            )

            compression_execution_times: ExecutionTimesWithOthers = {}  # type: ignore
            decompression_execution_times: ExecutionTimesWithOthers = {}  # type: ignore
            decompression_execution_times_ratios: Dict[str, float] = {}
            for key in execution_times_keys:
                compression_execution_times[key] = fmean(
                    compress["execution_times"][key] for compress in output["compress"]
                )
                compression_execution_times["others"] = fmean(
                    compress["elapsed_time_nanos"] for compress in output["compress"]
                ) - sum(compression_execution_times.values())

                decompression_execution_times[key] = fmean(
                    materialize["execution_times"][key]
                    for materialize in output["materialize"]
                )
                entiere_decompression_time = fmean(
                    materialize["elapsed_time_nanos"]
                    for materialize in output["materialize"]
                )
                decompression_execution_times["others"] = entiere_decompression_time - sum(decompression_execution_times.values())

                decompression_execution_times_ratios[key] = decompression_execution_times[key] / entiere_decompression_time

            compression_execution_times_data[method_name][dataset_index] = (
                compression_execution_times
            )
            decompression_execution_times_data[method_name][dataset_index] = (
                decompression_execution_times
            )
            decompression_execution_times_ratio_data[method_name][dataset_index] = (
                decompression_execution_times_ratios
            )

            compression_ratios_data[method_name][dataset_index] = ratio
            compression_throughput_data[method_name][dataset_index] = (
                compression_throughput
            )
            decompression_throughput_data[method_name][dataset_index] = (
                decompression_throughput
            )
            max_throughput_data[method_name][dataset_index] = max_throughput
            sum_throughput_data[method_name][dataset_index] = sum_throughput

            for filter_name in filter_methods:
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

                filter_execution_times: ExecutionTimesWithOthers = {}  # type: ignore
                filter_materialize_execution_times: ExecutionTimesWithOthers = {}  # type: ignore

                for key in execution_times_keys:
                    filter_execution_times[key] = fmean(
                        f["execution_times"][key]
                        for f in output["filters"][filter_name]["filter"]
                    )
                    filter_execution_times["others"] = fmean(
                        f["elapsed_time_nanos"]
                        for f in output["filters"][filter_name]["filter"]
                    ) - sum(filter_execution_times.values())

                    filter_materialize_execution_times[key] = fmean(
                        f["execution_times"][key]
                        for f in output["filters"][filter_name]["filter_materialize"]
                    )
                    filter_materialize_execution_times["others"] = fmean(
                        f["elapsed_time_nanos"]
                        for f in output["filters"][filter_name]["filter_materialize"]
                    ) - sum(filter_materialize_execution_times.values())

                filter_execution_times_data[filter_name][method_name][dataset_index] = (
                    filter_execution_times
                )
                filter_materialize_execution_times_data[filter_name][method_name][
                    dataset_index
                ] = filter_materialize_execution_times

    # Convert to DataFrame
    compression_ratios_df = pl.DataFrame(compression_ratios_data)
    compression_throughput_df = pl.DataFrame(compression_throughput_data)
    decompression_throughput_df = pl.DataFrame(decompression_throughput_data)
    max_throughput_df = pl.DataFrame(max_throughput_data)
    sum_throughput_df = pl.DataFrame(sum_throughput_data)
    filter_throughput_dfs = {
        key: pl.DataFrame(filter_throughput_data[key]) for key in filter_methods
    }
    filter_materialize_throughput_dfs = {
        key: pl.DataFrame(filter_materialize_throughput_data[key])
        for key in filter_methods
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
            note_str="*ALP utilizes SIMD instructions",
        )

        plot_stacked_execution_time_ratios_for_methods(
            compression_execution_times_data,
            dataset_index,
            dataset_name,
            f"{dataset_name}: Compression Execution Times",
            os.path.join(
                dataset_out_dir,
                f"compression_execution_times.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
        )
        plot_stacked_execution_time_ratios_for_methods(
            decompression_execution_times_data,
            dataset_index,
            dataset_name,
            f"{dataset_name}: Decompression Execution Times",
            os.path.join(
                dataset_out_dir,
                f"decompression_execution_times.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
        )

        dataset_filter_dir = os.path.join(dataset_out_dir, "filter")
        os.makedirs(dataset_filter_dir, exist_ok=True)
        for filter_name in filter_methods:
            plot_comparison(
                filter_throughput_dfs[filter_name].columns,
                filter_throughput_dfs[filter_name].row(dataset_index),
                f"{dataset_name}: {filter_name} Filter Throughput (bigger, better)",
                "Throughput (GB/s)",
                os.path.join(
                    dataset_filter_dir,
                    f"{filter_name}_filter_elapsed_seconds.png",
                ),
                note_str="*ALP,Buff utilize SIMD instructions",
            )
            plot_comparison(
                filter_materialize_throughput_dfs[filter_name].columns,
                filter_materialize_throughput_dfs[filter_name].row(dataset_index),
                f"{dataset_name}: {filter_name} Filter Materialize Throughput (bigger, better)",
                "Throughput (GB/s)",
                os.path.join(
                    dataset_filter_dir,
                    f"{filter_name}_filter_materialize_elapsed_seconds.png",
                ),
                note_str="*ALP,Buff utilize SIMD instructions",
            )

            plot_stacked_execution_time_ratios_for_methods(
                filter_execution_times_data[filter_name],
                dataset_index,
                dataset_name,
                f"{dataset_name}: {filter_name} Filter Execution Times",
                os.path.join(
                    dataset_filter_dir,
                    f"{filter_name}_filter_execution_times.png",
                ),
                note_str="*ALP,Buff utilize SIMD instructions",
            )
            plot_stacked_execution_time_ratios_for_methods(
                filter_materialize_execution_times_data[filter_name],
                dataset_index,
                dataset_name,
                f"{dataset_name}: {filter_name} Filter Materialize Execution Times",
                os.path.join(
                    dataset_filter_dir,
                    f"{filter_name}_filter_materialize_execution_times.png",
                ),
                note_str="*ALP,Buff utilize SIMD instructions",
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
            note_str="*ALP utilize SIMD instructions",
        )

        plot_comparison(
            max_throughput_df.columns,
            max_throughput_df.row(dataset_index),
            f"{dataset_name}: Max Throughput (bigger, better)",
            "Throughput (GB/s)",
            os.path.join(
                dataset_out_dir,
                f"max_elapsed_seconds.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
        )
        plot_comparison(
            sum_throughput_df.columns,
            sum_throughput_df.row(dataset_index),
            f"{dataset_name}: Sum Throughput (bigger, better)",
            "Throughput (GB/s)",
            os.path.join(
                dataset_out_dir,
                f"sum_elapsed_seconds.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
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
    max_throughput_df = max_throughput_df.filter(
        pl.all_horizontal(pl.col("*").is_not_null())
    )
    sum_throughput_df = sum_throughput_df.filter(
        pl.all_horizontal(pl.col("*").is_not_null())
    )
    for filter_name in filter_methods:
        filter_throughput_dfs[filter_name] = filter_throughput_dfs[filter_name].filter(
            pl.all_horizontal(pl.col("*").is_not_null())
        )
        filter_materialize_throughput_dfs[filter_name] = (
            filter_materialize_throughput_dfs[
                filter_name
            ].filter(pl.all_horizontal(pl.col("*").is_not_null()))
        )
    

    decompression_execution_times_throughput: Dict[str, List[Dict[str, float]]] = {
        method: [
            {
                key: fmean([d[key] for d in decompression_execution_times_ratio_data[method]]) / fmean(decompression_throughput_data[method])
                for key in decompression_execution_times_ratio_data[method][0].keys()
            }
        ]
        for method in decompression_execution_times_ratio_data
    }
    barchart_dir = os.path.join(out_dir, "barchart")
    boxplot_dir = os.path.join(out_dir, "boxplot")

    plot_absolute_stacked_execution_times_for_methods(
        decompression_execution_times_throughput,
        0,
        "Decompression Throughput",
        os.path.join(barchart_dir, "stacked_decompression_throughput.png"),
        note_str="*ALP utilize SIMD instructions",
        y_label="Average Execution Times (s/GB)",
    )

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
        note_str="*ALP utilizes SIMD instructions",
    )
    plot_comparison(
        max_throughput_df.columns,
        [max_throughput_df[column].mean() for column in max_throughput_df.columns],
        "Average Max Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_max_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )
    plot_comparison(
        sum_throughput_df.columns,
        [sum_throughput_df[column].mean() for column in sum_throughput_df.columns],
        "Average Sum Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_sum_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    os.makedirs(os.path.join(barchart_dir, "filter"), exist_ok=True)
    os.makedirs(os.path.join(boxplot_dir, "filter"), exist_ok=True)
    for filter_name in filter_methods:
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
            note_str="*ALP,Buff utilize SIMD instructions",
        )
        filter_throughput_without_buff_df = filter_throughput_dfs[filter_name].drop(
            "BUFF"
        )
        plot_comparison(
            filter_throughput_without_buff_df.columns,
            [
                filter_throughput_without_buff_df[column].mean()
                for column in filter_throughput_without_buff_df.columns
            ],
            f"Average {filter_name.upper()} Filter Throughput (bigger, better) without BUFF",
            "Throughput (GB/s)",
            os.path.join(
                os.path.join(barchart_dir, "filter"),
                f"average_{filter_name}_filter_throughput_without_buff.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
        )
        plot_comparison(
            filter_materialize_throughput_dfs[filter_name].columns,
            [
                filter_materialize_throughput_dfs[filter_name][column].mean()
                for column in filter_materialize_throughput_dfs[filter_name].columns
            ],
            f"Average {filter_name.upper()} Filter Materialize Throughput (bigger, better)",
            "Throughput (GB/s)",
            os.path.join(
                os.path.join(barchart_dir, "filter"),
                f"average_{filter_name}_filter_materialize_throughput.png",
            ),
            note_str="*ALP,Buff utilize SIMD instructions",
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
        note_str="*ALP utilize SIMD instructions",
    )

    plot_boxplot(
        max_throughput_df,
        "Boxplot for Average Max Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "boxplot_max_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )
    plot_boxplot(
        sum_throughput_df,
        "Boxplot for Average Sum Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "boxplot_sum_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    for filter_name in filter_methods:
        plot_boxplot(
            filter_throughput_dfs[filter_name],
            f"Boxplot for Average {filter_name.upper()} Filter Throughput (bigger, better)",
            f"Throughput (GB/s)",
            os.path.join(
                os.path.join(boxplot_dir, "filter"),
                f"boxplot_{filter_name}_filter_throughput.png",
            ),
            note_str="*ALP,Buff utilize SIMD instructions",
        )
        filter_throughput_without_buff_df = filter_throughput_dfs[filter_name].drop(
            "BUFF"
        )
        plot_boxplot(
            filter_throughput_without_buff_df,
            f"Boxplot for Average {filter_name.upper()} Filter Throughput (bigger, better) without BUFF",
            f"Throughput (GB/s)",
            os.path.join(
                os.path.join(boxplot_dir, "filter"),
                f"boxplot_{filter_name}_filter_throughput_without_buff.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
        )
        plot_boxplot(
            filter_materialize_throughput_dfs[filter_name],
            f"Boxplot for Average {filter_name.upper()} Filter Materialize Throughput (bigger, better)",
            f"Throughput (GB/s, bigger, better)",
            os.path.join(
                os.path.join(boxplot_dir, "filter"),
                f"boxplot_{filter_name}_filter_materialize_throughput.png",
            ),
            note_str="*ALP,Buff utilize SIMD instructions",
        )

    # Combined Radar Chart
    combined_radar_chart_dir = os.path.join(out_dir, "radar_chart")
    os.makedirs(combined_radar_chart_dir, exist_ok=True)

    compression_ratios_avg = compression_ratios_df.mean()
    compression_throughput_avg = compression_throughput_df.mean()
    decompression_throughput_avg = decompression_throughput_df.mean()
    max_throughput_avg = max_throughput_df.mean()
    sum_throughput_avg = sum_throughput_df.mean()

    filter_throughput_avg = {
        key: df.mean() for key, df in filter_throughput_dfs.items()
    }

    filter_materialize_throughput_avg = {
        key: df.mean() for key, df in filter_materialize_throughput_dfs.items()
    }

    def plot_radar_chart(pred: pl.Expr | List[str], title: str, output_path: str, kind: Literal["all", "comp", "db-query", "without-filter-mat"]="all"):
        data_dict = {
            "Compression Ratios": np.array(compression_ratios_avg.select(pred).row(0))
            ** -1,
            "Compression Throughput": np.array(
                compression_throughput_avg.select(pred).row(0)
            ),
            "Decompression Throughput": np.array(
                decompression_throughput_avg.select(pred).row(0)
            ),
            "Max Throughput": np.array(max_throughput_avg.select(pred).row(0)),
            "Sum Throughput": np.array(sum_throughput_avg.select(pred).row(0)),
        }

        match kind:
            case "all":
                pass
            case "comp":
                data_dict.pop("Max Throughput")
                data_dict.pop("Sum Throughput")
            case "db-query":
                data_dict.pop("Compression Ratios")
                data_dict.pop("Compression Throughput")
                data_dict.pop("Decompression Throughput")

        if kind != "comp":
            for key in filter_throughput_avg:
                data_dict[f"Filter Throughput \n({key})"] = np.array(
                    filter_throughput_avg[key].select(pred).row(0)
                )

            if kind != "without-filter-mat":
                for key in filter_materialize_throughput_avg:
                    data_dict[f"Filter Materialize Throughput \n({key})"] = np.array(
                        filter_materialize_throughput_avg[key].select(pred).row(0)
                    )

        labels = compression_ratios_df.select(pred).columns

        plot_combined_radar_chart(data_dict, title, labels, output_path)

    for kind in ["all", "comp", "db-query", "without-filter-mat"]:
        plot_radar_chart(
            pl.all().exclude("Uncompressed", "RLE"),
            "Average Performance Radar Chart All",
            os.path.join(combined_radar_chart_dir, f"all_radar_chart_{kind}.png"),
            kind=kind
        )
        # Xor Family
        plot_radar_chart(
            ["Gorilla", "Chimp", "Chimp128", "ElfOnChimp", "Elf"],
            "Average Performance Radar Chart Xor Family",
            os.path.join(combined_radar_chart_dir, f"xor_family_radar_chart_{kind}.png"),
            kind=kind
        )
    labels = compression_ratios_df.columns
    # Top 5 Compression Ratios
    compression_ratios_sorted_labels = list(
        map(
            lambda x: x[0],
            sorted(
                zip(labels, np.array(compression_ratios_df.mean().row(0))),
                key=lambda x: x[1],
            ),
        )
    )
    # Top 5 Compression Throughput
    compression_throughput_sorted_labels = list(
        map(
            lambda x: x[0],
            sorted(
                zip(labels, np.array(compression_throughput_df.mean().row(0))),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )
    # Top 5 Decompression Throughput
    decompressin_throughput_sorted_labels = list(
        map(
            lambda x: x[0],
            sorted(
                zip(labels, np.array(decompression_throughput_df.mean().row(0))),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )
    # Top 5 Max Throughput
    max_throughput_sorted_labels = list(
        map(
            lambda x: x[0],
            sorted(
                zip(labels, np.array(max_throughput_df.mean().row(0))),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )
    # Top 5 Sum Throughput
    sum_throughput_sorted_labels = list(
        map(
            lambda x: x[0],
            sorted(
                zip(labels, np.array(sum_throughput_df.mean().row(0))),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )
    # Top 5 Filter GREATER Throughput
    filter_greater_throughput_sorted_labels = list(
        map(
            lambda x: x[0],
            sorted(
                zip(
                    labels,
                    np.array(
                        filter_throughput_dfs["greater_10th_percentile"].mean().row(0)
                    ),
                ),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )
    for unnecessary_label in ["Uncompressed", "RLE"]:
        try:
            compression_ratios_sorted_labels.remove(unnecessary_label)
            compression_throughput_sorted_labels.remove(unnecessary_label)
            filter_greater_throughput_sorted_labels.remove(unnecessary_label)
            max_throughput_sorted_labels.remove(unnecessary_label)
            sum_throughput_sorted_labels.remove(unnecessary_label)
        except ValueError:
            print(f"{unnecessary_label} not found in labels list")
            pass

    for kind in ["all", "comp", "db-query", "without-filter-mat"]:
        plot_radar_chart(
            compression_ratios_sorted_labels[:5],
            "Top 5 Compression Ratios",
            os.path.join(
                combined_radar_chart_dir, f"top_5_compression_ratios_radar_chart_{kind}.png"
            ),
            kind=kind,
        )
        plot_radar_chart(
            compression_throughput_sorted_labels[:5],
            "Top 5 Compression Throughput",
            os.path.join(
                combined_radar_chart_dir, f"top_5_compression_throughput_radar_chart_{kind}.png"
            ),
            kind=kind
        )
        plot_radar_chart(
            decompressin_throughput_sorted_labels[:5],
            "Top 5 Decompression Throughput",
            os.path.join(
                combined_radar_chart_dir, f"top_5_decompression_throughput_radar_chart_{kind}.png"
            ),
            kind=kind
        )
        plot_radar_chart(
            max_throughput_sorted_labels[:5],
            "Top 5 Max Throughput",
            os.path.join(combined_radar_chart_dir, f"top_5_max_throughput_radar_chart_{kind}.png"),
            kind=kind
        )
        plot_radar_chart(
            sum_throughput_sorted_labels[:5],
            "Top 5 Sum Throughput",
            os.path.join(combined_radar_chart_dir, f"top_5_sum_throughput_radar_chart_{kind}.png"),
            kind=kind
        )
        plot_radar_chart(
            filter_greater_throughput_sorted_labels[:5],
            "Top 5 Filter GREATER then 10 th percentile Throughput",
            os.path.join(
                combined_radar_chart_dir, f"top_5_filter_greater_throughput_radar_chart_{kind}.png"
            ),
            kind=kind
        )


if __name__ == "__main__":
    main()
