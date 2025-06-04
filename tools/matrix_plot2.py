import json, re
import os
import sys
from typing import Dict, Iterable, Optional, TypedDict, List
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
from plot import (
    CompressStatistics,
    ExecutionTimes,
    ExecutionTimesWithOthers,
    SegmentLabelMapping,
    average_execution_times,
    compression_methods,
    CompressionMethodKeys,
    get_average_execution_times_ratios,
    segment_mapping
)
import pandas as pd

from plot2 import init_plt_font, plot_absolute_stacked_execution_times_for_methods
from common import default_compression_method_order
import seaborn as sns


class CompressionConfig(TypedDict):
    chunk_option: (
        dict  # Replace with the correct type if you have a more specific definition
    )
    # Replace with the correct type if you have a more specific definition
    compressor_config: dict

def average_compression_statistics(
    cs: list[CompressStatistics],
) -> CompressStatistics:
    if not cs:
        raise ValueError("The list of compression statistics is empty.")
    
    def avg(key: str) -> float:
        assert key in cs[0], f"Key '{key}' not found in compression statistics."
        return sum(stat[key] for stat in cs) / len(cs)
    
    return {
        "compression_elapsed_time_nano_secs": avg("compression_elapsed_time_nano_secs"),
        "uncompressed_size": cs[0]["uncompressed_size"],
        "compressed_size": avg("compressed_size"),
        "compressed_size_chunk_only": avg("compressed_size_chunk_only"),
        "compression_ratio": avg("compression_ratio"),
        "compression_ratio_chunk_only": avg("compression_ratio_chunk_only"),
    }

class MatrixResult(TypedDict):
    compression_config: CompressionConfig
    compression_statistics: CompressStatistics
    compression_elapsed_time_nano_secs: int

    matmul_elapsed_time_nano_secs: int
    matmul_segmented_execution_times: ExecutionTimes
    matmul_segmented_execution_times_with_others: ExecutionTimesWithOthers
    precision: int
    matrix_size: int

    result_string: str

    start_time: datetime
    end_time: datetime

def average_matrix_result(
    results: list[MatrixResult],
) -> MatrixResult:
    if not results:
        raise ValueError("The list of matrix results is empty.")
    
    def avg(key: str) -> float:
        assert key in results[0], f"Key '{key}' not found in matrix result."
        return sum(result[key] for result in results) / len(results)
    
    ret: MatrixResult =  {
        "compression_config": results[0]["compression_config"],
        "compression_statistics": average_compression_statistics(
            [result["compression_statistics"] for result in results]
        ),
        "compression_elapsed_time_nano_secs": avg("compression_elapsed_time_nano_secs"),
        "matmul_elapsed_time_nano_secs": avg("matmul_elapsed_time_nano_secs"),
        "matmul_segmented_execution_times": average_execution_times(
            [result["matmul_segmented_execution_times"] for result in results]
        ),
        "matmul_segmented_execution_times_with_others": get_average_execution_times_ratios(
            [result["matmul_segmented_execution_times"] for result in results],
            avg("matmul_elapsed_time_nano_secs")
        ),
        "precision": results[0]["precision"],
        "matrix_size": results[0]["matrix_size"],
        "result_string": results[0]["result_string"],
        "start_time": min(result["start_time"] for result in results),
        "end_time": max(result["end_time"] for result in results),
    }

    return ret


precisions = [1, 3, 5, 8]
matrix_sizes = [128, 512, 1024, 2048, 4096]

def load_json_files_from_directory(
    directory_path: str,
) -> tuple[Dict[str, Dict[str, MatrixResult]], Dict[str, pd.DataFrame]]:
    """
    Load JSON files from a directory, aggregate MatrixResult per matrix size,
    rename BUFF entries for non-8 precisions, and build a timers DataFrame
    for each size.

    Returns:
      - results: mapping from matrix_size (str) to dict of method_name -> averaged MatrixResult
      - timers: mapping from matrix_size (str) to pd.DataFrame with columns
                ["Throughput", "ExecTimeRatios"] and index = method names
    """
    if not os.path.isdir(directory_path):
        print(f"The provided path '{directory_path}' is not a directory.")
        sys.exit(1)

    # Temporarily hold per-size, per-method lists of MatrixResult
    size_results: Dict[str, Dict[str, List[MatrixResult]]] = {}

    # Iterate over all .json files in the directory
    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(directory_path, filename)
        base = os.path.splitext(filename)[0]

        # Match "<anything>_<matrix_size>_<precision>"
        m = re.match(r".*_(\d+)_(\d+)$", base)
        if not m:
            continue

        size, precision = m.group(1), m.group(2)

        # Load the list of single-run dicts from JSON
        with open(file_path, "r") as f:
            data: List[Dict[str, MatrixResult]] = json.load(f)

        # Aggregate runs by method name
        aggregated: Dict[str, List[MatrixResult]] = {}
        for entry in data:
            for method_name, matres in entry.items():
                aggregated.setdefault(method_name, []).append(matres)

        # If this is a non-8 precision file, rename "BUFF" -> "BUFF_<precision>"
        if precision != "8" and "BUFF" in aggregated:
            aggregated[f"BUFF_{precision}"] = aggregated.pop("BUFF")

        # Merge into the per-size map
        if size not in size_results:
            size_results[size] = {}
        for method_name, runs in aggregated.items():
            size_results[size].setdefault(method_name, []).extend(runs)

    # Now average across runs for each (size, method) pair
    final_results: Dict[str, Dict[str, MatrixResult]] = {}
    for size, methods in size_results.items():
        final_results[size] = {
            method: average_matrix_result(runs)
            for method, runs in methods.items()
        }

    # Build timers DataFrame for each size
    timers: Dict[str, pd.DataFrame] = {}
    for size, methods in final_results.items():
        index = []
        throughputs = []
        exec_time_ratios = []
        for method, res in methods.items():
            index.append(method)
            uncompressed = res["compression_statistics"]["uncompressed_size"]
            elapsed = res["matmul_elapsed_time_nano_secs"]
            throughputs.append(uncompressed / elapsed)
            exec_time_ratios.append(res["matmul_segmented_execution_times_with_others"])
        timers[size] = pd.DataFrame(
            {
                "Throughput": throughputs,
                "ExecTimeRatios": exec_time_ratios,
            },
            index=index,
        )

    return final_results, timers

compression_methods_with_precision = [
    *compression_methods,
    *[f"BUFF_{p}" for p in precisions],
]

def plot_comparison2(
    methods: Iterable[str],
    values: Iterable[float],
    title: str,
    y_label: str,
    out_path: str,
) -> None:
    """
    Plot a comparison bar chart for the given methods and values,
    matching the style of plot_bar_chart.
    """
    default_fontsize = 12
    small_default = 10

    print(f"Plotting {title}")

    # Create a pandas Series for plotting
    df = pd.Series(list(values), index=list(methods), name=y_label)

    # Setup figure and axis
    fig, ax = plt.subplots()

    # Plot bar chart
    df.plot(
        kind="bar",
        ax=ax,
        color=plt.get_cmap("tab10").colors,
        fontsize=small_default,
        legend=False,
    )

    # Remove x-axis label
    ax.xaxis.label.set_visible(False)

    # Set y-axis label
    ax.set_ylabel(y_label, fontsize=default_fontsize)
    ax.yaxis.set_label_coords(-0.075, 0.45)

    # Set title
    ax.set_title(title, fontsize=default_fontsize)

    # Rotate x tick labels
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=small_default
    )

    # Adjust tick parameters
    ax.tick_params(axis="both", labelsize=small_default)

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_bar_chart(
    data: pd.DataFrame,
    metric_cols: List[str],
    title: str,
    ylabel: str,
    save_path: str
) -> None:
    """
    Plot a bar chart for specified metric columns and save the figure.
    """
    order = [*default_compression_method_order(), *[f"BUFF_{p}" for p in [1, 3, 5]]]
    subset = data.loc[order, metric_cols].copy()
    subset = subset.rename_axis("method").reset_index()
    melted = subset.melt(id_vars="method", var_name="Metric", value_name=ylabel)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="method", y=ylabel, hue="Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        sys.exit(1)

    path = sys.argv[1]

    cuda_enabled = len(sys.argv) == 3 and sys.argv[2] == "-cuda"
    cuda_prefix = "_cuda" if cuda_enabled else ""
    if cuda_enabled:
        print("Plotting CUDA results...")

    out_dir = f"results/matmul{cuda_prefix}/"
    os.makedirs(out_dir, exist_ok=True)

    results, timers = load_json_files_from_directory(path)
     # Flatten into DataFrame for bar plots
    records = []
    for size, methods in results.items():
        for method, res in methods.items():
            stats = res["compression_statistics"]
            records.append({
                "matrix_size": int(size),
                "method": method,
                "compression_ratio": stats["compression_ratio"],
                "throughput": stats["uncompressed_size"] / res["matmul_elapsed_time_nano_secs"],
            })
    df = pd.DataFrame(records)

        # Generate bar charts and stacked plots per size
    for size, group in df.groupby("matrix_size"):
        tag = f"{size}x{size}"
        pivoted = group.set_index("method")[ ["compression_ratio", "throughput"] ]
        timer_df = timers[str(size)]

        # Compression ratio bar chart
        plot_bar_chart(
            pivoted,
            ["compression_ratio"],
            f"Matrix {tag} Compression Ratio",
            "Compression Ratio",
            os.path.join(out_dir, f"{tag}_compression_ratio_bar.png"),
        )
        # Throughput bar chart
        plot_bar_chart(
            pivoted,
            ["throughput"],
            f"Matrix {tag} Throughput",
            "Throughput (GB/s)",
            os.path.join(out_dir, f"{tag}_throughput_bar.png"),
        )

        # Stacked execution time plot
        def patch_label_mapping(d: Dict[CompressionMethodKeys, SegmentLabelMapping]):
            d["BUFF"]["sum_nanos"] = ["Sum"]
            for p in precisions:
                d[f"BUFF_{p}"] = d["BUFF"]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plot_absolute_stacked_execution_times_for_methods(
            ax,
            timer_df,
            "Throughput",
            "ExecTimeRatios",
            "Execution Time (s/GB)",
            segment_mapping,
            patch_label_mapping=patch_label_mapping,
        )
        fig.savefig(
            os.path.join(out_dir, f"{tag}_stacked.png"),
            bbox_inches='tight',
            dpi=300
        )

if __name__ == "__main__":
    init_plt_font()
    main()
    # main2()
