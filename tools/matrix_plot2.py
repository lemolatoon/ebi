import json
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
    plot_boxplot,
    plot_comparison,
    plot_absolute_stacked_execution_times_for_methods,
)
import polars as pl
import pandas as pd


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
        "precision": results[0]["precision"],
        "matrix_size": results[0]["matrix_size"],
        "result_string": results[0]["result_string"],
        "start_time": min(result["start_time"] for result in results),
        "end_time": max(result["end_time"] for result in results),
    }

    return ret


def load_json_files_from_directory(
    directory_path: str,
) -> Dict[str, Dict[str, MatrixResult]]:
    if not os.path.isdir(directory_path):
        print(f"The provided path '{directory_path}' is not a directory.")
        sys.exit(1)

    results = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data: list[Dict[str, MatrixResult]] = json.load(file)
                key = os.path.splitext(filename)[0]  # Remove the .json extension

                aggregated: Dict[str, list[MatrixResult]] = {}
                for entry in data:
                    for k, v in entry.items():
                        aggregated.setdefault(k, []).append(v)

                results[key] = {
                    k: average_matrix_result(vs)
                    for k, vs in aggregated.items()
                }
    return results


precisions = [1, 3, 5, 8]
matrix_sizes = [128, 512, 1024, 2048, 4096]

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

    results = load_json_files_from_directory(path)

    matrix_sizes = [128, 512, 1024, 2048, 4096]
    if not cuda_enabled:
        matrix_sizes = [128, 512, 1024]

    results_reformed: Dict[int, Dict[str, MatrixResult]] = {
        size: {
            **results[f"matrix{cuda_prefix}_{size}_8"],
            **{
                f"BUFF_{precision}": results[f"matrix{cuda_prefix}_{size}_{precision}"]["BUFF"]
                for precision in precisions
            },
        }
        for size in matrix_sizes
    }

    segmented_execution_times_prec8: Dict[int, Dict[str, ExecutionTimesWithOthers]] = {
        size: {
            method_name: {
                **results[f"matrix{cuda_prefix}_{size}_8"][method_name]["matmul_segmented_execution_times"],
                "others": results[f"matrix{cuda_prefix}_{size}_8"][method_name]["matmul_elapsed_time_nano_secs"],
            }
            for method_name in compression_methods
        }
        for size in matrix_sizes
    }

    for matrix_size in matrix_sizes:
        result = results_reformed[matrix_size]
        execution_times = {}
        segmented_execution_times: Dict[str, List[ExecutionTimesWithOthers]] = {}
        original_size: float = 0
        for method_name in compression_methods_with_precision:
            execution_times_with_others: ExecutionTimesWithOthers = {
                **result[method_name]["matmul_segmented_execution_times"],
                "others": result[method_name]["matmul_elapsed_time_nano_secs"],
            }
            original_size = result[method_name]["compression_statistics"][
                "uncompressed_size"
            ]
            print(
                f"{matrix_size}: {result[method_name]['matmul_elapsed_time_nano_secs']}"
            )
            execution_times[method_name] = result[method_name][
                "matmul_elapsed_time_nano_secs"
            ]

            segmented_execution_times[method_name] = [execution_times_with_others]

        throughput = {
            method_name: original_size / execution_time
            for method_name, execution_time in execution_times.items()
        }
        execution_times_df = pl.DataFrame(execution_times)
        throughput_df = pl.DataFrame(throughput)

        os.makedirs(os.path.join(out_dir, "exe_time"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "throughput"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "stacked"), exist_ok=True)
        dataset_symbol = f"{matrix_size}x{matrix_size}"
        plot_comparison(
            execution_times_df.columns,
            execution_times_df.row(0),
            f"MatMul Execution Time (ns) for {dataset_symbol}",
            "Execution Times (ns)",
            os.path.join(out_dir, "exe_time", f"{dataset_symbol}_exe_time_matmul.png"),
        )

        plot_comparison(
            throughput_df.columns,
            throughput_df.row(0),
            f"MatMul Throughput (GB/s) for {dataset_symbol}",
            "Throughput (GB/s)",
            os.path.join(out_dir, "throughput", f"{dataset_symbol}_throughput.png"),
        )

        def patch_label_mapping(d: Dict[CompressionMethodKeys, SegmentLabelMapping]):
            d["BUFF"]["sum_nanos"] = ["Sum"]
            for precision in precisions:
                d[f"BUFF_{precision}"] = d["BUFF"]  # type: ignore

        plot_absolute_stacked_execution_times_for_methods(
            segmented_execution_times,
            0,
            dataset_symbol,
            os.path.join(out_dir, "stacked", f"{dataset_symbol}_stacked_matmul.png"),
            patch_label_mapping=patch_label_mapping,
        )

def main2():
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
    print(f"Files will be saved to {out_dir}")

    # JSON を読み込んで平均化まで済ませた辞書を取得
    results = load_json_files_from_directory(path)

    # 利用する行列サイズの調整
    matrix_sizes = [128, 512, 1024, 2048, 4096]
    if not cuda_enabled:
        matrix_sizes = [128, 512, 1024]

    # 再形成：サイズごとのメソッド結果をまとめる
    results_reformed: Dict[int, Dict[str, dict]] = {
        size: {
            **results[f"matrix{cuda_prefix}_{size}_8"],
            **{
                f"BUFF_{p}": results[f"matrix{cuda_prefix}_{size}_{p}"]["BUFF"]
                for p in precisions
            },
        }
        for size in matrix_sizes
    }

    # 実行
    for size in matrix_sizes:
        result = results_reformed[size]

        # 各メソッドの生の実行時間を取り出し、pandas DataFrame に
        exe_times = { m: result[m]["matmul_elapsed_time_nano_secs"]
                      for m in list(compression_methods) + [f"BUFF_{p}" for p in precisions] }
        exe_df = pd.DataFrame([exe_times])  # shape: (1, methods)

        # スループット = 元サイズ / 実行時間（ns）
        original_size = next(iter(result.values()))["compression_statistics"]["uncompressed_size"]
        tp_df = exe_df.map(lambda x: original_size / x)

        # ディレクトリ準備
        exe_dir = os.path.join(out_dir, "exe_time")
        tp_dir = os.path.join(out_dir, "throughput")
        stack_dir = os.path.join(out_dir, "stacked")
        os.makedirs(exe_dir, exist_ok=True)
        os.makedirs(tp_dir, exist_ok=True)
        os.makedirs(stack_dir, exist_ok=True)

        symbol = f"{size}x{size}"

        # 実行時間プロット
        plot_comparison2(
            exe_df.columns.tolist(),
            exe_df.iloc[0].tolist(),
            f"MatMul Execution Time (ns) for {symbol}",
            "Execution Times (ns)",
            os.path.join(exe_dir, f"{symbol}_exe_time_matmul.png"),
        )

        # スループットプロット
        plot_comparison2(
            tp_df.columns.tolist(),
            tp_df.iloc[0].tolist(),
            f"MatMul Throughput (GB/s) for {symbol}",
            "Throughput (GB/s)",
            os.path.join(tp_dir, f"{symbol}_throughput.png"),
        )

        # セグメント化実行時間（Stacked bar 用）を準備
        seg_times: Dict[str, List[ExecutionTimesWithOthers]] = {}
        for m in compression_methods:
            seg = result[m]["matmul_segmented_execution_times"]
            # "others" に合計時間を追加
            seg["others"] = result[m]["matmul_elapsed_time_nano_secs"]
            seg_times[m] = [seg]

        # ラベルマッピングのパッチ関数
        def patch_label(d):
            d["BUFF"]["sum_nanos"] = ["Sum"]
            for p in precisions:
                d[f"BUFF_{p}"] = d["BUFF"]  # type: ignore

        # 積み上げ棒グラフ
        plot_absolute_stacked_execution_times_for_methods(
            seg_times,
            0,
            symbol,
            os.path.join(stack_dir, f"{symbol}_stacked_matmul.png"),
            patch_label_mapping=patch_label,
        )


if __name__ == "__main__":
    main()
    # main2()
