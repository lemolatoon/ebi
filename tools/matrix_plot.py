import json
import os
import sys
from typing import Dict, Optional, TypedDict, List
from datetime import datetime
from tqdm import tqdm
from plot import (
    ExecutionTimesWithOthers,
    SegmentLabelMapping,
    compression_methods,
    CompressionMethodKeys,
    plot_boxplot,
    plot_comparison,
    plot_absolute_stacked_execution_times_for_methods,
)
import polars as pl


class CompressionConfig(TypedDict):
    chunk_option: (
        dict  # Replace with the correct type if you have a more specific definition
    )
    # Replace with the correct type if you have a more specific definition
    compressor_config: dict


class CompressStatistics(TypedDict):
    compression_elapsed_time_nano_secs: int
    uncompressed_size: int
    compressed_size: int
    compressed_size_chunk_only: int
    compression_ratio: float
    compression_ratio_chunk_only: float


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
                data = json.load(file)
                key = os.path.splitext(filename)[0]  # Remove the .json extension
                results[key] = data

    return results


precisions = [1, 3, 5, 8]
matrix_sizes = [128, 512, 1024]

compression_methods_with_precision = [
    *compression_methods,
    *[f"BUFF_{p}" for p in precisions],
]


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        sys.exit(1)

    path = sys.argv[1]

    out_dir = "results/matmul/"
    os.makedirs(out_dir, exist_ok=True)

    results = load_json_files_from_directory(path)

    results_reformed: Dict[int, Dict[str, MatrixResult]] = {
        size: {
            **results[f"matrix_{size}_8"],
            **{
                f"BUFF_{precision}": results[f"matrix_{size}_{precision}"]["BUFF"]
                for precision in precisions
            },
        }
        for size in matrix_sizes
    }

    segmented_execution_times_prec8: Dict[int, Dict[str, ExecutionTimesWithOthers]] = {
        128: {
            method_name: {
                **results["matrix_128_8"][method_name][
                    "matmul_segmented_execution_times"
                ],
                "others": results["matrix_128_8"][method_name][
                    "matmul_elapsed_time_nano_secs"
                ],
            }
            for method_name in compression_methods
        },
        512: {
            method_name: {
                **results["matrix_512_8"][method_name][
                    "matmul_segmented_execution_times"
                ],
                "others": results["matrix_512_8"][method_name][
                    "matmul_elapsed_time_nano_secs"
                ],
            }
            for method_name in compression_methods
        },
        1024: {
            method_name: {
                **results["matrix_1024_8"][method_name][
                    "matmul_segmented_execution_times"
                ],
                "others": results["matrix_1024_8"][method_name][
                    "matmul_elapsed_time_nano_secs"
                ],
            }
            for method_name in compression_methods
        },
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
                f"{matrix_size}: {result[method_name]["matmul_elapsed_time_nano_secs"]}"
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


if __name__ == "__main__":
    main()
