import json
import os
import sys
from typing import Dict, Optional, TypedDict, List
from datetime import datetime
from tqdm import tqdm
from plot import compression_methods, plot_boxplot, plot_comparison
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


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        sys.exit(1)

    path = sys.argv[1]

    out_dir = "results/matmul/"
    os.makedirs(out_dir, exist_ok=True)

    results = load_json_files_from_directory(path)

    for key, result in tqdm(results.items()):
        execution_times = {}
        for method_name in compression_methods:
            execution_times[method_name] = result[method_name][
                "matmul_elapsed_time_nano_secs"
            ]

        execution_times_df = pl.DataFrame(execution_times)
        out_path = out_dir + f"{key}_matmul.png"
        plot_comparison(
            execution_times_df.columns,
            execution_times_df.row(0),
            f"MatMul Execution Time (ns) for {key}",
            "Execution Times (ns)",
            out_path,
        )


if __name__ == "__main__":
    main()
