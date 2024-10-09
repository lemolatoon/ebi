import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

from tqdm import tqdm
from plot import (
    CompressionConfig,
    ExecutionTimes,
    compression_methods,
    plot_boxplot,
    plot_comparison,
)
import polars as pl


class UCR2018Config(TypedDict):
    n: int
    precision: int
    k: int


class CompressStatistics(TypedDict):
    compression_elapsed_time_nano_secs: int
    uncompressed_size: int
    compressed_size: int
    compressed_size_chunk_only: int
    compression_ratio: float
    compression_ratio_chunk_only: float


class UCR2018ResultForOneDataset(TypedDict):
    dataset_name: str
    config: UCR2018Config
    compression_config: CompressionConfig
    elapsed_time_nanos: List[List[int]]
    execution_times: List[List[ExecutionTimes]]
    accuracy: float
    compression_statistics: CompressStatistics


class UCR2018Result(TypedDict):
    results: Dict[str, UCR2018ResultForOneDataset]
    scale_fallbacked_dataset: Optional[List[str]]
    start_time: datetime
    end_time: datetime


def load_json_files_from_directory(directory_path: str) -> Dict[str, UCR2018Result]:
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
compression_methods_with_precision = [
    *compression_methods,
    *[f"BUFF_{p}" for p in precisions],
]


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        sys.exit(1)

    path = sys.argv[1]

    out_dir = "results/ucr2018"
    os.makedirs(out_dir, exist_ok=True)

    results = load_json_files_from_directory(path)

    n_dataset: Optional[int] = None
    dataset_names: Optional[List[str]] = None
    for method_name, result in results.items():
        dataset_names = list(result["results"].keys())
        n_dataset = len(dataset_names)
        break

    knn1_throughput_per_target_vector_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * n_dataset
        for method_name in compression_methods_with_precision
    }

    knn1_throughput_per_all_target_vectors_data: dict[str, Optional[float]] = {
        method_name: [None] * n_dataset
        for method_name in compression_methods_with_precision
    }

    dataset_to_precision = {
        dataset_name: int(
            math.log10(
                dataset_result["compression_config"]["compressor_config"]["scale"]
            )
        )
        for (dataset_name, dataset_result) in results["BUFF"]["results"].items()
    }

    knn1_throughput_per_target_vector_data_per_precision: Dict[
        int, Dict[str, List[Optional[float]]]
    ] = {
        precision: {
            method_name: [] for method_name in compression_methods_with_precision
        }
        for precision in set(dataset_to_precision.values())
    }

    # throughput_per_vector * vector_length
    normalized_throughput_per_method: Dict[str, List[float]] = {
        method_name: [] for method_name in compression_methods_with_precision
    }

    with open("dataset_to_vector_length.json", "r") as json_file:
        dataset_to_vector_length: Dict[str, int] = json.load(json_file)

    for method_name, result in results.items():
        print(f"Method: {method_name}")
        print(f"Start time: {result['start_time']}")
        print(f"End time: {result['end_time']}")
        # print(f"Scale fallbacked datasets: {result['scale_fallbacked_dataset']}")
        # print()

        for i, (dataset_name, dataset_result) in enumerate(result["results"].items()):
            # print(f"Dataset: {dataset_name}, precision: {dataset_to_precision[dataset_name]}")
            # print(f"Config: {dataset_result['config']}")
            # print(f"Compression config: {dataset_result['compression_config']}")
            # print(f"Accuracy: {dataset_result['accuracy']}")
            # print(f"Compression statistics: {dataset_result['compression_statistics']}")

            original_dataset_size = dataset_result["compression_statistics"][
                "uncompressed_size"
            ]

            average_elapsed_time_per_vector = fmean(
                [fmean(times) for times in dataset_result["elapsed_time_nanos"]]
            )
            average_elapsed_time_per_all_vector = fmean(
                [sum(times) for times in dataset_result["elapsed_time_nanos"]]
            )

            average_throughput_per_target_vector = (
                original_dataset_size / average_elapsed_time_per_vector
            )
            average_throughput_per_all_vector = (
                original_dataset_size / average_elapsed_time_per_all_vector
            )

            normalized_throughput_per_method[method_name].append(
                average_throughput_per_target_vector
                * dataset_to_vector_length[dataset_name]
            )

            knn1_throughput_per_target_vector_data[method_name][i] = (
                average_throughput_per_target_vector
            )
            knn1_throughput_per_all_target_vectors_data[method_name][i] = (
                average_throughput_per_all_vector
            )

            knn1_throughput_per_target_vector_data_per_precision[
                dataset_to_precision[dataset_name]
            ][method_name].append(average_throughput_per_target_vector)

            # print(f"Average elapsed time: {average_elapsed_time} ns")

            # print()

    average_throughput_per_target_vector_df = pl.DataFrame(
        knn1_throughput_per_target_vector_data
    )
    average_throughput_per_all_target_vectors_df = pl.DataFrame(
        knn1_throughput_per_all_target_vectors_data
    )
    normalized_throughput_per_method_df = pl.DataFrame(normalized_throughput_per_method)

    average_throughput_per_target_vector_df_per_precision = {
        precision: pl.DataFrame(data)
        for precision, data in knn1_throughput_per_target_vector_data_per_precision.items()
    }

    for dataset_index, dataset_name in enumerate(tqdm(dataset_names)):
        dataset_out_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        plot_comparison(
            average_throughput_per_target_vector_df.columns,
            average_throughput_per_target_vector_df.row(dataset_index),
            f"{dataset_name}: 1-NN Throughput per Target Vector",
            "Throughput (GB/s)",
            os.path.join(
                dataset_out_dir,
                f"throughput_per_target_vector.png",
            ),
        )

        plot_comparison(
            average_throughput_per_all_target_vectors_df.columns,
            average_throughput_per_all_target_vectors_df.row(dataset_index),
            f"{dataset_name}: 1-NN Throughput per All Target Vectors",
            "Throughput (GB/s)",
            os.path.join(
                dataset_out_dir,
                f"throughput_per_all_target_vectors.png",
            ),
        )

    barchart_dir = os.path.join(out_dir, "barchart")
    boxplot_dir = os.path.join(out_dir, "boxplot")
    os.makedirs(barchart_dir, exist_ok=True)
    os.makedirs(boxplot_dir, exist_ok=True)
    plot_comparison(
        average_throughput_per_target_vector_df.columns,
        [
            average_throughput_per_target_vector_df[column].mean()
            for column in average_throughput_per_target_vector_df.columns
        ],
        "1-NN Average Throughput per Target Vector",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_throughput_per_target_vector.png"),
    )

    plot_comparison(
        average_throughput_per_all_target_vectors_df.columns,
        [
            average_throughput_per_all_target_vectors_df[column].mean()
            for column in average_throughput_per_all_target_vectors_df.columns
        ],
        "1-NN Average Throughput per All Target Vectors",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_throughput_per_all_target_vectors.png"),
    )

    plot_boxplot(
        average_throughput_per_target_vector_df,
        "1-NN Throughput per Target Vector",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "throughput_per_target_vector.png"),
    )

    plot_boxplot(
        average_throughput_per_all_target_vectors_df,
        "1-NN Throughput per All Target Vectors",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "throughput_per_all_target_vectors.png"),
    )

    plot_boxplot(
        normalized_throughput_per_method_df,
        "Normalized Throughput per Target Vector",
        "Throughput * Vector Length (GB/s)",
        os.path.join(boxplot_dir, "normalized_throughput_per_target_vector.png"),
    )

    for precision in set(dataset_to_precision.values()):
        plot_boxplot(
            average_throughput_per_target_vector_df_per_precision[precision],
            f"1-NN Throughput per Target Vector (Precision: {precision})",
            "Throughput (GB/s)",
            os.path.join(
                boxplot_dir, f"throughput_per_target_vector_precision_{precision}.png"
            ),
        )


if __name__ == "__main__":
    main()
