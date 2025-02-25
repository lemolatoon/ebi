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
    ExecutionTimesWithOthers,
    SegmentLabelMapping,
    CompressionMethodKeys,
    compression_methods,
    skip_methods,
    plot_absolute_stacked_execution_times_for_methods,
    plot_boxplot,
    execution_times_keys,
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


class UCR2018DecompressionResult(TypedDict):
    n: int
    elapsed_time_nanos: List[int]
    execution_times: List[ExecutionTimes]
    result_string: List[str]


class UCR2018ResultForOneDataset(TypedDict):
    dataset_name: str
    config: UCR2018Config
    compression_config: CompressionConfig
    elapsed_time_nanos: List[List[int]]
    execution_times: List[List[ExecutionTimes]]
    accuracy: float
    compression_statistics: CompressStatistics
    decompression_result: UCR2018DecompressionResult


class UCR2018Result(TypedDict):
    results: Dict[str, UCR2018ResultForOneDataset]
    scale_fallbacked_dataset: Optional[List[str]]
    start_time: datetime
    end_time: datetime


class UCR2018ForAllCompressionMethodsResult(TypedDict):
    results: Dict[str, UCR2018Result]
    start_time: datetime
    end_time: datetime
    dataset_to_vector_length: Dict[str, int]
    dataset_to_scale: Dict[str, int]


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

    knn1_elapsed_time_per_target_vector_data: dict[str, List[Optional[float]]] = {
        method_name: [None] * n_dataset
        for method_name in compression_methods_with_precision
    }

    knn1_throughput_per_all_target_vectors_data: dict[str, Optional[float]] = {
        method_name: [None] * n_dataset
        for method_name in compression_methods_with_precision
    }

    accuracy_per_methods: Dict[str, List[float]] = {
        method_name: [] for method_name in compression_methods_with_precision
    }
    execution_times_per_methods: Dict[str, List[ExecutionTimesWithOthers]] = {
        method_name: [] for method_name in compression_methods_with_precision
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

    compression_ratio_per_method: Dict[str, List[float]] = {
        method_name: [] for method_name in compression_methods
    }
    compression_throughput_per_method: Dict[str, List[float]] = {
        method_name: [] for method_name in compression_methods
    }
    decompression_throughput_per_method: Dict[str, List[float]] = {
        method_name: [] for method_name in compression_methods_with_precision
    }

    with open("dataset_to_vector_length.json", "r") as json_file:
        dataset_to_vector_length: Dict[str, int] = json.load(json_file)

    for method_name, result in results.items():
        print(f"Method: {method_name}")
        if method_name in skip_methods:
            continue
        # print(f"Start time: {result['start_time']}")
        # print(f"End time: {result['end_time']}")
        # print(f"Scale fallbacked datasets: {result['scale_fallbacked_dataset']}")
        # print()

        for i, dataset_name in enumerate(dataset_names):
            dataset_result = result["results"][dataset_name]
            # print(f"Dataset: {dataset_name}, precision: {dataset_to_precision[dataset_name]}")
            # print(f"Config: {dataset_result['config']}")
            # print(f"Compression config: {dataset_result['compression_config']}")
            # print(f"Accuracy: {dataset_result['accuracy']}")
            # print(f"Compression statistics: {dataset_result['compression_statistics']}")

            original_dataset_size = dataset_result["compression_statistics"][
                "uncompressed_size"
            ]

            if dataset_name == "FaceFour":
                compression_throughput = (
                    original_dataset_size
                    / dataset_result["compression_statistics"][
                        "compression_elapsed_time_nano_secs"
                    ]
                )
                time = dataset_result["compression_statistics"][
                    "compression_elapsed_time_nano_secs"
                ] / (10**9)
                print(
                    f"Original dataset size: {original_dataset_size}, Time: {time}, Compression throughput: {compression_throughput}"
                )

            if method_name in compression_methods:
                compression_ratio = (
                    dataset_result["compression_statistics"]["compressed_size"]
                    / original_dataset_size
                )
                compression_ratio_per_method[method_name].append(compression_ratio)
                compression_throughput = (
                    original_dataset_size
                    / dataset_result["compression_statistics"][
                        "compression_elapsed_time_nano_secs"
                    ]
                )
                compression_throughput_per_method[method_name].append(
                    compression_throughput
                )

            decompression_throughput = original_dataset_size / fmean(
                dataset_result["decompression_result"]["elapsed_time_nanos"]
            )

            decompression_throughput_per_method[method_name].append(
                decompression_throughput
            )

            average_elapsed_time_per_vector = fmean(
                [fmean(times) for times in dataset_result["elapsed_time_nanos"]]
            )
            average_elapsed_time_per_all_vector = fmean(
                [sum(times) for times in dataset_result["elapsed_time_nanos"]]
            )

            knn1_elapsed_time_per_target_vector_data[method_name][i] = (
                average_elapsed_time_per_vector
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

            accuracy_per_methods[method_name].append(dataset_result["accuracy"])

            knn1_execution_times: ExecutionTimesWithOthers = {}  # type: ignore
            for key in execution_times_keys:
                knn1_execution_times[key] = fmean(
                    fmean(map(lambda time: time[key], times))
                    for times in dataset_result["execution_times"]
                )

            knn1_execution_times["others"] = average_elapsed_time_per_vector - sum(
                knn1_execution_times.values()
            )
            execution_times_per_methods[method_name].append(knn1_execution_times)

    average_throughput_per_target_vector_df = pl.DataFrame(
        knn1_throughput_per_target_vector_data
    )
    average_throughput_per_all_target_vectors_df = pl.DataFrame(
        knn1_throughput_per_all_target_vectors_data
    )
    average_elapsed_time_per_vector_df = pl.DataFrame(
        knn1_elapsed_time_per_target_vector_data
    )
    normalized_throughput_per_method_df = pl.DataFrame(normalized_throughput_per_method)
    accuracy_per_methods_df = pl.DataFrame(accuracy_per_methods)

    average_throughput_per_target_vector_df_per_precision = {
        precision: pl.DataFrame(data)
        for precision, data in knn1_throughput_per_target_vector_data_per_precision.items()
    }

    compression_ratio_per_method_df = pl.DataFrame(compression_ratio_per_method)
    compression_throughput_per_method_df = pl.DataFrame(
        compression_throughput_per_method
    )
    decompression_throughput_per_method_df = pl.DataFrame(
        decompression_throughput_per_method
    )

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
            note_str="*ALP utilizes SIMD instructions",
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
            note_str="*ALP utilizes SIMD instructions",
        )

        plot_comparison(
            average_elapsed_time_per_vector_df.columns,
            average_elapsed_time_per_vector_df.row(dataset_index),
            f"{dataset_name}: 1-NN Average Elapsed Time per Target Vector",
            "Elapsed Time (ns)",
            os.path.join(
                dataset_out_dir,
                f"elapsed_time_per_target_vector.png",
            ),
            note_str="*ALP utilizes SIMD instructions",
        )

        def patch_label_mapping(d: Dict[CompressionMethodKeys, SegmentLabelMapping]):
            d["BUFF"]["sum_nanos"] = ["Sum"]
            for precision in precisions:
                d[f"BUFF_{precision}"] = d["BUFF"]  # type: ignore

        plot_absolute_stacked_execution_times_for_methods(
            execution_times_per_methods,
            dataset_index,
            dataset_name,
            os.path.join(
                dataset_out_dir,
                f"stacked_execution_times.png",
            ),
            patch_label_mapping=patch_label_mapping,
            note_str="*ALP utilizes SIMD instructions",
        )

    barchart_dir = os.path.join(out_dir, "barchart")
    boxplot_dir = os.path.join(out_dir, "boxplot")
    compression_dir = os.path.join(out_dir, "compression")
    os.makedirs(barchart_dir, exist_ok=True)
    os.makedirs(boxplot_dir, exist_ok=True)
    os.makedirs(compression_dir, exist_ok=True)

    # Compression

    plot_comparison(
        compression_ratio_per_method_df.columns,
        [
            compression_ratio_per_method_df[column].mean()
            for column in compression_ratio_per_method_df.columns
        ],
        "Compression Ratio",
        "Compression Ratio",
        os.path.join(compression_dir, "compression_ratio.png"),
    )

    plot_comparison(
        compression_throughput_per_method_df.columns,
        [
            compression_throughput_per_method_df[column].mean()
            for column in compression_throughput_per_method_df.columns
        ],
        "Compression Throughput",
        "Throughput (GB/s)",
        os.path.join(compression_dir, "compression_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    plot_comparison(
        decompression_throughput_per_method_df.columns,
        [
            decompression_throughput_per_method_df[column].mean()
            for column in decompression_throughput_per_method_df.columns
        ],
        "Decompression Throughput",
        "Throughput (GB/s)",
        os.path.join(compression_dir, "decompression_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    # Compression End

    plot_comparison(
        average_throughput_per_target_vector_df.columns,
        [
            average_throughput_per_target_vector_df[column].mean()
            for column in average_throughput_per_target_vector_df.columns
        ],
        "1-NN Average Throughput per Target Vector",
        "Throughput (GB/s)",
        os.path.join(barchart_dir, "average_throughput_per_target_vector.png"),
        note_str="*ALP utilizes SIMD instructions",
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
        note_str="*ALP utilizes SIMD instructions",
    )

    # Compression

    plot_boxplot(
        compression_ratio_per_method_df,
        "Average Compression Ratio",
        "Compression Ratio",
        os.path.join(compression_dir, "boxplot_compression_ratio.png"),
    )

    plot_boxplot(
        compression_throughput_per_method_df,
        "Average Compression Throughput",
        "Throughput (GB/s)",
        os.path.join(compression_dir, "boxplot_compression_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    plot_boxplot(
        decompression_throughput_per_method_df,
        "Average Decompression Throughput",
        "Throughput (GB/s)",
        os.path.join(compression_dir, "boxplot_decompression_throughput.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    # Compression End

    plot_boxplot(
        average_throughput_per_target_vector_df,
        "1-NN Throughput per Target Vector",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "throughput_per_target_vector.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    plot_boxplot(
        average_throughput_per_all_target_vectors_df,
        "1-NN Throughput per All Target Vectors",
        "Throughput (GB/s)",
        os.path.join(boxplot_dir, "throughput_per_all_target_vectors.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    plot_boxplot(
        normalized_throughput_per_method_df,
        "Normalized Throughput per Target Vector",
        "Throughput * Vector Length (GB/s)",
        os.path.join(boxplot_dir, "normalized_throughput_per_target_vector.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    plot_boxplot(
        accuracy_per_methods_df,
        "Accuracy",
        "Accuracy",
        os.path.join(boxplot_dir, "accuracy.png"),
    )

    plot_boxplot(
        accuracy_per_methods_df.select(["BUFF", *[f"BUFF_{p}" for p in precisions]]),
        "Accuracy on Buff's Different Precision",
        "Accuracy",
        os.path.join(boxplot_dir, "accuracy_on_buffs_different_precision.png"),
        note_str="*ALP utilizes SIMD instructions",
    )

    for precision in set(dataset_to_precision.values()):
        plot_boxplot(
            average_throughput_per_target_vector_df_per_precision[precision],
            f"1-NN Throughput per Target Vector (Precision: {precision})",
            "Throughput (GB/s)",
            os.path.join(
                boxplot_dir, f"throughput_per_target_vector_precision_{precision}.png"
            ),
            note_str="*ALP utilizes SIMD instructions",
        )


if __name__ == "__main__":
    main()
