import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(data, title, y_label, output_path, max_value=None):
    methods = list(data.keys())
    values = list(data.values())

    if max_value is None:
        max_value = max(values) * 1.1

    plt.figure(figsize=(12, 8))
    plt.bar(methods, values, color='blue')
    plt.ylim(0, max_value)
    plt.title(title)
    plt.xlabel('Compression Method')
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_boxplot(data, title, y_label, output_path):
    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    data.boxplot(grid=False)

    # Customize the plot
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        sys.exit(1)

    path = sys.argv[1]

    with open(path, 'r') as file:
        all_output = json.load(file)

    # smaller is better
    average_compression_ratios = {}
    average_compression_throughput = {}
    average_decompression_throughput = {}

    for dataset_name, methods in all_output.items():
        print(f"Processing {dataset_name}")
        if dataset_name == "command_type":
            continue
        for method_name, output in methods.items():
            compression_statistics = output['compress']["command_specific"]
            ratio = compression_statistics['compression_ratio']
            throughput = compression_statistics['uncompressed_size'] / \
                compression_statistics['compression_elapsed_time_nano_secs']
            decompression_throughput = compression_statistics['uncompressed_size'] / \
                output["materialize"]["elapsed_time_nanos"]

            if method_name in average_compression_ratios:
                average_compression_ratios[method_name] += ratio
                average_compression_throughput[method_name] += throughput
                average_decompression_throughput[method_name] += decompression_throughput
            else:
                average_compression_ratios[method_name] = ratio
                average_compression_throughput[method_name] = throughput
                average_decompression_throughput[method_name] = decompression_throughput

    num_datasets = len(all_output)
    for method in average_compression_ratios:
        average_compression_ratios[method] /= num_datasets
        average_compression_throughput[method] /= num_datasets
        average_decompression_throughput[method] /= num_datasets

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    plot_comparison(
        average_compression_ratios,
        "Average Compression Ratio",
        "Compression Ratio (smaller, better)",
        os.path.join(out_dir, "average_compression_ratios.png"),
        max_value=1.1
    )
    plot_comparison(
        average_compression_throughput,
        "Average Compression Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(out_dir, "average_compression_throughput.png")
    )
    plot_comparison(
        average_decompression_throughput,
        "Average Decompression Throughput (bigger, better)",
        "Throughput (GB/s)",
        os.path.join(out_dir, "average_decompression_throughput.png")
    )

    for dataset_name, output_per_compression_methods in all_output.items():
        if dataset_name == "command_type":
            continue
        print(f"Processing {dataset_name}")
        compression_ratios = {}
        compression_throughputs = {}
        filters_elapsed_nanos = {key: {} for key in ["eq", "ne", "greater"]}
        filter_materializes_elapsed_nanos = {
            key: {} for key in ["eq", "ne", "greater"]}
        materialize_elapsed_nanos = {}

        for compression_method, output in output_per_compression_methods.items():
            compression_statistics = output['compress']["command_specific"]
            compression_ratio = compression_statistics['compression_ratio']
            compression_ratios[compression_method] = compression_ratio

            compression_throughput = compression_statistics['uncompressed_size'] / \
                compression_statistics['compression_elapsed_time_nano_secs']
            compression_throughputs[compression_method] = compression_throughput

            for filter_name in ["eq", "ne", "greater"]:
                filters_elapsed_nanos[filter_name][compression_method] = output['filters'][
                    filter_name]['filter']['elapsed_time_nanos'] / 1_000_000_000.0
                filter_materializes_elapsed_nanos[filter_name][compression_method] = output[
                    'filters'][filter_name]['filter_materialize']['elapsed_time_nanos'] / 1_000_000_000.0

            materialize_elapsed_nanos[compression_method] = output['materialize']['elapsed_time_nanos'] / 1_000_000_000.0

        dataset_out_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)

        plot_comparison(
            compression_ratios,
            f"{dataset_name}: Compression Ratio",
            "Compression Ratio (smaller, better)",
            os.path.join(dataset_out_dir, f"{
                         dataset_name}_compression_ratios.png"),
            max_value=1.1
        )
        plot_comparison(
            compression_throughputs,
            f"{dataset_name}: Compression Throughput",
            "Throughput (GB/s)",
            os.path.join(dataset_out_dir, f"{
                         dataset_name}_compression_throughputs.png")
        )
        for filter_name in ["eq", "ne", "greater"]:
            plot_comparison(
                filters_elapsed_nanos[filter_name],
                f"{dataset_name}: {filter_name} Filter Elapsed Time",
                "Elapsed Time (s)",
                os.path.join(dataset_out_dir, f"{dataset_name}_{
                             filter_name}_filter_elapsed_seconds.png")
            )
            plot_comparison(
                filter_materializes_elapsed_nanos[filter_name],
                f"{dataset_name}: {filter_name} Filter Materialize Elapsed Time",
                "Elapsed Time (s)",
                os.path.join(dataset_out_dir, f"{dataset_name}_{
                             filter_name}_filter_materialize_elapsed_seconds.png")
            )

        plot_comparison(
            materialize_elapsed_nanos,
            f"{dataset_name}: Materialize Elapsed Time",
            "Elapsed Time (s)",
            os.path.join(dataset_out_dir, f"{
                         dataset_name}_materialize_elapsed_seconds.png")
        )

    print("Creating boxplot")
    compression_ratios_data = {method_name: []
                               for method_name in average_compression_ratios.keys()}
    compression_throughput_data = {
        method_name: [] for method_name in average_compression_throughput.keys()}
    decompression_throughput_data = {
        method_name: [] for method_name in average_decompression_throughput.keys()}

    for dataset_name, methods in all_output.items():
        if dataset_name == "command_type":
            continue
        for method_name, output in methods.items():
            compression_statistics = output['compress']["command_specific"]
            ratio = compression_statistics['compression_ratio']
            compression_throughput = compression_statistics['uncompressed_size'] / \
                compression_statistics['compression_elapsed_time_nano_secs']
            decompression_throughput = compression_statistics['uncompressed_size'] / \
                output["materialize"]["elapsed_time_nanos"]

            compression_ratios_data[method_name].append(ratio)
            compression_throughput_data[method_name].append(
                compression_throughput)
            decompression_throughput_data[method_name].append(
                decompression_throughput)

    # Convert to DataFrame
    compression_ratios_df = pd.DataFrame(compression_ratios_data)
    compression_throughput_df = pd.DataFrame(compression_throughput_data)
    decompression_throughput_df = pd.DataFrame(decompression_throughput_data)

    plot_boxplot(compression_ratios_df, "Boxplot for Compression Ratios", "Compression Ratio(smaller, better)", os.path.join(
        out_dir, "boxplot_compression_ratios.png"))
    plot_boxplot(
        compression_throughput_df,
        "Boxplot for Compression Throughput",
        "Compression Throughput (GB/s, bigger, better)",
        os.path.join(out_dir, "boxplot_compression_throughput.png")
    )

    plot_boxplot(decompression_throughput_df,
                 "Boxplot for Decompression Throughput",
                 "Decompression Throughput (GB/s, bigger, better)",
                 os.path.join(out_dir, "boxplot_decompression_throughput.png"))


if __name__ == "__main__":
    main()
