import os
import sys
from collections import defaultdict
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from common import default_compression_method_order
from plot import get_average_execution_times_ratios, segment_mapping
from plot2 import plot_absolute_stacked_execution_times_for_methods, init_plt_font

def average_tpch_results(data: list[dict]) -> tuple[dict, pd.DataFrame]: 
    """
    Compute average compression statistics and query elapsed time for each method.
    Ignores the 'result' field.
    """
    n = len(data)
    methods = data[0].keys()
    aggregated = {}

    segmented_exec_times_pd = {
        "Inverse Query Elapsed Time": [],
        "ExecTimeRatios": [],
    }
    index = []

    for i, method in enumerate(methods):
        comp_acc = defaultdict(lambda: defaultdict(float))
        query_acc = 0

        for run in data:
            m = run[method]
            # Accumulate compression statistics
            for col, stats in m['compression'].items():
                for stat_name, value in stats.items():
                    comp_acc[col][stat_name] += value
            # Accumulate query elapsed time
            query_acc += m['query_elapsed_time']

        # Compute averages
        comp_avg = {
            col: {stat: comp_acc[col][stat] / n for stat in comp_acc[col]}
            for col in comp_acc
        }
        query_avg = query_acc / n

        aggregated[method] = {
            'compression': comp_avg,
            'query_elapsed_time': query_avg,
        }

        # execute time ratios
        index.append(method)
        segmented_exec_times_pd["Inverse Query Elapsed Time"].append(
            query_avg / 1e9 # Convert to seconds
        )
        timer = get_average_execution_times_ratios(
                [run[method]['timer'] for run in data],
                query_avg
            )
        for segment in timer.keys():
            timer[segment] *= query_avg / 1e9  # Convert to seconds
        segmented_exec_times_pd["ExecTimeRatios"].append(
            timer
        )

    # Create DataFrame for execution times
    segmented_exec_times_df = pd.DataFrame(segmented_exec_times_pd, index=index)
    return aggregated, segmented_exec_times_df

def create_dataframe(aggregated: dict) -> pd.DataFrame:
    """
    Convert averaged results into a pandas DataFrame.

    @return: 
    DataFrame with methods as index and compression statistics as columns.
    Columns are named as 'compression_(column)_(statistic)'.
    Or 'query_elapsed_time' for query elapsed time.
    e.g. compression_l_quantity_compression_elapsed_time_nano_secs, 
    compression_l_extendedprice_compression_ratio,
    compression_l_extendedprice_uncompressed_size,
    """
    rows = {}

    for method, v in aggregated.items():
        row = {}
        for col, stats in v['compression'].items():
            for stat_name, val in stats.items():
                row[f"compression_{col}_{stat_name}"] = val
        row['query_elapsed_time'] = v['query_elapsed_time']
        rows[method] = row

    return pd.DataFrame.from_dict(rows, orient='index')

def load_result(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        data = json.load(f)
    avg, timer_df = average_tpch_results(data)
    timer_df = timer_df.loc[default_compression_method_order()]
    return create_dataframe(avg), timer_df

def main():
    if len(sys.argv) < 2:
        print("No directory path provided")
        sys.exit(1)
    yes = False
    if len(sys.argv) == 3 and sys.argv[2] == "-y":
        yes = True

    base_dir = sys.argv[1]
    assert os.path.isdir(base_dir)
    save_dir = os.path.join(".", "results", "tpch")
    if not yes and os.path.exists(save_dir):
        print(f"Directory {save_dir} already exists. Please remove it first.")
        if input("Do you want to remove it? (y/n): ").strip().lower() != 'y':
            sys.exit(1)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to '{save_dir}'")
    
    df01, timer_df01 = load_result(f"{base_dir}/tpch_01.json")
    df06, timer_df06 = load_result(f"{base_dir}/tpch_06.json")
    df01_simplified, timer_df01_simplified = load_result(f"{base_dir}/tpch_01_simplified.json")

    process_h01(df01, df01_simplified, timer_df01, save_dir)
    process_h06(df06, timer_df06, save_dir)
    process_tpch(df01_simplified, timer_df01_simplified, tpch01_columns, "tpch_01_simplified", save_dir)

def plot_bar_chart(data: pd.DataFrame, metric_cols: list[str], title: str, ylabel: str, save_path: str):
    """
    Plot a bar chart for specified metric columns and save the figure.
    """
    subset = data.loc[default_compression_method_order(), metric_cols].copy()
    subset = subset.rename_axis("method").reset_index()
    melted = subset.melt(id_vars="method", var_name="Metric", value_name=ylabel)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="method", y=ylabel, hue="Metric")
    plt.xlabel(None)
    # plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_box_chart(data: pd.DataFrame, metric_cols: list[str], title: str, ylabel: str, save_path: str):
    """
    Plot a box chart for specified metric columns (one box per method across multiple columns).
    """
    melted = []
    for col in metric_cols:
        for method in data.index:
            melted.append({"method": method, ylabel: data.loc[method, col]})
    melted_df = pd.DataFrame(melted)

    melted_df["method"] = pd.Categorical(
        melted_df["method"],
        categories=default_compression_method_order(),
        ordered=True
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_df, x="method", y=ylabel)
    # plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_query_elapsed_time_with_overlay(
    base_df: pd.DataFrame,
    overlay_df: pd.DataFrame,
    base_label: str,
    overlay_label: str,
    save_path: str,
) -> None:
    """
    Draw a bar chart of query-elapsed time (base_df) and overlay a **horizontal**
    dotted line for each bar showing the corresponding value from overlay_df.
    """
    # make sure both dataframes have the seconds column
    for df in (base_df, overlay_df):
        if "query_elapsed_time_sec" not in df.columns:
            df["query_elapsed_time_sec"] = df["query_elapsed_time"] / 1e9

    methods = default_compression_method_order()
    y_base = base_df.loc[methods, "query_elapsed_time_sec"]
    y_overlay = overlay_df.loc[methods, "query_elapsed_time_sec"]

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        x=methods,
        y=y_base,
        color="tab:blue",
        label=base_label,
    )

    # add a horizontal dotted line across each bar
    first = True
    for bar, y in zip(ax.patches, y_overlay):
        x_left = bar.get_x()
        x_right = x_left + bar.get_width()
        ax.hlines(
            y,
            xmin=x_left,
            xmax=x_right,
            linestyle="--",
            linewidth=2,
            color="black",
            label=overlay_label if first else None,  # add legend label only once
        )
        first = False

    ax.set_ylabel("Execution Time (s)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_query_time(
    df: pd.DataFrame,
    tag: str,
) -> None:
    """
    Print the query-elapsed-time Series (sorted) to stdout
    and draw the bar chart.
    """
    # ensure the seconds column exists
    if "query_elapsed_time_sec" not in df.columns:
        df["query_elapsed_time_sec"] = df["query_elapsed_time"] / 1e9

    s = df.loc[default_compression_method_order(), "query_elapsed_time_sec"]
    s_sorted = s.sort_values()  # ascending
    # stdout
    print(f"\n=== {tag}: sorted query_elapsed_time_sec (s) ===")
    print(s_sorted.to_string())

def process_tpch(df: pd.DataFrame, timer_df: pd.DataFrame, columns: list[str], tag: str, save_dir: str):
    """
    Process a TPCH DataFrame and generate compression ratio, throughput and query time plots.
    """
    throughput_cols = []
    for col in columns:
        size_col = f"compression_{col}_uncompressed_size"
        time_col = f"compression_{col}_compression_elapsed_time_nano_secs"
        throughput_col = f"compression_{col}_compression_throughput"
        df[throughput_col] = df[size_col] / (1024 ** 3) / (df[time_col] / 1e9)  # GB/s
        throughput_cols.append(throughput_col)

    for col in columns:
        ratio_col = f"compression_{col}_compression_ratio"
        throughput_col = f"compression_{col}_compression_throughput"
        plot_bar_chart(
            df,
            [ratio_col],
            f"{tag}: Compression Ratio for {col}",
            "Compression Ratio",
            os.path.join(save_dir, f"{tag}_{col}_compression_ratio_bar.png"),
        )
        plot_bar_chart(
            df,
            [throughput_col],
            f"{tag}: Compression Throughput for {col}",
            "Throughput (GB/s)",
            os.path.join(save_dir, f"{tag}_{col}_compression_throughput_bar.png"),
        )

    ratio_cols = [f"compression_{col}_compression_ratio" for col in columns]
    plot_box_chart(
        df,
        ratio_cols,
        f"{tag}: Compression Ratio (All Columns)",
        "Compression Ratio",
        os.path.join(save_dir, f"{tag}_compression_ratio_box.png"),
    )
    plot_box_chart(
        df,
        throughput_cols,
        f"{tag}: Compression Throughput (All Columns)",
        "Throughput (GB/s)",
        os.path.join(save_dir, f"{tag}_compression_throughput_box.png"),
    )

    df["query_elapsed_time_sec"] = df["query_elapsed_time"] / 1e9
    plot_bar_chart(
        df,
        ["query_elapsed_time_sec"],
        f"{tag}: Query Elapsed Time",
        "Time (s)",
        os.path.join(save_dir, f"{tag}_query_elapsed_time_bar.png"),
    )
    print_query_time(df, tag)

    fig, ax = plt.subplots()
    plot_absolute_stacked_execution_times_for_methods(
        ax,
        timer_df,
        "Throughput",
        "ExecTimeRatios",
        "Execution Time (s)",
        segment_mapping,
        normalized=True,
    )
    fig.savefig(
        os.path.join(save_dir, f"{tag}_execution_time_stacked.png"),
        bbox_inches='tight',
        dpi=300
    )


tpch01_columns = [
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
]

def process_h01(
    df: pd.DataFrame,
    overlay_df: pd.DataFrame,
    timer_df: pd.DataFrame,
    save_dir: str,
) -> None:
    """Process TPC-H Query 1 and generate the extra overlay plot."""
    process_tpch(df, timer_df, tpch01_columns, "tpch_01", save_dir)

    # extra overlay plot
    plot_query_elapsed_time_with_overlay(
        base_df=df,
        overlay_df=overlay_df,
        base_label="Q1",
        overlay_label="Q1 Simplified",
        save_path=os.path.join(save_dir, "tpch_01_query_elapsed_time_overlay.png"),
    )


tpch06_columns = [
    "l_extendedprice",
    "l_discount",
    "l_quantity",
]
def process_h06(df: pd.DataFrame, timer_df: pd.DataFrame, save_dir: str):
    process_tpch(df, timer_df, tpch06_columns, "tpch_06", save_dir)

if __name__ == "__main__":
    init_plt_font()
    main()
