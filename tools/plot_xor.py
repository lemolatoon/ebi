import sys
from statistics import fmean
import sys
import os
import json
from typing import Dict, List, Optional, TypedDict, cast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from plot import AllOutput, AllOutputHandler
from plot2 import (
    create_all_df,
    default_omit_methods,
    default_compression_method_order,
    init_plt_font,
)


def plot_comp_ratio_bar_chart(df: pd.DataFrame, out_path: Path):
    """
    Creates a bar chart for the "Comp Ratio" column from the given DataFrame
    and saves it to the specified directory.

    Args:
        df (pd.DataFrame): DataFrame containing "Method" and "Comp Ratio" columns.
        out_dir (Path): Path to the output directory where the chart will be saved.

    The y-axis is labeled as "Comp Ratio (Smaller is Better)" to indicate that lower values are preferred.
    The figure height is adjusted to be smaller.
    """
    default_fontsize = 12
    small_fontsize = 10
    # Adjust figure size (reduce height)
    plt.figure(figsize=(10, 4))
    plt.bar(df.index, df["Comp Ratio"], color="blue", alpha=0.7)

    plt.xlabel("Method", fontsize=small_fontsize)
    plt.ylabel("Comp Ratio (Smaller is Better)", fontsize=default_fontsize)
    plt.xticks(rotation=45, ha="right", fontsize=small_fontsize)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the chart as an image
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot2.py <json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    out_dir = Path(json_file_path).parent.joinpath(Path(json_file_path).stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    with open(json_file_path, "r") as f:
        all_output = cast(AllOutput, json.load(f))
    all_output: AllOutputHandler = AllOutputHandler(all_output)
    assert "xor_dataset256x1024x1024" in all_output.datasets()
    # make sure there's only one dataset, otherwise we delete others
    for dataset in all_output.datasets():
        if dataset != "xor_dataset256x1024x1024":
            del all_output.data[dataset]
    assert len(all_output.datasets()) == 1

    df = create_all_df(all_output.averaged().map_to_stats())
    df.reset_index(inplace=True)
    df.drop(columns=["Dataset"], inplace=True)
    df = df[~df["Method"].isin(default_omit_methods())]
    df.set_index("Method", inplace=True)

    # Sort the DataFrame by the order of the compression methods
    df = df.reindex(default_compression_method_order())
    print(df)

    plot_comp_ratio_bar_chart(df, out_dir.joinpath("xor_comp_ratio.png"))


if __name__ == "__main__":
    init_plt_font()
    main()
