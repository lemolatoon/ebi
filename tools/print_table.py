from pathlib import Path
import sys
from typing import List
import pandas as pd
from common import default_omit_methods
from plot2 import load_stats_df

datasets_inorder = [
    "Air-Pressure",
    "Basel-temp",
    "Basel-wind",
    "Bird-migration",
    "Bitcoin-price",
    "City-Temp",
    "Dew-Point-Temp",
    "IR-bio-temp",
    "PM10-dust",
    "Stocks-DE",
    "Stocks-UK",
    "Stocks-USA",
    "Wind-dir",
    "Arade/4",
    "Blockchain-tr",
    "CMS/1",
    "CMS/25",
    "CMS/9",
    "Food-prices",
    "Gov/10",
    "Gov/26",
    "Gov/30",
    "Gov/31",
    "Gov/40",
    "Medicare/1",
    "Medicare/9",
    "NYC/29",
    "POI-lat",
    "POI-lon",
    "SD-bench",
]

embedding_data = [
    {
        "Dataset": "Arxiv-Embed",
        "Semantics": "Text Embeddings",
        "Source": "Hugging Face",
        "Num Values": 12800000,
        "Dataset-Type": "Embedding",
        "Tex Name": "mongodb2024arxivembedding",
    },
    {
        "Dataset": "Airbnb-Embed",
        "Semantics": "Text Embeddings",
        # "Source": "MongoDB",
        "Source": "Hugging Face",
        "Num Values": 8532480,
        "Dataset-Type": "Embedding",
        "Tex Name": "mongodb2024airbnbembedding",
    },
]

ucr2018_data = [
    {
        "Dataset": "UCRArchive2018's 129 Datasets",
        "Semantics": "2D Time Series",
        "Source": "UCR Time Series Classification Archive",
        "Num Values": None,
        "Dataset-Type": "2D Time Series",
        "Tex Name": "UCRArchive2018",
    }
]


def generate_combined_latex_table(df: pd.DataFrame) -> str:
    """Generate a combined LaTeX table with dataset overview and statistics."""

    headings = [
        "",
        "Dataset",
        "Semantics",
        # "Source",
        "\# of Records",
    ]
    n_columns = len(headings)
    # loc_specifier = "c" + "|l" * 3 + "|c" * (n_columns - 5) + "|r"
    loc_specifier = "c" + "|l" * 2 + "|r"
    assert n_columns == len(loc_specifier.lower().replace("|", ""))
    latex_table = f"""
    \\begin{{tabular}}{{{loc_specifier}}}
        \\toprule"""

    latex_table += "        " + " & ".join(headings) + " \\\\\n"
    latex_table += """
        \\midrule
"""

    dataset_types = df["Dataset-Type"].unique()

    for dtype in dataset_types:
        subset = df[df["Dataset-Type"] == dtype]
        n_rows = len(subset)

        latex_table += f"        \\multirow{{{n_rows}}}{{*}}"
        if dtype == "Embedding":
            dtype = "Emb"
        latex_table += f"{{\\rotatebox{{90}}{{\\textbf{{{dtype}}}}}}}\n"

        for _, row in subset.iterrows():
            number_of_records_with_commas = "{:,}".format(row["Num Values"])

            row_values = [
                f"{row['Dataset']}~\\cite{{{row['Tex Name']}}}",
                row["Semantics"],
                # row["Source"],
                number_of_records_with_commas,
            ]
            assert len(row_values) + 1 == n_columns

            latex_table += "        & " + " & ".join(row_values) + " \\\\\n"

        midrule_str = "        \\midrule\n"
        latex_table += midrule_str

    # remove the last midrule
    latex_table = latex_table[: -len(midrule_str)]

    latex_table += """
        \\bottomrule
    \\end{tabular}
"""

    return latex_table


def generate_compact_latex_table(df: pd.DataFrame) -> str:
    """Generate a compact LaTeX table with dataset types and datasets listed in a single cell per type."""

    headings = ["Type", "Datasets"]
    loc_specifier = "|l|p{10cm}|"

    latex_table = f"""
    \\begin{{tabular}}{{{loc_specifier}}}
        \\hline"""

    latex_table += "        " + " & ".join(headings) + " \\\\\n"
    latex_table += "        \\hline\n"

    type_mapping = {
        "Time Series": "Time Series",
        "Non-Time Series": "Non-Time Series",
        "Embedding": "Embedding",
    }

    dataset_types = df["Dataset-Type"].unique()

    for dtype in dataset_types:
        datasets = (
            df[df["Dataset-Type"] == dtype]
            .apply(lambda row: f"{row['Dataset']}~\\cite{{{row['Tex Name']}}}", axis=1)
            .tolist()
        )
        datasets_str = ", ".join(datasets)

        latex_table += (
            f"        {type_mapping.get(dtype, dtype)} & {datasets_str} \\\\\n"
        )
        latex_table += "        \\hline\n"

    # remove the last midrule
    latex_table = latex_table.strip("\n")

    latex_table += """
    \\end{tabular}
"""

    return latex_table


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_table.py <stats.json>")
        sys.exit(1)
    json_file_path = sys.argv[1]

    df = load_stats_df(json_file_path)
    df["Dataset-Type"] = df["Time Series"].apply(
        lambda x: "Time Series" if x else "Non-Time Series"
    )

    df = pd.concat([df, pd.DataFrame(embedding_data)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(ucr2018_data)], ignore_index=True)
    # sort df by dataset order
    dataset_order = [
        *datasets_inorder,
        "UCR2018 129 Datasets",
        "Arxiv-Embed",
        "Airbnb-Embed",
    ]
    df = df.sort_values(
        "Dataset", key=lambda x: x.apply(lambda x: dataset_order.index(x))
    )
    compact_table = generate_compact_latex_table(df)
    print(compact_table)


if __name__ == "__main__":
    main()
