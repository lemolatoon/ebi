/// This file is used to write data processing and plotting.
/// But plotting in Rust is quite hard.
/// I like to use Rust's good typing for data processing,
/// and Python's good plotting libraries for plotting.
/// Once I have written the data processing code in Rust,
/// pass the code to ChatGPT, and get the python equivalent code.
/// Then do the plotting in Python.
use std::{collections::HashMap, path::PathBuf};

use anyhow::Context as _;
use experimenter::AllOutput;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or(anyhow::anyhow!("No path provided"))?;

    let file = std::fs::File::open(path).context("Failed to open file")?;
    let all_output: AllOutput = serde_json::from_reader(file).context("Failed to parse JSON")?;

    // smaller is better
    let average_compression_ratios: HashMap<String, f64> = all_output
        .clone()
        .aggregate_per_compression_method(AllOutput::average_compression_ratio);

    // byte per nano sec == GB/s
    let average_compression_throughput: HashMap<String, f64> = all_output
        .clone()
        .aggregate_per_compression_method(AllOutput::average_compression_throughput);

    let out_dir: PathBuf = "results".into();
    plot_comparison(
        average_compression_ratios,
        "Average Compression Ratio",
        "Compression Ratio(smaller, better)",
        out_dir.join("average_compression_ratios.png"),
        Some(1.1),
    )?;
    plot_comparison(
        average_compression_throughput,
        "Average Compression Throughput",
        "Throughput (GB/s)",
        out_dir.join("average_compression_throughput.png"),
        None,
    )?;

    for (dataset_name, output_per_compression_methods) in all_output.0 {
        let mut compression_ratios = HashMap::new();
        let mut compression_throughputs = HashMap::new();
        let mut filters_elapsed_nanos = HashMap::new();
        let mut filter_materializes_elapsed_nanos = HashMap::new();
        for filter_name in ["eq", "ne", "greater"] {
            filters_elapsed_nanos.insert(filter_name, HashMap::new());
            filter_materializes_elapsed_nanos.insert(filter_name, HashMap::new());
        }
        let mut materialize_elapsed_nanos = HashMap::new();
        for (compression_method, output) in output_per_compression_methods {
            let compression_ratio = output.compress.command_specific.compression_ratio;
            compression_ratios.insert(compression_method.clone(), compression_ratio);
            // byte per nano sec
            let compression_throughput = output.compress.command_specific.uncompressed_size as f64
                / output
                    .compress
                    .command_specific
                    .compression_elapsed_time_nano_secs as f64;
            compression_throughputs.insert(compression_method.clone(), compression_throughput);
            for filter_name in ["eq", "ne", "greater"] {
                let filter_elapsed_time_nanos =
                    output.filters[filter_name].filter.elapsed_time_nanos;
                filters_elapsed_nanos
                    .get_mut(filter_name)
                    .unwrap()
                    .insert(compression_method.clone(), filter_elapsed_time_nanos);
                let filter_materialize_time_nanos = output.filters[filter_name]
                    .filter_materialize
                    .elapsed_time_nanos;
                filter_materializes_elapsed_nanos
                    .get_mut(filter_name)
                    .unwrap()
                    .insert(compression_method.clone(), filter_materialize_time_nanos);
            }
            let materialize_elapsed_time_nanos = output.materialize.elapsed_time_nanos;
            materialize_elapsed_nanos
                .insert(compression_method.clone(), materialize_elapsed_time_nanos);
        }

        let filters_elapsed_seconds: HashMap<_, _> = filters_elapsed_nanos
            .into_iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    v.into_iter()
                        .map(|(k, v)| (k, v as f64 / 1_000_000_000.0))
                        .collect::<HashMap<String, f64>>(),
                )
            })
            .collect();
        let filter_materializes_elapsed_seconds: HashMap<_, _> = filter_materializes_elapsed_nanos
            .into_iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    v.into_iter()
                        .map(|(k, v)| (k, v as f64 / 1_000_000_000.0))
                        .collect::<HashMap<String, f64>>(),
                )
            })
            .collect();
        let materialize_elapsed_seconds: HashMap<_, _> = materialize_elapsed_nanos
            .into_iter()
            .map(|(k, v)| (k, v as f64 / 1_000_000_000.0))
            .collect();

        let out_dir = out_dir.join(dataset_name.as_str());
        plot_comparison(
            compression_ratios,
            format!("{dataset_name}: Compression Ratio"),
            "Compression Ratio(smaller, better)",
            out_dir.join(format!("{dataset_name}_compression_ratios.png")),
            Some(1.1),
        )?;
        plot_comparison(
            compression_throughputs,
            format!("{dataset_name}: Compression Throughput"),
            "Throughput (GB/s)",
            out_dir.join(format!("{dataset_name}_compression_throughputs.png")),
            None,
        )?;
        for filter_name in ["eq", "ne", "greater"] {
            plot_comparison(
                filters_elapsed_seconds[filter_name].clone(),
                format!("{dataset_name}: {filter_name} Filter Elapsed Time"),
                "Elapsed Time (s)",
                out_dir.join(format!(
                    "{dataset_name}_{filter_name}_filter_elapsed_seconds.png"
                )),
                None,
            )?;
            plot_comparison(
                filter_materializes_elapsed_seconds[filter_name].clone(),
                format!("{dataset_name}: {filter_name} Filter Materialize Elapsed Time"),
                "Elapsed Time (s)",
                out_dir.join(format!(
                    "{dataset_name}_{filter_name}_filter_materialize_elapsed_seconds.png"
                )),
                None,
            )?;
        }

        plot_comparison(
            materialize_elapsed_seconds,
            format!("{dataset_name}: Materialize Elapsed Time"),
            "Elapsed Time (s)",
            out_dir.join(format!("{dataset_name}_materialize_elapsed_seconds.png")),
            None,
        )?;
    }

    Ok(())
}

fn plot_comparison(
    _data: HashMap<String, f64>,
    _title: impl AsRef<str>,
    _y_label: &str,
    _output_path: PathBuf,
    _max_value: Option<f64>,
) -> anyhow::Result<()> {
    unimplemented!()
}
