use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::BufRead,
};

use anyhow::Context as _;
use ebi::{
    api::decoder::ChunkId, compressor::CompressorConfig, decoder::query::Predicate,
    encoder::ChunkOption, time::SerializableSegmentedExecutionTimes,
};
use quick_impl::QuickImpl;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub chunk_option: ChunkOption,
    pub compressor_config: CompressorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub predicate: Predicate,
    pub chunk_id: Option<ChunkId>,
    pub bitmask: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterMaterializeConfig {
    pub predicate: Predicate,
    pub chunk_id: Option<ChunkId>,
    pub bitmask: Option<Vec<u32>>,
}

impl From<FilterConfig> for FilterMaterializeConfig {
    fn from(value: FilterConfig) -> Self {
        let FilterConfig {
            predicate,
            chunk_id,
            bitmask,
        } = value;
        Self {
            predicate,
            chunk_id,
            bitmask,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializeConfig {
    pub chunk_id: Option<ChunkId>,
    pub bitmask: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressStatistics {
    pub compression_elapsed_time_nano_secs: u64,
    pub uncompressed_size: u64,
    pub compressed_size: u64,
    pub compressed_size_chunk_only: u64,
    pub compression_ratio: f64,
    pub compression_ratio_chunk_only: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputWrapper<T> {
    pub in_memory: bool,
    pub version: String,
    pub config_path: String,
    pub compression_config: CompressionConfig,
    pub command_specific: T,
    pub elapsed_time_nanos: u64,
    pub execution_times: SerializableSegmentedExecutionTimes,
    pub input_filename: String,
    pub datetime: chrono::DateTime<chrono::Utc>,
    pub result_string: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
pub struct FilterFamilyOutput {
    pub filter: Vec<OutputWrapper<FilterConfig>>,
    pub filter_materialize: Vec<OutputWrapper<FilterMaterializeConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
pub struct AllOutputInner {
    pub compress: Vec<OutputWrapper<CompressStatistics>>,
    pub filters: HashMap<String, FilterFamilyOutput>,
    pub materialize: Vec<OutputWrapper<MaterializeConfig>>,
}

impl AllOutputInner {
    pub fn clear(&mut self) {
        self.compress.clear();
        self.filters.clear();
        self.materialize.clear();
    }
}

/// dataset_name -> compression_method -> AllOutputInner
#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
pub struct AllOutput(pub HashMap<String, HashMap<String, AllOutputInner>>);

impl AllOutput {
    pub fn map<T>(self, f: impl Fn(AllOutputInner) -> T) -> HashMap<String, HashMap<String, T>> {
        self.0
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.into_iter()
                        .map(|(k, v)| (k, f(v)))
                        .collect::<HashMap<String, T>>(),
                )
            })
            .collect()
    }

    pub fn compression_methods(self) -> HashSet<String> {
        self.0.values().flat_map(|v| v.keys()).cloned().collect()
    }

    pub fn dataset_names(self) -> HashSet<String> {
        self.0.keys().cloned().collect()
    }

    pub fn transpose(self) -> HashMap<String, HashMap<String, AllOutputInner>> {
        let mut result = HashMap::new();
        for (dataset_name, result_per_dataset) in self.0 {
            for (compression_method, output) in result_per_dataset {
                let entry = result
                    .entry(compression_method)
                    .or_insert_with(HashMap::new);
                entry.insert(dataset_name.clone(), output);
            }
        }
        result
    }

    pub fn aggregate_per_compression_method<T>(
        self,
        f: impl Fn(HashMap<String, AllOutputInner>) -> T,
    ) -> HashMap<String, T> {
        self.transpose()
            .into_iter()
            .map(|(k, v)| (k, f(v)))
            .collect()
    }

    pub fn average_compression_ratio(dataset_to_output: HashMap<String, AllOutputInner>) -> f64 {
        let total_compression_ratio = dataset_to_output
            .values()
            .map(|output| {
                let n = output.compress.len() as f64;
                output
                    .compress
                    .iter()
                    .map(|c| c.command_specific.compression_ratio)
                    .sum::<f64>()
                    / n
            })
            .sum::<f64>();
        total_compression_ratio / dataset_to_output.len() as f64
    }

    pub fn average_compression_throughput(
        dataset_to_output: HashMap<String, AllOutputInner>,
    ) -> f64 {
        let total_compression_throughput = dataset_to_output
            .values()
            .map(|output| {
                let n = output.compress.len() as f64;
                let elapsed_time = output
                    .compress
                    .iter()
                    .map(|c| c.elapsed_time_nanos as f64)
                    .sum::<f64>();
                let uncompressed_size = output
                    .compress
                    .iter()
                    .map(|c| c.command_specific.uncompressed_size as f64)
                    .sum::<f64>();
                (uncompressed_size / elapsed_time) / n
            })
            .sum::<f64>();

        total_compression_throughput / dataset_to_output.len() as f64
    }

    pub fn average_decompression_throughput(
        dataset_to_output: HashMap<String, AllOutputInner>,
    ) -> f64 {
        let total_decompression_throughput = dataset_to_output
            .values()
            .map(|output| {
                let n = output.compress.len() as f64;
                let elapsed_time = output
                    .materialize
                    .iter()
                    .map(|c| c.elapsed_time_nanos as f64)
                    .sum::<f64>();
                let uncompressed_size = output
                    .compress
                    .iter()
                    .map(|c| c.command_specific.uncompressed_size as f64)
                    .sum::<f64>();
                (uncompressed_size / elapsed_time) / n
            })
            .sum::<f64>();

        total_decompression_throughput / dataset_to_output.len() as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
#[serde(tag = "command_type")]
pub enum Output {
    #[quick_impl(impl From)]
    Compress(OutputWrapper<CompressStatistics>),
    #[quick_impl(impl From)]
    Filter(OutputWrapper<FilterConfig>),
    #[quick_impl(impl From)]
    FilterMaterialize(OutputWrapper<FilterMaterializeConfig>),
    #[quick_impl(impl From)]
    Materialize(OutputWrapper<MaterializeConfig>),
    #[quick_impl(impl From)]
    All(AllOutput),
}

/// # Examples
/// ```rust
/// use experimenter::get_decimal_precision;
/// assert_eq!(get_decimal_precision("1.0"), 1);
/// assert_eq!(get_decimal_precision("1.0e-1"), 2);
/// assert_eq!(get_decimal_precision("3.33"), 2);
/// assert_eq!(get_decimal_precision("100"), 0);
/// assert_eq!(get_decimal_precision("10e-18"), 18);
/// assert_eq!(get_decimal_precision("1.234e-3"), 6);
/// assert_eq!(get_decimal_precision("1e3"), 0);
///
/// assert_eq!(get_decimal_precision("64.2"), 1);
/// assert_eq!(get_decimal_precision("-99"), 0);
/// ```
pub fn get_decimal_precision(s: &str) -> usize {
    if let Some(e_pos) = s.find('e') {
        let (base, exponent) = s.split_at(e_pos);
        let exponent_value: i32 = exponent[1..].parse().unwrap_or(0);

        if let Some(dot_pos) = base.find('.') {
            let decimal_length = base.len() - dot_pos - 1;
            (decimal_length as i32 - exponent_value).max(0) as usize
        } else if exponent_value < 0 {
            (-exponent_value) as usize
        } else {
            0
        }
    } else if let Some(dot_pos) = s.find('.') {
        s.len() - dot_pos - 1
    } else {
        0
    }
}

/// # Examples
/// ```rust
/// use experimenter::get_appropriate_scale;
/// use experimenter::get_decimal_precision;
/// use std::io::BufRead;
/// let data = "1.2,2.0,3.31\n4.0,5.0,6.0\n7.0,8.0,9.0";
/// assert_eq!(get_appropriate_scale(data.as_bytes()).unwrap(), 100, "data: {}", data);
///
/// let data2 = "100,2.0,3.31\n4.0,5.0,6.0\n7.0009e-3,8.0,9.0";
/// assert_eq!(get_appropriate_scale(data2.as_bytes()).unwrap(), 10000_000, "data2: {}", data2);
///
/// let data3 = "100,2.0,3.31\n4.0,5.0,6.0\n7.0009,8.0,9.0";
/// assert_eq!(get_appropriate_scale(data3.as_bytes()).unwrap(), 10000, "data3: {}", data3);
///
/// let data4 = "64.2,49.4,48.8,46.4,47.9,48.7,48.9,49.1,49.0,51.9,51.7,51.3,47.0,46.9,47.5,45.9,44.5,50.7,54.0,52.6,54.2,51.0,53.5,54.2,54.2,52.6,55.5,53.8,54.3,57.4,56.9,50.4,50.1,54.1,49.1,48.8,50.7,51.6,52.6,56.3,59.0,59.4,55.5,57.0,60.8,61.8,57.7,56.1,53.4,51.4,52.6,52.5,57.5,55.1,54.3,63.0,60.0,48.3,55.3,52.2,56.6,54.7,51.9,54.5,58.5,53.4,51.8,53.3,65.6,68.7,58.4,55.1,52.8,53.9,54.8,55.0,52.8,56.1,56.5,56.7,51.4,51.6,53.3,56.4,54.7,54.5,53.4,56.6,53.2,46.6,47.4,52.0,62.2,64.2,59.5,59.0,54.9,54.2,57.8,60.0,61.1,56.2,56.1,54.6,54.5,52.0,56.6,60.4,62.7,61.0,56.5,56.0,53.1,51.1,57.2,56.3,56.5,60.8,60.4,61.5,60.4,59.7,61.5,63.9,63.7,66.7,75.1,77.2,74.6,69.4,67.4,68.2,63.3,60.4,65.6,68.9,68.4,70.1,70.3,68.0,62.6,64.3,72.0,73.9,68.2,67.7,67.2,67.7,68.3,68.2,66.8,64.6,66.7,67.7,68.6,69.3,70.4,72.6,70.6,70.5,69.1,68.6,70.3,69.2,69.2,71.5,71.4,71.9,70.0,76.5,73.4,75.9,74.5,73.8,72.8,74.6,75.8,75.8,75.1,73.6,77.7,78.6,77.4,76.2,73.0,73.9,72.9,74.1,76.6,75.9,76.7,76.1,75.5,74.6,76.7,76.5,76.4,76.6,77.7,78.3,76.9,83.4,81.6,77.2,80.4,79.3,81.1,80.7,80.3,79.4,78.8,79.7,80.1,79.9,77.6,78.6,78.5,78.3,78.4,78.9,-99,-99,85.7,82.1,79.5,81.0,78.9,77.6,79.4,79.8,79.0,77.5,78.3,76.2,77.4,76.0,74.9,73.8,75.6,75.8,76.0,73.0,72.9,76.3,79.6,84.2,76.0,76.0,76.0,75.9,75.7,70.6,73.6,75.1,75.2,72.5,69.4,69.9,71.0,72.0,74.0,71.9,67.9,67.8,69.3,68.6,66.0,65.3,62.6,65.2,66.7,67.0,70.6,69.0,69.0,74.0,72.9,73.3,73.6,71.1,72.4,70.6,69.8,67.8,66.0,71.6,68.9,69.1,66.0,67.3,69.5,65.4,63.0,62.2,63.2,63.4,62.4,67.4,66.4,66.9,68.9,70.3,71.6,71.3,67.9,69.9,65.3,60.6,61.6,55.0,52.5,57.2,58.3,74.0,74.8,68.6,59.9,58.4,59.5,64.3,66.8,71.7,65.7,59.0,63.5,60.5,56.4,56.4,60.5,59.3,56.8,52.6,56.6,57.5,57.3,55.9,55.1,57.7,53.8,51.8,59.3,58.2,53.2,52.6,52.9,52.1,54.4,52.3,62.7,62.2,55.6,54.1,52.1,58.1,57.0,55.3,60.8,67.0,66.9,65.7,64.2,63.9,57.3,64.3,66.1,67.4,60.0,54.4,57.7,57.6,62.2,59.8,56.8,64.5,56.7,53.5,54.3,54.1,53.3,59.4,53.2,56.8,55.3,53.3,55.8,58.6,66.0,61.1,56.8,51.9,52.1,58.6,62.8,56.3,54.2,54.0,60.6,58.3,54.7,56.0,51.8,55.8,55.9,55.8,48.9,54.6,51.0,51.4,-99,-99,49.4,48.4,51.2,48.5,54.4,48.5,47.0,42.3,45.9,45.9,54.4,52.5,49.0,50.2,51.7,48.6,49.5,51.6,50.3,50.7,57.7,56.5,55.3,59.8,55.4,53.3,51.4,53.8,52.4,50.9,51.0,49.4,50.1,50.4,56.1,57.2,60.4,65.5,72.4,64.2,61.5,60.2,60.3,58.9,63.6,62.4,60.5,61.6,55.4,50.6,53.2,55.1,55.0,56.9,56.3,55.6,59.5,59.1,61.0,61.2,57.9,58.9,61.3,54.0,60.4,74.1,68.0,68.6,64.7,59.0,58.2,59.4,58.8,57.7,58.6,60.5,60.7,62.6,63.3,63.2,62.5,66.4,63.1,64.9,64.6,62.2,61.8,61.1,60.5,61.0,63.6,64.4,69.0,67.0,64.7,61.8,60.6,61.0,62.4,64.6,66.2,66.3,67.0,67.5,66.3,66.0,66.1,68.0,67.1,65.0,63.8,65.8,67.6,68.0,71.6,71.6,72.0,72.3,73.0,72.7,72.6,71.6,72.7,74.1,72.6,74.0,78.9,81.4,74.4,69.6,65.2,64.7,67.0,67.7,68.8,72.2,71.5,72.8,77.4,73.7,74.4,78.5,78.8,79.7,74.3,69.3,71.2,71.6,71.0,69.5,70.6,72.0,76.0,74.0,73.6,75.7,77.1,76.0,75.6,82.5,87.9,88.1,81.2,78.0,75.1,77.7,79.2,78.6,76.0,76.8,77.9,79.0,77.9,79.0,81.6,80.5,77.8,77.7,81.1,77.8,77.3,79.2,80.2,79.5,77.3,73.6,75.2,78.1,77.6,74.0,76.2,75.6,77.7,86.4,82.4,78.4,76.1,75.4,75.0,72.4,71.0,68.0,70.1,71.9,73.9,76.2,77.1,73.0,68.9,76.4,75.4,72.3,69.4,69.5,69.7,69.8,72.5,72.1,68.2,72.8,68.8,67.5,65.9,69.1,69.6,65.7,66.4,66.4,65.5,67.4,67.9,63.4,63.1,60.8,57.3,57.9,61.7,61.9,61.2,62.2,62.0,64.4,67.4,64.5,62.3,61.9,65.5,63.7,63.7,60.2,60.7,61.1,61.8,61.4,60.1,61.1,61.3,60.1,63.8,57.0,56.9,55.3,55.1,56.5,58.8,59.9,54.7,55.6,56.3,57.1,65.7,76.9,72.1,61.0,60.6,59.1,57.9,56.8,54.6,62.5,61.8,61.4,57.8,57.1,56.1,62.7,63.4,59.8,52.9,51.1,59.7,56.0,52.4,52.1,60.4,52.1,50.5,48.1,48.9,52.7,51.2,53.7,57.1,64.1,59.5,57.2,54.3,56.6,59.8,58.4,66.9,71.1,58.9,59.3,58.7,53.2,49.2,43.9,55.2,51.6,56.6,57.7,59.0,57.7,53.7,49.3,52.8,48.6,50.0,58.2,55.3,50.2,48.2,49.5,54.6,58.7,60.1,52.8,53.9,57.2,67.3,67.9,65.0,62.6,54.3,53.7,52.7,53.8,50.1,59.7,57.7,55.0,55.4,56.2,54.0,54.0,52.7,49.8,49.5,49.6,51.0,51.4,51.0,52.4,56.0,59.0,57.4,52.5,51.4,53.9,50.7,50.8,51.9,53.4,58.3,56.3,54.6,55.8,54.2,49.9,50.2,52.0,52.3,52.0,53.6,50.7,55.6,53.4,52.9,54.0,49.9,54.8,50.6,52.2,52.3,57.0,53.5,54.0,56.1,54.6,54.2,53.6,53.0,54.3,54.5,54.9,54.6,58.1,55.1,52.3,52.9,52.3,54.0,53.9,57.2,56.6,56.1,62.3,63.1,62.1,60.7,60.3,55.4,58.0,62.7,60.1,62.0,59.6,59.0,61.5,68.9,58.1,58.1,57.8,59.1,59.0,59.4,66.9,67.7,67.5,61.8,59.4,60.6,63.4,65.4,69.7,70.9,63.0,59.3,60.4,64.7,65.8,65.8,68.0,66.2,65.2,65.5,63.6,67.1,68.8,71.2,66.3,67.2,71.1,76.4,82.7,72.9,70.8,72.2,70.2,72.6,70.0,70.7,71.1,69.6,74.2,70.9,73.8,75.1,75.3,75.0,77.4,75.4,74.0,76.1,77.2,76.1,75.0,77.8,73.6,72.7,74.8,77.9,75.5,72.4,73.8,73.3,73.7,75.8,71.9,70.2,68.0,71.7,73.7,73.0,71.6,71.8,73.4,73.2,71.6,71.9,72.8,73.9,74.7,80.6,79.1,77.5,77.4,77.0,77.1,77.4,75.5,75.8,77.9,76.6,74.3,75.1,75.7,76.8,76.1,75.0,76.2,77.4,77.4,77.1,80.2,79.7,78.0,76.7,81.4,78.5,84.3,86.8,80.2,75.2,76.0,75.9,78.3,78.5,78.6,77.5,76.6,78.3,78.0,77.0,76.3,75.4,79.4,77.3,77.8,78.4,76.0,74.5,78.3,81.7,79.1,76.6,70.9,73.9,75.5,75.3,73.5,75.2,76.4,79.2,81.4,81.5,76.2,75.9,75.0,75.7,76.1,76.2,74.8,72.6,72.1,70.8,70.0,71.3,70.7";
/// assert_eq!(get_appropriate_scale(data4.as_bytes()).unwrap(), 10, "data4: {}", data4);
///
pub fn get_appropriate_scale<R: BufRead>(reader: R) -> anyhow::Result<u32> {
    let mut max_decimal_precision = 0;
    let mut float_strs = BTreeMap::new();
    for line in tqdm::tqdm(reader.lines()) {
        let line = line.context("Failed to read line")?;
        let strs = line.split(',').filter_map(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.trim().to_string())
            }
        });

        for (precision, s) in strs.map(|s| (get_decimal_precision(&s), s)) {
            max_decimal_precision = max_decimal_precision.max(precision);
            float_strs.entry(precision).or_insert_with(Vec::new).push(s);
        }
    }

    let number_of_records = float_strs.values().map(|v| v.len()).sum::<usize>();
    let mut scale = None;
    const IGNORE_THRESHOLD: f64 = 0.01;
    for (&precision, strs) in tqdm::tqdm(float_strs.iter().rev()) {
        let number_of_records_of_precision = strs.len();
        println!(
            "number_of_records_of_precision: {}",
            number_of_records_of_precision
        );
        if (number_of_records_of_precision as f64 / number_of_records as f64) < IGNORE_THRESHOLD {
            println!(
                "Skipping precision: {}, {} / {} ({}) < THRESHOLD ({})",
                precision,
                number_of_records_of_precision,
                number_of_records,
                number_of_records_of_precision as f64 / number_of_records as f64,
                IGNORE_THRESHOLD
            );
            continue;
        }
        scale = Some(10u32.checked_pow(precision as u32).context(format!(
            "Failed to calculate appropriate scale. decimal_precision: {}, decimal_repr: {:?}",
            max_decimal_precision, strs
        ))?);
        println!(
            "Use precision: {}, {} / {} ({})",
            precision,
            number_of_records_of_precision,
            number_of_records,
            number_of_records_of_precision as f64 / number_of_records as f64,
        );
        break;
    }

    scale.context(format!(
        "Failed to calculate appropriate scale from {:?}",
        float_strs
    ))
}
