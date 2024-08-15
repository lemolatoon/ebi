use std::collections::{HashMap, HashSet};

use ebi::{
    api::decoder::ChunkId, compressor::CompressorConfig, decoder::query::Predicate,
    encoder::ChunkOption,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializeConfig {
    pub chunk_id: Option<ChunkId>,
    pub bitmask: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressStatistics {
    pub compression_elapsed_time_nano_secs: u128,
    pub uncompressed_size: u64,
    pub compressed_size: u64,
    pub compressed_size_chunk_only: u64,
    pub compression_ratio: f64,
    pub compression_ratio_chunk_only: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputWrapper<T> {
    pub version: String,
    pub config_path: String,
    pub compression_config: CompressionConfig,
    pub command_specific: T,
    pub elapsed_time_nanos: u128,
    pub input_filename: String,
    pub datetime: chrono::DateTime<chrono::Utc>,
    pub result_string: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
pub struct FilterFamilyOutput {
    pub filter: OutputWrapper<FilterConfig>,
    pub filter_materialize: OutputWrapper<FilterMaterializeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
pub struct AllOutputInner {
    pub compress: OutputWrapper<CompressStatistics>,
    pub filters: HashMap<String, FilterFamilyOutput>,
    pub materialize: OutputWrapper<MaterializeConfig>,
}

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
            .map(|output| output.compress.command_specific.compression_ratio)
            .sum::<f64>();
        total_compression_ratio / dataset_to_output.len() as f64
    }

    pub fn average_compression_throughput(
        dataset_to_output: HashMap<String, AllOutputInner>,
    ) -> f64 {
        let total_compression_throughput = dataset_to_output
            .values()
            .map(|output| {
                let elapsed_time = output.compress.elapsed_time_nanos as f64;
                let uncompressed_size = output.compress.command_specific.uncompressed_size as f64;
                uncompressed_size / elapsed_time
            })
            .sum::<f64>();

        total_compression_throughput / dataset_to_output.len() as f64
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
