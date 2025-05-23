use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{Cursor, Seek as _},
    path::Path,
};

use anyhow::Context;
use arrow::{
    array::{Array as _, ArrayRef, Decimal128Array},
    datatypes::DataType,
};
use ebi::{
    api::{
        decoder::{Aggregation, AggregationKind, AggregationResult, Decoder, DecoderInput, Expr},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::{
        expr::Op,
        query::{Predicate, Range, RangeValue},
    },
    encoder::ChunkOption,
    time::{SegmentedExecutionTimes, SerializableSegmentedExecutionTimes},
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpchResult01 {
    pub name: String,
    // Column name to Compression statistics
    pub compression: HashMap<String, CompressStatistics>,
    pub queries_elapsed_time: Vec<u64>,
    pub query_names: Vec<String>,
    pub query_elapsed_time: u64,
    pub result: Vec<AggregationResult>,
    pub timer: SerializableSegmentedExecutionTimes,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpchResult06 {
    pub name: String,
    // Column name to Compression statistics
    pub compression: HashMap<String, CompressStatistics>,
    pub queries_elapsed_time: Vec<u64>,
    pub query_names: Vec<String>,
    pub query_elapsed_time: u64,
    pub result: Vec<AggregationResult>,
    pub timer: SerializableSegmentedExecutionTimes,
}

use crate::{get_compress_statistics, CompressStatistics, DEFAULT_CHUNK_OPTION};

fn extract_decimal_as_f64(
    array: &ArrayRef,
    scale: i8,
    column_name: &str,
) -> anyhow::Result<Vec<f64>> {
    let array = array
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .with_context(|| format!("Expected Decimal128 on column: {}", column_name))?;

    let factor = 10f64.powi(scale as i32);

    Ok((0..array.len())
        .map(|i| {
            if array.is_null(i) {
                f64::NAN
            } else {
                array.value(i) as f64 / factor
            }
        })
        .collect())
}

fn extract_columns(file: File) -> anyhow::Result<BTreeMap<String, Vec<f64>>> {
    let batch_reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut columns = BTreeMap::new();
    for maybe_batch in batch_reader {
        let batch = maybe_batch.with_context(|| "Failed to read batch")?;
        let scheme = batch.schema();
        for (i, field) in scheme.fields().iter().enumerate() {
            let col_name = field.name();
            let data_type = field.data_type();
            let DataType::Decimal128(_, scale) = data_type else {
                anyhow::ensure!(
                    col_name == "l_returnflag" || col_name == "l_linestatus",
                    "Expected l_returnflag or l_linestatus on not Decimal128 data_type {}",
                    data_type
                );
                continue;
            };

            let array = batch.column(i);
            let values = extract_decimal_as_f64(array, *scale, col_name)?;

            columns
                .entry(col_name.to_string())
                .or_insert_with(Vec::new)
                .extend(values);
        }
    }

    Ok(columns)
}

pub fn tpch_command(
    file_dir: impl AsRef<Path>,
) -> anyhow::Result<(HashMap<String, TpchResult01>, HashMap<String, TpchResult06>)> {
    let mut h01_result = None;
    let mut h06_result = None;
    for file in file_dir.as_ref().read_dir()? {
        let path = file?.path();

        match path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Failed to get file name"))?
            .to_str()
            .with_context(|| format!("Invalid Path: {}", path.display()))?
        {
            "h01.parquet" => h01_result = Some(process_h01(&path)?),
            "h06.parquet" => h06_result = Some(process_h06(&path)?),
            _ => {
                println!("Skipping file: {}", path.display());
                continue;
            }
        }
    }

    anyhow::ensure!(h01_result.is_some(), "Failed to process h01.parquet file");
    anyhow::ensure!(h06_result.is_some(), "Failed to process h06.parquet file");

    Ok((h01_result.unwrap(), h06_result.unwrap()))
}

fn decoder(
    values: &[f64],
    config: &CompressorConfig,
    chunk_option: ChunkOption,
) -> anyhow::Result<Decoder<Cursor<Vec<u8>>>> {
    let ei = EncoderInput::from_f64_slice(values);
    let eo = EncoderOutput::from_vec(Vec::new());
    let mut encoder = Encoder::new(ei, eo, chunk_option, *config);

    encoder.encode().context("Failed to encode")?;

    let mut encoder_output = encoder.into_output().into_inner();
    encoder_output.seek(std::io::SeekFrom::Start(0))?;
    let decoder = Decoder::new(DecoderInput::from_reader(encoder_output))
        .context("Failed to create decoder from encoded memory")?;

    Ok(decoder)
}

fn all_configs(scale: u64) -> Vec<CompressorConfig> {
    vec![
        CompressorConfig::uncompressed().build().into(),
        CompressorConfig::chimp128().build().into(),
        CompressorConfig::chimp().build().into(),
        CompressorConfig::elf_on_chimp().build().into(),
        CompressorConfig::elf().build().into(),
        CompressorConfig::gorilla().build().into(),
        CompressorConfig::rle().build().into(),
        CompressorConfig::zstd().build().into(),
        CompressorConfig::gzip().build().into(),
        CompressorConfig::snappy().build().into(),
        CompressorConfig::ffi_alp().build().into(),
        CompressorConfig::delta_sprintz()
            .scale(scale)
            .build()
            .into(),
        CompressorConfig::buff().scale(scale).build().into(),
    ]
}

fn h01_aggregations(column_to_index: &HashMap<String, usize>) -> Vec<Aggregation> {
    vec![
        Aggregation::sum(Expr::Index(column_to_index["l_quantity"])),
        Aggregation::sum(Expr::Index(column_to_index["l_extendedprice"])),
        Aggregation::sum(Expr::Binary {
            op: Op::Mul,
            lhs: Box::new(Expr::Index(column_to_index["l_extendedprice"])),
            rhs: Box::new(Expr::Binary {
                op: Op::Sub,
                lhs: Box::new(Expr::Literal(1.0)),
                rhs: Box::new(Expr::Index(column_to_index["l_discount"])),
            }),
        }),
        Aggregation::sum(Expr::Binary {
            op: Op::Mul,
            lhs: Box::new(Expr::Binary {
                op: Op::Mul,
                lhs: Box::new(Expr::Index(column_to_index["l_extendedprice"])),
                rhs: Box::new(Expr::Binary {
                    op: Op::Sub,
                    lhs: Box::new(Expr::Literal(1.0)),
                    rhs: Box::new(Expr::Index(column_to_index["l_discount"])),
                }),
            }),
            rhs: Box::new(Expr::Binary {
                op: Op::Add,
                lhs: Box::new(Expr::Literal(1.0)),
                rhs: Box::new(Expr::Index(column_to_index["l_tax"])),
            }),
        }),
        Aggregation::sum(Expr::Index(column_to_index["l_discount"])),
    ]
}

pub fn process_h01(path: impl AsRef<Path>) -> anyhow::Result<HashMap<String, TpchResult01>> {
    let mut results = HashMap::new();
    let file = File::open(path.as_ref())?;
    let columns = extract_columns(file).with_context(|| {
        format!(
            "Failed to extract columns from file: {}",
            path.as_ref().display()
        )
    })?;
    let column_to_index = columns
        .keys()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect::<HashMap<_, _>>();

    for comp_conf in tqdm::tqdm(all_configs(100)) {
        let mut compression_statistics = HashMap::new();
        let method_name = format!("{:?}", comp_conf.compression_scheme());
        let mut decoders = Vec::with_capacity(columns.len());
        for (col_name, values) in columns.iter() {
            let decoder = decoder(values, &comp_conf, DEFAULT_CHUNK_OPTION)?;
            compression_statistics.insert(col_name.clone(), get_compress_statistics(&decoder)?);
            decoders.push(decoder);
        }

        let h01_aggregations = h01_aggregations(&column_to_index);

        let number_of_records = decoders[0].footer().number_of_records() as usize;
        // run aggregations
        let mut queries_elapsed_time = Vec::new();
        let mut query_names = Vec::new();
        let mut timer = SegmentedExecutionTimes::new();
        let mut result = {
            let start = std::time::Instant::now();
            let result =
                Aggregation::compute_multiple(&h01_aggregations, &mut decoders, None, &mut timer)?;
            let elapsed_time = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            queries_elapsed_time.push(elapsed_time);
            query_names.push(
                [
                    "SUM(l_quantity)",
                    "SUM(l_extendedprice)",
                    "SUM(l_extendedprice * (1 - l_discount))",
                    "SUM((l_extendedprice * (1 - l_discount)) * (1 + l_tax))",
                    "SUM(l_discount)",
                ]
                .into_iter()
                .fold(String::new(), |acc, x| acc + "\n" + x),
            );
            result
        };
        // push avg,count results manually

        {
            // result[0] is SUM(l_quantity)
            let start = std::time::Instant::now();
            let v =
                AggregationResult::Scalar(result[0].scalar().unwrap() / number_of_records as f64);
            let elapsed = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            result.push(v);
            queries_elapsed_time.push(elapsed);
            query_names.push("AVG(l_quantity)".to_string());
        }
        {
            // result[1] is SUM(l_extendedprice)
            let start = std::time::Instant::now();
            let v =
                AggregationResult::Scalar(result[1].scalar().unwrap() / number_of_records as f64);
            let elapsed = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            result.push(v);
            queries_elapsed_time.push(elapsed);
            query_names.push("AVG(l_extendedprice)".to_string());
        }
        {
            // result[4] is SUM(l_discount)
            let start = std::time::Instant::now();
            let v =
                AggregationResult::Scalar(result[4].scalar().unwrap() / number_of_records as f64);
            let elapsed = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            result[4] = v;
            query_names.push("AVG(l_discount)".to_string());
            queries_elapsed_time.push(elapsed);
        }
        result.push(AggregationResult::Integer(number_of_records));
        // count(*)

        let total_elapsed_time = queries_elapsed_time.iter().sum::<u64>();
        // swap the order of the result for the consistency with SQL
        result.swap(4, 6);
        query_names.swap(4 - 3, 6 - 3);

        let result = TpchResult01 {
            name: "h01".to_string(),
            compression: compression_statistics,
            queries_elapsed_time,
            query_names,
            query_elapsed_time: total_elapsed_time,
            result,
            timer: timer.into(),
        };
        results.insert(method_name, result);
    }

    Ok(results)
}

fn get_h06_predicate(column_name: &str) -> Option<Predicate> {
    match column_name {
        "l_discount" => Some(Predicate::Range(Range::new(
            RangeValue::Inclusive(0.05),
            RangeValue::Inclusive(0.07),
        ))),
        "l_quantity" => Some(Predicate::Range(Range::new(
            RangeValue::None,
            RangeValue::Exclusive(24.0),
        ))),
        _ => None,
    }
}

fn h06_aggregations(column_to_index: &HashMap<String, usize>) -> Vec<Aggregation> {
    vec![Aggregation::new(
        AggregationKind::Sum,
        Expr::Binary {
            op: Op::Mul,
            lhs: Box::new(Expr::Index(
                *column_to_index
                    .get("l_extendedprice")
                    .expect("Column not found"),
            )),
            rhs: Box::new(Expr::Index(
                *column_to_index.get("l_discount").expect("Column not found"),
            )),
        },
    )]
}

pub fn process_h06(path: impl AsRef<Path>) -> anyhow::Result<HashMap<String, TpchResult06>> {
    let mut results = HashMap::new();
    let file = File::open(path.as_ref())?;
    let columns = extract_columns(file).with_context(|| {
        format!(
            "Failed to extract columns from file: {}",
            path.as_ref().display()
        )
    })?;

    let column_to_index = columns
        .keys()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect::<HashMap<_, _>>();

    for comp_conf in tqdm::tqdm(all_configs(100)) {
        let mut compression_statistics = HashMap::new();
        let method_name = format!("{:?}", comp_conf.compression_scheme());
        let mut decoders = Vec::with_capacity(columns.len());
        for (col_name, values) in columns.iter() {
            let decoder = decoder(values, &comp_conf, DEFAULT_CHUNK_OPTION)?;
            compression_statistics.insert(col_name.clone(), get_compress_statistics(&decoder)?);
            decoders.push(decoder);
        }

        let predicate0 = get_h06_predicate("l_discount").unwrap();
        let predicate0_index = column_to_index
            .get("l_discount")
            .copied()
            .expect("Column not found");
        let predicate1 = get_h06_predicate("l_quantity").unwrap();
        let predicate1_index = column_to_index
            .get("l_quantity")
            .copied()
            .expect("Column not found");
        let h06_aggregations = h06_aggregations(&column_to_index);

        let mut query_names = Vec::new();
        let mut queries_elapsed_time = Vec::new();
        let mut timer = SegmentedExecutionTimes::new();

        // generate bitmask
        let bitmask = {
            let start = std::time::Instant::now();
            let bitmask = decoders[predicate0_index].filter(predicate0, None, None)?;
            let elapsed_time = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            queries_elapsed_time.push(elapsed_time);
            query_names.push("l_discount BETWEEN 0.05 AND 0.07".to_string());
            timer += decoders[predicate0_index].segmented_execution_times();
            bitmask
        };
        let bitmask = {
            let start = std::time::Instant::now();
            let bitmask = decoders[predicate1_index].filter(predicate1, Some(&bitmask), None)?;
            let elapsed_time = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            queries_elapsed_time.push(elapsed_time);
            query_names.push("l_quantity < 24".to_string());
            timer += decoders[predicate1_index].segmented_execution_times();
            bitmask
        };
        // run aggregations
        let result = {
            let start = std::time::Instant::now();
            let result = Aggregation::compute_multiple(
                &h06_aggregations,
                &mut decoders,
                Some(&bitmask),
                &mut timer,
            )?;
            let elapsed_time = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            queries_elapsed_time.push(elapsed_time);
            query_names.push("SUM(l_extendedprice * l_discount)".to_string());
            result
        };
        let total_elapsed_time = queries_elapsed_time.iter().sum::<u64>();

        let result = TpchResult06 {
            name: "h06".to_string(),
            compression: compression_statistics,
            query_elapsed_time: total_elapsed_time,
            queries_elapsed_time,
            query_names,
            timer: timer.into(),
            result,
        };
        results.insert(method_name, result);
    }

    Ok(results)
}
