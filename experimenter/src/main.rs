use glob::glob;
use std::{
    fs::File,
    io::{self, BufRead as _, BufReader, Read, Seek, Write as _},
    iter::{self},
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use clap::{Args, Parser, Subcommand};
use ebi::{
    api::{
        decoder::{ChunkId, Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::query::{Predicate, Range, RangeValue, RoaringBitmap},
    encoder::ChunkOption,
};
use quick_impl::QuickImpl;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Cli {
    #[arg(long, short)]
    input: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressionConfig {
    chunk_option: ChunkOption,
    compressor_config: CompressorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FilterConfig {
    predicate: Predicate,
    chunk_id: Option<ChunkId>,
    bitmask: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FilterMaterializeConfig {
    predicate: Predicate,
    chunk_id: Option<ChunkId>,
    bitmask: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MaterializeConfig {
    chunk_id: Option<ChunkId>,
    bitmask: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressStatistics {
    compression_elapsed_time_nano_secs: u128,
    uncompressed_size: u64,
    compressed_size: u64,
    compressed_size_chunk_only: u64,
    compression_ratio: f64,
    compression_ratio_chunk_only: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutputWrapper<T> {
    version: String,
    config_path: String,
    compression_config: CompressionConfig,
    command_specific: T,
    elapsed_time_nanos: u128,
    input_filename: String,
    datetime: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
#[serde(tag = "command_type")]
enum Output {
    #[quick_impl(impl From)]
    Compress(OutputWrapper<CompressStatistics>),
    #[quick_impl(impl From)]
    Filter(OutputWrapper<FilterConfig>),
    #[quick_impl(impl From)]
    FilterMaterialize(OutputWrapper<FilterMaterializeConfig>),
    #[quick_impl(impl From)]
    Materialize(OutputWrapper<MaterializeConfig>),
}

#[derive(Debug, Clone, Args)]
struct ConfigPath {
    #[arg(long, short)]
    config: String,
}

impl ConfigPath {
    fn read<T: serde::de::DeserializeOwned>(&self) -> anyhow::Result<T> {
        let file = std::fs::File::open(&self.config)
            .context(format!("Failed to open config file: {}", self.config))?;
        let reader = std::io::BufReader::new(file);
        let config = serde_json::from_reader(reader)
            .context(format!("Failed to parse config file: {}", self.config))?;
        Ok(config)
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    CreateFilterConfig,
    CreateDefaultCompressorConfig,
    Compress(ConfigPath),
    Filter(ConfigPath),
    FilterMaterialize(ConfigPath),
    Materialize(ConfigPath),
}

impl Commands {
    pub fn dir_name(&self) -> &'static str {
        match self {
            Commands::Compress(_) => "compress",
            Commands::Filter(_) => "filter",
            Commands::FilterMaterialize(_) => "filter_materialize",
            Commands::Materialize(_) => "materialize",
            Commands::CreateFilterConfig => "filter_config",
            Commands::CreateDefaultCompressorConfig => "compressor_config",
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let files = glob(&cli.input)
        .context("Failed to read glob pattern")?
        .collect::<Result<Vec<PathBuf>, _>>()
        .context("Failed to collect glob pattern")?;
    let n_files = files.len();
    for (i, path) in files.into_iter().enumerate() {
        println!(
            "[{:03}/{:03}] Processing file: {}",
            i,
            n_files,
            path.display(),
        );
        if let Err(e) = process_file(&path, cli.clone()) {
            eprintln!("Failed to process file {}: {:?}", path.display(), e);
        } else {
            println!(
                "[{:03}/{:03}] Finished processing file: {}",
                i + 1,
                n_files,
                path.display(),
            );
        }
    }

    Ok(())
}

fn process_file(filename: impl AsRef<Path>, cli: Cli) -> anyhow::Result<()> {
    let filename2 = filename.as_ref().to_string_lossy().to_string();
    let cli2 = cli.clone();

    let output: Output = match cli.command {
        Commands::Compress(args) => compress_command(filename, args.read()?, args)?.into(),
        Commands::Filter(args) => filter_command(filename, args.read()?, args)?.into(),
        Commands::FilterMaterialize(args) => {
            filter_materialize_command(filename, args.read()?, args)?.into()
        }
        Commands::Materialize(args) => materialize_command(filename, args.read()?, args)?.into(),
        Commands::CreateFilterConfig => {
            return create_config_command(filename);
        }
        Commands::CreateDefaultCompressorConfig => {
            return create_default_compressor_config(filename);
        }
    };

    save_output_json(output, &filename2, &cli2.command)?;

    Ok(())
}

fn save_output_json(output: Output, filename: &str, command: &Commands) -> anyhow::Result<()> {
    // Get the directory where the filename is located
    let file_dir = Path::new(filename)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("result");
    let base_dir = Path::new(filename).file_stem().unwrap().to_str().unwrap();
    let command_dir = command.dir_name();

    // Create the save directory path based on the file's location
    let save_dir = file_dir.join(base_dir).join(command_dir);

    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&save_dir)?;

    // Determine the next available filename
    let mut file_number = 0;
    let output_file = loop {
        let output_filename = save_dir.join(format!("{:03}.json", file_number));
        if !output_filename.exists() {
            break output_filename;
        }
        file_number += 1;
    };

    // Write the output to JSON
    write_output_json(output, output_file)?;

    Ok(())
}

fn write_output_json(
    output: impl Into<Output>,
    output_filename: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let output_json =
        serde_json::to_string_pretty(&output.into()).context("Failed to serialize output")?;

    let output_file = std::fs::File::create(output_filename.as_ref()).context(format!(
        "Failed to create output file: {}",
        output_filename.as_ref().display()
    ));
    let output_file = match output_file {
        Ok(file) => file,
        Err(e) => {
            eprintln!("{}\n", output_filename.as_ref().display());
            eprintln!("{}", output_json);
            return Err(e);
        }
    };

    let mut writer = std::io::BufWriter::new(output_file);
    let write_result = writer
        .write_all(output_json.as_bytes())
        .context("Failed to write output");
    if let Err(e) = write_result {
        eprintln!("{}\n", output_filename.as_ref().display());
        eprintln!("{}", output_json);
        return Err(e);
    }

    Ok(())
}

fn get_compress_statistics<R: Read + Seek>(
    decoder: &Decoder<R>,
) -> anyhow::Result<CompressStatistics> {
    let compression_elapsed_time_nano_secs = decoder.footer().compression_elapsed_time_nano_secs();
    let uncompressed_size = decoder.footer().number_of_records() * size_of::<f64>() as u64;
    let compressed_size = decoder.total_file_size();
    let compressed_size_chunk_only = decoder.total_chunk_size();
    let compression_ratio = compressed_size as f64 / uncompressed_size as f64;
    let compression_ratio_chunk_only = compressed_size_chunk_only as f64 / uncompressed_size as f64;

    Ok(CompressStatistics {
        compression_elapsed_time_nano_secs,
        uncompressed_size,
        compressed_size,
        compressed_size_chunk_only,
        compression_ratio,
        compression_ratio_chunk_only,
    })
}

/// # Examples
/// ```rust
/// assert_eq!(decimal_places("1.0"), 1);
/// assert_eq!(decimal_places("1.0e-1"), 2);
/// assert_eq!(decimal_places("3.33", 2));
/// assert_eq!(decimal_places("100", 0));
/// assert_eq!(decimal_places("10e-18", 18));
/// assert_eq!(decimal_places("1.234e-3", 6));
/// assert_eq!(decimal_places("1e3", 0));
/// ```
fn decimal_places(s: &str) -> usize {
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

fn get_appropriate_scale(filename: impl AsRef<Path>) -> anyhow::Result<Option<u32>> {
    let reader = BufReader::new(File::open(filename.as_ref()).context(format!(
        "Failed to open file.: {}",
        filename.as_ref().display()
    ))?);
    let mut decimal_position = 0;
    for line in reader.lines() {
        let line = line.context("Failed to read line")?;
        let max_decimal_position = line
            .split(',')
            .filter_map(|s| {
                if s.is_empty() {
                    None
                } else {
                    let s = s.trim();
                    Some(decimal_places(s))
                }
            })
            .max();

        if let Some(max_scale) = max_decimal_position {
            decimal_position = decimal_position.max(max_scale);
        }
    }

    Ok(10u32.checked_pow(decimal_position as u32))
}

fn create_default_compressor_config(filename: impl AsRef<Path>) -> anyhow::Result<()> {
    let scale = get_appropriate_scale(filename.as_ref())?;
    let mut configs: Vec<(&'static str, CompressorConfig)> = vec![
        (
            "uncompressed",
            CompressorConfig::uncompressed().build().into(),
        ),
        ("chimp128", CompressorConfig::chimp128().build().into()),
        ("chimp", CompressorConfig::chimp().build().into()),
        (
            "elf_on_chimp",
            CompressorConfig::elf_on_chimp().build().into(),
        ),
        ("elf", CompressorConfig::elf().build().into()),
        ("gorilla", CompressorConfig::gorilla().build().into()),
        ("rle", CompressorConfig::rle().build().into()),
    ];
    if let Some(scale) = scale {
        configs.push((
            "delta_sprintz",
            CompressorConfig::delta_sprintz()
                .scale(scale)
                .build()
                .into(),
        ));
        configs.push(("buff", CompressorConfig::buff().scale(scale).build().into()));
    } else {
        println!(
            "Failed to get appropriate scale for {}",
            filename.as_ref().display()
        );
    }
    let outdir = std::env::current_dir()
        .context("Failed to get current directory")?
        .join("compressor_configs")
        .join(
            filename
                .as_ref()
                .file_stem()
                .context("Failed to get file stem")?,
        );
    std::fs::create_dir_all(&outdir).context(format!(
        "Failed to create output directory: {}",
        outdir.display()
    ))?;
    for (name, config) in configs {
        let output_filename = outdir.join(format!("{name}.json"));
        let mut f = File::create(&output_filename).context(format!(
            "Failed to create output file: {}",
            output_filename.as_path().display()
        ))?;
        serde_json::to_writer_pretty(&mut f, &config)
            .context(format!("Failed to write {} compressor config", name))?;
    }

    Ok(())
}

fn create_config_command(filename: impl AsRef<Path>) -> anyhow::Result<()> {
    let encoder_input = EncoderInput::from_file(filename.as_ref()).context(format!(
        "Failed to create encoder input from input file.: {}",
        filename.as_ref().display()
    ))?;
    let encoder_output = EncoderOutput::from_vec(Vec::new());

    let mut encoder = Encoder::new(
        encoder_input,
        encoder_output,
        ChunkOption::RecordCount(8192),
        CompressorConfig::uncompressed().build(),
    );
    encoder.encode().context("Failed to encode")?;
    let mut writer = encoder.into_output().into_inner();
    writer.seek(io::SeekFrom::Start(0))?;
    let decoder_input = DecoderInput::from_reader(writer);
    let mut decoder = Decoder::new(decoder_input)?;

    let _min = decoder
        .min(None, None)
        .context("Failed to get min value from decoder")?;
    let _max = decoder
        .max(None, None)
        .context("Failed to get max value from decoder")?;
    let sum = decoder
        .sum(None, None)
        .context("Failed to get sum value from decoder")?;
    let avg = sum / decoder.footer().number_of_records() as f64;

    let base_dir = Path::new(filename.as_ref())
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let output_dir = filename
        .as_ref()
        .parent()
        .ok_or(anyhow::anyhow!(
            "Failed to get parent directory: {}",
            filename.as_ref().display()
        ))?
        .parent()
        .ok_or(anyhow::anyhow!(
            "Failed to get parent directory: {}",
            filename.as_ref().display()
        ))?
        .join("filter_config")
        .join(base_dir);

    let greater_predicate =
        Predicate::Range(Range::new(RangeValue::Inclusive(avg), RangeValue::None));

    let mut out_buf = [0u8; 8];
    let mut out = DecoderOutput::from_writer(&mut out_buf[..]);
    let number_of_records = decoder.footer().number_of_records();
    let bitmask = RoaringBitmap::from_iter(iter::once(number_of_records as u32 / 2));
    decoder
        .materialize(&mut out, Some(&bitmask), None)
        .context("Failed to materialize")?;

    let eq_predicate = Predicate::Eq(f64::from_le_bytes(out_buf));

    let ne_predicate = Predicate::Ne(f64::from_le_bytes(out_buf));

    std::fs::create_dir_all(&output_dir).context(format!(
        "Failed to create output directory: {}",
        output_dir.display()
    ))?;

    for (pred, name) in &[
        (greater_predicate, "greater"),
        (eq_predicate, "eq"),
        (ne_predicate, "ne"),
    ] {
        let output_filename = output_dir.join(format!("{name}.json"));
        dbg!(&output_filename);
        let mut f = File::create(&output_filename).context(format!(
            "Failed to create output file: {}",
            output_filename.display()
        ))?;
        serde_json::to_writer_pretty(
            &mut f,
            &FilterConfig {
                predicate: *pred,
                chunk_id: None,
                bitmask: None,
            },
        )
        .context(format!("Failed to write {} predicate config", name))?;
    }

    Ok(())
}

fn compress_command(
    filename: impl AsRef<Path>,
    config: CompressionConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<CompressStatistics>> {
    let CompressionConfig {
        compressor_config,
        chunk_option,
    } = config.clone();
    let now = chrono::Utc::now();

    let encoded_filename = filename.as_ref().with_extension("ebi");
    let encoder_input = EncoderInput::from_file(filename.as_ref()).context(format!(
        "Failed to create encoder input from input file.: {}",
        filename.as_ref().display()
    ))?;
    let encoder_output = EncoderOutput::from_file(encoded_filename.as_path()).context(format!(
        "Failed to create encoder output from output file.: {}",
        encoded_filename.as_path().display()
    ))?;

    let mut encoder = Encoder::new(
        encoder_input,
        encoder_output,
        chunk_option,
        compressor_config,
    );

    let start = std::time::Instant::now();
    encoder.encode().context("Failed to encode")?;
    let elapsed_time_nanos = start.elapsed().as_nanos();

    drop(encoder);

    let decoder = Decoder::new(DecoderInput::from_file(encoded_filename.as_path())?)
        .context("Failed to create decoder from encoded file")?;

    let statistics = get_compress_statistics(&decoder)?;

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config,
        compression_config: config,
        command_specific: statistics,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
    };

    Ok(output)
}

fn filter_command(
    filename: impl AsRef<Path>,
    config: FilterConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<FilterConfig>> {
    let FilterConfig {
        predicate,
        chunk_id,
        bitmask,
    } = config.clone();
    let bitmask = bitmask.map(ebi::decoder::query::RoaringBitmap::from_iter);
    let now = chrono::Utc::now();
    println!("{:?}", predicate);

    let mut decoder =
        Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
            "Failed to create decoder from input file.: {}",
            filename.as_ref().display()
        ))?;

    let start = std::time::Instant::now();
    let bitmask = decoder
        .filter(predicate, bitmask.as_ref(), chunk_id)
        .context("Failed to perform filter")?;
    let elapsed_time_nanos = start.elapsed().as_nanos();

    println!("filtered: {:?}", bitmask);

    let compression_config = CompressionConfig {
        chunk_option: *decoder.header().config().chunk_option(),
        compressor_config: *decoder.footer().compressor_config(),
    };

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
    };

    Ok(output)
}

fn filter_materialize_command(
    filename: impl AsRef<Path>,
    config: FilterMaterializeConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<FilterMaterializeConfig>> {
    let FilterMaterializeConfig {
        predicate,
        chunk_id,
        bitmask,
    } = config.clone();
    let bitmask = bitmask.map(ebi::decoder::query::RoaringBitmap::from_iter);
    let now = chrono::Utc::now();
    println!("{:?}", predicate);

    let mut decoder =
        Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
            "Failed to create decoder from input file.: {}",
            filename.as_ref().display()
        ))?;
    let decoded_filename = filename.as_ref().with_extension("filter_materialized.bin");
    let mut decoder_output =
        DecoderOutput::from_file(decoded_filename.as_path()).context(format!(
            "Failed to create decoder output from output file.: {}",
            decoded_filename.as_path().display()
        ))?;

    let start = std::time::Instant::now();
    decoder
        .filter_materialize(&mut decoder_output, predicate, bitmask.as_ref(), chunk_id)
        .context("Failed to perform filter materialize")?;
    let elapsed_time_nanos = start.elapsed().as_nanos();

    let compression_config = CompressionConfig {
        chunk_option: *decoder.header().config().chunk_option(),
        compressor_config: *decoder.footer().compressor_config(),
    };

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
    };

    Ok(output)
}

fn materialize_command(
    filename: impl AsRef<Path>,
    config: MaterializeConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<MaterializeConfig>> {
    let MaterializeConfig { chunk_id, bitmask } = config.clone();
    let bitmask = bitmask.map(ebi::decoder::query::RoaringBitmap::from_iter);
    let now = chrono::Utc::now();
    println!("{:?} {:?}", chunk_id, &bitmask);

    let mut decoder =
        Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
            "Failed to create decoder from input file.: {}",
            filename.as_ref().display()
        ))?;
    let decoded_filename = filename.as_ref().with_extension("materialized.bin");
    let mut decoder_output =
        DecoderOutput::from_file(decoded_filename.as_path()).context(format!(
            "Failed to create decoder output from output file.: {}",
            decoded_filename.as_path().display()
        ))?;

    let start = std::time::Instant::now();
    decoder
        .materialize(&mut decoder_output, bitmask.as_ref(), chunk_id)
        .context("Failed to perform filter materialize")?;
    let elapsed_time_nanos = start.elapsed().as_nanos();

    let compression_config = CompressionConfig {
        chunk_option: *decoder.header().config().chunk_option(),
        compressor_config: *decoder.footer().compressor_config(),
    };

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
    };

    Ok(output)
}
