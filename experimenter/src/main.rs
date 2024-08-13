use std::{io::Write as _, path::Path};

use anyhow::Context as _;
use clap::{Args, Parser, Subcommand};
use ebi::{
    api::{
        decoder::{ChunkId, Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::query::Predicate,
    encoder::ChunkOption,
};
use quick_impl::QuickImpl;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Cli {
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
struct OutputWrapper<T> {
    version: String,
    config_path: String,
    config: T,
    elapsed_time_nanos: u128,
    input_filename: String,
    datetime: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, QuickImpl)]
#[serde(tag = "command_type")]
enum Output {
    #[quick_impl(impl From)]
    Compress(OutputWrapper<CompressionConfig>),
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

    fn empty() -> Self {
        Self {
            config: String::from("<in-memory>"),
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
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
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let cli2 = cli.clone();

    let filename = cli.input;

    let output: Output = match cli.command {
        Commands::Compress(args) => compress_command(filename, args.read()?, args)?.into(),
        Commands::Filter(args) => filter_command(filename, args.read()?, args)?.into(),
        Commands::FilterMaterialize(args) => {
            filter_materialize_command(filename, args.read()?, args)?.into()
        }
        Commands::Materialize(args) => materialize_command(filename, args.read()?, args)?.into(),
    };

    save_output_json(output, &cli2.input, &cli2.command)?;

    Ok(())
}

fn save_output_json(output: Output, filename: &str, command: &Commands) -> anyhow::Result<()> {
    // Get the directory where the filename is located
    let file_dir = Path::new(filename).parent().unwrap().join("result");
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

fn compress_command(
    filename: impl AsRef<Path>,
    config: CompressionConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<CompressionConfig>> {
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

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config,
        config,
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

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        config,
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

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
    };

    // let timestamp = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
    // let output_filename = filename.as_ref().with_file_name(format!(
    //     "{}_{}.filter_materialize.json",
    //     filename.as_ref().file_stem().unwrap().to_string_lossy(),
    //     timestamp
    // ));

    // write_output_json(output, output_filename)?;
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

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
    };

    Ok(output)
}
