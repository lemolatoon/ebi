use experimenter::{
    AllOutput, AllOutputInner, CompressStatistics, CompressionConfig, FilterConfig,
    FilterFamilyOutput, FilterMaterializeConfig, MaterializeConfig, Output, OutputWrapper,
};
use glob::glob;
use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fs::{self, File},
    io::{self, BufRead as _, BufReader, Read, Seek, Write as _},
    iter::{self},
    path::{Path, PathBuf},
};
use tqdm::tqdm;

use anyhow::Context as _;
use clap::{Args, Parser, Subcommand};
use ebi::{
    api::{
        decoder::{Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::query::{Predicate, Range, RangeValue, RoaringBitmap},
    encoder::ChunkOption,
};

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Cli {
    #[arg(long, short)]
    input: Option<String>,
    #[command(subcommand)]
    command: Commands,
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

#[derive(Debug, Clone, Args)]
struct AllArgs {
    #[arg(long)]
    create_config: bool,
    #[arg(long, short('c'))]
    compressor_config_dir: PathBuf,
    #[arg(long, short('f'))]
    filter_config_dir: PathBuf,
    #[arg(long, short('b'))]
    binary_dir: PathBuf,
    #[arg(long, short('s'))]
    save_dir: PathBuf,
    #[arg(long, short('s'))]
    n: Option<usize>,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    CreateFilterConfig {
        #[arg(long, short)]
        output_dir: Option<PathBuf>,
    },
    CreateDefaultCompressorConfig {
        #[arg(long, short)]
        output_dir: Option<PathBuf>,
    },
    Compress(ConfigPath),
    Filter(ConfigPath),
    FilterMaterialize(ConfigPath),
    Materialize(ConfigPath),
    All(AllArgs),
}

impl Commands {
    pub fn dir_name(&self) -> &'static str {
        match self {
            Commands::Compress(_) => "compress",
            Commands::Filter(_) => "filter",
            Commands::FilterMaterialize(_) => "filter_materialize",
            Commands::Materialize(_) => "materialize",
            Commands::CreateFilterConfig { .. } => "filter_config",
            Commands::CreateDefaultCompressorConfig { .. } => "compressor_config",
            Commands::All { .. } => "all",
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Commands::All(args) = &cli.command {
        let args = args.clone();
        let save_dir = args
            .binary_dir
            .parent()
            .context("Failed to get parent directory")?
            .join("result")
            .join("all");
        let all_outputs = all_command(args)?;
        save_output_json(all_outputs.into(), save_dir)?;

        return Ok(());
    }

    let Some(input) = &cli.input else {
        anyhow::bail!("Input file is required");
    };

    let files = glob(input)
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
        Commands::CreateFilterConfig { output_dir } => {
            return create_config_command(filename, output_dir);
        }
        Commands::CreateDefaultCompressorConfig { output_dir } => {
            return create_default_compressor_config(filename, output_dir);
        }
        Commands::All { .. } => unreachable!(),
    };

    // Get the directory where the filename is located
    let file_dir = Path::new(&filename2)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("result");
    let base_dir = Path::new(&filename2).file_stem().unwrap().to_str().unwrap();
    let command_dir = cli2.command.dir_name();

    // Create the save directory path based on the file's location
    let save_dir = file_dir.join(base_dir).join(command_dir);
    save_output_json(output, save_dir)?;

    Ok(())
}

fn save_output_json(output: Output, save_dir: impl AsRef<Path>) -> anyhow::Result<()> {
    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&save_dir)?;

    // Determine the next available filename
    let mut file_number = 0;
    let output_file = loop {
        let output_filename = save_dir.as_ref().join(format!("{:03}.json", file_number));
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

fn all_command(args: AllArgs) -> anyhow::Result<AllOutput> {
    let AllArgs {
        create_config,
        compressor_config_dir,
        filter_config_dir,
        binary_dir,
        save_dir,
        n,
    } = args;
    let n = n.unwrap_or(1);
    anyhow::ensure!(
        compressor_config_dir.is_dir(),
        "Config directory does not exist or not directory: {}",
        compressor_config_dir.display()
    );
    anyhow::ensure!(
        filter_config_dir.is_dir(),
        "Config directory does not exist or not directory: {}",
        filter_config_dir.display()
    );
    anyhow::ensure!(
        binary_dir.is_dir(),
        "Binary directory does not exist or not directory: {}",
        binary_dir.display()
    );
    anyhow::ensure!(
        save_dir.is_dir(),
        "Save directory does not exist or not directory: {}",
        save_dir.display()
    );

    let entries = fs::read_dir(&binary_dir).context(format!(
        "Failed to read directory: {}",
        binary_dir.display()
    ))?;
    let entries = entries
        .filter_map(|e| {
            let e = e.ok()?;
            let path = e.path();
            if path.is_file() && matches!(path.extension().and_then(|s| s.to_str()), Some("bin")) {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let csv_dir = binary_dir.parent().unwrap();
    let stem_to_csv_entry = fs::read_dir(csv_dir)
        .context(format!("Failed to read directory: {}", csv_dir.display()))?
        .filter_map(|e| {
            let e = e.ok()?;
            let path = e.path();
            if path.is_file()
                && matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("csv" | "txt") | None
                )
            {
                Some(path)
            } else {
                None
            }
        })
        .map(|path| {
            let stem = path.file_stem().unwrap().to_owned();
            (stem, path)
        })
        .collect::<HashMap<_, _>>();

    let mut skip_files = HashSet::new();
    // Create Config!
    if create_config {
        for (_i, binary_file) in
            tqdm(entries.clone().into_iter().enumerate()).desc(Some("Create Config"))
        {
            let binary_file_stem = binary_file.file_stem().context("Failed to get file stem")?;
            let compressor_config_dir = compressor_config_dir.join(binary_file_stem);
            let filter_config_dir = filter_config_dir.join(binary_file_stem);
            if let Err(e) =
                create_config_command(binary_file.as_path(), Some(filter_config_dir.clone()))
            {
                eprintln!(
                    "Failed to create filter config. Add to skip list...: {:?}",
                    e
                );
                skip_files.insert(binary_file_stem.to_string_lossy().to_string());
                continue;
            }
            let csv_path = stem_to_csv_entry.get(binary_file_stem).context(format!(
                "Failed to get csv path, stem: {}",
                binary_file_stem.to_string_lossy()
            ))?;
            if let Err(e) =
                create_default_compressor_config(csv_path.as_path(), Some(compressor_config_dir))
            {
                eprintln!(
                    "Failed to create compressor config. Add to skip list...: {:?}",
                    e
                );
                skip_files.insert(binary_file_stem.to_string_lossy().to_string());
            }
        }
    }
    drop(stem_to_csv_entry);

    fn check_result_already_exist(
        save_dir: impl AsRef<Path>,
        binary_file_stem: &OsStr,
    ) -> anyhow::Result<Option<HashMap<String, AllOutputInner>>> {
        let saved_file_name = save_dir
            .as_ref()
            .join(binary_file_stem)
            .with_extension("json");

        if !saved_file_name.exists() {
            return Ok(None);
        }

        let saved_file = File::open(saved_file_name).context("Failed to open saved file")?;

        let all_output =
            serde_json::from_reader(saved_file).context("Failed to read saved file")?;

        Ok(Some(all_output))
    }

    fn create_saved_file_at(
        save_dir: impl AsRef<Path>,
        binary_file_stem: &OsStr,
        all_output: &AllOutputInner,
    ) -> anyhow::Result<()> {
        let saved_file_name = save_dir
            .as_ref()
            .join(binary_file_stem)
            .with_extension("json");

        let saved_file = File::create(saved_file_name).context("Failed to create saved file")?;
        serde_json::to_writer_pretty(saved_file, &all_output)
            .context("Failed to write saved file")?;

        Ok(())
    }
    let mut all_outputs = HashMap::new();
    for (_i, binary_file) in tqdm(entries.into_iter().enumerate()).desc(Some("Datasets")) {
        if skip_files.contains(binary_file.file_stem().unwrap().to_str().unwrap()) {
            println!("Skip processing file: {}", binary_file.display());
            continue;
        }

        let binary_file_stem = binary_file.file_stem().context("Failed to get file stem")?;

        if let Ok(Some(all_output)) =
            check_result_already_exist(save_dir.as_path(), binary_file_stem)
        {
            println!(
                "Result already exists. Skip processing file: {}",
                binary_file.display()
            );
            all_outputs.insert(binary_file_stem.to_string_lossy().to_string(), all_output);
            continue;
        }

        let filter_config_dir = filter_config_dir.join(binary_file_stem);
        let compressor_config_dir = compressor_config_dir.join(binary_file_stem);

        let mut outputs = HashMap::new();

        let entries = fs::read_dir(compressor_config_dir)
            .context("Failed to read compressor config directory")?;
        let entries = entries
            .filter_map(|e| {
                let e = e.ok()?;
                let path = e.path();
                if path.is_file()
                    && matches!(path.extension().and_then(|s| s.to_str()), Some("json"))
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for compressor_config_file in tqdm(entries.into_iter()).desc(Some("Compressor Configs")) {
            let compressor_config = ConfigPath {
                config: compressor_config_file.to_string_lossy().to_string(),
            };
            let compression_config: CompressionConfig = compressor_config.read()?;
            let compression_scheme = compression_config.compressor_config.compression_scheme();
            let mut compress_results = Vec::with_capacity(n);
            for _i in tqdm(0..n).desc(Some("Compress")) {
                let compress_result = compress_command(
                    binary_file.as_path(),
                    compression_config.clone(),
                    compressor_config.clone(),
                )?;
                compress_results.push(compress_result);
            }

            let compressed_file = binary_file.with_extension("ebi");

            let mut filter_outputs = HashMap::new();
            for filter_config_file in fs::read_dir(filter_config_dir.as_path())
                .context("Failed to read filter config directory")?
            {
                let filter_config_file = filter_config_file.context("Failed to read file")?;
                let filter_config_file = filter_config_file.path();
                if !filter_config_file.is_file()
                    || filter_config_file.extension().and_then(|s| s.to_str()) != Some("json")
                {
                    continue;
                }
                let filter_config_stem = filter_config_file
                    .file_stem()
                    .context("Failed to get file stem from filter config file")?;

                let filter_config = ConfigPath {
                    config: filter_config_file.to_string_lossy().to_string(),
                };

                let mut filter_results = Vec::with_capacity(n);
                let filter_config_content: FilterConfig = filter_config.read()?;
                for _i in tqdm(0..n).desc(Some(format!(
                    "Filter: {}",
                    filter_config_stem.to_str().unwrap_or("")
                ))) {
                    let filter_result = filter_command(
                        compressed_file.as_path(),
                        filter_config_content.clone(),
                        filter_config.clone(),
                    )?;
                    filter_results.push(filter_result);
                }

                let mut filter_materialize_results = Vec::with_capacity(n);
                let filter_materialize_config_content: FilterMaterializeConfig =
                    filter_config.read()?;

                for _i in tqdm(0..n).desc(Some(format!(
                    "Filter Materialize: {}",
                    filter_config_stem.to_str().unwrap_or("")
                ))) {
                    let filter_materialize_result = filter_materialize_command(
                        compressed_file.as_path(),
                        filter_materialize_config_content.clone(),
                        filter_config.clone(),
                    )?;
                    filter_materialize_results.push(filter_materialize_result);
                }

                filter_outputs.insert(
                    filter_config_stem.to_string_lossy().to_string(),
                    FilterFamilyOutput {
                        filter: filter_results,
                        filter_materialize: filter_materialize_results,
                    },
                );
            }

            let mut materialize_results = Vec::with_capacity(n);
            for _i in tqdm(0..n).desc(Some("Materialize")) {
                let materialize_result = materialize_command(
                    compressed_file.as_path(),
                    MaterializeConfig {
                        chunk_id: None,
                        bitmask: None,
                    },
                    ConfigPath::empty(),
                )?;
                materialize_results.push(materialize_result);
            }

            let all_output = AllOutputInner {
                compress: compress_results,
                filters: filter_outputs,
                materialize: materialize_results,
            };

            create_saved_file_at(save_dir.as_path(), binary_file_stem, &all_output)?;
            outputs.insert(format!("{:?}", compression_scheme), all_output);
        }
        all_outputs.insert(binary_file_stem.to_string_lossy().to_string(), outputs);
    }

    Ok(AllOutput(all_outputs))
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

fn create_default_compressor_config(
    filename: impl AsRef<Path>,
    output_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
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
        ("zstd", CompressorConfig::zstd().build().into()),
        ("gzip", CompressorConfig::gzip().build().into()),
        ("snappy", CompressorConfig::snappy().build().into()),
        ("ffi_alp", CompressorConfig::ffi_alp().build().into()),
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
    let configs = configs.into_iter().map(|(name, config)| {
        (
            name,
            CompressionConfig {
                chunk_option: ChunkOption::RecordCount(128 * 1024 * 1024 / 8),
                compressor_config: config,
            },
        )
    });

    let output_dir = Ok(output_dir).transpose().unwrap_or_else(
        || -> anyhow::Result<PathBuf, anyhow::Error> {
            Ok(std::env::current_dir()
                .context("Failed to get current directory")?
                .join("compressor_configs")
                .join(
                    filename
                        .as_ref()
                        .file_stem()
                        .context("Failed to get file stem")?,
                ))
        },
    )?;
    std::fs::create_dir_all(&output_dir).context(format!(
        "Failed to create output directory: {}",
        output_dir.display()
    ))?;
    for (name, config) in configs {
        let output_filename = output_dir.join(format!("{name}.json"));
        let mut f = File::create(&output_filename).context(format!(
            "Failed to create output file: {}",
            output_filename.as_path().display()
        ))?;
        serde_json::to_writer_pretty(&mut f, &config)
            .context(format!("Failed to write {} compressor config", name))?;
    }

    Ok(())
}

fn create_config_command(
    filename: impl AsRef<Path>,
    output_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let encoder_input = EncoderInput::from_file(filename.as_ref()).context(format!(
        "Failed to create encoder input from input file.: {}",
        filename.as_ref().display()
    ))?;
    let encoder_output = EncoderOutput::from_vec(Vec::new());

    let mut encoder = Encoder::new(
        encoder_input,
        encoder_output,
        ChunkOption::RecordCount(128 * 1024 * 1024 / 8),
        CompressorConfig::uncompressed().build(),
    );
    encoder.encode().context("Failed to encode")?;
    let mut writer = encoder.into_output().into_inner();
    writer.seek(io::SeekFrom::Start(0))?;
    let decoder_input = DecoderInput::from_reader(writer);
    let mut decoder = Decoder::new(decoder_input)?;

    let sum = decoder
        .sum(None, None)
        .context("Failed to get sum value from decoder")?;
    let avg = sum / decoder.footer().number_of_records() as f64;

    let base_dir = Path::new(filename.as_ref())
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let output_dir = Ok(output_dir)
        .transpose()
        .unwrap_or_else(|| -> anyhow::Result<PathBuf> {
            Ok(filename
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
                .join(base_dir))
        })?;

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

    let mut created_files = Vec::new();
    let mut err = None;
    for (pred, name) in &[
        (greater_predicate, "greater"),
        (eq_predicate, "eq"),
        (ne_predicate, "ne"),
    ] {
        let output_filename = output_dir.join(format!("{name}.json"));
        let mut f = File::create(&output_filename).context(format!(
            "Failed to create output file: {}",
            output_filename.display()
        ))?;
        created_files.push(output_filename.clone());
        serde_json::to_writer_pretty(
            &mut f,
            &FilterConfig {
                predicate: *pred,
                chunk_id: None,
                bitmask: None,
            },
        )
        .context(format!("Failed to write {} predicate config", name))?;

        // validness check
        drop(f);
        let mut f = File::open(&output_filename).context(format!(
            "Failed to open file: {}",
            output_filename.display()
        ))?;
        f.seek(io::SeekFrom::Start(0))?;
        if let Err(e) = serde_json::from_reader::<_, FilterConfig>(&mut f) {
            err = Some(e);
            break;
        }
        f.seek(io::SeekFrom::Start(0))?;
        if let Err(e) = serde_json::from_reader::<_, FilterMaterializeConfig>(&mut f) {
            err = Some(e);
            break;
        }
    }

    if let Some(e) = err {
        for file in created_files {
            if !file.exists() {
                continue;
            }
            std::fs::remove_file(file.as_path()).context(format!(
                "Failed to remove file: {}",
                file.as_path().display()
            ))?;
        }

        return Err(e).context("Failed to re-parse filter config");
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
    let result_string = format!(
        "number_of_records: {}",
        decoder.footer().number_of_records()
    );

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config,
        compression_config: config,
        command_specific: statistics,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
        result_string,
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

    let compression_config = CompressionConfig {
        chunk_option: *decoder.header().config().chunk_option(),
        compressor_config: *decoder.footer().compressor_config(),
    };

    let result_string = format!("{:?}", bitmask);

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
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

    let mut decoder =
        Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
            "Failed to create decoder from input file.: {}",
            filename.as_ref().display()
        ))?;
    let decoded_filename = filename.as_ref().with_extension("filter_materialized");
    let mut decoder_output = DecoderOutput::from_file(decoded_filename.as_path())
        .context(format!(
            "Failed to create decoder output from output file.: {}",
            decoded_filename.as_path().display()
        ))?
        .into_buffered();

    let start = std::time::Instant::now();
    decoder
        .filter_materialize(&mut decoder_output, predicate, bitmask.as_ref(), chunk_id)
        .context("Failed to perform filter materialize")?;
    let elapsed_time_nanos = start.elapsed().as_nanos();

    let compression_config = CompressionConfig {
        chunk_option: *decoder.header().config().chunk_option(),
        compressor_config: *decoder.footer().compressor_config(),
    };
    let metadata = decoder_output.into_writer().into_inner()?.metadata()?;
    let result_string = format!("{:?}", metadata);

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
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
    let decoded_filename = filename.as_ref().with_extension("materialized");
    let mut decoder_output = DecoderOutput::from_file(decoded_filename.as_path())
        .context(format!(
            "Failed to create decoder output from output file.: {}",
            decoded_filename.as_path().display()
        ))?
        .into_buffered();

    let start = std::time::Instant::now();
    decoder
        .materialize(&mut decoder_output, bitmask.as_ref(), chunk_id)
        .context("Failed to perform filter materialize")?;
    let elapsed_time_nanos = start.elapsed().as_nanos();

    let compression_config = CompressionConfig {
        chunk_option: *decoder.header().config().chunk_option(),
        compressor_config: *decoder.footer().compressor_config(),
    };

    let metadata = decoder_output.into_writer().into_inner()?.metadata()?;
    let result_string = format!("{:?}", metadata);

    let output = OutputWrapper {
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
        datetime: now,
    };

    Ok(output)
}
