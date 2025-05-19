use cfg_if::cfg_if;
use either::Either;
#[cfg(feature = "cuda")]
use experimenter::matmul_cuda::matrix_cuda_command;
use experimenter::{
    get_appropriate_precision, get_compress_statistics, round_by_scale, save_json_safely,
    tpch::{self},
    AllOutput, AllOutputInner, CompressStatistics, CompressionConfig, FilterConfig,
    FilterFamilyOutput, FilterMaterializeConfig, MaterializeConfig, MatrixResult, MaxConfig,
    Output, OutputWrapper, SimpleCompressionPerformanceForOneDataset, SumConfig, UCR2018Config,
    UCR2018DecompressionResult, UCR2018ForAllCompressionMethodsResult, UCR2018Result,
    UCR2018ResultForOneDataset, DEFAULT_CHUNK_OPTION,
};
use glob::glob;
use rand::Rng as _;
use rand_chacha::{rand_core::SeedableRng as _, ChaCha20Rng};
use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Cursor, Read, Seek, Write as _},
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
    decoder::query::{Predicate, Range, RangeValue},
    encoder::ChunkOption,
    format::CompressionScheme,
    time::SerializableSegmentedExecutionTimes,
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
    #[arg(long, short)]
    in_memory: Option<bool>,
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
            in_memory: None,
        }
    }
}

#[derive(Debug, Clone, Args)]
struct AllPatchArgs {
    #[arg(long)]
    create_config: bool,
    #[arg(long)]
    exact_precision: bool,
    #[arg(long)]
    in_memory: bool,
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

    #[arg(long("pd"))]
    patch_dataset: Vec<String>,
    #[arg(long("pc"))]
    patch_compressor: Vec<String>,
}

impl AllPatchArgs {
    fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.compressor_config_dir.is_dir(),
            "Config directory does not exist or not directory: {}",
            self.compressor_config_dir.display()
        );
        anyhow::ensure!(
            self.filter_config_dir.is_dir(),
            "Config directory does not exist or not directory: {}",
            self.filter_config_dir.display()
        );
        anyhow::ensure!(
            self.binary_dir.is_dir(),
            "Binary directory does not exist or not directory: {}",
            self.binary_dir.display()
        );
        anyhow::ensure!(
            self.save_dir.is_dir(),
            "Save directory does not exist or not directory: {}",
            self.save_dir.display()
        );

        const VALID_COMPRESSOR: &[&str] = &[
            "uncompressed",
            "chimp128",
            "chimp",
            "elf_on_chimp",
            "elf",
            "gorilla",
            "rle",
            "zstd",
            "gzip",
            "snappy",
            "ffi_alp",
            "delta_sprintz",
            "buff",
        ];
        for patch_compressor in &self.patch_compressor {
            anyhow::ensure!(
                VALID_COMPRESSOR.contains(&patch_compressor.as_str()),
                "Invalid compressor: {}, valid compressors: {:?}",
                patch_compressor,
                VALID_COMPRESSOR
            );
        }

        Ok(())
    }

    fn from_all_args(
        args: AllArgs,
        patch_dataset: Vec<String>,
        patch_compressor: Vec<String>,
    ) -> Self {
        Self {
            create_config: args.create_config,
            exact_precision: args.exact_precision,
            in_memory: args.in_memory,
            compressor_config_dir: args.compressor_config_dir,
            filter_config_dir: args.filter_config_dir,
            binary_dir: args.binary_dir,
            save_dir: args.save_dir,
            n: args.n,
            patch_dataset,
            patch_compressor,
        }
    }
}

#[derive(Debug, Clone, Args)]
struct AllArgs {
    #[arg(long)]
    create_config: bool,
    #[arg(long)]
    exact_precision: bool,
    #[arg(long)]
    in_memory: bool,
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
impl AllArgs {
    fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.compressor_config_dir.is_dir(),
            "Config directory does not exist or not directory: {}",
            self.compressor_config_dir.display()
        );
        anyhow::ensure!(
            self.filter_config_dir.is_dir(),
            "Config directory does not exist or not directory: {}",
            self.filter_config_dir.display()
        );
        anyhow::ensure!(
            self.binary_dir.is_dir(),
            "Binary directory does not exist or not directory: {}",
            self.binary_dir.display()
        );
        anyhow::ensure!(
            self.save_dir.is_dir(),
            "Save directory does not exist or not directory: {}",
            self.save_dir.display()
        );

        Ok(())
    }
}

fn config_entries(config_dir: impl AsRef<Path>) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    Ok(entry_file_iter(config_dir)?
        .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("json"))))
}

fn csv_entries(csv_dir: impl AsRef<Path>) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    Ok(entry_file_iter(csv_dir)?.filter(|p| {
        matches!(
            p.extension().and_then(|s| s.to_str()),
            Some("csv" | "txt") | None
        )
    }))
}

fn binary_entries(binary_dir: impl AsRef<Path>) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    Ok(entry_file_iter(binary_dir)?
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("bin")))
}

fn entry_file_iter(dir: impl AsRef<Path>) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    let entries = fs::read_dir(dir.as_ref()).context(format!(
        "Failed to read directory: {}",
        dir.as_ref().display()
    ))?;
    Ok(entries.filter_map(|e| {
        let e = e.ok()?;
        let path = e.path();
        if path.is_file() {
            Some(path)
        } else {
            None
        }
    }))
}

fn entry_dir_iter(dir: impl AsRef<Path>) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    let entries = fs::read_dir(dir.as_ref()).context(format!(
        "Failed to read directory: {}",
        dir.as_ref().display()
    ))?;
    Ok(entries.filter_map(|e| {
        let e = e.ok()?;
        let path = e.path();
        if path.is_dir() {
            Some(path)
        } else {
            None
        }
    }))
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
        #[arg(long)]
        exact_precision: bool,
    },
    AllPatch(AllPatchArgs),
    Compress(ConfigPath),
    Filter(ConfigPath),
    FilterMaterialize(ConfigPath),
    Materialize(ConfigPath),
    Max(ConfigPath),
    Sum(ConfigPath),
    All(AllArgs),
    UCR2018 {
        #[arg(long, short('o'))]
        output_dir: PathBuf,
    },
    Matrix {
        #[arg(long, short('o'))]
        output_dir: PathBuf,
    },
    MatrixCuda {
        #[arg(long, short('o'))]
        output_dir: PathBuf,
    },
    Embedding {
        #[arg(long, short('o'))]
        output_dir: PathBuf,
    },
    Tpch {
        #[arg(long, short('i'))]
        input_dir: PathBuf,
        #[arg(long, short('o'))]
        output_dir: PathBuf,
    },
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
            Commands::AllPatch { .. } => "all_patch",
            Commands::Max { .. } => "max",
            Commands::Sum { .. } => "sum",
            Commands::UCR2018 { .. } => "ucr2018",
            Commands::Matrix { .. } => "matrix",
            Commands::MatrixCuda { .. } => "matrix_cuda",
            Commands::Embedding { .. } => "embedding",
            Commands::Tpch { .. } => "tpch",
        }
    }

    pub fn is_matrix_type(&self) -> Option<&PathBuf> {
        if let Commands::Matrix { output_dir } = self {
            return Some(output_dir);
        }

        #[cfg(feature = "cuda")]
        if let Commands::MatrixCuda { output_dir } = self {
            return Some(output_dir);
        }

        None
    }

    pub fn is_matrix_cuda(&self) -> bool {
        matches!(self, Commands::MatrixCuda { .. })
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

    if let Some(output_dir) = cli.command.is_matrix_type() {
        let is_matrix_cuda = cli.command.is_matrix_cuda();
        let mut save_dir = output_dir.join("result");
        if is_matrix_cuda {
            save_dir = save_dir.join("matrix_cuda");
        } else {
            save_dir = save_dir.join("matrix");
        }
        // Create the directory if it doesn't exist
        std::fs::create_dir_all(&save_dir)?;
        println!("Save dir: {}", save_dir.display());

        // Determine the next available directory name
        let mut dir_number = 0;
        let output_dir = loop {
            let output_dirname = save_dir.join(format!("{:03}", dir_number));
            if !output_dirname.exists() {
                break output_dirname;
            }
            dir_number += 1;
        };

        // Create the directory
        std::fs::create_dir_all(&output_dir)?;

        let precisions = [1, 3, 5, 8];
        for precision in precisions {
            assert!(
                10u32.checked_pow(precision).is_some(),
                "10^{} is too large for u32",
                precision
            );
        }
        let mut matrix_sizes = vec![128, 512, 1024];
        if is_matrix_cuda {
            matrix_sizes.extend_from_slice(&[2048, 4096]);
        }
        for &matrix_size in matrix_sizes.iter() {
            match DEFAULT_CHUNK_OPTION {
                ChunkOption::RecordCount(c) => assert!(
                    c % (matrix_size * matrix_size) == 0,
                    "{} % ({} * {}) != 0",
                    c,
                    matrix_size,
                    matrix_size
                ),
                ChunkOption::ByteSizeBestEffort(_) | ChunkOption::Full => unreachable!(),
            }
        }
        for precision in tqdm(precisions).desc(Some("Precision")) {
            for &matrix_size in
                tqdm(matrix_sizes.iter()).desc(Some(format!("Matrix Size(p:{precision})")))
            {
                let filename = if is_matrix_cuda {
                    format!("matrix_cuda_{}_{}.json", matrix_size, precision)
                } else {
                    format!("matrix_{}_{}.json", matrix_size, precision)
                };
                let do_with_only_compression_methods_with_controlled_precision_support =
                    precision != 8;
                let output = if is_matrix_cuda {
                    cfg_if! {
                        if #[cfg(feature = "cuda")] {
                            matrix_cuda_command(
                                precision,
                                matrix_size,
                                do_with_only_compression_methods_with_controlled_precision_support,
                            )?
                        } else {
                            panic!("Matrix CUDA command is not supported in this build. Please enable the 'cuda' feature.")
                        }
                    }
                } else {
                    matrix_command(
                        precision,
                        matrix_size,
                        do_with_only_compression_methods_with_controlled_precision_support,
                    )?
                };

                serde_json::to_writer(
                    BufWriter::new(File::create(output_dir.join(filename))?),
                    &output,
                )?;
            }
        }

        return Ok(());
    }

    if let Commands::Tpch {
        input_dir,
        output_dir,
    } = &cli.command
    {
        let save_dir = output_dir.join("result").join("tpch");
        anyhow::ensure!(
            input_dir.is_dir(),
            "Input directory does not exist or not directory: {}",
            input_dir.display()
        );

        std::fs::create_dir_all(&save_dir)?;

        // Determine the next available directory name
        let mut dir_number = 0;
        let unique_output_dir = loop {
            let output_dirname = save_dir.join(format!("{:03}", dir_number));
            if !output_dirname.exists() {
                break output_dirname;
            }
            dir_number += 1;
        };
        std::fs::create_dir_all(&unique_output_dir)?;
        println!("Save dir: {}", unique_output_dir.display());

        const N_BATCH: usize = 5;
        let mut results_01 = Vec::with_capacity(N_BATCH);
        let mut results_06 = Vec::with_capacity(N_BATCH);
        for _ in 0..N_BATCH {
            let (result01, result06) = tpch::tpch_command(input_dir)?;
            results_01.push(result01);
            results_06.push(result06);
        }

        save_json_safely(&results_01, unique_output_dir.join("tpch_01.json"))?;
        save_json_safely(&results_06, unique_output_dir.join("tpch_06.json"))?;

        return Ok(());
    }

    let Some(input) = &cli.input else {
        anyhow::bail!("Input file is required");
    };

    if let Commands::AllPatch(args) = &cli.command {
        let args = args.clone();
        let patched = PathBuf::from(input);
        anyhow::ensure!(
            patched.exists(),
            "Patched dataset does not exist: {}",
            patched.display()
        );
        anyhow::ensure!(
            patched.is_file(),
            "Patched dataset is not a file: {}",
            patched.display()
        );
        let Output::All(patched) =
            serde_json::from_reader::<_, Output>(&mut BufReader::new(File::open(&patched)?))
                .context("Failed to read results being patched")?
        else {
            anyhow::bail!("Patched dataset is not an AllOutput");
        };
        let save_dir = args
            .binary_dir
            .parent()
            .context("Failed to get parent directory")?
            .join("result")
            .join("all_patch");
        let all_outputs = all_patch_command(args, patched)?;
        save_output_json(all_outputs.into(), save_dir)?;

        return Ok(());
    }

    if let Commands::UCR2018 { output_dir } = &cli.command {
        let input_dir = Path::new(input);
        let output_dir = output_dir.join("result").join("1nn_ucr2018");
        anyhow::ensure!(
            input_dir.is_dir(),
            "Input directory does not exist or not directory: {}",
            input_dir.display()
        );

        std::fs::create_dir_all(&output_dir)?;

        let precisions = [1, 3, 5, 8];
        for precision in precisions {
            assert!(
                10u32.checked_pow(precision).is_some(),
                "10^{} is too large for u32",
                precision
            );
        }

        let save_dir = input_dir.parent().unwrap().join("result").join("ucr2018");
        // Determine the next available directory name
        let mut dir_number = 0;
        let unique_output_dir = loop {
            let output_dirname = save_dir.join(format!("{:03}", dir_number));
            if !output_dirname.exists() {
                break output_dirname;
            }
            dir_number += 1;
        };

        // without controlled precision support
        let output = ucr2018_command(input_dir, 5, 1, -1, false)?;
        save_ucr2018_json(output, &unique_output_dir, None)?;

        for precision in tqdm(precisions).desc(Some("Precision")) {
            let output = ucr2018_command(input_dir, 5, 1, precision as i32, true)?;
            save_ucr2018_json(output, &unique_output_dir, Some(precision))?;
        }

        return Ok(());
    }

    if let Commands::Embedding { output_dir } = &cli.command {
        let input_dir = Path::new(input);
        let output_dir = output_dir.join("result").join("embedding");
        anyhow::ensure!(
            input_dir.is_dir(),
            "Input directory does not exist or not directory: {}",
            input_dir.display()
        );

        std::fs::create_dir_all(&output_dir)?;

        let save_dir = input_dir.parent().unwrap().join("result").join("embedding");
        // Determine the next available directory name
        let mut dir_number = 0;
        let unique_output_dir = loop {
            let output_dirname = save_dir.join(format!("{:03}", dir_number));
            if !output_dirname.exists() {
                break output_dirname;
            }
            dir_number += 1;
        };

        // without controlled precision support
        let output = embedding_command(input_dir)?;
        save_embedding_json(output, &unique_output_dir)?;

        return Ok(());
    }

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
        if path.is_dir() {
            eprintln!("Skipping directory: {}", path.display());
            continue;
        }
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
        Commands::Compress(args) => compress_command(
            args.in_memory.unwrap_or(false),
            filename,
            args.read()?,
            args,
        )?
        .into(),
        Commands::Filter(args) => filter_command(
            args.in_memory.unwrap_or(false),
            filename,
            args.read()?,
            args,
        )?
        .into(),
        Commands::FilterMaterialize(args) => filter_materialize_command(
            args.in_memory.unwrap_or(false),
            filename,
            args.read()?,
            args,
        )?
        .into(),
        Commands::Materialize(args) => materialize_command(
            args.in_memory.unwrap_or(false),
            filename,
            args.read()?,
            args,
        )?
        .into(),
        Commands::CreateFilterConfig { output_dir } => {
            return create_config_command(filename, output_dir);
        }
        Commands::CreateDefaultCompressorConfig {
            output_dir,
            exact_precision,
        } => {
            return create_default_compressor_config(filename, output_dir, exact_precision);
        }
        Commands::Max(args) => max_command(
            args.in_memory.unwrap_or(false),
            filename,
            args.read()?,
            args,
        )?
        .into(),
        Commands::Sum(args) => sum_command(
            args.in_memory.unwrap_or(false),
            filename,
            args.read()?,
            args,
        )?
        .into(),
        Commands::AllPatch { .. } => unreachable!(),
        Commands::All { .. } => unreachable!(),
        Commands::UCR2018 { .. } => unreachable!(),
        Commands::Matrix { .. } => unreachable!(),
        Commands::MatrixCuda { .. } => unreachable!(),
        Commands::Embedding { .. } => unreachable!(),
        Commands::Tpch { .. } => unreachable!(),
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

fn save_ucr2018_json(
    output: UCR2018ForAllCompressionMethodsResult,
    save_dir: impl AsRef<Path>,
    precision: Option<u32>,
) -> anyhow::Result<()> {
    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&save_dir)?;

    // Write the output to JSON
    for (dataset, result) in output.results.into_iter() {
        let filename = match precision {
            Some(precision) => format!("{}_{}.json", dataset, precision),
            None => format!("{}.json", dataset),
        };
        let output_file = save_dir.as_ref().join(filename);
        write_output_json(result, output_file)?;
    }

    Ok(())
}

fn save_embedding_json(
    output: HashMap<String, HashMap<String, SimpleCompressionPerformanceForOneDataset>>,
    save_dir: impl AsRef<Path>,
) -> anyhow::Result<()> {
    // Create the directory if it doesn't exist
    std::fs::create_dir_all(&save_dir)?;

    let output_file = save_dir.as_ref().join("embedding_result.json");
    // Write the output to JSON
    write_output_json(Output::Embedding(output), output_file)?;

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
    serde_json::to_writer_pretty(saved_file, &all_output).context("Failed to write saved file")?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_experiment_for_compressor(
    in_memory: bool,
    binary_file: impl AsRef<Path>,
    compression_config: CompressionConfig,
    compressor_config: ConfigPath,
    filter_config_dir: impl AsRef<Path>,
    save_dir: impl AsRef<Path>,
    all_output_inner: &mut AllOutputInner,
    n: usize,
    max_value: f64,
) -> anyhow::Result<()> {
    let compression_scheme = compression_config.compressor_config.compression_scheme();
    if compression_scheme == CompressionScheme::BUFF
        || compression_scheme == CompressionScheme::DeltaSprintz
    {
        let scale = compression_config.compressor_config.scale().unwrap();
        let precision = if scale == 0 {
            0
        } else {
            (scale as f64).log10() as usize
        };
        // Check if the quantization will be done correctly
        let quantized = max_value * scale as f64;
        if !quantized.is_finite() || quantized.abs() > i64::MAX as f64 {
            eprintln!(
                "Quantization will not be done correctly. Skip: {:?}",
                compressor_config
            );
            return Ok(());
        }
        if compression_scheme == CompressionScheme::BUFF && precision > 12 {
            eprintln!(
                "Precision({}) is too high for BUFF. Skip: {:?}",
                precision, compressor_config
            );
            return Ok(());
        }
    }
    let binary_file_stem = binary_file
        .as_ref()
        .file_stem()
        .context("Failed to get file stem")?;

    // Compress
    for _i in tqdm(0..n).desc(Some("Compress")) {
        let compress_result = compress_command(
            in_memory,
            binary_file.as_ref(),
            compression_config.clone(),
            compressor_config.clone(),
        )?;
        all_output_inner.compress.push(compress_result);
    }

    let compressed_file = binary_file.as_ref().with_extension("ebi");

    // filter
    let filter_config_entries = config_entries(filter_config_dir.as_ref())
        .context("Failed to read filter config directory")?;
    for filter_config_file in filter_config_entries {
        let filter_config_stem = filter_config_file
            .file_stem()
            .context("Failed to get file stem from filter config file")?;
        let filter_config_stem_str = filter_config_stem.to_string_lossy().to_string();
        let filters_output = all_output_inner
            .filters
            .entry(filter_config_stem_str.clone())
            .or_insert(FilterFamilyOutput {
                filter: Vec::with_capacity(n),
                filter_materialize: Vec::with_capacity(n),
            });

        let filter_config = ConfigPath {
            config: filter_config_file.to_string_lossy().to_string(),
            in_memory: None,
        };

        let filter_config_content: FilterConfig = filter_config.read()?;
        for _i in tqdm(0..n).desc(Some(format!(
            "Filter: {}",
            filter_config_stem.to_str().unwrap_or("")
        ))) {
            let filter_result = filter_command(
                in_memory,
                compressed_file.as_path(),
                filter_config_content.clone(),
                filter_config.clone(),
            )?;
            filters_output.filter.push(filter_result);
        }

        let filter_materialize_config_content: FilterMaterializeConfig =
            filter_config_content.into();

        for _i in tqdm(0..n).desc(Some(format!(
            "Filter Materialize: {}",
            filter_config_stem.to_str().unwrap_or("")
        ))) {
            let filter_materialize_result = filter_materialize_command(
                in_memory,
                compressed_file.as_path(),
                filter_materialize_config_content.clone(),
                filter_config.clone(),
            )?;
            filters_output
                .filter_materialize
                .push(filter_materialize_result);
        }
    }

    for _i in tqdm(0..n).desc(Some("Materialize")) {
        let materialize_result = materialize_command(
            in_memory,
            compressed_file.as_path(),
            MaterializeConfig {
                chunk_id: None,
                bitmask: None,
            },
            ConfigPath::empty(),
        )?;
        all_output_inner.materialize.push(materialize_result);
    }

    for _ in tqdm(0..n).desc(Some("Max")) {
        let max_result = max_command(
            in_memory,
            compressed_file.as_path(),
            MaxConfig {
                chunk_id: None,
                bitmask: None,
            },
            ConfigPath::empty(),
        )?;
        all_output_inner.max.push(max_result);
    }

    for _ in tqdm(0..n).desc(Some("Sum")) {
        let sum_result = sum_command(
            in_memory,
            compressed_file.as_path(),
            SumConfig {
                chunk_id: None,
                bitmask: None,
            },
            ConfigPath::empty(),
        )?;
        all_output_inner.sum.push(sum_result);
    }

    create_saved_file_at(save_dir.as_ref(), binary_file_stem, all_output_inner)?;
    Ok(())
}

fn process_experiment_for_dataset(
    in_memory: bool,
    binary_file: impl AsRef<Path>,
    compressor_config_entries: impl IntoIterator<Item = PathBuf>,
    filter_config_dir: impl AsRef<Path>,
    save_dir: impl AsRef<Path>,
    n: usize,
    outputs: &mut HashMap<String, AllOutputInner>,
) -> anyhow::Result<()> {
    for compressor_config_file in tqdm(compressor_config_entries).desc(Some("Compressor Configs")) {
        let compressor_config = ConfigPath {
            config: compressor_config_file.to_string_lossy().to_string(),
            in_memory: None,
        };
        let compression_config: CompressionConfig = compressor_config.read()?;
        let compression_scheme = compression_config.compressor_config.compression_scheme();
        let compression_scheme_key = format!("{:?}", compression_scheme);

        // Clear all the result for this compression
        let all_output_inner = outputs
            .entry(compression_scheme_key.clone())
            .and_modify(AllOutputInner::clear)
            .or_insert(AllOutputInner {
                compress: Vec::with_capacity(n),
                filters: HashMap::with_capacity(3),
                materialize: Vec::with_capacity(n),
                max: Vec::with_capacity(n),
                sum: Vec::with_capacity(n),
            });

        // Find the max value for the dataset
        let max_value = {
            let mut max_value: f64 = 0.0;
            let mut reader = BufReader::new(File::open(&binary_file)?);
            let mut buffer = [0; 8];
            while let Ok(n) = reader.read(&mut buffer) {
                if n == 0 {
                    break;
                }
                let value = u64::from_le_bytes(buffer);
                max_value = max_value.max(f64::from_bits(value));
            }
            max_value
        };

        process_experiment_for_compressor(
            in_memory,
            &binary_file,
            compression_config,
            compressor_config,
            &filter_config_dir,
            &save_dir,
            all_output_inner,
            n,
            max_value,
        )?;
    }

    Ok(())
}

fn all_patch_command(args: AllPatchArgs, mut patched: AllOutput) -> anyhow::Result<AllOutput> {
    args.verify()?;
    let AllPatchArgs {
        create_config,
        exact_precision,
        in_memory,
        compressor_config_dir,
        filter_config_dir,
        binary_dir,
        save_dir,
        n,
        patch_dataset,
        patch_compressor,
    } = args;
    assert!(
        !exact_precision || create_config,
        "Exact precision requires create config"
    );

    let n = n.unwrap_or(10);

    let binary_entries_all = binary_entries(&binary_dir)?.collect::<Vec<_>>();

    let csv_dir = binary_dir.parent().unwrap();
    let stem_to_csv_entry = csv_entries(csv_dir)?
        .map(|path| {
            let stem = path.file_stem().unwrap().to_owned();
            (stem, path)
        })
        .collect::<HashMap<_, _>>();

    let mut skip_files = HashSet::new();
    // Create Config!
    if create_config {
        for (_i, binary_file) in
            tqdm(binary_entries_all.clone().into_iter().enumerate()).desc(Some("Create Config"))
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
            if let Err(e) = create_default_compressor_config(
                csv_path.as_path(),
                Some(compressor_config_dir),
                exact_precision,
            ) {
                eprintln!(
                    "Failed to create compressor config. Add to skip list...: {:?}",
                    e
                );
                skip_files.insert(binary_file_stem.to_string_lossy().to_string());
            }
        }
    }
    drop(stem_to_csv_entry);

    let binary_entries_patching = binary_entries_all
        .clone()
        .into_iter()
        .filter(|path| {
            let stem = path.file_stem().unwrap().to_string_lossy();
            patch_dataset.contains(&stem.to_string())
        })
        .collect::<HashSet<_>>();

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

    // first patch compressor for exsiting dataset in the result
    for binary_file in tqdm(binary_entries_all).desc(Some("Datasets")) {
        let binary_file_stem = binary_file.file_stem().context("Failed to get file stem")?;
        let binary_file_stem_str = binary_file_stem.to_string_lossy().to_string();

        if skip_files.contains(&binary_file_stem_str) {
            println!("Skip processing file: {}", binary_file.display());
            continue;
        }
        if binary_entries_patching.contains(&PathBuf::from(binary_file_stem)) {
            // Skip temporarily processing dataset to be patched
            continue;
        }

        let filter_config_dir = filter_config_dir.join(binary_file_stem);
        let compressor_config_dir = compressor_config_dir.join(binary_file_stem);

        let compressor_config_entries_patching = config_entries(compressor_config_dir.clone())
            .context("Failed to read compressor config directory")?
            .filter(|path| {
                let stem = path.file_stem().unwrap().to_string_lossy();
                patch_compressor.contains(&stem.to_string())
            })
            .collect::<Vec<_>>();
        println!(
            "Compressor Configs to be patched: {:?}",
            &compressor_config_entries_patching
        );

        if let Ok(Some(all_output)) =
            check_result_already_exist(save_dir.as_path(), binary_file_stem)
        {
            println!(
                "Result already exists. Skip processing file: {}",
                binary_file.display()
            );
            patched.0.insert(binary_file_stem_str, all_output);
            continue;
        }

        let outputs = patched.0.entry(binary_file_stem_str.clone()).or_default();

        process_experiment_for_dataset(
            in_memory,
            binary_file,
            compressor_config_entries_patching.clone(),
            filter_config_dir,
            &save_dir,
            n,
            outputs,
        )?;
    }

    println!("Dataset to be patched: {:?}", &binary_entries_patching);
    // second patch dataset for all the compressors
    for binary_file in tqdm(binary_entries_patching).desc(Some("Datasets")) {
        let binary_file_stem = binary_file.file_stem().context("Failed to get file stem")?;
        let binary_file_stem_str = binary_file_stem.to_string_lossy().to_string();

        if skip_files.contains(&binary_file_stem_str) {
            println!("Skip processing file: {}", binary_file.display());
            continue;
        }

        if let Ok(Some(all_output)) =
            check_result_already_exist(save_dir.as_path(), binary_file_stem)
        {
            println!(
                "Result already exists. Skip processing file: {}",
                binary_file.display()
            );
            patched.0.insert(binary_file_stem_str, all_output);
            continue;
        }

        let filter_config_dir = filter_config_dir.join(binary_file_stem);
        let compressor_config_dir = compressor_config_dir.join(binary_file_stem);

        let outputs = patched.0.entry(binary_file_stem_str.clone()).or_default();

        let compressor_config_entries_all = config_entries(compressor_config_dir.clone())
            .context("Failed to read compressor config directory")?
            .collect::<Vec<_>>();
        process_experiment_for_dataset(
            in_memory,
            binary_file,
            compressor_config_entries_all.clone().into_iter(),
            filter_config_dir,
            &save_dir,
            n,
            outputs,
        )?;
    }

    Ok(patched)
}

fn all_command(args: AllArgs) -> anyhow::Result<AllOutput> {
    args.verify()?;

    let all_outputs = AllOutput(HashMap::new());
    let patch_dataset = binary_entries(&args.binary_dir)?
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let patch_compressor = config_entries(&args.compressor_config_dir)?
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let patch_args = AllPatchArgs::from_all_args(args, patch_dataset, patch_compressor);
    all_patch_command(patch_args, all_outputs)
}

fn create_default_compressor_config(
    filename: impl AsRef<Path>,
    output_dir: Option<PathBuf>,
    exact_precision: bool,
) -> anyhow::Result<()> {
    let reader = BufReader::new(File::open(filename.as_ref()).context(format!(
        "Failed to open file.: {}",
        filename.as_ref().display()
    ))?);
    println!("get scale for {}", filename.as_ref().display());
    let prec = get_appropriate_precision(reader, ',', exact_precision)?;
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
    if let Some(scale) = 10u64.checked_pow(prec) {
        configs.push((
            "delta_sprintz",
            CompressorConfig::delta_sprintz()
                .scale(scale)
                .build()
                .into(),
        ));
        configs.push(("buff", CompressorConfig::buff().scale(scale).build().into()));
    } else {
        eprintln!(
            "Precision is too high for delta_sprintz and buff. Skip creating config for them of {}",
            filename.as_ref().display()
        );
    }
    let configs = configs.into_iter().map(|(name, config)| {
        (
            name,
            CompressionConfig {
                chunk_option: DEFAULT_CHUNK_OPTION,
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
    println!("create filter config for {}", filename.as_ref().display());
    let mut buf = Vec::new();
    File::open(filename.as_ref())
        .context(format!(
            "Failed to create encoder input from input file.: {}",
            filename.as_ref().display()
        ))?
        .read_to_end(&mut buf)
        .context(format!(
            "Failed to read input file.: {}",
            filename.as_ref().display()
        ))?;
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

    let mut floats = buf
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();

    let number_of_records = floats.len() as u64;

    anyhow::ensure!(
        !floats.iter().any(|&f| f.is_nan()),
        "Found NaN in the input file"
    );
    floats.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("Failed to compare {} and {}", a, b))
    });
    let ten_percentile = floats[(number_of_records * 10 / 100) as usize];
    let half_percentile = floats[(number_of_records / 2) as usize];
    let ninety_percentile = floats[(number_of_records * 90 / 100) as usize];

    let greater_predicate_ten_percentile = Predicate::Range(Range::new(
        RangeValue::Exclusive(ten_percentile),
        RangeValue::None,
    ));
    let greater_predicate_half_percentile = Predicate::Range(Range::new(
        RangeValue::Exclusive(half_percentile),
        RangeValue::None,
    ));
    let greater_predicate_ninety_percentile = Predicate::Range(Range::new(
        RangeValue::Exclusive(ninety_percentile),
        RangeValue::None,
    ));

    let eq_predicate = Predicate::Eq(half_percentile);

    let ne_predicate = Predicate::Ne(half_percentile);

    std::fs::create_dir_all(&output_dir).context(format!(
        "Failed to create output directory: {}",
        output_dir.display()
    ))?;

    let mut created_files = Vec::new();
    let mut err = None;
    for (pred, name) in &[
        (greater_predicate_ten_percentile, "greater_10th_percentile"),
        (greater_predicate_half_percentile, "greater_50th_percentile"),
        (
            greater_predicate_ninety_percentile,
            "greater_90th_percentile",
        ),
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
    in_memory: bool,
    filename: impl AsRef<Path>,
    config: CompressionConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<CompressStatistics>> {
    let CompressionConfig {
        compressor_config,
        chunk_option,
    } = config.clone();
    let now = chrono::Utc::now();

    let (statistics, elapsed_time_nanos, execution_times, result_string) = if in_memory {
        let mut content = Vec::new();
        File::open(filename.as_ref())
            .context(format!(
                "Failed to open file.: {}",
                filename.as_ref().display()
            ))?
            .read_to_end(&mut content)
            .context(format!(
                "Failed to read file.: {}",
                filename.as_ref().display()
            ))?;
        let encoder_input = EncoderInput::from_reader(&content[..]);
        let encoder_output = EncoderOutput::from_vec(Vec::new());
        let mut encoder = Encoder::new(
            encoder_input,
            encoder_output,
            chunk_option,
            compressor_config,
        );

        let start = std::time::Instant::now();
        encoder.encode().context("Failed to encode")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let mut encoder_output = encoder.into_output().into_inner();
        encoder_output.seek(io::SeekFrom::Start(0))?;

        File::create(filename.as_ref().with_extension("ebi"))
            .context("Failed to create output file")?
            .write_all(encoder_output.get_ref())
            .context("Failed to write encoded memory to file")?;

        let decoder = Decoder::new(DecoderInput::from_reader(encoder_output))
            .context("Failed to create decoder from encoded memory")?;
        let result_string = format!(
            "number_of_records: {}",
            decoder.footer().number_of_records()
        );
        let segmented_execution_times = *decoder.footer().segmented_execution_times();

        (
            get_compress_statistics(&decoder),
            elapsed_time_nanos,
            segmented_execution_times,
            result_string,
        )
    } else {
        let encoded_filename = filename.as_ref().with_extension("ebi");
        let encoder_input = EncoderInput::from_file(filename.as_ref()).context(format!(
            "Failed to create encoder input from input file.: {}",
            filename.as_ref().display()
        ))?;
        let encoder_output =
            EncoderOutput::from_file(encoded_filename.as_path()).context(format!(
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
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        drop(encoder);

        let decoder = Decoder::new(DecoderInput::from_file(encoded_filename.as_path())?)
            .context("Failed to create decoder from encoded file")?;
        let result_string = format!(
            "number_of_records: {}",
            decoder.footer().number_of_records()
        );
        let segmented_execution_times = *decoder.footer().segmented_execution_times();

        (
            get_compress_statistics(&decoder),
            elapsed_time_nanos,
            segmented_execution_times,
            result_string,
        )
    };

    let output = OutputWrapper {
        in_memory,
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config,
        compression_config: config,
        command_specific: statistics?,
        elapsed_time_nanos,
        execution_times: execution_times.into(),
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        datetime: now,
        result_string,
    };

    Ok(output)
}

fn filter_command(
    in_memory: bool,
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

    let (bitmask, elapsed_time_nanos, execution_times, compression_config) = if in_memory {
        let mut content = Vec::new();
        File::open(filename.as_ref())
            .context(format!(
                "Failed to open file.: {}",
                filename.as_ref().display()
            ))?
            .read_to_end(&mut content)
            .context(format!(
                "Failed to read file.: {}",
                filename.as_ref().display()
            ))?;
        let mut decoder = Decoder::new(DecoderInput::from_reader(Cursor::new(&content[..])))
            .context("Failed to create decoder from input memory")?;

        let start = std::time::Instant::now();
        let bitmask = decoder
            .filter(predicate, bitmask.as_ref(), chunk_id)
            .context("Failed to perform filter")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        (
            bitmask,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            compression_config,
        )
    } else {
        let mut decoder =
            Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
                "Failed to create decoder from input file.: {}",
                filename.as_ref().display()
            ))?;

        let start = std::time::Instant::now();
        let bitmask = decoder
            .filter(predicate, bitmask.as_ref(), chunk_id)
            .context("Failed to perform filter")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        (
            bitmask,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            compression_config,
        )
    };

    let result_string = format!("{:?}", bitmask);

    let output = OutputWrapper {
        in_memory,
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        execution_times: execution_times.into(),
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
        datetime: now,
    };

    Ok(output)
}

fn filter_materialize_command(
    in_memory: bool,
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

    let (compression_config, elapsed_time_nanos, execution_times, result_string) = if in_memory {
        let mut content = Vec::new();
        File::open(filename.as_ref())
            .context(format!(
                "Failed to open file.: {}",
                filename.as_ref().display()
            ))?
            .read_to_end(&mut content)
            .context(format!(
                "Failed to read file.: {}",
                filename.as_ref().display()
            ))?;
        let mut decoder = Decoder::new(DecoderInput::from_reader(Cursor::new(&content[..])))
            .context("Failed to create decoder from input memory")?;

        let start = std::time::Instant::now();
        let mut decoder_output = DecoderOutput::from_writer(Cursor::new(Vec::new()));
        decoder
            .filter_materialize(&mut decoder_output, predicate, bitmask.as_ref(), chunk_id)
            .context("Failed to perform filter materialize")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };
        let output = decoder_output.into_writer().into_inner();
        let result_string = format!("bytes: {:?}", output.len());

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    } else {
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
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };
        let metadata = decoder_output.into_writer().into_inner()?.metadata()?;
        let result_string = format!("{:?}", metadata);

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    };

    let output = OutputWrapper {
        in_memory,
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        execution_times: execution_times.into(),
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
        datetime: now,
    };

    Ok(output)
}

fn materialize_command(
    in_memory: bool,
    filename: impl AsRef<Path>,
    config: MaterializeConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<MaterializeConfig>> {
    let MaterializeConfig { chunk_id, bitmask } = config.clone();
    let bitmask = bitmask.map(ebi::decoder::query::RoaringBitmap::from_iter);
    let now = chrono::Utc::now();

    let (compression_config, elapsed_time_nanos, execution_times, result_string) = if in_memory {
        let mut content = Vec::new();
        File::open(filename.as_ref())
            .context(format!(
                "Failed to open file.: {}",
                filename.as_ref().display()
            ))?
            .read_to_end(&mut content)
            .context(format!(
                "Failed to read file.: {}",
                filename.as_ref().display()
            ))?;
        let mut decoder = Decoder::new(DecoderInput::from_reader(Cursor::new(&content[..])))
            .context("Failed to create decoder from input memory")?;

        let start = std::time::Instant::now();
        let mut decoder_output = DecoderOutput::from_writer(Cursor::new(Vec::new()));
        decoder
            .materialize(&mut decoder_output, bitmask.as_ref(), chunk_id)
            .context("Failed to perform filter materialize")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        let bytes = decoder_output.into_writer().into_inner();
        let result_string = format!("bytes: {:?}", bytes.len());

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    } else {
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
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        let metadata = decoder_output.into_writer().into_inner()?.metadata()?;
        let result_string = format!("{:?}", metadata);

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    };

    let output = OutputWrapper {
        in_memory,
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        execution_times: execution_times.into(),
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
        datetime: now,
    };

    Ok(output)
}

fn max_command(
    in_memory: bool,
    filename: impl AsRef<Path>,
    config: MaxConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<MaxConfig>> {
    let MaxConfig { chunk_id, bitmask } = config.clone();
    let bitmask = bitmask.map(ebi::decoder::query::RoaringBitmap::from_iter);
    let now = chrono::Utc::now();

    let (compression_config, elapsed_time_nanos, execution_times, result_string) = if in_memory {
        let mut content = Vec::new();
        File::open(filename.as_ref())
            .context(format!(
                "Failed to open file.: {}",
                filename.as_ref().display()
            ))?
            .read_to_end(&mut content)
            .context(format!(
                "Failed to read file.: {}",
                filename.as_ref().display()
            ))?;
        let mut decoder = Decoder::new(DecoderInput::from_reader(Cursor::new(&content[..])))
            .context("Failed to create decoder from input memory")?;

        let start = std::time::Instant::now();
        let max_fp = decoder
            .max(bitmask.as_ref(), chunk_id)
            .context("Failed to perform max")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        let result_string = max_fp.to_string();

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    } else {
        let mut decoder =
            Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
                "Failed to create decoder from input file.: {}",
                filename.as_ref().display()
            ))?;
        let start = std::time::Instant::now();
        let max_fp = decoder
            .max(bitmask.as_ref(), chunk_id)
            .context("Failed to perform max")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        let result_string = max_fp.to_string();

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    };

    let output = OutputWrapper {
        in_memory,
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        execution_times: execution_times.into(),
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
        datetime: now,
    };

    Ok(output)
}

fn sum_command(
    in_memory: bool,
    filename: impl AsRef<Path>,
    config: SumConfig,
    path: ConfigPath,
) -> anyhow::Result<OutputWrapper<SumConfig>> {
    let SumConfig { chunk_id, bitmask } = config.clone();
    let bitmask = bitmask.map(ebi::decoder::query::RoaringBitmap::from_iter);
    let now = chrono::Utc::now();

    let (compression_config, elapsed_time_nanos, execution_times, result_string) = if in_memory {
        let mut content = Vec::new();
        File::open(filename.as_ref())
            .context(format!(
                "Failed to open file.: {}",
                filename.as_ref().display()
            ))?
            .read_to_end(&mut content)
            .context(format!(
                "Failed to read file.: {}",
                filename.as_ref().display()
            ))?;
        let mut decoder = Decoder::new(DecoderInput::from_reader(Cursor::new(&content[..])))
            .context("Failed to create decoder from input memory")?;

        let start = std::time::Instant::now();
        let sum_fp = decoder
            .sum(bitmask.as_ref(), chunk_id)
            .context("Failed to perform max")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        let result_string = sum_fp.to_string();

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    } else {
        let mut decoder =
            Decoder::new(DecoderInput::from_file(filename.as_ref())?).context(format!(
                "Failed to create decoder from input file.: {}",
                filename.as_ref().display()
            ))?;
        let start = std::time::Instant::now();
        let sum_fp = decoder
            .sum(bitmask.as_ref(), chunk_id)
            .context("Failed to perform max")?;
        let elapsed_time_nanos = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);

        let compression_config = CompressionConfig {
            chunk_option: *decoder.header().config().chunk_option(),
            compressor_config: *decoder.footer().compressor_config(),
        };

        let result_string = sum_fp.to_string();

        (
            compression_config,
            elapsed_time_nanos,
            decoder.segmented_execution_times(),
            result_string,
        )
    };

    let output = OutputWrapper {
        in_memory,
        version: env!("CARGO_PKG_VERSION").to_string(),
        config_path: path.config.clone(),
        compression_config,
        command_specific: config,
        elapsed_time_nanos,
        execution_times: execution_times.into(),
        input_filename: filename.as_ref().to_string_lossy().to_string(),
        result_string,
        datetime: now,
    };

    Ok(output)
}

pub struct LabelPixel {
    pub label: isize,
    pub pixels: Vec<f64>,
}
pub fn slurp_file(file: impl AsRef<Path>, prec: i32) -> Vec<LabelPixel> {
    fn nan_to_zero(x: f64) -> f64 {
        if x.is_nan() {
            0.0
        } else {
            x
        }
    }
    let file = file.as_ref();
    BufReader::new(File::open(file).unwrap())
        .lines()
        .skip(1)
        .map(|line| {
            let line = line.unwrap();
            let mut iter = line.trim().split('\t').map(|x| x.parse::<f64>().unwrap());

            LabelPixel {
                label: iter.next().unwrap() as isize,
                pixels: {
                    if prec < 0 {
                        iter.map(nan_to_zero).collect()
                    } else {
                        iter.map(|x| round_by_scale(nan_to_zero(x), prec as usize))
                            .collect()
                    }
                },
            }
        })
        .collect()
}

/// Return `compression_method -> dataset_name -> [output]`
fn ucr2018_command(
    input_dir: impl AsRef<Path>,
    n: usize,
    k: usize,
    precision: i32,
    do_with_only_compression_methods_with_controlled_precision_support: bool,
) -> anyhow::Result<UCR2018ForAllCompressionMethodsResult> {
    let start_time = chrono::Utc::now();
    anyhow::ensure!(k == 1, "k must be 1");
    anyhow::ensure!(
        precision >= -1,
        "precision must be greater than or equal to -1"
    );
    anyhow::ensure!(
        do_with_only_compression_methods_with_controlled_precision_support || precision == -1,
        "precision must be -1 if do_with_only_compression_methods_with_controlled_precision_support is false"
    );
    if precision == -1 {
        println!("Precision is -1, It means the precision will be automatically detected.");
    }
    let input_dir = input_dir.as_ref();
    dbg!(input_dir);
    let dataset_entries = entry_dir_iter(input_dir)?
        .flat_map(|path| {
            const MISSING_VALUE_AND_VARIABLE_LENGTH_DATASETS_ADJUSTED: &str =
                "Missing_value_and_variable_length_datasets_adjusted";
            if path.file_name().unwrap() == MISSING_VALUE_AND_VARIABLE_LENGTH_DATASETS_ADJUSTED {
                Either::Left(entry_dir_iter(path).unwrap())
            } else {
                Either::Right(iter::once(path))
            }
        })
        .collect::<Vec<_>>();
    dbg!(&dataset_entries);

    let mut dataset_scales = HashMap::new();
    let scale_fallbacked_datasets = Vec::new();
    for dataset_entry in tqdm(dataset_entries.iter()) {
        if !dataset_entry.is_dir() {
            continue;
        }

        let dataset_name = dataset_entry
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let train_file = dataset_entry.join(format!("{}_TRAIN.tsv", dataset_name));
        let train_file_reader = BufReader::new(
            File::open(&train_file)
                .context(format!("Failed to open file.: {}", train_file.display()))?,
        );
        let prec = get_appropriate_precision(train_file_reader, '\t', false).context(format!(
            "Failed to get appropriate prec for {}, Fallback to precision 9, scale: 10^9",
            &dataset_name
        ))?;
        let scale = 10u64.checked_pow(prec).context(format!(
            "Failed to calculate scale for {} with precision {}",
            &dataset_name, prec
        ))?;
        dataset_scales.insert(dataset_name, scale);
    }
    dbg!(&dataset_scales);

    let mut configs: Vec<CompressorConfig> = vec![CompressorConfig::buff()
        .scale(0) // scale will be set later
        .build()
        .into()];

    if !do_with_only_compression_methods_with_controlled_precision_support {
        configs.extend(&[
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
                .scale(0) // scale will be set later
                .build()
                .into(),
        ]);
    }

    let mut dataset_to_vector_length = HashMap::new();

    for dataset_entry in tqdm(dataset_entries.iter()).desc(Some("dataset")) {
        if !dataset_entry.is_dir() {
            continue;
        }

        let dataset_name = dataset_entry
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let test_file = dataset_entry.join(format!("{}_TEST.tsv", dataset_name));
        let test_data = slurp_file(&test_file, dataset_scales[&dataset_name] as i32);
        let test_vectors = test_data
            .iter()
            .map(|x| x.pixels.clone())
            .collect::<Vec<_>>();
        dataset_to_vector_length.insert(dataset_name.clone(), test_vectors[0].len());
    }

    let mut results_for_all_ucr2018 = HashMap::new();
    for config in tqdm(configs).desc(Some("compression_method")) {
        let start_time = chrono::Utc::now();
        let mut results = HashMap::new();
        for dataset_entry in tqdm(dataset_entries.iter()).desc(Some("dataset")) {
            if !dataset_entry.is_dir() {
                continue;
            }

            let dataset_name = dataset_entry
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            println!(
                "dataset: {}\tcompression_method: {:?}",
                &dataset_name,
                config.compression_scheme()
            );
            let config = {
                // set scale if precision is -1 (auto detect)
                let mut config = config;
                config.set_scale(dataset_scales[&dataset_name]);
                config
            };
            let train_file = dataset_entry.join(format!("{}_TRAIN.tsv", dataset_name));
            let test_file = dataset_entry.join(format!("{}_TEST.tsv", dataset_name));

            let train_data = slurp_file(&train_file, dataset_scales[&dataset_name] as i32);
            let test_data = slurp_file(&test_file, dataset_scales[&dataset_name] as i32);
            let train_vectors = train_data
                .iter()
                .flat_map(|x| x.pixels.clone())
                .collect::<Vec<_>>();
            let train_labels = train_data.iter().map(|x| x.label).collect::<Vec<_>>();

            let test_vectors = test_data
                .iter()
                .map(|x| x.pixels.clone())
                .collect::<Vec<_>>();
            let test_labels = test_data.iter().map(|x| x.label).collect::<Vec<_>>();

            let train_vectors_encoded = {
                let input = EncoderInput::from_f64_slice(&train_vectors);
                let output = EncoderOutput::from_vec(Vec::new());

                let mut encoder = Encoder::new(input, output, DEFAULT_CHUNK_OPTION, config);

                encoder
                    .encode()
                    .context(format!("Failed to encode {}", dataset_name))?;

                encoder.into_output().into_inner().into_inner()
            };

            let (compression_statistics, decompression_result) = {
                let n = 5;
                let mut statistics = None;
                let mut elapsed_time_nanos = Vec::with_capacity(n);
                let mut segmented_execution_times = Vec::with_capacity(n);
                let mut result_string = Vec::with_capacity(n);
                for _ in 0..n {
                    let decoder_input =
                        DecoderInput::from_reader(Cursor::new(&train_vectors_encoded[..]));
                    let mut decoder = Decoder::new(decoder_input)
                        .context(format!("Failed to create decoder for {}", dataset_name))?;
                    statistics = Some(get_compress_statistics(&decoder)?);
                    if precision >= 0
                        && do_with_only_compression_methods_with_controlled_precision_support
                    {
                        let original_precision =
                            (dataset_scales[&dataset_name] as f64).log10().round() as u32;
                        decoder.with_precision((precision as u32).min(original_precision));
                    }

                    let start = std::time::Instant::now();
                    let mut output = DecoderOutput::from_vec(Vec::new());
                    decoder.materialize(&mut output, None, None)?;
                    let elapsed_time_nano = start.elapsed().as_nanos();
                    let segmented_execution_time: SerializableSegmentedExecutionTimes =
                        decoder.segmented_execution_times().into();
                    let result_str = format!(
                        "{:?}",
                        output
                            .into_writer()
                            .into_inner()
                            .chunks(8)
                            .take(5)
                            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
                            .collect::<Vec<_>>()
                    );
                    elapsed_time_nanos.push(elapsed_time_nano.try_into().unwrap_or(u64::MAX));
                    segmented_execution_times.push(segmented_execution_time);
                    result_string.push(result_str);
                }

                let decompression_result = UCR2018DecompressionResult {
                    n,
                    elapsed_time_nanos,
                    execution_times: segmented_execution_times,
                    result_string,
                };
                (statistics.unwrap(), decompression_result)
            };

            let mut n_correct = 0;
            let n_test_vectors = test_vectors.len();
            let mut elapsed_time_nanos = Vec::with_capacity(n_test_vectors);
            let mut segmented_execution_times = Vec::with_capacity(n_test_vectors);
            for (i, test_vector) in
                tqdm(test_vectors.into_iter().enumerate()).desc(Some("test_vectors"))
            {
                let mut is_correct = false;
                let mut elapsed_time_nanos_for_one_test_vector = Vec::with_capacity(n);
                let mut segmented_execution_times_for_one_test_vector = Vec::with_capacity(n);
                for _ in 0..n {
                    let decoder_input =
                        DecoderInput::from_reader(Cursor::new(&train_vectors_encoded[..]));
                    let mut decoder = Decoder::new(decoder_input)
                        .context(format!("Failed to create decoder for {}", dataset_name))?;

                    if precision >= 0
                        && do_with_only_compression_methods_with_controlled_precision_support
                    {
                        let original_precision =
                            (dataset_scales[&dataset_name] as f64).log10().round() as u32;
                        decoder.with_precision((precision as u32).min(original_precision));
                    }

                    let start = std::time::Instant::now();
                    let knn_result = decoder.knn1(&test_vector[..])?;
                    let elapsed_time_nano = start.elapsed().as_nanos();

                    elapsed_time_nanos_for_one_test_vector.push(elapsed_time_nano);
                    segmented_execution_times_for_one_test_vector
                        .push(decoder.segmented_execution_times().into());

                    is_correct = test_labels[i] == train_labels[knn_result.index];
                }

                if is_correct {
                    n_correct += 1;
                }

                elapsed_time_nanos.push(elapsed_time_nanos_for_one_test_vector);
                segmented_execution_times.push(segmented_execution_times_for_one_test_vector);
            }
            let accuracy = n_correct as f64 / n_test_vectors as f64;

            let compression_config = CompressionConfig::new(DEFAULT_CHUNK_OPTION, config);
            let result = UCR2018ResultForOneDataset {
                dataset_name: dataset_name.clone(),
                config: UCR2018Config {
                    n,
                    precision: precision as isize,
                    k,
                },
                compression_config,
                elapsed_time_nanos,
                execution_times: segmented_execution_times,
                accuracy,
                compression_statistics,
                decompression_result,
            };

            results.insert(dataset_name, result);
        }

        let end_time = chrono::Utc::now();
        let ucr2018_result = UCR2018Result {
            start_time,
            end_time,
            results,
            scale_fallbacked_dataset: if precision == -1 {
                Some(scale_fallbacked_datasets.clone())
            } else {
                None
            },
        };

        let compression_scheme_key = format!("{:?}", config.compression_scheme());
        results_for_all_ucr2018.insert(compression_scheme_key, ucr2018_result);
    }

    let end_time = chrono::Utc::now();
    let result_for_all = UCR2018ForAllCompressionMethodsResult {
        results: results_for_all_ucr2018,
        dataset_to_vector_length,
        dataset_to_scale: dataset_scales,
        start_time,
        end_time,
    };

    Ok(result_for_all)
}

fn matrix_command(
    precision: u32,
    matrix_size: usize,
    do_with_only_compression_methods_with_controlled_precision_support: bool,
) -> anyhow::Result<HashMap<String, MatrixResult>> {
    let original_data_precision = 8;
    let scale = 10u64.pow(original_data_precision);
    let seed: u64 = (original_data_precision as usize * matrix_size + 42).try_into()?;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    const N_DATA_MATRIX: usize = 40;

    let data_array_size = N_DATA_MATRIX * matrix_size * matrix_size;
    let mut data_matrices = vec![0.0; data_array_size];

    rng.fill(&mut data_matrices[..]);

    let mut target_matrix = vec![0.0; matrix_size * matrix_size];

    rng.fill(&mut target_matrix[..]);

    /// Map value from [0, 1) to [-4, 4)
    fn map_value(x: f64) -> f64 {
        (x - 0.5) * 8.0
    }

    data_matrices
        .iter_mut()
        .for_each(|x| *x = round_by_scale(map_value(*x), scale as usize));
    target_matrix
        .iter_mut()
        .for_each(|x| *x = round_by_scale(map_value(*x), scale as usize));

    match DEFAULT_CHUNK_OPTION {
        ChunkOption::RecordCount(c) => assert!(
            c % (matrix_size * matrix_size) == 0,
            "{} % ({} * {}) != 0",
            c,
            matrix_size,
            matrix_size
        ),
        ChunkOption::ByteSizeBestEffort(_) | ChunkOption::Full => unreachable!(),
    }

    // Configs with controlled precision support
    let mut configs: Vec<CompressorConfig> =
        vec![CompressorConfig::buff().scale(scale).build().into()];

    if !do_with_only_compression_methods_with_controlled_precision_support {
        configs.extend_from_slice(&[
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
        ]);
    }
    let mut results = HashMap::new();
    for config in /*tqdm(configs)*/ configs {
        let start_time = chrono::Utc::now();
        let (encoded, compression_elapsed_time_nano_secs) = {
            let input = EncoderInput::from_f64_slice(&data_matrices);
            let output = EncoderOutput::from_vec(Vec::new());

            let mut encoder = Encoder::new(input, output, DEFAULT_CHUNK_OPTION, config);

            let start = std::time::Instant::now();
            encoder.encode().context("Failed to encode")?;
            let elapsed_time_nanos = start.elapsed().as_nanos();

            (
                encoder.into_output().into_inner().into_inner(),
                elapsed_time_nanos,
            )
        };

        let decoder_input = DecoderInput::from_reader(Cursor::new(&encoded[..]));
        let mut decoder = Decoder::new(decoder_input)?;
        decoder.with_precision(precision);

        let compression_statistics = get_compress_statistics(&decoder)?;

        let start = std::time::Instant::now();
        let matmul_result = decoder.matmul(
            &target_matrix,
            (matrix_size, matrix_size),
            (matrix_size, matrix_size),
        )?;
        let elapsed_time_nanos = start.elapsed().as_nanos();
        let segmented_execution_times = decoder.segmented_execution_times();

        let end_time = chrono::Utc::now();
        let result = MatrixResult {
            n_data_matrix: N_DATA_MATRIX,
            compression_config: CompressionConfig::new(DEFAULT_CHUNK_OPTION, config),
            compression_elapsed_time_nano_secs: compression_elapsed_time_nano_secs as u64,
            compression_statistics,
            matmul_elapsed_time_nano_secs: elapsed_time_nanos as u64,
            matmul_segmented_execution_times: segmented_execution_times.into(),
            precision,
            matrix_size,
            result_string: format!("{:?}", &matmul_result[..5.min(matmul_result.len())]),

            start_time,
            end_time,
        };

        results.insert(format!("{:?}", config.compression_scheme()), result);
    }

    Ok(results)
}

fn embedding_command(
    input_dir: impl AsRef<Path>,
    // dataaset_name -> method_name -> result
) -> anyhow::Result<HashMap<String, HashMap<String, SimpleCompressionPerformanceForOneDataset>>> {
    let datasets_exclude = [];

    let dataset_entries = entry_file_iter(input_dir.as_ref())?
        .filter(|entry| {
            let dataset_name = entry.file_stem().unwrap().to_string_lossy();
            let ext = entry.as_path().extension().unwrap().to_string_lossy();
            !datasets_exclude.contains(&dataset_name.as_ref()) && ext == "bin"
        })
        .collect::<Vec<_>>();

    let precision_map: HashMap<String, f64> = serde_json::from_reader(
        File::open(input_dir.as_ref().join("precision_data.json"))
            .context("Failed to open precision_data.json")?,
    )
    .context("Failed to parse precision_data.json")?;
    let precision_map: HashMap<String, u64> = precision_map
        .into_iter()
        .map(|(k, v)| (k, v.round() as u64))
        .collect();
    let scale_map: HashMap<String, Option<u64>> = precision_map
        .clone()
        .into_iter()
        .map(|(k, v)| (k, 10u64.checked_pow(v as u32)))
        .collect();
    // scale is set to 10^precision later
    let configs: Vec<CompressorConfig> = vec![
        CompressorConfig::buff().scale(0).build().into(),
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
        CompressorConfig::delta_sprintz().scale(0).build().into(),
    ];

    let mut results = HashMap::new();
    let n = 10;
    for dataset_entry in tqdm(dataset_entries) {
        let dataset_key = dataset_entry
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let scale = scale_map[&dataset_key];

        // encode
        let mut input_in_memory = Vec::new();
        BufReader::new(File::open(dataset_entry.as_path())?).read_to_end(&mut input_in_memory)?;

        let mut results_of_this_dataset = HashMap::new();
        for config in tqdm(configs.clone()) {
            let config = {
                let mut config = config;
                if matches!(
                    config.compression_scheme(),
                    CompressionScheme::DeltaSprintz | CompressionScheme::BUFF
                ) {
                    let scale = match scale {
                        // if precision > 12, skip for the method
                        Some(scale) if scale > 10u64.pow(12) => {
                            eprintln!(
                                "Scale is too large for {}, skipping for the method {:?}",
                                dataset_key,
                                config.compression_scheme()
                            );
                            continue;
                        }
                        None => {
                            eprintln!(
                                "Failed to get scale for {}, skipping for the method {:?}",
                                dataset_key,
                                config.compression_scheme()
                            );
                            continue;
                        }
                        Some(scale) => scale,
                    };
                    config.set_scale(scale);
                }
                config
            };
            let mut compression_elapsed_time_nanos = Vec::with_capacity(n);
            let mut compression_segmented_execution_times = Vec::with_capacity(n);
            let mut compression_result_string = Vec::with_capacity(n);
            let mut compression_statistics = Vec::with_capacity(n);

            let mut decompression_elapsed_time_nanos = Vec::with_capacity(n);
            let mut decompression_segmented_execution_times = Vec::with_capacity(n);
            let mut decompression_result_string = Vec::with_capacity(n);

            for _ in 0..n {
                let mut encoder = Encoder::new(
                    EncoderInput::from_reader(Cursor::new(&input_in_memory[..])),
                    EncoderOutput::from_vec(Vec::new()),
                    DEFAULT_CHUNK_OPTION,
                    config,
                );
                let start = std::time::Instant::now();
                encoder.encode()?;
                let compression_elapsed_time_nano_secs = start.elapsed().as_nanos();
                let encoded = encoder.into_output().into_inner().into_inner();

                let decoder_input = DecoderInput::from_reader(Cursor::new(&encoded[..]));
                let mut decoder = Decoder::new(decoder_input)?;

                compression_elapsed_time_nanos.push(
                    compression_elapsed_time_nano_secs
                        .try_into()
                        .unwrap_or(u64::MAX),
                );
                compression_segmented_execution_times
                    .push((*decoder.footer().segmented_execution_times()).into());
                compression_result_string.push(format!("{:?}", &encoded[..5.min(encoded.len())]));

                let statistics = get_compress_statistics(&decoder)?;
                compression_statistics.push(statistics);

                // decode
                let mut decoder_output = DecoderOutput::from_vec(Vec::new());
                let start = std::time::Instant::now();
                decoder.materialize(&mut decoder_output, None, None)?;
                let decompression_elapsed_time_nano_secs = start.elapsed().as_nanos();

                decompression_elapsed_time_nanos.push(
                    decompression_elapsed_time_nano_secs
                        .try_into()
                        .unwrap_or(u64::MAX),
                );
                decompression_segmented_execution_times
                    .push(decoder.segmented_execution_times().into());
                decompression_result_string.push(format!(
                    "{:?}",
                    &decoder_output.into_writer().into_inner()[..5.min(encoded.len())]
                ));
            }

            let result = SimpleCompressionPerformanceForOneDataset {
                n,
                dataset_name: dataset_key.clone(),
                compression_config: CompressionConfig::new(DEFAULT_CHUNK_OPTION, config),
                compression_elapsed_time_nanos,
                compression_segmented_execution_times,
                compression_result_string,
                compression_statistics,
                decompression_elapsed_time_nanos,
                decompression_segmented_execution_times,
                decompression_result_string,
                precision: precision_map[&dataset_key],
            };

            results_of_this_dataset.insert(format!("{:?}", config.compression_scheme()), result);
        }

        results.insert(dataset_key.clone(), results_of_this_dataset);
    }

    Ok(results)
}
