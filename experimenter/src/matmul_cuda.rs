use std::{collections::HashMap, io::Cursor};
use tqdm::tqdm;

use anyhow::Context as _;
use ebi::{
    api::{
        decoder::{Decoder, DecoderInput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    encoder::ChunkOption,
};
use rand::{Rng as _, SeedableRng as _};
use rand_chacha::ChaCha20Rng;

use crate::{
    get_compress_statistics, round_by_scale, CompressionConfig, MatrixResult, DEFAULT_CHUNK_OPTION,
};

pub fn matrix_cuda_command(
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
    for config in /*tqdm(configs)*/ tqdm(configs) {
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
        let matmul_result = decoder.matmul_cuda(
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
