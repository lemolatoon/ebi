use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};

#[allow(unused_imports)]
use ebi::compressor::{
    gorilla::GorillaCompressor, run_length::RunLengthCompressor,
    uncompressed::UncompressedCompressor,
};

use ebi::{
    api::{
        decoder::{ChunkId, Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::query::Predicate,
    encoder::ChunkOption,
};
use rand::Rng;

#[allow(dead_code)]
fn generate_and_write_random_f64(path: impl AsRef<Path>, n: usize, scale: usize) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut random_values: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if rng.gen_bool(0.0) && random_values.last().is_some() {
            random_values.push(random_values[i - 1]);
        } else {
            let mut fp: f64 = rng.gen_range(0.0..1000.0);
            // let mut fp: f64 = rng.gen_range(0.0..1.0);
            fp = 10.0 + (fp * scale as f64).floor() / scale as f64;
            random_values.push(fp);
        }
    }

    let mut file = File::create(path)?;

    for &value in &random_values {
        let bytes = value.to_ne_bytes();
        file.write_all(&bytes)?;
    }

    println!("{:?}", &random_values[0..std::cmp::min(10, n)]);

    Ok(())
}

fn main() {
    const RECORD_COUNT: usize = 32;
    let scale = 10;
    generate_and_write_random_f64("uncompressed.bin", RECORD_COUNT + 3, scale).unwrap();
    // let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new(100));
    // let compressor_config = CompressorConfig::rle().build();
    // let compressor_config = CompressorConfig::gorilla().build();
    let compressor_config = CompressorConfig::buff().scale(scale).build();
    // let chunk_option = ChunkOption::RecordCount(RECORD_COUNT * 3000 + 3);
    let chunk_option = ChunkOption::ByteSizeBestEffort(1024 * 8);
    dbg!(chunk_option);

    // let binding = vec![0.4, 100000000.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0];
    let mut encoder = Encoder::new(
        EncoderInput::from_file_with_capacity("uncompressed.bin", 1024 * 16).unwrap(),
        // EncoderInput::from_f64_slice(&binding),
        EncoderOutput::from_file("compressed.bin").unwrap(),
        chunk_option,
        compressor_config,
    );
    encoder.encode().unwrap();

    let input = DecoderInput::from_file("compressed.bin")
        .unwrap()
        .into_buffered();
    let mut decoder = Decoder::new(input).unwrap();

    let reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();
    dbg!(reader.footer().number_of_records() as usize * size_of::<f64>());
    dbg!(reader.footer().number_of_chunks());
    let chunk_footers = reader.footer().chunk_footers();
    println!("{:X?}", &chunk_footers[0..chunk_footers.len().min(10)]);

    let mut output = DecoderOutput::from_vec(Vec::with_capacity(RECORD_COUNT * 3000 + 3));
    let mut output2 = DecoderOutput::from_vec(Vec::with_capacity(RECORD_COUNT * 3000 + 3));

    let bitmask = decoder.filter(Predicate::Eq(21.68), None, None).unwrap();
    println!("{:?}", bitmask);

    decoder.materialize(&mut output, None, None).unwrap();
    decoder
        .filter_materialize(&mut output2, Predicate::Eq(0.9), None, None)
        .unwrap();

    let output = output.into_writer().into_inner();

    println!("{:?}", output.len());

    let mut input_bytes = Vec::new();
    File::open("uncompressed.bin")
        .unwrap()
        .read_to_end(&mut input_bytes)
        .unwrap();

    assert_eq!(output.len(), input_bytes.len(), "Length mismatch");

    let output_f64: Vec<f64> = output
        .chunks_exact(8)
        .map(|chunk| {
            let fp = f64::from_ne_bytes(chunk.try_into().unwrap());
            (fp * scale as f64).round() / scale as f64
        })
        .collect();

    let input_f64: Vec<f64> = input_bytes
        .chunks_exact(8)
        .map(|chunk| {
            let fp = f64::from_ne_bytes(chunk.try_into().unwrap());
            (fp * scale as f64).round() / scale as f64
        })
        .collect();
    // let input_f64 = binding;

    assert_eq!(
        &output_f64[..10],
        &input_f64[..10],
        "Decompressed Floats mismatch"
    );

    assert_eq!(&output_f64, &input_f64, "Decompressed Floats mismatch");
}
