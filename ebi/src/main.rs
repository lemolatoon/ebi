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
fn generate_and_write_random_f64(path: impl AsRef<Path>, n: usize) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut random_values: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if rng.gen_bool(0.0) && random_values.last().is_some() {
            random_values.push(random_values[i - 1]);
        } else {
            let mut fp: f64 = rng.gen_range(0.0..1000.0);
            // let mut fp: f64 = rng.gen_range(0.0..1.0);
            fp = 10.0 + (fp * 100.0).floor() / 100.0;
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
    const RECORD_COUNT: usize = 5;
    // generate_and_write_random_f64("uncompressed.bin", RECORD_COUNT * 3).unwrap();
    // let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new(100));
    // let compressor_config = CompressorConfig::rle().build();
    // let compressor_config = CompressorConfig::gorilla().build();
    let scale = 10;
    let compressor_config = CompressorConfig::buff().scale(scale).build();
    let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);

    let binding = vec![0.4, 0.5];
    let mut encoder = Encoder::new(
        // EncoderInput::from_file("uncompressed.bin").unwrap(),
        EncoderInput::from_f64_slice(&binding),
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
    let chunk_footers = reader.footer().chunk_footers();
    println!("{:X?}", chunk_footers);

    let mut output = DecoderOutput::from_vec(Vec::with_capacity(RECORD_COUNT * 3000 + 3));

    let bitmask = decoder.filter(Predicate::Eq(21.68), None, None).unwrap();
    println!("{:?}", bitmask);

    decoder.materialize(&mut output, None, None).unwrap();

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
            // (fp * 100.0).round() / 100.0
            fp
        })
        .collect();

    let input_f64: Vec<f64> = input_bytes
        .chunks_exact(8)
        .map(|chunk| {
            let fp = f64::from_ne_bytes(chunk.try_into().unwrap());
            (fp * scale as f64).round() / scale as f64
        })
        .collect();

    assert_eq!(&output_f64, &input_f64, "Decompressed Floats mismatch");
    println!("{:?}", &output_f64);
}
