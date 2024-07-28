use std::{
    fs::File,
    io::{self, Write},
    path::Path,
};

#[allow(unused_imports)]
use ebi::compressor::{
    gorilla::GorillaCompressor, run_length::RunLengthCompressor,
    uncompressed::UncompressedCompressor,
};

use ebi::{
    api::{
        decoder::{Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    encoder::ChunkOption,
};
use rand::Rng;

#[allow(dead_code)]
fn generate_and_write_random_f64(path: impl AsRef<Path>, n: usize) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut random_values: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if rng.gen_bool(0.5) && random_values.last().is_some() {
            random_values.push(random_values[i - 1]);
        } else {
            let mut fp: f64 = rng.gen();
            fp = (fp * 10.0).floor() / 10.0;
            random_values.push(fp);
        }
    }

    let mut file = File::create(path)?;

    for value in random_values {
        let bytes = value.to_ne_bytes();
        file.write_all(&bytes)?;
    }

    Ok(())
}

fn main() {
    const RECORD_COUNT: usize = 100;
    generate_and_write_random_f64("uncompressed.bin", RECORD_COUNT * 3000 + 3).unwrap();
    // let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new(100));
    // let compressor_config = CompressorConfig::rle().build();
    // let compressor_config = CompressorConfig::gorilla().build();
    let compressor_config = CompressorConfig::buff().build();
    let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);

    let mut encoder = Encoder::new(
        EncoderInput::from_file("uncompressed.bin").unwrap(),
        EncoderOutput::from_file("compressed.bin").unwrap(),
        chunk_option,
        compressor_config,
    );
    encoder.encode().unwrap();

    let input = DecoderInput::from_file("compressed.bin")
        .unwrap()
        .into_buffered();
    let mut decoder = Decoder::new(input).unwrap();

    let mut output = DecoderOutput::from_vec(Vec::with_capacity(RECORD_COUNT * 3000 + 3));

    decoder.materialize(&mut output, None, None).unwrap();

    let output = output.into_writer().into_inner();

    println!("{:?}", output.len());
}
