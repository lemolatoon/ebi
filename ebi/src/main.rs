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
    const RECORD_COUNT: usize = 3;
    let scale = 100;
    // generate_and_write_random_f64("uncompressed.bin", RECORD_COUNT * 3 + 3, scale).unwrap();
    // let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new(100));
    // let compressor_config = CompressorConfig::rle().build();
    // let compressor_config = CompressorConfig::gorilla().build();
    // let compressor_config = CompressorConfig::buff().scale(scale).build();
    // let compressor_config = CompressorConfig::chimp128().build();
    // let compressor_config = CompressorConfig::elf_on_chimp().build();
    // let compressor_config = CompressorConfig::elf().build();
    // let compressor_config = CompressorConfig::chimp().build();
    let compressor_config = CompressorConfig::delta_sprintz()
        .scale(scale as u32)
        .build();
    let chunk_option = ChunkOption::RecordCount(RECORD_COUNT * 10 + 3);
    // let chunk_option = ChunkOption::ByteSizeBestEffort(1024 * 8);
    dbg!(chunk_option);

    const _XOR_VALUES: [f64; 66] = const {
        let mut values = [0.0f64; 66];
        values[0] = 3.3;
        values[1] = 3.3;
        let mut i = 1;
        while i < 64 {
            let last = values[i];
            let last_bits = unsafe { std::mem::transmute::<f64, u64>(last) };
            let next = (last_bits) ^ (1 << (i - 1));
            let next = unsafe { std::mem::transmute::<u64, f64>(next) };
            // `f64::is_nan` is const unstable
            #[allow(clippy::eq_op)]
            let is_nan = next != next;
            let is_infinite = (next == f64::INFINITY) | (next == f64::NEG_INFINITY);
            if is_nan || is_infinite {
                values[i] = last * -2.0;
            } else {
                values[i + 1] = next;
                i += 1;
            }
        }

        values
    };
    // let binding = &XOR_VALUES;
    // let binding = &[3.0, 40000000000000000003.0, 5.0];
    // let binding = &[24903104499507892000.0];
    // let binding = vec![0.4, 100000000.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0];
    let binding = &[(i64::MAX / (2 * scale) as i64) as f64];
    println!("binding: {}", binding[0]);
    let mut encoder = Encoder::new(
        // EncoderInput::from_file_with_capacity("uncompressed.bin", 1024 * 16).unwrap(),
        EncoderInput::from_f64_slice(binding),
        EncoderOutput::from_file("compressed.bin").unwrap(),
        chunk_option,
        compressor_config,
    );
    encoder.encode().unwrap();
    println!("encoded!!!!!");

    let input = DecoderInput::from_file("compressed.bin")
        .unwrap()
        .into_buffered();
    let mut decoder = Decoder::new(input).unwrap();

    let reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();
    dbg!(reader.footer().number_of_records() as usize * size_of::<f64>());
    dbg!(reader.footer().number_of_chunks());
    let chunk_footers = reader.footer().chunk_footers();
    println!("{:X?}", &chunk_footers[0..chunk_footers.len().min(10)]);

    let mut output = DecoderOutput::from_vec(Vec::new());
    decoder.materialize(&mut output, None, None).unwrap();

    let mut input_bytes = Vec::new();
    File::open("uncompressed.bin")
        .unwrap()
        .read_to_end(&mut input_bytes)
        .unwrap();
    let _input_floats: Vec<f64> = input_bytes
        .chunks_exact(8)
        .map(|b| {
            let fp = f64::from_le_bytes(b.try_into().unwrap());
            (fp * scale as f64).round() / scale as f64
        })
        .collect();
    let input_floats = binding;

    let output_bytes = output.into_writer().into_inner();
    let output_floats: Vec<f64> = output_bytes
        .chunks_exact(8)
        .map(|b| {
            let fp = f64::from_le_bytes(b.try_into().unwrap());
            (fp * scale as f64).round() / scale as f64
        })
        .collect();

    assert_eq!(input_floats.len(), output_floats.len());

    assert_eq!(&input_floats[..], &output_floats[..]);
}
