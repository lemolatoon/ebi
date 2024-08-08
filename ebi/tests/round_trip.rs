use std::io::Cursor;

use ebi::{
    api::{
        decoder::{Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::query::{Predicate, Range, RangeValue},
    encoder::ChunkOption,
};
use rand::Rng;

#[cfg(test)]
pub(crate) fn generate_and_write_random_f64(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut random_values: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if rng.gen_bool(0.5) && random_values.last().is_some() {
            random_values.push(random_values[i - 1]);
        } else {
            random_values.push(rng.gen());
        }
    }

    random_values
}

#[cfg(test)]
pub(crate) fn generate_and_write_random_f64_with_precision(n: usize, scale: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut random_values: Vec<f64> = Vec::with_capacity(n);

    fn round_by_scale(value: f64, scale: usize) -> f64 {
        let scale = scale as f64;
        (value * scale).round() / scale
    }

    for i in 0..n {
        if rng.gen_bool(0.5) && random_values.last().is_some() {
            random_values.push(random_values[i - 1]);
        } else {
            random_values.push(round_by_scale(rng.gen_range(-10.0..10000.0), scale));
        }
    }

    random_values
}

fn record_count_chunk_option_by_n(n: usize) -> ChunkOption {
    ChunkOption::RecordCount(n / 5)
}

fn bytesize_chunk_option_by_n(n: usize) -> ChunkOption {
    ChunkOption::ByteSizeBestEffort(n / 3)
}

#[test]
fn test_api_round_trip_uncompressed() {
    let header = b"my_header".to_vec().into_boxed_slice();
    let compressor_config = CompressorConfig::uncompressed()
        .capacity(8000)
        .header(header)
        .build();
    test_round_trip(compressor_config.into(), record_count_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_uncompressed_bytesize() {
    let header = b"my_header".to_vec().into_boxed_slice();
    let compressor_config = CompressorConfig::uncompressed()
        .capacity(8000)
        .header(header)
        .build();
    test_round_trip(compressor_config.into(), bytesize_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_gorilla() {
    let compressor_config = CompressorConfig::gorilla().capacity(8000).build();
    test_round_trip(compressor_config.into(), record_count_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_gorilla_bytesize() {
    let compressor_config = CompressorConfig::gorilla().capacity(8000).build();
    test_round_trip(compressor_config.into(), bytesize_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_rle() {
    let compressor_config = CompressorConfig::rle().build();
    test_round_trip(compressor_config.into(), record_count_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_rle_bytesize() {
    let compressor_config = CompressorConfig::rle().build();
    test_round_trip(compressor_config.into(), bytesize_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_chimp() {
    let compressor_config = CompressorConfig::chimp().build();
    test_round_trip(compressor_config.into(), record_count_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_chimp_bytesize() {
    let compressor_config = CompressorConfig::chimp().build();
    test_round_trip(compressor_config.into(), bytesize_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_elf_on_chimp() {
    let compressor_config = CompressorConfig::elf_on_chimp().build();
    test_round_trip(compressor_config.into(), record_count_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_elf_on_chimp_bytesize() {
    let compressor_config = CompressorConfig::elf_on_chimp().build();
    test_round_trip(compressor_config.into(), bytesize_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_chimp128() {
    let compressor_config = CompressorConfig::chimp128().build();
    test_round_trip(compressor_config.into(), record_count_chunk_option_by_n);
}

#[test]
fn test_api_round_trip_chimp128_bytesize() {
    let compressor_config = CompressorConfig::chimp128().build();
    test_round_trip(compressor_config.into(), bytesize_chunk_option_by_n);
}

fn test_round_trip_with_scale(
    generator: fn(usize, usize) -> Vec<f64>,
    compressor_config: CompressorConfig,
    scale: usize,
    chunk_option: impl Fn(usize) -> ChunkOption,
) {
    for n in [103, 1003, 100005] {
        #[cfg(miri)] // miri is too slow
        if n > 103 {
            continue;
        }

        let random_values = generator(n, scale);
        // encode
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&random_values);

            let encoder_output = EncoderOutput::from_vec(Vec::new());

            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                chunk_option(n),
                compressor_config.clone(),
            );

            encoder.encode().unwrap();

            encoder.into_output().into_vec()
        };

        // decode
        let decoded_filter_materialize = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);

            let mut decoder_output = DecoderOutput::from_vec(Vec::new());

            let mut decoder = Decoder::new(decoder_input).unwrap();

            decoder
                .filter_materialize(
                    &mut decoder_output,
                    Predicate::Range(Range::new(
                        RangeValue::Inclusive(-1000.0),
                        RangeValue::Inclusive(10000000.0),
                    )),
                    None,
                    None,
                )
                .unwrap();

            decoder_output.into_writer().into_inner()
        };

        let decoded_materialize = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);

            let mut decoder_output = DecoderOutput::from_vec(Vec::new());

            let mut decoder = Decoder::new(decoder_input).unwrap();

            let bm0 = decoder.filter(Predicate::Ne(0.5), None, None).unwrap();
            let bm1 = decoder.filter(Predicate::Eq(0.5), None, None).unwrap();
            decoder
                .materialize(&mut decoder_output, Some(&(bm0 | bm1)), None)
                .unwrap();

            decoder_output.into_writer().into_inner()
        };

        for decoded in [decoded_filter_materialize, decoded_materialize] {
            let random_values_bytes = random_values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>();
            assert_eq!(
                random_values_bytes.len(),
                decoded.len(),
                "The length of the decoded values is not equal to the length of the original values. {} != {}", random_values_bytes.len(), decoded.len()
            );
            println!("passed");

            let random_floats_rounded = random_values
                .iter()
                .map(|&x| {
                    let scale = scale as f64;
                    (x * scale).round() / scale
                })
                .collect::<Vec<f64>>();

            let decoded_floats_rounded = decoded
                .chunks_exact(8)
                .map(|x| {
                    let x = f64::from_ne_bytes(x.try_into().unwrap());
                    let scale = scale as f64;
                    (x * scale).round() / scale
                })
                .collect::<Vec<f64>>();

            assert_eq!(
                random_floats_rounded.len(),
                decoded_floats_rounded.len(),
                "The length of the decoded values is not equal to the length of the original values"
            );

            assert_eq!(
                random_floats_rounded, decoded_floats_rounded,
                "The decoded values are not equal to the original values"
            );
        }
    }
}

fn round_trip_assert(
    values: &[f64],
    compressor_config: CompressorConfig,
    chunk_option: ChunkOption,
) {
    // encode
    let encoded = {
        let encoder_input = EncoderInput::from_f64_slice(values);

        let encoder_output = EncoderOutput::from_vec(Vec::new());

        let mut encoder = Encoder::new(
            encoder_input,
            encoder_output,
            chunk_option,
            compressor_config,
        );

        encoder.encode().unwrap();

        encoder.into_output().into_vec()
    };

    // decode
    let decoded_filter_materialize = {
        let input_cursor = Cursor::new(&encoded[..]);
        let decoder_input = DecoderInput::from_reader(input_cursor);

        let mut decoder_output = DecoderOutput::from_vec(Vec::new());

        let mut decoder = Decoder::new(decoder_input).unwrap();

        decoder
            .filter_materialize(
                &mut decoder_output,
                Predicate::Range(Range::new(
                    RangeValue::Inclusive(f64::MIN),
                    RangeValue::Inclusive(f64::MAX),
                )),
                None,
                None,
            )
            .unwrap();

        decoder_output.into_writer().into_inner()
    };

    let decoded_materialize = {
        let input_cursor = Cursor::new(&encoded[..]);
        let decoder_input = DecoderInput::from_reader(input_cursor);

        let mut decoder_output = DecoderOutput::from_vec(Vec::new());

        let mut decoder = Decoder::new(decoder_input).unwrap();

        let bm0 = decoder.filter(Predicate::Ne(0.5), None, None).unwrap();
        let bm1 = decoder.filter(Predicate::Eq(0.5), None, None).unwrap();
        decoder
            .materialize(&mut decoder_output, Some(&(bm0 | bm1)), None)
            .unwrap();

        decoder_output.into_writer().into_inner()
    };

    for decoded in [decoded_filter_materialize, decoded_materialize] {
        let decoded_floats: Vec<f64> = decoded
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
                values.len(),
                decoded_floats.len(),
                "The length of the decoded values is not equal to the length of the original values. {} != {}", values.len(), decoded_floats.len()
            );

        assert_eq!(
            values, decoded_floats,
            "The decoded values are not equal to the original values"
        );
    }
}

fn test_round_trip(
    compressor_config: CompressorConfig,
    chunk_option: impl Fn(usize) -> ChunkOption,
) {
    for n in [103, 1003, 10003, 100004, 100005] {
        #[cfg(miri)] // miri is too slow
        if n > 103 {
            continue;
        }

        let random_values = generate_and_write_random_f64(n);

        round_trip_assert(&random_values, compressor_config.clone(), chunk_option(n));
    }

    const XOR_VALUES: [f64; 66] = const {
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
    round_trip_assert(
        &XOR_VALUES,
        compressor_config.clone(),
        ChunkOption::RecordCount(XOR_VALUES.len()),
    );

    let xor_values2 = {
        let mut xor_values2 = [2.3f64, 3.3f64, 0.0f64];
        xor_values2[2] = xor_values2[1];
        for i in 0..2 {
            xor_values2[2] = f64::from_bits(xor_values2[2].to_bits() ^ (1 << i));
        }

        xor_values2
    };
    round_trip_assert(&xor_values2, compressor_config, ChunkOption::RecordCount(3));
}

#[test]
fn test_api_round_trip_buff() {
    for scale in [1, 10, 100, 1000] {
        println!("scale: {}", scale);
        let compressor_config = CompressorConfig::buff().scale(scale).build();
        test_round_trip_with_scale(
            generate_and_write_random_f64_with_precision,
            compressor_config.into(),
            scale,
            record_count_chunk_option_by_n,
        );
    }

    let compressor_config = CompressorConfig::buff().scale(100).build();

    println!("single values");
    // sigle value
    test_round_trip_with_scale(
        |n, _scale| vec![100.19; n],
        compressor_config.into(),
        100,
        record_count_chunk_option_by_n,
    );
}

#[test]
fn test_api_round_trip_buff_bytesize() {
    for scale in [1, 10, 100, 1000] {
        println!("scale: {}", scale);
        let compressor_config = CompressorConfig::buff().scale(scale).build();
        test_round_trip_with_scale(
            generate_and_write_random_f64_with_precision,
            compressor_config.into(),
            scale,
            bytesize_chunk_option_by_n,
        );
    }

    let compressor_config = CompressorConfig::buff().scale(100).build();

    println!("single values");
    // sigle value
    test_round_trip_with_scale(
        |n, _scale| vec![100.19; n],
        compressor_config.into(),
        100,
        bytesize_chunk_option_by_n,
    );
}
