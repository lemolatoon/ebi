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

#[test]
fn test_api_round_trip_uncompressed() {
    let header = b"my_header".to_vec().into_boxed_slice();
    let compressor_config = CompressorConfig::uncompressed()
        .capacity(8000)
        .header(header)
        .build();
    test_round_trip(compressor_config.into());
}

#[test]
fn test_api_round_trip_gorilla() {
    let compressor_config = CompressorConfig::gorilla().capacity(8000).build();
    test_round_trip(compressor_config.into());
}

#[test]
fn test_api_round_trip_rle() {
    let compressor_config = CompressorConfig::rle().build();
    test_round_trip(compressor_config.into());
}

#[test]
fn test_api_round_trip_buff() {
    for scale in [1, 10, 100, 1000] {
        let compressor_config = CompressorConfig::buff().scale(scale).build();
        test_round_trip_with_scale(
            generate_and_write_random_f64_with_precision,
            compressor_config.into(),
            scale,
        );
    }

    let compressor_config = CompressorConfig::buff().scale(100).build();

    // sigle value
    test_round_trip_with_scale(|n, _scale| vec![100.19; n], compressor_config.into(), 100);
}

fn test_round_trip_with_scale(
    generator: fn(usize, usize) -> Vec<f64>,
    compressor_config: CompressorConfig,
    scale: usize,
) {
    for n in [1003, 10003, 100004, 100005] {
        #[cfg(miri)] // miri is too slow
        if n > 1003 {
            continue;
        }

        let random_values = generator(n, scale);
        // encode
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&random_values);

            let encoder_output = EncoderOutput::from_vec(Vec::new());

            let chunk_option = ChunkOption::RecordCount(1024 * 8);

            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                chunk_option,
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
            let random_values_bytes = random_values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>();
            assert_eq!(
                random_values_bytes.len(),
                decoded.len(),
                "The length of the decoded values is not equal to the length of the original values. {} != {}", random_values_bytes.len(), decoded.len()
            );

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
                random_floats_rounded, decoded_floats_rounded,
                "The decoded values are not equal to the original values"
            );
        }
    }
}

fn test_round_trip(compressor_config: CompressorConfig) {
    for n in [1003, 10003, 100004, 100005] {
        #[cfg(miri)] // miri is too slow
        if n > 1003 {
            continue;
        }

        let random_values = generate_and_write_random_f64(n);
        // encode
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&random_values);

            let encoder_output = EncoderOutput::from_vec(Vec::new());

            let chunk_option = ChunkOption::RecordCount(1024 * 8);

            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                chunk_option,
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
            let random_values_bytes = random_values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>();
            assert_eq!(
                random_values_bytes.len(),
                decoded.len(),
                "The length of the decoded values is not equal to the length of the original values. {} != {}", random_values_bytes.len(), decoded.len()
            );

            assert_eq!(
                random_values_bytes, decoded,
                "The decoded values are not equal to the original values"
            );
        }
    }
}
