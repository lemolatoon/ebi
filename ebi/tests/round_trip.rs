use std::io::Cursor;

use ebi::{
    api::{
        decoder::{Decoder, DecoderInput, DecoderOutput},
        encoder::{Encoder, EncoderInput, EncoderOutput},
    },
    compressor::CompressorConfig,
    decoder::query::Predicate,
    encoder::ChunkOption,
};
use rand::Rng;

fn generate_and_write_random_f64(n: usize) -> Vec<f64> {
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
        let decoded = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);

            let mut decoder_output = DecoderOutput::from_vec(Vec::new());

            let mut decoder = Decoder::new(decoder_input).unwrap();

            let bm0 = decoder.filter(Predicate::Gt(0.5), None, None).unwrap();
            let bm1 = decoder.filter(Predicate::Le(0.5), None, None).unwrap();
            let bm2 = decoder.filter(Predicate::Eq(0.5), None, None).unwrap();
            decoder
                .scan(&mut decoder_output, Some(&(bm0 | bm1 | bm2)), None)
                .unwrap();

            decoder_output.into_writer().into_inner()
        };

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
