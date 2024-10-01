use ebi::{compressor::CompressorConfig, encoder::ChunkOption};

mod helper {
    use std::{fmt::Display, io::Cursor};

    use ebi::{
        api::{
            decoder::{Decoder, DecoderInput},
            encoder::{Encoder, EncoderInput, EncoderOutput},
        },
        compressor::CompressorConfig,
        encoder::ChunkOption,
    };

    pub fn test_knn1(
        config: CompressorConfig,
        data: &[f64],
        chunk_option: ChunkOption,
        target: &[f64],
        expected_index: usize,
        test_name: impl Display,
    ) {
        println!("=========== {} ===========", test_name);
        // Dimention check
        assert!(
            data.len() % target.len() == 0,
            "Data and target dimention mismatch"
        );

        // Expected validity check
        assert!(
            expected_index < data.len() / target.len(),
            "Expected index out of bound"
        );

        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(data);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let nearest = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();

            decoder
                .knn1(target)
                .unwrap_or_else(|_| panic!("{}: knn1 failed", test_name))
        };

        assert_eq!(
            nearest.index, expected_index,
            "{}: Expected index mismatch, expected: {}, got: {:?}",
            test_name, expected_index, nearest
        );
    }
}

pub fn test_knn1_manual(config: impl Into<CompressorConfig>) {
    let config = config.into();

    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Test 1: Simple 1-NN
    for i in 0..3 {
        helper::test_knn1(
            config,
            &data,
            ChunkOption::RecordCount(2),
            &data[i * 2..(i + 1) * 2],
            i,
            format!("Test 1: Simple 1-NN ({i})"),
        );
    }
    // Test 2: Simple 1-NN, but vector is laid over multiple chunks (0)
    for i in 0..3 {
        helper::test_knn1(
            config,
            &data,
            ChunkOption::RecordCount(3),
            &data[i * 2..(i + 1) * 2],
            i,
            format!("Test 2: Simple 1-NN, but vector is laid over multiple chunks ({i})"),
        );
    }
}

macro_rules! declare_ml_query_tests {
    ($($method:ident,)*) => {
        $(
            mod $method {
                #[test]
                fn test_knn1() {
                    let config = super::CompressorConfig::$method().build();
                    super::test_knn1_manual(config);
                }
            }
        )*
    };
}

declare_ml_query_tests!(uncompressed,);
#[cfg(not(miri))]
declare_ml_query_tests!(
    rle,
    gorilla,
    chimp,
    chimp128,
    elf_on_chimp,
    elf,
    zstd,
    gzip,
    snappy,
);
