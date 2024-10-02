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

    pub fn test_matmul(
        config: CompressorConfig,
        // (Batch, H, W)
        data: &[&[&[f64]]],
        chunk_option: ChunkOption,
        // (W, H)
        target: &[&[f64]],
        // (Batch, H, W)
        expected: &[&[&[f64]]],
        test_name: impl Display,
    ) {
        println!("=========== {} ===========", test_name);
        let data_batch = data.len();
        let data_h = data[0].len();
        let data_w = data[0][0].len();

        let data = data
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|row| row.iter().copied()))
            .collect::<Box<[f64]>>();

        let target_w = target.len();
        let target_h = target[0].len();

        let target = target
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Box<[f64]>>();

        let expected_batch = expected.len();
        let expected_h = expected[0].len();
        let expected_w = expected[0][0].len();

        let expected = expected
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|row| row.iter().copied()))
            .collect::<Box<[f64]>>();
        // Dimention check
        // Check multipliable
        assert_eq!(
            data_w, target_h,
            "Data height and target width dimention mismatch: {} != {}",
            data_w, target_h
        );

        // Check expected dimention
        assert_eq!(
            data_batch, expected_batch,
            "Data batch dimention mismatch: {} != {}",
            data_batch, expected_batch
        );

        assert_eq!(
            data_h, expected_h,
            "Data height dimention mismatch: {} != {}",
            data_h, expected_h
        );

        assert_eq!(
            target_w, expected_w,
            "Data width dimention mismatch: {} != {}",
            target_w, expected_w
        );

        // chunk_option check
        assert!(
            matches!(chunk_option, ChunkOption::RecordCount(_)),
            "ChunkOption must be RecordCount"
        );

        let chunk_option_record_count = match chunk_option {
            ChunkOption::RecordCount(record_count) => record_count,
            _ => unreachable!(),
        };

        assert!(
            chunk_option_record_count % (data_h * data_w) == 0,
            "ChunkOption record count must be multiple of data size: {} % {} != 0",
            chunk_option_record_count,
            data_h * data_w
        );

        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&data);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let result = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();

            decoder
                .matmul(&target, (target_h, target_w), (data_h, data_w))
                .unwrap_or_else(|_| panic!("{}: matmul failed", test_name))
        };

        assert_eq!(
            result, expected,
            "{}: Expected result mismatch, expected: {:?}, got: {:?}",
            test_name, expected, result
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

pub fn test_matmul_manual(config: impl Into<CompressorConfig>) {
    let config = config.into();

    let data: &[&[&[f64]]] = &[
        &[&[1.0, 2.0], &[3.0, 4.0], &[0.0, 1.0]],
        &[&[3.0, 2.0], &[-1.0, 8.0], &[0.0, 1.0]],
    ];

    // column first layout
    let target: &[&[f64]] = &[&[1.0, 1.0], &[5.0, 4.0], &[6.0, 3.0], &[2.0, 1.0]];

    let expected: &[&[&[f64]]] = &[
        &[
            &[3.0, 13.0, 12.0, 4.0],
            &[7.0, 31.0, 30.0, 10.0],
            &[1.0, 4.0, 3.0, 1.0],
        ],
        &[
            &[5.0, 23.0, 24.0, 8.0],
            &[7.0, 27.0, 18.0, 6.0],
            &[1.0, 4.0, 3.0, 1.0],
        ],
    ];

    helper::test_matmul(
        config,
        data,
        ChunkOption::RecordCount(6),
        target,
        expected,
        "Test 1: Simple matmul",
    );

    let data: &[&[&[f64]]] = &[
        &[&[1.0, 2.0], &[3.0, 4.0], &[0.0, 1.0]],
        &[&[3.0, 2.0], &[-1.0, 8.0], &[0.0, 1.0]],
        &[&[2.0, 1.0], &[0.0, 3.0], &[1.0, 0.0]],
        &[&[4.0, -1.0], &[2.0, 2.0], &[0.0, 0.5]],
    ];

    // Column-first layout
    let target: &[&[f64]] = &[&[1.0, 1.0], &[5.0, 4.0], &[6.0, 3.0], &[2.0, 1.0]];

    let expected: &[&[&[f64]]] = &[
        &[
            &[3.0, 13.0, 12.0, 4.0],
            &[7.0, 31.0, 30.0, 10.0],
            &[1.0, 4.0, 3.0, 1.0],
        ],
        &[
            &[5.0, 23.0, 24.0, 8.0],
            &[7.0, 27.0, 18.0, 6.0],
            &[1.0, 4.0, 3.0, 1.0],
        ],
        &[
            &[3.0, 14.0, 15.0, 5.0],
            &[3.0, 12.0, 9.0, 3.0],
            &[1.0, 5.0, 6.0, 2.0],
        ],
        &[
            &[3.0, 16.0, 21.0, 7.0],
            &[4.0, 18.0, 18.0, 6.0],
            &[0.5, 2.0, 1.5, 0.5],
        ],
    ];

    helper::test_matmul(
        config,
        data,
        ChunkOption::RecordCount(6),
        target,
        expected,
        "Test 2: Updated matmul with four matrices",
    );

    let data: &[&[&[f64]]] = &[
        &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[0.0, 1.0, 2.0],
            &[3.0, 4.0, 5.0],
        ],
        &[
            &[2.0, 1.0, 0.0],
            &[3.0, 2.0, 1.0],
            &[4.0, 3.0, 2.0],
            &[5.0, 4.0, 3.0],
            &[6.0, 5.0, 4.0],
        ],
        &[
            &[0.0, 1.0, 2.0],
            &[1.0, 0.0, 3.0],
            &[2.0, 1.0, 4.0],
            &[3.0, 2.0, 5.0],
            &[4.0, 3.0, 6.0],
        ],
        &[
            &[3.0, 2.0, 1.0],
            &[4.0, 3.0, 2.0],
            &[5.0, 4.0, 3.0],
            &[6.0, 5.0, 4.0],
            &[7.0, 6.0, 5.0],
        ],
    ];

    // Transposed target matrix (9x3)
    let target: &[&[f64]] = &[
        &[1.0, 2.0, 3.0],
        &[2.0, 4.0, 6.0],
        &[3.0, 6.0, 9.0],
        &[4.0, 8.0, 12.0],
        &[5.0, 10.0, 15.0],
        &[6.0, 12.0, 18.0],
        &[7.0, 14.0, 21.0],
        &[8.0, 16.0, 24.0],
        &[9.0, 18.0, 27.0],
    ];

    let expected: &[&[&[f64]]] = &[
        &[
            &[14.0, 28.0, 42.0, 56.0, 70.0, 84.0, 98.0, 112.0, 126.0],
            &[32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0],
            &[50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0],
            &[8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0],
            &[26.0, 52.0, 78.0, 104.0, 130.0, 156.0, 182.0, 208.0, 234.0],
        ],
        &[
            &[4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            &[16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0, 144.0],
            &[22.0, 44.0, 66.0, 88.0, 110.0, 132.0, 154.0, 176.0, 198.0],
            &[28.0, 56.0, 84.0, 112.0, 140.0, 168.0, 196.0, 224.0, 252.0],
        ],
        &[
            &[8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            &[16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0, 144.0],
            &[22.0, 44.0, 66.0, 88.0, 110.0, 132.0, 154.0, 176.0, 198.0],
            &[28.0, 56.0, 84.0, 112.0, 140.0, 168.0, 196.0, 224.0, 252.0],
        ],
        &[
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            &[16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0, 144.0],
            &[22.0, 44.0, 66.0, 88.0, 110.0, 132.0, 154.0, 176.0, 198.0],
            &[28.0, 56.0, 84.0, 112.0, 140.0, 168.0, 196.0, 224.0, 252.0],
            &[34.0, 68.0, 102.0, 136.0, 170.0, 204.0, 238.0, 272.0, 306.0],
        ],
    ];

    helper::test_matmul(
        config,
        data,
        ChunkOption::RecordCount(30),
        target,
        expected,
        "Test 3: Updated matmul with 5x3 matrices and target 3x9",
    );
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

                #[test]
                fn test_matmul() {
                    let config = super::CompressorConfig::$method().build();
                    super::test_matmul_manual(config);
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
