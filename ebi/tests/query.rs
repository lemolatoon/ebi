use ebi::compressor::CompressorConfig;

#[cfg(test)]
mod helper {
    use ebi::api::decoder::ChunkId;
    use ebi::compressor::CompressorConfig;
    use ebi::decoder::query::{Range, RangeValue};
    use std::io::Cursor;

    use ebi::{
        api::{
            decoder::{Decoder, DecoderInput, DecoderOutput},
            encoder::{Encoder, EncoderInput, EncoderOutput},
        },
        decoder::query::Predicate,
        encoder::ChunkOption,
    };
    use roaring::RoaringBitmap;

    #[allow(clippy::too_many_arguments)]
    fn test_query_filter_inner(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        expected: RoaringBitmap,
        test_name: String,
    ) {
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let bitmap = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();
            decoder.filter(predicate, bitmask, chunk_id).unwrap()
        };

        assert_eq!(bitmap, expected, "[{test_name}]: Filter result mismatch");
    }

    /// Test filter_materialize
    /// `SCALE` is the scale to round the input, output.
    #[allow(clippy::too_many_arguments)]
    fn test_query_filter_materialize_inner(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        expected: RoaringBitmap,
        test_name: String,
        round_scale: Option<usize>,
    ) {
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let materialized: Vec<f64> = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();
            let mut output = DecoderOutput::from_vec(Vec::new());
            decoder
                .filter_materialize(&mut output, predicate, bitmask, chunk_id)
                .unwrap();

            output
                .into_writer()
                .into_inner()
                .chunks_exact(8)
                .map(|chunk| {
                    let fp = f64::from_ne_bytes(chunk.try_into().unwrap());
                    if let Some(scale) = round_scale {
                        (fp * scale as f64).round() / scale as f64
                    } else {
                        fp
                    }
                })
                .collect()
        };

        let expected: Vec<f64> = values
            .into_iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if expected.contains(i as u32) {
                    Some(if let Some(scale) = round_scale {
                        (f * scale as f64).round() / scale as f64
                    } else {
                        f
                    })
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            materialized, expected,
            "[{test_name}]: Filter materialize result mismatch"
        );
    }

    fn test_query_filter_optionally_materialize(
        config: impl Into<CompressorConfig>,
        test_name: impl std::fmt::Display,
        is_filter_materialize: bool,
        round_scale: Option<usize>,
    ) {
        let config = config.into();
        let t_name = |n: String| format!("{test_name}: {n}");

        let query_name = if is_filter_materialize {
            "Filter Materialize"
        } else {
            "Filter"
        };

        type TestFn = Box<
            dyn Fn(
                CompressorConfig,
                Vec<f64>,
                ChunkOption,
                Predicate,
                Option<&RoaringBitmap>,
                Option<ChunkId>,
                RoaringBitmap,
                String,
            ),
        >;
        let test_fn: TestFn = if is_filter_materialize {
            Box::new(
                move |config, values, option, pred, bm, id, expected, name| {
                    test_query_filter_materialize_inner(
                        config,
                        values,
                        option,
                        pred,
                        bm,
                        id,
                        expected,
                        name,
                        round_scale,
                    )
                },
            )
        } else {
            Box::new(test_query_filter_inner)
        };

        // Range filter
        {
            // Test 1: Range filter (Inclusive, Inclusive)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Inclusive(3.8),
                RangeValue::Inclusive(9.9),
            ));
            let expected = RoaringBitmap::from_iter(vec![1, 2, 3, 5]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Inclusive, Inclusive)")),
            );

            // Test 2: Range filter (Exclusive, Inclusive)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Exclusive(3.8),
                RangeValue::Inclusive(9.9),
            ));
            let expected = RoaringBitmap::from_iter(vec![1, 3, 5]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Exclusive, Inclusive)")),
            );

            // Test 3: Range filter (Inclusive, Exclusive)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Inclusive(3.8),
                RangeValue::Exclusive(9.9),
            ));
            let expected = RoaringBitmap::from_iter(vec![1, 2, 3]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Inclusive, Exclusive)")),
            );

            // Test 4: Range filter (Exclusive, Exclusive)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Exclusive(3.8),
                RangeValue::Exclusive(9.9),
            ));
            let expected = RoaringBitmap::from_iter(vec![1, 3]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Exclusive, Exclusive)")),
            );

            // Test 5: Range filter (Inclusive, None)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate =
                Predicate::Range(Range::new(RangeValue::Inclusive(3.8), RangeValue::None));
            let expected = RoaringBitmap::from_iter(vec![1, 2, 3, 5, 6]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Inclusive, None)")),
            );

            // Test 6: Range filter (None, Exclusive)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate =
                Predicate::Range(Range::new(RangeValue::None, RangeValue::Exclusive(9.9)));
            let expected = RoaringBitmap::from_iter(vec![0, 1, 2, 3, 4]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (None, Exclusive)")),
            );
        }

        // Other predicates filter
        {
            // Test 1: Eq filter
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 3.8];
            let predicate = Predicate::Eq(3.8);
            let expected = RoaringBitmap::from_iter(vec![2, 7]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Eq {query_name}")),
            );

            // Test 2: Ne filter
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 3.8];
            let predicate = Predicate::Ne(3.8);
            let expected = RoaringBitmap::from_iter(vec![0, 1, 3, 4, 5, 6]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Ne {query_name}")),
            );
        }

        // Filter Arguments
        {
            // Test 1: Filter with bitmask
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 3.8];
            let predicate =
                Predicate::Range(Range::new(RangeValue::Exclusive(3.8), RangeValue::None));
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 4, 6, 7]);
            let expected = RoaringBitmap::from_iter(vec![3, 6]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                predicate,
                Some(&bitmask),
                None,
                expected,
                t_name(format!("{query_name} with bitmask")),
            );

            // Test 2: Filter with chunk_id
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 3.8];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Inclusive(3.8),
                RangeValue::Exclusive(9.9),
            ));
            let chunk_option = ChunkOption::RecordCount(3);
            let chunk_id = ChunkId::new(2);
            let expected = RoaringBitmap::from_iter(vec![7]);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                Some(chunk_id),
                expected,
                t_name(format!("{query_name} with chunk_id")),
            );

            // Test 3: Filter with bitmask and chunk_id
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 3.8];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Exclusive(3.8),
                RangeValue::Inclusive(9.9),
            ));
            let chunk_option = ChunkOption::RecordCount(3);
            let chunk_id = ChunkId::new(1);
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 5, 7]);
            let expected = RoaringBitmap::from_iter(vec![3, 5]);
            test_fn(
                config,
                values,
                chunk_option,
                predicate,
                Some(&bitmask),
                Some(chunk_id),
                expected,
                t_name(format!("{query_name} with bitmask and chunk_id")),
            );
        }
    }

    pub(crate) fn test_query_filter(
        config: impl Into<CompressorConfig>,
        test_name: impl std::fmt::Display,
    ) {
        test_query_filter_optionally_materialize(config, test_name, false, None);
    }

    pub(crate) fn test_query_filter_materialize(
        config: impl Into<CompressorConfig>,
        round_scale: Option<usize>,
        test_name: impl std::fmt::Display,
    ) {
        test_query_filter_optionally_materialize(config, test_name, true, round_scale);
    }

    #[allow(clippy::too_many_arguments)]
    fn test_query_materialize_inner(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        round_scale: Option<usize>,
        expected: Vec<f64>,
        test_name: impl std::fmt::Display,
    ) {
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let decoded: Vec<f64> = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();
            let mut output = DecoderOutput::from_vec(Vec::new());
            decoder.materialize(&mut output, bitmask, chunk_id).unwrap();

            let decoded_bytes = output.into_writer().into_inner();

            decoded_bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_ne_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        let (expected, decoded) = if let Some(scale) = round_scale {
            let round = |x: f64| (x * scale as f64).round() / scale as f64;

            let rounded_expected = expected.into_iter().map(round).collect::<Vec<f64>>();
            let rounded_decoded = decoded.into_iter().map(round).collect::<Vec<f64>>();

            (rounded_expected, rounded_decoded)
        } else {
            (expected, decoded)
        };

        assert_eq!(
            expected, decoded,
            "[{test_name}]: Materialize result mismatch"
        );
    }

    pub(crate) fn test_query_materialize(
        config: impl Into<CompressorConfig>,
        round_scale: Option<usize>,
        test_name: impl std::fmt::Display,
    ) {
        let config = config.into();
        let t_name = |n: &str| format!("{test_name}: {n}");
        // Test 1: Materialize without bitmask and chunk_id
        {
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let expected = values.clone();
            test_query_materialize_inner(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                None,
                None,
                round_scale,
                expected,
                t_name("Materialize without bitmask and chunk_id"),
            );
        }

        // Test 2: Materialize with bitmask
        {
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 5]);
            let expected = vec![3.3, 3.8, 6.7, 9.9];
            test_query_materialize_inner(
                config.clone(),
                values,
                ChunkOption::RecordCount(2),
                Some(&bitmask),
                None,
                round_scale,
                expected,
                t_name("Materialize with bitmask"),
            );
        }

        // Test 3: Materialize with chunk_id
        {
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 20.3];
            let chunk_option = ChunkOption::RecordCount(3);
            let chunk_id = ChunkId::new(2);
            let expected = vec![10.1, 20.3];
            test_query_materialize_inner(
                config.clone(),
                values,
                chunk_option,
                None,
                Some(chunk_id),
                round_scale,
                expected,
                t_name("Materialize with chunk_id"),
            );
        }

        // Test 4: Materialize with bitmask and chunk_id
        {
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let chunk_option = ChunkOption::RecordCount(3);
            let chunk_id = ChunkId::new(1);
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 5]);
            let expected = vec![6.7, 9.9];
            test_query_materialize_inner(
                config,
                values,
                chunk_option,
                Some(&bitmask),
                Some(chunk_id),
                round_scale,
                expected,
                t_name("Materialize with bitmask and chunk_id"),
            );
        }
    }
}

#[test]
fn test_uncompressed_filter() {
    let config = CompressorConfig::uncompressed().build();
    helper::test_query_filter(config, "Uncompressed");
}

#[test]
fn test_rle_filter() {
    let config = CompressorConfig::rle().build();
    helper::test_query_filter(config, "RLE");
}

#[test]
fn test_gorilla_filter() {
    let config = CompressorConfig::gorilla().build();
    helper::test_query_filter(config, "Gorilla");
}

#[test]
fn test_buff_filter() {
    let config = CompressorConfig::buff().scale(100).build();
    helper::test_query_filter(config, "BUFF");
}

#[test]
fn test_uncompressed_materialize() {
    let config = CompressorConfig::uncompressed().build();
    helper::test_query_materialize(config, None, "Uncompressed");
}

#[test]
fn test_rle_materialize() {
    let config = CompressorConfig::rle().build();
    helper::test_query_materialize(config, None, "RLE");
}

#[test]
fn test_gorilla_materialize() {
    let config = CompressorConfig::gorilla().build();
    helper::test_query_materialize(config, None, "Gorilla");
}

#[test]
fn test_buff_materialize() {
    let config = CompressorConfig::buff().scale(10).build();
    helper::test_query_materialize(config, Some(10), "BUFF");
}

#[test]
fn test_uncompressed_filter_materialize() {
    let config = CompressorConfig::uncompressed().build();
    helper::test_query_filter_materialize(config, None, "Uncompressed");
}

#[test]
fn test_rle_filter_materialize() {
    let config = CompressorConfig::rle().build();
    helper::test_query_filter_materialize(config, None, "RLE");
}

#[test]
fn test_gorilla_filter_materialize() {
    let config = CompressorConfig::gorilla().build();
    helper::test_query_filter_materialize(config, None, "Gorilla");
}

#[test]
fn test_buff_filter_materialize() {
    let config = CompressorConfig::buff().scale(10).build();
    helper::test_query_filter_materialize(config, Some(10), "BUFF");
}
