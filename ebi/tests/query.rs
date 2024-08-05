use ebi::compressor::CompressorConfig;

#[cfg(test)]
mod helper {
    use ebi::api::decoder::ChunkId;
    use ebi::compressor::CompressorConfig;
    use ebi::decoder::query::{Range, RangeValue};
    use rand::prelude::Distribution;
    use rand::rngs::StdRng;
    #[cfg(not(miri))]
    use rand::seq::SliceRandom as _;
    use rand::{Rng, SeedableRng};
    use std::cmp::min;
    use std::io::Cursor;
    use std::iter;

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
        println!("=========== {} ===========", test_name);
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

        // println!("bitmap: {:?}", bitmap.iter().collect::<Vec<_>>());
        // println!("expected: {:?}", expected.iter().collect::<Vec<_>>());
        assert_eq!(bitmap, expected, "[{test_name}]: Filter result mismatch");
    }

    fn test_query_filter_inner_with_expected_calculation(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        test_name: String,
    ) {
        let encoded = {
            let uncompressed_config = CompressorConfig::uncompressed().build();
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                chunk_option,
                uncompressed_config,
            );
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let expected = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();
            decoder.filter(predicate, bitmask, chunk_id).unwrap()
        };

        test_query_filter_inner(
            config,
            values,
            chunk_option,
            predicate,
            bitmask,
            chunk_id,
            expected,
            test_name,
        );
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
                    print!("{} ", fp);
                    if let Some(scale) = round_scale {
                        (fp * scale as f64).round() / scale as f64
                    } else {
                        fp
                    }
                })
                .collect()
        };
        println!();

        let expected: Vec<f64> = values
            .into_iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if expected.contains(i as u32) {
                    print!("{} ", f);
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
        println!();

        assert_eq!(
            materialized.len(),
            expected.len(),
            "[{test_name}]: Filter materialize result length mismatch"
        );

        assert_eq!(
            materialized, expected,
            "[{test_name}]: Filter materialize result mismatch"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn test_query_filter_materialize_inner_with_expected_calculation(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        test_name: String,
        round_scale: Option<usize>,
    ) {
        let encoded = {
            let uncompressed_config = CompressorConfig::uncompressed().build();
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                chunk_option,
                uncompressed_config,
            );
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let expected = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();
            decoder.filter(predicate, bitmask, chunk_id).unwrap()
        };

        test_query_filter_materialize_inner(
            config,
            values,
            chunk_option,
            predicate,
            bitmask,
            chunk_id,
            expected,
            test_name,
            round_scale,
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
            let values = vec![-300.3, -5.6, -3.8, -699.7, 2.1, 9.9, 100.1];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Inclusive(-3.8),
                RangeValue::Inclusive(9.9),
            ));
            let expected = RoaringBitmap::from_iter(vec![2, 4, 5]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
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
                ChunkOption::RecordCount(5),
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
                ChunkOption::RecordCount(5),
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
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Exclusive, Exclusive)")),
            );

            // Test 5: Range filter (Exclusive, None)
            let values = vec![3.3, 52.6, 3.8, 6.7, 2.1, 9.9, 100.1];
            let predicate =
                Predicate::Range(Range::new(RangeValue::Exclusive(3.8), RangeValue::None));
            let expected = RoaringBitmap::from_iter(vec![1, 3, 5, 6]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Exclusive, None)")),
            );

            // Test 6: Range filter (Inclusive, None)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate =
                Predicate::Range(Range::new(RangeValue::Inclusive(3.8), RangeValue::None));
            let expected = RoaringBitmap::from_iter(vec![1, 2, 3, 5, 6]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (Inclusive, None)")),
            );

            // Test 7: Range filter (None, Exclusive)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate =
                Predicate::Range(Range::new(RangeValue::None, RangeValue::Exclusive(9.9)));
            let expected = RoaringBitmap::from_iter(vec![0, 1, 2, 3, 4]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} (None, Exclusive)")),
            );

            // Test 7: Range filter with bitmask (None, None)
            let values = vec![3.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let predicate = Predicate::Range(Range::new(RangeValue::None, RangeValue::None));
            let bitmask = RoaringBitmap::from_iter(vec![0, 1, 3, 4, 6]);
            let expected = bitmask.clone();
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                Some(&bitmask),
                None,
                expected,
                t_name(format!("Range {query_name} (None, None) with bitmask")),
            );

            // Test 8: Range filter large range (Inclusive, Inclusive)
            // let values = vec![3.3, 5.6, -333.8, 6.7, 2.1, 9.9, 10.1];
            let values = vec![9.9, 10.1];
            let predicate = Predicate::Range(Range::new(
                RangeValue::Inclusive(-1000.0),
                RangeValue::Inclusive(10000000.0),
            ));
            let expected = RoaringBitmap::from_iter(0..values.len() as u32);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!(
                    "Range {query_name} large range (Inclusive, Inclusive)"
                )),
            );

            // Test 9: Range filter simd (Inclusive, None)
            #[rustfmt::skip]
            let values = vec![/*  0 */1.0, 1.0, 1.0, 1.0, 1.0,
                                        /*  5 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 10 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 15 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 20 */10.5, 10.6, 10.7, 10.9, 10.9,
                                        /* 25 */100.0, 100.0, 100.0, 100.0, 100.0,
                                        /* 30 */10.0, 10.0, 10.0, 10.0, 10.0];
            let predicate =
                Predicate::Range(Range::new(RangeValue::Inclusive(10.7), RangeValue::None));
            let expected = RoaringBitmap::from_iter(22..30);
            let chunk_option = ChunkOption::RecordCount(35);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} simd (Inclusive, None)")),
            );

            // Test 10: Range filter simd (None, Exclusive)
            #[rustfmt::skip]
            let values = vec![/*  0 */1.0, 1.0, 1.0, 1.0, 1.0,
                                        /*  5 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 10 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 15 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 20 */10.5, 10.6, 10.7, 10.9, 10.9,
                                        /* 25 */100.0, 100.0, 100.0, 100.0, 100.0,
                                        /* 30 */10.0, 10.0, 10.0, 10.0, 10.0];
            let predicate =
                Predicate::Range(Range::new(RangeValue::None, RangeValue::Exclusive(10.7)));
            let expected = RoaringBitmap::from_iter((0..=21).chain(30..35));
            let chunk_option = ChunkOption::RecordCount(35);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                None,
                expected,
                t_name(format!("Range {query_name} simd (None, Exclusive)")),
            );
        }

        // Other predicates filter
        {
            // Test 1: Eq filter
            let values = vec![322.3, 5204029348.6, 3.8, -6.7, 2.1, 99.9, 1000220.1, 3.8];
            let predicate = Predicate::Eq(3.8);
            let expected = RoaringBitmap::from_iter(vec![2, 7]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Eq {query_name}")),
            );

            // Test 2: Ne filter
            let values = vec![3.7, 522.6, 3.8, 6099.7, -22390.1, 908.9, -10.1, 3.8];
            let predicate = Predicate::Ne(3.8);
            let expected = RoaringBitmap::from_iter(vec![0, 1, 3, 4, 5, 6]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                None,
                None,
                expected,
                t_name(format!("Ne {query_name}")),
            );

            // Test 3: Eq simd
            #[rustfmt::skip]
            let values = vec![/*  0 */1.0, 1.0, 1.0, 1.0, 1.0,
                                        /*  5 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 10 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 15 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 20 */10.5, 10.6, 10.7, 10.9, 10.9,
                                        /* 25 */100.0, 100.0, 100.0, 100.0, 100.0,
                                        /* 30 */10.0, 10.0, 10.0, 10.0, 10.0];
            let predicate = Predicate::Eq(10.5);
            let expected = RoaringBitmap::from_iter(iter::once(20));
            let chunk_option = ChunkOption::RecordCount(35);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                None,
                expected,
                t_name(format!("{query_name} Eq simd")),
            );

            // Test 3: Ne simd
            #[rustfmt::skip]
            let values = vec![/*  0 */1.0, 1.0, 1.0, 1.0, 1.0,
                                        /*  5 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 10 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 15 */2.0, 2.0, 2.0, 2.0, 2.0,
                                        /* 20 */10.5, 10.6, 10.7, 10.9, 10.9,
                                        /* 25 */100.0, 100.0, 100.0, 100.0, 100.0,
                                        /* 30 */10.0, 10.0, 10.0, 10.0, 10.0];
            let predicate = Predicate::Ne(10.5);
            let expected = RoaringBitmap::from_iter((0..20).chain(21..35));
            let chunk_option = ChunkOption::RecordCount(35);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                None,
                expected,
                t_name(format!("{query_name} Ne simd")),
            );

            // Test 4: Random Failure Ne simd
            let values: Vec<f64> = vec![-7072.1; 32]
                .into_iter()
                .chain(iter::once(-7071.2))
                .collect();
            let predicate = Predicate::Ne(-7071.2);
            let expected = RoaringBitmap::from_iter(0..32);
            let chunk_option = ChunkOption::RecordCount(35);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                None,
                expected,
                t_name(format!("{query_name} Random failure Ne simd")),
            );

            // Test 4: Random Failure Range simd (low selectivity, more than 2 subcolumns check will be run and more than 2 chunks)
            let values = vec![41.4; 15].into_iter().chain(vec![41.5; 15]);
            // let values = vec![/*0.4*/0.0; 15].into_iter().chain(vec![0.5; 15]);
            let values: Vec<f64> = values.clone().chain(values).collect();
            let predicate =
                Predicate::Range(Range::new(RangeValue::Inclusive(41.4), RangeValue::None));
            let expected = RoaringBitmap::from_iter(0..values.len() as u32);
            // let expected = RoaringBitmap::from_iter(0..0);
            let chunk_option = ChunkOption::RecordCount(32 + 1);
            test_fn(
                config.clone(),
                values,
                chunk_option,
                predicate,
                None,
                None,
                expected,
                t_name(format!(
                    "{query_name} Random failure Eq simd (low selectivity, 2 chunks)"
                )),
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
                ChunkOption::RecordCount(5),
                predicate,
                Some(&bitmask),
                None,
                expected,
                t_name(format!("{query_name} with bitmask")),
            );

            // Test 1: Filter with bitmask Ne
            let values = vec![3.3, 5.6, 3.8, 3.3, 2.1, 9.9, 10.1, 3.8];
            let predicate = Predicate::Ne(3.3);
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 4, 6, 7]);
            let expected = RoaringBitmap::from_iter(vec![2, 4, 6, 7]);
            test_fn(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                predicate,
                Some(&bitmask),
                None,
                expected,
                t_name(format!("{query_name} with bitmask Ne")),
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

    #[allow(clippy::too_many_arguments)]
    fn test_query_filter_optionally_materialize_random(
        config: impl Into<CompressorConfig>,
        test_name: impl std::fmt::Display,
        is_filter_materialize: bool,
        round_scale: Option<usize>,
        mean: Option<f64>,
        std_dev: Option<f64>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) {
        let config = config.into();
        let t_name = |n: String| format!("{test_name}: {n}");

        let query_name = if is_filter_materialize {
            "Filter Materialize"
        } else {
            "Filter"
        };

        type TestFnRandom = Box<
            dyn Fn(
                CompressorConfig,
                Vec<f64>,
                ChunkOption,
                Predicate,
                Option<&RoaringBitmap>,
                Option<ChunkId>,
                String,
            ),
        >;
        let test_fn_random: TestFnRandom = if is_filter_materialize {
            Box::new(move |config, values, option, pred, bm, id, name| {
                test_query_filter_materialize_inner_with_expected_calculation(
                    config,
                    values,
                    option,
                    pred,
                    bm,
                    id,
                    name,
                    round_scale,
                )
            })
        } else {
            Box::new(test_query_filter_inner_with_expected_calculation)
        };

        let mut rng = StdRng::from_entropy();
        let round = |x: f64| {
            if let Some(scale) = round_scale {
                (x * scale as f64).round() / scale as f64
            } else {
                x
            }
        };
        let clamp = |x: f64| {
            if let Some(lower_bound) = lower_bound {
                if x < lower_bound {
                    return round(lower_bound);
                }
            }

            if let Some(upper_bound) = upper_bound {
                if x > upper_bound {
                    return round(upper_bound);
                }
            }

            round(x)
        };
        let gen_random_float = || {
            let mut rng = StdRng::from_entropy();
            let fp = match rng.gen_range(0..=10) {
                0 => rng.gen_range(f64::MIN / 2.0..=0.0),
                1..3 => rng.gen_range(0.0..=f64::MAX / 2.0),
                3..6 => rng.gen_range(-10000.0..=10000.0),
                6..=10 => rng.gen_range(0.0..=100.0),
                _ => unreachable!(),
            };

            clamp(fp)
        };
        let mean = mean.unwrap_or_else(gen_random_float);
        let std_dev = std_dev.unwrap_or_else(|| rng.gen_range(0.001..=10.0));
        println!("mean: {}, std_dev: {}", mean, std_dev);
        let distribution = rand_distr::Normal::new(mean, std_dev).unwrap();
        #[cfg(not(miri))]
        let mut n_records = *[10, 100, 1000, 10000, 100000, 1000000]
            .as_slice()
            .choose(&mut rng)
            .unwrap();
        #[cfg(miri)]
        let mut n_records = 100;
        n_records += rng.gen_range(0..n_records - 1);
        let chunk_option = if rng.gen_bool(0.5) {
            let mut rng = StdRng::from_entropy();
            let n_chunks_max = n_records / min(rng.gen_range(1..=10000), n_records);
            let n_chunks = rng.gen_range(1..=n_chunks_max);
            ChunkOption::RecordCount(n_records / n_chunks)
        } else {
            let byte_size = rng.gen_range(100..=1024 * 10);
            ChunkOption::ByteSize(byte_size)
        };
        println!("n_records: {}, chunk_option: {:?}", n_records, chunk_option);
        let gen_values = || {
            let mut rng = StdRng::from_entropy();
            let mut values = Vec::with_capacity(n_records);
            for _ in 0..n_records {
                let fp = clamp(distribution.sample(&mut rng));
                if let Some(scale) = round_scale {
                    values.push((fp * scale as f64).round() / scale as f64);
                } else {
                    values.push(fp);
                }
            }
            values
        };

        let gen_chunk_id = || {
            let mut rng = StdRng::from_entropy();
            if let ChunkOption::RecordCount(record_counts) = chunk_option {
                if rng.gen_bool(0.5) {
                    Some(ChunkId::new(rng.gen_range(0..=(n_records / record_counts))))
                } else {
                    None
                }
            } else {
                None
            }
        };

        let gen_bitmask = || {
            let mut rng = StdRng::from_entropy();
            if rng.gen_bool(0.5) {
                let mut bitmask = RoaringBitmap::new();
                let p = rng.gen_range(0.0..=1.0);
                for i in 0..n_records {
                    if rng.gen_bool(p) {
                        bitmask.insert(i as u32);
                    }
                }
                Some(bitmask)
            } else {
                None
            }
        };

        // Range
        {
            let mut gen_range_value = || match rng.gen_range(0..=3) {
                0 => RangeValue::None,
                1 => RangeValue::Inclusive(clamp(distribution.sample(&mut rng))),
                2 => RangeValue::Exclusive(clamp(distribution.sample(&mut rng))),
                3 => RangeValue::Inclusive(clamp(distribution.sample(&mut rng))),
                _ => unreachable!(),
            };
            let range = {
                let mut range = Range::new(gen_range_value(), gen_range_value());
                if let (
                    RangeValue::Inclusive(start) | RangeValue::Exclusive(start),
                    RangeValue::Inclusive(end) | RangeValue::Exclusive(end),
                ) = (range.start(), range.end())
                {
                    if start > end {
                        range.swap();
                        range
                    } else {
                        range
                    }
                } else {
                    range
                }
            };
            let predicate = Predicate::Range(range);
            let values = gen_values();
            println!(
                "values max: {}, min: {}",
                values.iter().copied().reduce(f64::max).unwrap(),
                values.iter().copied().reduce(f64::min).unwrap()
            );
            let bitmask = gen_bitmask();
            let chunk_id = gen_chunk_id();

            test_fn_random(
                config.clone(),
                values.clone(),
                chunk_option,
                predicate,
                bitmask.as_ref(),
                chunk_id,
                t_name(format!(
                    "Range({:?}), {:?}, bitmask?: {} {query_name}",
                    predicate,
                    chunk_id,
                    bitmask.is_some()
                )),
            );
        }

        // Eq
        {
            let predicate = Predicate::Eq(clamp(distribution.sample(&mut rng)));
            let values = gen_values();
            let bitmask = gen_bitmask();
            let chunk_id = gen_chunk_id();

            test_fn_random(
                config.clone(),
                values.clone(),
                chunk_option,
                predicate,
                bitmask.as_ref(),
                chunk_id,
                t_name(format!(
                    "Eq({:?}), {:?}, bitmask?: {} {query_name}",
                    predicate,
                    chunk_id,
                    bitmask.is_some()
                )),
            );
        }

        // Ne
        {
            let predicate = Predicate::Ne(clamp(distribution.sample(&mut rng)));
            let values = gen_values();
            let bitmask = gen_bitmask();
            let chunk_id = gen_chunk_id();

            test_fn_random(
                config,
                values,
                chunk_option,
                predicate,
                bitmask.as_ref(),
                chunk_id,
                t_name(format!(
                    "Ne({:?}), {:?}, bitmask?: {} {query_name}",
                    predicate,
                    chunk_id,
                    bitmask.is_some()
                )),
            );
        }
    }

    pub(crate) fn test_query_filter(
        config: impl Into<CompressorConfig>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        round_scale: Option<usize>,
        test_name: impl std::fmt::Display + Clone,
    ) {
        let config = config.into();
        test_query_filter_optionally_materialize(
            config.clone(),
            test_name.clone(),
            false,
            round_scale,
        );
        test_query_filter_optionally_materialize_random(
            config,
            test_name,
            false,
            round_scale,
            None,
            None,
            lower_bound,
            upper_bound,
        );
    }

    pub(crate) fn test_query_filter_materialize(
        config: impl Into<CompressorConfig>,
        round_scale: Option<usize>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        test_name: impl std::fmt::Display + Clone,
    ) {
        let config = config.into();
        test_query_filter_optionally_materialize(
            config.clone(),
            test_name.clone(),
            true,
            round_scale,
        );
        test_query_filter_optionally_materialize_random(
            config,
            test_name,
            true,
            round_scale,
            None,
            None,
            lower_bound,
            upper_bound,
        );
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
            let values = vec![33.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let expected = values.clone();
            test_query_materialize_inner(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                None,
                None,
                round_scale,
                expected,
                t_name("Materialize without bitmask and chunk_id"),
            );

            let values = vec![0.4, 0.4, 0.5];
            let expected = values.clone();
            test_query_materialize_inner(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                None,
                None,
                round_scale,
                expected,
                t_name("Materialize without bitmask and chunk_id almost the same values"),
            );
        }

        // Test 2: Materialize with bitmask
        {
            let values = vec![33.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1];
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 5]);
            let expected = vec![33.3, 3.8, 6.7, 9.9];
            test_query_materialize_inner(
                config.clone(),
                values,
                ChunkOption::RecordCount(5),
                Some(&bitmask),
                None,
                round_scale,
                expected,
                t_name("Materialize with bitmask"),
            );
        }

        // Test 3: Materialize with chunk_id
        {
            let values = vec![33.3, 5.6, 3.8, 6.7, 2.1, 9.9, 10.1, 200.3];
            let chunk_option = ChunkOption::RecordCount(3);
            let chunk_id = ChunkId::new(2);
            let expected = vec![10.1, 200.3];
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
            let values = vec![33.3, 5.6, 3.8, 67.7, -2.1, 9999.9, 10.1];
            let chunk_option = ChunkOption::RecordCount(3);
            let chunk_id = ChunkId::new(1);
            let bitmask = RoaringBitmap::from_iter(vec![0, 2, 3, 5]);
            let expected = vec![67.7, 9999.9];
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
    helper::test_query_filter(config, None, None, None, "Uncompressed");
}

#[test]
fn test_rle_filter() {
    let config = CompressorConfig::rle().build();
    helper::test_query_filter(config, None, None, None, "RLE");
}

#[test]
fn test_gorilla_filter() {
    let config = CompressorConfig::gorilla().build();
    helper::test_query_filter(config, None, None, None, "Gorilla");
}

#[test]
fn test_buff_filter() {
    let scale = 100;
    let fractional_part_bits_length = 8;
    let integer_part_max_bits_length = 32 - fractional_part_bits_length;

    let config = CompressorConfig::buff().scale(scale).build();

    let upper_bound = 2.0f64.powf(integer_part_max_bits_length as f64);
    let lower_bound = -upper_bound;

    let upper_bound = (upper_bound * scale as f64).floor() / scale as f64;
    let lower_bound = (lower_bound * scale as f64).floor() / scale as f64;
    helper::test_query_filter(
        config,
        Some(lower_bound),
        Some(upper_bound),
        Some(scale),
        "BUFF",
    );
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
    let scale = 10;
    let config = CompressorConfig::buff().scale(scale).build();
    helper::test_query_materialize(config, Some(scale), "BUFF");
}

#[test]
fn test_uncompressed_filter_materialize() {
    let config = CompressorConfig::uncompressed().build();
    helper::test_query_filter_materialize(config, None, None, None, "Uncompressed");
}

#[test]
fn test_rle_filter_materialize() {
    let config = CompressorConfig::rle().build();
    helper::test_query_filter_materialize(config, None, None, None, "RLE");
}

#[test]
fn test_gorilla_filter_materialize() {
    let config = CompressorConfig::gorilla().build();
    helper::test_query_filter_materialize(config, None, None, None, "Gorilla");
}

#[test]
fn test_buff_filter_materialize() {
    let scale = 10;
    let fractional_part_bits_length = 8;
    let integer_part_max_bits_length = 32 - fractional_part_bits_length;

    let config = CompressorConfig::buff().scale(scale).build();

    let upper_bound = 2.0f64.powf(integer_part_max_bits_length as f64);
    let lower_bound = -upper_bound;

    let upper_bound = (upper_bound * scale as f64).floor() / scale as f64;
    let lower_bound = (lower_bound * scale as f64).floor() / scale as f64;
    helper::test_query_filter_materialize(
        config,
        Some(10),
        Some(lower_bound),
        Some(upper_bound),
        "BUFF",
    );
}
