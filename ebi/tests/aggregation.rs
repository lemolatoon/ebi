mod common;

use core::f64;

use ebi::{
    api::decoder::{Aggregation, AggregationKind, AggregationResult, Expr},
    compressor::CompressorConfig,
    decoder::{self, expr::Op},
    encoder::ChunkOption,
    time::SegmentedExecutionTimes,
};
use roaring::RoaringBitmap;

fn round_at(value: f64, scale: u32) -> f64 {
    (value * scale as f64).round() / scale as f64
}

mod helper {
    use std::{fmt::Display, io::Cursor};

    use ebi::{
        api::{
            decoder::{ChunkId, Decoder, DecoderInput},
            encoder::{Encoder, EncoderInput, EncoderOutput},
        },
        compressor::CompressorConfig,
        encoder::ChunkOption,
    };
    use roaring::RoaringBitmap;

    use crate::round_at;

    #[allow(clippy::too_many_arguments)]
    pub fn test_sum_inner(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        expected: f64,
        round_scale: Option<u32>,
        test_name: impl Display,
    ) {
        println!("=========== {} ===========", test_name);
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let sum = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();

            decoder.sum(bitmask, chunk_id).unwrap()
        };

        println!("(sum, expected): ({}, {})", sum, expected);
        let sum = match round_scale {
            Some(scale) => round_at(sum, scale),
            None => sum,
        };
        let expected = match round_scale {
            Some(scale) => round_at(expected, scale),
            None => expected,
        };
        println!("(sum, expected): ({}, {})", sum, expected);

        let eps = if let CompressorConfig::BUFF(_) = config {
            0.5 * values.len() as f64
        } else {
            1e-3
        };
        assert!(
            (sum - expected).abs() < eps,
            "[{test_name}]: Sum result mismatch (expected: {expected}, got: {sum}, diff: {})",
            (sum - expected).abs()
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn test_max_inner(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        expected: f64,
        round_scale: Option<u32>,
        test_name: impl Display,
    ) {
        println!("=========== {} ===========", test_name);
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let max = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();

            decoder.max(bitmask, chunk_id).unwrap()
        };

        println!("(max, expected): ({}, {})", max, expected);
        let max = match round_scale {
            Some(scale) => round_at(max, scale),
            None => max,
        };
        let expected = match round_scale {
            Some(scale) => round_at(expected, scale),
            None => expected,
        };
        println!("(max, expected): ({}, {})", max, expected);

        assert_eq!(
            max, expected,
            "[{test_name}]: Max result mismatch (expected: {expected}, got: {max})"
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn test_min_inner(
        config: CompressorConfig,
        values: Vec<f64>,
        chunk_option: ChunkOption,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
        expected: f64,
        round_scale: Option<u32>,
        test_name: impl Display,
    ) {
        println!("=========== {} ===========", test_name);
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(&values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(encoder_input, encoder_output, chunk_option, config);
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        let min = {
            let input_cursor = Cursor::new(&encoded[..]);
            let decoder_input = DecoderInput::from_reader(input_cursor);
            let mut decoder = Decoder::new(decoder_input).unwrap();

            decoder.min(bitmask, chunk_id).unwrap()
        };

        let min = match round_scale {
            Some(scale) => round_at(min, scale),
            None => min,
        };
        let expected = match round_scale {
            Some(scale) => round_at(expected, scale),
            None => expected,
        };

        assert_eq!(
            min, expected,
            "[{test_name}]: min result mismatch (expected: {expected}, got: {min})"
        );
    }

    pub fn decoder(values: &[f64]) -> Decoder<Cursor<Vec<u8>>> {
        let compressor_config: CompressorConfig = CompressorConfig::uncompressed().build().into();
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                ChunkOption::RecordCount(3),
                compressor_config,
            );
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        Decoder::new(DecoderInput::from_reader(Cursor::new(encoded))).unwrap()
    }

    pub fn decoders(values: Vec<&[f64]>) -> Vec<Decoder<Cursor<Vec<u8>>>> {
        values.into_iter().map(decoder).collect::<Vec<_>>()
    }
}

fn test_sum(
    config: impl Into<CompressorConfig>,
    round_scale: Option<u32>,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) {
    let config = config.into();
    helper::test_sum_inner(
        config,
        vec![0.0, 2.0, 3.0, 4.0, 5.0],
        ChunkOption::RecordCount(8),
        None,
        None,
        14.0,
        round_scale,
        "Basic Sum(base0)",
    );
    helper::test_sum_inner(
        config,
        vec![11.0, 12.0],
        ChunkOption::RecordCount(8),
        Some(&RoaringBitmap::from_iter(vec![0])),
        None,
        11.0,
        round_scale,
        "Basic Sum(bitmask)",
    );
    helper::test_sum_inner(
        config,
        vec![0.0, 2.0, 3.0, 4.0, 500.0],
        ChunkOption::RecordCount(8),
        None,
        None,
        509.0,
        round_scale,
        "Basic Sum(multi-subcolumn)",
    );
    helper::test_sum_inner(
        config,
        vec![-1.0, 2.0, 3.0, 4.0, 5.0],
        ChunkOption::RecordCount(8),
        None,
        None,
        13.0,
        round_scale,
        "Basic Sum(base -1)",
    );

    let mut gen = common::RandomGen::new(round_scale, lower_bound, upper_bound);
    let n_records = gen.n_records();
    let lower_bound_lower_bound = -1.0e7 / n_records as f64;
    let upper_bound_upper_bound = 1.0e7 / n_records as f64;
    let lower_bound =
        lower_bound.map_or(lower_bound_lower_bound, |v| v.max(lower_bound_lower_bound));
    let upper_bound =
        upper_bound.map_or(upper_bound_upper_bound, |v| v.min(upper_bound_upper_bound));
    gen.update_bounds(Some(lower_bound), Some(upper_bound));

    let values = gen.values();
    let chunk_option = gen.chunk_option();
    let bitmask = gen.bitmask();
    let expected = if let Some(bitmask) = bitmask.as_ref() {
        let mut sum = 0.0;
        for i in bitmask {
            sum += values[i as usize];
        }

        sum
    } else {
        values.iter().sum()
    };
    let n_records = gen.n_records();
    let (values, expected) = if expected.is_infinite() {
        let values: Vec<f64> = values
            .into_iter()
            .map(|v| gen.round(v / n_records as f64))
            .collect();
        let expected = if let Some(bitmask) = bitmask.as_ref() {
            let mut sum = 0.0;
            for i in bitmask {
                sum += values[i as usize];
            }

            sum
        } else {
            values.iter().sum()
        };
        assert!(
            expected.is_finite(),
            "Random Sum: Expected sum is infinite, but the sum of the values is finite"
        );
        (values, expected)
    } else {
        (values, expected)
    };
    println!("Random Info: {:?}", gen);
    helper::test_sum_inner(
        config,
        values,
        chunk_option,
        bitmask.as_ref(),
        None,
        expected,
        round_scale,
        "Random Sum",
    );
}

fn test_max(
    config: impl Into<CompressorConfig>,
    round_scale: Option<u32>,
    mut lower_bound: Option<f64>,
    mut upper_bound: Option<f64>,
) {
    lower_bound.get_or_insert(1.0e-15);
    upper_bound.get_or_insert(1.0e15);

    let config = config.into();
    helper::test_max_inner(
        config,
        vec![-1.0, 2.0, 3.0, 4.0, 5.0],
        ChunkOption::RecordCount(8),
        None,
        None,
        5.0,
        round_scale,
        "Basic Max",
    );

    let mut gen = common::RandomGen::new(round_scale, lower_bound, upper_bound);

    let values = gen.values();
    let chunk_option = gen.chunk_option();
    let bitmask = gen.bitmask();
    let expected = if let Some(bitmask) = bitmask.as_ref() {
        let mut max = f64::NEG_INFINITY;
        for i in bitmask {
            max = max.max(values[i as usize]);
        }
        max
    } else {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    println!("Random Info: {:?}", gen);
    helper::test_max_inner(
        config,
        values,
        chunk_option,
        bitmask.as_ref(),
        None,
        expected,
        round_scale,
        "Random Max",
    );
}

fn test_min(
    config: impl Into<CompressorConfig>,
    round_scale: Option<u32>,
    mut lower_bound: Option<f64>,
    mut upper_bound: Option<f64>,
) {
    lower_bound.get_or_insert(1.0e-15);
    upper_bound.get_or_insert(1.0e15);
    let config = config.into();
    helper::test_min_inner(
        config,
        vec![-1.0, 2.0, 3.0, 4.0, 5.0],
        ChunkOption::RecordCount(8),
        None,
        None,
        -1.0,
        round_scale,
        "Basic Min",
    );

    let mut gen = common::RandomGen::new(round_scale, lower_bound, upper_bound);

    let values = gen.values();
    let chunk_option = gen.chunk_option();
    let bitmask = gen.bitmask();
    let expected = if let Some(bitmask) = bitmask.as_ref() {
        let mut min = f64::INFINITY;
        for i in bitmask {
            min = min.min(values[i as usize]);
        }
        min
    } else {
        values.iter().copied().fold(f64::INFINITY, f64::min)
    };
    println!("Random Info: {:?}", gen);
    helper::test_min_inner(
        config,
        values,
        chunk_option,
        bitmask.as_ref(),
        None,
        expected,
        round_scale,
        "Random Min",
    );
}

macro_rules! declare_aggregation_tests {
    ($($method:ident, $round_scale:expr, $lower_bound:expr, $upper_bound:expr,)*) => {
        $(
            mod $method {
                #[test]
                fn test_sum() {
                    let config = super::CompressorConfig::$method().scale($round_scale).build();
                    super::test_sum(config, Some($round_scale), Some($lower_bound), Some($upper_bound));
                }

                #[test]
                fn test_max() {
                    let config = super::CompressorConfig::$method().scale($round_scale).build();
                    super::test_max(config, Some($round_scale), Some($lower_bound), Some($upper_bound));
                }

                #[test]
                fn test_min() {
                    let config = super::CompressorConfig::$method().scale($round_scale).build();
                    super::test_min(config, Some($round_scale), Some($lower_bound), Some($upper_bound));
                }
            }
        )*
    };
    ($($method:ident, $round_scale:expr,)*) => {
        $(
            mod $method {
                #[test]
                fn test_sum() {
                    let config = super::CompressorConfig::$method().scale($round_scale).build();
                    super::test_sum(config, Some($round_scale), None, None);
                }

                #[test]
                fn test_max() {
                    let config = super::CompressorConfig::$method().scale($round_scale).build();
                    super::test_max(config, Some($round_scale), None, None);
                }

                #[test]
                fn test_min() {
                    let config = super::CompressorConfig::$method().scale($round_scale).build();
                    super::test_min(config, Some($round_scale), None, None);
                }
            }
        )*
    };
    ($($method:ident,)*) => {
        $(
            mod $method {
                #[test]
                fn test_sum() {
                    let config = super::CompressorConfig::$method().build();
                    super::test_sum(config, None, None, None);
                }

                #[test]
                fn test_max() {
                    let config = super::CompressorConfig::$method().build();
                    super::test_max(config, None, None, None);
                }

                #[test]
                fn test_min() {
                    let config = super::CompressorConfig::$method().build();
                    super::test_min(config, None, None, None);
                }

            }
        )*
    };
}

#[test]
fn test_expr1() {
    // Test sum
    let mut decoders = helper::decoders(vec![&[0.0, 2.0, 3.0, 4.0, 5.0]]);
    let result = Aggregation::new(AggregationKind::Sum, Expr::Index(0))
        .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
        .unwrap();
    assert_eq!(result, AggregationResult::Scalar(14.0));
}

#[cfg(not(miri))]
mod not_miri {
    use super::*;

    #[test]
    fn test_expr2() {
        // Test sum with expr
        let mut decoders = helper::decoders(vec![
            &[0.0, 2.0, -3.0, 4.0, 7.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        ]);
        let result = Aggregation::new(
            AggregationKind::Sum,
            Expr::Binary {
                op: decoder::expr::Op::Add,
                lhs: Box::new(Expr::Binary {
                    op: decoder::expr::Op::Mul,
                    lhs: Box::new(Expr::Index(0)),
                    rhs: Box::new(Expr::Index(1)),
                }),
                rhs: Box::new(Expr::Index(1)),
            },
        );
        assert_eq!(
            result
                .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
                .unwrap(),
            AggregationResult::Scalar(61.0)
        );
    }

    #[test]
    fn test_expr3() {
        // Sum with Index only
        let mut decoders = helper::decoders(vec![&[0.0, 2.0, 3.0, 4.0, 5.0]]);
        let result = Aggregation::new(AggregationKind::Sum, Expr::Index(0))
            .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
            .unwrap();
        assert_eq!(result, AggregationResult::Scalar(14.0));
    }

    #[test]
    fn test_expr4() {
        // Sum with a Binary Expr: (a0 * a1) + a1
        let mut decoders = helper::decoders(vec![
            &[0.0, 2.0, -3.0, 4.0, 7.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        ]);
        let sum_expr = Expr::Binary {
            op: Op::Add,
            lhs: Box::new(Expr::Binary {
                op: Op::Mul,
                lhs: Box::new(Expr::Index(0)),
                rhs: Box::new(Expr::Index(1)),
            }),
            rhs: Box::new(Expr::Index(1)),
        };
        let result = Aggregation::new(AggregationKind::Sum, sum_expr)
            .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
            .unwrap();
        // (-0*1+1) + (2*2+2) + (-3*3+3) + (4*4+4) + (7*5+5) = 1 + 6 + (-6) + 20 + 40 = 61
        assert_eq!(result, AggregationResult::Scalar(61.0));
    }

    // --- Max ---

    #[test]
    fn test_expr5() {
        // Max with Index only
        let mut decoders = helper::decoders(vec![&[1.0, 5.0, 2.0, 4.0]]);
        let result = Aggregation::new(AggregationKind::Max, Expr::Index(0))
            .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
            .unwrap();
        assert_eq!(result, AggregationResult::Scalar(5.0));
    }

    #[test]
    fn test_expr6() {
        // Max with Binary expr: a0 * a1
        let mut decoders = helper::decoders(vec![&[1.0, 2.0, 3.0], &[2.0, 1.0, 4.0]]);
        let max_expr = Expr::Binary {
            op: Op::Mul,
            lhs: Box::new(Expr::Index(0)),
            rhs: Box::new(Expr::Index(1)),
        };
        let result = Aggregation::new(AggregationKind::Max, max_expr)
            .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
            .unwrap();
        // products: [2, 2, 12] → max = 12
        assert_eq!(result, AggregationResult::Scalar(12.0));
    }

    // --- Min ---

    #[test]
    fn test_expr7() {
        // Min with Index only
        let mut decoders = helper::decoders(vec![&[1.0, 5.0, 2.0, 4.0]]);
        let result = Aggregation::new(AggregationKind::Min, Expr::Index(0))
            .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
            .unwrap();
        assert_eq!(result, AggregationResult::Scalar(1.0));
    }

    #[test]
    fn test_expr8() {
        // Min with Binary expr: a0 - a1
        let mut decoders = helper::decoders(vec![&[3.0, 1.0, 4.0], &[2.0, 2.0, 1.0]]);
        let min_expr = Expr::Binary {
            op: Op::Sub,
            lhs: Box::new(Expr::Index(0)),
            rhs: Box::new(Expr::Index(1)),
        };
        let result = Aggregation::new(AggregationKind::Min, min_expr)
            .compute(&mut decoders, None, &mut SegmentedExecutionTimes::new())
            .unwrap();
        // differences: [1, -1, 3] → min = -1
        assert_eq!(result, AggregationResult::Scalar(-1.0));
    }

    // --- Count ---

    #[test]
    fn test_expr9() {
        // Count with Index only, full bitmask
        let mut decoders = helper::decoders(vec![&[0.0, 0.8, 0.3], &[1.0, 1.0, 20.0]]);
        let result = Aggregation::new(AggregationKind::Sum, Expr::Index(0))
            .compute(
                &mut decoders,
                Some(&RoaringBitmap::from_sorted_iter(0..2).unwrap()),
                &mut SegmentedExecutionTimes::new(),
            )
            .unwrap();
        assert_eq!(result, AggregationResult::Scalar(0.8));
    }

    #[test]
    fn test_expr10() {
        // Count with Index only, custom bitmask
        // bitmask selects indices 0 and 2 of each decoder
        let mut bm = RoaringBitmap::new();
        bm.insert(0);
        bm.insert(2);
        let mut decoders = helper::decoders(vec![&[1.0, 2.2, 5.0, 4.0]]);
        let result = Aggregation::new(AggregationKind::Sum, Expr::Index(0))
            .compute(
                &mut decoders,
                Some(&bm),
                &mut SegmentedExecutionTimes::new(),
            )
            .unwrap();
        // from one decoder of length 4, bitmask len = 2
        assert_eq!(result, AggregationResult::Scalar(6.0));
    }

    #[test]
    fn test_multiple_sum_index_and_binary() {
        let mut decoders = helper::decoders(vec![&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]]);
        let aggs = vec![
            Aggregation::new(AggregationKind::Sum, Expr::Index(0)), // 1 + 2 + 3 = 6
            Aggregation::new(
                AggregationKind::Sum,
                Expr::Binary {
                    op: Op::Add,
                    lhs: Box::new(Expr::Index(0)),
                    rhs: Box::new(Expr::Index(1)), // [11, 22, 33] → sum = 66
                },
            ),
        ];
        let result = Aggregation::compute_multiple(
            &aggs,
            &mut decoders,
            None,
            &mut SegmentedExecutionTimes::new(),
        )
        .unwrap();
        assert_eq!(
            result,
            vec![
                AggregationResult::Scalar(6.0),
                AggregationResult::Scalar(66.0),
            ]
        );
    }

    #[test]
    fn test_multiple_max_index_and_binary() {
        let mut decoders = helper::decoders(vec![&[3.0, 8.0, 5.0], &[2.0, 1.0, 7.0]]);
        let aggs = vec![
            Aggregation::new(AggregationKind::Max, Expr::Index(1)), // max = 7
            Aggregation::new(
                AggregationKind::Max,
                Expr::Binary {
                    op: Op::Mul,
                    lhs: Box::new(Expr::Index(0)),
                    rhs: Box::new(Expr::Index(1)), // [6, 8, 35] → max = 35
                },
            ),
        ];
        let result = Aggregation::compute_multiple(
            &aggs,
            &mut decoders,
            None,
            &mut SegmentedExecutionTimes::new(),
        )
        .unwrap();
        assert_eq!(
            result,
            vec![
                AggregationResult::Scalar(7.0),
                AggregationResult::Scalar(35.0),
            ]
        );
    }

    #[test]
    fn test_multiple_min_index_and_binary() {
        let mut decoders = helper::decoders(vec![&[6.0, 2.0, 9.0], &[1.0, 2.0, 3.0]]);
        let aggs = vec![
            Aggregation::new(AggregationKind::Min, Expr::Index(0)), // min = 2
            Aggregation::new(
                AggregationKind::Min,
                Expr::Binary {
                    op: Op::Sub,
                    lhs: Box::new(Expr::Index(0)),
                    rhs: Box::new(Expr::Index(1)), // [5, 0, 6] → min = 0
                },
            ),
        ];
        let result = Aggregation::compute_multiple(
            &aggs,
            &mut decoders,
            None,
            &mut SegmentedExecutionTimes::new(),
        )
        .unwrap();
        assert_eq!(
            result,
            vec![
                AggregationResult::Scalar(2.0),
                AggregationResult::Scalar(0.0),
            ]
        );
    }

    #[test]
    fn test_multiple_count_index_and_binary() {
        let mut decoders = helper::decoders(vec![&[1.0, 0.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
        let mut bm = RoaringBitmap::new();
        bm.insert(0);
        bm.insert(2); // only indices 0 and 2 will be counted

        let aggs = vec![
            Aggregation::new(AggregationKind::Sum, Expr::Index(0)), // count = 2
            Aggregation::new(
                AggregationKind::Max,
                Expr::Binary {
                    op: Op::Add,
                    lhs: Box::new(Expr::Index(0)),
                    rhs: Box::new(Expr::Index(1)), // count = 2
                },
            ),
        ];
        let result = Aggregation::compute_multiple(
            &aggs,
            &mut decoders,
            Some(&bm),
            &mut SegmentedExecutionTimes::new(),
        )
        .unwrap();
        assert_eq!(
            result,
            vec![
                AggregationResult::Scalar(4.0),
                AggregationResult::Scalar(10.0),
            ]
        );
    }
}

declare_aggregation_tests!(uncompressed,);
#[cfg(not(miri))]
declare_aggregation_tests!(
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
#[cfg(all(feature = "ffi_alp", not(miri)))]
declare_aggregation_tests!(ffi_alp,);

declare_aggregation_tests!(delta_sprintz, 100, 1.0e-15, 1.0e15,);

declare_aggregation_tests!(
    buff,
    100,
    super::round_at(-(2.0f64.powf(24.0)), 100),
    super::round_at(2.0f64.powf(24.0), 100),
);
