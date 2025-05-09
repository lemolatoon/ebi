use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ebi::compressor::{run_length::RunLengthCompressor, Compressor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn run_length_compressor_bench(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::from_seed([42; 32]);
    const VALUE_LENGTH_HIGH_CARDINALITY: usize = 100000;
    let mut high_cardinality_random_floats: Vec<f64> =
        Vec::with_capacity(VALUE_LENGTH_HIGH_CARDINALITY);
    for _ in 0..VALUE_LENGTH_HIGH_CARDINALITY {
        if rng.gen_bool(0.9) && high_cardinality_random_floats.last().is_some() {
            high_cardinality_random_floats.push(*high_cardinality_random_floats.last().unwrap());
        } else {
            high_cardinality_random_floats.push(rng.gen());
        }
    }

    const RECORD_COUNTS: [usize; 3] = [1000, 10000, 100000];
    for &record_count in RECORD_COUNTS.iter() {
        c.bench_with_input(
            BenchmarkId::new(
                "RunLengthCompressor Compress Chunk at Once with Allocated Compressor",
                format!("{record_count} floating values with high cardinality"),
            ),
            &&high_cardinality_random_floats[..record_count],
            |b, random_values| {
                let mut compressor = RunLengthCompressor::new();
                b.iter(|| {
                    compressor.compress(black_box(random_values)).unwrap();
                    compressor.reset();
                });
            },
        );
    }

    let mut rng = ChaCha20Rng::from_seed([42; 32]);
    const VALUE_LENGTH_LOW_CARDINALITY: usize = 100000;
    let mut high_cardinality_random_floats: Vec<f64> =
        Vec::with_capacity(VALUE_LENGTH_LOW_CARDINALITY);
    for _ in 0..VALUE_LENGTH_LOW_CARDINALITY {
        if rng.gen_bool(0.3) && high_cardinality_random_floats.last().is_some() {
            high_cardinality_random_floats.push(*high_cardinality_random_floats.last().unwrap());
        } else {
            high_cardinality_random_floats.push(rng.gen());
        }
    }

    for &record_count in RECORD_COUNTS.iter() {
        c.bench_with_input(
            BenchmarkId::new(
                "RunLengthCompressor Compress Chunk at Once with Allocated Compressor",
                format!("{record_count} floating values with high cardinality"),
            ),
            &&high_cardinality_random_floats[..record_count],
            |b, random_values| {
                let mut compressor = RunLengthCompressor::new();
                b.iter(|| {
                    compressor.compress(black_box(random_values)).unwrap();
                    compressor.reset();
                });
            },
        );
    }
}

criterion_group!(benches, run_length_compressor_bench);
criterion_main!(benches);
