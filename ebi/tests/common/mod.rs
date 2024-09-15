use std::cmp::min;

use ebi::{
    api::decoder::ChunkId,
    decoder::query::{Predicate, Range, RangeValue},
    encoder::ChunkOption,
};
#[cfg(not(miri))]
use rand::prelude::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Distribution;
use roaring::RoaringBitmap;

#[derive(Debug, Clone)]
pub struct RandomGen {
    round_scale: Option<u32>,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    normal_dist: rand_distr::Normal<f64>,
    n_records: Option<usize>,
    chunk_id: Option<Option<ChunkId>>,
    chunk_option: Option<ChunkOption>,
    bitmask: Option<Option<RoaringBitmap>>,
    predicate: Option<Predicate>,
    rng: StdRng,
}

impl RandomGen {
    pub fn new(
        round_scale: Option<u32>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        let rng = StdRng::from_entropy();
        let mut gen = Self {
            round_scale,
            lower_bound,
            upper_bound,
            normal_dist: rand_distr::Normal::new(0.0, 1.0).unwrap(),
            n_records: None,
            chunk_option: None,
            chunk_id: None,
            bitmask: None,
            predicate: None,
            rng,
        };

        let mean = gen.gen_random_float();
        let std_dev = gen.rng.gen_range(0.001..=10.0);
        gen.update_normal_dist(mean, std_dev);

        gen
    }

    pub fn update_bounds(&mut self, lower_bound: Option<f64>, upper_bound: Option<f64>) {
        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;

        let mean = self.gen_random_float();
        let std_dev = self.rng.gen_range(0.001..=10.0);

        self.update_normal_dist(mean, std_dev);
    }

    #[allow(dead_code)]
    pub fn update_n_records(&mut self, n_records: usize) {
        let is_n_records_set = self.n_records.is_some();
        self.n_records = Some(n_records);

        if !is_n_records_set {
            return;
        }

        self.chunk_option = None;
        self.chunk_id = None;
        self.bitmask = None;
        self.predicate = None;
    }

    fn update_normal_dist(&mut self, mean: f64, std_dev: f64) {
        self.normal_dist = rand_distr::Normal::new(mean, std_dev).unwrap();
    }

    #[inline]
    pub fn round(&self, x: f64) -> f64 {
        if let Some(scale) = self.round_scale {
            (x * scale as f64).round() / scale as f64
        } else {
            x
        }
    }

    #[inline]
    pub fn clamp(&self, x: f64) -> f64 {
        if let Some(lower_bound) = self.lower_bound {
            if x < lower_bound {
                return lower_bound;
            }
        }

        if let Some(upper_bound) = self.upper_bound {
            if x > upper_bound {
                return upper_bound;
            }
        }

        x
    }

    #[inline]
    pub fn gen_random_float(&mut self) -> f64 {
        let fp = match self.rng.gen_range(0..=10) {
            0 => self.rng.gen_range(f64::MIN / 2.0..=0.0),
            1..3 => self.rng.gen_range(0.0..=f64::MAX / 2.0),
            3..6 => self.rng.gen_range(-10000.0..=10000.0),
            6..=10 => self.rng.gen_range(0.0..=100.0),
            _ => unreachable!(),
        };

        self.clamp(fp)
    }

    #[inline]
    pub fn n_records(&mut self) -> usize {
        if let Some(n_records) = self.n_records {
            return n_records;
        }

        #[cfg(not(miri))]
        let mut n_records = *[10, 100, 1000, 10000, 100000, 1000000]
            .as_slice()
            .choose(&mut self.rng)
            .unwrap();
        #[cfg(miri)]
        let mut n_records = 100;
        n_records += self.rng.gen_range(0..n_records - 1);

        self.n_records = Some(n_records);

        n_records
    }

    #[inline]
    pub fn chunk_option(&mut self) -> ChunkOption {
        if let Some(chunk_option) = self.chunk_option {
            return chunk_option;
        }
        let p = self.rng.gen_bool(0.5);
        let n_records = self.n_records();
        let chunk_option = if p {
            let n_chunks_max = n_records / min(self.rng.gen_range(1..=10000), n_records);
            let n_chunks = self.rng.gen_range(1..=n_chunks_max);
            ChunkOption::RecordCount(n_records / n_chunks)
        } else {
            let byte_size = self.rng.gen_range(100..=1024 * 10);
            ChunkOption::ByteSizeBestEffort(byte_size)
        };

        self.chunk_option = Some(chunk_option);
        chunk_option
    }

    #[inline]
    pub fn values(&mut self) -> Vec<f64> {
        let n_records = self.n_records();
        let mut values = Vec::with_capacity(n_records);
        for _ in 0..n_records {
            let fp = self.normal_dist.sample(&mut self.rng);
            let fp = self.clamp(fp);
            values.push(self.round(fp));
        }
        values
    }

    #[inline]
    #[allow(dead_code)]
    pub fn chunk_id(&mut self) -> Option<ChunkId> {
        if let Some(chunk_id) = self.chunk_id {
            return chunk_id;
        }
        let chunk_id = if let ChunkOption::RecordCount(record_counts) = self.chunk_option() {
            let p = self.rng.gen_bool(0.5);
            let n_records = self.n_records();
            if p {
                Some(ChunkId::new(
                    self.rng.gen_range(0..=(n_records / record_counts)),
                ))
            } else {
                None
            }
        } else {
            None
        };

        self.chunk_id = Some(chunk_id);

        chunk_id
    }

    #[inline]
    pub fn bitmask(&mut self) -> Option<RoaringBitmap> {
        if self.bitmask.is_some() {
            return self.bitmask.as_ref().unwrap().clone();
        }
        let p = self.rng.gen_bool(0.5);
        let bitmask = if p {
            let n_records = self.n_records();
            let mut bitmask = RoaringBitmap::new();
            let p = self.rng.gen_range(0.0..=1.0);
            for i in 0..n_records {
                if self.rng.gen_bool(p) {
                    bitmask.insert(i as u32);
                }
            }
            Some(bitmask)
        } else {
            None
        };

        self.bitmask = Some(bitmask);

        self.bitmask.as_ref().unwrap().clone()
    }

    #[inline]
    #[allow(dead_code)]
    fn sample_and_clamp(&mut self) -> f64 {
        let fp = self.normal_dist.sample(&mut self.rng);
        self.clamp(fp)
    }

    #[inline]
    #[allow(dead_code)]
    fn gen_range_value(&mut self) -> RangeValue {
        match self.rng.gen_range(0..=3) {
            0 => RangeValue::None,
            1 => RangeValue::Inclusive(self.sample_and_clamp()),
            2 => RangeValue::Exclusive(self.sample_and_clamp()),
            3 => RangeValue::Inclusive(self.sample_and_clamp()),
            _ => unreachable!(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn gen_range(&mut self) -> Range {
        let mut range = Range::new(self.gen_range_value(), self.gen_range_value());
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
    }

    #[inline]
    #[allow(dead_code)]
    pub fn predicate(&mut self) -> Predicate {
        if let Some(predicate) = self.predicate {
            return predicate;
        }
        let p = self.rng.gen_range(0..=2);
        let predicate = match p {
            0 => Predicate::Eq(self.sample_and_clamp()),
            1 => Predicate::Ne(self.sample_and_clamp()),
            2 => Predicate::Range(self.gen_range()),
            _ => unreachable!(),
        };

        self.predicate = Some(predicate);

        predicate
    }

    #[allow(dead_code)]
    pub fn mean(&self) -> f64 {
        self.normal_dist.mean()
    }

    #[allow(dead_code)]
    pub fn std_dev(&self) -> f64 {
        self.normal_dist.std_dev()
    }
}
