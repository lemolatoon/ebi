use std::mem::{size_of, size_of_val};

use crate::format::{
    run_length::RunLengthHeader,
    serialize::{AsBytes, ToLe},
};

use super::Compressor;

/// Run Length Encoding (RLE) Compression Scheme
/// Chunk Layout:
/// Values (f64) | RunCounts (u32)
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct RunLengthCompressor {
    total_bytes_in: usize,
    header: RunLengthHeader,
    run_counts: Vec<u32>,
    values: Vec<f64>,
}
const DEFAULT_CAPACITY: usize = 8000;

impl RunLengthCompressor {
    pub fn new() -> Self {
        Self {
            total_bytes_in: 0,
            header: Default::default(),
            run_counts: Vec::with_capacity(DEFAULT_CAPACITY),
            values: Vec::with_capacity(DEFAULT_CAPACITY),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            total_bytes_in: 0,
            header: Default::default(),
            run_counts: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn reset_capacity(&mut self, capacity: usize) {
        self.run_counts.reserve(capacity);
        self.values.reserve(capacity);
    }

    pub fn reset_capacity_to_default(&mut self) {
        self.run_counts.shrink_to(DEFAULT_CAPACITY);
        self.values.shrink_to(DEFAULT_CAPACITY);
    }
}

impl Default for RunLengthCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for RunLengthCompressor {
    fn compress(&mut self, input: &[f64]) -> usize {
        for value in input {
            let Some(last_value) = self.values.last() else {
                self.values.push(*value);
                self.run_counts.push(1);
                continue;
            };

            // byte level comparison
            if last_value.as_bytes() == value.as_bytes() {
                let last_run_count = self.run_counts.last_mut().unwrap();
                *last_run_count += 1;
            } else {
                self.run_counts.push(1);
                self.values.push(*value);
            }
        }

        let n_bytes_compressed = size_of_val(input);
        self.total_bytes_in += n_bytes_compressed;

        n_bytes_compressed
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        size_of::<RunLengthHeader>()
            + size_of_val(&self.run_counts[..])
            + size_of_val(&self.values[..])
    }

    /// Prepare the current run count for serialization
    /// Essentially, this function converts the current run count into little endian
    fn prepare(&mut self) {
        // turn the current run count into little endian
        #[cfg(target_endian = "big")]
        {
            for count in self.run_counts.iter_mut() {
                *count = count.to_le();
            }
        }

        let number_fields = self.values.len() as u64;
        self.header.set_number_of_fields(number_fields);
        self.header.to_le();
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        let header = self.header.as_bytes();
        let run_counts = self.run_counts.as_slice().as_bytes();
        let values = self.values.as_slice().as_bytes();

        const EMPTY: &[u8] = &[];
        [header, values, run_counts, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        self.run_counts.clear();
        self.values.clear();

        self.total_bytes_in = 0;
    }
}
