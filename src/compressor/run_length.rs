use std::mem::size_of_val;

use crate::format::serialize::AsBytes;

use super::Compressor;

/// Run Length Encoding (RLE) Compression Scheme
/// Chunk Layout:
/// RunCounts (u32) | Values (f64)
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct RunLengthCompressor {
    total_bytes_in: usize,
    current_run_count: Vec<u32>,
    current_run_value: Vec<f64>,
}
const DEFAULT_CAPACITY: usize = 8000;

impl RunLengthCompressor {
    pub fn new() -> Self {
        Self {
            total_bytes_in: 0,
            current_run_count: Vec::with_capacity(DEFAULT_CAPACITY),
            current_run_value: Vec::with_capacity(DEFAULT_CAPACITY),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            total_bytes_in: 0,
            current_run_count: Vec::with_capacity(capacity),
            current_run_value: Vec::with_capacity(capacity),
        }
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
            let Some(last_value) = self.current_run_value.last() else {
                self.current_run_value.push(*value);
                self.current_run_count.push(1);
                continue;
            };

            // byte level comparison
            if last_value.as_bytes() == value.as_bytes() {
                let last_run_count = self.current_run_count.last_mut().unwrap();
                *last_run_count += 1;
            } else {
                self.current_run_count.push(1);
                self.current_run_value.push(*value);
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
        size_of_val(&self.current_run_count[..]) + size_of_val(&self.current_run_value[..])
    }

    /// Prepare the current run count for serialization
    /// Essentially, this function converts the current run count into little endian
    fn prepare(&mut self) {
        // turn the current run count into little endian
        {
            for count in self.current_run_count.iter_mut() {
                *count = count.to_le();
            }
        }
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        let run_counts = self.current_run_count.as_slice().as_bytes();
        let values = self.current_run_value.as_slice().as_bytes();

        const EMPTY: &[u8] = &[];
        [run_counts, values, EMPTY, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        self.current_run_count.clear();
        self.current_run_value.clear();

        self.total_bytes_in = 0;
    }
}
