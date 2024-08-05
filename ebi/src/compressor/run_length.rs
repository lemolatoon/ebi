use std::{
    iter,
    mem::{size_of, size_of_val},
};

use crate::format::{
    run_length::RunLengthHeader,
    serialize::{AsBytes, ToLe},
};

use super::{size_estimater::AppendCompressingSizeEstimator, AppendableCompressor, Compressor};

/// Run Length Encoding (RLE) Compression Scheme
/// Chunk Layout:
/// Array of (RunCounts (u8), values (f64))
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct RunLengthCompressor {
    total_bytes_in: usize,
    header: RunLengthHeader,
    bytes: Vec<u8>,
    previous_value: f64,
    previous_run_count: u8,
}
const DEFAULT_CAPACITY: usize = 8000;

impl RunLengthCompressor {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            total_bytes_in: 0,
            header: Default::default(),
            bytes: Vec::with_capacity(capacity),
            previous_value: 0.0,
            previous_run_count: 0,
        }
    }

    fn add_field(&mut self, value: f64, run_count: u8) {
        self.bytes.extend_from_slice(&run_count.to_le_bytes());
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.header
            .set_number_of_fields(self.header.number_of_fields() + 1);
    }
}

impl Default for RunLengthCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for RunLengthCompressor {
    // TODO: Implement RLE size estimator
    type SizeEstimatorImpl<'comp, 'buf> = AppendCompressingSizeEstimator<'comp, 'buf, Self>;

    fn compress(&mut self, input: &[f64]) {
        self.reset();
        let is_starting = self.previous_run_count == 0;
        if is_starting {
            self.previous_value = input[0];
            self.previous_run_count = 1;
        }
        let n_bytes_compressed = size_of_val(input);

        let input = if is_starting { &input[1..] } else { input };
        for value in input {
            // byte level comparison
            if self.previous_value.to_le_bytes() == value.to_le_bytes()
                && self.previous_run_count < u8::MAX
            {
                self.previous_run_count += 1;
                continue;
            };

            self.add_field(self.previous_value, self.previous_run_count);

            self.previous_value = *value;
            self.previous_run_count = 1;
        }

        self.total_bytes_in += n_bytes_compressed;
    }

    fn size_estimator<'comp, 'buf>(
        &'comp mut self,
        input: &'buf [f64],
        estimate_option: super::size_estimater::EstimateOption,
    ) -> Option<Self::SizeEstimatorImpl<'comp, 'buf>> {
        Some(AppendCompressingSizeEstimator::new(
            self,
            input,
            estimate_option,
        ))
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        let is_already_started = self.previous_run_count != 0;
        size_of::<RunLengthHeader>()
            + size_of_val(self.bytes.as_slice())
            + if is_already_started {
                size_of::<u8>() + size_of::<f64>()
            } else {
                0
            }
    }

    /// Prepare the current run count for serialization
    /// Essentially, this function converts the current run count into little endian
    fn prepare(&mut self) {
        self.header.to_le();
        if self.previous_run_count != 0 {
            self.add_field(self.previous_value, self.previous_run_count);
            self.previous_run_count = 0;
        }
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        let header = self.header.as_bytes();
        let bytes = self.bytes.as_slice();

        const EMPTY: &[u8] = &[];
        [header, bytes, EMPTY, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        self.bytes.clear();
        self.header.set_number_of_fields(0);
        self.previous_run_count = 0;

        self.total_bytes_in = 0;
    }
}

impl AppendableCompressor for RunLengthCompressor {
    fn append_compress(&mut self, input: &[f64]) {
        let is_starting = self.previous_run_count == 0;
        if is_starting {
            self.previous_value = input[0];
            self.previous_run_count = 1;
        }
        let n_bytes_compressed = size_of_val(input);

        let input = if is_starting { &input[1..] } else { input };
        for value in input {
            // byte level comparison
            if self.previous_value.to_le_bytes() == value.to_le_bytes()
                && self.previous_run_count < u8::MAX
            {
                self.previous_run_count += 1;
                continue;
            };

            self.add_field(self.previous_value, self.previous_run_count);

            self.previous_value = *value;
            self.previous_run_count = 1;
        }

        self.total_bytes_in += n_bytes_compressed;
    }

    fn rewind(&mut self, n: usize) -> bool {
        let mut remaining_records = n;
        if self.previous_run_count == 0 {
            return false;
        }

        let mut previous_value_run_count_set = false;

        let mut number_of_fields_to_rewind = 0;
        let fileds_iter = self.bytes.chunks_exact(9).rev().map(|field| {
            let run_count = u8::from_le_bytes(field[..1].try_into().unwrap());
            let value = f64::from_le_bytes(field[1..9].try_into().unwrap());
            (run_count, value)
        });
        for (run_count, value) in
            iter::once((self.previous_run_count, self.previous_value)).chain(fileds_iter)
        {
            if remaining_records >= run_count as usize {
                remaining_records -= run_count as usize;
                number_of_fields_to_rewind += 1;
                continue;
            }

            self.previous_value = value;
            self.previous_run_count = run_count - remaining_records as u8;
            remaining_records = 0;
            number_of_fields_to_rewind += 1;
            previous_value_run_count_set = true;
            break;
        }

        if remaining_records != 0 {
            return false;
        }

        if !previous_value_run_count_set {
            self.previous_run_count = 0;
        }

        // rewind fields
        self.bytes
            .truncate(self.bytes.len() - (number_of_fields_to_rewind - 1) * 9);
        self.header.set_number_of_fields(
            self.header.number_of_fields() - (number_of_fields_to_rewind - 1) as u64,
        );
        self.total_bytes_in -= n * size_of::<f64>();

        true
    }
}
