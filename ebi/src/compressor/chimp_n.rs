use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder,
    format::{deserialize, serialize},
    io::bit_write::{BitWrite as _, BufferedBitWriter},
    time::{SegmentKind, SegmentedExecutionTimes},
};

use super::{Capacity, Compressor};

pub type Chimp128Compressor = ChimpNCompressor<128>;

#[derive(Debug, Clone, PartialEq)]
pub struct ChimpNCompressor<const N: usize> {
    encoder: ChimpNEncoder<N>,
    total_bytes_in: usize,
    timer: SegmentedExecutionTimes,
}

impl<const N: usize> ChimpNCompressor<N> {
    pub fn new() -> Self {
        Self {
            encoder: ChimpNEncoder::new(),
            total_bytes_in: 0,
            timer: SegmentedExecutionTimes::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            encoder: ChimpNEncoder::with_capacity(capacity),
            total_bytes_in: 0,
            timer: SegmentedExecutionTimes::new(),
        }
    }
}

pub type Chimp128CompressorConfig = ChimpCompressorConfig<128>;
pub type Chimp128CompressorConfigBuilder = ChimpCompressorConfigBuilder<128>;

#[derive(Builder, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, packed)]
pub struct ChimpCompressorConfig<const N: usize> {
    #[builder(setter(into), default)]
    pub(crate) capacity: Capacity,
}
serialize::impl_to_le!(Chimp128CompressorConfig, capacity);
deserialize::impl_from_le_bytes!(Chimp128CompressorConfig, chimp128, (capacity, Capacity));

impl<const N: usize> From<ChimpCompressorConfig<N>> for ChimpNCompressor<N> {
    fn from(config: ChimpCompressorConfig<N>) -> ChimpNCompressor<N> {
        let cap = config.capacity.0 as usize;
        ChimpNCompressor::with_capacity(cap)
    }
}

impl<const N: usize> ChimpCompressorConfigBuilder<N> {
    pub fn build(self) -> ChimpCompressorConfig<N> {
        let Self { capacity } = self;
        ChimpCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
        }
    }
}

impl<const N: usize> Compressor for ChimpNCompressor<N> {
    fn compress(&mut self, input: &[f64]) -> encoder::Result<()> {
        self.reset();
        self.total_bytes_in += size_of_val(input);

        let xor_timer = self.timer.start_measurement(SegmentKind::Xor);
        for &value in input {
            self.encoder.add_value(value);
        }
        xor_timer.stop();

        Ok(())
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        debug_assert_eq!(
            self.encoder.size().next_multiple_of(8) as usize / 8,
            self.encoder.get_out().len()
        );
        // EOF: f64::NAN + 1 bit
        let n_bits = self.encoder.simulate_add_value(f64::NAN) + 1;

        n_bits.next_multiple_of(8) as usize / 8
    }

    fn prepare(&mut self) {
        self.encoder.close();
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];
        [self.encoder.get_out(), E, E, E, E]
    }

    fn reset(&mut self) {
        self.encoder.reset();
        self.total_bytes_in = 0;
    }

    fn execution_times(&self) -> Option<&SegmentedExecutionTimes> {
        Some(&self.timer)
    }
}

impl<const N: usize> Default for ChimpNCompressor<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpNEncoder<const N_PREVIOUS_VALUES: usize> {
    w: BufferedBitWriter,
    stored_leading_zeros: u32,
    stored_values: Box<[u64]>,
    first: bool,
    size: u32,
    indices: Box<[usize]>,
    index: usize,
    current: usize,
}

impl<const N_PREVIOUS_VALUES: usize> ChimpNEncoder<N_PREVIOUS_VALUES> {
    const PREVIOUS_VALUES_LOG2: u32 = (N_PREVIOUS_VALUES as u64).trailing_zeros();
    const THRESHOLD: u32 = /* log_2 64 */ 6 + Self::PREVIOUS_VALUES_LOG2;
    const SET_LSB: u32 = (1 << (Self::THRESHOLD + 1)) - 1;
    const FLAG_ZERO_SIZE: u32 = Self::PREVIOUS_VALUES_LOG2 + 2;
    const FLAG_ONE_SIZE: u32 = Self::PREVIOUS_VALUES_LOG2 + 11;

    pub const LEADING_REPRESENTATION: [u8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7,
    ];

    pub const LEADING_ROUND: [u8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];

    pub fn new() -> Self {
        Self::with_capacity(1024 * 8)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            w: BufferedBitWriter::with_capacity(capacity),
            stored_leading_zeros: u32::MAX,
            stored_values: vec![0; N_PREVIOUS_VALUES].into_boxed_slice(),
            first: true,
            size: 0,
            indices: vec![0usize; 1 << (Self::THRESHOLD + 1)].into_boxed_slice(),
            index: 0,
            current: 0,
        }
    }

    pub fn add_value(&mut self, value: f64) {
        let value_bits = value.to_bits();
        if self.first {
            self.write_first(value_bits);
        } else {
            self.compress_value(value_bits);
        }
    }

    fn write_first(&mut self, value: u64) {
        self.first = false;
        self.stored_values[self.current] = value;
        self.w.write_bits(value, 64);
        self.indices[(value & Self::SET_LSB as u64) as usize] = self.index;
        self.size += 64;
    }

    pub fn close(&mut self) {
        self.add_value(f64::NAN);
        self.w.write_bit(false);
    }

    fn compress_value(&mut self, value: u64) {
        let key = (value & Self::SET_LSB as u64) as usize;
        let xor;
        let previous_index;
        let mut trailing_zeros = 0;
        let curr_index = self.indices[key];

        if (self.index - curr_index) < N_PREVIOUS_VALUES {
            let temp_xor = value ^ self.stored_values[curr_index % N_PREVIOUS_VALUES];
            trailing_zeros = temp_xor.trailing_zeros();
            if trailing_zeros > Self::THRESHOLD {
                previous_index = curr_index % N_PREVIOUS_VALUES;
                xor = temp_xor;
            } else {
                previous_index = self.index % N_PREVIOUS_VALUES;
                xor = self.stored_values[previous_index] ^ value;
            }
        } else {
            previous_index = self.index % N_PREVIOUS_VALUES;
            xor = self.stored_values[previous_index] ^ value;
        }

        if xor == 0 {
            self.w
                .write_bits(previous_index as u64, Self::FLAG_ZERO_SIZE);
            self.size += Self::FLAG_ZERO_SIZE;
            self.stored_leading_zeros = 65;
        } else {
            let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;

            if trailing_zeros > Self::THRESHOLD {
                let significant_bits = 64 - leading_zeros - trailing_zeros;
                self.w.write_bits(
                    (512 * (N_PREVIOUS_VALUES + previous_index)
                        + 64 * Self::LEADING_REPRESENTATION[leading_zeros as usize] as usize
                        + significant_bits as usize) as u64,
                    Self::FLAG_ONE_SIZE,
                );
                self.w.write_bits(xor >> trailing_zeros, significant_bits);
                self.size += significant_bits + Self::FLAG_ONE_SIZE;
                self.stored_leading_zeros = 65;
            } else if leading_zeros == self.stored_leading_zeros {
                self.w.write_bits(2, 2);
                let significant_bits = 64 - leading_zeros;
                self.w.write_bits(xor, significant_bits);
                self.size += 2 + significant_bits;
            } else {
                self.stored_leading_zeros = leading_zeros;
                let significant_bits = 64 - leading_zeros;
                self.w.write_bits(
                    24 + Self::LEADING_REPRESENTATION[leading_zeros as usize] as u64,
                    5,
                );
                self.w.write_bits(xor, significant_bits);
                self.size += 5 + significant_bits;
            }
        }

        self.current = (self.current + 1) % N_PREVIOUS_VALUES;
        self.stored_values[self.current] = value;
        self.index += 1;
        self.indices[key] = self.index;
    }

    /// ```
    /// use ebi::compressor::chimp_n::ChimpNEncoder;
    /// let mut encoder = ChimpNEncoder::<128>::new();
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_add_value(42.0);
    /// assert_eq!(simulated_size, 64);
    ///
    /// // Actual addition of the value
    /// encoder.add_value(42.0);
    /// assert_eq!(encoder.size(), 64);
    ///
    /// encoder.add_value(44.0);
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_add_value(22.0);
    /// // Actual addition of the value
    /// encoder.add_value(22.0);
    /// assert_eq!(simulated_size, encoder.size());
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_add_value(111122.0);
    /// // Actual addition of the value
    /// encoder.add_value(111122.0);
    /// assert_eq!(simulated_size, encoder.size());
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_add_value(44.0);
    /// // Actual addition of the value
    /// encoder.add_value(44.0);
    /// assert_eq!(simulated_size, encoder.size());
    /// ```
    pub fn simulate_add_value(&self, value: f64) -> u32 {
        let value = value.to_bits();
        if self.first {
            return 64;
        }
        let mut simulated_size = self.size;

        let key = (value & Self::SET_LSB as u64) as usize;
        let xor;
        let mut trailing_zeros = 0;
        let curr_index = self.indices[key];

        if (self.index - curr_index) < N_PREVIOUS_VALUES {
            let temp_xor = value ^ self.stored_values[curr_index % N_PREVIOUS_VALUES];
            trailing_zeros = temp_xor.trailing_zeros();
            if trailing_zeros > Self::THRESHOLD {
                xor = temp_xor;
            } else {
                let previous_index = self.index % N_PREVIOUS_VALUES;
                xor = self.stored_values[previous_index] ^ value;
            }
        } else {
            let previous_index = self.index % N_PREVIOUS_VALUES;
            xor = self.stored_values[previous_index] ^ value;
        }

        if xor == 0 {
            simulated_size += Self::FLAG_ZERO_SIZE;
        } else {
            let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;

            if trailing_zeros > Self::THRESHOLD {
                let significant_bits = 64 - leading_zeros - trailing_zeros;
                simulated_size += significant_bits + Self::FLAG_ONE_SIZE;
            } else if leading_zeros == self.stored_leading_zeros {
                let significant_bits = 64 - leading_zeros;
                simulated_size += 2 + significant_bits;
            } else {
                let significant_bits = 64 - leading_zeros;
                simulated_size += 5 + significant_bits;
            }
        }

        simulated_size
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn get_out(&self) -> &[u8] {
        self.w.as_slice()
    }

    pub fn reset(&mut self) {
        self.w.reset();
        self.stored_leading_zeros = u32::MAX;
        self.stored_values.fill(0);
        self.first = true;
        self.size = 0;
        self.indices.fill(0);
        self.index = 0;
        self.current = 0;
    }
}

impl<const PREVIOUS_VALUES: usize> Default for ChimpNEncoder<PREVIOUS_VALUES> {
    fn default() -> Self {
        Self::new()
    }
}
