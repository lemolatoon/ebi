use derive_builder::Builder;
use tsz::{stream::Write as _, Bit};

use crate::io::buffered_bit_writer::BufferedWriterExt;

use super::{Capacity, Compressor};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpCompressor {
    encoder: ChimpEncoder,
    total_bytes_in: usize,
}

impl ChimpCompressor {
    pub fn new() -> Self {
        ChimpCompressor {
            encoder: ChimpEncoder::new(),
            total_bytes_in: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        ChimpCompressor {
            encoder: ChimpEncoder::with_capacity(capacity),
            total_bytes_in: 0,
        }
    }
}

#[derive(Builder, Debug, Clone, Copy)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct ChimpCompressorConfig {
    #[builder(setter(into), default)]
    pub(crate) capacity: Capacity,
}

impl ChimpCompressorConfigBuilder {
    pub fn build(self) -> ChimpCompressorConfig {
        let Self { capacity } = self;
        ChimpCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
        }
    }
}

impl Compressor for ChimpCompressor {
    fn compress(&mut self, input: &[f64]) {
        self.reset();
        self.total_bytes_in += size_of_val(input);

        for &value in input {
            self.encoder.add_value(value);
        }
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        debug_assert_eq!(
            self.encoder.size().next_multiple_of(8) as usize / 8,
            self.encoder.get_out().len()
        );
        self.encoder.get_out().len()
    }

    fn prepare(&mut self) {}

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];
        [self.encoder.get_out(), E, E, E, E]
    }

    fn reset(&mut self) {
        self.encoder.reset();
        self.total_bytes_in = 0;
    }
}

impl Default for ChimpCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpEncoder {
    w: BufferedWriterExt,
    stored_leading_zeros: u32,
    stored_val: u64,
    first: bool,
    size: u32,
}

impl ChimpEncoder {
    pub const THRESHOLD: u32 = 6;

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
        ChimpEncoder {
            w: BufferedWriterExt::with_capacity(capacity),
            stored_leading_zeros: u32::MAX,
            stored_val: 0,
            first: true,
            size: 0,
        }
    }

    #[inline]
    pub fn add_value(&mut self, value: f64) {
        let value = value.to_bits();
        if self.first {
            self.write_first(value);
        } else {
            self.compress_value(value);
        }
    }

    #[inline]
    fn write_first(&mut self, value: u64) {
        self.first = false;
        self.stored_val = value;
        self.w.write_bits(value, 64);
        self.size += 64;
    }

    pub fn close(&mut self) {
        self.add_value(f64::NAN);
        self.w.write_bit(Bit::Zero);
    }

    #[inline]
    fn compress_value(&mut self, value: u64) {
        let xor = self.stored_val ^ value;
        if xor == 0 {
            self.w.write_bit(Bit::Zero);
            self.w.write_bit(Bit::Zero);
            self.size += 2;
            self.stored_leading_zeros = 65;
        } else {
            let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;
            let trailing_zeros = xor.trailing_zeros();

            if trailing_zeros > Self::THRESHOLD {
                let significant_bits = 64 - leading_zeros - trailing_zeros;
                self.w.write_bit(Bit::Zero);
                self.w.write_bit(Bit::One);
                self.w.write_bits(
                    Self::LEADING_REPRESENTATION[leading_zeros as usize] as u64,
                    3,
                );
                self.w.write_bits(significant_bits as u64, 6);
                self.w.write_bits(xor >> trailing_zeros, significant_bits);
                self.size += 11 + significant_bits;
                self.stored_leading_zeros = 65;
            } else if leading_zeros == self.stored_leading_zeros {
                self.w.write_bit(Bit::One);
                self.w.write_bit(Bit::Zero);
                let significant_bits = 64 - leading_zeros;
                self.w.write_bits(xor, significant_bits);
                self.size += 2 + significant_bits;
            } else {
                self.stored_leading_zeros = leading_zeros;
                let significant_bits = 64 - leading_zeros;
                self.w.write_bit(Bit::One);
                self.w.write_bit(Bit::One);
                self.w.write_bits(
                    Self::LEADING_REPRESENTATION[leading_zeros as usize] as u64,
                    3,
                );
                self.w.write_bits(xor, significant_bits);
                self.size += 5 + significant_bits;
            }
        }
        self.stored_val = value;
    }

    /// Returns the number of bits written to the buffer.
    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn get_out(&self) -> &[u8] {
        self.w.as_slice()
    }

    pub fn reset(&mut self) {
        self.w.reset();
        self.stored_leading_zeros = u32::MAX;
        self.stored_val = 0;
        self.first = true;
        self.size = 0;
    }
}

impl Default for ChimpEncoder {
    fn default() -> Self {
        Self::new()
    }
}
