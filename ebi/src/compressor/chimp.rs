use tsz::{stream::Write as _, Bit};

use crate::io::buffered_bit_writer::BufferedWriterExt;

use super::general_xor::{
    GeneralXorCompressor, GeneralXorCompressorConfig, GeneralXorCompressorConfigBuilder, XorEncoder,
};

pub type ChimpCompressor = GeneralXorCompressor<ChimpEncoder>;
pub type ChimpCompressorConfig = GeneralXorCompressorConfig<ChimpEncoder>;
pub type ChimpCompressorConfigBuilder = GeneralXorCompressorConfigBuilder<ChimpEncoder>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpEncoder {
    stored_leading_zeros: u32,
    stored_val: u64,
    first: bool,
    size: usize,
}

impl XorEncoder for ChimpEncoder {
    fn compress_float(&mut self, w: &mut BufferedWriterExt, bits: u64) -> usize {
        let preserved_size = self.size();
        self.add_value(w, f64::from_bits(bits));

        self.size() - preserved_size
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn close(&mut self, w: &mut BufferedWriterExt) {
        ChimpEncoder::close(self, w);
    }

    fn simulate_close(&self) -> usize {
        let preserved_size = self.size();

        self.simulate_add_value(f64::NAN) + 1 - preserved_size
    }
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
        ChimpEncoder {
            stored_leading_zeros: u32::MAX,
            stored_val: 0,
            first: true,
            size: 0,
        }
    }

    #[inline]
    pub fn add_value(&mut self, w: &mut BufferedWriterExt, value: f64) {
        let value = value.to_bits();
        if self.first {
            self.write_first(w, value);
        } else {
            self.compress_value(w, value);
        }
    }

    #[inline]
    fn write_first(&mut self, w: &mut BufferedWriterExt, value: u64) {
        self.first = false;
        self.stored_val = value;
        w.write_bits(value, 64);
        self.size += 64;
    }

    pub fn close(&mut self, w: &mut BufferedWriterExt) {
        self.add_value(w, f64::NAN);
        w.write_bit(Bit::Zero);
    }

    #[inline]
    fn compress_value(&mut self, w: &mut BufferedWriterExt, value: u64) {
        let xor = self.stored_val ^ value;
        if xor == 0 {
            w.write_bit(Bit::Zero);
            w.write_bit(Bit::Zero);
            self.size += 2;
            self.stored_leading_zeros = 65;
        } else {
            let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;
            let trailing_zeros = xor.trailing_zeros();

            if trailing_zeros > Self::THRESHOLD {
                let significant_bits = 64 - leading_zeros - trailing_zeros;
                w.write_bit(Bit::Zero);
                w.write_bit(Bit::One);
                w.write_bits(
                    Self::LEADING_REPRESENTATION[leading_zeros as usize] as u64,
                    3,
                );
                w.write_bits(significant_bits as u64, 6);
                w.write_bits(xor >> trailing_zeros, significant_bits);
                self.size += (11 + significant_bits) as usize;
                self.stored_leading_zeros = 65;
            } else if leading_zeros == self.stored_leading_zeros {
                w.write_bit(Bit::One);
                w.write_bit(Bit::Zero);
                let significant_bits = 64 - leading_zeros;
                w.write_bits(xor, significant_bits);
                self.size += (2 + significant_bits) as usize;
            } else {
                self.stored_leading_zeros = leading_zeros;
                let significant_bits = 64 - leading_zeros;
                w.write_bit(Bit::One);
                w.write_bit(Bit::One);
                w.write_bits(
                    Self::LEADING_REPRESENTATION[leading_zeros as usize] as u64,
                    3,
                );
                w.write_bits(xor, significant_bits);
                self.size += (5 + significant_bits) as usize;
            }
        }
        self.stored_val = value;
    }

    /// Simulates adding a value to the encoder and returns the resulting size.
    ///
    /// This function calculates the size of the encoded data after adding a
    /// given value without actually modifying the encoder's state. It is useful
    /// for estimating the impact on size before performing the actual addition.
    ///
    /// # Arguments
    ///
    /// * `value` - The floating point value to be added to the encoder.
    ///
    /// # Returns
    ///
    /// * `u32` - The size of the encoded data in bits after adding the value.
    ///     /// # Example
    ///
    /// ```
    /// use ebi::compressor::chimp::ChimpEncoder;
    /// use ebi::io::buffered_bit_writer::BufferedWriterExt;
    /// let mut encoder = ChimpEncoder::new();
    /// let mut w = BufferedWriterExt::new();
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_add_value(42.0);
    /// assert_eq!(simulated_size, 64);
    ///
    /// // Actual addition of the value
    /// encoder.add_value(&mut w, 42.0);
    /// assert_eq!(encoder.size(), 64);
    ///
    /// encoder.add_value(&mut w, 44.0);
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_add_value(22.0);
    /// // Actual addition of the value
    /// encoder.add_value(&mut w, 22.0);
    /// assert_eq!(simulated_size, encoder.size());
    /// ```
    pub fn simulate_add_value(&self, value: f64) -> usize {
        let value_bits = value.to_bits();
        let mut simulated_size = self.size;
        let xor = self.stored_val ^ value_bits;

        if self.first {
            simulated_size += 64;
        } else if xor == 0 {
            simulated_size += 2;
        } else {
            let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;
            let trailing_zeros = xor.trailing_zeros();

            if trailing_zeros > Self::THRESHOLD {
                let significant_bits = 64 - leading_zeros - trailing_zeros;
                simulated_size += (11 + significant_bits) as usize;
            } else if leading_zeros == self.stored_leading_zeros {
                let significant_bits = 64 - leading_zeros;
                simulated_size += (2 + significant_bits) as usize;
            } else {
                let significant_bits = 64 - leading_zeros;
                simulated_size += (5 + significant_bits) as usize;
            }
        }

        simulated_size
    }

    /// Returns the number of bits written to the buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn reset(&mut self) {
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
