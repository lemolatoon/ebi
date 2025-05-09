use std::mem;

use crate::{
    compression_common::elf,
    format::deserialize,
    io::bit_write::{BitWrite, BufferedBitWriter},
};

use super::{
    chimp::ChimpEncoder,
    general_xor::{
        GeneralXorCompressor, GeneralXorCompressorConfig, GeneralXorCompressorConfigBuilder,
        XorEncoder,
    },
    Capacity,
};

type ElfOnTCompressor<T> = GeneralXorCompressor<BufferedBitWriter, ElfEncoderWrapper<T>>;
type ElfOnTCompressorConfig<T> = GeneralXorCompressorConfig<ElfEncoderWrapper<T>>;
type ElfOnTCompressorConfigBuilder<T> = GeneralXorCompressorConfigBuilder<ElfEncoderWrapper<T>>;

pub type ElfCompressor = ElfOnTCompressor<ElfXorEncoder>;
pub type ElfCompressorConfig = ElfOnTCompressorConfig<ElfXorEncoder>;
pub type ElfCompressorConfigBuilder = ElfOnTCompressorConfigBuilder<ElfXorEncoder>;

macro_rules! declare_elf_on_t_compressor {
    ($mod_name:ident, $encoder_ty:ty) => {
        pub mod $mod_name {
            pub type ElfCompressor = super::ElfOnTCompressor<$encoder_ty>;
            pub type ElfCompressorConfig = super::ElfOnTCompressorConfig<$encoder_ty>;
            pub type ElfCompressorConfigBuilder = super::ElfOnTCompressorConfigBuilder<$encoder_ty>;
        }
    };
}

declare_elf_on_t_compressor!(on_chimp, super::ChimpEncoder);
deserialize::impl_from_le_bytes!(
    on_chimp::ElfCompressorConfig,
    elf_on_chimp,
    (capacity, Capacity)
);
deserialize::impl_from_le_bytes!(ElfCompressorConfig, elf, (capacity, Capacity));

impl<T: XorEncoder> XorEncoder for ElfEncoderWrapper<T> {
    fn compress_float<W: BitWrite>(&mut self, w: W, bits: u64) -> usize {
        let preserved_size = self.size();
        self.add_value(w, f64::from_bits(bits));

        self.size() - preserved_size
    }

    fn reset(&mut self) {
        self.size = 0;
        self.last_beta_star = i32::MAX;
        self.xor_encoder.reset();
    }

    fn close<W: BitWrite>(&mut self, mut w: W) {
        self.size += 2;
        // case 10
        w.write_bits(2, 2);
        self.xor_encoder.close(w);
    }

    fn simulate_close(&self) -> usize {
        2 + self.xor_encoder.simulate_close()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ElfEncoderWrapper<T: XorEncoder> {
    size: usize,
    last_beta_star: i32,
    xor_encoder: T,
}

impl<T: XorEncoder> Default for ElfEncoderWrapper<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: XorEncoder> ElfEncoderWrapper<T> {
    pub fn new() -> Self {
        Self {
            size: 0,
            last_beta_star: i32::MAX,
            xor_encoder: T::default(),
        }
    }

    pub fn add_value<W: BitWrite>(&mut self, mut w: W, v: f64) {
        let v_bits = v.to_bits();

        let v_prime_u64 = if v == 0.0 || v.is_infinite() {
            // case 10
            self.size += 2;
            w.write_bits(2, 2);
            v_bits
        } else if v.is_nan() {
            // case 10
            self.size += 2;
            w.write_bits(2, 2);
            f64::NAN.to_bits()
        } else {
            // C1: v is a normal or subnormal
            let (alpha, beta_star) = elf::get_alpha_and_beta_star(v, self.last_beta_star);
            let mut mask = u64::MAX;
            let mut delta = 0;
            let mut erase_bits = 0;

            if beta_star < 16 {
                let e = (v_bits >> 52) & 0x7ff;
                let g_alpha = elf::get_f_alpha(alpha) + e as u32 - 1023;
                erase_bits = 52 - g_alpha as i32;
                if erase_bits > 4 {
                    mask = u64::MAX << erase_bits;
                    delta = (!mask) & v_bits;
                } else {
                    mask = u64::MAX;
                    delta = 0;
                }
            }

            if beta_star < 16 && erase_bits > 4 && delta != 0 {
                // C2
                if beta_star == self.last_beta_star {
                    // case 0
                    self.size += 1;
                    w.write_bit(false);
                } else {
                    // case 11, 2 + 4 = 6
                    self.size += 6;
                    let bits: u32 = unsafe { mem::transmute(beta_star | 0x30) };
                    w.write_bits(bits as u64, 6);
                    self.last_beta_star = beta_star;
                }
                mask & v_bits
            } else {
                // case 10
                self.size += 2;
                w.write_bits(2, 2);
                v_bits
            }
        };

        self.xor_encoder.compress_float(w, v_prime_u64);
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ElfXorEncoder {
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    stored_val: u64,
    first: bool,
    size: usize,
}

impl ElfXorEncoder {
    const END_SIGN: u64 = f64::NAN.to_bits();
    const LEADING_REPRESENTATION: [u8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7,
    ];

    const LEADING_ROUND: [u8; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];

    pub fn new() -> Self {
        Self {
            stored_leading_zeros: u32::MAX,
            stored_trailing_zeros: u32::MAX,
            stored_val: 0,
            first: true,
            size: 0,
        }
    }

    /// Compress the passed float,
    /// Returns the increased size by bits
    ///
    /// This method should called at the first float of the entire floats
    /// The 2nd and later, use [`ElfXorEncoder::compress_float_inner`] instead
    fn write_first<W: BitWrite>(&mut self, mut w: W, bits: u64) -> usize {
        self.first = false;
        self.stored_val = bits;
        let trailing_zeros = bits.trailing_zeros();
        w.write_bits(trailing_zeros as u64, 7);
        if trailing_zeros < 64 {
            w.write_bits(bits >> (trailing_zeros + 1), 63 - trailing_zeros);
            self.size += 70 - trailing_zeros as usize;
            70 - trailing_zeros as usize
        } else {
            self.size += 7;
            7
        }
    }

    /// Compress the passed float,
    /// Returns the increased size by bits
    ///
    /// For the first float, use [`ElfXorEncoder::write_first`] instead
    fn compress_float_inner<W: BitWrite>(&mut self, mut w: W, bits: u64) -> usize {
        let mut this_size = 0;
        let xor = self.stored_val ^ bits;

        if xor == 0 {
            // case 01
            w.write_bits(1, 2);
            self.size += 2;
            this_size += 2;
        } else {
            let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;
            let trailing_zeros = xor.trailing_zeros();

            if leading_zeros == self.stored_leading_zeros
                && trailing_zeros >= self.stored_trailing_zeros
            {
                // case 00
                let center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros;
                let len = 2 + center_bits as usize;
                if len > 64 {
                    w.write_bits(0, 2);
                    w.write_bits(xor >> self.stored_trailing_zeros, center_bits);
                } else {
                    w.write_bits(xor >> self.stored_trailing_zeros, len as u32);
                }
                self.size += len;
                this_size += len;
            } else {
                self.stored_leading_zeros = leading_zeros;
                self.stored_trailing_zeros = trailing_zeros;
                let center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros;

                let leading_repr =
                    Self::LEADING_REPRESENTATION[self.stored_leading_zeros as usize] as u32;
                if center_bits <= 16 {
                    // case 10
                    w.write_bits(
                        ((((0x2 << 3) | leading_repr) << 4) | (center_bits & 0xf)) as u64,
                        9,
                    );
                    w.write_bits(xor >> (self.stored_trailing_zeros + 1), center_bits - 1);
                    self.size += 8 + center_bits as usize;
                    this_size += 8 + center_bits as usize;
                } else {
                    // case 11
                    w.write_bits(
                        ((((0x3 << 3) | leading_repr) << 6) | (center_bits & 0x3f)) as u64,
                        11,
                    );
                    w.write_bits(xor >> (self.stored_trailing_zeros + 1), center_bits - 1);
                    self.size += 10 + center_bits as usize;
                    this_size += 10 + center_bits as usize;
                }
            }

            self.stored_val = bits;
        }

        this_size
    }

    /// Simulates compressing a float value to the encoder
    /// and returns the incresed size in bits.
    ///
    /// This function calculates the size of the encoded float
    /// without actually modifying the encoder's state. It is useful
    /// for estimating the impact on size before performing the actual addition.
    ///
    /// # Arguments
    ///
    /// * `value` - The floating point value to be added to the encoder.
    ///
    /// # Returns
    ///
    /// * `usize` - The size of the encoded float in bits
    ///
    /// # Example
    /// ```
    /// use ebi::compressor::elf::ElfXorEncoder;
    /// use crate::ebi::compressor::general_xor::XorEncoder;
    /// use ebi::io::bit_write::BufferedBitWriter;
    /// let mut encoder = ElfXorEncoder::new();
    /// let mut w = BufferedBitWriter::new();
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_compress_float(42.0f64.to_bits());
    ///
    /// // Actual addition of the value
    /// let actual_size = encoder.compress_float(&mut w, 42.0f64.to_bits());
    /// assert_eq!(simulated_size, actual_size);
    ///
    /// encoder.compress_float(&mut w, 44.0f64.to_bits());
    ///
    /// // Simulate adding a value
    /// let simulated_size = encoder.simulate_compress_float(22.0f64.to_bits());
    /// // Actual addition of the value
    /// let actual_size = encoder.compress_float(&mut w, 22.0f64.to_bits());
    /// assert_eq!(simulated_size, actual_size);
    ///
    /// let simulated_size = encoder.simulate_compress_float(f64::NAN.to_bits());
    /// let actual_size = encoder.compress_float(&mut w, f64::NAN.to_bits());
    /// assert_eq!(simulated_size, actual_size);
    /// ```
    pub fn simulate_compress_float(&self, bits: u64) -> usize {
        let mut size: usize = 0;
        if self.first {
            let trailing_zeros = bits.trailing_zeros();
            if trailing_zeros < 64 {
                size += 7 + 63 - trailing_zeros as usize;
            } else {
                size += 7;
            }
        } else {
            let xor = self.stored_val ^ bits;

            if xor == 0 {
                size += 2;
            } else {
                let leading_zeros = Self::LEADING_ROUND[xor.leading_zeros() as usize] as u32;
                let trailing_zeros = xor.trailing_zeros();

                if leading_zeros == self.stored_leading_zeros
                    && trailing_zeros >= self.stored_trailing_zeros
                {
                    let center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros;
                    let len = 2 + center_bits as usize;
                    size += len;
                } else {
                    let center_bits = 64 - leading_zeros - trailing_zeros;

                    if center_bits <= 16 {
                        size += 8 + center_bits as usize;
                    } else {
                        size += 10 + center_bits as usize;
                    }
                }
            }
        }

        size
    }
}

impl Default for ElfXorEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl XorEncoder for ElfXorEncoder {
    fn compress_float<W: BitWrite>(&mut self, w: W, bits: u64) -> usize {
        if self.first {
            self.write_first(w, bits)
        } else {
            self.compress_float_inner(w, bits)
        }
    }

    fn close<W: BitWrite>(&mut self, mut w: W) {
        self.compress_float(&mut w, Self::END_SIGN);
        w.write_bit(false);
    }

    fn simulate_close(&self) -> usize {
        self.simulate_compress_float(Self::END_SIGN) + 1
    }

    fn reset(&mut self) {
        self.stored_leading_zeros = u32::MAX;
        self.stored_trailing_zeros = u32::MAX;
        self.stored_val = 0;
        self.first = true;
        self.size = 0;
    }
}
