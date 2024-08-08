use std::mem;

use tsz::{stream::Write as _, Bit};

use crate::{compression_common::elf, io::buffered_bit_writer::BufferedWriterExt};

use super::{
    chimp::ChimpEncoder,
    general_xor::{
        GeneralXorCompressor, GeneralXorCompressorConfig, GeneralXorCompressorConfigBuilder,
        XorEncoder,
    },
};

type ElfOnTCompressor<T> = GeneralXorCompressor<ElfEncoderWrapper<T>>;
type ElfOnTCompressorConfig<T> = GeneralXorCompressorConfig<ElfEncoderWrapper<T>>;
type ElfOnTCompressorConfigBuilder<T> = GeneralXorCompressorConfigBuilder<ElfEncoderWrapper<T>>;

macro_rules! declare_elf_on_t_compressor {
    ($mod_name:ident, $encoder_ty:ty) => {
        pub mod $mod_name {
            pub type ElfCompressor = super::ElfOnTCompressor<$encoder_ty>;
            pub type ElfCompressorConfig = super::ElfOnTCompressorConfig<$encoder_ty>;
            pub type ElfCompressorConfigBuilder = super::ElfOnTCompressorConfigBuilder<$encoder_ty>;
        }
    };
}

declare_elf_on_t_compressor!(chimp, super::ChimpEncoder);

impl<T: XorEncoder> XorEncoder for ElfEncoderWrapper<T> {
    fn compress_float(&mut self, w: &mut BufferedWriterExt, bits: u64) -> usize {
        let preserved_size = self.size();
        self.add_value(w, f64::from_bits(bits));

        self.size() - preserved_size
    }

    fn reset(&mut self) {
        self.size = 0;
        self.last_beta_star = i32::MAX;
        self.xor_encoder.reset();
    }

    fn close(&mut self, w: &mut BufferedWriterExt) {
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

    pub fn add_value(&mut self, w: &mut BufferedWriterExt, v: f64) {
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
                    w.write_bit(Bit::Zero);
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
