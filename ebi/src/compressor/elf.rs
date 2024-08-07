use std::mem;

use tsz::{stream::Write as _, Bit};

use crate::{compression_common::elf, io::buffered_bit_writer::BufferedWriterExt};

pub trait XorEncoder {
    /// Compress the passed floats,
    /// Returns the increased size by bits
    fn compress_float(&mut self, w: &mut BufferedWriterExt, bits: u64) -> usize;
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ElfEncoderWrapper<T: XorEncoder> {
    size: usize,
    last_beta_star: i32,
    w: BufferedWriterExt,
    xor_encoder: T,
}

impl<T: XorEncoder> ElfEncoderWrapper<T> {
    pub fn new(w: BufferedWriterExt, xor_encoder: T) -> Self {
        Self {
            size: 0,
            last_beta_star: i32::MAX,
            w,
            xor_encoder,
        }
    }

    pub fn add_value(&mut self, v: f64) {
        let v_bits = v.to_bits();

        let v_prime_u64 = if v == 0.0 || v.is_infinite() {
            // case 10
            self.size += 2;
            self.w.write_bits(2, 2);
            v_bits
        } else if v.is_nan() {
            // case 10
            self.size += 2;
            self.w.write_bits(2, 2);
            f64::NAN.to_bits()
        } else {
            // C1: v is a normal or subnormal
            let (alpha, beta_star) = elf::get_alpha_and_beta_star(v, self.last_beta_star);
            let e = (v_bits >> 52) & 0x7ff;
            let g_alpha = elf::get_f_alpha(alpha) + e as u32 - 1023;
            let erase_bits = 52 - g_alpha;
            let mask = u64::MAX << erase_bits;
            let delta = (!mask) & v_bits;

            if delta != 0 && erase_bits > 4 {
                // C2
                if beta_star == self.last_beta_star {
                    // case 0
                    self.size += 1;
                    self.w.write_bit(Bit::Zero);
                } else {
                    // case 11, 2 + 4 = 6
                    self.size += 6;
                    let bits: u32 = unsafe { mem::transmute(beta_star | 0x30) };
                    self.w.write_bits(bits as u64, 6);
                    self.last_beta_star = beta_star;
                }
                mask & v_bits
            } else {
                // case 10
                self.size += 2;
                self.w.write_bits(2, 2);
                v_bits
            }
        };

        self.xor_encoder.compress_float(&mut self.w, v_prime_u64);
    }

    pub fn size(&self) -> usize {
        self.size
    }
}
