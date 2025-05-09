use super::{
    chimp::ChimpDecoder,
    general_xor::{GeneralXorDecompressIterator, GeneralXorReader, XorDecoder},
};
use crate::{
    compression_common,
    decoder::{self, error::DecoderError},
    io::bit_read::BitRead2,
};
use std::mem;

type ElfOnTReader<T> = GeneralXorReader<ElfDecoderWrapper<T>>;
type ElfOnTDecompressIterator<'a, T> = GeneralXorDecompressIterator<'a, ElfDecoderWrapper<T>>;

macro_rules! declare_elf_on_t_compressor {
    ($mod_name:ident, $decoder_ty:ty) => {
        pub mod $mod_name {
            pub type ElfReader = super::ElfOnTReader<$decoder_ty>;
            pub type ElfDecompressIterator<'a> = super::ElfOnTDecompressIterator<'a, $decoder_ty>;
        }
    };
}

pub type ElfReader = ElfOnTReader<ElfXorDecoder>;
pub type ElfDecompressIterator<'a> = ElfOnTDecompressIterator<'a, ElfXorDecoder>;

declare_elf_on_t_compressor!(on_chimp, super::ChimpDecoder);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ElfDecoderWrapper<T: XorDecoder> {
    last_beta_star: i32,
    xor_decoder: T,
}

impl<T: XorDecoder> ElfDecoderWrapper<T> {
    pub fn new() -> Self {
        Self {
            last_beta_star: i32::MAX,
            xor_decoder: T::default(),
        }
    }

    fn next_value<R: BitRead2>(&mut self, r: &mut R) -> decoder::Result<Option<f64>> {
        let v;

        if !r.read_bit().ok_or(DecoderError::UnexpectedEndOfChunk)?
        /* 0 */
        {
            v = self.recover_v_by_beta_star(r)?; // case 0
        } else if !r.read_bit().ok_or(DecoderError::UnexpectedEndOfChunk)?
        /* 0 */
        {
            v = self.xor_decompress(r)?; // case 10
        } else {
            self.last_beta_star = unsafe {
                mem::transmute::<u32, i32>(
                    r.read_bits(4).ok_or(DecoderError::UnexpectedEndOfChunk)? as u32,
                )
            }; // case 11
            v = self.recover_v_by_beta_star(r)?;
        }
        Ok(v)
    }

    fn recover_v_by_beta_star<R: BitRead2>(&mut self, r: &mut R) -> decoder::Result<Option<f64>> {
        let Some(v_prime) = self.xor_decompress(r)? else {
            return Ok(None);
        };
        let sp = compression_common::elf::get_sp(v_prime.abs());
        let v = if self.last_beta_star == 0 {
            let mut v = compression_common::elf::get_10i_n((-sp - 1) as u32);
            if v_prime < 0.0 {
                v = -v;
            }
            v
        } else {
            let alpha = self.last_beta_star - sp - 1;
            compression_common::elf::round_up(v_prime, alpha as u32)
        };
        Ok(Some(v))
    }

    fn xor_decompress<R: BitRead2>(&mut self, r: &mut R) -> decoder::Result<Option<f64>> {
        self.xor_decoder.decompress_float(r)
    }
}

impl<T: XorDecoder> Default for ElfDecoderWrapper<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: XorDecoder> XorDecoder for ElfDecoderWrapper<T> {
    fn decompress_float<R: BitRead2>(&mut self, mut r: R) -> decoder::Result<Option<f64>> {
        self.next_value(&mut r)
    }

    fn reset(&mut self) {
        self.last_beta_star = i32::MAX;
        self.xor_decoder.reset();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElfXorDecoder {
    stored_val: u64,
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    first: bool,
    end_of_stream: bool,
}

impl ElfXorDecoder {
    const END_SIGN: u64 = f64::NAN.to_bits();
    const LEADING_REPRESENTATION: [u8; 8] = [0, 8, 12, 16, 18, 20, 22, 24];

    pub fn new() -> Self {
        Self {
            stored_val: 0,
            stored_leading_zeros: u32::MAX,
            stored_trailing_zeros: u32::MAX,
            first: true,
            end_of_stream: false,
        }
    }

    pub fn read_first<R: BitRead2>(&mut self, r: &mut R) -> decoder::Result<Option<f64>> {
        self.first = false;
        let trailing_zeros = r.read_bits(7).ok_or(DecoderError::UnexpectedEndOfChunk)? as u32;
        if trailing_zeros < 64 {
            self.stored_val = ((r
                .read_bits(63 - trailing_zeros as u8)
                .ok_or(DecoderError::UnexpectedEndOfChunk)?
                << 1)
                | 1)
                << trailing_zeros;
        } else {
            self.stored_val = 0;
        }
        if self.stored_val == Self::END_SIGN {
            self.end_of_stream = true;
            return Ok(None);
        }

        Ok(Some(f64::from_bits(self.stored_val)))
    }

    pub fn decompress_float_inner<R: BitRead2>(
        &mut self,
        r: &mut R,
    ) -> decoder::Result<Option<f64>> {
        let flag = r.read_bits(2).ok_or(DecoderError::UnexpectedEndOfChunk)? as u32;
        match flag {
            3 => {
                // case 11
                let lead_and_center =
                    r.read_bits(9).ok_or(DecoderError::UnexpectedEndOfChunk)? as u32;
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[(lead_and_center >> 6) as usize] as u32;
                let mut center_bits = lead_and_center & 0x3f;
                if center_bits == 0 {
                    center_bits = 64;
                }
                self.stored_trailing_zeros = 64 - self.stored_leading_zeros - center_bits;
                let mut value = ((r
                    .read_bits((center_bits - 1) as u8)
                    .ok_or(DecoderError::UnexpectedEndOfChunk)?
                    << 1)
                    | 1)
                    << self.stored_trailing_zeros;
                value ^= self.stored_val;
                if value == Self::END_SIGN {
                    self.end_of_stream = true;
                    return Ok(None);
                } else {
                    self.stored_val = value;
                }
            }
            2 => {
                // case 10
                let lead_and_center =
                    r.read_bits(7).ok_or(DecoderError::UnexpectedEndOfChunk)? as u32;
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[(lead_and_center >> 4) as usize] as u32;
                let mut center_bits = lead_and_center & 0xf;
                if center_bits == 0 {
                    center_bits = 16;
                }
                self.stored_trailing_zeros = 64 - self.stored_leading_zeros - center_bits;
                let mut value = if center_bits == 1 {
                    1 << self.stored_trailing_zeros
                } else {
                    ((r.read_bits((center_bits - 1) as u8)
                        .ok_or(DecoderError::UnexpectedEndOfChunk)?
                        << 1)
                        | 1)
                        << self.stored_trailing_zeros
                };
                value ^= self.stored_val;
                if value == Self::END_SIGN {
                    self.end_of_stream = true;
                    return Ok(None);
                } else {
                    self.stored_val = value;
                }
            }
            1 => {
                // case 01, we do nothing, the same value as before
            }
            _ => {
                // case 00
                let center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros;
                let mut value = r
                    .read_bits(center_bits as u8)
                    .ok_or(DecoderError::UnexpectedEndOfChunk)?
                    << self.stored_trailing_zeros;
                value ^= self.stored_val;
                if value == Self::END_SIGN {
                    self.end_of_stream = true;
                    return Ok(None);
                } else {
                    self.stored_val = value;
                }
            }
        }

        Ok(Some(f64::from_bits(self.stored_val)))
    }
}

impl Default for ElfXorDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl XorDecoder for ElfXorDecoder {
    fn decompress_float<R: BitRead2>(&mut self, mut r: R) -> decoder::Result<Option<f64>> {
        if self.end_of_stream {
            return Ok(None);
        }

        if self.first {
            self.read_first(&mut r)
        } else {
            self.decompress_float_inner(&mut r)
        }
    }

    fn reset(&mut self) {
        self.stored_val = 0;
        self.stored_leading_zeros = u32::MAX;
        self.stored_trailing_zeros = u32::MAX;
        self.first = true;
        self.end_of_stream = false;
    }
}
