use super::{
    chimp::ChimpDecoder,
    general_xor::{GeneralXorDecompressIterator, GeneralXorReader, XorDecoder},
};
use crate::{compression_common, io::bit_read::BitRead};
use std::{io, mem};

type ElfOnTReader<R, T> = GeneralXorReader<R, ElfDecoderWrapper<T>>;
type ElfOnTDecompressIterator<'a, R, T> = GeneralXorDecompressIterator<'a, R, ElfDecoderWrapper<T>>;

macro_rules! declare_elf_on_t_compressor {
    ($mod_name:ident, $decoder_ty:ty) => {
        pub mod $mod_name {
            pub type ElfReader<R> = super::ElfOnTReader<R, $decoder_ty>;
            pub type ElfDecompressIterator<'a, R> =
                super::ElfOnTDecompressIterator<'a, R, $decoder_ty>;
        }
    };
}

declare_elf_on_t_compressor!(chimp, super::ChimpDecoder);

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

    fn next_value<R: BitRead>(&mut self, r: &mut R) -> io::Result<Option<f64>> {
        let v;

        if !r.read_bit()?
        /* 0 */
        {
            v = self.recover_v_by_beta_star(r)?; // case 0
        } else if !r.read_bit()?
        /* 0 */
        {
            v = self.xor_decompress(r)?; // case 10
        } else {
            self.last_beta_star = unsafe { mem::transmute::<u32, i32>(r.read_bits(4)? as u32) }; // case 11
            v = self.recover_v_by_beta_star(r)?;
        }
        Ok(v)
    }

    fn recover_v_by_beta_star<R: BitRead>(&mut self, r: &mut R) -> io::Result<Option<f64>> {
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

    fn xor_decompress<R: BitRead>(&mut self, r: &mut R) -> io::Result<Option<f64>> {
        self.xor_decoder.decompress_float(r)
    }
}

impl<T: XorDecoder> Default for ElfDecoderWrapper<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: XorDecoder> XorDecoder for ElfDecoderWrapper<T> {
    fn decompress_float<R: BitRead>(&mut self, mut r: R) -> io::Result<Option<f64>> {
        self.next_value(&mut r)
    }

    fn reset(&mut self) {
        self.last_beta_star = i32::MAX;
        self.xor_decoder.reset();
    }
}
