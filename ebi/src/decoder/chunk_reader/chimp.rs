use std::io::{self};

use crate::io::bit_read::BitRead;

use super::general_xor::{GeneralXorDecompressIterator, GeneralXorReader, XorDecoder};

pub type ChimpReader<R> = GeneralXorReader<R, ChimpDecoder>;
pub type ChimpDecompressIterator<'a, R> = GeneralXorDecompressIterator<'a, R, ChimpDecoder>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpDecoder {
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    stored_val: u64,
    first: bool,
    end_of_stream: bool,
}

impl XorDecoder for ChimpDecoder {
    fn decompress_float<R: BitRead>(&mut self, r: R) -> io::Result<Option<f64>> {
        self.read_value(r)
    }

    fn reset(&mut self) {
        ChimpDecoder::reset(self)
    }
}

impl Default for ChimpDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChimpDecoder {
    const NAN_LONG: u64 = 0x7ff8000000000000;
    const LEADING_REPRESENTATION: [u8; 8] = [0, 8, 12, 16, 18, 20, 22, 24];

    pub fn new() -> Self {
        ChimpDecoder {
            stored_leading_zeros: u32::MAX,
            stored_trailing_zeros: 0,
            stored_val: 0,
            first: true,
            end_of_stream: false,
        }
    }

    pub fn get_values<R: BitRead>(&mut self, mut r: R) -> io::Result<Vec<f64>> {
        let mut values = Vec::new();
        while let Some(value) = self.read_value(&mut r)? {
            values.push(value);
        }
        Ok(values)
    }

    pub fn read_value<R: BitRead>(&mut self, mut r: R) -> io::Result<Option<f64>> {
        if self.end_of_stream {
            return Ok(None);
        }

        if self.first {
            self.first = false;
            self.stored_val = r.read_bits(64)?;
            if self.stored_val == Self::NAN_LONG {
                self.end_of_stream = true;
                return Ok(None);
            }
        } else {
            self.next_value(r)?;
        }

        if self.end_of_stream {
            return Ok(None);
        }

        Ok(Some(f64::from_bits(self.stored_val)))
    }

    fn next_value<R: BitRead>(&mut self, mut r: R) -> io::Result<()> {
        let flag = r.read_bits(2)? as u8;
        match flag {
            3 => {
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[r.read_bits(3)? as usize] as u32;
                let significant_bits = 64 - self.stored_leading_zeros;
                let value = r.read_bits(significant_bits as u8)?;
                self.stored_val ^= value;
            }
            2 => {
                let significant_bits = 64 - self.stored_leading_zeros;
                let value = r.read_bits(significant_bits as u8)?;
                self.stored_val ^= value;
            }
            1 => {
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[r.read_bits(3)? as usize] as u32;
                let mut significant_bits = r.read_bits(6)? as u32;
                if significant_bits == 0 {
                    significant_bits = 64;
                }
                self.stored_trailing_zeros = 64 - significant_bits - self.stored_leading_zeros;
                let value = r.read_bits(
                    (64 - self.stored_leading_zeros - self.stored_trailing_zeros) as u8,
                )?;
                self.stored_val ^= value << self.stored_trailing_zeros;
            }
            _ => {}
        }

        if self.stored_val == Self::NAN_LONG {
            self.end_of_stream = true;
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
