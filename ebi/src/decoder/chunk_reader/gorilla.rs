use crate::{
    decoder::{self},
    io::bit_read::BitRead2,
};

use super::general_xor::{GeneralXorDecompressIterator, GeneralXorReader, XorDecoder};

pub type GorillaReader = GeneralXorReader<modified_tsz::GorillaDecoder>;
pub type GorillaDecompressIterator<'a> =
    GeneralXorDecompressIterator<'a, modified_tsz::GorillaDecoder>;

impl XorDecoder for modified_tsz::GorillaDecoder {
    fn decompress_float<R: BitRead2>(&mut self, r: R) -> decoder::Result<Option<f64>> {
        self.next(r)
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

mod modified_tsz {
    //! This original implementation is from tsz crate. It is modified to just use the floating point values compression.
    //! Original tsz crate is licensed under MIT.
    //! Original LICENSE is below:
    //!
    //! Copyright (c) 2016 Jerome Froelich
    //! Permission is hereby granted, free of charge, to any person obtaining a copy
    //! of this software and associated documentation files (the "Software"), to deal
    //! in the Software without restriction, including without limitation the rights
    //! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    //! copies of the Software, and to permit persons to whom the Software is
    //! furnished to do so, subject to the following conditions:
    //!
    //! The above copyright notice and this permission notice shall be included in all
    //! copies or substantial portions of the Software.
    //!
    //! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    //! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    //! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    //! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    //! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    //! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    //! SOFTWARE.
    //!

    // END_MARKER relies on the fact that when we encode the delta of delta for a number that requires
    // more than 12 bits we write four control bits 1111 followed by the 32 bits of the value. Since
    // encoding assumes the value is greater than 12 bits, we can store the value 0 to signal the end
    // of the stream

    use std::fmt::Debug;

    use crate::{
        decoder::{self, error::DecoderError},
        io::bit_read::BitRead2,
    };

    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    pub struct GorillaDecoder {
        /// current float value as bits
        value_bits: u64,

        leading_zeroes: u8,  // leading zeroes
        trailing_zeroes: u8, // trailing zeroes

        first: bool, // will next DataPoint be the first DataPoint decoded
        done: bool,
    }

    impl Default for GorillaDecoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GorillaDecoder {
        const NAN_BITS: u64 = const { f64::NAN.to_bits() };
        /// new creates a new StdDecoder which will read bytes from r
        pub fn new() -> Self {
            Self {
                value_bits: 0,
                leading_zeroes: 0,
                trailing_zeroes: 0,
                first: true,
                done: false,
            }
        }

        fn read_first_value(&mut self, r: &mut impl BitRead2) -> decoder::Result<u64> {
            #[allow(clippy::useless_conversion)]
            r.read_bits(64)
                .map(|bits| {
                    self.value_bits = bits;
                    self.value_bits
                })
                .ok_or(DecoderError::UnexpectedEndOfChunk.into())
        }

        fn read_next_value(&mut self, r: &mut impl BitRead2) -> decoder::Result<u64> {
            let contol_bit = r.read_bit().ok_or(DecoderError::UnexpectedEndOfChunk)?;

            if !contol_bit
            /* 0 */
            {
                return Ok(self.value_bits);
            }

            let zeroes_bit = r.read_bit().ok_or(DecoderError::UnexpectedEndOfChunk)?;

            if zeroes_bit
            /* 1 */
            {
                self.leading_zeroes = r
                    .read_bits(5)
                    .map(|n| n as u8)
                    .ok_or(DecoderError::UnexpectedEndOfChunk)?;
                let significant_digits = r
                    .read_bits(6)
                    .map(|n| (n + 1) as u8)
                    .ok_or(DecoderError::UnexpectedEndOfChunk)?;
                self.trailing_zeroes = 64 - self.leading_zeroes - significant_digits;
            }

            let size = 64 - self.leading_zeroes - self.trailing_zeroes;
            #[allow(clippy::useless_conversion)]
            r.read_bits(size)
                .map(|bits| {
                    self.value_bits ^= bits << self.trailing_zeroes;
                    self.value_bits
                })
                .ok_or(DecoderError::UnexpectedEndOfChunk.into())
        }
    }

    impl GorillaDecoder {
        pub fn next(&mut self, mut r: impl BitRead2) -> decoder::Result<Option<f64>> {
            if self.done {
                return Ok(None);
            }

            let value_bits = if self.first {
                self.first = false;
                self.read_first_value(&mut r)?
            } else {
                self.read_next_value(&mut r)?
            };

            if value_bits == Self::NAN_BITS {
                self.done = true;
                return Ok(None);
            }

            let value = f64::from_bits(value_bits);

            Ok(Some(value))
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{
            compressor::{general_xor::XorEncoder, gorilla::modified_tsz::GorillaFloatEncoder},
            decoder::chunk_reader::general_xor::XorDecoder,
            io::{
                bit_read::BufferedBitReader,
                bit_write::{BitWrite as _, BufferedBitWriter},
            },
        };

        #[test]
        fn gorilla() {
            let _encoded: [u8; 15] = [
                0x40,
                0x28,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00, /* 12.0 as double */
                #[allow(clippy::unusual_byte_groupings)]
                0b11__0_1011__0, /* control bit 11, # of leading zeros (11 (5 bits)) */
                #[allow(clippy::unusual_byte_groupings)]
                0b0_0000__1__11, /* # of meaning bits (1 (6 bits)), actual meaning bits 1, END_MARKER starts (2 bits) */
                #[allow(clippy::unusual_byte_groupings)]
                0b11__00_0000,
                0,
                0,
                0,
                0, /* END_MARKER */
            ];

            let mut encoder = GorillaFloatEncoder::new();
            let mut bitwriter = BufferedBitWriter::new();
            encoder.compress_float(&mut bitwriter, 12.0f64.to_bits());
            encoder.compress_float(&mut bitwriter, 24.0f64.to_bits());
            encoder.close(&mut bitwriter);
            let encoded = bitwriter.as_slice();
            let mut b = BufferedBitReader::new(encoded);
            let mut decoder = super::GorillaDecoder::new();

            assert_eq!(dbg!(decoder.decompress_float(&mut b).unwrap()), Some(12.0));
            assert_eq!(dbg!(decoder.decompress_float(&mut b).unwrap()), Some(24.0));
            assert_eq!(dbg!(decoder.decompress_float(&mut b).unwrap()), None);
        }
    }
}
