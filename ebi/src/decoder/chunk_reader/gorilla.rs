use std::io::Read;

use crate::{
    decoder::{self, query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    io::bit_read::BufferedBitReader,
};

use super::Reader;

type BitReader = BufferedBitReader<Vec<u8>>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GorillaReader {
    decoder: modified_tsz::GorillaDecoder<BitReader>,
    number_of_records: usize,
    decompressed: Option<Vec<f64>>,
}

impl GorillaReader {
    pub fn new<T: FileMetadataLike, R: Read>(
        handle: &GeneralChunkHandle<T>,
        mut reader: R,
    ) -> Self {
        let number_of_records = handle.number_of_records() as usize;
        let chunk_size = handle.chunk_size() as usize;
        let mut chunk_in_memory = vec![0; chunk_size];
        reader.read_exact(&mut chunk_in_memory).unwrap();
        let r = BufferedBitReader::new(chunk_in_memory);
        let decoder = modified_tsz::GorillaDecoder::new(r, Some(number_of_records));

        Self {
            decoder,
            number_of_records,
            decompressed: None,
        }
    }
}

pub type GorillaIterator<'a> = &'a mut modified_tsz::GorillaDecoder<BitReader>;

impl Reader for GorillaReader {
    type NativeHeader = ();
    type DecompressIterator<'a> = GorillaIterator<'a> where Self: 'a;

    fn decompress(&mut self) -> decoder::Result<&[f64]> {
        if self.decompressed.is_some() {
            return Ok(self.decompressed.as_ref().unwrap());
        }

        let mut buf = Vec::with_capacity(self.number_of_records);
        while let Some(value) = self.decoder.next()? {
            buf.push(value);
        }

        self.decompressed = Some(buf);

        Ok(self.decompressed.as_ref().unwrap())
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
    }

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        &mut self.decoder
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed = Some(data);

        self.decompressed.as_ref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.decompressed.as_deref()
    }
}

impl QueryExecutor for GorillaReader {}

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
    pub struct GorillaDecoder<T: BitRead2> {
        /// current float value as bits
        value_bits: u64,

        /// number_of_records is the number of records to decode
        number_of_records: Option<usize>,

        /// number_of_records_read is the number of records decoded
        number_of_records_read: usize,

        leading_zeroes: u8,  // leading zeroes
        trailing_zeroes: u8, // trailing zeroes

        first: bool, // will next DataPoint be the first DataPoint decoded
        done: bool,

        r: T,
    }

    impl<T> GorillaDecoder<T>
    where
        T: BitRead2,
    {
        /// new creates a new StdDecoder which will read bytes from r
        pub fn new(r: T, number_of_records: Option<usize>) -> Self {
            Self {
                value_bits: 0,
                number_of_records,
                number_of_records_read: 0,
                leading_zeroes: 0,
                trailing_zeroes: 0,
                first: true,
                done: false,
                r,
            }
        }

        fn read_first_value(&mut self) -> decoder::Result<u64> {
            self.r
                .read_bits(64)
                .map(|bits| {
                    self.value_bits = bits;
                    self.value_bits
                })
                .ok_or(DecoderError::UnexpectedEndOfChunk.into())
        }

        fn read_next_value(&mut self) -> decoder::Result<u64> {
            let contol_bit = self
                .r
                .read_bit()
                .ok_or(DecoderError::UnexpectedEndOfChunk)?;

            if !contol_bit
            /* 0 */
            {
                return Ok(self.value_bits);
            }

            let zeroes_bit = self
                .r
                .read_bit()
                .ok_or(DecoderError::UnexpectedEndOfChunk)?;

            if zeroes_bit
            /* 1 */
            {
                self.leading_zeroes = self
                    .r
                    .read_bits(5)
                    .map(|n| n as u8)
                    .ok_or(DecoderError::UnexpectedEndOfChunk)?;
                let significant_digits = self
                    .r
                    .read_bits(6)
                    .map(|n| (n + 1) as u8)
                    .ok_or(DecoderError::UnexpectedEndOfChunk)?;
                self.trailing_zeroes = 64 - self.leading_zeroes - significant_digits;
            }

            let size = 64 - self.leading_zeroes - self.trailing_zeroes;
            self.r
                .read_bits(size)
                .map(|bits| {
                    self.value_bits ^= bits << self.trailing_zeroes;
                    self.value_bits
                })
                .ok_or(DecoderError::UnexpectedEndOfChunk.into())
        }
    }

    impl<'a, T: BitRead2> Iterator for &'a mut GorillaDecoder<T> {
        type Item = decoder::Result<f64>;

        fn next(&mut self) -> Option<Self::Item> {
            GorillaDecoder::next(self).transpose()
        }
    }

    impl<T> GorillaDecoder<T>
    where
        T: BitRead2,
    {
        pub fn next(&mut self) -> decoder::Result<Option<f64>> {
            if self.done
                || self.number_of_records.map_or(false, |number_of_records| {
                    self.number_of_records_read >= number_of_records
                })
            {
                debug_assert!(self.number_of_records == Some(self.number_of_records_read));
                return Ok(None);
            }

            if self.r.peak_bits(1).is_none() {
                self.done = true;
                return Ok(None);
            }

            let value_bits = if self.first {
                self.first = false;
                self.read_first_value()?
            } else {
                self.read_next_value()?
            };

            let value = f64::from_bits(value_bits);

            self.number_of_records_read += 1;

            Ok(Some(value))
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::io::bit_read::BufferedBitReader;

        #[test]
        fn gorilla() {
            let encoded: [u8; 15] = [
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

            let b = BufferedBitReader::new(&encoded[..]);
            let mut decoder = super::GorillaDecoder::new(b, Some(2));

            assert_eq!(decoder.next().unwrap(), Some(12.0));
            assert_eq!(decoder.next().unwrap(), Some(24.0));
            assert_eq!(decoder.next().unwrap(), None);
        }
    }
}
