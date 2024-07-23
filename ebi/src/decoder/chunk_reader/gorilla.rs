use crate::decoder::{FileMetadataLike, GeneralChunkHandle};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GorillaReader<'chunk> {
    decoder: modified_tsz::GorillaDecoder<modified_tsz::BufferRefReader<'chunk>>,
    number_of_records: usize,
    decompressed: Option<Vec<f64>>,
}

impl<'chunk> GorillaReader<'chunk> {
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, chunk: &'chunk [u8]) -> Self {
        let number_of_records = handle.number_of_records() as usize;
        let decoder = modified_tsz::GorillaDecoder::new(
            modified_tsz::BufferRefReader::new(chunk),
            Some(number_of_records),
        );

        Self {
            decoder,
            number_of_records,
            decompressed: None,
        }
    }
}

impl<'chunk> Reader for GorillaReader<'chunk> {
    type NativeHeader = ();

    fn decompress(&mut self) -> &[f64] {
        if self.decompressed.is_some() {
            return self.decompressed.as_ref().unwrap();
        }

        let mut buf = Vec::with_capacity(self.number_of_records);
        while let Ok(value) = self.decoder.next() {
            buf.push(value);
        }

        self.decompressed = Some(buf);

        self.decompressed.as_ref().unwrap()
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
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

    use tsz::{
        decode::Error,
        stream::{self, Read},
        Bit,
    };

    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    pub struct GorillaDecoder<T: Read> {
        /// current float value as bits
        value_bits: u64,

        /// number_of_records is the number of records to decode
        number_of_records: Option<usize>,

        /// number_of_records_read is the number of records decoded
        number_of_records_read: usize,

        leading_zeroes: u32,  // leading zeroes
        trailing_zeroes: u32, // trailing zeroes

        first: bool, // will next DataPoint be the first DataPoint decoded
        done: bool,

        r: T,
    }

    impl<T> GorillaDecoder<T>
    where
        T: Read,
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

        fn read_first_value(&mut self) -> Result<u64, Error> {
            self.r.read_bits(64).map_err(Error::Stream).map(|bits| {
                self.value_bits = bits;
                self.value_bits
            })
        }

        fn read_next_value(&mut self) -> Result<u64, Error> {
            let contol_bit = self.r.read_bit()?;

            if contol_bit == Bit::Zero {
                return Ok(self.value_bits);
            }

            let zeroes_bit = self.r.read_bit()?;

            if zeroes_bit == Bit::One {
                self.leading_zeroes = self.r.read_bits(5).map(|n| n as u32)?; // not 6, but 5
                let significant_digits = self.r.read_bits(6).map(|n| (n + 1) as u32)?;
                self.trailing_zeroes = 64 - self.leading_zeroes - significant_digits;
            }

            let size = 64 - self.leading_zeroes - self.trailing_zeroes;
            self.r.read_bits(size).map_err(Error::Stream).map(|bits| {
                self.value_bits ^= bits << self.trailing_zeroes;
                self.value_bits
            })
        }
    }

    impl<T> GorillaDecoder<T>
    where
        T: Read,
    {
        pub fn next(&mut self) -> Result<f64, Error> {
            if self.done
                || self.number_of_records.map_or(false, |number_of_records| {
                    self.number_of_records_read >= number_of_records
                })
            {
                debug_assert!(self.number_of_records == Some(self.number_of_records_read));
                return Err(Error::EndOfStream);
            }

            if self.r.peak_bits(1) == Err(tsz::stream::Error::EOF) {
                self.done = true;
                return Err(Error::EndOfStream);
            };

            let value_bits = if self.first {
                self.first = false;
                self.read_first_value()?
            } else {
                self.read_next_value()?
            };

            let value = f64::from_bits(value_bits);

            self.number_of_records_read += 1;

            Ok(value)
        }
    }

    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    pub struct BufferRefReader<'a> {
        bytes: &'a [u8], // internal buffer of bytes
        index: usize,    // index into bytes
        pos: u32,        // position in the byte we are currenlty reading
    }

    impl<'a> BufferRefReader<'a> {
        /// new creates a new `BufferedReader` from `bytes`
        pub fn new(bytes: &'a [u8]) -> Self {
            Self {
                bytes,
                index: 0,
                pos: 0,
            }
        }

        fn get_byte(&mut self) -> Result<u8, stream::Error> {
            self.bytes
                .get(self.index)
                .cloned()
                .ok_or(stream::Error::EOF)
        }
    }

    impl<'a> Read for BufferRefReader<'a> {
        fn read_bit(&mut self) -> Result<Bit, stream::Error> {
            if self.pos == 8 {
                self.index += 1;
                self.pos = 0;
            }

            let byte = self.get_byte()?;

            let bit = if byte & 1u8.wrapping_shl(7 - self.pos) == 0 {
                Bit::Zero
            } else {
                Bit::One
            };

            self.pos += 1;

            Ok(bit)
        }

        fn read_byte(&mut self) -> Result<u8, stream::Error> {
            if self.pos == 0 {
                self.pos += 8;
                return self.get_byte();
            }

            if self.pos == 8 {
                self.index += 1;
                return self.get_byte();
            }

            let mut byte = 0;
            let mut b = self.get_byte()?;

            byte |= b.wrapping_shl(self.pos);

            self.index += 1;
            b = self.get_byte()?;

            byte |= b.wrapping_shr(8 - self.pos);

            Ok(byte)
        }

        fn read_bits(&mut self, mut num: u32) -> Result<u64, stream::Error> {
            // can't read more than 64 bits into a u64
            if num > 64 {
                num = 64;
            }

            let mut bits: u64 = 0;
            while num >= 8 {
                let byte = self.read_byte().map(u64::from)?;
                bits = bits.wrapping_shl(8) | byte;
                num -= 8;
            }

            while num > 0 {
                self.read_bit()
                    .map(|bit| bits = bits.wrapping_shl(1) | bit.to_u64())?;

                num -= 1;
            }

            Ok(bits)
        }

        fn peak_bits(&mut self, num: u32) -> Result<u64, stream::Error> {
            // save the current index and pos so we can reset them after calling `read_bits`
            let index = self.index;
            let pos = self.pos;

            let bits = self.read_bits(num)?;

            self.index = index;
            self.pos = pos;

            Ok(bits)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::BufferRefReader;
        use tsz::stream::{Error, Read};
        use tsz::Bit;

        #[test]
        fn read_bit() {
            let bytes = [0b01101100, 0b11101001];
            let mut b = BufferRefReader::new(&bytes[..]);

            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);

            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bit().unwrap(), Bit::One);

            assert_eq!(b.read_bit().err().unwrap(), Error::EOF);
        }

        #[test]
        fn read_byte() {
            let bytes = vec![100, 25, 0, 240, 240];
            let mut b = BufferRefReader::new(&bytes);

            assert_eq!(b.read_byte().unwrap(), 100);
            assert_eq!(b.read_byte().unwrap(), 25);
            assert_eq!(b.read_byte().unwrap(), 0);

            // read some individual bits we can test `read_byte` when the position in the
            // byte we are currently reading is non-zero
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);
            assert_eq!(b.read_bit().unwrap(), Bit::One);

            assert_eq!(b.read_byte().unwrap(), 15);

            assert_eq!(b.read_byte().err().unwrap(), Error::EOF);
        }

        #[test]
        fn read_bits() {
            let bytes = vec![0b01010111, 0b00011101, 0b11110101, 0b00010100];
            let mut b = BufferRefReader::new(&bytes);

            assert_eq!(b.read_bits(3).unwrap(), 0b010);
            assert_eq!(b.read_bits(1).unwrap(), 0b1);
            assert_eq!(b.read_bits(20).unwrap(), 0b01110001110111110101);
            assert_eq!(b.read_bits(8).unwrap(), 0b00010100);
            assert_eq!(b.read_bits(4).err().unwrap(), Error::EOF);
        }

        #[test]
        fn read_mixed() {
            let bytes = vec![0b01101101, 0b01101101];
            let mut b = BufferRefReader::new(&bytes);

            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bits(3).unwrap(), 0b110);
            assert_eq!(b.read_byte().unwrap(), 0b11010110);
            assert_eq!(b.read_bits(2).unwrap(), 0b11);
            assert_eq!(b.read_bit().unwrap(), Bit::Zero);
            assert_eq!(b.read_bits(1).unwrap(), 0b1);
            assert_eq!(b.read_bit().err().unwrap(), Error::EOF);
        }

        #[test]
        fn peak_bits() {
            let bytes = vec![0b01010111, 0b00011101, 0b11110101, 0b00010100];
            let mut b = BufferRefReader::new(&bytes);

            assert_eq!(b.peak_bits(1).unwrap(), 0b0);
            assert_eq!(b.peak_bits(4).unwrap(), 0b0101);
            assert_eq!(b.peak_bits(8).unwrap(), 0b01010111);
            assert_eq!(b.peak_bits(20).unwrap(), 0b01010111000111011111);

            // read some individual bits we can test `peak_bits` when the position in the
            // byte we are currently reading is non-zero
            assert_eq!(b.read_bits(12).unwrap(), 0b010101110001);

            assert_eq!(b.peak_bits(1).unwrap(), 0b1);
            assert_eq!(b.peak_bits(4).unwrap(), 0b1101);
            assert_eq!(b.peak_bits(8).unwrap(), 0b11011111);
            assert_eq!(b.peak_bits(20).unwrap(), 0b11011111010100010100);

            assert_eq!(b.peak_bits(22).err().unwrap(), Error::EOF);
        }
    }
}
