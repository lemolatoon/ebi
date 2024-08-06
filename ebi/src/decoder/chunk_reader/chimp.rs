use std::io::{self, Read};

use crate::{
    decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    io::bit_read::{self, BitRead},
};

use super::Reader;

type BitReader<R> = bit_read::BitReader<R>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpReader<R: Read> {
    bit_reader: BitReader<R>,
    number_of_records: usize,
    decompressed_result: Option<Vec<f64>>,
}

impl<R: Read> ChimpReader<R> {
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, reader: R) -> Self {
        let number_of_records = handle.number_of_records() as usize;
        ChimpReader {
            bit_reader: BitReader::new(reader),
            number_of_records,
            decompressed_result: None,
        }
    }
}

pub type ChimpDecompressIterator<'a, R> = ChimpDecompressIteratorImpl<&'a mut BitReader<R>>;

impl<R: Read> Reader for ChimpReader<R> {
    type NativeHeader = ();

    type DecompressIterator<'a> = ChimpDecompressIterator<'a, R>
    where
        R: 'a;

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        ChimpDecompressIteratorImpl::new(&mut self.bit_reader, self.number_of_records)
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed_result = Some(data);
        self.decompressed_result.as_ref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.decompressed_result.as_deref()
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
    }
}

impl<R: Read> QueryExecutor for ChimpReader<R> {}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpDecompressIteratorImpl<R: BitRead> {
    decoder: ChimpDecoder<R>,
    number_of_records: usize,
}

impl<R: BitRead> ChimpDecompressIteratorImpl<R> {
    pub fn new(bit_reader: R, number_of_records: usize) -> Self {
        ChimpDecompressIteratorImpl {
            decoder: ChimpDecoder::new(bit_reader),
            number_of_records,
        }
    }
}

impl<R: BitRead> ExactSizeIterator for ChimpDecompressIteratorImpl<R> {
    fn len(&self) -> usize {
        self.number_of_records
    }
}

impl<R: BitRead> Iterator for ChimpDecompressIteratorImpl<R> {
    type Item = io::Result<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.read_value() {
            Ok(Some(value)) => Some(Ok(value)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.number_of_records, Some(self.number_of_records))
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpDecoder<R: BitRead> {
    bit_reader: R,
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    stored_val: u64,
    first: bool,
    end_of_stream: bool,
}

impl<R: BitRead> ChimpDecoder<R> {
    const NAN_LONG: u64 = 0x7ff8000000000000;
    const LEADING_REPRESENTATION: [u8; 8] = [0, 8, 12, 16, 18, 20, 22, 24];

    pub fn new(bit_reader: R) -> Self {
        ChimpDecoder {
            bit_reader,
            stored_leading_zeros: u32::MAX,
            stored_trailing_zeros: 0,
            stored_val: 0,
            first: true,
            end_of_stream: false,
        }
    }

    pub fn get_values(&mut self) -> io::Result<Vec<f64>> {
        let mut values = Vec::new();
        while let Some(value) = self.read_value()? {
            values.push(value);
        }
        Ok(values)
    }

    pub fn read_value(&mut self) -> io::Result<Option<f64>> {
        if self.end_of_stream {
            return Ok(None);
        }

        if self.first {
            self.first = false;
            self.stored_val = self.bit_reader.read_bits(64)?;
            if self.stored_val == Self::NAN_LONG {
                self.end_of_stream = true;
                return Ok(None);
            }
        } else {
            self.next_value()?;
        }

        if self.end_of_stream {
            return Ok(None);
        }

        Ok(Some(f64::from_bits(self.stored_val)))
    }

    fn next_value(&mut self) -> io::Result<()> {
        let flag = self.bit_reader.read_bits(2)? as u8;
        match flag {
            3 => {
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[self.bit_reader.read_bits(3)? as usize] as u32;
                let significant_bits = 64 - self.stored_leading_zeros;
                let value = self.bit_reader.read_bits(significant_bits as u8)?;
                self.stored_val ^= value;
            }
            2 => {
                let significant_bits = 64 - self.stored_leading_zeros;
                let value = self.bit_reader.read_bits(significant_bits as u8)?;
                self.stored_val ^= value;
            }
            1 => {
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[self.bit_reader.read_bits(3)? as usize] as u32;
                let mut significant_bits = self.bit_reader.read_bits(6)? as u32;
                if significant_bits == 0 {
                    significant_bits = 64;
                }
                dbg!(significant_bits, self.stored_leading_zeros);
                self.stored_trailing_zeros = 64 - significant_bits - self.stored_leading_zeros;
                let value = self.bit_reader.read_bits(
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
}
