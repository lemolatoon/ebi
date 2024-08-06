use std::io::{self, Read};

use crate::{
    decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    io::bit_read::{self, BitRead},
};

use super::Reader;

type BitReader<R> = bit_read::BitReader<R>;

pub type Chimp128Reader<R> = ChimpNReader<128, R>;
pub type Chimp128DecompressIterator<'a, R> = ChimpNDecompressIterator<'a, 128, R>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpNReader<const N: usize, R: Read> {
    bit_reader: BitReader<R>,
    number_of_records: usize,
    decompressed_result: Option<Vec<f64>>,
}

impl<const N: usize, R: Read> ChimpNReader<N, R> {
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, reader: R) -> Self {
        let number_of_records = handle.number_of_records() as usize;
        ChimpNReader {
            bit_reader: BitReader::new(reader),
            number_of_records,
            decompressed_result: None,
        }
    }
}

pub type ChimpNDecompressIterator<'a, const N: usize, R> =
    ChimpNDecompressIteratorImpl<N, &'a mut BitReader<R>>;

impl<const N: usize, R: Read> Reader for ChimpNReader<N, R> {
    type NativeHeader = ();

    type DecompressIterator<'a> = ChimpNDecompressIterator<'a, N, R>
    where
        R: 'a;

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        ChimpNDecompressIteratorImpl::new(&mut self.bit_reader, self.number_of_records)
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

impl<const N: usize, R: Read> QueryExecutor for ChimpNReader<N, R> {}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ChimpNDecompressIteratorImpl<const N: usize, R: BitRead> {
    decoder: ChimpNDecoder<N, R>,
    number_of_records: usize,
}

impl<const N: usize, R: BitRead> ChimpNDecompressIteratorImpl<N, R> {
    pub fn new(bit_reader: R, number_of_records: usize) -> Self {
        ChimpNDecompressIteratorImpl {
            decoder: ChimpNDecoder::new(bit_reader),
            number_of_records,
        }
    }
}

impl<const N: usize, R: BitRead> ExactSizeIterator for ChimpNDecompressIteratorImpl<N, R> {
    fn len(&self) -> usize {
        self.number_of_records
    }
}

impl<const N: usize, R: BitRead> Iterator for ChimpNDecompressIteratorImpl<N, R> {
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
pub struct ChimpNDecoder<const N_PREVIOUS_VALUES: usize, R: BitRead> {
    bit_reader: R,
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    stored_val: u64,
    stored_values: Box<[u64]>,
    current: usize,
    first: bool,
    end_of_stream: bool,
}

impl<const N_PREVIOUS_VALUES: usize, R: BitRead> ChimpNDecoder<N_PREVIOUS_VALUES, R> {
    pub const LEADING_REPRESENTATION: [u8; 8] = [0, 8, 12, 16, 18, 20, 22, 24];
    pub const NAN_LONG: u64 = 0x7ff8000000000000;
    const PREVIOUS_VALUES_LOG2: u32 = (N_PREVIOUS_VALUES as u64).trailing_zeros();
    const INITIAL_FILL: u32 = Self::PREVIOUS_VALUES_LOG2 + /* leading_count, center_count */9;

    pub fn new(bit_reader: R) -> Self {
        ChimpNDecoder {
            bit_reader,
            stored_leading_zeros: u32::MAX,
            stored_trailing_zeros: 0,
            stored_val: 0,
            stored_values: vec![0; N_PREVIOUS_VALUES].into_boxed_slice(),
            current: 0,
            first: true,
            end_of_stream: false,
        }
    }

    pub fn read_value(&mut self) -> io::Result<Option<f64>> {
        if self.end_of_stream {
            return Ok(None);
        }

        self.next()?;

        if self.end_of_stream {
            return Ok(None);
        }

        Ok(Some(f64::from_bits(self.stored_val)))
    }

    pub fn get_values(&mut self) -> io::Result<Vec<f64>> {
        let mut values = Vec::new();
        while let Some(value) = self.read_value()? {
            values.push(value);
        }
        Ok(values)
    }

    fn next(&mut self) -> io::Result<()> {
        if self.first {
            self.first = false;
            self.stored_val = self.bit_reader.read_bits(64)?;
            self.stored_values[self.current] = self.stored_val;
            if self.stored_val == Self::NAN_LONG {
                self.end_of_stream = true;
                return Ok(());
            }
        } else {
            self.next_value()?;
        }

        Ok(())
    }

    fn next_value(&mut self) -> io::Result<()> {
        let flag = self.bit_reader.read_bits(2)? as u8;
        match flag {
            3 => {
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[self.bit_reader.read_bits(3)? as usize] as u32;
                let value = self
                    .bit_reader
                    .read_bits(64 - self.stored_leading_zeros as u8)?;
                self.stored_val ^= value;

                if self.stored_val == Self::NAN_LONG {
                    self.end_of_stream = true;
                } else {
                    self.current = (self.current + 1) % N_PREVIOUS_VALUES;
                    self.stored_values[self.current] = self.stored_val;
                }
            }
            2 => {
                let value = self
                    .bit_reader
                    .read_bits(64 - self.stored_leading_zeros as u8)?;
                self.stored_val ^= value;

                if self.stored_val == Self::NAN_LONG {
                    self.end_of_stream = true;
                } else {
                    self.current = (self.current + 1) % N_PREVIOUS_VALUES;
                    self.stored_values[self.current] = self.stored_val;
                }
            }
            1 => {
                let mut fill = Self::INITIAL_FILL;
                let temp = self.bit_reader.read_bits(fill as u8)?;
                fill -= Self::PREVIOUS_VALUES_LOG2;
                let index = (temp >> fill) & ((1 << Self::PREVIOUS_VALUES_LOG2) - 1);
                fill -= 3;
                self.stored_leading_zeros =
                    Self::LEADING_REPRESENTATION[((temp >> fill) & ((1 << 3) - 1)) as usize] as u32;
                fill -= 6;
                let mut significant_bits = (temp >> fill) & ((1 << 6) - 1);
                self.stored_val = self.stored_values[index as usize];
                if significant_bits == 0 {
                    significant_bits = 64;
                }
                dbg!(significant_bits, self.stored_leading_zeros);
                self.stored_trailing_zeros =
                    64 - significant_bits as u32 - self.stored_leading_zeros;
                let value = self.bit_reader.read_bits(
                    (64 - self.stored_leading_zeros - self.stored_trailing_zeros) as u8,
                )?;
                self.stored_val ^= value << self.stored_trailing_zeros;

                if self.stored_val == Self::NAN_LONG {
                    self.end_of_stream = true;
                } else {
                    self.current = (self.current + 1) % N_PREVIOUS_VALUES;
                    self.stored_values[self.current] = self.stored_val;
                }
            }
            _ => {
                self.stored_val = self.stored_values[self
                    .bit_reader
                    .read_bits(Self::PREVIOUS_VALUES_LOG2 as u8)?
                    as usize];
                self.current = (self.current + 1) % N_PREVIOUS_VALUES;
                self.stored_values[self.current] = self.stored_val;
            }
        }

        Ok(())
    }
}
