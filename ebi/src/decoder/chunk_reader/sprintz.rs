use std::io::Read;

use roaring::RoaringBitmap;

use crate::{
    decoder::{
        self,
        error::DecoderError,
        query::{default_filter, default_materialize, Predicate, QueryExecutor},
        FileMetadataLike, GeneralChunkHandle,
    },
    io::bit_read::{self, BitRead2, BufferedBitReader},
};

use super::Reader;

pub type BitReader = bit_read::BufferedBitReader<Vec<u8>>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct DeltaSprintzReader {
    bit_reader: BitReader,
    number_of_records: u64,
    decompressed: Option<Vec<f64>>,
}

impl DeltaSprintzReader {
    pub fn new<F: FileMetadataLike, R: Read>(
        handle: &GeneralChunkHandle<F>,
        mut r: R,
    ) -> decoder::Result<Self> {
        let number_of_records = handle.number_of_records();
        let chunk_size = handle.chunk_size() as usize;
        let mut chunk_in_memory = vec![0; chunk_size];
        r.read_exact(&mut chunk_in_memory)?;
        let bit_reader = BufferedBitReader::new(chunk_in_memory);
        Ok(Self {
            bit_reader,
            number_of_records,
            decompressed: None,
        })
    }
}

pub type DeltaSprintzDecompressIterator<'a> = DeltaSprintzDecompressIteratorImpl<&'a mut BitReader>;

impl Reader for DeltaSprintzReader {
    type NativeHeader = ();

    type DecompressIterator<'a> = DeltaSprintzDecompressIterator<'a>
    where
        Self: 'a;

    fn decompress_iter(&mut self) -> decoder::Result<Self::DecompressIterator<'_>> {
        Self::DecompressIterator::new(&mut self.bit_reader, self.number_of_records)
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed = Some(data);
        self.decompressed.as_ref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.decompressed.as_deref()
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
    }
}

impl QueryExecutor for DeltaSprintzReader {
    fn filter(
        &mut self,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
    ) -> decoder::Result<RoaringBitmap> {
        self.bit_reader.reset();
        if self.decompressed.is_some() {
            return default_filter(self, predicate, bitmask, logical_offset);
        }

        let initial_value_bits = self
            .bit_reader
            .read_bits(64)
            .ok_or(DecoderError::UnexpectedEndOfChunk)?;
        let initial_value = unsafe { std::mem::transmute::<u64, i64>(initial_value_bits) };
        let scale = self
            .bit_reader
            .read_bits(32)
            .ok_or(DecoderError::UnexpectedEndOfChunk)? as u32;
        let number_of_bits_needed = self
            .bit_reader
            .read_byte()
            .ok_or(DecoderError::UnexpectedEndOfChunk)?;

        let predicate_encoded = predicate.map(|v| (v * scale as f64).round() as i64);
        let all = || {
            let mut all = roaring::RoaringBitmap::new();
            all.insert_range(
                logical_offset as u32..(self.number_of_records as u32 + logical_offset as u32),
            );
            if let Some(bitmask) = bitmask {
                all &= bitmask;
            }
            all
        };

        if self.number_of_records == 0 {
            if predicate_encoded.eval(0) {
                return Ok(all());
            } else {
                return Ok(RoaringBitmap::new());
            }
        }

        let mut bm = RoaringBitmap::new();
        let mut previous_value_quantized = initial_value;
        for i in 0..self.number_of_records {
            let zigzag_delta = self
                .bit_reader
                .read_bits(number_of_bits_needed)
                .ok_or(DecoderError::UnexpectedEndOfChunk)?;
            let delta = unzigzag(zigzag_delta);
            let quantized = previous_value_quantized + delta;
            previous_value_quantized = quantized;

            if predicate_encoded.eval(quantized) {
                let record_offset = logical_offset as u32 + i as u32;
                bm.insert(record_offset);
            }
        }

        if let Some(bitmask) = bitmask {
            bm &= bitmask;
        }

        Ok(bm)
    }

    fn materialize<W: std::io::Write>(
        &mut self,
        output: &mut W,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
    ) -> decoder::Result<()> {
        self.bit_reader.reset();

        default_materialize(self, output, bitmask, logical_offset)
    }
}

pub struct DeltaSprintzDecompressIteratorImpl<R: BitRead2> {
    bit_reader: R,
    previous_value_quantized: i64,
    scale: u32,
    number_of_bits_needed: u8,
    number_of_records: u64,
    record_index: u64,
}

impl<R: BitRead2> DeltaSprintzDecompressIteratorImpl<R> {
    pub fn new(mut bit_reader: R, number_of_records: u64) -> decoder::Result<Self> {
        let initial_value_bits = bit_reader
            .read_bits(64)
            .ok_or(DecoderError::UnexpectedEndOfChunk)?;
        let initial_value = unsafe { std::mem::transmute::<u64, i64>(initial_value_bits) };
        let scale = bit_reader
            .read_bits(32)
            .ok_or(DecoderError::UnexpectedEndOfChunk)? as u32;
        let number_of_bits_needed = bit_reader
            .read_byte()
            .ok_or(DecoderError::UnexpectedEndOfChunk)?;
        Ok(Self {
            bit_reader,
            previous_value_quantized: initial_value,
            scale,
            number_of_bits_needed,
            number_of_records,
            record_index: 0,
        })
    }
}

impl<R: BitRead2> Iterator for DeltaSprintzDecompressIteratorImpl<R> {
    type Item = decoder::Result<f64>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.record_index >= self.number_of_records {
            debug_assert!(self.record_index == self.number_of_records);
            return None;
        }

        self.record_index += 1;
        if self.number_of_bits_needed == 0 {
            return Some(Ok(self.previous_value_quantized as f64 / self.scale as f64));
        }
        let delta = match self.bit_reader.read_bits(self.number_of_bits_needed) {
            Some(delta) => delta,
            None => return Some(Err(DecoderError::UnexpectedEndOfChunk.into())),
        };
        let quantized = unzigzag(delta) + self.previous_value_quantized;
        self.previous_value_quantized = quantized;

        Some(Ok(quantized as f64 / self.scale as f64))
    }
}

/// decode zigzag encoded number
///
/// # Example
/// ```rust
/// use ebi::decoder::chunk_reader::sprintz::unzigzag;
///
/// assert_eq!(unzigzag(0), 0);
/// assert_eq!(unzigzag(1), -1);
/// assert_eq!(unzigzag(2), 1);
/// assert_eq!(unzigzag(199999), -100000);
/// assert_eq!(unzigzag(i64::MAX as u64 - 1), i64::MAX / 2);
/// ```
#[inline]
pub const fn unzigzag(origin: u64) -> i64 {
    (origin >> 1) as i64 ^ -((origin & 1) as i64)
}
