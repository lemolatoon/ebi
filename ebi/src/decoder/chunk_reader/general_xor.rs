use std::{
    io::{self, Read},
    marker::PhantomData,
};

use crate::{
    decoder::{self, query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    io::bit_read::{self, BitRead2},
};

use super::Reader;

pub trait XorDecoder: Default {
    /// Decompress one float from the reader
    /// If already reached the end of stream, returns None
    fn decompress_float<R: BitRead2>(&mut self, r: R) -> decoder::Result<Option<f64>>;

    /// Reset the state of decoder
    fn reset(&mut self);
}

type BitReader = bit_read::BufferedBitReader<Vec<u8>>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralXorReader<T: XorDecoder> {
    bit_reader: BitReader,
    number_of_records: usize,
    decompressed_result: Option<Vec<f64>>,
    xor_decoder: PhantomData<T>,
}

impl<T: XorDecoder> GeneralXorReader<T> {
    pub fn new<F: FileMetadataLike, R: Read>(
        handle: &GeneralChunkHandle<F>,
        mut reader: R,
    ) -> io::Result<Self> {
        let number_of_records = handle.number_of_records() as usize;
        let chunk_size = handle.chunk_size() as usize;
        let mut chunk_in_memory = vec![0; chunk_size];
        reader.read_exact(&mut chunk_in_memory)?;
        let bit_reader = BitReader::new(chunk_in_memory);
        Ok(GeneralXorReader {
            bit_reader,
            number_of_records,
            decompressed_result: None,
            xor_decoder: PhantomData,
        })
    }
}

pub type GeneralXorDecompressIterator<'a, T> =
    GeneralXorDecompressIteratorImpl<&'a mut BitReader, T>;

impl<T: XorDecoder> Reader for GeneralXorReader<T> {
    type NativeHeader = ();

    type DecompressIterator<'a> = GeneralXorDecompressIterator<'a, T>
    where
        T: 'a;

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        Self::DecompressIterator::new(&mut self.bit_reader, self.number_of_records)
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

impl<T: XorDecoder> QueryExecutor for GeneralXorReader<T> {}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralXorDecompressIteratorImpl<R: BitRead2, T: XorDecoder> {
    decoder: T,
    number_of_records: usize,
    bit_reader: R,
}

impl<R: BitRead2, T: XorDecoder> GeneralXorDecompressIteratorImpl<R, T> {
    pub fn new(bit_reader: R, number_of_records: usize) -> Self {
        GeneralXorDecompressIteratorImpl {
            decoder: T::default(),
            number_of_records,
            bit_reader,
        }
    }
}

impl<R: BitRead2, T: XorDecoder> ExactSizeIterator for GeneralXorDecompressIteratorImpl<R, T> {
    fn len(&self) -> usize {
        self.number_of_records
    }
}

impl<R: BitRead2, T: XorDecoder> Iterator for GeneralXorDecompressIteratorImpl<R, T> {
    type Item = decoder::Result<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decompress_float(&mut self.bit_reader) {
            Ok(Some(value)) => Some(Ok(value)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.number_of_records, Some(self.number_of_records))
    }
}
