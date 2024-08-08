use std::{
    io::{self, Read},
    marker::PhantomData,
};

use crate::{
    decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    io::bit_read::{self, BitRead},
};

use super::Reader;

pub trait XorDecoder: Default {
    /// Decompress one float from the reader
    /// If already reached the end of stream, returns None
    fn decompress_float<R: BitRead>(&mut self, r: R) -> io::Result<Option<f64>>;

    /// Reset the state of decoder
    fn reset(&mut self);
}

type BitReader<R> = bit_read::BitReader<R>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralXorReader<R: Read, T: XorDecoder> {
    bit_reader: BitReader<R>,
    number_of_records: usize,
    decompressed_result: Option<Vec<f64>>,
    xor_decoder: PhantomData<T>,
}

impl<R: Read, T: XorDecoder> GeneralXorReader<R, T> {
    pub fn new<F: FileMetadataLike>(handle: &GeneralChunkHandle<F>, reader: R) -> Self {
        let number_of_records = handle.number_of_records() as usize;
        GeneralXorReader {
            bit_reader: BitReader::new(reader),
            number_of_records,
            decompressed_result: None,
            xor_decoder: PhantomData,
        }
    }
}

pub type GeneralXorDecompressIterator<'a, R, T> =
    GeneralXorDecompressIteratorImpl<&'a mut BitReader<R>, T>;

impl<R: Read, T: XorDecoder> Reader for GeneralXorReader<R, T> {
    type NativeHeader = ();

    type DecompressIterator<'a> = GeneralXorDecompressIterator<'a, R, T>
    where
        R: 'a, T: 'a;

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

impl<R: Read, T: XorDecoder> QueryExecutor for GeneralXorReader<R, T> {}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralXorDecompressIteratorImpl<R: BitRead, T: XorDecoder> {
    decoder: T,
    number_of_records: usize,
    bit_reader: R,
}

impl<R: BitRead, T: XorDecoder> GeneralXorDecompressIteratorImpl<R, T> {
    pub fn new(bit_reader: R, number_of_records: usize) -> Self {
        GeneralXorDecompressIteratorImpl {
            decoder: T::default(),
            number_of_records,
            bit_reader,
        }
    }
}

impl<R: BitRead, T: XorDecoder> ExactSizeIterator for GeneralXorDecompressIteratorImpl<R, T> {
    fn len(&self) -> usize {
        self.number_of_records
    }
}

impl<R: BitRead, T: XorDecoder> Iterator for GeneralXorDecompressIteratorImpl<R, T> {
    type Item = io::Result<f64>;

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
