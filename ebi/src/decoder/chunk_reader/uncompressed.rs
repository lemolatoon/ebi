use std::{
    io::{self, Read},
    mem::size_of,
};

use crate::{
    decoder::{self, query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    format::{
        deserialize::FromLeBytesExt,
        uncompressed::{NativeUncompressedHeader, UncompressedHeader0},
    },
    io::as_bytes_mut::AsBytesMut,
};

use super::Reader;

pub type UncompressedReader = UncompressedReaderImpl<io::Cursor<Vec<u8>>>;
pub type UncompressedIterator<'a> = UncompressedIteratorImpl<'a, io::Cursor<Vec<u8>>>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct UncompressedReaderImpl<R: Read> {
    header: NativeUncompressedHeader,
    number_of_records: usize,
    values: Option<Vec<f64>>,
    reader: R,
}

impl UncompressedReader {
    /// Create a new UncompressedReader.
    /// Caller must guarantee that the input chunk is valid for Uncompressed Chunk.
    pub fn new<T: FileMetadataLike, R: Read>(
        handle: &GeneralChunkHandle<T>,
        mut reader: R,
    ) -> io::Result<UncompressedReaderImpl<io::Cursor<Vec<u8>>>> {
        let header_head = UncompressedHeader0::from_le_bytes_by_reader(&mut reader)?;
        let header_size = header_head.header_size as usize;
        let header1_size = header_size - size_of::<UncompressedHeader0>();
        let mut header = vec![0u8; header1_size].into_boxed_slice();
        reader.read_exact(&mut header)?;

        let header = NativeUncompressedHeader::new(header_head, header);
        let number_of_records = handle.number_of_records() as usize;

        let chunk_size = handle.chunk_size() as usize - header_size;
        let mut chunk_in_memory = vec![0; chunk_size];
        reader.read_exact(&mut chunk_in_memory)?;
        let reader = io::Cursor::new(chunk_in_memory);

        Ok(UncompressedReaderImpl {
            header,
            number_of_records,
            reader,
            values: None,
        })
    }
}

impl<R: Read> Reader for UncompressedReaderImpl<R> {
    type NativeHeader = NativeUncompressedHeader;
    type DecompressIterator<'a> = UncompressedIteratorImpl<'a, R> where Self: 'a;

    fn decompress(&mut self) -> decoder::Result<&[f64]> {
        if self.values.is_some() {
            return Ok(self.values.as_ref().unwrap());
        }

        let mut buf: Vec<f64> = vec![0.0; self.number_of_records];
        let buf_ref = buf.as_mut_slice();
        let bytes = buf_ref.as_bytes_mut();
        self.reader.read_exact(bytes)?;

        self.values = Some(buf);

        Ok(self.values.as_ref().unwrap())
    }

    fn header_size(&self) -> usize {
        *self.header.header_size() as usize
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &self.header
    }

    fn decompress_iter(&mut self) -> decoder::Result<Self::DecompressIterator<'_>> {
        Ok(UncompressedIteratorImpl::new(self))
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.values = Some(data);

        self.values.as_deref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.values.as_deref()
    }
}

pub struct UncompressedIteratorImpl<'a, R: Read> {
    reader: &'a mut UncompressedReaderImpl<R>,
    count: usize,
}

impl<'a, R: Read> UncompressedIteratorImpl<'a, R> {
    pub fn new(reader: &'a mut UncompressedReaderImpl<R>) -> Self {
        UncompressedIteratorImpl { reader, count: 0 }
    }
}

impl<'a, R: Read> Iterator for UncompressedIteratorImpl<'a, R> {
    type Item = decoder::Result<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.reader.number_of_records {
            return None;
        }

        let mut buf = [0u8; size_of::<f64>()];
        match self.reader.reader.read_exact(&mut buf) {
            Ok(_) => {
                self.count += 1;
                Some(Ok(f64::from_le_bytes(buf)))
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

impl<R: Read> QueryExecutor for UncompressedReaderImpl<R> {}
