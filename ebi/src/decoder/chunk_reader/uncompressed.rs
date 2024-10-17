use std::{
    io::{self, Read},
    mem::size_of,
};

use crate::{
    decoder::{self, query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    io::as_bytes_mut::AsBytesMut,
    time::{SegmentKind, SegmentedExecutionTimes},
};

use super::Reader;

pub type UncompressedReader = UncompressedReaderImpl<io::Cursor<Vec<u8>>>;
pub type UncompressedIterator<'a> = UncompressedIteratorImpl<'a, io::Cursor<Vec<u8>>>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct UncompressedReaderImpl<R: Read> {
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
        timer: &mut SegmentedExecutionTimes,
    ) -> io::Result<UncompressedReaderImpl<io::Cursor<Vec<u8>>>> {
        let number_of_records = handle.number_of_records() as usize;

        let chunk_size = handle.chunk_size() as usize;
        let mut chunk_in_memory = vec![0; chunk_size];
        let io_read_timer = timer.start_addition_measurement(SegmentKind::IORead);
        reader.read_exact(&mut chunk_in_memory)?;
        io_read_timer.stop();
        let reader = io::Cursor::new(chunk_in_memory);

        Ok(UncompressedReaderImpl {
            number_of_records,
            reader,
            values: None,
        })
    }
}

impl<R: Read> Reader for UncompressedReaderImpl<R> {
    type NativeHeader = ();
    type DecompressIterator<'a>
        = UncompressedIteratorImpl<'a, R>
    where
        Self: 'a;

    fn decompress(&mut self, timer: &mut SegmentedExecutionTimes) -> decoder::Result<&[f64]> {
        if self.values.is_some() {
            return Ok(self.values.as_ref().unwrap());
        }

        let mut buf: Vec<f64> = vec![0.0; self.number_of_records];
        let buf_ref = buf.as_mut_slice();
        let bytes = buf_ref.as_bytes_mut();
        let io_read_timer = timer.start_addition_measurement(SegmentKind::IORead);
        self.reader.read_exact(bytes)?;
        io_read_timer.stop();

        self.values = Some(buf);

        Ok(self.values.as_ref().unwrap())
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
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
