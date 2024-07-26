use std::{
    io::{self, Read},
    mem::size_of,
};

use crate::{
    decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    format::{deserialize::FromLeBytesExt, run_length::RunLengthHeader},
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct RunLengthReader<R: Read> {
    number_of_records: usize,
    header: RunLengthHeader,
    reader: R,
    decompressed: Option<Vec<f64>>,
}

impl<R: Read> RunLengthReader<R> {
    /// Create a new RunLengthReader.
    pub fn new<T: FileMetadataLike>(
        handle: &GeneralChunkHandle<T>,
        mut reader: R,
    ) -> io::Result<Self> {
        let number_of_records = handle.number_of_records() as usize;

        let header = RunLengthHeader::from_le_bytes_by_reader(&mut reader)?;

        Ok(Self {
            number_of_records,
            header,
            reader,
            decompressed: None,
        })
    }

    pub fn number_of_fields(&self) -> usize {
        self.header.number_of_fields() as usize
    }

    pub fn number_of_records(&self) -> usize {
        self.number_of_records
    }
}

impl<R: Read> Reader for RunLengthReader<R> {
    type NativeHeader = RunLengthHeader;
    type DecompressIterator<'a> = RunLengthIterator<'a, R> where Self: 'a;

    fn decompress(&mut self) -> io::Result<&[f64]> {
        if self.decompressed.is_some() {
            return Ok(self.decompressed.as_ref().unwrap());
        }

        let mut decompressed = Vec::with_capacity(self.number_of_records());

        for _ in 0..self.number_of_fields() {
            let mut buf = [0u8; size_of::<u8>() + size_of::<f64>()];
            self.reader.read_exact(&mut buf)?;
            let run_count = buf[0];
            let value = f64::from_le_bytes(buf[1..9].try_into().unwrap());

            for _ in 0..run_count {
                decompressed.push(value);
            }
        }

        self.decompressed = Some(decompressed);

        Ok(self.decompressed.as_ref().unwrap())
    }

    fn header_size(&self) -> usize {
        size_of::<RunLengthHeader>()
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &self.header
    }

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        RunLengthIterator::new(self)
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed = Some(data);

        self.decompressed.as_ref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.decompressed.as_deref()
    }
}

pub struct RunLengthIterator<'a, R: Read> {
    reader: &'a mut RunLengthReader<R>,
    current_field: usize,
    current_run: u8,
    current_value: f64,
}

impl<'a, R: Read> RunLengthIterator<'a, R> {
    pub fn new(reader: &'a mut RunLengthReader<R>) -> Self {
        Self {
            reader,
            current_field: 0,
            current_run: 0,
            current_value: 0.0,
        }
    }
}

impl<R: Read> Iterator for RunLengthIterator<'_, R> {
    type Item = io::Result<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_run == 0 {
            self.current_field += 1;
            if self.current_field > self.reader.number_of_fields() {
                return None;
            }
            let mut buf = [0u8; size_of::<u8>() + size_of::<f64>()];
            match self.reader.reader.read_exact(&mut buf) {
                Ok(_) => {
                    self.current_run = buf[0];
                    self.current_value = f64::from_le_bytes(buf[1..9].try_into().unwrap());
                }
                Err(e) => return Some(Err(e)),
            }
        }

        self.current_run -= 1;

        Some(Ok(self.current_value))
    }
}

// TODO: Implement specialized scan
impl<R: Read> QueryExecutor for RunLengthReader<R> {}
