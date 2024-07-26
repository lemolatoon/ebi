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

    fn decompress(&mut self) -> io::Result<&[f64]> {
        if self.decompressed.is_some() {
            return Ok(self.decompressed.as_ref().unwrap());
        }

        let mut decompressed = Vec::with_capacity(self.number_of_records());

        for _ in 0..dbg!(self.number_of_fields()) {
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
}

// TODO: Implement specialized scan
impl<R: Read> QueryExecutor for RunLengthReader<R> {}
