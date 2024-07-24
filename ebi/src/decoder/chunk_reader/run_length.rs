use std::{mem::size_of, slice};

use cfg_if::cfg_if;

use crate::{
    decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    format::{deserialize::FromLeBytes, run_length::RunLengthHeader},
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct RunLengthReader<'chunk> {
    number_of_records: usize,
    header: RunLengthHeader,
    values: &'chunk [f64],
    #[cfg(target_endian = "little")]
    run_counts: &'chunk [u32],
    #[cfg(not(target_endian = "little"))]
    run_counts: Vec<u32>,
    decompressed: Option<Vec<f64>>,
}

impl<'chunk> RunLengthReader<'chunk> {
    /// Create a new RunLengthReader.
    /// Caller must guarantee that the input chunk is valid for Run Length Encoding Chunk.
    /// The input chunk must begin with `RunLengthHeader`, followed by f64 values and u32 run counts.
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, chunk: &'chunk [u8]) -> Self {
        let number_of_records = handle.number_of_records() as usize;

        let header = RunLengthHeader::from_le_bytes(chunk);
        let number_of_fields = header.number_of_fields() as usize;
        // Safety:
        // This is safe because this chunk starts with RunLengthHeader.
        let values_ptr = unsafe {
            chunk
                .as_ptr()
                .cast::<RunLengthHeader>()
                .add(1)
                .cast::<f64>()
        };
        debug_assert!(
            values_ptr.is_aligned(),
            "values_ptr is not aligned: {:p}",
            values_ptr
        );
        // Safety:
        // This is safe because f64 values are followed by the header.
        let values = unsafe { slice::from_raw_parts(values_ptr, number_of_fields) };

        // Safety:
        // This is safe because u32 run_counts are followed by f64 values.
        let run_counts_ptr = unsafe { values_ptr.add(number_of_fields).cast::<u32>() };
        let run_counts = unsafe { slice::from_raw_parts(run_counts_ptr, number_of_fields) };

        cfg_if! {
            if #[cfg(target_endian = "little")] {
                Self {
                    number_of_records,
                    header,
                    run_counts,
                    values,
                    decompressed: None,
                }
            } else {
                let run_counts = run_counts.to_vec();
                Self {
                    number_of_records,
                    header,
                    run_counts,
                    values,
                    decompressed: None,
                }
            }
        }
    }

    pub fn number_of_fields(&self) -> usize {
        self.header.number_of_fields() as usize
    }

    pub fn number_of_records(&self) -> usize {
        self.number_of_records
    }
}

impl<'chunk> Reader for RunLengthReader<'chunk> {
    type NativeHeader = RunLengthHeader;

    fn decompress(&mut self) -> &[f64] {
        if self.decompressed.is_some() {
            return self.decompressed.as_ref().unwrap();
        }

        let mut buf = Vec::with_capacity(self.number_of_records());

        // self.run_counts is Vec<u32> on big endian.
        #[allow(clippy::redundant_slicing)]
        for (&value, &count) in self.values.iter().zip(&self.run_counts[..]) {
            for _ in 0..count {
                buf.push(value);
            }
        }

        self.decompressed = Some(buf);

        self.decompressed.as_ref().unwrap()
    }

    fn header_size(&self) -> usize {
        size_of::<RunLengthHeader>()
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &self.header
    }
}

// TODO: Implement specialized scan
impl QueryExecutor for RunLengthReader<'_> {}
