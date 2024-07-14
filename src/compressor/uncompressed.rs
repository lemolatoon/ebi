use std::{
    cmp::min,
    mem::{size_of, size_of_val},
    slice,
};

use crate::format::{
    serialize::{AsBytes, ToLe},
    uncompressed::UncompressedHeader0,
};

use super::Compressor;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UncompressedCompressor {
    total_bytes_in: usize,
    total_bytes_out: usize,
    header: Option<Box<[u8]>>,
}

impl UncompressedCompressor {
    pub fn new() -> Self {
        Self {
            total_bytes_in: 0,
            total_bytes_out: 0,
            header: None,
        }
    }

    pub fn header(mut self, header: Box<[u8]>) -> Self {
        self.header = Some(header);
        self
    }
}

impl Default for UncompressedCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for UncompressedCompressor {
    fn compress<'a>(&mut self, input: &'a [f64], output: &mut [u8]) -> usize {
        let input_bytes_ptr = input.as_ptr().cast::<u8>();
        let len = size_of_val(input) / size_of::<u8>();
        // Safety:
        // `input_bytes_ptr` is valid for `len * size_of::<u8>()` -> `size_of::<f64> * input.len()`.
        // `input_bytes_ptr` is non null because it is originated from slice.
        // `input_bytes_slice`'s lifetime is the same as `input`.
        // Consecutive `len` u8 memory is properly initialized, since u8 is valid whatever values it has as a memory representation.
        // `len * size_of::<u8>()` -> `size_of::<f64> * input.len()` is no larger than `isize::MAX`
        debug_assert!(size_of::<u8>() * len <= isize::MAX as usize);
        let input_bytes_slice: &'a [u8] = unsafe { slice::from_raw_parts(input_bytes_ptr, len) };

        let copyable_bytes = min(input_bytes_slice.len(), output.len());
        let number_of_tuples_copied = copyable_bytes / size_of::<f64>();
        let number_of_bytes_copied = number_of_tuples_copied * size_of::<f64>();

        output[0..number_of_bytes_copied]
            .copy_from_slice(&input_bytes_slice[0..number_of_bytes_copied]);

        self.total_bytes_in += number_of_bytes_copied;
        self.total_bytes_out += number_of_bytes_copied;

        number_of_bytes_copied
    }

    fn header_size(&self) -> usize {
        size_of::<UncompressedHeader0>() + self.header.as_ref().map_or(0, |h| h.len())
    }

    fn write_header(&mut self, output: &mut [u8]) {
        let header_size = self.header_size() as u8;
        let mut header0 = UncompressedHeader0 { header_size };
        output[..size_of::<UncompressedHeader0>()].copy_from_slice(header0.to_le().as_bytes());
        if let Some(header) = self.header.as_ref() {
            output[size_of::<UncompressedHeader0>()..self.header_size()]
                .copy_from_slice(&header[..]);
        }
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_out(&self) -> usize {
        self.total_bytes_out
    }
}
