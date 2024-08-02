use std::{
    mem::{size_of, size_of_val},
    slice,
};

use crate::format::{
    serialize::{AsBytes, ToLe},
    uncompressed::UncompressedHeader0,
};

use super::{size_estimater::StaticSizeEstimator, AppendableCompressor, Compressor, MAX_BUFFERS};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct UncompressedCompressor {
    total_bytes_in: usize,
    header0: UncompressedHeader0,
    buffer: Vec<f64>,
    header: Option<Box<[u8]>>,
}

impl UncompressedCompressor {
    pub fn new(capacity: usize) -> Self {
        Self {
            total_bytes_in: 0,
            buffer: Vec::with_capacity(capacity),
            header0: UncompressedHeader0 { header_size: 0 },
            header: None,
        }
    }

    pub fn header(mut self, header: Box<[u8]>) -> Self {
        self.header = Some(header);
        self
    }
}

impl Compressor for UncompressedCompressor {
    type SizeEstimatorImpl<'comp, 'buf> = StaticSizeEstimator<'comp, 'buf, Self>;
    fn compress(&mut self, input: &[f64]) {
        self.reset();
        for &value in input {
            self.buffer.push(value);
        }

        let size = size_of_val(input);
        self.total_bytes_in += size;
    }

    fn estimate_size_static(
        &self,
        number_of_records: usize,
        _estimate_option: super::size_estimater::EstimateOption,
    ) -> Option<usize> {
        Some(
            size_of::<UncompressedHeader0>()
                + self.header.as_ref().map_or(0, |h| h.len())
                + size_of::<f64>() * number_of_records,
        )
    }

    fn size_estimater<'comp, 'buf>(
        &'comp mut self,
        input: &'buf [f64],
        estimate_option: super::size_estimater::EstimateOption,
    ) -> Option<Self::SizeEstimatorImpl<'comp, 'buf>> {
        Some(StaticSizeEstimator::new(self, input, estimate_option))
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        size_of::<UncompressedHeader0>()
            + self.header.as_ref().map_or(0, |h| h.len())
            + size_of_val(&self.buffer[..])
    }

    fn prepare(&mut self) {
        let header0_size = size_of_val(&self.header0);
        let flexible_header_size = self.header.as_ref().map(|h| h.len()).unwrap_or(0);
        let mut header0 = UncompressedHeader0 {
            header_size: header0_size as u8 + flexible_header_size as u8,
        };
        header0.to_le();
        self.header0 = header0;
    }

    fn buffers<'a>(&'a self) -> [&'a [u8]; MAX_BUFFERS] {
        let header0 = self.header0.as_bytes();
        const EMPTY: &[u8] = &[];
        let header = self.header.as_ref().map_or(EMPTY, |h| h.as_ref());

        let f64buf: &'a [f64] = &self.buffer[..];
        let buffer: &'a [u8] = unsafe {
            slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>(), size_of_val(f64buf))
        };

        [header0, header, buffer, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        self.total_bytes_in = 0;
        self.buffer.clear();
    }
}

impl AppendableCompressor for UncompressedCompressor {
    fn append_compress(&mut self, input: &[f64]) {
        for &value in input {
            self.buffer.push(value);
        }

        let size = size_of_val(input);
        self.total_bytes_in += size;
    }

    fn rewind(&mut self, n: usize) -> bool {
        if self.buffer.len() < n {
            return false;
        }
        self.buffer.truncate(self.buffer.len() - n);

        true
    }
}
