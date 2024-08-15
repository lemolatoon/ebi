use std::{mem::size_of_val, slice};

use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::format::{deserialize, serialize};

use super::{AppendableCompressor, Capacity, Compressor, RewindableCompressor, MAX_BUFFERS};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct UncompressedCompressor {
    total_bytes_in: usize,
    buffer: Vec<f64>,
}

impl UncompressedCompressor {
    pub fn new(capacity: usize) -> Self {
        Self {
            total_bytes_in: 0,
            buffer: Vec::with_capacity(capacity),
        }
    }
}

impl Compressor for UncompressedCompressor {
    fn compress(&mut self, input: &[f64]) {
        self.reset();
        for &value in input {
            self.buffer.push(value);
        }

        let size = size_of_val(input);
        self.total_bytes_in += size;
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        size_of_val(&self.buffer[..])
    }

    fn prepare(&mut self) {}

    fn buffers<'a>(&'a self) -> [&'a [u8]; MAX_BUFFERS] {
        const EMPTY: &[u8] = &[];

        let f64buf: &'a [f64] = &self.buffer[..];
        let buffer: &'a [u8] = unsafe {
            slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>(), size_of_val(f64buf))
        };

        [buffer, EMPTY, EMPTY, EMPTY, EMPTY]
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
}

impl RewindableCompressor for UncompressedCompressor {
    fn rewind(&mut self, n: usize) -> bool {
        if self.buffer.len() < n {
            return false;
        }
        self.buffer.truncate(self.buffer.len() - n);

        true
    }
}

#[derive(Builder, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, packed)]
pub struct UncompressedCompressorConfig {
    #[builder(setter(into), default)]
    capacity: Capacity,
}

serialize::impl_to_le!(UncompressedCompressorConfig, capacity);
deserialize::impl_from_le_bytes!(
    UncompressedCompressorConfig,
    uncompressed,
    (capacity, Capacity)
);

impl From<UncompressedCompressorConfig> for UncompressedCompressor {
    fn from(config: UncompressedCompressorConfig) -> UncompressedCompressor {
        let cap = config.capacity.0 as usize;
        UncompressedCompressor::new(cap)
    }
}

impl UncompressedCompressorConfigBuilder {
    pub fn build(self) -> UncompressedCompressorConfig {
        let Self { capacity } = self;
        UncompressedCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
        }
    }
}
