use alp_binding::compress_double;
use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder,
    format::{deserialize, serialize},
};

use super::Compressor;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FFIAlpCompressor {
    total_bytes_in: usize,
    compressed: Vec<u8>,
    is_compressed: bool,
}

#[derive(Builder, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, packed)]
pub struct FFIAlpCompressorConfig {}

impl FFIAlpCompressorConfigBuilder {
    pub fn build(self) -> FFIAlpCompressorConfig {
        FFIAlpCompressorConfig {}
    }
}

serialize::impl_to_le!(FFIAlpCompressorConfig,);
deserialize::impl_from_le_bytes!(FFIAlpCompressorConfig, ffi_alp,);

impl From<FFIAlpCompressorConfig> for FFIAlpCompressor {
    fn from(_value: FFIAlpCompressorConfig) -> Self {
        Self {
            compressed: Vec::new(),
            total_bytes_in: 0,
            is_compressed: false,
        }
    }
}

impl Compressor for FFIAlpCompressor {
    fn compress(&mut self, input: &[f64]) -> encoder::Result<()> {
        self.reset();

        compress_double(input, &mut self.compressed);
        self.is_compressed = true;

        self.total_bytes_in += size_of_val(input);

        Ok(())
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        self.compressed.len()
    }

    fn prepare(&mut self) {}

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];

        [self.compressed.as_slice(), E, E, E, E]
    }

    fn reset(&mut self) {
        self.compressed.clear();
        self.is_compressed = false;
        self.total_bytes_in = 0;
    }
}
