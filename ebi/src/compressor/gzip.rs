use std::io::Write as _;

use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder,
    format::{deserialize, serialize},
};

use super::Compressor;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GzipCompressor {
    level: u32,
    total_bytes_in: usize,
    compressed: Option<Vec<u8>>,
}

#[derive(Builder, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, packed)]
pub struct GzipCompressorConfig {
    /// Compression level (1-9)
    /// Default: 6
    level: u32,
}

impl GzipCompressorConfigBuilder {
    pub fn build(self) -> GzipCompressorConfig {
        let Self { level } = self;
        let level = level.unwrap_or(6);
        GzipCompressorConfig { level }
    }
}

serialize::impl_to_le!(GzipCompressorConfig, level);
deserialize::impl_from_le_bytes!(GzipCompressorConfig, gzip, (level, u32));

impl From<GzipCompressorConfig> for GzipCompressor {
    fn from(value: GzipCompressorConfig) -> Self {
        Self {
            level: value.level,
            compressed: None,
            total_bytes_in: 0,
        }
    }
}

impl Compressor for GzipCompressor {
    fn compress(&mut self, input: &[f64]) -> encoder::Result<()> {
        self.reset();
        let bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), size_of_val(input)) };
        self.total_bytes_in += bytes.len();

        let mut compressed = Vec::<u8>::new();
        let mut encoder =
            flate2::GzBuilder::new().write(&mut compressed, flate2::Compression::new(self.level));
        encoder.write_all(bytes)?;
        encoder.finish()?;

        self.compressed = Some(compressed);

        Ok(())
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        self.compressed.as_ref().map_or(0, |c| c.len())
    }

    fn prepare(&mut self) {}

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];
        let compressed = self.compressed.as_ref().map_or(E, |c| c.as_slice());

        [compressed, E, E, E, E]
    }

    fn reset(&mut self) {
        self.compressed = None;
        self.total_bytes_in = 0;
    }
}
