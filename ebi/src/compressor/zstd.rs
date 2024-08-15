use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder,
    format::{deserialize, serialize},
};

use super::Compressor;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ZstdCompressor {
    level: i32,
    total_bytes_in: usize,
    compressed: Option<Vec<u8>>,
}

#[derive(Builder, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, packed)]
pub struct ZstdCompressorConfig {
    level: i32,
}

impl ZstdCompressorConfigBuilder {
    pub fn build(self) -> ZstdCompressorConfig {
        let Self { level } = self;
        let level = level.unwrap_or(3);
        ZstdCompressorConfig { level }
    }
}

serialize::impl_to_le!(ZstdCompressorConfig, level);
deserialize::impl_from_le_bytes!(ZstdCompressorConfig, zstd, (level, i32));

impl From<ZstdCompressorConfig> for ZstdCompressor {
    fn from(value: ZstdCompressorConfig) -> Self {
        Self {
            level: value.level,
            compressed: None,
            total_bytes_in: 0,
        }
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&mut self, input: &[f64]) -> encoder::Result<()> {
        let bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), size_of_val(input)) };
        self.total_bytes_in += bytes.len();

        let compressed = zstd::encode_all(bytes, self.level)?;
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
