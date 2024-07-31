use buff::BUFFCompressor;
use derive_builder::Builder;
use gorilla::GorillaCompressor;
use quick_impl::QuickImpl;
use run_length::RunLengthCompressor;
// use run_length::RunLengthCompressor;
use uncompressed::UncompressedCompressor;

use crate::format::{self, CompressionScheme};

pub mod buff;
pub mod gorilla;
pub mod run_length;
pub mod uncompressed;

const MAX_BUFFERS: usize = 5;
pub trait Compressor {
    /// Take as many values from input, and compress and write it to the internal buffer as possible.
    /// Returns the number of bytes consumed from input.
    fn compress(&mut self, input: &[f64]) -> usize;

    /// Returns the total number of input bytes which have been processed by this Compressor.
    fn total_bytes_in(&self) -> usize;

    /// Returns the total bytes buffered in the compressor.
    fn total_bytes_buffered(&self) -> usize; // NOTE: used for stop compression for the chunk with ChunkOption::ByteSize(usize)

    /// Prepare the internal buffer for reading.
    /// This method should be called before reading the compressed data using `buffers`.
    /// Typically, this method is used to write the header to the internal buffer.
    fn prepare(&mut self);

    /// Returns the buffered data as an array of the write order.
    /// MAX_BUFFERS can be changed, do not rely on the exact number of buffers.
    fn buffers(&self) -> [&[u8]; MAX_BUFFERS];

    /// Reset the internal state of the compressor.
    fn reset(&mut self);
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum GenericCompressor {
    Uncompressed(UncompressedCompressor),
    RLE(RunLengthCompressor),
    Gorilla(GorillaCompressor),
    BUFF(BUFFCompressor),
}

macro_rules! impl_generic_compressor {
    ($enum_name:ident, $($variant:ident),*) => {
        impl $enum_name {
            pub fn compress(&mut self, input: &[f64]) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.compress(input), )*
                }
            }

            pub fn total_bytes_in(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_in(), )*
                }
            }

            pub fn total_bytes_buffered(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_buffered(), )*
                }
            }

            pub fn prepare(&mut self) {
                match self {
                    $( $enum_name::$variant(c) => c.prepare(), )*
                }
            }

            pub fn buffers(&mut self) -> [&[u8]; MAX_BUFFERS] {
                match self {
                    $( $enum_name::$variant(c) => c.buffers(), )*
                }
            }

            pub fn reset(&mut self) {
                match self {
                    $( $enum_name::$variant(c) => c.reset(), )*
                }
            }
        }
    };
}

impl GenericCompressor {
    pub fn compression_scheme(&self) -> format::CompressionScheme {
        match self {
            GenericCompressor::Uncompressed(_) => format::CompressionScheme::Uncompressed,
            GenericCompressor::RLE(_) => format::CompressionScheme::RLE,
            GenericCompressor::Gorilla(_) => format::CompressionScheme::Gorilla,
            GenericCompressor::BUFF(_) => format::CompressionScheme::BUFF,
        }
    }
}

impl_generic_compressor!(GenericCompressor, Uncompressed, RLE, Gorilla, BUFF);

#[derive(QuickImpl, Debug, Clone)]
pub enum CompressorConfig {
    #[quick_impl(impl From)]
    Uncompressed(UncompressedCompressorConfig),
    #[quick_impl(impl From)]
    RLE(RunLengthCompressorConfig),
    #[quick_impl(impl From)]
    Gorilla(GorillaCompressorConfig),
    #[quick_impl(impl From)]
    BUFF(BUFFCompressorConfig),
}

impl CompressorConfig {
    pub fn build(self) -> GenericCompressor {
        match self {
            CompressorConfig::Uncompressed(c) => GenericCompressor::Uncompressed({
                let mut comp = UncompressedCompressor::new(c.capacity.0);
                if let Some(header) = c.header {
                    comp = comp.header(header);
                }
                comp
            }),
            CompressorConfig::RLE(c) => {
                GenericCompressor::RLE(RunLengthCompressor::with_capacity(c.capacity.0))
            }
            CompressorConfig::Gorilla(c) => {
                GenericCompressor::Gorilla(GorillaCompressor::with_capacity(c.capacity.0))
            }
            CompressorConfig::BUFF(c) => GenericCompressor::BUFF(BUFFCompressor::new(c.scale)),
        }
    }

    pub fn compression_scheme(&self) -> CompressionScheme {
        CompressionScheme::from(self)
    }
}

impl From<&CompressorConfig> for CompressionScheme {
    fn from(value: &CompressorConfig) -> Self {
        match value {
            CompressorConfig::Uncompressed(_) => Self::Uncompressed,
            CompressorConfig::RLE(_) => Self::RLE,
            CompressorConfig::Gorilla(_) => Self::Gorilla,
            CompressorConfig::BUFF(_) => Self::BUFF,
        }
    }
}

impl CompressorConfig {
    pub fn uncompressed() -> UncompressedCompressorConfigBuilder {
        UncompressedCompressorConfigBuilder::default()
    }

    pub fn rle() -> RunLengthCompressorConfigBuilder {
        RunLengthCompressorConfigBuilder::default()
    }

    pub fn gorilla() -> GorillaCompressorConfigBuilder {
        GorillaCompressorConfigBuilder::default()
    }

    pub fn buff() -> BUFFCompressorConfigBuilder {
        BUFFCompressorConfigBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
struct Capacity(usize);

const DEFAULT_CAPACITY: usize = 1024 * 8;
impl Default for Capacity {
    fn default() -> Self {
        Capacity(DEFAULT_CAPACITY)
    }
}
impl From<usize> for Capacity {
    fn from(value: usize) -> Self {
        Capacity(value)
    }
}

#[derive(Builder, Debug, Clone)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct UncompressedCompressorConfig {
    #[builder(setter(into), default)]
    capacity: Capacity,
    #[builder(setter(into, strip_option), default)]
    header: Option<Box<[u8]>>,
}

impl UncompressedCompressorConfigBuilder {
    pub fn build(self) -> UncompressedCompressorConfig {
        let Self { capacity, header } = self;
        UncompressedCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
            header: header.unwrap_or(None),
        }
    }
}

#[derive(Builder, Debug, Clone, Copy)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct RunLengthCompressorConfig {
    #[builder(setter(into), default)]
    capacity: Capacity,
}

impl RunLengthCompressorConfigBuilder {
    pub fn build(self) -> RunLengthCompressorConfig {
        let Self { capacity } = self;
        RunLengthCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
        }
    }
}

#[derive(Builder, Debug, Clone, Copy)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct GorillaCompressorConfig {
    #[builder(setter(into), default)]
    capacity: Capacity,
}

impl GorillaCompressorConfigBuilder {
    pub fn build(self) -> GorillaCompressorConfig {
        let Self { capacity } = self;
        GorillaCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
        }
    }
}

#[derive(Builder, Debug, Clone, Copy)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct BUFFCompressorConfig {
    scale: usize,
}

impl BUFFCompressorConfigBuilder {
    pub fn build(self) -> BUFFCompressorConfig {
        BUFFCompressorConfig {
            scale: self.scale.unwrap_or(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of_val;

    use super::{gorilla::GorillaCompressor, CompressorConfig, GenericCompressor};

    fn test_total_bytes_in(compressor: &mut GenericCompressor) {
        let mut floats: Vec<f64> = (0..10).map(|x| (x / 2) as f64).collect();
        assert_eq!(
            compressor.total_bytes_in(),
            0,
            "total_bytes_in() should be 0 before compressing"
        );

        let bytes0 = compressor.compress(&floats);

        assert_eq!(
            bytes0,
            size_of_val(&floats[..]),
            "the return value of compress() should be the size of the input"
        );

        assert_eq!(
            compressor.total_bytes_in(),
            size_of_val(&floats[..]),
            "total_bytes_in() should be the size of the input after compressing"
        );

        floats.reverse();
        let bytes1 = compressor.compress(&floats[..3]);

        assert_eq!(
            bytes1,
            size_of_val(&floats[..3]),
            "the return value of compress() should be the size of the input"
        );

        assert_eq!(
            compressor.total_bytes_in(),
            size_of_val(&floats[..]) + size_of_val(&floats[..3]),
            "total_bytes_in() should be the size of the total bytes of input"
        );
    }

    fn test_total_bytes_buffered(compressor: &mut GenericCompressor) {
        let mut floats: Vec<f64> = (0..10).map(|x| (x / 2) as f64).collect();

        compressor.compress(&floats);

        floats.reverse();
        compressor.compress(&floats[..3]);

        let total_bytes_buffered = compressor.total_bytes_buffered();

        let mut total_bytes_buffered_calculated = 0;

        compressor.prepare();
        for b in compressor.buffers() {
            total_bytes_buffered_calculated += b.len();
        }

        assert_eq!(
            total_bytes_buffered, total_bytes_buffered_calculated,
            "total_bytes_buffered() should be the sum of the buffer sizes"
        );
    }

    fn test_reset(compressor: &mut GenericCompressor) {
        compressor.reset();

        assert_eq!(
            compressor.total_bytes_in(),
            0,
            "total_bytes_in() should be 0 after reset"
        );
    }

    #[test]
    fn test_uncompressed() {
        let mut compressor =
            GenericCompressor::Uncompressed(super::uncompressed::UncompressedCompressor::new(10));

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_rle() {
        let mut compressor = GenericCompressor::RLE(super::run_length::RunLengthCompressor::new());

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_gorilla() {
        let mut compressor = GenericCompressor::Gorilla(GorillaCompressor::new());

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_buff() {
        return;
        let config: CompressorConfig = CompressorConfig::buff().build().into();
        let mut compressor = config.build();

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }
}
