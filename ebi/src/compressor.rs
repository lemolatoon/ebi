use buff::BUFFCompressor;
use chimp::{ChimpCompressor, ChimpCompressorConfig};
use chimp_n::{Chimp128Compressor, Chimp128CompressorConfig};
use derive_builder::Builder;
use gorilla::GorillaCompressor;
use quick_impl::QuickImpl;
use run_length::RunLengthCompressor;
// use run_length::RunLengthCompressor;
use uncompressed::UncompressedCompressor;

use crate::format::{self, CompressionScheme};

pub mod buff;
pub mod chimp;
pub mod chimp_n;
pub mod elf;
pub mod general_xor;
pub mod gorilla;
pub mod run_length;
pub mod uncompressed;

const MAX_BUFFERS: usize = 5;
pub trait Compressor {
    /// Perform the compression and return the size of the compressed data.
    fn compress(&mut self, input: &[f64]);

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

pub trait AppendableCompressor: Compressor {
    /// Append the input data and compress with the existing data.
    /// This method is NOT re-compressing the existing data.
    /// Returns the total size of the compressed data.
    fn append_compress(&mut self, input: &[f64]);
}

pub trait RewindableCompressor: AppendableCompressor {
    /// Rewind the n records from the end of the compressed data
    /// Returns true if the rewind is successful, false otherwise.
    fn rewind(&mut self, n: usize) -> bool;
}

#[derive(QuickImpl, Debug, Clone, PartialEq, PartialOrd)]
pub enum GenericCompressor {
    Uncompressed(UncompressedCompressor),
    RLE(RunLengthCompressor),
    Gorilla(GorillaCompressor),
    BUFF(BUFFCompressor),
    Chimp(ChimpCompressor),
    Chimp128(Chimp128Compressor),
    ElfOnChimp(elf::on_chimp::ElfCompressor),
    Elf(elf::ElfCompressor),
}

macro_rules! impl_generic_compressor {
    ($enum_name:ident, $($variant:ident),*) => {
        impl Compressor for $enum_name {
            fn compress(&mut self, input: &[f64]) {
                match self {
                    $( $enum_name::$variant(c) => c.compress(input), )*
                }
            }

            fn total_bytes_in(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_in(), )*
                }
            }

            fn total_bytes_buffered(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_buffered(), )*
                }
            }

            fn prepare(&mut self) {
                match self {
                    $( $enum_name::$variant(c) => c.prepare(), )*
                }
            }

            fn buffers(&self) -> [&[u8]; MAX_BUFFERS] {
                match self {
                    $( $enum_name::$variant(c) => c.buffers(), )*
                }
            }

            fn reset(&mut self) {
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
            GenericCompressor::Chimp(_) => format::CompressionScheme::Chimp,
            GenericCompressor::Chimp128(_) => format::CompressionScheme::Chimp128,
            GenericCompressor::ElfOnChimp(_) => format::CompressionScheme::ElfOnChimp,
            GenericCompressor::Elf(_) => format::CompressionScheme::Elf,
        }
    }
}

impl_generic_compressor!(
    GenericCompressor,
    Uncompressed,
    RLE,
    Gorilla,
    BUFF,
    Chimp,
    Chimp128,
    ElfOnChimp,
    Elf
);

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
    #[quick_impl(impl From)]
    Chimp(ChimpCompressorConfig),
    #[quick_impl(impl From)]
    Chimp128(Chimp128CompressorConfig),
    #[quick_impl(impl From)]
    ElfOnChimp(elf::on_chimp::ElfCompressorConfig),
    #[quick_impl(impl From)]
    Elf(elf::ElfCompressorConfig),
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
            CompressorConfig::Chimp(c) => {
                GenericCompressor::Chimp(ChimpCompressor::with_capacity(c.capacity.0))
            }
            CompressorConfig::Chimp128(c) => {
                GenericCompressor::Chimp128(Chimp128Compressor::with_capacity(c.capacity.0))
            }
            CompressorConfig::ElfOnChimp(c) => GenericCompressor::ElfOnChimp(c.into()),
            CompressorConfig::Elf(c) => GenericCompressor::Elf(c.into()),
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
            CompressorConfig::Chimp(_) => Self::Chimp,
            CompressorConfig::Chimp128(_) => Self::Chimp128,
            CompressorConfig::ElfOnChimp(_) => Self::ElfOnChimp,
            CompressorConfig::Elf(_) => Self::Elf,
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

    pub fn chimp() -> chimp::ChimpCompressorConfigBuilder {
        chimp::ChimpCompressorConfigBuilder::default()
    }

    pub fn chimp128() -> chimp_n::Chimp128CompressorConfigBuilder {
        chimp_n::Chimp128CompressorConfigBuilder::default()
    }

    pub fn elf_on_chimp() -> elf::on_chimp::ElfCompressorConfigBuilder {
        elf::on_chimp::ElfCompressorConfigBuilder::default()
    }

    pub fn elf() -> elf::ElfCompressorConfigBuilder {
        elf::ElfCompressorConfigBuilder::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Capacity(usize);

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

    use crate::compressor::Compressor;

    use super::{gorilla::GorillaCompressor, CompressorConfig, GenericCompressor};

    fn test_total_bytes_in(compressor: &mut GenericCompressor) {
        let floats: Vec<f64> = (0..10).map(|x| (x / 2) as f64).collect();
        assert_eq!(
            compressor.total_bytes_in(),
            0,
            "total_bytes_in() should be 0 before compressing"
        );

        compressor.compress(&floats);

        assert_eq!(
            compressor.total_bytes_in(),
            size_of_val(&floats[..]),
            "total_bytes_in() should be the size of the input after compressing"
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
        let config: CompressorConfig = CompressorConfig::buff().build().into();
        let mut compressor = config.build();

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_chimp() {
        let config: CompressorConfig = CompressorConfig::chimp().build().into();
        let mut compressor = config.build();

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_chimp128() {
        let config: CompressorConfig = CompressorConfig::chimp128().build().into();
        let mut compressor = config.build();

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_elf_on_chimp() {
        let config: CompressorConfig = CompressorConfig::elf_on_chimp().build().into();
        let mut compressor = config.build();

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }

    #[test]
    fn test_elf() {
        let config: CompressorConfig = CompressorConfig::elf().build().into();
        let mut compressor = config.build();

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);

        test_reset(&mut compressor);

        test_total_bytes_in(&mut compressor);
        test_total_bytes_buffered(&mut compressor);
    }
}
