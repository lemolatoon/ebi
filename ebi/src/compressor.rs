use std::io;

use buff::BUFFCompressor;
use chimp::ChimpCompressor;
use chimp_n::Chimp128Compressor;
use gorilla::GorillaCompressor;
use quick_impl::QuickImpl;
use run_length::RunLengthCompressor;
use uncompressed::{UncompressedCompressor, UncompressedCompressorConfig};

use crate::{
    encoder,
    format::{
        self,
        deserialize::FromLeBytes as _,
        serialize::{AsBytes, ToLe},
        CompressionScheme,
    },
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub mod buff;
pub mod chimp;
pub mod chimp_n;
pub mod elf;
pub mod general_xor;
pub mod gorilla;
pub mod run_length;
pub mod sprintz;
pub mod uncompressed;
// pub mod zstd;

const MAX_BUFFERS: usize = 5;
pub trait Compressor {
    /// Perform the compression and return the size of the compressed data.
    fn compress(&mut self, input: &[f64]) -> encoder::Result<()>;

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
    DeltaSprintz(sprintz::DeltaSprintzCompressor),
}

macro_rules! impl_generic_compressor {
    ($enum_name:ident, $($variant:ident),*) => {
        impl Compressor for $enum_name {
            fn compress(&mut self, input: &[f64]) -> encoder::Result<()> {
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

        impl $enum_name {
            pub fn compression_scheme(&self) -> format::CompressionScheme {
                match self {
                    $( $enum_name::$variant(_c) => format::CompressionScheme::$variant, )*
                }
            }
        }

    };
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
    Elf,
    DeltaSprintz
);

#[derive(QuickImpl, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum CompressorConfig {
    #[quick_impl(impl From)]
    Uncompressed(uncompressed::UncompressedCompressorConfig),
    #[quick_impl(impl From)]
    RLE(run_length::RunLengthCompressorConfig),
    #[quick_impl(impl From)]
    Gorilla(gorilla::GorillaCompressorConfig),
    #[quick_impl(impl From)]
    BUFF(buff::BUFFCompressorConfig),
    #[quick_impl(impl From)]
    Chimp(chimp::ChimpCompressorConfig),
    #[quick_impl(impl From)]
    Chimp128(chimp_n::Chimp128CompressorConfig),
    #[quick_impl(impl From)]
    ElfOnChimp(elf::on_chimp::ElfCompressorConfig),
    #[quick_impl(impl From)]
    Elf(elf::ElfCompressorConfig),
    #[quick_impl(impl From)]
    DeltaSprintz(sprintz::DeltaSprintzCompressorConfig),
}

macro_rules! impl_compressor_config {
    ($enum_name:ident, $($variant:ident),*) => {
        impl $enum_name {
            pub fn build(self) -> GenericCompressor {
                match self {
                    $( $enum_name::$variant(c) => GenericCompressor::$variant(c.into()), )*
                }
            }
        }

        impl From<&$enum_name> for CompressionScheme {
            fn from(value: &$enum_name) -> Self {
                match value {
                    $( $enum_name::$variant(_) => CompressionScheme::$variant, )*
                }
            }
        }
    };
}

impl_compressor_config!(
    CompressorConfig,
    Uncompressed,
    RLE,
    Gorilla,
    BUFF,
    Chimp,
    Chimp128,
    ElfOnChimp,
    Elf,
    DeltaSprintz
);

impl CompressorConfig {
    pub fn compression_scheme(&self) -> CompressionScheme {
        CompressionScheme::from(self)
    }

    /// Returns the size of the serialized configuration in bytes.
    /// Includes the first byte that represents the size of the configuration itself.
    pub fn serialized_size(&self) -> usize {
        // 1 byte for the size of the config
        1 + match self {
            CompressorConfig::Uncompressed(c) => c.serialized_size(),
            CompressorConfig::RLE(c) => c.serialized_size(),
            CompressorConfig::Gorilla(c) => c.serialized_size(),
            CompressorConfig::BUFF(c) => c.serialized_size(),
            CompressorConfig::Chimp(c) => c.serialized_size(),
            CompressorConfig::Chimp128(c) => c.serialized_size(),
            CompressorConfig::ElfOnChimp(c) => c.serialized_size(),
            CompressorConfig::Elf(c) => c.serialized_size(),
            CompressorConfig::DeltaSprintz(c) => c.serialized_size(),
        }
    }

    /// Serialize the compressor configuration to the given writer.
    /// The first byte is the size of the configuration in bytes.
    /// The rest of the bytes are the serialized configuration.
    pub fn serialize<W: io::Write>(self, mut w: W) -> io::Result<()> {
        macro_rules! just_write {
            ($comp:expr, $w:expr) => {{
                let size = size_of_val(&$comp) as u8;
                w.write_all(&[size])?;
                let le_bytes = $comp.to_le().as_bytes();
                w.write_all(le_bytes)?;
            }};
        }
        macro_rules! into_packed_and_write {
            ($comp:expr, $w:expr) => {{
                let mut packed = $comp.into_packed();
                let size = size_of_val(&packed) as u8;
                w.write_all(&[size])?;
                w.write_all(packed.to_le().as_bytes())?;
            }};
        }
        match self {
            CompressorConfig::Uncompressed(mut c) => just_write!(c, w),
            CompressorConfig::RLE(mut c) => just_write!(c, w),
            CompressorConfig::Gorilla(mut c) => just_write!(c, w),
            CompressorConfig::BUFF(mut c) => just_write!(c, w),
            CompressorConfig::Chimp(c) => into_packed_and_write!(c, w),
            CompressorConfig::Chimp128(mut c) => just_write!(c, w),
            CompressorConfig::ElfOnChimp(c) => into_packed_and_write!(c, w),
            CompressorConfig::Elf(c) => into_packed_and_write!(c, w),
            CompressorConfig::DeltaSprintz(mut c) => just_write!(c, w),
        }

        Ok(())
    }

    pub fn from_le_bytes(compression_scheme: &CompressionScheme, bytes: &[u8]) -> Self {
        match compression_scheme {
            CompressionScheme::Uncompressed => {
                Self::Uncompressed(UncompressedCompressorConfig::from_le_bytes(bytes))
            }
            CompressionScheme::RLE => {
                Self::RLE(run_length::RunLengthCompressorConfig::from_le_bytes(bytes))
            }
            CompressionScheme::BUFF => Self::BUFF(buff::BUFFCompressorConfig::from_le_bytes(bytes)),
            CompressionScheme::Gorilla => {
                Self::Gorilla(gorilla::GorillaCompressorConfig::from_le_bytes(bytes))
            }
            CompressionScheme::Chimp => {
                Self::Chimp(chimp::ChimpCompressorConfig::from_le_bytes(bytes))
            }
            CompressionScheme::Chimp128 => {
                Self::Chimp128(chimp_n::Chimp128CompressorConfig::from_le_bytes(bytes))
            }
            CompressionScheme::ElfOnChimp => {
                Self::ElfOnChimp(elf::on_chimp::ElfCompressorConfig::from_le_bytes(bytes))
            }
            CompressionScheme::Elf => Self::Elf(elf::ElfCompressorConfig::from_le_bytes(bytes)),
            CompressionScheme::DeltaSprintz => {
                Self::DeltaSprintz(sprintz::DeltaSprintzCompressorConfig::from_le_bytes(bytes))
            }
        }
    }
}

impl CompressorConfig {
    pub fn uncompressed() -> uncompressed::UncompressedCompressorConfigBuilder {
        uncompressed::UncompressedCompressorConfigBuilder::default()
    }

    pub fn rle() -> run_length::RunLengthCompressorConfigBuilder {
        run_length::RunLengthCompressorConfigBuilder::default()
    }

    pub fn gorilla() -> gorilla::GorillaCompressorConfigBuilder {
        gorilla::GorillaCompressorConfigBuilder::default()
    }

    pub fn buff() -> buff::BUFFCompressorConfigBuilder {
        buff::BUFFCompressorConfigBuilder::default()
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

    pub fn delta_sprintz() -> sprintz::DeltaSprintzCompressorConfigBuilder {
        sprintz::DeltaSprintzCompressorConfigBuilder::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct Capacity(pub(crate) u64);

const DEFAULT_CAPACITY: u64 = 1024 * 8;
impl Default for Capacity {
    fn default() -> Self {
        Capacity(DEFAULT_CAPACITY)
    }
}
impl From<u64> for Capacity {
    fn from(value: u64) -> Self {
        Capacity(value)
    }
}
impl Capacity {
    fn to_le(self) -> Capacity {
        Self(self.0.to_le())
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of_val;

    use crate::compressor::Compressor;

    use super::{CompressorConfig, GenericCompressor};

    fn test_total_bytes_in(compressor: &mut GenericCompressor) {
        let floats: Vec<f64> = (0..10).map(|x| (x / 2) as f64).collect();
        assert_eq!(
            compressor.total_bytes_in(),
            0,
            "total_bytes_in() should be 0 before compressing"
        );

        compressor.compress(&floats).unwrap();

        assert_eq!(
            compressor.total_bytes_in(),
            size_of_val(&floats[..]),
            "total_bytes_in() should be the size of the input after compressing"
        );
    }

    fn test_total_bytes_buffered(compressor: &mut GenericCompressor) {
        let mut floats: Vec<f64> = (0..10).map(|x| (x / 2) as f64).collect();

        compressor.compress(&floats).unwrap();

        floats.reverse();
        compressor.compress(&floats[..3]).unwrap();

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

    macro_rules! declare_test_compressor {
        ($method:ident) => {
            declare_test_compressor!($method, super::CompressorConfig::$method().build());
        };
        ($method:ident, $custom_config:expr) => {
            mod $method {
                #[test]
                fn test_compressor_consistency() {
                    let config: super::CompressorConfig = $custom_config.into();
                    let mut compressor = config.build();

                    super::test_total_bytes_in(&mut compressor);
                    super::test_total_bytes_buffered(&mut compressor);

                    super::test_reset(&mut compressor);

                    super::test_total_bytes_in(&mut compressor);
                    super::test_total_bytes_buffered(&mut compressor);
                }

                #[test]
                fn test_compressor_config_consistency() {
                    let config: super::CompressorConfig = $custom_config.into();
                    let size = config.serialized_size();
                    let mut cursor = std::io::Cursor::new(Vec::new());
                    config.serialize(&mut cursor).unwrap();

                    let bytes = cursor.into_inner();

                    assert_eq!(
                        bytes.len(),
                        size,
                        "serialized size should be the size of the configuration plus 1"
                    );

                    assert_eq!(
                        config,
                        super::CompressorConfig::from_le_bytes(
                            &config.compression_scheme(),
                            &bytes[1..]
                        ),
                    )
                }
            }
        };
    }

    declare_test_compressor!(uncompressed);
    declare_test_compressor!(rle);
    declare_test_compressor!(gorilla);
    declare_test_compressor!(chimp);
    declare_test_compressor!(chimp128);
    declare_test_compressor!(elf_on_chimp);
    declare_test_compressor!(elf);
    declare_test_compressor!(delta_sprintz);
    declare_test_compressor!(buff, super::CompressorConfig::buff().scale(100).build());
}
