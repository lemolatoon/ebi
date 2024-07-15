use uncompressed::UncompressedCompressor;

use crate::format;

pub mod uncompressed;

pub trait Compressor {
    /// Take as many values from input, and compress and write it to output as possible.
    /// Returns the number of bytes consumed from input.
    fn compress(&mut self, input: &[f64], output: &mut [u8]) -> usize;

    /// Returns the number of bytes of the method specific header.
    fn header_size(&self) -> usize;

    /// Write header bytes into output buffer.
    /// # Panics
    /// If `output.len()` < `Self::HEADER_SIZE()`, panics.
    fn write_header(&mut self, output: &mut [u8]);

    /// Returns the total number of input bytes which have been processed by this Compressor.
    fn total_bytes_in(&self) -> usize;

    /// Returns the total number of output bytes which have been produced by this Compressor.
    fn total_bytes_out(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GenericCompressor {
    Uncompressed(UncompressedCompressor),
}

macro_rules! impl_generic_compressor {
    ($enum_name:ident, $($variant:ident),*) => {
        impl $enum_name {
            pub fn compress(&mut self, input: &[f64], output: &mut [u8]) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.compress(input, output), )*
                }
            }

            pub fn header_size(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.header_size(), )*
                }
            }

            pub fn write_header(&mut self, output: &mut [u8]) {
                match self {
                    $( $enum_name::$variant(c) => c.write_header(output), )*
                }
            }

            pub fn total_bytes_in(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_in(), )*
                }
            }

            pub fn total_bytes_out(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_out(), )*
                }
            }
        }
    };
}

impl GenericCompressor {
    pub fn compression_scheme(&self) -> format::CompressionScheme {
        match self {
            GenericCompressor::Uncompressed(_) => format::CompressionScheme::Uncompressed,
        }
    }
}

impl_generic_compressor!(GenericCompressor, Uncompressed);
