use std::any::Any;

use uncompressed::UncompressedCompressor;

pub mod uncompressed;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkOption {
    Full,
    RecordCount(usize),
    ByteSize(usize),
}

pub trait Compressor {
    /// Take as many values from input, and compress and write it to output as possible.
    /// Returns the number of bytes consumed from input.
    fn compress(&mut self, input: &[f64], output: &mut [u8]) -> usize;

    /// Returns the total number of input bytes which have been processed by this Compressor.
    fn total_bytes_in(&self) -> usize;

    /// Returns the total number of output bytes which have been produced by this Compressor.
    fn total_bytes_out(&self) -> usize;
}

pub enum GenericCompressor {
    Uncompressed(UncompressedCompressor),
}

macro_rules! impl_compressor_for_enum {
    ($enum_name:ident, $($variant:ident),*) => {
        impl Compressor for $enum_name {
            fn compress(&mut self, input: &[f64], output: &mut [u8]) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.compress(input, output), )*
                }
            }

            fn total_bytes_in(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_in(), )*
                }
            }

            fn total_bytes_out(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.total_bytes_out(), )*
                }
            }
        }
    };
}

impl_compressor_for_enum!(GenericCompressor, Uncompressed);
