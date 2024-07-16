use run_length::RunLengthCompressor;
// use run_length::RunLengthCompressor;
use uncompressed::UncompressedCompressor;

use crate::format;

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
        }
    }
}

impl_generic_compressor!(GenericCompressor, Uncompressed, RLE);
