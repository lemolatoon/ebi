use gorilla::GorillaCompressor;
use run_length::RunLengthCompressor;
// use run_length::RunLengthCompressor;
use uncompressed::UncompressedCompressor;

use crate::format;

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
        }
    }
}

impl_generic_compressor!(GenericCompressor, Uncompressed, RLE, Gorilla);

#[cfg(test)]
mod tests {
    use std::mem::size_of_val;

    use super::{gorilla::GorillaCompressor, GenericCompressor};

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
}
