pub mod run_length;
pub mod uncompressed;

use std::mem::{align_of, size_of};

use crate::format::native::{NativeFileFooter, NativeFileHeader};
use crate::format::{CompressionScheme, GeneralChunkHeader};

use super::Result;
use super::{error::DecoderError, GeneralChunkHandle};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralChunkReader<'reader, 'chunk> {
    handle: &'reader GeneralChunkHandle<'reader>,
    chunk: &'chunk [u8],
    reader: GeneralChunkReaderInner<'chunk>,
}

impl<'reader, 'chunk> GeneralChunkReader<'reader, 'chunk> {
    /// Create a new GeneralChunkReader.
    /// Caller must guarantee that the input chunk is valid.
    /// # chunk format
    /// ```text
    /// GeneralChunkHeader | MethodSpecificHeader | (padding for 64bit alignment) | Data
    /// ```
    /// The chunk begins with the header including the method specific header.
    /// The header is followed by the data.
    /// The data is 64bit aligned, so there may be padding between the header and the data.
    pub fn new(handle: &'reader GeneralChunkHandle<'reader>, chunk: &'chunk [u8]) -> Result<Self> {
        let chunk_size = handle.chunk_size() as usize;
        if chunk.len() < chunk_size.next_multiple_of(align_of::<u64>()) / size_of::<u64>() + 1 {
            return Err(DecoderError::BufferTooSmall);
        }

        let compression_scheme = handle.header().config().compression_scheme();

        let reader_inner = GeneralChunkReaderInner::new(handle, chunk, *compression_scheme);

        Ok(Self {
            handle,
            chunk,
            reader: reader_inner,
        })
    }

    /// Returns the raw chunk data including the chunk header.
    pub fn chunk(&self) -> &[u8] {
        self.chunk
    }

    /// Returns the raw data of the chunk, which does not contain the chunk header.
    pub fn data(&self) -> &[u8] {
        let header_size = self.reader.header_size();

        &self.chunk[size_of::<GeneralChunkHeader>() + header_size..]
    }

    pub fn header(&self) -> &NativeFileHeader {
        self.handle.header()
    }

    pub fn footer(&self) -> &NativeFileFooter {
        self.handle.footer()
    }

    pub fn inner(&self) -> &GeneralChunkReaderInner<'chunk> {
        &self.reader
    }

    pub fn inner_mut(&mut self) -> &mut GeneralChunkReaderInner<'chunk> {
        &mut self.reader
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum GeneralChunkReaderInner<'chunk> {
    Uncompressed(uncompressed::UncompressedReader<'chunk>),
    RLE(run_length::RunLengthReader<'chunk>),
}

impl<'chunk> GeneralChunkReaderInner<'chunk> {
    pub fn new(
        handle: &GeneralChunkHandle,
        chunk: &'chunk [u8],
        compression_scheme: CompressionScheme,
    ) -> Self {
        match compression_scheme {
            CompressionScheme::Uncompressed => GeneralChunkReaderInner::Uncompressed(
                uncompressed::UncompressedReader::new(handle, chunk),
            ),
            CompressionScheme::RLE => {
                GeneralChunkReaderInner::RLE(run_length::RunLengthReader::new(handle, chunk))
            }
            c => unimplemented!("Unimplemented compression scheme: {:?}", c),
        }
    }
}

impl From<&GeneralChunkReaderInner<'_>> for CompressionScheme {
    fn from(value: &GeneralChunkReaderInner<'_>) -> Self {
        match value {
            GeneralChunkReaderInner::Uncompressed(_) => CompressionScheme::Uncompressed,
            GeneralChunkReaderInner::RLE(_) => CompressionScheme::RLE,
        }
    }
}

pub trait Reader {
    type NativeHeader;

    /// Decompress the whole chunk and return the slice of the decompressed values.
    fn decompress(&mut self) -> &[f64];

    /// Returns the number of bytes of the method specific header.
    fn header_size(&self) -> usize;

    /// Read header from the internal buffer and return the reference to the header.
    /// Potentially, the header is cached internally.
    fn read_header(&mut self) -> &Self::NativeHeader;
}

macro_rules! impl_generic_reader {
    ($enum_name:ident, $($variant:ident),*) => {
        impl<'a> $enum_name<'a> {
            pub fn decompress(&mut self) -> &[f64] {
                match self {
                    $( $enum_name::$variant(c) => c.decompress(), )*
                }
            }

            pub fn header_size(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.header_size(), )*
                }
            }
        }
    };
}

impl_generic_reader!(GeneralChunkReaderInner, Uncompressed, RLE);
