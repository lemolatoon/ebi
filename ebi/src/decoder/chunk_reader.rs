pub mod gorilla;
pub mod run_length;
pub mod uncompressed;

use std::io::{self, Read, Write};

use roaring::RoaringBitmap;

use crate::format::native::{NativeFileFooter, NativeFileHeader};
use crate::format::CompressionScheme;

use super::query::{Predicate, QueryExecutor};
use super::GeneralChunkHandle;
use super::{FileMetadataLike, Result};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralChunkReader<'handle, R: Read, T: FileMetadataLike> {
    handle: &'handle GeneralChunkHandle<T>,
    reader: GeneralChunkReaderInner<R>,
}

impl<'handle, R: Read, T: FileMetadataLike> GeneralChunkReader<'handle, R, T> {
    /// Create a new GeneralChunkReader.
    /// Caller must guarantee that the input chunk is valid.
    /// # chunk format
    /// ```text
    /// GeneralChunkHeader | MethodSpecificHeader | (padding for 64bit alignment) | Data
    /// ```
    /// The chunk begins with the header including the method specific header.
    /// The header is followed by the data.
    /// The data is 64bit aligned, so there may be padding between the header and the data.
    pub fn new(handle: &'handle GeneralChunkHandle<T>, reader: R) -> Result<Self> {
        let compression_scheme = handle.header().config().compression_scheme();

        let reader_inner = GeneralChunkReaderInner::new(handle, reader, *compression_scheme)?;

        Ok(Self {
            handle,
            reader: reader_inner,
        })
    }

    pub fn header(&self) -> &NativeFileHeader {
        self.handle.header()
    }

    pub fn footer(&self) -> &NativeFileFooter {
        self.handle.footer()
    }

    pub fn inner(&self) -> &GeneralChunkReaderInner<R> {
        &self.reader
    }

    pub fn inner_mut(&mut self) -> &mut GeneralChunkReaderInner<R> {
        &mut self.reader
    }

    /// Materialize the values filtered by the bitmask and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index must be global to the whole chunks.
    pub fn materialize(
        &mut self,
        output: &mut impl Write,
        bitmask: Option<&RoaringBitmap>,
    ) -> Result<()> {
        let logical_offset = self.handle.chunk_footer().logical_offset() as usize;
        self.reader.materialize(output, bitmask, logical_offset)
    }

    /// Filter the values by the predicate and return the result as a bitmask.
    /// The result bitmask is global to the whole chunks.
    /// But the result bitmask is guaranteed to only contain the record offsets in the current chunk.
    ///
    /// `bitmask` is optional. If it is None, all values are evaluated by `predicate`.
    ///
    /// `bitmask`'s index must be global to the whole chunks.
    pub fn filter(
        &mut self,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
    ) -> Result<RoaringBitmap> {
        let logical_offset = self.handle.chunk_footer().logical_offset() as usize;
        self.reader.filter(predicate, bitmask, logical_offset)
    }

    /// Filter the values by the predicate and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are filtered and then scaned.
    ///
    /// `bitmask`'s index must be global to the whole chunks.
    pub fn filter_materialize(
        &mut self,
        output: &mut impl Write,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
    ) -> Result<()> {
        let logical_offset = self.handle.chunk_footer().logical_offset() as usize;
        self.reader
            .filter_materialize(output, predicate, bitmask, logical_offset)
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum GeneralChunkReaderInner<R: Read> {
    Uncompressed(uncompressed::UncompressedReader<R>),
    RLE(run_length::RunLengthReader<R>),
    Gorilla(gorilla::GorillaReader<R>),
}

impl<R: Read> GeneralChunkReaderInner<R> {
    pub fn new<T: FileMetadataLike>(
        handle: &GeneralChunkHandle<T>,
        reader: R,
        compression_scheme: CompressionScheme,
    ) -> io::Result<Self> {
        Ok(match compression_scheme {
            CompressionScheme::Uncompressed => GeneralChunkReaderInner::Uncompressed(
                uncompressed::UncompressedReader::new(handle, reader)?,
            ),
            CompressionScheme::RLE => {
                GeneralChunkReaderInner::RLE(run_length::RunLengthReader::new(handle, reader)?)
            }
            CompressionScheme::Gorilla => {
                GeneralChunkReaderInner::Gorilla(gorilla::GorillaReader::new(handle, reader))
            }
            c => unimplemented!("Unimplemented compression scheme: {:?}", c),
        })
    }
}

impl<R: Read> From<&GeneralChunkReaderInner<R>> for CompressionScheme {
    fn from(value: &GeneralChunkReaderInner<R>) -> Self {
        match value {
            GeneralChunkReaderInner::Uncompressed(_) => CompressionScheme::Uncompressed,
            GeneralChunkReaderInner::RLE(_) => CompressionScheme::RLE,
            GeneralChunkReaderInner::Gorilla(_) => CompressionScheme::Gorilla,
        }
    }
}

pub trait Reader {
    type NativeHeader;

    /// Decompress the whole chunk and return the slice of the decompressed values.
    fn decompress(&mut self) -> io::Result<&[f64]>;

    /// Returns the number of bytes of the method specific header.
    fn header_size(&self) -> usize;

    /// Read header from the internal buffer and return the reference to the header.
    /// Potentially, the header is cached internally.
    fn read_header(&mut self) -> &Self::NativeHeader;
}

macro_rules! impl_generic_reader {
    ($enum_name:ident, $($variant:ident),*) => {
        impl<R: Read> $enum_name<R> {
            pub fn decompress(&mut self) -> io::Result<&[f64]> {
                match self {
                    $( $enum_name::$variant(c) => c.decompress(), )*
                }
            }

            pub fn header_size(&self) -> usize {
                match self {
                    $( $enum_name::$variant(c) => c.header_size(), )*
                }
            }

            /// Materialize the values filtered by the bitmask and write the results as IEEE754 double array to the output.
            ///
            /// `bitmask` is optional. If it is None, all values are written.
            ///
            /// `bitmask`'s index is global to the whole chunks.
            /// That is why `logical_offset` is necessary to access bitmask.
            pub fn materialize(
                &mut self,
                output: &mut impl Write,
                bitmask: Option<&RoaringBitmap>,
                logical_offset: usize,
            ) -> Result<()> {
                match self {
                    $( $enum_name::$variant(c) => c.materialize(output, bitmask, logical_offset), )*
                }
            }

            /// Filter the values by the predicate and return the result as a bitmask.
            /// The result bitmask is global to the whole chunks.
            /// But the result bitmask is guaranteed to only contain the record offsets in the current chunk.
            ///
            /// `bitmask` is optional. If it is None, all values are evaluated by `predicate`.
            ///
            /// `bitmask`'s index is global to the whole chunks.
            /// That is why `logical_offset` is necessary to access bitmask.
            pub fn filter(
                &mut self,
                predicate: Predicate,
                bitmask: Option<&RoaringBitmap>,
                logical_offset: usize,
            ) -> Result<RoaringBitmap> {
                match self {
                    $( $enum_name::$variant(c) => c.filter(predicate, bitmask, logical_offset), )*
                }
            }

            /// Filter the values by the predicate and write the results as IEEE754 double array to the output.
            ///
            /// `bitmask` is optional. If it is None, all values are filtered and then scaned.
            ///
            /// `bitmask`'s index is global to the whole chunks.
            /// That is why `logical_offset` is necessary to access bitmask.
            pub fn filter_materialize(
                &mut self,
                output: &mut impl Write,
                predicate: Predicate,
                bitmask: Option<&RoaringBitmap>,
                logical_offset: usize,
            ) -> Result<()> {
                match self {
                    $( $enum_name::$variant(c) => c.filter_materialize(output, predicate, bitmask, logical_offset), )*
                }
            }
        }
    };
}

impl_generic_reader!(GeneralChunkReaderInner, Uncompressed, RLE, Gorilla);
