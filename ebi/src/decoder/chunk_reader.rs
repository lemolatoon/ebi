pub mod buff;
pub mod gorilla;
pub mod run_length;
pub mod uncompressed;

use std::io::{self, Read, Write};

use quick_impl::QuickImpl;
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
    BUFF(buff::BUFFReader<R>),
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
            GeneralChunkReaderInner::BUFF(_) => CompressionScheme::BUFF,
        }
    }
}

pub trait Reader {
    type NativeHeader;
    type DecompressIterator<'a>: Iterator<Item = io::Result<f64>>
    where
        Self: 'a;

    /// Returns `impl Iterator<Item = io::Result<f64>>`, which decompresses the chunk iteratively.
    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_>;

    /// Decompress the whole chunk and return the slice of the decompressed values.
    fn decompress(&mut self) -> io::Result<&[f64]> {
        if self.decompress_result().is_some() {
            return Ok(self.decompress_result().unwrap());
        }

        let data = self.decompress_iter().collect::<io::Result<Vec<f64>>>()?;
        let result = self.set_decompress_result(data);

        Ok(result)
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64];

    fn decompress_result(&mut self) -> Option<&[f64]>;

    /// Returns the number of bytes of the method specific header.
    fn header_size(&self) -> usize;

    /// Read header from the internal buffer and return the reference to the header.
    /// Potentially, the header is cached internally.
    fn read_header(&mut self) -> &Self::NativeHeader;
}

#[derive(QuickImpl)]
pub enum GeneralDecompressIterator<'a, R: Read> {
    #[quick_impl(impl From)]
    Uncompressed(uncompressed::UncompressedIterator<'a, R>),
    #[quick_impl(impl From)]
    RLE(run_length::RunLengthIterator<'a, R>),
    #[quick_impl(impl From)]
    Gorilla(gorilla::GorillaIterator<'a, R>),
    #[quick_impl(impl From)]
    BUFF(buff::BUFFIterator<'a>),
}

impl<'a, R: Read> Iterator for GeneralDecompressIterator<'a, R> {
    type Item = io::Result<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            GeneralDecompressIterator::Uncompressed(c) => c.next(),
            GeneralDecompressIterator::RLE(c) => c.next(),
            GeneralDecompressIterator::Gorilla(c) => c.next(),
            GeneralDecompressIterator::BUFF(c) => c.next(),
        }
    }
}

macro_rules! impl_generic_reader {
    ($enum_name:ident, $($variant:ident),*) => {
        impl<R: Read> $enum_name<R> {
            /// Decompress the whole chunk and return the slice of the decompressed values.
            pub fn decompress(&mut self) -> io::Result<&[f64]> {
                match self {
                    $( $enum_name::$variant(c) => c.decompress(), )*
                }
            }

            /// Returns `impl Iterator<Item = io::Result<f64>>`, which decompresses the chunk iteratively.
            pub fn decompress_iter(&mut self) -> GeneralDecompressIterator<'_, R> {
                match self {
                    $( $enum_name::$variant(c) => c.decompress_iter().into(), )*
                }
            }

            pub fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
                match self {
                    $( $enum_name::$variant(c) => c.set_decompress_result(data), )*
                }
            }

            pub fn decompress_result(&mut self) -> Option<&[f64]> {
                match self {
                    $( $enum_name::$variant(c) => c.decompress_result(), )*
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

impl_generic_reader!(GeneralChunkReaderInner, Uncompressed, RLE, Gorilla, BUFF);

#[cfg(test)]
mod tests {
    use io::Seek;
    use rand::Rng;

    use crate::{
        api::{
            decoder::{ChunkId, Decoder, DecoderInput},
            encoder::{Encoder, EncoderInput, EncoderOutput},
        },
        compressor::CompressorConfig,
        encoder::ChunkOption,
    };

    use super::*;

    fn generate_and_write_random_f64(n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut random_values: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            if rng.gen_bool(0.5) && random_values.last().is_some() {
                random_values.push(random_values[i - 1]);
            } else {
                random_values.push(rng.gen());
            }
        }

        random_values
    }

    fn decoder(
        values: &[f64],
        compressor_config: impl Into<CompressorConfig>,
    ) -> Decoder<io::Cursor<Vec<u8>>> {
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(values);

            let encoder_output = EncoderOutput::from_vec(Vec::new());

            let chunk_option = ChunkOption::RecordCount(512);

            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                chunk_option,
                compressor_config.into(),
            );

            encoder.encode().unwrap();

            encoder.into_output().into_vec()
        };

        let decoder_input = DecoderInput::from_reader(io::Cursor::new(encoded));

        Decoder::new(decoder_input).unwrap()
    }

    fn test_all(compresor_config: impl Into<CompressorConfig>) {
        let values = generate_and_write_random_f64(1003);
        let mut decoder = decoder(values.as_slice(), compresor_config.into());

        let mut reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        test_compressed_result(reader.inner_mut());

        test_decompress_iter(&mut decoder);
    }

    fn test_compressed_result<R: Read>(reader: &mut GeneralChunkReaderInner<R>) {
        let result = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = reader.set_decompress_result(result);

        assert_eq!(
            data,
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            "set_decompress_result should return the data set by set_decompress_result"
        );

        assert_eq!(
            reader.decompress_result().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            "decompress_result should return the data set by set_decompress_result"
        );
    }

    fn test_decompress_iter<R: Read + Seek>(decoder: &mut Decoder<R>) {
        let mut reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();
        let iter_result = reader
            .inner_mut()
            .decompress_iter()
            .collect::<io::Result<Vec<f64>>>()
            .unwrap();

        let mut reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();
        let decompress_result = reader.inner_mut().decompress().unwrap();

        assert_eq!(
            iter_result.len(),
            decompress_result.len(),
            "decompress_iter should return the same length of result as decompress"
        );

        assert_eq!(
            iter_result, decompress_result,
            "decompress_iter should return the same result as decompress"
        );
    }

    #[test]
    fn test_gorilla() {
        test_all(CompressorConfig::gorilla().build());
    }

    #[test]
    fn test_rle() {
        test_all(CompressorConfig::rle().build());
    }

    #[test]
    fn test_uncompressed() {
        test_all(CompressorConfig::uncompressed().build());
    }

    #[test]
    fn test_buff() {
        test_all(CompressorConfig::buff().build());
    }
}
