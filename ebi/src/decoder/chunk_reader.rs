pub mod buff;
pub mod chimp;
pub mod chimp_n;
pub mod elf;
#[cfg(feature = "ffi_alp")]
pub mod ffi_alp;
pub mod general_xor;
pub mod gorilla;
pub mod gzip;
pub mod run_length;
pub mod snappy;
pub mod sprintz;
pub mod uncompressed;
pub mod zstd;

use std::io::{Read, Write};

use iter_enum::Iterator;
use quick_impl::QuickImpl;
use roaring::RoaringBitmap;

use crate::decoder;

use crate::format::native::{NativeChunkFooter, NativeFileFooter, NativeFileHeader};
use crate::format::CompressionScheme;
use crate::time::SegmentedExecutionTimes;

use super::query::{Predicate, QueryExecutor};
use super::GeneralChunkHandle;
use super::{FileMetadataLike, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GeneralChunkReader<'handle, T: FileMetadataLike, R: Read> {
    handle: &'handle GeneralChunkHandle<T>,
    reader: GeneralChunkReaderInner<R>,
    timer: SegmentedExecutionTimes,
}

impl<'handle, T: FileMetadataLike, R: Read> GeneralChunkReader<'handle, T, R> {
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

        let mut timer = SegmentedExecutionTimes::new();
        let reader_inner =
            GeneralChunkReaderInner::new(handle, reader, *compression_scheme, &mut timer)?;

        Ok(Self {
            handle,
            reader: reader_inner,
            timer,
        })
    }

    /// Advance the reader to the next chunk.
    ///
    /// # Preconditions
    /// - The reader must be at the end of the current chunk.
    pub fn advance(&mut self, next_handle: &'handle GeneralChunkHandle<T>) {
        self.handle = next_handle;
        self.timer = SegmentedExecutionTimes::new();
    }

    /// Set the precision of the reader, which is used when materializing.
    pub fn with_precision(&mut self, precision: u32) {
        if let GeneralChunkReaderInner::BUFF(buff) = &mut self.reader {
            buff.with_controlled_precision(precision);
        }
    }

    pub fn is_last_chunk(&self) -> bool {
        self.handle.is_last_chunk()
    }

    /// Reset the timer explicitly. Calling this method means that you have to responsible for the timer mesurements,
    /// since api::Decoder assumes that the timer never got reset.
    pub fn reset_timer(&mut self) {
        self.timer = SegmentedExecutionTimes::new();
    }

    /// Returns the execution times of the method, which is measured by the previous operation.
    pub fn segmented_execution_times(&self) -> SegmentedExecutionTimes {
        self.timer
    }

    pub fn header(&self) -> &NativeFileHeader {
        self.handle.header()
    }

    pub fn footer(&self) -> &NativeFileFooter {
        self.handle.footer()
    }

    pub fn chunk_footer(&self) -> &NativeChunkFooter {
        self.handle.chunk_footer()
    }

    pub fn number_of_records(&self) -> u64 {
        self.handle.number_of_records()
    }

    pub fn chunk_index(&self) -> usize {
        self.handle.chunk_index
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
        self.reader
            .materialize(output, bitmask, logical_offset, &mut self.timer)?;

        Ok(())
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
        let bm = self
            .reader
            .filter(predicate, bitmask, logical_offset, &mut self.timer)?;

        Ok(bm)
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
        self.reader.filter_materialize(
            output,
            predicate,
            bitmask,
            logical_offset,
            &mut self.timer,
        )?;

        Ok(())
    }

    /// Calculate the sum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    pub fn sum(&mut self, bitmask: Option<&RoaringBitmap>) -> Result<f64> {
        let logical_offset = self.handle.chunk_footer().logical_offset() as usize;
        let v = self.reader.sum(bitmask, logical_offset, &mut self.timer)?;

        Ok(v)
    }

    /// Calculate the minimum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    pub fn min(&mut self, bitmask: Option<&RoaringBitmap>) -> Result<f64> {
        let logical_offset = self.handle.chunk_footer().logical_offset() as usize;
        let v = self.reader.min(bitmask, logical_offset, &mut self.timer)?;

        Ok(v)
    }

    /// Calculate the maximum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    pub fn max(&mut self, bitmask: Option<&RoaringBitmap>) -> Result<f64> {
        let logical_offset = self.handle.chunk_footer().logical_offset() as usize;
        let v = self.reader.max(bitmask, logical_offset, &mut self.timer)?;

        Ok(v)
    }

    /// Calculates the distance^2 between the data at the specified offset in the chunk and the target slice.
    ///
    /// Let `chunk` is the logical chunk of f64 array.
    /// The distance will be calculated with the vector of length: `min(chunk[offset_in_chunk..].len(), target.len())`.
    ///
    /// # Parameters
    ///
    /// - `offset_in_chunk`: The offset within the chunk where the data is located.
    /// - `target`: A slice of `f64` values representing the target data to compare against.
    /// - `timer`: A mutable reference to `SegmentedExecutionTimes` for recording execution times.
    ///
    /// # Returns
    ///
    /// A `Result` containing the calculated distance^2 as an `f64` value, or an error if the calculation fails.
    ///
    pub fn distance_squared(
        &mut self,
        offset_in_chunk: usize,
        target: &[f64],
    ) -> decoder::Result<f64> {
        self.reader
            .distance_squared(offset_in_chunk, target, &mut self.timer)
    }

    /// Calculates the dot product between the data at the specified offset in the chunk and the target slice.
    ///
    /// Let `chunk` is the logical chunk of f64 array.
    /// The dot product will be calculated with the vector of length: `min(chunk[offset_in_chunk..].len(), target.len())`.
    ///
    /// # Parameters
    ///
    /// - `offset_in_chunk`: The offset within the chunk where the data is located.
    /// - `target`: A slice of `f64` values representing the target data to compare against.
    /// - `timer`: A mutable reference to `SegmentedExecutionTimes` for recording execution times.
    ///
    /// # Returns
    ///
    /// A `Result` containing the calculated dot product as an `f64` value, or an error if the calculation fails.
    pub fn dot_product(&mut self, offset_in_chunk: usize, target: &[f64]) -> decoder::Result<f64> {
        self.reader
            .dot_product(offset_in_chunk, target, &mut self.timer)
    }

    #[cfg(feature = "cuda")]
    pub fn gemm(
        &mut self,
        ctx: &mut cuda_binding::Context,
        a_range: std::ops::Range<usize>,
        b: &[f64],
        c: &mut [f64],
        swap_a_b: bool,
        cfg: cuda_binding::GemmConfig<f64>,
    ) -> Result<()> {
        let decompressed = self.reader.decompress(&mut self.timer)?;
        let a = &decompressed[a_range];
        if swap_a_b {
            ctx.gemm(b, a, c, cfg)?;
        } else {
            ctx.gemm(a, b, c, cfg)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeneralChunkReaderInner<R: Read> {
    Uncompressed(uncompressed::UncompressedReader),
    RLE(run_length::RunLengthReader),
    Gorilla(gorilla::GorillaReader),
    BUFF(buff::BUFFReader),
    Chimp(chimp::ChimpReader),
    Chimp128(chimp_n::Chimp128Reader),
    ElfOnChimp(elf::on_chimp::ElfReader),
    Elf(elf::ElfReader),
    DeltaSprintz(sprintz::DeltaSprintzReader),
    Zstd(zstd::ZstdReader<R>),
    Gzip(gzip::GzipReader<R>),
    Snappy(snappy::SnappyReader<R>),
    #[cfg(feature = "ffi_alp")]
    FFIAlp(ffi_alp::FFIAlpReader<R>),
}

impl<R: Read> GeneralChunkReaderInner<R> {
    pub fn new<T: FileMetadataLike>(
        handle: &GeneralChunkHandle<T>,
        reader: R,
        compression_scheme: CompressionScheme,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<Self> {
        Ok(match compression_scheme {
            CompressionScheme::Uncompressed => GeneralChunkReaderInner::Uncompressed(
                uncompressed::UncompressedReader::new(handle, reader, timer)?,
            ),
            CompressionScheme::RLE => GeneralChunkReaderInner::RLE(
                run_length::RunLengthReader::new(handle, reader, timer)?,
            ),
            CompressionScheme::Gorilla => GeneralChunkReaderInner::Gorilla(
                gorilla::GorillaReader::new(handle, reader, timer)?,
            ),
            CompressionScheme::BUFF => {
                GeneralChunkReaderInner::BUFF(buff::BUFFReader::new(handle, reader, timer))
            }
            CompressionScheme::Chimp => {
                GeneralChunkReaderInner::Chimp(chimp::ChimpReader::new(handle, reader, timer)?)
            }
            CompressionScheme::Chimp128 => GeneralChunkReaderInner::Chimp128(
                chimp_n::Chimp128Reader::new(handle, reader, timer),
            ),
            CompressionScheme::ElfOnChimp => GeneralChunkReaderInner::ElfOnChimp(
                elf::on_chimp::ElfReader::new(handle, reader, timer)?,
            ),
            CompressionScheme::Elf => {
                GeneralChunkReaderInner::Elf(elf::ElfReader::new(handle, reader, timer)?)
            }
            CompressionScheme::DeltaSprintz => GeneralChunkReaderInner::DeltaSprintz(
                sprintz::DeltaSprintzReader::new(handle, reader, timer)?,
            ),
            CompressionScheme::Zstd => {
                GeneralChunkReaderInner::Zstd(zstd::ZstdReader::new(handle, reader))
            }
            CompressionScheme::Gzip => {
                GeneralChunkReaderInner::Gzip(gzip::GzipReader::new(handle, reader))
            }
            CompressionScheme::Snappy => {
                GeneralChunkReaderInner::Snappy(snappy::SnappyReader::new(handle, reader))
            }
            #[cfg(feature = "ffi_alp")]
            CompressionScheme::FFIAlp => {
                GeneralChunkReaderInner::FFIAlp(ffi_alp::FFIAlpReader::new(handle, reader))
            }
        })
    }
}

pub fn default_decompress(r: &mut impl Reader) -> decoder::Result<&[f64]> {
    if r.decompress_result().is_some() {
        return Ok(r.decompress_result().unwrap());
    }

    let data = r
        .decompress_iter()?
        .collect::<decoder::Result<Vec<f64>>>()?;
    let result = r.set_decompress_result(data);

    Ok(result)
}

pub trait Reader {
    type NativeHeader;
    type DecompressIterator<'a>: Iterator<Item = decoder::Result<f64>>
    where
        Self: 'a;

    /// Returns `impl Iterator<Item = io::Result<f64>>`, which decompresses the chunk iteratively.
    fn decompress_iter(&mut self) -> decoder::Result<Self::DecompressIterator<'_>>;

    /// Decompress the whole chunk and return the slice of the decompressed values.
    fn decompress(&mut self, timer: &mut SegmentedExecutionTimes) -> decoder::Result<&[f64]>;

    /// Decompress the whole chunk and return the slice of the decompressed values with the specified precision.
    /// If the Reader does not support controlled precision, it will simply fall back to [`Reader::decompress`].
    /// Otherwise, if the Reader supports controlled precision, it will decompress the chunk with the specified precision.
    /// For example, the original data is `[11.23456789, 20.3456789, 35.456789]` and the precision is `2`.
    /// The decompressed data will be like `[11.225, 20.348, 35.462]`.
    /// The original data and the decompressed data must be the same after rounding at the precision.
    fn decompress_with_precision(
        &mut self,
        _precision: u32,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<&[f64]> {
        self.decompress(timer)
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64];

    fn decompress_result(&mut self) -> Option<&[f64]>;

    /// Returns the number of bytes of the method specific header.
    fn header_size(&self) -> usize;

    /// Read header from the internal buffer and return the reference to the header.
    /// Potentially, the header is cached internally.
    fn read_header(&mut self) -> &Self::NativeHeader;
}

#[derive(QuickImpl, Iterator)]
pub enum GeneralDecompressIterator<'a> {
    #[quick_impl(impl From)]
    BUFF(buff::BUFFIterator<'a>),
    #[quick_impl(impl From)]
    Gorilla(gorilla::GorillaDecompressIterator<'a>),
    #[quick_impl(impl From)]
    RLE(run_length::RunLengthIterator<'a>),
    #[quick_impl(impl From)]
    Uncompressed(uncompressed::UncompressedIterator<'a>),
    #[quick_impl(impl From)]
    Chimp(chimp::ChimpDecompressIterator<'a>),
    #[quick_impl(impl From)]
    Chimp128(chimp_n::Chimp128DecompressIterator<'a>),
    #[quick_impl(impl From)]
    ElfOnChimp(elf::on_chimp::ElfDecompressIterator<'a>),
    #[quick_impl(impl From)]
    Elf(elf::ElfDecompressIterator<'a>),
    #[quick_impl(impl From)]
    DeltaSprintz(sprintz::DeltaSprintzDecompressIterator<'a>),
}

macro_rules! impl_generic_reader {
    ($enum_name:ident, $($variant:ident $(#[$meta:meta])?),*) => {
        impl<R: Read> $enum_name<R> {
            /// Decompress the whole chunk and return the slice of the decompressed values.
            pub fn decompress(&mut self, timer: &mut SegmentedExecutionTimes) -> decoder::Result<&[f64]> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.decompress(timer),
                    )*
                }
            }

            /// Returns `impl Iterator<Item = io::Result<f64>>`, which decompresses the chunk iteratively.
            pub fn decompress_iter(&mut self) -> decoder::Result<GeneralDecompressIterator<'_>> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.decompress_iter().map(|c| c.into()),
                    )*
                }
            }

            pub fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.set_decompress_result(data),
                    )*
                }
            }

            pub fn decompress_result(&mut self) -> Option<&[f64]> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.decompress_result(),
                    )*
                }
            }

            pub fn header_size(&self) -> usize {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.header_size(),
                    )*
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
                timer: &mut SegmentedExecutionTimes,
            ) -> Result<()> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.materialize(output, bitmask, logical_offset, timer),
                    )*
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
                timer: &mut SegmentedExecutionTimes,
            ) -> Result<RoaringBitmap> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.filter(predicate, bitmask, logical_offset, timer),
                    )*
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
                timer: &mut SegmentedExecutionTimes,
            ) -> Result<()> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.filter_materialize(output, predicate, bitmask, logical_offset, timer),
                    )*
                }
            }

            /// Calculate the sum of the values filtered by the bitmask.
            /// `bitmask` is optional. If it is None, all values are written.
            ///
            /// `bitmask`'s index is global to the whole chunks.
            /// That is why `logical_offset` is necessary to access bitmask.
            pub fn sum(
                &mut self,
                bitmask: Option<&RoaringBitmap>,
                logical_offset: usize,
                timer: &mut SegmentedExecutionTimes,
            ) -> Result<f64> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.sum(bitmask, logical_offset, timer),
                    )*
                }
            }

            /// Calculate the minimum of the values filtered by the bitmask.
            /// `bitmask` is optional. If it is None, all values are written.
            ///
            /// `bitmask`'s index is global to the whole chunks.
            /// That is why `logical_offset` is necessary to access bitmask.
            pub fn min(
                &mut self,
                bitmask: Option<&RoaringBitmap>,
                logical_offset: usize,
                timer: &mut SegmentedExecutionTimes,
            ) -> Result<f64> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.min(bitmask, logical_offset, timer),
                    )*
                }
            }

            /// Calculate the maximum of the values filtered by the bitmask.
            /// `bitmask` is optional. If it is None, all values are written.
            ///
            /// `bitmask`'s index is global to the whole chunks.
            /// That is why `logical_offset` is necessary to access bitmask.
            pub fn max(
                &mut self,
                bitmask: Option<&RoaringBitmap>,
                logical_offset: usize,
                timer: &mut SegmentedExecutionTimes,
            ) -> Result<f64> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.max(bitmask, logical_offset, timer),
                    )*
                }
            }

            /// Calculates the distance^2 between the data at the specified offset in the chunk and the target slice.
            ///
            /// Let `chunk` is the logical chunk of f64 array.
            /// The distance will be calculated with the vector of length: `min(chunk[offset_in_chunk..].len(), target.len())`.
            ///
            /// # Parameters
            ///
            /// - `offset_in_chunk`: The offset within the chunk where the data is located.
            /// - `target`: A slice of `f64` values representing the target data to compare against.
            /// - `timer`: A mutable reference to `SegmentedExecutionTimes` for recording execution times.
            ///
            /// # Returns
            ///
            /// A `Result` containing the calculated distance^2 as an `f64` value, or an error if the calculation fails.
            ///
            pub fn distance_squared(
                &mut self,
                offset_in_chunk: usize,
                target: &[f64],
                timer: &mut SegmentedExecutionTimes,
            ) -> decoder::Result<f64> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.distance_squared(offset_in_chunk, target, timer),
                    )*
                }
            }


            /// Calculates the dot product between the data at the specified offset in the chunk and the target slice.
            ///
            /// Let `chunk` is the logical chunk of f64 array.
            /// The dot product will be calculated with the vector of length: `min(chunk[offset_in_chunk..].len(), target.len())`.
            ///
            /// # Parameters
            ///
            /// - `offset_in_chunk`: The offset within the chunk where the data is located.
            /// - `target`: A slice of `f64` values representing the target data to compare against.
            /// - `timer`: A mutable reference to `SegmentedExecutionTimes` for recording execution times.
            ///
            /// # Returns
            ///
            /// A `Result` containing the calculated dot product as an `f64` value, or an error if the calculation fails.
            pub fn dot_product(
                &mut self,
                offset_in_chunk: usize,
                target: &[f64],
                timer: &mut SegmentedExecutionTimes,
            ) -> decoder::Result<f64> {
                match self {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(c) => c.dot_product(offset_in_chunk, target, timer),
                    )*
                }
            }
        }

        impl<R: Read> From<&$enum_name<R>> for CompressionScheme {
            fn from(value: &$enum_name<R>) -> Self {
                match value {
                    $(
                        $(#[$meta])?
                        $enum_name::$variant(_) => CompressionScheme::$variant,
                    )*
                }
            }
        }
    };
}

impl_generic_reader!(
    GeneralChunkReaderInner,
    Uncompressed,
    RLE,
    Gorilla,
    BUFF,
    Chimp,
    Chimp128,
    ElfOnChimp,
    Elf,
    DeltaSprintz,
    Zstd,
    Gzip,
    Snappy,
    FFIAlp
    #[cfg(feature = "ffi_alp")]
);

#[cfg(test)]
mod tests {
    use rand::Rng;
    use std::io::{self, Seek};

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
                random_values.push(rng.gen_range(0.0..10000.0));
            }
        }

        random_values
    }

    fn generate_and_write_random_f64_with_precision(n: usize, scale: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut random_values: Vec<f64> = Vec::with_capacity(n);

        fn round_by_scale(value: f64, scale: usize) -> f64 {
            let scale = scale as f64;
            (value * scale).round() / scale
        }

        for i in 0..n {
            if rng.gen_bool(0.5) && random_values.last().is_some() {
                random_values.push(random_values[i - 1]);
            } else {
                random_values.push(round_by_scale(rng.gen_range(0.0..10000.0), scale));
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

            let chunk_option = ChunkOption::RecordCount(64);

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
        let values = generate_and_write_random_f64(103);
        let mut decoder = decoder(values.as_slice(), compresor_config.into());

        let mut reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        test_compressed_result(reader.inner_mut());

        test_decompress_iter(&mut decoder);
    }

    fn test_all_with_precision(compresor_config: CompressorConfig, scale: usize) {
        let values = generate_and_write_random_f64_with_precision(103, scale);
        let mut decoder = decoder(values.as_slice(), compresor_config);

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
            .unwrap()
            .collect::<decoder::Result<Vec<f64>>>()
            .unwrap();

        let mut reader = decoder.chunk_reader(ChunkId::new(0)).unwrap();
        let decompress_result = reader
            .inner_mut()
            .decompress(&mut SegmentedExecutionTimes::new())
            .unwrap();

        assert_eq!(
            iter_result.len(),
            decompress_result.len(),
            "decompress_iter should return the same length of result as decompress"
        );

        for (i, (a, b)) in iter_result.iter().zip(decompress_result.iter()).enumerate() {
            assert_eq!(
                a, b,
                "[{i} th]decompress_iter should return the same result as decompress"
            );
        }
        assert_eq!(
            iter_result, decompress_result,
            "decompress_iter should return the same result as decompress"
        );
    }

    macro_rules! declare_test_reader {
        ($method:ident) => {
            #[cfg(test)]
            mod $method {
                #[test]
                fn test_reader_consistency() {
                    let config = super::CompressorConfig::$method().build();
                    super::test_all(config);
                }
            }
        };
    }

    declare_test_reader!(uncompressed);
    declare_test_reader!(rle);
    declare_test_reader!(gorilla);
    declare_test_reader!(chimp);
    declare_test_reader!(chimp128);
    declare_test_reader!(elf_on_chimp);
    declare_test_reader!(elf);
    declare_test_reader!(delta_sprintz);
    #[cfg(not(miri))]
    declare_test_reader!(zstd);
    declare_test_reader!(gzip);
    declare_test_reader!(snappy);
    #[cfg(all(feature = "ffi_alp", not(miri)))]
    declare_test_reader!(ffi_alp);

    #[test]
    fn test_buff() {
        let scale = 100;
        test_all_with_precision(
            CompressorConfig::buff().scale(scale).build().into(),
            scale as usize,
        );
    }
}
