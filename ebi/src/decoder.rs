pub mod chunk_reader;
pub mod error;
pub mod query;
pub mod timer;

use chunk_reader::GeneralChunkReader;
use error::DecoderError;
pub use error::Result;
use std::{
    io::{self, Read, Seek},
    mem::size_of,
    ops::Deref,
};

use crate::{
    compressor::CompressorConfig,
    format::{
        deserialize::{FromLeBytes, TryFromLeBytes},
        native::{NativeChunkFooter, NativeFileFooter, NativeFileHeader},
        ChunkFooter, FileFooter0, FileFooter3, FileHeader,
    },
};

pub trait FileMetadataLike {
    fn header(&self) -> &NativeFileHeader;
    fn footer(&self) -> &NativeFileFooter;
}

impl<T> FileMetadataLike for T
where
    T: Deref<Target = Metadata>,
{
    fn header(&self) -> &NativeFileHeader {
        self.deref().header()
    }

    fn footer(&self) -> &NativeFileFooter {
        self.deref().footer()
    }
}

#[derive(Debug, Clone)]
pub struct Metadata {
    header: NativeFileHeader,
    footer: NativeFileFooter,
}

impl FileMetadataLike for Metadata {
    fn header(&self) -> &NativeFileHeader {
        &self.header
    }

    fn footer(&self) -> &NativeFileFooter {
        &self.footer
    }
}

impl Metadata {
    /// Returns an iterator over the chunks in the file.
    /// If the header or the footer is not read yet, returns `None`.
    pub fn chunks_iter(&self) -> impl Iterator<Item = GeneralChunkHandle<&Self>> {
        let n_chunks = self.footer().number_of_chunks() as usize;

        (0..n_chunks).map(move |i| GeneralChunkHandle::new(self, i))
    }

    pub fn chunks_iter_with_mapping_metadata<T: FileMetadataLike>(
        &self,
        f: impl Fn() -> T,
    ) -> impl Iterator<Item = GeneralChunkHandle<T>> {
        let n_chunks = self.footer().number_of_chunks() as usize;

        (0..n_chunks).map(move |i| GeneralChunkHandle::new(f(), i))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetadataRef<'a, 'b> {
    header: &'a NativeFileHeader,
    footer: &'b NativeFileFooter,
}

impl MetadataRef<'_, '_> {
    /// Returns an iterator over the chunks in the file.
    /// If the header or the footer is not read yet, returns `None`.
    pub fn chunks_iter(&self) -> impl Iterator<Item = GeneralChunkHandle<Self>> {
        let n_chunks = self.footer().number_of_chunks() as usize;

        let metadata_ref = *self;
        (0..n_chunks).map(move |i| GeneralChunkHandle::new(metadata_ref, i))
    }
}

impl FileMetadataLike for MetadataRef<'_, '_> {
    fn header(&self) -> &NativeFileHeader {
        self.header
    }

    fn footer(&self) -> &NativeFileFooter {
        self.footer
    }
}

/// A reader for the file format.
/// This struct reads the file format from the input stream.
#[derive(Debug, Clone)]
pub struct FileReader {
    header: Option<NativeFileHeader>,
    footer: Option<NativeFileFooter>,
}

impl Default for FileReader {
    fn default() -> Self {
        Self::new()
    }
}

impl FileReader {
    pub fn new() -> Self {
        Self {
            header: None,
            footer: None,
        }
    }

    /// Fetches the header of the file.
    /// If the header is already read, returns the reference to the header.
    /// Otherwise, reads the header from the input stream and returns the reference to the header.
    /// Caller must guarantee that the input stream is at the beginning of the file.
    pub fn fetch_header<R: Read>(&mut self, mut input: R) -> Result<&NativeFileHeader> {
        if self.header.is_some() {
            return Ok(self.header.as_ref().unwrap());
        }
        // If header is not read yet, input stream is still at the beginning.

        let mut buf = [0u8; size_of::<FileHeader>()];
        input.read_exact(&mut buf)?;
        let header = FileHeader::try_from_le_bytes(&buf)?;

        self.header = Some(NativeFileHeader::from(&header));

        Ok(self.header.as_ref().unwrap())
    }

    /// Returns the header of the file if it is already read.
    pub fn header(&self) -> Option<&NativeFileHeader> {
        self.header.as_ref()
    }

    /// Seeks the input stream to the footer of the file.
    /// Caller must guarantee that the input stream is at the beginning of the file.
    /// After calling this method, the input stream is at the footer of the file.
    ///
    /// # Errors
    /// Returns an error if the header is not read yet.
    pub fn seek_to_footer<R: Read + Seek>(&mut self, input: &mut R) -> Result<()> {
        let footer_offset = self
            .header()
            .ok_or(DecoderError::PreconditionsNotMet)?
            .footer_offset();
        input.seek(io::SeekFrom::Start(footer_offset))?;
        Ok(())
    }

    /// Fetches the footer of the file.
    /// If the footer is already read, returns the reference to the footer.
    /// Otherwise, reads the footer from the input stream and returns the reference to the footer.
    /// Caller must guarantee that the input stream is at the footer of the file.
    ///
    /// # Errors
    /// - Returns an error if the header is not read yet.
    /// - Returns an error if I/O error occurs.
    pub fn fetch_footer<R: Read>(&mut self, mut input: R) -> Result<&NativeFileFooter> {
        if self.footer.is_some() {
            return Ok(self.footer.as_ref().unwrap());
        }

        let Some(header) = self.header() else {
            #[allow(clippy::useless_conversion)]
            return Err(DecoderError::PreconditionsNotMet.into());
        };

        let mut buf = vec![0u8; size_of::<FileFooter0>()];
        input.read_exact(&mut buf)?;
        let footer0 = FileFooter0::from_le_bytes(&buf);
        let n_chunks = footer0.number_of_chunks as usize;

        let chunk_footers_size = size_of::<ChunkFooter>() * n_chunks;

        buf.resize(chunk_footers_size, 0);
        input.read_exact(&mut buf[..])?;

        let chunk_footers = buf[..]
            .chunks_exact(size_of::<ChunkFooter>())
            .map(ChunkFooter::from_le_bytes)
            .collect::<Vec<_>>();

        input.read_exact(&mut buf[..1])?;
        let config_size = buf[0] as usize;
        buf.resize(config_size, 0);
        input.read_exact(&mut buf[..config_size])?;

        let compressor_config =
            CompressorConfig::from_le_bytes(header.config().compression_scheme(), &buf[..]);

        let footer3_size = size_of::<FileFooter3>();

        buf.resize(footer3_size, 0);
        input.read_exact(&mut buf[..])?;

        let footer2 = FileFooter3::from_le_bytes(&buf);

        let footer =
            NativeFileFooter::new(&footer0, &chunk_footers[..], compressor_config, &footer2);

        self.footer = Some(footer);

        Ok(self.footer.as_ref().unwrap())
    }

    /// Returns the footer of the file if it is already read.
    pub fn footer(&self) -> Option<&NativeFileFooter> {
        self.footer.as_ref()
    }

    pub fn metadata(&self) -> Option<MetadataRef<'_, '_>> {
        Some(MetadataRef {
            header: self.header.as_ref()?,
            footer: self.footer.as_ref()?,
        })
    }

    /// Converts the `FileReader` into `Metadata`.
    /// If the header or the footer is not read yet, returns `None`.
    pub fn into_metadata(self) -> Option<Metadata> {
        Some(Metadata {
            header: self.header?,
            footer: self.footer?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GeneralChunkHandle<T: FileMetadataLike> {
    file_metadata: T,
    chunk_index: usize,
}

impl<T: FileMetadataLike> GeneralChunkHandle<T> {
    /// Creates a new `GeneralChunkHandle`.
    /// Caller must guarantee that the `chunk_index` is valid.
    /// `chunk_index` is valid if `0 <= chunk_index < number_of_chunks`.
    ///     
    /// Caller must fetch the header and the footer before creating `GeneralChunkHandle`.
    fn new(file_metadata: T, chunk_index: usize) -> Self {
        Self {
            file_metadata,
            chunk_index,
        }
    }

    pub fn is_last_chunk(&self) -> bool {
        self.chunk_index + 1 == self.file_metadata.footer().number_of_chunks() as usize
    }

    pub fn header(&self) -> &NativeFileHeader {
        self.file_metadata.header()
    }

    pub fn footer(&self) -> &NativeFileFooter {
        self.file_metadata.footer()
    }

    pub fn chunk_footer(&self) -> &NativeChunkFooter {
        &self.footer().chunk_footers()[self.chunk_index]
    }

    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    pub fn chunk_size(&self) -> u64 {
        let physical_offset = self.footer().chunk_footers()[self.chunk_index].physical_offset();
        let next_physical_offset = self
            .footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.physical_offset())
            .unwrap_or_else(|| self.header().footer_offset());
        next_physical_offset - physical_offset
    }

    pub fn physical_offset(&self) -> u64 {
        self.chunk_footer().physical_offset()
    }

    pub fn logical_offset(&self) -> u64 {
        self.chunk_footer().logical_offset()
    }

    pub fn next_physical_offset(&self) -> u64 {
        self.footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.physical_offset())
            .unwrap_or_else(|| self.header().footer_offset())
    }

    pub fn next_logical_offset(&self) -> u64 {
        self.footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.logical_offset())
            .unwrap_or_else(|| self.footer().number_of_records())
    }

    pub fn number_of_records(&self) -> u64 {
        let logical_offset = self.chunk_footer().logical_offset();
        let next_physical_offset = self
            .footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.logical_offset())
            .unwrap_or_else(|| self.footer().number_of_records());

        next_physical_offset - logical_offset
    }

    pub fn logical_record_range(&self) -> std::ops::Range<u64> {
        let logical_offset = self.chunk_footer().logical_offset();
        let next_logical_offset = self
            .footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.logical_offset())
            .unwrap_or_else(|| self.footer().number_of_records());

        logical_offset..next_logical_offset
    }

    pub fn logical_record_range_u32(&self) -> std::ops::Range<u32> {
        let logical_offset = self.chunk_footer().logical_offset() as u32;
        let next_logical_offset =
            self.footer()
                .chunk_footers()
                .get(self.chunk_index + 1)
                .map(|footer| footer.logical_offset())
                .unwrap_or_else(|| self.footer().number_of_records()) as u32;

        logical_offset..next_logical_offset
    }

    /// Seeks the input stream to the beginning of the chunk.
    pub fn seek_to_chunk<R: Read + Seek>(&self, input: &mut R) -> io::Result<()> {
        let physical_offset = self.footer().chunk_footers()[self.chunk_index].physical_offset();
        input.seek(io::SeekFrom::Start(physical_offset))?;
        Ok(())
    }

    /// Seeks the input stream to the end of the chunk.
    pub fn seek_to_chunk_end<R: Read + Seek>(&self, input: &mut R) -> io::Result<()> {
        let next_physical_offset = self
            .footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.physical_offset())
            .unwrap_or_else(|| self.header().footer_offset());
        input.seek(io::SeekFrom::Start(next_physical_offset))?;
        Ok(())
    }

    pub fn make_chunk_reader<R: Read>(&self, reader: R) -> Result<GeneralChunkReader<'_, T, R>> {
        GeneralChunkReader::new(self, reader)
    }
}
