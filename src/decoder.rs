pub mod chunk_reader;
pub mod error;

use chunk_reader::GeneralChunkReader;
use error::DecoderError;
pub use error::Result;
use std::{
    io::{self, Read, Seek},
    mem::{self, align_of, size_of, size_of_val},
    slice,
};

use crate::format::{
    deserialize::{FromLeBytes, TryFromLeBytes},
    native::{NativeChunkFooter, NativeFileFooter, NativeFileHeader},
    run_length::RunLengthHeader,
    uncompressed::UncompressedHeader0,
    ChunkFooter, CompressionScheme, FileFooter0, FileFooter2, FileHeader, GeneralChunkHeader,
};

/// A reader for the file format.
/// This struct reads the file format from the input stream.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

        if self.header().is_none() {
            return Err(DecoderError::PreconditionsNotMet);
        }

        let mut buf = vec![0u8; size_of::<FileFooter0>()];
        input.read_exact(&mut buf)?;
        let footer0 = FileFooter0::from_le_bytes(&buf);
        let n_chunks = footer0.number_of_chunks as usize;

        let footer_left_size = size_of::<ChunkFooter>() * n_chunks + size_of::<FileFooter2>();

        buf.resize(footer_left_size, 0);
        input.read_exact(&mut buf[..])?;

        let chunk_footers = buf[..footer_left_size - size_of::<FileFooter2>()]
            .chunks_exact(size_of::<ChunkFooter>())
            .map(ChunkFooter::from_le_bytes)
            .collect::<Vec<_>>();

        let footer2 =
            FileFooter2::from_le_bytes(&buf[footer_left_size - size_of::<FileFooter2>()..]);

        let footer = NativeFileFooter::new(&footer0, &chunk_footers[..], &footer2);

        self.footer = Some(footer);

        Ok(self.footer.as_ref().unwrap())
    }

    /// Returns the footer of the file if it is already read.
    pub fn footer(&self) -> Option<&NativeFileFooter> {
        self.footer.as_ref()
    }

    /// Returns an iterator over the chunks in the file.
    /// If the header or the footer is not read yet, returns `None`.
    pub fn chunks_iter(&self) -> Option<impl Iterator<Item = GeneralChunkHandle<'_>>> {
        if self.header.is_none() || self.footer.is_none() {
            return None;
        }

        let n_chunks = self.footer().unwrap().number_of_chunks() as usize;

        Some((0..n_chunks).map(move |i| GeneralChunkHandle::new(self, i)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GeneralChunkHandle<'reader> {
    file_reader: &'reader FileReader,
    chunk_index: usize,
}

impl<'reader> GeneralChunkHandle<'reader> {
    /// Creates a new `GeneralChunkHandle`.
    /// Caller must guarantee that the `chunk_index` is valid.
    /// `chunk_index` is valid if `0 <= chunk_index < number_of_chunks`.
    ///     
    /// Caller must fetch the header and the footer before creating `GeneralChunkHandle`.
    fn new(file_reader: &'reader FileReader, chunk_index: usize) -> Self {
        Self {
            file_reader,
            chunk_index,
        }
    }

    pub fn header(&self) -> &NativeFileHeader {
        // Safety:
        // `header` is always read before `GenericChunkHandle` is created.
        unsafe { self.file_reader.header().unwrap_unchecked() }
    }

    pub fn footer(&self) -> &NativeFileFooter {
        // Safety:
        // `footer` is always read before `GenericChunkHandle` is created.
        unsafe { self.file_reader.footer().unwrap_unchecked() }
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

    /// Fetches the chunk from the input stream.
    /// Caller must guarantee that the input stream is at the beginning of the chunk.
    ///
    /// The reason why this method requires buffer as `&mut [u64]` is that the buffer must be aligned to 8 bytes.
    ///
    /// # Errors
    /// - Returns an error if the buffer is too small.
    /// Especially, even if buffer size(bytes) are the same as chunk size, it can return an error if the buffer size is not a multiple of 8.
    /// You can resize the buffer like this:
    /// ```no_run
    /// use std::mem::{align_of, size_of};
    /// let mut buf = Vec::new();
    /// let chunk_size: usize = 1017/* = chunk_handle.chunk_size() as usize */;
    /// let expected_chunk_size = chunk_size.next_multiple_of(align_of::<u64>()) + 1;
    /// if buf.len() * size_of::<u64>() < expected_chunk_size {
    ///     buf.resize(expected_chunk_size.next_power_of_two(), 0);
    /// }
    /// ```
    /// This size of buffer is guaranteed to be enough for `fetch`.
    /// - Returns an error if I/O error occurs.
    pub fn fetch<'buf, R: Read>(&mut self, mut input: R, buf: &'buf mut [u64]) -> Result<()> {
        let chunk_size = self.chunk_size() as usize;
        // Safety:
        // - we can transmute u64 slice as u8 slice safely.
        let buf: &'buf mut [u8] = unsafe {
            slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), mem::size_of_val(buf))
        };

        if buf.len() < chunk_size.next_multiple_of(align_of::<u64>()) + 1 {
            return Err(DecoderError::BufferTooSmall);
        }

        let header_size = self.fetch_header(&mut input, buf)?;
        let data_offset = header_size.next_multiple_of(align_of::<u64>());

        input.read_exact(&mut buf[data_offset..data_offset + chunk_size - header_size])?;

        Ok(())
    }

    /// Fetches the header of the chunk including the method specific header.
    /// The header is stored in the buffer.
    ///
    /// Returns the header_size of the chunk including `GeneralChunkHeader`.
    fn fetch_header<R: Read>(&self, mut input: R, buf: &mut [u8]) -> Result<usize> {
        input.read_exact(&mut buf[..size_of::<GeneralChunkHeader>()])?;

        let compression_scheme = self.header().config().compression_scheme();

        let header_size = match *compression_scheme {
            CompressionScheme::Uncompressed => {
                input.read_exact(
                    &mut buf[size_of::<GeneralChunkHeader>()..size_of::<UncompressedHeader0>()],
                )?;
                let header_head = UncompressedHeader0::from_le_bytes(
                    &buf[size_of::<GeneralChunkHeader>()..size_of::<UncompressedHeader0>()],
                );
                let header_size = header_head.header_size as usize;
                input.read_exact(&mut buf[size_of::<UncompressedHeader0>()..header_size])?;
                header_size
            }
            CompressionScheme::RLE => {
                let header_size = size_of::<RunLengthHeader>();
                input.read_exact(&mut buf[size_of::<GeneralChunkHeader>()..header_size])?;

                header_size
            }
            c => unimplemented!("Unimplemented compression scheme: {:?}", c),
        };

        Ok(size_of::<GeneralChunkHeader>() + header_size)
    }

    pub fn make_chunk_reader<'chunk>(
        &'reader self,
        input: &'chunk [u64],
    ) -> Result<GeneralChunkReader<'reader, 'chunk>> {
        // Safety:
        // - we can transmute u64 slice as u8 slice safely.
        let input: &'chunk [u8] =
            unsafe { slice::from_raw_parts(input.as_ptr().cast::<u8>(), size_of_val(input)) };
        GeneralChunkReader::new(self, input)
    }
}
