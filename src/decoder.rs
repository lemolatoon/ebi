pub mod error;

use error::DecoderError;
pub use error::Result;
use std::{
    io,
    io::{Read, Seek},
    mem::size_of,
};

use crate::format::{
    deserialize::{FromLeBytes, TryFromLeBytes},
    native::{NativeFileFooter, NativeFileHeader},
    ChunkFooter, FileFooter0, FileFooter2, FileHeader,
};

/// A reader for the file format.
/// This struct reads the file format from the input stream.
///
/// Seek State Rule:
/// ```text
/// match (header, footer) {
///   (None, None) => input stream's position is at the beginning of the file,
///   (Some(_), None) => input stream's position is at the end of the header,
///   (Some(_), Some(_)) => input stream's position is arbitrary,
///   _ => unreachable!(),
/// }
/// ```
pub struct FileReader<R: Read + Seek> {
    input: R,
    header: Option<NativeFileHeader>,
    footer: Option<NativeFileFooter>,
}

impl<R: Read + Seek> FileReader<R> {
    pub fn new(input: R) -> Self {
        Self {
            input,
            header: None,
            footer: None,
        }
    }

    /// Fetches the header of the file.
    /// If the header is already read, returns the reference to the header.
    /// Otherwise, reads the header from the input stream and returns the reference to the header.
    pub fn fetch_header(&mut self) -> Result<&NativeFileHeader> {
        if let Some(header) = &self.header {
            return Ok(header);
        }
        // If header is not read yet, input stream is still at the beginning.

        let mut buf = [0u8; size_of::<FileHeader>()];
        self.input.read_exact(&mut buf)?;
        FileHeader::try_from_le_bytes(&buf)?;

        unimplemented!();
    }

    /// Returns the header of the file if it is already read.
    pub fn header(&self) -> Option<&NativeFileHeader> {
        self.header.as_ref()
    }

    /// Fetches the footer of the file.
    /// If the footer is already read, returns the reference to the footer.
    /// Otherwise, reads the footer from the input stream and returns the reference to the footer.
    pub fn fetch_footer(&mut self) -> Result<&NativeFileFooter> {
        if self.footer.is_some() {
            return Ok(self.footer.as_ref().unwrap());
        }

        let header = self.fetch_header()?;
        let footer_offset = *header.footer_offset();

        self.input.seek(io::SeekFrom::Start(footer_offset))?;
        let mut buf = vec![0u8; size_of::<FileFooter0>()];
        self.input.read_exact(&mut buf)?;
        let footer0 = FileFooter0::from_le_bytes(&buf);
        let n_chunks = footer0.number_of_chunks as usize;

        let footer_left_size = size_of::<ChunkFooter>() * n_chunks + size_of::<FileFooter2>();

        buf.resize(footer_left_size, 0);
        self.input.read_exact(&mut buf[..])?;

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
    pub fn chunks_iter<'reader>(
        &'reader mut self,
    ) -> Option<impl Iterator<Item = GeneralChunkHandle<'reader, R>>> {
        if self.header.is_none() || self.footer.is_none() {
            return None;
        }

        Some(GeneralChunkHandleIter::new(self))
    }

    fn input_mut(&mut self) -> &mut R {
        &mut self.input
    }
}

pub struct GeneralChunkHandleIter<'reader, R: 'reader + Read + Seek> {
    file_reader: &'reader mut FileReader<R>,
    chunk_index: usize,
}

impl<'reader, R: 'reader + Read + Seek> GeneralChunkHandleIter<'reader, R> {
    /// Creates a new iterator over the chunks in the file.
    /// Caller must ensure that the header and the footer are already read.
    pub fn new(file_reader: &'reader mut FileReader<R>) -> Self {
        debug_assert!(file_reader.header().is_some());
        debug_assert!(file_reader.footer().is_some());

        Self {
            file_reader,
            chunk_index: 0,
        }
    }

    pub fn footer(&self) -> &NativeFileFooter {
        // Safety:
        // `footer` is always read before `GenericChunkHandlerIter` is created.
        unsafe { self.file_reader.footer().unwrap_unchecked() }
    }

    pub fn header(&self) -> &NativeFileHeader {
        // Safety:
        // `header` is always read before `GenericChunkHandlerIter` is created.
        unsafe { self.file_reader.header().unwrap_unchecked() }
    }
}

impl<'reader, R: 'reader + Read + Seek> Iterator for GeneralChunkHandleIter<'reader, R> {
    type Item = GeneralChunkHandle<'reader, R>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_chunks = *self.footer().number_of_chunks() as usize;
        if n_chunks <= self.chunk_index {
            return None;
        }

        let handle = GeneralChunkHandle::new(self.file_reader, self.chunk_index);
        self.chunk_index += 1;

        // TODO: check if this is safe
        let handle: GeneralChunkHandle<'reader, R> = unsafe {
            // Safety:
            // `GenericChunkHandleIter` is bounded by the lifetime of `'reader`.
            // `handle` is created from `self` and is returned.
            // `self` never outlives `'reader`.
            std::mem::transmute(handle)
        };

        Some(handle)
    }
}

pub struct GeneralChunkHandle<'reader, R: 'reader + Read + Seek> {
    file_reader: &'reader mut FileReader<R>,
    chunk_index: usize,
}

impl<'reader, R: 'reader + Read + Seek> GeneralChunkHandle<'reader, R> {
    pub fn new(file_reader: &'reader mut FileReader<R>, chunk_index: usize) -> Self {
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

    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    pub fn chunk_size(&self) -> u64 {
        let &physical_offset = self.footer().chunk_footers()[self.chunk_index].physical_offset();
        let &next_physical_offset = self
            .footer()
            .chunk_footers()
            .get(self.chunk_index + 1)
            .map(|footer| footer.physical_offset())
            .unwrap_or_else(|| self.header().footer_offset());
        next_physical_offset - physical_offset
    }

    pub fn fetch(&mut self, buf: &mut [u8]) -> Result<()> {
        let &physical_offset = self.footer().chunk_footers()[self.chunk_index].physical_offset();
        let chunk_size = self.chunk_size();

        if buf.len() < chunk_size as usize {
            return Err(DecoderError::BufferTooSmall);
        }

        self.file_reader
            .input_mut()
            .seek(io::SeekFrom::Start(physical_offset))?;

        self.file_reader
            .input_mut()
            .read_exact(&mut buf[..chunk_size as usize])?;

        Ok(())
    }
}
