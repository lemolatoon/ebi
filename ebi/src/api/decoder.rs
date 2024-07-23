use std::{
    fs::File,
    io::{self, Cursor, Read, Seek, Write},
    path::Path,
};

use crate::{
    decoder::{self, FileMetadataLike, FileReader, GeneralChunkHandle, Metadata},
    format::native::{NativeFileFooter, NativeFileHeader},
};

pub struct DecoderInput<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> DecoderInput<R> {
    pub fn from_reader(reader: R) -> Self {
        Self { inner: reader }
    }

    pub fn reader(&self) -> &R {
        &self.inner
    }

    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.inner
    }
}

impl DecoderInput<File> {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self { inner: file })
    }
}

pub struct DecoderOutput<W: Write> {
    inner: W,
}

impl<W: Write> DecoderOutput<W> {
    pub fn from_writer(writer: W) -> Self {
        Self { inner: writer }
    }
}

impl DecoderOutput<File> {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self { inner: file })
    }
}

impl DecoderOutput<Cursor<Vec<u8>>> {
    pub fn from_vec() -> Self {
        Self {
            inner: Cursor::new(Vec::new()),
        }
    }
}

pub struct Decoder<R: Read + Seek> {
    input: DecoderInput<R>,
    file_metadata_ref: &'static Metadata,
    chunk_handles: Box<[GeneralChunkHandle<&'static Metadata>]>,
}

impl<R: Read + Seek> Drop for Decoder<R> {
    fn drop(&mut self) {
        // Safety:
        // file_metadata_ref is created from Box::leak.
        // file_metadata_ref will never used right after this point because it will be dropped.
        drop(unsafe { Box::from_raw(self.file_metadata_ref as *const Metadata as *mut Metadata) });
    }
}

impl<R: Read + Seek> Decoder<R> {
    pub fn new(mut input: DecoderInput<R>) -> decoder::Result<Self> {
        let mut file_reader = FileReader::new();
        file_reader.fetch_header(&mut input.reader_mut())?;
        file_reader.seek_to_footer(input.reader_mut())?;
        file_reader.fetch_footer(&mut input.reader_mut())?;

        let file_metadata = file_reader.into_metadata().unwrap();
        let file_metadata = Box::new(file_metadata);
        let file_metadata_ref: &'static Metadata = Box::leak(file_metadata);

        let chunk_handles = file_metadata_ref.chunks_iter().collect();

        Ok(Self {
            input,
            file_metadata_ref,
            chunk_handles,
        })
    }

    pub fn header(&self) -> &NativeFileHeader {
        self.file_metadata_ref.header()
    }

    pub fn footer(&self) -> &NativeFileFooter {
        self.file_metadata_ref.footer()
    }
}
