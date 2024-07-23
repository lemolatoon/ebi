use std::{
    fs::File,
    io::{self, Cursor, Read, Seek, Write},
    mem::align_of,
    path::Path,
};

use roaring::RoaringBitmap;

use crate::{
    decoder::{
        self, chunk_reader::GeneralChunkReader, query::Predicate, FileMetadataLike, FileReader,
        GeneralChunkHandle, Metadata,
    },
    format::native::{NativeChunkFooter, NativeFileFooter, NativeFileHeader},
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

    pub fn writer(&self) -> &W {
        &self.inner
    }

    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    pub fn into_writer(self) -> W {
        self.inner
    }
}

impl DecoderOutput<File> {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self { inner: file })
    }
}

impl DecoderOutput<Cursor<Vec<u8>>> {
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            inner: Cursor::new(vec),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkId(usize);

impl ChunkId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

pub struct Decoder<R: Read + Seek> {
    input: DecoderInput<R>,
    file_metadata_ref: &'static Metadata,
    chunk_handles: Box<[GeneralChunkHandle<&'static Metadata>]>,
    buffer: Box<[u64]>,
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

        let chunk_handles: Box<[GeneralChunkHandle<&'static Metadata>]> =
            file_metadata_ref.chunks_iter().collect();

        let max_chunk_size = chunk_handles.iter().map(|x| x.chunk_size()).max().unwrap() as usize;
        let buf_size = (max_chunk_size.next_multiple_of(align_of::<u64>()) + 1).next_power_of_two();
        let buffer: Box<[u64]> = vec![0; buf_size].into_boxed_slice();

        Ok(Self {
            input,
            file_metadata_ref,
            chunk_handles,
            buffer,
        })
    }

    pub fn header(&self) -> &NativeFileHeader {
        self.file_metadata_ref.header()
    }

    pub fn footer(&self) -> &NativeFileFooter {
        self.file_metadata_ref.footer()
    }

    pub fn chunk_footers(&self) -> &[NativeChunkFooter] {
        &self.footer().chunk_footers()[..]
    }

    /// Scan the values filtered by the bitmask and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are written.
    pub fn scan<W: Write>(
        &mut self,
        output: &mut DecoderOutput<W>,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<()> {
        let Self {
            input,
            chunk_handles,
            buffer,
            ..
        } = self;

        for chunk_handle in chunk_handles
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| chunk_id.map_or(true, |x| x.index() == *i))
            .map(|(_, x)| x)
        {
            let mut chunk_range_bitmap = RoaringBitmap::new();
            chunk_range_bitmap.insert_range(chunk_handle.logical_record_range_u32());
            if bitmask.is_some_and(|bm| (bm & chunk_range_bitmap).is_empty()) {
                continue;
            }

            let mut chunk_reader = Self::chunk_reader_from_handle(input, chunk_handle, buffer)?;

            chunk_reader.scan(output.writer_mut(), bitmask)?;
        }

        Ok(())
    }

    /// Filter the values by the predicate and return the result as a bitmask.
    ///
    /// `bitmask` is optional. If it is None, all values are evaluated by `predicate`.
    pub fn filter(
        &mut self,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<RoaringBitmap> {
        let Self {
            input,
            chunk_handles,
            buffer,
            ..
        } = self;

        let mut result = RoaringBitmap::new();

        for chunk_handle in chunk_handles
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| chunk_id.map_or(true, |x| x.index() == *i))
            .map(|(_, x)| x)
        {
            let mut chunk_range_bitmap = RoaringBitmap::new();
            chunk_range_bitmap.insert_range(chunk_handle.logical_record_range_u32());
            if bitmask.is_some_and(|bm| (bm & chunk_range_bitmap).is_empty()) {
                continue;
            }

            let mut chunk_reader = Self::chunk_reader_from_handle(input, chunk_handle, buffer)?;

            let filtered = chunk_reader.filter(predicate, bitmask);

            result |= filtered;
        }

        Ok(result)
    }

    pub fn filter_scan(
        &mut self,
        output: &mut DecoderOutput<File>,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<()> {
        let Self {
            input,
            chunk_handles,
            buffer,
            ..
        } = self;

        for chunk_handle in chunk_handles
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| chunk_id.map_or(true, |x| x.index() == *i))
            .map(|(_, x)| x)
        {
            let mut chunk_range_bitmap = RoaringBitmap::new();
            chunk_range_bitmap.insert_range(chunk_handle.logical_record_range_u32());
            if !bitmask.is_some_and(|bm| (bm & chunk_range_bitmap).is_empty()) {
                continue;
            }

            let mut chunk_reader = Self::chunk_reader_from_handle(input, chunk_handle, buffer)?;

            chunk_reader.filter_scan(output.writer_mut(), predicate, bitmask)?;
        }

        Ok(())
    }

    fn chunk_reader_from_handle<'a>(
        input: &'a mut DecoderInput<R>,
        chunk_handle: &'a mut GeneralChunkHandle<&'static Metadata>,
        buffer: &'a mut [u64],
    ) -> decoder::Result<GeneralChunkReader<'a, 'a, &'static Metadata>> {
        chunk_handle.seek_to_chunk(input.reader_mut())?;
        chunk_handle.fetch(input.reader_mut(), buffer)?;
        let chunk_reader = chunk_handle.make_chunk_reader(buffer)?;

        Ok(chunk_reader)
    }
}
