use std::{
    fs::File,
    io::{self, Cursor, Read, Seek, Write},
    path::Path,
    sync::Arc,
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
    file_metadata_ref: Arc<Metadata>,
    chunk_handles: Box<[GeneralChunkHandle<Arc<Metadata>>]>,
}

impl<R: Read + Seek> Decoder<R> {
    pub fn new(mut input: DecoderInput<R>) -> decoder::Result<Self> {
        let mut file_reader = FileReader::new();
        file_reader.fetch_header(input.reader_mut())?;
        file_reader.seek_to_footer(input.reader_mut())?;
        file_reader.fetch_footer(input.reader_mut())?;

        let file_metadata = file_reader.into_metadata().unwrap();
        let file_metadata_ref = Arc::new(file_metadata);

        let chunk_handles: Box<[GeneralChunkHandle<Arc<Metadata>>]> = file_metadata_ref
            .chunks_iter_with_mapping_metadata(|| Arc::clone(&file_metadata_ref))
            .collect();

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

    pub fn chunk_footers(&self) -> &[NativeChunkFooter] {
        &self.footer().chunk_footers()[..]
    }

    /// Scan the values filtered by the bitmask and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are written.
    pub fn materialize<W: Write>(
        &mut self,
        output: &mut DecoderOutput<W>,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<()> {
        let Self {
            input,
            chunk_handles,
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

            let mut chunk_reader = Self::chunk_reader_from_handle(input, chunk_handle)?;

            chunk_reader.materialize(output.writer_mut(), bitmask)?;
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

            let mut chunk_reader = Self::chunk_reader_from_handle(input, chunk_handle)?;

            let filtered = chunk_reader.filter(predicate, bitmask)?;

            result |= filtered;
        }

        Ok(result)
    }

    pub fn filter_materialize<W: Write>(
        &mut self,
        output: &mut DecoderOutput<W>,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<()> {
        let Self {
            input,
            chunk_handles,
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

            let mut chunk_reader = Self::chunk_reader_from_handle(input, chunk_handle)?;

            chunk_reader.filter_materialize(output.writer_mut(), predicate, bitmask)?;
        }

        Ok(())
    }

    fn chunk_reader_from_handle<'a>(
        input: &'a mut DecoderInput<R>,
        chunk_handle: &'a mut GeneralChunkHandle<Arc<Metadata>>,
    ) -> decoder::Result<GeneralChunkReader<'a, &'a mut R, Arc<Metadata>>> {
        chunk_handle.seek_to_chunk(input.reader_mut())?;
        let chunk_reader = chunk_handle.make_chunk_reader(input.reader_mut())?;

        Ok(chunk_reader)
    }
}
