use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Cursor, Read, Seek, Write},
    path::Path,
    sync::Arc,
};

use roaring::RoaringBitmap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{
        self, chunk_reader::GeneralChunkReader, query::Predicate, FileMetadataLike, FileReader,
        GeneralChunkHandle, Metadata,
    },
    format::native::{NativeChunkFooter, NativeFileFooter, NativeFileHeader},
    time::SegmentedExecutionTimes,
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

    pub fn into_buffered(self) -> DecoderInput<BufReader<R>> {
        let buf_reader = BufReader::new(self.inner);
        DecoderInput { inner: buf_reader }
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

    pub fn into_buffered(self) -> DecoderOutput<BufWriter<W>> {
        let buf_writer = BufWriter::new(self.inner);
        DecoderOutput { inner: buf_writer }
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    timer: SegmentedExecutionTimes,
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
            timer: SegmentedExecutionTimes::new(),
        })
    }

    /// Returns the segmented execution times of the previous operation.
    pub fn segmented_execution_times(&self) -> SegmentedExecutionTimes {
        self.timer
    }

    pub fn footer_size(&self) -> u64 {
        self.file_metadata_ref.footer().size() as u64
    }

    pub fn total_file_size(&self) -> u64 {
        self.file_metadata_ref.header().footer_offset() + self.footer_size()
    }

    pub fn total_chunk_size(&self) -> u64 {
        let chunk_head = self
            .chunk_handles
            .first()
            .map(|x| x.physical_offset())
            .unwrap_or(self.header().footer_offset());
        let chunk_tail = self.header().footer_offset();

        chunk_tail - chunk_head
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

        self.timer = SegmentedExecutionTimes::new();

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

            self.timer += chunk_reader.segmented_execution_times();
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

        self.timer = SegmentedExecutionTimes::new();

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

            self.timer += chunk_reader.segmented_execution_times();
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

        self.timer = SegmentedExecutionTimes::new();

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

            self.timer += chunk_reader.segmented_execution_times();
        }

        Ok(())
    }

    /// Calculate the sum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    pub fn sum(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<f64> {
        let Self {
            input,
            chunk_handles,
            ..
        } = self;

        self.timer = SegmentedExecutionTimes::new();

        let mut result = 0.0;

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

            result += chunk_reader.sum(bitmask)?;

            self.timer += chunk_reader.segmented_execution_times();
        }

        Ok(result)
    }

    /// Calculate the min of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    pub fn min(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<f64> {
        let Self {
            input,
            chunk_handles,
            ..
        } = self;

        self.timer = SegmentedExecutionTimes::new();

        let mut result = f64::INFINITY;

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

            result = result.min(chunk_reader.min(bitmask)?);

            self.timer += chunk_reader.segmented_execution_times();
        }

        Ok(result)
    }

    /// Calculate the max of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    pub fn max(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        chunk_id: Option<ChunkId>,
    ) -> decoder::Result<f64> {
        let Self {
            input,
            chunk_handles,
            ..
        } = self;

        self.timer = SegmentedExecutionTimes::new();

        let mut result = f64::NEG_INFINITY;

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

            result = result.max(chunk_reader.max(bitmask)?);

            self.timer += chunk_reader.segmented_execution_times();
        }

        Ok(result)
    }

    pub fn chunk_reader(
        &mut self,
        chunk_id: ChunkId,
    ) -> decoder::Result<GeneralChunkReader<'_, Arc<Metadata>, &mut R>> {
        let Self {
            input,
            chunk_handles,
            ..
        } = self;

        Self::chunk_reader_from_handle(input, &mut chunk_handles[chunk_id.index()])
    }

    fn chunk_reader_from_handle<'a>(
        input: &'a mut DecoderInput<R>,
        chunk_handle: &'a mut GeneralChunkHandle<Arc<Metadata>>,
    ) -> decoder::Result<GeneralChunkReader<'a, Arc<Metadata>, &'a mut R>> {
        chunk_handle.seek_to_chunk(input.reader_mut())?;
        let chunk_reader = chunk_handle.make_chunk_reader(input.reader_mut())?;

        Ok(chunk_reader)
    }
}
