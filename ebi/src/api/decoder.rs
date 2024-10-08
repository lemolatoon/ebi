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
        self, chunk_reader::GeneralChunkReader, error::DecoderError, query::Predicate,
        FileMetadataLike, FileReader, GeneralChunkHandle, Metadata,
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

/// Represents the result of a K-nearest neighbors search.
///
/// Each `KnnResult` contains the index and the label of this neighbor.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KnnResult {
    /// The index of this neighbor in the original data.
    pub index: usize,
    /// The distance of this neighbor.
    pub distance: f64,
}

pub struct Decoder<R: Read + Seek> {
    input: DecoderInput<R>,
    file_metadata_ref: Arc<Metadata>,
    chunk_handles: Box<[GeneralChunkHandle<Arc<Metadata>>]>,
    /// The precision used when materializing the values. If the compression method does not support controlled precision,
    /// this field will be simply ignored.
    ///
    /// If this field is `Some(n)`, the materialized values rounded by `n + 1` decimal places must be equal to the original values.
    /// For example, if the original value is `1.2345` and `n = 2`, the materialized value can be `1.228`.
    /// `round_at(1.228, 2 + 1) == 1.23 == round_at(1.2345, 2 + 1)`.
    precision: Option<u32>,
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
            precision: None,
            chunk_handles,
            timer: SegmentedExecutionTimes::new(),
        })
    }

    /// Sets the precision used when materializing the values.
    pub fn with_precision(&mut self, precision: u32) {
        self.precision = Some(precision);
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

            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

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

            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

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

            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

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

            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

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

            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

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

            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

            result = result.max(chunk_reader.max(bitmask)?);

            self.timer += chunk_reader.segmented_execution_times();
        }

        Ok(result)
    }

    /// Finds the K-nearest neighbors to the target slice within the data.
    ///
    /// The data will be interpreted as a 2D `f64` array with tags logically. Each entry in the array
    /// a vector of `f64` values. The function calculates the distances between
    /// the target slice and each vector in the data, and returns the K-nearest neighbors.
    ///
    /// # Parameters
    ///
    /// - `target`: A slice of `f64` values representing the target data to compare against.
    ///
    /// # Returns
    ///
    /// A vector of `KnnResult` representing the K-nearest neighbors. Each `KnnResult` contains the
    /// index and the distance of the neighbor.
    ///
    /// # Example
    /// ```text
    /// Data: [1.0, 1.2, 1.3, 1.4, 2.0, 1.5, 1.7, 1.8, 1.0, 1.9, 3.0, 4.0]
    /// where target.len() == 4
    /// ```
    /// will be interpreted as:
    /// ```text
    /// [
    ///     [1.0, 1.2, 1.3, 1.4],
    ///     [2.0, 1.5, 1.7, 1.8],
    ///     [1.0, 1.9, 3.0, 4.0],
    /// ]
    /// ```
    pub fn knn1(&mut self, target: &[f64]) -> decoder::Result<KnnResult> {
        let target_len = target.len();

        let Self {
            input,
            chunk_handles,
            ..
        } = self;

        self.timer = SegmentedExecutionTimes::new();

        let mut processing_vector = KnnResult {
            index: 0,
            distance: 0.0,
        };
        let mut min_result = KnnResult {
            index: usize::MAX,
            distance: f64::INFINITY,
        };
        let mut remaining_records = 0;
        for chunk_handle in chunk_handles.iter_mut() {
            let mut offset = 0;
            let chunk_number_of_records = chunk_handle.number_of_records() as usize;
            let mut chunk_reader =
                Self::chunk_reader_from_handle(input, chunk_handle, self.precision)?;

            if remaining_records != 0 {
                // If the vector is not fully processed, continue processing it with the partial distance of the previous chunk.
                let remaining_target_start = target_len - remaining_records;
                processing_vector.distance +=
                    chunk_reader.distance_squared(0, &target[remaining_target_start..])?;

                if processing_vector.distance < min_result.distance {
                    min_result = processing_vector;
                }

                remaining_records = 0;

                processing_vector.index += 1;
                offset += remaining_records;
            }

            while offset < chunk_number_of_records {
                let n_records_in_vector = target_len.min(chunk_number_of_records - offset);
                processing_vector.distance =
                    chunk_reader.distance_squared(offset, &target[..n_records_in_vector])?;

                if n_records_in_vector == target_len {
                    if processing_vector.distance <= min_result.distance {
                        min_result = processing_vector;
                    }
                    processing_vector.index += 1;
                } else {
                    // If the vector is not fully processed, save the partial distance for the next chunk.
                    remaining_records = target_len - n_records_in_vector;
                }

                offset += n_records_in_vector;
            }

            self.timer += chunk_reader.segmented_execution_times();
        }

        Ok(min_result)
    }

    /// Performs matrix multiplication between the target matrix and the data matrix within the chunk.
    ///
    /// The target matrix should be provided in a 'column first' layout, while the data matrix within the chunk
    /// will be interpreted in a 'row first' layout. The function interprets the chunk as a 3-D array, where the first
    /// dimension represents the batch size. The result of the matrix multiplication is returned in a
    /// 'row first' layout.
    ///
    /// # Parameters
    ///
    /// - `target_matrix`: A slice of `f64` values representing the target matrix in 'column first' layout.
    /// - `target_matrix_shape`: A tuple representing the shape of the target matrix (rows, columns).
    /// - `data_matrix_shape`: A tuple representing the shape of the data matrix (rows, columns).
    /// - `timer`: A mutable reference to `SegmentedExecutionTimes` for recording execution times.
    ///
    /// # Returns
    ///
    /// A `Result` containing a boxed slice of `f64` values representing the result of the matrix multiplication
    /// in 'row first' layout, or an error if the calculation fails.
    ///
    /// `result` = `data_matrix` (matmul op) `target_matrix`
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an issue accessing the data chunk or performing the calculation.
    ///
    /// # Preconditions
    ///
    /// - The number of columns in the target matrix must be equal to the number of rows in the data matrix.
    /// - The total number of elements in the data matrix must be a multiple of the product of its dimensions.
    ///
    ///
    /// # Example
    ///
    /// Given the following matrices:
    ///
    /// ```text
    /// chunk (row first layout): [a11, a12, a13, a21, a22, a23, a31, a32, a33]
    /// ```
    /// which corresponds to:
    ///
    /// ```text
    ///       | a11  a12  a13 |
    /// A =   | a21  a22  a23 |
    ///       | a31  a32  a33 |
    /// ```
    ///
    /// and
    ///
    /// ```text
    /// target_matrix (column first layout): [b11, b21, b31, b12, b22, b32, b13, b23, b33]
    /// ```
    /// which corresponds to:
    ///
    /// ```text
    ///         | b11  b12  b13 |
    ///   B =   | b21  b22  b23 |
    ///         | b31  b32  b33 |
    /// ```
    ///
    /// The result will be: C = AB
    ///
    /// ```text
    /// results (row first layout): [c11, c12, c13, c21, c22, c23, c31, c32, c33]
    /// ```
    /// which corresponds to:
    ///
    /// ```text
    ///         | c11  c12  c13 |
    ///   C =   | c21  c22  c23 |
    ///         | c31  c32  c33 |
    /// ```
    pub fn matmul(
        &mut self,
        target_matrix: &[f64],
        target_matrix_shape: (usize, usize),
        data_matrix_shape: (usize, usize),
    ) -> decoder::Result<Box<[f64]>> {
        let mut timer = SegmentedExecutionTimes::new();
        let number_of_records = self.footer().number_of_records() as usize;

        let (target_rows, target_columns) = target_matrix_shape;
        let (data_rows, data_columns) = data_matrix_shape;

        let is_matrix_shape_invalid = data_columns != target_rows;
        let is_data_matrix_shape_aligned_to_number_of_records_invalid =
            number_of_records % (data_rows * data_columns) != 0;
        if is_matrix_shape_invalid || is_data_matrix_shape_aligned_to_number_of_records_invalid {
            return Err(DecoderError::PreconditionsNotMet.into());
        }

        let batch_size = number_of_records / (data_rows * data_columns);

        let mut result_matrices = Vec::with_capacity(batch_size * data_rows * target_columns);

        let mut offset_in_chunk = 0;
        let mut chunk_index = 0;
        let mut chunk_reader = self.chunk_reader(ChunkId::new(chunk_index))?;

        for _ in 0..batch_size {
            if offset_in_chunk >= chunk_reader.number_of_records() as usize {
                offset_in_chunk = 0;
                chunk_index += 1;
                timer += chunk_reader.segmented_execution_times();
                chunk_reader = self.chunk_reader(ChunkId::new(chunk_index))?;
            }
            for row_index in 0..data_rows {
                for target_column in target_matrix.chunks_exact(target_rows) {
                    let result = chunk_reader
                        .dot_product(offset_in_chunk + row_index * data_columns, target_column)?;

                    result_matrices.push(result);
                }
            }
            offset_in_chunk += data_rows * data_columns;
        }
        debug_assert!(chunk_reader.is_last_chunk());
        timer += chunk_reader.segmented_execution_times();

        self.timer = timer;

        Ok(result_matrices.into_boxed_slice())
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

        Self::chunk_reader_from_handle(input, &mut chunk_handles[chunk_id.index()], self.precision)
    }

    fn chunk_reader_from_handle<'a>(
        input: &'a mut DecoderInput<R>,
        chunk_handle: &'a mut GeneralChunkHandle<Arc<Metadata>>,
        precision: Option<u32>,
    ) -> decoder::Result<GeneralChunkReader<'a, Arc<Metadata>, &'a mut R>> {
        chunk_handle.seek_to_chunk(input.reader_mut())?;
        let mut chunk_reader = chunk_handle.make_chunk_reader(input.reader_mut())?;

        if let Some(precision) = precision {
            chunk_reader.with_precision(precision);
        }

        Ok(chunk_reader)
    }
}
