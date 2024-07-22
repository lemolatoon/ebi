use std::{
    fs::File,
    io::{Cursor, Read, Seek, Write},
    mem::size_of_val,
    path::Path,
};

use thiserror::Error;

use crate::{
    compressor::CompressorConfig,
    encoder::ChunkOption,
    io::aligned_buf_reader::{AlignedBufRead, AlignedBufReader, AlignedByteSliceBufReader},
};

pub struct EncoderInput<R: AlignedBufRead> {
    inner: R,
}

impl<R: Read> EncoderInput<AlignedBufReader<R>> {
    pub fn from_reader(reader: R) -> Self {
        Self {
            inner: AlignedBufReader::new(reader),
        }
    }
}

impl<'a> EncoderInput<AlignedBufReader<&'a [u8]>> {
    pub fn from_slice(slice: &'a [u8]) -> Self {
        Self::from_reader(slice)
    }
}

impl<'a> EncoderInput<AlignedByteSliceBufReader<'a>> {
    pub fn empty() -> Self {
        Self {
            inner: AlignedByteSliceBufReader::new_from_f64_slice(&[]),
        }
    }

    pub fn from_aligned_slice(slice: &'a [u8]) -> Option<Self> {
        let inner = AlignedByteSliceBufReader::new(slice)?;
        Some(Self { inner })
    }

    pub fn from_u64_slice(slice: &'a [u64]) -> Self {
        debug_assert!(
            slice.as_ptr().is_aligned(),
            "slice must be always aligned to its element's alignment(u64)"
        );
        let inner = AlignedByteSliceBufReader::new_from_u64_slice(slice);
        Self { inner }
    }

    pub fn from_f64_slice(slice: &'a [f64]) -> Self {
        debug_assert!(
            slice.as_ptr().is_aligned(),
            "slice must be always aligned to its element's alignment(f64)"
        );
        let inner = AlignedByteSliceBufReader::new_from_f64_slice(slice);
        Self { inner }
    }
}

impl EncoderInput<AlignedBufReader<File>> {
    pub fn from_file(file_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(file_path)?;
        Ok(Self::from_reader(file))
    }
}

pub struct EncoderOutput<W: Write + Seek> {
    inner: W,
}

impl<W: Write + Seek> EncoderOutput<W> {
    pub fn from_writer(writer: W) -> Self {
        Self { inner: writer }
    }
}

impl EncoderOutput<File> {
    /// Create a new `EncoderOutput` from a file path.
    /// This will create a new file at the given path.
    /// If the file already exists, it will be overwritten.
    pub fn from_file(file_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::create(file_path)?;
        Ok(Self::from_writer(file))
    }
}

impl EncoderOutput<Cursor<Vec<u8>>> {
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::from_writer(Cursor::new(v))
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.inner.into_inner()
    }
}

struct AppendableEncoderInput<R: AlignedBufRead> {
    inner: R,
    additional_data: Vec<f64>,
    additional_data_cursor: Option<usize>,
}

#[derive(Error, Debug, Clone, Copy)]
pub enum AppendError {
    #[error("Read has already started")]
    ReadStarted,
}

impl<R: AlignedBufRead> AppendableEncoderInput<R> {
    pub fn with_capacity(inner: R, capacity: usize) -> Self {
        Self {
            inner,
            additional_data: Vec::with_capacity(capacity),
            additional_data_cursor: None,
        }
    }

    pub fn bulk_append(&mut self, additional_data: &[f64]) -> Result<(), AppendError> {
        if self.additional_data_cursor.is_some() {
            return Err(AppendError::ReadStarted);
        }

        self.additional_data.extend_from_slice(additional_data);

        Ok(())
    }

    pub fn append(&mut self, additional_data: f64) -> Result<(), AppendError> {
        if self.additional_data_cursor.is_some() {
            return Err(AppendError::ReadStarted);
        }

        self.additional_data.push(additional_data);

        Ok(())
    }

    /// Returns the additional data as a slice of `u8`.
    /// This slice is guaranteed to be aligned to 8 bytes.
    fn additional_data_bytes(additional_data: &[f64]) -> &[u8] {
        let ptr = additional_data.as_ptr().cast::<u8>();
        let byte_len = size_of_val(additional_data);

        // Safety:
        // Any slice can be safely casted to a slice of u8.
        unsafe { std::slice::from_raw_parts(ptr, byte_len) }
    }
}

unsafe impl AlignedBufRead for AppendableEncoderInput<AlignedBufReader<File>> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        let Self {
            inner,
            additional_data,
            additional_data_cursor,
        } = self;
        let result = AlignedBufRead::fill_buf(inner)?;

        if !result.is_empty() {
            return Ok(result);
        }

        let cursor = *additional_data_cursor.get_or_insert(0);

        if Self::additional_data_bytes(additional_data).is_empty() {
            return Ok(&[]);
        }

        if Self::additional_data_bytes(additional_data).len() <= cursor {
            return Ok(&[]);
        }

        Ok(&Self::additional_data_bytes(additional_data)[cursor..])
    }

    fn consume(&mut self, amt: usize) {
        let Self {
            inner,
            additional_data_cursor,
            ..
        } = self;

        match additional_data_cursor {
            Some(ref mut cursor) => {
                *cursor += amt;
            }
            None => {
                AlignedBufRead::consume(inner, amt);
            }
        }
    }
}

pub struct Encoder<R: AlignedBufRead, W: Write + Seek> {
    input: AppendableEncoderInput<R>,
    output: EncoderOutput<W>,
    chunk_option: ChunkOption,
    compressor_config: CompressorConfig,
}

impl<R: AlignedBufRead, W: Write + Seek> Encoder<R, W> {
    pub fn new(
        input: EncoderInput<R>,
        output: EncoderOutput<W>,
        chunk_option: ChunkOption,
        compressor_config: impl Into<CompressorConfig>,
    ) -> Self {
        Self::with_additional_capacity(input, output, chunk_option, compressor_config, 0)
    }

    pub fn with_additional_capacity(
        input: EncoderInput<R>,
        output: EncoderOutput<W>,
        chunk_option: ChunkOption,
        compressor_config: impl Into<CompressorConfig>,
        capacity: usize,
    ) -> Self {
        let compressor_config = compressor_config.into();
        Self {
            input: AppendableEncoderInput::with_capacity(input.inner, capacity),
            output,
            chunk_option,
            compressor_config,
        }
    }

    pub fn append(&mut self, additional_data: f64) -> Result<(), AppendError> {
        self.input.append(additional_data)
    }

    pub fn bulk_append(&mut self, additional_data: &[f64]) -> Result<(), AppendError> {
        self.input.bulk_append(additional_data)
    }
}
