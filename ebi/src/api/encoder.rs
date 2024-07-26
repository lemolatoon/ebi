use std::{
    fs::File,
    io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    mem::size_of_val,
    path::Path,
};

use thiserror::Error;

use crate::{
    compressor::CompressorConfig,
    encoder::{ChunkOption, FileWriter},
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

    fn writer_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    pub fn into_inner(self) -> W {
        self.inner
    }

    pub fn into_buffered(self) -> EncoderOutput<BufWriter<W>> {
        let buf_writer = BufWriter::new(self.inner);
        EncoderOutput { inner: buf_writer }
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

unsafe impl<R: AlignedBufRead> AlignedBufRead for AppendableEncoderInput<R> {
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
    file_writer: FileWriter<AppendableEncoderInput<R>>,
    output: EncoderOutput<W>,
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
        let input = AppendableEncoderInput::with_capacity(input.inner, capacity);
        let compressor_config = compressor_config.into();
        let file_writer = FileWriter::new(input, compressor_config, chunk_option);
        Self {
            file_writer,
            output,
        }
    }

    pub fn append(&mut self, additional_data: f64) -> Result<(), AppendError> {
        self.file_writer.input_mut().append(additional_data)
    }

    pub fn bulk_append(&mut self, additional_data: &[f64]) -> Result<(), AppendError> {
        self.file_writer.input_mut().bulk_append(additional_data)
    }

    /// Encode the data.
    /// This will write the data to the Output.
    /// # Errors
    /// This function will return an error if I/O Error happends.
    /// The output will be in an invalid state if an error occurs.
    pub fn encode(&mut self) -> std::io::Result<()> {
        let Self {
            file_writer,
            output,
        } = self;
        // write header leaving footer offset blank
        file_writer.write_header(output.writer_mut())?;

        let mut compressor = None;
        // loop until there are no data left
        while let Some(mut chunk_writer) =
            file_writer.chunk_writer_with_compressor(compressor.take())
        {
            chunk_writer.write(output.writer_mut())?;
            output.writer_mut().flush()?;

            compressor = Some(chunk_writer.into_compressor());
        }

        file_writer.write_footer(output.writer_mut())?;
        let footer_offset_slot_offset = file_writer.footer_offset_slot_offset();
        output
            .writer_mut()
            .seek(SeekFrom::Start(footer_offset_slot_offset as u64))?;
        file_writer.write_footer_offset(output.writer_mut())?;

        let elapsed_time_slot_offset = file_writer.elapsed_time_slot_offset();
        output
            .writer_mut()
            .seek(SeekFrom::Start(elapsed_time_slot_offset as u64))?;
        file_writer.write_elapsed_time(output.writer_mut())?;
        // you can flush if you want
        output.writer_mut().flush()?;

        Ok(())
    }

    pub fn output(&self) -> &EncoderOutput<W> {
        &self.output
    }

    pub fn into_output(self) -> EncoderOutput<W> {
        self.output
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use rand::Rng;

    use crate::{compressor::CompressorConfig, encoder::ChunkOption};

    use super::{Encoder, EncoderInput, EncoderOutput};

    fn generate_and_write_random_f64(n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut random_values: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            if rng.gen_bool(0.5) && random_values.last().is_some() {
                random_values.push(random_values[i - 1]);
            } else {
                random_values.push(rng.gen());
            }
        }

        random_values
    }

    #[test]
    fn encode_test() {
        let header = b"my_header".to_vec().into_boxed_slice();
        let compressor_config = CompressorConfig::uncompressed()
            .capacity(8000)
            .header(header)
            .build();
        for n in [1003, 10003, 100004, 100005] {
            #[cfg(miri)] // miri is too slow
            if n > 1003 {
                continue;
            }

            let random_values = generate_and_write_random_f64(n);

            let input = EncoderInput::from_f64_slice(&random_values);
            let output = EncoderOutput::from_vec(Vec::with_capacity(n));
            let chunk_option = ChunkOption::RecordCount(1024 * 3);
            let mut encoder = Encoder::new(input, output, chunk_option, compressor_config.clone());

            for v in generate_and_write_random_f64(1000) {
                encoder.append(v).unwrap();
            }
            encoder.bulk_append(&[1.0, 2.0, 3.0]).unwrap();

            encoder.encode().unwrap();

            let output = encoder.into_output().into_vec();

            assert!(
                (random_values.len() + 1000) * size_of::<f64>() < output.len() * size_of::<u8>() ,
                "output must be bigger than input because this is uncompressed: input({}), output({})", (random_values.len() + 1000) * size_of::<f64>(), output.len() * size_of::<u8>()
            );
        }
    }
}
