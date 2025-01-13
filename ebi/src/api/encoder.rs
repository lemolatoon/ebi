use std::{
    fs::File,
    io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    mem::size_of_val,
    path::Path,
    slice,
};

use crate::{
    compressor::CompressorConfig,
    encoder::{self, ChunkOption, FileWriter},
    time::SegmentedExecutionTimes,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EncoderInput<R: Read> {
    inner: R,
}

impl<R: Read> EncoderInput<R> {
    pub fn from_reader(reader: R) -> Self {
        Self { inner: reader }
    }
}

impl<'a> EncoderInput<&'a [u8]> {
    pub fn from_slice(slice: &'a [u8]) -> Self {
        Self::from_reader(slice)
    }
}

impl<'a> EncoderInput<&'a [u8]> {
    pub fn empty() -> Self {
        Self::from_slice(&[])
    }

    pub fn from_f64_slice(slice: &'a [f64]) -> Self {
        let u8_slice: &'a [u8] =
            unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) };
        Self::from_slice(u8_slice)
    }
}

impl EncoderInput<File> {
    pub fn from_file(file_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(file_path)?;
        Ok(Self::from_reader(file))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

impl<R: Read> Read for EncoderInput<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Read::read(&mut self.inner, buf)
    }

    fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        Read::read_vectored(&mut self.inner, bufs)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> std::io::Result<usize> {
        Read::read_to_end(&mut self.inner, buf)
    }

    fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize> {
        Read::read_to_string(&mut self.inner, buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        Read::read_exact(&mut self.inner, buf)
    }
}

pub struct Encoder<R: Read, W: Write + Seek> {
    file_writer: FileWriter<EncoderInput<R>>,
    output: EncoderOutput<W>,
    timer: SegmentedExecutionTimes,
}

impl<R: Read, W: Write + Seek> Encoder<R, W> {
    pub fn new(
        input: EncoderInput<R>,
        output: EncoderOutput<W>,
        chunk_option: ChunkOption,
        compressor_config: impl Into<CompressorConfig>,
    ) -> Self {
        Self {
            file_writer: FileWriter::new(input, compressor_config.into(), chunk_option),
            output,
            timer: SegmentedExecutionTimes::new(),
        }
    }

    /// Encode the data.
    /// This will write the data to the Output.
    /// # Errors
    /// This function will return an error if I/O Error happends.
    /// The output will be in an invalid state if an error occurs.
    pub fn encode(&mut self) -> encoder::Result<()> {
        let Self {
            file_writer,
            output,
            ..
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

            self.timer += *chunk_writer.timer();

            compressor = Some(chunk_writer.into_compressor());
        }

        file_writer.write_footer(output.writer_mut(), self.timer)?;
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
        let compressor_config = CompressorConfig::uncompressed().capacity(8000).build();
        for n in [1003, 10003, 100004, 100005] {
            #[cfg(miri)] // miri is too slow
            if n > 1003 {
                continue;
            }

            let random_values = generate_and_write_random_f64(n);

            let input = EncoderInput::from_f64_slice(&random_values);
            let output = EncoderOutput::from_vec(Vec::with_capacity(n));
            let chunk_option = ChunkOption::RecordCount(1024 * 3);
            let mut encoder = Encoder::new(input, output, chunk_option, compressor_config);

            encoder.encode().unwrap();

            let output = encoder.into_output().into_vec();

            assert!(
                random_values.len() * size_of::<f64>() < output.len() * size_of::<u8>() ,
                "output must be bigger than input because this is uncompressed: input({}), output({})", (random_values.len() + 1000) * size_of::<f64>(), output.len() * size_of::<u8>()
            );
        }
    }
}
