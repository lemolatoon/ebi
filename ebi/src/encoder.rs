pub mod error;

pub use error::Result;

use cfg_if::cfg_if;

use crate::{
    compressor::{Compressor, CompressorConfig, GenericCompressor},
    format::{
        self,
        serialize::{AsBytes, ToLe},
        ChunkFooter, FieldType, FileConfig, FileFooter0, FileFooter3, FileHeader,
        GeneralChunkHeader,
    },
};
use core::slice;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Read, Write},
    mem::{offset_of, size_of},
    time::Instant,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ChunkOption {
    Full,
    RecordCount(usize),
    ByteSizeBestEffort(usize),
}

impl From<&ChunkOption> for format::ChunkOption {
    fn from(value: &ChunkOption) -> Self {
        match value {
            ChunkOption::Full => format::ChunkOption {
                kind: format::ChunkOptionKind::Full,
                reserved: 0,
                value: 0,
            },
            ChunkOption::RecordCount(value) => format::ChunkOption {
                kind: format::ChunkOptionKind::RecordCount,
                reserved: 0,
                value: *value as u64,
            },
            ChunkOption::ByteSizeBestEffort(value) => format::ChunkOption {
                kind: format::ChunkOptionKind::ByteSize,
                reserved: 0,
                value: *value as u64,
            },
        }
    }
}

impl ChunkOption {
    pub fn reach_limit(&self, total_bytes_in: usize, total_bytes_out: usize) -> bool {
        match self {
            ChunkOption::Full => false,
            ChunkOption::RecordCount(count) => count * size_of::<f64>() <= total_bytes_in,
            ChunkOption::ByteSizeBestEffort(n_bytes) => *n_bytes <= total_bytes_out,
        }
    }
}

pub struct FileWriter<R: Read> {
    input: R,
    start_time: Instant,
    chunk_option: ChunkOption,
    compressor: CompressorConfig,
    total_bytes_in: usize,
    total_bytes_out: usize,
    /// The vector of `ChunkFooter`, including the next `ChunkFooter`
    chunk_footers: Vec<ChunkFooter>,
    reaches_eof: bool,
}

impl<R: Read> FileWriter<R> {
    const MAGIC_NUMBER: &'static [u8; 4] = b"EBI1";
    pub fn new(input: R, compressor: CompressorConfig, chunk_option: ChunkOption) -> Self {
        let start_time = Instant::now();
        let chunk_footer = ChunkFooter {
            physical_offset: size_of::<FileHeader>() as u64,
            logical_offset: 0,
        };
        Self {
            input,
            start_time,
            compressor,
            chunk_option,
            chunk_footers: vec![chunk_footer],
            total_bytes_in: 0,
            total_bytes_out: 0,
            reaches_eof: false,
        }
    }

    pub fn input(&self) -> &R {
        &self.input
    }

    pub fn input_mut(&mut self) -> &mut R {
        &mut self.input
    }

    pub fn file_header_size(&self) -> usize {
        size_of::<FileHeader>()
    }

    fn total_bytes_out(&self) -> usize {
        self.total_bytes_out
    }

    fn set_reaches_eof(&mut self) {
        self.reaches_eof = true;
    }

    fn push_chunk_footer(&mut self, chunk_footer: ChunkFooter) {
        self.chunk_footers.push(chunk_footer);
    }

    fn n_chunks(&self) -> usize {
        self.chunk_footers().len()
    }

    pub fn footer_offset_slot_offset(&self) -> usize {
        offset_of!(FileHeader, footer_offset)
    }

    fn chunk_footers_with_next_chunk_footer(&self) -> &Vec<ChunkFooter> {
        &self.chunk_footers
    }

    fn chunk_footers(&self) -> &[ChunkFooter] {
        &self.chunk_footers[..(self.chunk_footers.len() - 1)]
    }

    pub fn chunk_option(&self) -> &ChunkOption {
        &self.chunk_option
    }

    /// Returns the ChunkWriter unless input `Read` stream reaches EOF.
    /// If it is the first time of calling this method for this `FileWriter`,
    /// it is guaranteed that returned `Option` is `Some`.
    pub fn chunk_writer(&mut self) -> Option<ChunkWriter<'_, R>> {
        self.chunk_writer_with_compressor(None)
    }

    pub fn chunk_writer_with_compressor(
        &mut self,
        compressor: Option<GenericCompressor>,
    ) -> Option<ChunkWriter<'_, R>> {
        if self.reaches_eof {
            return None;
        }
        let mut compressor = compressor.unwrap_or_else(|| self.compressor.build());
        compressor.reset();
        Some(ChunkWriter::new(self, compressor))
    }

    fn increment_total_bytes_in_by(&mut self, incremented: usize) {
        self.total_bytes_in += incremented;
    }

    fn increment_total_bytes_out_by(&mut self, incremented: usize) {
        self.total_bytes_out += incremented;
    }

    pub fn write_header<W: Write>(&mut self, mut f: W) -> io::Result<()> {
        // can be static?
        let version: [u16; 3] = [
            env!("CARGO_PKG_VERSION_MAJOR"),
            env!("CARGO_PKG_VERSION_MINOR"),
            env!("CARGO_PKG_VERSION_PATCH"),
        ]
        .map(|v| v.parse().unwrap_or(u16::MAX).to_le());
        let mut header = FileHeader {
            magic_number: *Self::MAGIC_NUMBER,
            version,
            // leave footer_offset blank
            footer_offset: 0,
            config: FileConfig {
                field_type: FieldType::F64,
                chunk_option: self.chunk_option().into(),
                compression_scheme: self.compressor.compression_scheme(),
            },
        };

        f.write_all(header.to_le().as_bytes())
    }

    pub fn write_footer<W: Write + std::io::Seek>(&mut self, mut f: W) -> io::Result<()> {
        let number_of_chunks = self.n_chunks() as u64;
        let number_of_records = self
            .chunk_footers_with_next_chunk_footer()
            .last()
            .map_or(0, |f| f.logical_offset);
        let mut footer0 = FileFooter0 {
            number_of_chunks,
            number_of_records,
        };
        f.write_all(footer0.to_le().as_bytes())?;

        cfg_if! {
            if #[cfg(target_endian = "little")] {
                // We can write all chunk footers at once.
                f.write_all(self.chunk_footers().as_bytes())?
            } else {
                // We need to write chunk footers one by one.
                for chunk_footer in self.chunk_footers().iter() {
                    let mut chunk_footer = *chunk_footer;
                    f.write_all(chunk_footer.to_le().as_bytes())?;
                }
            }
        }

        self.compressor.serialize(&mut f)?;

        // TODO: calculate `crc` here.
        let crc = 0u32;

        let mut footer3 = FileFooter3 {
            compression_elapsed_time_nano_secs: 0, // must be written later
            crc,
        };

        let le_bytes = footer3.to_le().as_bytes();
        f.write_all(le_bytes)
    }

    pub fn write_footer_offset<W: Write>(&mut self, mut f: W) -> io::Result<()> {
        let footer_offset: u64 = (size_of::<FileHeader>() + self.total_bytes_out()) as u64;
        f.write_all(&footer_offset.to_le_bytes())?;

        Ok(())
    }

    pub fn elapsed_time(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    pub fn elapsed_time_slot_offset(&self) -> usize {
        let footer_offset = size_of::<FileHeader>() + self.total_bytes_out();
        let slot_offset = size_of::<FileFooter0>()
            + self.n_chunks() * size_of::<ChunkFooter>()
            + self.compressor.serialized_size()
            + offset_of!(FileFooter3, compression_elapsed_time_nano_secs);

        footer_offset + slot_offset
    }

    /// Writes the elapsed time from the start of the compression to the given writer.
    /// Caller must ensure that the writer has already seeked to the elapsed_time_slot
    pub fn write_elapsed_time<W: Write>(&self, mut f: W) -> io::Result<()> {
        let elapsed_time = self.elapsed_time();
        f.write_all(&elapsed_time.as_nanos().to_le_bytes())?;
        Ok(())
    }
}

pub struct ChunkWriter<'a, R: Read> {
    file_writer: &'a mut FileWriter<R>,
    compressor: GenericCompressor,
}

fn read_less_or_equal<R: Read>(mut f: R, n_bytes: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0.0; (n_bytes + 7) / size_of::<f64>()];
    let u8_slice = unsafe {
        let ptr = buf.as_mut_ptr().cast::<u8>();
        slice::from_raw_parts_mut(ptr, n_bytes)
    };
    let mut n_read = 0;
    while n_read < n_bytes {
        match f.read(&mut u8_slice[n_read..]) {
            Ok(0) => break,
            Ok(n) => n_read += n,
            Err(e)
                if e.kind() == io::ErrorKind::Interrupted
                    || e.kind() == io::ErrorKind::UnexpectedEof =>
            {
                if n_read == 0 {
                    return Err(e.into());
                }
                break;
            }
            Err(e) => return Err(e.into()),
        }
    }

    buf.truncate(n_read / size_of::<f64>());
    Ok(buf)
}

impl<'a, R: Read> ChunkWriter<'a, R> {
    fn new(file_writer: &'a mut FileWriter<R>, compressor: GenericCompressor) -> Self {
        Self {
            file_writer,
            compressor,
        }
    }

    pub fn into_compressor(self) -> GenericCompressor {
        self.compressor
    }

    pub fn write<W: Write>(&mut self, mut f: W) -> Result<()> {
        let chunk_option = *self.file_writer.chunk_option();

        match chunk_option {
            opt @ (ChunkOption::RecordCount(_) | ChunkOption::Full) => {
                let mut buf = if let ChunkOption::RecordCount(count) = opt {
                    read_less_or_equal(self.file_writer.input_mut(), count * size_of::<f64>())?
                } else {
                    let mut buf = Vec::new();
                    self.file_writer.input_mut().read_to_end(&mut buf)?;

                    buf.chunks_exact(size_of::<f64>())
                        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect::<Vec<f64>>()
                };
                if buf.is_empty() {
                    self.file_writer.set_reaches_eof();
                    return Ok(());
                }

                self.compressor.compress(buf.as_mut_slice())?;
            }
            ChunkOption::ByteSizeBestEffort(n) => {
                // TODO: implement good estimation
                // Simple heuristic to estimate the number of records to read.
                let record_count = ((n as f64 / size_of::<f64>() as f64) * 1.5).ceil() as usize;
                let mut buf = read_less_or_equal(
                    self.file_writer.input_mut(),
                    record_count * size_of::<f64>(),
                )?;
                if buf.is_empty() {
                    self.file_writer.set_reaches_eof();
                    return Ok(());
                }

                self.compressor.compress(buf.as_mut_slice())?;
            }
        }
        if self.compressor.total_bytes_in() == 0 {
            self.file_writer.set_reaches_eof();
            return Ok(());
        }

        let mut general_header = GeneralChunkHeader {};

        let mut total_bytes_out = 0;
        // TODO: handle error especially for `f` is too small to write.
        f.write_all(general_header.to_le().as_bytes())?;
        total_bytes_out += size_of::<GeneralChunkHeader>();

        self.compressor.prepare();

        for bytes in self.compressor.buffers() {
            total_bytes_out += bytes.len();
            f.write_all(bytes)?;
        }

        self.file_writer
            .increment_total_bytes_in_by(self.compressor.total_bytes_in());
        self.file_writer
            .increment_total_bytes_out_by(total_bytes_out);

        let next_physical_offset =
            (size_of::<FileHeader>() + self.file_writer.total_bytes_out()) as u64;
        let next_logical_offset = self
            .file_writer
            .chunk_footers_with_next_chunk_footer()
            .last()
            .map_or(0, |f| f.logical_offset)
            + (self.compressor.total_bytes_in() / size_of::<f64>()) as u64;
        let next_chunk_footer = ChunkFooter {
            physical_offset: next_physical_offset,
            logical_offset: next_logical_offset,
        };

        self.file_writer.push_chunk_footer(next_chunk_footer);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use format::{
        deserialize::{FromLeBytes, TryFromLeBytes},
        ChunkOptionKind, CompressionScheme,
    };

    use crate::compressor::uncompressed::UncompressedCompressorConfig;

    use super::*;
    use std::{io::Cursor, ptr::addr_of};

    #[test]
    fn test_write() -> Result<()> {
        let data = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
        let u8_data = unsafe {
            let ptr = data.as_ptr().cast::<u8>();
            slice::from_raw_parts(ptr, data.len() * size_of::<f64>())
        };
        const RECORD_COUNT: usize = 100;
        let compressor_config = CompressorConfig::uncompressed()
            .capacity(data.len() as u64)
            .build()
            .into();
        let mut in_f = u8_data;
        let mut out_f = Cursor::new(Vec::new());
        let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);
        let mut file_writer = FileWriter::new(&mut in_f, compressor_config, chunk_option);

        // write header leaving footer offset blank
        file_writer.write_header(&mut out_f)?;

        // loop until there are no data left
        while let Some(mut chunk_context) = file_writer.chunk_writer() {
            chunk_context.write(&mut out_f)?;
            // you can flush if you want
            out_f.flush()?;
        }

        file_writer.write_footer(&mut out_f)?;
        let footer_offset_slot_offset = file_writer.footer_offset_slot_offset();
        let mut dest_vec = out_f.into_inner();
        file_writer.write_footer_offset(&mut dest_vec[footer_offset_slot_offset..])?;
        let elapsed_time_slot_offset = file_writer.elapsed_time_slot_offset();
        file_writer.write_elapsed_time(&mut dest_vec[elapsed_time_slot_offset..])?;

        let file_header = FileHeader::try_from_le_bytes(&dest_vec[..]);
        match file_header {
            Ok(_) => {}
            Err(e) => {
                panic!("Failed to parse written FileHeader: {:?}", e);
            }
        }
        let file_header = file_header.unwrap();
        assert_eq!(
            file_header.magic_number,
            *FileWriter::<Cursor<Vec<u8>>>::MAGIC_NUMBER
        );
        let version = file_header.version;
        assert_eq!(
            version,
            [
                env!("CARGO_PKG_VERSION_MAJOR"),
                env!("CARGO_PKG_VERSION_MINOR"),
                env!("CARGO_PKG_VERSION_PATCH")
            ]
            .map(|v| v.parse::<u16>().unwrap())
        );
        assert_eq!(file_header.config.field_type, FieldType::F64);
        assert_eq!(
            file_header.config.chunk_option.kind,
            ChunkOptionKind::RecordCount
        );

        assert_eq!(
            unsafe { addr_of!(file_header.config.chunk_option.value).read_unaligned() },
            RECORD_COUNT as u64
        );

        assert_eq!(
            unsafe { addr_of!(file_header.config.compression_scheme).read_unaligned() },
            CompressionScheme::Uncompressed
        );

        let footer_offset = file_header.footer_offset as usize;

        let file_footer0 = FileFooter0::from_le_bytes(&dest_vec[footer_offset..]);

        let number_of_chunks = file_footer0.number_of_chunks;
        assert!(
            number_of_chunks <= 10,
            "number_of_chunks: {} must be less than or equal to 10",
            number_of_chunks
        );
        let number_of_records = file_footer0.number_of_records;
        assert_eq!(number_of_records, 1000);

        let fifth_chunk_footer = ChunkFooter::from_le_bytes(
            &dest_vec[footer_offset + size_of::<FileFooter0>() + 5 * size_of::<ChunkFooter>()..],
        );
        let fifth_chunk_physical_offset = fifth_chunk_footer.physical_offset as usize;
        let fifth_chunk_logical_offset = fifth_chunk_footer.logical_offset as usize;
        let data_in_src = data[fifth_chunk_logical_offset].to_le_bytes();

        let val_head = fifth_chunk_physical_offset;
        let data_in_dest = &dest_vec[val_head..(val_head + size_of::<f64>())];
        assert_eq!(data_in_src, data_in_dest);

        // TODO: check crc here if implemented

        let footer_size = size_of::<FileFooter0>()
            + number_of_chunks as usize * size_of::<ChunkFooter>()
            + size_of::<u8>()
            + size_of::<UncompressedCompressorConfig>()
            + size_of::<FileFooter3>();
        assert_eq!(
            dest_vec.len(),
            footer_offset + footer_size,
            "file size mismatch"
        );

        Ok(())
    }
}
