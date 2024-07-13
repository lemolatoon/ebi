use crate::{
    compressor::GenericCompressor,
    format::{
        self,
        serialize::{AsBytes, ToLe},
        ChunkFooter, FieldType, FileConfig, FileFooter0, FileFooter2, FileHeader,
        GeneralChunkHeader,
    },
};
use core::slice;
use std::{
    io::{self, BufRead, Write},
    mem::{offset_of, size_of, size_of_val},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkOption {
    Full,
    RecordCount(usize),
    ByteSize(usize),
}

impl From<&ChunkOption> for format::ChunkOption {
    fn from(value: &ChunkOption) -> Self {
        match value {
            ChunkOption::Full => format::ChunkOption {
                kind: format::ChunkOptionKind::Full,
                value: 0,
            },
            ChunkOption::RecordCount(value) => format::ChunkOption {
                kind: format::ChunkOptionKind::RecordCount,
                value: (*value as u64).to_le(),
            },
            ChunkOption::ByteSize(value) => format::ChunkOption {
                kind: format::ChunkOptionKind::ByteSize,
                value: (*value as u64).to_le(),
            },
        }
    }
}

impl ChunkOption {
    pub fn reach_limit(&self, total_bytes_in: usize, total_bytes_out: usize) -> bool {
        match self {
            ChunkOption::Full => false,
            ChunkOption::RecordCount(count) => count * size_of::<f64>() <= total_bytes_in,
            ChunkOption::ByteSize(n_bytes) => *n_bytes <= total_bytes_out,
        }
    }
}

pub struct FileWriter<R: BufRead> {
    input: R,
    chunk_option: ChunkOption,
    compressor: GenericCompressor,
    total_bytes_in: usize,
    total_bytes_out: usize,
    /// The vector of `ChunkFooter`, including the next `ChunkFooter`
    chunk_footers: Vec<ChunkFooter>,
    reaches_eof: bool,
}

struct BufWrapper<'a, R: BufRead> {
    input: &'a mut R,
    buf: &'a [f64],
    n_consumed_bytes: usize,
}

impl<'a, R: BufRead> BufWrapper<'a, R> {
    /// Provides the wrapper of internal buffer of BufRead.
    /// Tuple's first element is the `BufWrapper`, the second element indicates
    /// reader reaches EOF or not.
    pub fn new(input: &'a mut R) -> Result<(Self, bool), io::Error> {
        let buf = input.fill_buf()?;
        let reaches_eof = buf.is_empty();
        let buf_ptr = buf.as_ptr().cast::<f64>();
        let len = size_of_val(buf) / size_of::<f64>();
        // Safety:
        // input buffer is safely interpreted because the user of this struct guarantees
        // this byte stream is a f64 array.
        debug_assert!(size_of::<f64>() * len <= isize::MAX as usize);
        let buf: &'a [f64] = unsafe { slice::from_raw_parts(buf_ptr, len) };
        Ok((
            Self {
                input,
                buf,
                n_consumed_bytes: len * size_of::<f64>(),
            },
            reaches_eof,
        ))
    }

    pub fn set_n_consumed_bytes(&mut self, consumed: usize) {
        self.n_consumed_bytes = consumed;
    }
}

impl<'a, R: BufRead> AsRef<[f64]> for BufWrapper<'a, R> {
    fn as_ref(&self) -> &'a [f64] {
        self.buf
    }
}

impl<'a, R: BufRead> Drop for BufWrapper<'a, R> {
    fn drop(&mut self) {
        self.input.consume(self.n_consumed_bytes);
    }
}

impl<R: BufRead> FileWriter<R> {
    const MAGIC_NUMBER: &'static [u8; 4] = b"EBI1";
    pub fn new(input: R, compressor: GenericCompressor, chunk_option: ChunkOption) -> Self {
        let chunk_footer = ChunkFooter {
            physical_offset: size_of::<FileHeader>() as u64,
            logical_offset: 0,
        };
        Self {
            input,
            compressor,
            chunk_option,
            chunk_footers: vec![chunk_footer],
            total_bytes_in: 0,
            total_bytes_out: 0,
            reaches_eof: false,
        }
    }

    /// Returns the wrapper of the contents of the internal buffer, filling it with more data from the inner reader.
    /// If input has already reached EOF, the second element of tuple will be set.
    fn f64_buf(&mut self) -> Result<(BufWrapper<'_, R>, bool), io::Error> {
        BufWrapper::new(&mut self.input)
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
    pub fn chunk_writer<'a, 'b>(
        &'a mut self,
        buf: &'b mut Vec<u8>,
    ) -> Option<ChunkWriter<'a, 'b, R>> {
        if self.reaches_eof {
            return None;
        }
        Some(ChunkWriter::new(self, self.compressor.clone(), buf))
    }

    fn increment_total_bytes_in_by(&mut self, incremented: usize) {
        self.total_bytes_in += incremented;
    }

    fn increment_total_bytes_out_by(&mut self, incremented: usize) {
        self.total_bytes_out += incremented;
    }

    pub fn write_header<W: Write>(&mut self, mut f: W, buf: &mut Vec<u8>) -> io::Result<()> {
        let header_size = size_of::<FileHeader>();
        if buf.len() * size_of::<u8>() < header_size {
            let next_len = header_size.next_power_of_two();
            buf.resize(next_len, 0);
        }
        let slice = &mut buf[..header_size];
        let header_ptr = slice.as_mut_ptr().cast::<FileHeader>();

        // can be static?
        let version: [u16; 3] = [
            env!("CARGO_PKG_VERSION_MAJOR"),
            env!("CARGO_PKG_VERSION_MINOR"),
            env!("CARGO_PKG_VERSION_PATCH"),
        ]
        .map(|v| v.parse().unwrap_or(u16::MAX).to_le());
        let header = FileHeader {
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
        // Safety:
        // these memory regions are allocated as Vec<u8>
        unsafe {
            header_ptr.write(header);
        };

        f.write_all(&buf[..header_size])
    }

    pub fn write_footer<W: Write>(&mut self, mut f: W, buf: &mut Vec<u8>) -> io::Result<()> {
        let footer_size = size_of::<FileFooter0>()
            + size_of::<ChunkFooter>() * self.n_chunks()
            + size_of::<FileFooter2>();

        if buf.len() < footer_size {
            let next_len = buf.len().next_power_of_two();
            buf.resize(next_len, 0);
        }

        let footer0_ptr = buf.as_mut_ptr().cast::<FileFooter0>();
        let number_of_chunks = (self.n_chunks() as u64).to_le();
        let number_of_records = self
            .chunk_footers_with_next_chunk_footer()
            .last()
            .map_or(0, |f| f.logical_offset);
        let footer0 = FileFooter0 {
            number_of_chunks,
            number_of_records,
        };
        // Safety:
        // the pointer destination is valid, this is originated from `buf`.
        unsafe {
            footer0_ptr.write_unaligned(footer0);
        }

        // Safety:
        // the pointer will be in bounds of `buf`.
        let footer1_ptr = unsafe { footer0_ptr.offset(1).cast::<ChunkFooter>() };
        for (i, chunk_footer) in self.chunk_footers().iter().enumerate() {
            let mut chunk_footer = *chunk_footer;
            chunk_footer = chunk_footer.to_le();
            // Safety:
            //  - The pointer will be in bounds of `buf`.
            //  - Also that's why the pointer is valid.
            unsafe { footer1_ptr.add(i).write_unaligned(chunk_footer) }
        }

        // Safety:
        // The pointer will be in bounds in `buf`.
        let footer2_ptr = unsafe { footer1_ptr.add(self.n_chunks()).cast::<FileFooter2>() };
        // TODO: calculate `crc` here.
        let footer2 = FileFooter2 { crc: 0u32.to_le() };
        // Safety:
        //  - The pointer will be in bounds of `buf`.
        //  - Also that's why the pointer is valid.
        unsafe {
            footer2_ptr.write_unaligned(footer2);
        }

        f.write_all(&buf[..footer_size])
    }

    pub fn write_footer_offset<W: Write>(&mut self, mut f: W) -> io::Result<()> {
        let footer_offset: u64 = (size_of::<FileHeader>() + self.total_bytes_out()) as u64;
        f.write_all(&footer_offset.to_le_bytes())?;

        Ok(())
    }
}

pub struct ChunkWriter<'a, 'b, R: BufRead> {
    file_writer: &'a mut FileWriter<R>,
    compressor: GenericCompressor,
    buf: &'b mut Vec<u8>,
}

impl<'a, 'b, R: BufRead> ChunkWriter<'a, 'b, R> {
    fn new(
        file_writer: &'a mut FileWriter<R>,
        compressor: GenericCompressor,
        buf: &'b mut Vec<u8>,
    ) -> Self {
        Self {
            file_writer,
            compressor,
            buf,
        }
    }

    pub fn write<W: Write>(&'a mut self, mut f: W) -> Result<(), io::Error> {
        let header_size = size_of::<GeneralChunkHeader>() + self.compressor.header_size();

        if self.buf.len() < header_size {
            let next_len = header_size.next_power_of_two();
            self.buf.resize(next_len, 0);
        } else if !self.buf.len().is_power_of_two() {
            let next_len = self.buf.len().next_power_of_two();
            self.buf.resize(next_len, 0);
        }

        loop {
            let reaches_limit = self.file_writer.chunk_option().reach_limit(
                self.compressor.total_bytes_in(),
                self.compressor.total_bytes_out(),
            );
            if reaches_limit {
                break;
            }
            let (mut buf, reaches_eof) = self.file_writer.f64_buf()?;
            if reaches_eof {
                drop(buf);
                self.file_writer.set_reaches_eof();
                break;
            }

            let total_bytes_out = self.compressor.total_bytes_out();
            let n_bytes_compressed = self.compressor.compress(
                buf.as_ref(),
                &mut self.buf[(header_size + total_bytes_out)..],
            );
            buf.set_n_consumed_bytes(n_bytes_compressed);

            if n_bytes_compressed == 0 {
                // expand `self.buf` x2.
                let buf_len = self.buf.len();
                self.buf.resize(buf_len * 2, 0);
            }
        }

        if self.compressor.total_bytes_in() == 0 {
            return Ok(());
        }

        let mut header = GeneralChunkHeader {};
        self.buf[..size_of::<GeneralChunkHeader>()].copy_from_slice(header.to_le().as_bytes());
        self.compressor
            .write_header(&mut self.buf[size_of::<GeneralChunkHeader>()..header_size]);

        // TODO: handle error especially for `f` is too small to write.
        f.write_all(&self.buf[..(header_size + self.compressor.total_bytes_out())])?;

        self.file_writer
            .increment_total_bytes_in_by(self.compressor.total_bytes_in());
        self.file_writer
            .increment_total_bytes_out_by(header_size + self.compressor.total_bytes_out());

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
        uncompressed::UncompressedHeader0,
        ChunkOptionKind, CompressionScheme,
    };

    use crate::compressor::uncompressed::UncompressedCompressor;

    use super::*;
    use std::{
        io::{self, BufReader, Cursor},
        ptr::addr_of,
    };

    #[test]
    fn test_write() -> io::Result<()> {
        let data = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
        let u8_data = unsafe {
            let ptr = data.as_ptr().cast::<u8>();
            slice::from_raw_parts(ptr, data.len() * size_of::<f64>())
        };
        const RECORD_COUNT: usize = 100;
        let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new());
        let mut in_f = BufReader::new(&u8_data[..]);
        let mut out_f = Cursor::new(Vec::new());
        let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);
        let mut file_writer = FileWriter::new(&mut in_f, compressor, chunk_option);

        let mut buf = Vec::<u8>::new();
        // write header leaving footer offset blank
        file_writer.write_header(&mut out_f, &mut buf)?;

        // loop until there are no data left
        while let Some(mut chunk_context) = file_writer.chunk_writer(&mut buf) {
            chunk_context.write(&mut out_f)?;
            // you can flush if you want
            out_f.flush()?;
        }

        file_writer.write_footer(&mut out_f, &mut buf)?;
        let footer_offset_slot_offset = file_writer.footer_offset_slot_offset();
        let mut dest_vec = out_f.into_inner();
        file_writer.write_footer_offset(&mut dest_vec[footer_offset_slot_offset..])?;

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
            *FileWriter::<BufReader<Cursor<Vec<u8>>>>::MAGIC_NUMBER
        );
        assert_eq!(
            unsafe { addr_of!(file_header.version).read_unaligned() },
            [
                env!("CARGO_PKG_VERSION_MAJOR"),
                env!("CARGO_PKG_VERSION_MINOR"),
                env!("CARGO_PKG_VERSION_PATCH")
            ]
            .map(|v| v.parse().unwrap())
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

        let header_head =
            UncompressedHeader0::from_le_bytes(&dest_vec[fifth_chunk_physical_offset..]);
        let header_size = header_head.header_size as usize;
        let val_head = header_size + fifth_chunk_physical_offset;
        let data_in_dest = &dest_vec[val_head..(val_head + size_of::<f64>())];
        assert_eq!(data_in_src, data_in_dest);

        // TODO: check crc here if implemented

        let footer_size = size_of::<FileFooter0>()
            + number_of_chunks as usize * size_of::<ChunkFooter>()
            + size_of::<FileFooter2>();
        assert_eq!(
            dest_vec.len(),
            footer_offset + footer_size,
            "file size mismatch"
        );

        Ok(())
    }
}
