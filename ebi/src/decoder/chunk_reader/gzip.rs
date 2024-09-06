use std::{io::Read, iter, slice};

use flate2::read::GzDecoder;

use crate::{
    decoder::{self, query::QueryExecutor, FileMetadataLike, GeneralChunkHandle},
    time::{SegmentKind, SegmentedExecutionTimes},
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GzipReader<R: Read> {
    reader: R,
    chunk_size: usize,
    decompressed: Vec<f64>,
    is_decompressed: bool,
}

impl<R: Read> GzipReader<R> {
    pub fn new<F: FileMetadataLike>(handle: &GeneralChunkHandle<F>, reader: R) -> Self {
        let number_of_records = handle.number_of_records() as usize;
        let chunk_size = handle.chunk_size() as usize;
        Self {
            reader,
            chunk_size,
            decompressed: vec![0.0; number_of_records],
            is_decompressed: false,
        }
    }
}

type F = fn(&f64) -> decoder::Result<f64>;
pub type ZstdDecompressIterator<'a> = iter::Map<slice::Iter<'a, f64>, F>;
impl<R: Read> Reader for GzipReader<R> {
    type NativeHeader = ();

    type DecompressIterator<'a> = ZstdDecompressIterator<'a>
    where
        Self: 'a;

    fn decompress(&mut self, timer: &mut SegmentedExecutionTimes) -> decoder::Result<&[f64]> {
        if self.is_decompressed {
            return Ok(self.decompressed.as_slice());
        }

        let buf_ptr = self.decompressed.as_mut_ptr().cast::<u8>();
        let buf_len = size_of_val(self.decompressed.as_slice());
        let buf = unsafe { slice::from_raw_parts_mut(buf_ptr, buf_len) };

        // TODO: If underlying reader is in-memory, we can avoid copying
        let mut src_buf = vec![0; self.chunk_size];
        let io_read_timer = timer.start_addition_measurement(SegmentKind::IORead);
        self.reader.read_exact(&mut src_buf)?;
        io_read_timer.stop();

        let mut decoder = GzDecoder::new(src_buf.as_slice());
        let decompression_timer = timer.start_addition_measurement(SegmentKind::Decompression);
        decoder.read_exact(buf)?;
        decompression_timer.stop();
        debug_assert_eq!(
            decoder.read(&mut [0]).ok(),
            Some(0),
            "Should have read all bytes"
        );

        self.is_decompressed = true;

        Ok(self.decompressed.as_slice())
    }

    fn decompress_iter(&mut self) -> decoder::Result<Self::DecompressIterator<'_>> {
        if !self.is_decompressed {
            self.decompress(&mut SegmentedExecutionTimes::new())?;
        }

        Ok(self.decompressed.iter().map(|&x| Ok(x)))
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed = data;
        self.is_decompressed = true;

        self.decompressed.as_slice()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        if self.is_decompressed {
            Some(self.decompressed.as_slice())
        } else {
            None
        }
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
    }
}

impl<R: Read> QueryExecutor for GzipReader<R> {}
