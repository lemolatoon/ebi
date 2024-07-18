use tsz::{stream::BufferedWriter, StdEncoder};

use super::Compressor;

pub struct GorillaCompressor {
    encoder: tsz::StdEncoder<BufferedWriter>,
}

impl GorillaCompressor {
    pub fn new() -> Self {
        let w = BufferedWriter::new();
        // TODO: timestamp?
        let encoder = StdEncoder::new(0, w);
        Self { encoder }
    }
}

impl Compressor for GorillaCompressor {
    fn compress(&mut self, _input: &[f64]) -> usize {
        unimplemented!()
    }

    fn total_bytes_in(&self) -> usize {
        todo!()
    }

    fn total_bytes_buffered(&self) -> usize {
        todo!()
    }

    fn prepare(&mut self) {
        todo!()
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        todo!()
    }

    fn reset(&mut self) {
        todo!()
    }
}
