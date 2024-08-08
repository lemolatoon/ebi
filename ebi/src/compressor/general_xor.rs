use std::marker::PhantomData;

use derive_builder::Builder;

use crate::io::bit_write::{BitWrite, BufferedBitWriter};

use super::{Capacity, Compressor};

pub trait XorEncoder: Default {
    /// Compress the passed floats,
    /// Returns the increased size by bits
    fn compress_float<W: BitWrite>(&mut self, w: W, bits: u64) -> usize;

    /// Write end marker at the last
    fn close<W: BitWrite>(&mut self, w: W);

    /// Simulate the close operation and returns incresed size by bits
    fn simulate_close(&self) -> usize;

    /// Reset the state of encoder
    fn reset(&mut self);
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GeneralXorCompressor<W: BitWrite, T: XorEncoder + Default> {
    w: W,
    encoder: T,
    total_bytes_in: usize,
}

type BitWriter = BufferedBitWriter;

impl<T: XorEncoder> GeneralXorCompressor<BitWriter, T> {
    pub fn new() -> Self {
        Self::with_capacity(1024 * 8)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        GeneralXorCompressor {
            w: BitWriter::with_capacity(capacity),
            encoder: T::default(),
            total_bytes_in: 0,
        }
    }
}

impl<T: XorEncoder> Default for GeneralXorCompressor<BitWriter, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Builder, Debug, Clone, Copy)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct GeneralXorCompressorConfig<T: XorEncoder> {
    #[builder(setter(into), default)]
    pub(crate) capacity: Capacity,
    #[builder(setter(skip))]
    _encoder_type: PhantomData<T>,
}

impl<T: XorEncoder> GeneralXorCompressorConfigBuilder<T> {
    pub fn build(self) -> GeneralXorCompressorConfig<T> {
        let Self { capacity, .. } = self;
        GeneralXorCompressorConfig {
            capacity: capacity.unwrap_or(Capacity::default()),
            _encoder_type: PhantomData,
        }
    }
}

impl<T: XorEncoder> From<GeneralXorCompressorConfig<T>> for GeneralXorCompressor<BitWriter, T> {
    fn from(value: GeneralXorCompressorConfig<T>) -> Self {
        let capacity = value.capacity;
        Self::with_capacity(capacity.0)
    }
}

impl<W: BitWrite, T: XorEncoder> Compressor for GeneralXorCompressor<W, T> {
    fn compress(&mut self, input: &[f64]) {
        self.reset();
        self.total_bytes_in += size_of_val(input);

        for value in input {
            self.encoder.compress_float(&mut self.w, value.to_bits());
        }
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        // EOF: f64::NAN + 1 bit
        let n_bits = self.w.as_slice().len() * 8 + self.encoder.simulate_close();

        n_bits.next_multiple_of(8) / 8
    }

    fn prepare(&mut self) {
        self.encoder.close(&mut self.w);
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];
        [self.w.as_slice(), E, E, E, E]
    }

    fn reset(&mut self) {
        self.w.reset();
        self.encoder.reset();
        self.total_bytes_in = 0;
    }
}
