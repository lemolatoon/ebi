use std::mem;

use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::io::bit_write::{BitWrite, BufferedBitWriter};

use super::{Capacity, Compressor};

type BitWriter = BufferedBitWriter;

pub type DeltaSprintzCompressor = DeltaSprintzCompressorImpl<BitWriter>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct DeltaSprintzCompressorImpl<W: BitWrite> {
    w: W,
    /// Used for temporarily store delta_zigzag_encoded quantized floats
    buffer: Vec<u64>,
    scale: u32,
    total_bytes_in: usize,
}

#[derive(Builder, Debug, Clone, Copy)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeltaSprintzCompressorConfig {
    #[builder(setter(into), default)]
    capacity: Capacity,
    scale: u32,
}

impl DeltaSprintzCompressorConfigBuilder {
    pub fn build(self) -> DeltaSprintzCompressorConfig {
        DeltaSprintzCompressorConfig {
            capacity: self.capacity.unwrap_or_default(),
            scale: self.scale.unwrap_or(1),
        }
    }
}

impl From<&DeltaSprintzCompressorConfig> for DeltaSprintzCompressorImpl<BitWriter> {
    fn from(c: &DeltaSprintzCompressorConfig) -> Self {
        Self::new(BufferedBitWriter::with_capacity(c.capacity.0), c.scale)
    }
}

impl<W: BitWrite> DeltaSprintzCompressorImpl<W> {
    pub fn new(bit_writer: W, scale: u32) -> Self {
        Self {
            w: bit_writer,
            buffer: Vec::new(),
            total_bytes_in: 0,
            scale,
        }
    }
}

impl<W: BitWrite> Compressor for DeltaSprintzCompressorImpl<W> {
    fn compress(&mut self, input: &[f64]) {
        self.reset();
        // quantize
        self.total_bytes_in += size_of_val(input);
        delta_impl::delta_sprintz_compress_impl(input, self.scale, &mut self.buffer, &mut self.w);
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        self.w.as_slice().len()
    }

    fn prepare(&mut self) {}

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];

        [self.w.as_slice(), E, E, E, E]
    }

    fn reset(&mut self) {
        self.w.reset();
        self.buffer.clear();
        self.total_bytes_in = 0;
    }
}

mod delta_impl {
    use std::{iter, mem};

    use super::{zigzag, BitWrite};

    /// Compress input floats and writes to w
    /// buffer is provided as already allocated buffer
    /// to store delta_zigzag_encoded before writing as bit packed form
    /// Caller must ensure buffer is empty.
    pub fn delta_sprintz_compress_impl<W: BitWrite>(
        input: &[f64],
        scale: u32,
        buffer: &mut Vec<u64>,
        mut w: W,
    ) {
        debug_assert!(buffer.is_empty());
        let quantized = input
            .iter()
            .copied()
            .map(|fp| (fp * scale as f64).round() as i64);
        let (initial_number, number_of_bits_needed) = zigzag_delta_num_bits(quantized, buffer);

        // write header
        let initial_number_bits = unsafe { mem::transmute::<i64, u64>(initial_number) };
        w.write_bits(initial_number_bits, 64);
        w.write_bits(scale as u64, 32);
        w.write_byte(number_of_bits_needed);

        if number_of_bits_needed == 0 {
            return;
        }
        for v in buffer.iter().copied() {
            w.write_bits(v, number_of_bits_needed as u32);
        }
    }
    /// zigzag_delta encode `quantized_input` and store them into buffer.
    /// Also returns the initial_number of quantized_input and number_of_bits_needed for
    /// bitpacking the zigzag_delta encoded quantized_input
    pub fn zigzag_delta_num_bits(
        mut quantized_input: impl ExactSizeIterator<Item = i64>,
        buffer: &mut Vec<u64>,
    ) -> (i64, u8) {
        let number_of_records = quantized_input.len();
        let delta_zigzag_encoded = buffer;
        delta_zigzag_encoded.reserve(number_of_records);

        let initial_number = quantized_input.next().unwrap();
        let mut previous = initial_number;

        let mut encoded_ptr = delta_zigzag_encoded.as_mut_ptr();
        let mut accumulated_or = 0;

        for f in iter::once(initial_number).chain(quantized_input) {
            let delta = f - previous;
            let zz = zigzag(delta);
            // Safety:
            // We called delta_zigzag_encoded.reserve in advance to ensure
            // the internal buffer of vector is allocated
            unsafe {
                encoded_ptr.write(zz);
                encoded_ptr = encoded_ptr.add(1);
            };
            accumulated_or |= zz;
            previous = f;
        }
        // Safety:
        // 0..quantized_input.len() is properly initialized the loop above
        unsafe {
            delta_zigzag_encoded.set_len(number_of_records);
        };
        let lead = accumulated_or.leading_zeros();
        let number_of_bits_needed: u8 = (64 - lead) as u8;
        (initial_number, number_of_bits_needed)
    }
}

/// Do zigzag encoding
/// Maps negative values to positive values while going back and forth
/// (0 = 0, -1 = 1, 1 = 2, -2 = 3, 2 = 4, -3 = 5, 3 = 6 ...)
/// # Example
/// ```
/// use ebi::compressor::sprintz::zigzag;
/// assert_eq!(zigzag(0), 0);
/// assert_eq!(zigzag(-1), 1);
/// assert_eq!(zigzag(2), 4);
/// assert_eq!(zigzag(-100000), 199999);
///
/// assert_eq!(zigzag(i64::MAX as i64 / 2), i64::MAX as u64 - 1);
/// ```
#[inline]
pub const fn zigzag(origin: i64) -> u64 {
    let zzu = (origin << 1) ^ (origin >> 63);
    unsafe { mem::transmute::<i64, u64>(zzu) }
}
