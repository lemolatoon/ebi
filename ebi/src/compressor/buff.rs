use derive_builder::Builder;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    compression_common::buff::precision_bound::{self, PRECISION_MAP},
    format::{deserialize, serialize},
};

use super::Compressor;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFCompressor {
    total_bytes_in: usize,
    data: Vec<f64>,
    scale: u32,
    compressed: Option<Vec<u8>>,
}

impl BUFFCompressor {
    pub fn new(scale: u32) -> Self {
        Self {
            total_bytes_in: 0,
            data: Vec::new(),
            scale,
            compressed: None,
        }
    }

    fn compress_with_precalculated(&mut self, precalculated: Precalculated) {
        let number_of_records = precalculated.number_of_records();
        self.compressed = Some(internal::buff_simd256_encode(precalculated));

        let n_bytes_compressed = number_of_records * size_of::<f64>();
        self.total_bytes_in += n_bytes_compressed;
    }
}

impl Compressor for BUFFCompressor {
    fn compress(&mut self, data: &[f64]) {
        self.compress_with_precalculated(Precalculated::precalculate(self.scale as usize, data));
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        self.compressed.as_ref().map_or(0, |v| v.len())
    }

    fn prepare(&mut self) {}

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const EMPTY: &[u8] = &[];
        let data = self.compressed.as_ref().map_or(EMPTY, |v| v.as_slice());
        [data, EMPTY, EMPTY, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        self.total_bytes_in = 0;
        self.data.clear();
        self.compressed = None;
    }
}

#[derive(Builder, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[builder(pattern = "owned", build_fn(skip))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, packed)]
pub struct BUFFCompressorConfig {
    scale: u32,
}

serialize::impl_to_le!(BUFFCompressorConfig, scale);
deserialize::impl_from_le_bytes!(BUFFCompressorConfig, buff, (scale, u32));

impl From<BUFFCompressorConfig> for BUFFCompressor {
    fn from(c: BUFFCompressorConfig) -> Self {
        Self::new(c.scale)
    }
}

impl BUFFCompressorConfigBuilder {
    pub fn build(self) -> BUFFCompressorConfig {
        BUFFCompressorConfig {
            scale: self.scale.unwrap_or(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Precalculated {
    min: usize,
    max: usize,
    fixed_representation_values: Vec<i64>,
    precision: usize,
}

impl Precalculated {
    pub fn precalculate(scale: usize, floats: &[f64]) -> Self {
        let mut fixed_representation_values = Vec::new();

        let precision = if scale == 0 {
            0
        } else {
            (scale as f32).log10() as usize
        };

        let fractional_part_bits_length = *PRECISION_MAP.get(precision).unwrap();
        let mut min = i64::MAX;
        let mut min_index = 0;
        let mut max = i64::MIN;
        let mut max_index = 0;

        for (i, &f) in floats.iter().enumerate() {
            let fixed = precision_bound::into_fixed_representation_with_fractional_part_bits_length(
                f,
                fractional_part_bits_length as i32,
            );
            if fixed < min {
                min = fixed;
                min_index = i;
            }
            if fixed > max {
                max = fixed;
                max_index = i;
            }
            fixed_representation_values.push(fixed);
        }

        Self {
            min: min_index,
            max: max_index,
            fixed_representation_values,
            precision,
        }
    }

    #[inline]
    pub fn max_fixed(&self) -> Option<i64> {
        self.fixed_representation_values.get(self.max).copied()
    }

    #[inline]
    pub fn min_fixed(&self) -> Option<i64> {
        self.fixed_representation_values.get(self.min).copied()
    }

    #[inline]
    pub fn delta(&self) -> Option<i64> {
        match (self.min_fixed(), self.max_fixed()) {
            (Some(min), Some(max)) => Some(max - min),
            _ => None,
        }
    }

    pub fn fixed_representation_bits_length(&self) -> Option<u32> {
        let delta = self.delta()?;
        Some(if delta == 0 {
            0
        } else {
            64 - delta.leading_zeros()
        })
    }

    pub fn fixed_representation_values(&self) -> &[i64] {
        &self.fixed_representation_values[..]
    }

    pub fn fractional_part_bits_length(&self) -> u64 {
        *PRECISION_MAP.get(self.precision).unwrap()
    }

    pub fn number_of_records(&self) -> usize {
        self.fixed_representation_values.len()
    }

    pub fn compressed_size(&self) -> usize {
        let number_of_records = self.number_of_records();

        let header_size = size_of::<u32>() * 5;
        if number_of_records == 0 {
            return header_size;
        }

        let fixed_representation_bits_length =
            self.fixed_representation_bits_length().unwrap() as usize;

        let fixed_floats_size =
            (fixed_representation_bits_length * number_of_records).next_multiple_of(8) / 8;

        header_size + fixed_floats_size
    }
}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{bit_packing::BitPack, flip};

    use super::Precalculated;

    pub fn buff_simd256_encode(precalculated: Precalculated) -> Vec<u8> {
        let fixed_representation_values = precalculated.fixed_representation_values();
        let base_fixed64 = precalculated.min_fixed().unwrap();

        let delta_fixed_representation_values = fixed_representation_values
            .iter()
            .map(|x| (x - base_fixed64) as u64)
            .collect::<Vec<u64>>();

        let fixed_representation_bits_length =
            precalculated.fixed_representation_bits_length().unwrap();
        let fractional_part_bits_length = precalculated.fractional_part_bits_length() as usize;
        let number_of_records = precalculated.number_of_records() as u32;

        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(precalculated.compressed_size());

        let base_fixed64_bits: u64 = unsafe { mem::transmute(base_fixed64) };
        let base_fixed64_low = base_fixed64_bits as u32;
        let base_fixed64_high = (base_fixed64_bits >> 32) as u32;

        // Safety:
        // The following unwraps are safe because bits length is always less than 32
        unsafe {
            bitpack_vec.write(base_fixed64_low, 32).unwrap_unchecked();
            bitpack_vec.write(base_fixed64_high, 32).unwrap_unchecked();
            bitpack_vec.write(number_of_records, 32).unwrap_unchecked();
            bitpack_vec
                .write(fixed_representation_bits_length, 32)
                .unwrap_unchecked();
            bitpack_vec
                .write(fractional_part_bits_length as u32, 32)
                .unwrap_unchecked();
        }

        let mut remaining_bits_length = fixed_representation_bits_length as usize;

        if 0 < remaining_bits_length && remaining_bits_length < 8 {
            for delta_fixed in delta_fixed_representation_values {
                bitpack_vec
                    .write_bits(delta_fixed as u32, remaining_bits_length)
                    .unwrap();
            }
        } else {
            while remaining_bits_length >= 8 {
                remaining_bits_length -= 8;
                if remaining_bits_length > 0 {
                    for &delta_fixed in &delta_fixed_representation_values {
                        bitpack_vec.write_byte(flip((delta_fixed >> remaining_bits_length) as u8));
                    }
                } else {
                    for &delta_fixed in &delta_fixed_representation_values {
                        bitpack_vec.write_byte(flip(delta_fixed as u8));
                    }
                }
            }
            if remaining_bits_length > 0 {
                bitpack_vec.finish_write_byte();
                for d in delta_fixed_representation_values {
                    bitpack_vec
                        .write_bits(d as u32, remaining_bits_length)
                        .unwrap();
                }
            }
        }
        bitpack_vec.into_vec()
    }
}
