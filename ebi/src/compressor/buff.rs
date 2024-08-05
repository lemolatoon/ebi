use crate::compression_common::buff::precision_bound::{self, PRECISION_MAP};

use super::{size_estimater::NaiveSlowSizeEstimator, Compressor};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFCompressor {
    total_bytes_in: usize,
    data: Vec<f64>,
    scale: usize,
    compressed: Option<Vec<u8>>,
}

impl BUFFCompressor {
    pub fn new(scale: usize) -> Self {
        Self {
            total_bytes_in: 0,
            data: Vec::new(),
            scale,
            compressed: None,
        }
    }
}

impl Compressor for BUFFCompressor {
    // TODO: Implement BUFF size estimator
    type SizeEstimatorImpl<'comp, 'buf> = NaiveSlowSizeEstimator<'comp, 'buf, Self>;

    fn compress(&mut self, data: &[f64]) {
        self.compressed = Some(internal::buff_simd256_encode(Precalculated::precalculate(
            self.scale, data,
        )));

        let n_bytes_compressed = size_of_val(data);
        self.total_bytes_in += n_bytes_compressed;
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

    pub fn max(&self) -> i64 {
        self.fixed_representation_values[self.max]
    }

    pub fn min(&self) -> i64 {
        self.fixed_representation_values[self.min]
    }

    pub fn delta(&self) -> i64 {
        self.max() - self.min()
    }

    pub fn base_fixed64(&self) -> i64 {
        self.fixed_representation_values[self.min]
    }

    pub fn fixed_representation_bits_length(&self) -> u32 {
        if self.delta() == 0 {
            0
        } else {
            64 - self.delta().leading_zeros()
        }
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
        let fixed_representation_bits_length = self.fixed_representation_bits_length() as usize;

        let number_of_records = self.number_of_records();

        let header_size = size_of::<u32>() * 5;

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
        let base_fixed64 = precalculated.base_fixed64();

        let delta_fixed_representation_values = fixed_representation_values
            .iter()
            .map(|x| (x - base_fixed64) as u64)
            .collect::<Vec<u64>>();

        let fixed_representation_bits_length = precalculated.fixed_representation_bits_length();
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
