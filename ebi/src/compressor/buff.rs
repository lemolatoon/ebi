use super::Compressor;

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
    fn compress(&mut self, data: &[f64]) -> usize {
        if self.compressed.is_some() {
            return 0;
        }
        self.compressed = Some(internal::buff_simd256_encode(self.scale, data));

        let n_bytes_compressed = size_of_val(data);
        self.total_bytes_in += n_bytes_compressed;

        n_bytes_compressed
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

mod internal {
    use std::mem;

    use crate::compression_common::buff::{
        bit_packing::BitPack,
        flip,
        precision_bound::{self, PRECISION_MAP},
    };

    pub fn buff_simd256_encode(scale: usize, floats: &[f64]) -> Vec<u8> {
        let mut fixed_representation_values = Vec::new();

        let number_of_records: u32 = floats.len() as u32;
        let precision = if scale == 0 {
            0
        } else {
            (scale as f32).log10() as i32
        };

        let fractional_part_bits_length = *PRECISION_MAP.get(precision as usize).unwrap();
        let mut min = i64::MAX;
        let mut max = i64::MIN;

        for &f in floats {
            let fixed = precision_bound::into_fixed_representation_with_fractional_part_bits_length(
                f,
                fractional_part_bits_length as i32,
            );
            if fixed < min {
                min = fixed;
            }
            if fixed > max {
                max = fixed;
            }
            fixed_representation_values.push(fixed);
        }
        let delta = max - min;
        let base_fixed64 = min;

        let delta_fixed_representation_values = fixed_representation_values
            .into_iter()
            .map(|x| (x - base_fixed64) as u64)
            .collect::<Vec<u64>>();

        let fixed_representation_bits_length = if delta == 0 {
            0
        } else {
            64 - delta.leading_zeros()
        };

        let fractional_part_bits_length = fractional_part_bits_length as usize;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);

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
