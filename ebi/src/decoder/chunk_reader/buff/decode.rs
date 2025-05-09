use std::mem;

use crate::compression_common::buff::{bit_packing::BitPack, flip};

pub fn buff_simd256_decode(bytes: &[u8]) -> Vec<f64> {
    let mut bitpack = BitPack::<&[u8]>::new(bytes);
    let lower = bitpack.read_u32().unwrap();
    let higher = bitpack.read_u32().unwrap();
    let base_fixed64_bits = (lower as u64) | ((higher as u64) << 32);
    let base_fixed64 = unsafe { mem::transmute::<u64, i64>(base_fixed64_bits) };

    let number_of_records = bitpack.read_u32().unwrap();

    let fixed_representation_bits_length = bitpack.read_u32().unwrap();

    let fractional_part_bits_length = bitpack.read_u32().unwrap();

    let scale: f64 = 2.0f64.powi(fractional_part_bits_length as i32);

    let mut remaining_bits_length = fixed_representation_bits_length;

    let expected_datapoints: Vec<f64> = if remaining_bits_length == 0 {
        vec![base_fixed64 as f64 / scale; number_of_records as usize]
    } else if remaining_bits_length < 8 {
        let mut expected_datapoints = Vec::with_capacity(number_of_records as usize);
        for _ in 0..number_of_records {
            let cur = bitpack.read_bits(remaining_bits_length as usize).unwrap();
            expected_datapoints.push((base_fixed64 + cur as i64) as f64 / scale);
        }

        expected_datapoints
    } else {
        let mut fixed_vec: Vec<u64> = Vec::with_capacity(number_of_records as usize);
        remaining_bits_length -= 8;
        let subcolumn_chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();

        for x in subcolumn_chunk {
            fixed_vec.push((flip(*x) as u64) << remaining_bits_length)
        }

        while remaining_bits_length >= 8 {
            remaining_bits_length -= 8;
            let subcolumn_chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();

            for (fixed, chunk) in fixed_vec.iter_mut().zip(subcolumn_chunk.iter()) {
                *fixed |= (flip(*chunk) as u64) << remaining_bits_length;
            }
        }

        if remaining_bits_length > 0 {
            let last_subcolumn_chunk = (0..number_of_records)
                .map(|_| bitpack.read_bits(remaining_bits_length as usize).unwrap() as u64);

            fixed_vec
                .into_iter()
                .zip(last_subcolumn_chunk)
                .map(|(fixed, last_subcolumn)| {
                    let delta_fixed = fixed | last_subcolumn;
                    (base_fixed64 + delta_fixed as i64) as f64 / scale
                })
                .collect()
        } else {
            fixed_vec
                .into_iter()
                .map(|x| (base_fixed64 + x as i64) as f64 / scale)
                .collect()
        }
    };
    expected_datapoints
}
