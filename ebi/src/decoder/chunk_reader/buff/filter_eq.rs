use roaring::RoaringBitmap;
#[cfg(all(target_arch = "x86_64", not(miri)))]
use simd_x86_64::equal_simd;

use crate::compression_common::buff::{
    bit_packing::BitPack,
    flip,
    precision_bound::{self},
};
use cfg_if::cfg_if;

/// Returns `next_result_bitmask`
#[inline]
fn buff_simd256_equal_filter_subcolumn_without_bitmask_nonsimd(
    chunk: &[u8],
    delta_fixed_pred_subcolumn: u8,
    logical_offset: u32,
) -> RoaringBitmap {
    let mut bitmask = RoaringBitmap::new();
    for (record_index, subcolumn) in chunk
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32 + logical_offset, x))
    {
        let subcolumn = flip(subcolumn);
        if subcolumn == delta_fixed_pred_subcolumn {
            bitmask.insert(record_index);
        }
    }

    bitmask
}

/// Returns `next_result_bitmask`
#[inline]
fn buff_simd256_equal_filter_subcolumn_without_bitmask(
    chunk: &[u8],
    delta_fixed_pred_subcolumn: u8,
    logical_offset: u32,
) -> RoaringBitmap {
    cfg_if! {
        if #[cfg(any(not(target_arch = "x86_64"), miri))] {
            buff_simd256_equal_filter_subcolumn_without_bitmask_nonsimd(
                chunk,
                delta_fixed_pred_subcolumn,
                logical_offset
            )
        } else {
            let (initial_part, simd_part, remaining_part) =
                super::filter::split_for_simd_processing(chunk, logical_offset);
            let next_result_bitmask0 =
                buff_simd256_equal_filter_subcolumn_without_bitmask_nonsimd(
                    initial_part,
                    delta_fixed_pred_subcolumn,
                    logical_offset,
                );
            let next_result_bitmask1 =
                unsafe {
                    equal_simd(simd_part, delta_fixed_pred_subcolumn, logical_offset + initial_part.len() as u32)
                };
            let next_result_bitmask2 =
                buff_simd256_equal_filter_subcolumn_without_bitmask_nonsimd(
                    remaining_part,
                    delta_fixed_pred_subcolumn,
                    logical_offset + initial_part.len() as u32 + simd_part.len() as u32,
                );

            next_result_bitmask0 | next_result_bitmask1 | next_result_bitmask2
        }
    }
}

#[inline]
pub(crate) fn buff_simd256_equal_filter(
    bytes: &[u8],
    pred: f64,
    bitmask: Option<RoaringBitmap>,
    logical_offset: u32,
) -> RoaringBitmap {
    // Read the header
    let mut bitpack = BitPack::<&[u8]>::new(bytes);

    let fixed_base64_lower = bitpack.read_u32().unwrap();
    let fixed_base64_higher = bitpack.read_u32().unwrap();
    let fixed_base64_bits = (fixed_base64_lower as u64) | ((fixed_base64_higher as u64) << 32);
    let fixed_base64 = u64::cast_signed(fixed_base64_bits);
    let number_of_records = bitpack.read_u32().unwrap();
    let fixed_representation_bits_length = bitpack.read_u32().unwrap();
    let fractional_part_bits_length = bitpack.read_u32().unwrap();

    let fixed_pred = precision_bound::into_fixed_representation_with_fractional_part_bits_length(
        pred,
        fractional_part_bits_length as i32,
    );

    let all = || {
        let mut all = RoaringBitmap::new();
        all.insert_range(logical_offset..(number_of_records + logical_offset));
        all
    };
    let none = RoaringBitmap::new;
    // First Skip all the evaluation based on delta base
    if fixed_pred < fixed_base64 {
        return none();
    }
    if fixed_representation_bits_length == 0 {
        return if fixed_pred == fixed_base64 {
            all()
        } else {
            none()
        };
    }

    let delta_fixed_pred = (fixed_pred - fixed_base64) as u64;

    let mut remaining_bits_length = fixed_representation_bits_length;

    let mut result_bitmask: Option<RoaringBitmap> = bitmask;

    while remaining_bits_length >= 8 {
        remaining_bits_length -= 8;

        let delta_fixed_pred_subcolumn = (delta_fixed_pred >> remaining_bits_length) as u8;
        let next_result_bitmask = if let Some(bitmask) = &result_bitmask {
            let mut next_result_bitmask = RoaringBitmap::new();
            let mut expected_record_index_if_sequential = 0;
            for record_index in bitmask.iter().map(|x| x - logical_offset) {
                bitpack
                    .skip_n_byte((record_index - expected_record_index_if_sequential) as usize)
                    .unwrap();
                let subcolumn = bitpack.read_byte().unwrap();
                let subcolumn = flip(subcolumn);
                if subcolumn == delta_fixed_pred_subcolumn {
                    next_result_bitmask.insert(logical_offset + record_index);
                }

                expected_record_index_if_sequential = record_index + 1;
            }
            bitpack
                .skip_n_byte(
                    number_of_records as usize - expected_record_index_if_sequential as usize,
                )
                .unwrap();

            next_result_bitmask
        } else {
            let chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();
            buff_simd256_equal_filter_subcolumn_without_bitmask(
                chunk,
                delta_fixed_pred_subcolumn,
                logical_offset,
            )
        };

        if next_result_bitmask.is_empty() {
            return next_result_bitmask;
        }

        result_bitmask.replace(next_result_bitmask);
    }

    // last subcolumn
    if remaining_bits_length > 0 {
        if let Some(bitmask) = &result_bitmask {
            let mask_for_remaining_bits = (1 << remaining_bits_length) - 1;
            let delta_fixed_pred_subcolumn = ((delta_fixed_pred) & mask_for_remaining_bits) as u8;
            let mut next_result_bitmask = RoaringBitmap::new();
            let mut expected_record_index_if_sequential = 0;
            for record_index in bitmask.iter().map(|x| x - logical_offset) {
                let n_bits_skipped =
                    (record_index - expected_record_index_if_sequential) * remaining_bits_length;
                bitpack.skip(n_bits_skipped as usize).unwrap();

                let subcolumn = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                // no flip for the last bits(less than byte) column
                if subcolumn == delta_fixed_pred_subcolumn {
                    next_result_bitmask.insert(logical_offset + record_index);
                }

                expected_record_index_if_sequential = record_index + 1;
            }
            let n_bits_skipped =
                (number_of_records - expected_record_index_if_sequential) * remaining_bits_length;
            bitpack.skip(n_bits_skipped as usize).unwrap();

            next_result_bitmask
        } else {
            let mut next_result_bitmask = RoaringBitmap::new();
            for i in 0..number_of_records {
                let subcolumn = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                if subcolumn == delta_fixed_pred as u8 {
                    next_result_bitmask.insert(logical_offset + i);
                }
            }

            next_result_bitmask
        }
    } else {
        result_bitmask.unwrap_or_else(all)
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod simd_x86_64 {
    use std::arch::x86_64::{
        __m256i, _mm256_cmpeq_epi8, _mm256_lddqu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
    };
    pub(super) const VECTOR_SIZE: usize = size_of::<__m256i>();

    use roaring::RoaringBitmap;

    use crate::compression_common::buff::flip;

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[inline]
    pub unsafe fn equal_simd(x: &[u8], pred: u8, logical_offset: u32) -> RoaringBitmap {
        if x.is_empty() {
            return RoaringBitmap::new();
        }
        debug_assert_eq!(
            x.len() % VECTOR_SIZE,
            0,
            "simd chunk size is not a multiple of 32"
        );
        debug_assert_eq!(logical_offset % 8, 0);

        let mut bitmasks = Vec::with_capacity(x.len() / VECTOR_SIZE);

        let pred_word = _mm256_set1_epi8(std::mem::transmute::<u8, i8>(flip(pred)));
        for word in x
            .chunks_exact(VECTOR_SIZE)
            .map(|chunk| _mm256_lddqu_si256(chunk.as_ptr().cast::<__m256i>()))
        {
            let equal = _mm256_cmpeq_epi8(word, pred_word);
            let eq_mask = _mm256_movemask_epi8(equal);

            bitmasks.push(eq_mask as u32);
        }

        RoaringBitmap::from_lsb0_bytes(logical_offset, unsafe {
            std::slice::from_raw_parts(
                bitmasks.as_ptr().cast::<u8>(),
                size_of_val(bitmasks.as_slice()),
            )
        })
    }
}
