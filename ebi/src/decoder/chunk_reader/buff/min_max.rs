use std::mem;

use roaring::RoaringBitmap;

use crate::{
    compression_common::buff::{bit_packing::BitPack, flip},
    decoder::{self, error::DecoderError},
};

/// Returns `next_to_check` bitmask
fn find_max_subcolumn(
    chunk: &[u8],
    to_check: &roaring::RoaringBitmap,
    logical_offset: u32,
) -> decoder::Result<roaring::RoaringBitmap> {
    let mut next_to_check = roaring::RoaringBitmap::new();
    let mut max_subcolumn = u8::MIN;
    for i in to_check.iter().map(|x| x - logical_offset) {
        let subcolumn = chunk[i as usize];
        let subcolumn = flip(subcolumn);
        if subcolumn == max_subcolumn {
            next_to_check.insert(i + logical_offset);
        } else if subcolumn > max_subcolumn {
            next_to_check.clear();
            max_subcolumn = subcolumn;
            next_to_check.insert(i + logical_offset);
        }
    }

    Ok(next_to_check)
}

pub(super) fn max_with_bitmask(
    bytes: &[u8],
    bitmask: roaring::RoaringBitmap,
    logical_offset: u32,
) -> decoder::Result<f64> {
    let mut bitpack = BitPack::new(bytes);
    // Read header values
    let lower = bitpack.read_u32().unwrap();
    let higher = bitpack.read_u32().unwrap();
    let base_fixed64_bits = (lower as u64) | ((higher as u64) << 32);
    let base_fixed64 = unsafe { mem::transmute::<u64, i64>(base_fixed64_bits) };

    let number_of_records = bitpack.read_u32().unwrap();

    let fixed_representation_bits_length = bitpack.read_u32().unwrap();

    let fractional_part_bits_length = bitpack.read_u32().unwrap();

    let scale: f64 = 2.0f64.powi(fractional_part_bits_length as i32);

    let mut remaining_bits_length = fixed_representation_bits_length;

    // All values are the same
    if remaining_bits_length == 0 {
        return Ok(base_fixed64 as f64 / scale);
    }

    let mut to_check = bitmask;
    while remaining_bits_length > 0 {
        let mut next_to_check = RoaringBitmap::new();
        // If the remaining bits are less than 8, it will be the last subcolumn
        // we treat it separately
        if remaining_bits_length < 8 {
            let mut expected_record_index_if_sequential = 0;
            let mut max_subcolumn = u8::MIN;
            for i in to_check.iter().map(|x| x - logical_offset) {
                let n_records_to_skip = i - expected_record_index_if_sequential;
                expected_record_index_if_sequential = i + 1;
                let n_bits_to_skip = n_records_to_skip * remaining_bits_length;
                bitpack
                    .skip(n_bits_to_skip as usize)
                    .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
                let subcolumn = bitpack
                    .read_bits(remaining_bits_length as usize)
                    .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
                let subcolumn = flip(subcolumn);
                if subcolumn == max_subcolumn {
                    next_to_check.insert(i + logical_offset);
                } else if subcolumn > max_subcolumn {
                    next_to_check.clear();
                    max_subcolumn = subcolumn;
                    next_to_check.insert(i + logical_offset);
                }
            }

            to_check = next_to_check;
            break;
        }

        // Process subcolumns(bytes)
        let subcolumn_chunk = bitpack
            .read_n_byte(number_of_records as usize)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        to_check = find_max_subcolumn(subcolumn_chunk, &to_check, logical_offset)?;

        remaining_bits_length -= 8;
    }

    let max_value_index = (to_check.iter().next().unwrap() - logical_offset) as usize;

    let mut remaining_bits_length = fixed_representation_bits_length;
    // Re-initialize the bitpack
    let mut bitpack = BitPack::new(&bytes[4 * 5..]);
    let mut delta_fixed_value = 0;
    while remaining_bits_length > 0 {
        if remaining_bits_length < 8 {
            let n_bits_to_skip = max_value_index * remaining_bits_length as usize;
            bitpack
                .skip(n_bits_to_skip)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            let last_subcolumn = bitpack
                .read_bits(remaining_bits_length as usize)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            delta_fixed_value =
                (delta_fixed_value << remaining_bits_length) | last_subcolumn as u64;
            break;
        }

        bitpack
            .skip_n_byte(max_value_index)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        let subcolumn = bitpack
            .read_byte()
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        let subcolumn = flip(subcolumn);
        bitpack
            .skip_n_byte(number_of_records as usize - max_value_index - 1)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;

        delta_fixed_value = (delta_fixed_value << 8) | subcolumn as u64;
        remaining_bits_length -= 8;
    }

    let fixed_value = base_fixed64 + delta_fixed_value as i64;
    let original_value = fixed_value as f64 / scale;

    Ok(original_value)
}

pub(super) fn max_without_bitmask(bytes: &[u8], logical_offset: u32) -> decoder::Result<f64> {
    let mut bitpack = BitPack::new(bytes);

    // Read header values
    let lower = bitpack.read_u32().unwrap();
    let higher = bitpack.read_u32().unwrap();
    let base_fixed64_bits = (lower as u64) | ((higher as u64) << 32);
    let base_fixed64 = unsafe { mem::transmute::<u64, i64>(base_fixed64_bits) };

    let number_of_records = bitpack.read_u32().unwrap();

    let fixed_representation_bits_length = bitpack.read_u32().unwrap();

    let fractional_part_bits_length = bitpack.read_u32().unwrap();

    let scale: f64 = 2.0f64.powi(fractional_part_bits_length as i32);

    let mut remaining_bits_length = fixed_representation_bits_length;

    // First subcolumn

    // Find the indices of the maximum values in subcolumn_chunks
    let mut max_subcolumn = u8::MIN;
    let mut to_check = RoaringBitmap::new();

    // All values are the same
    if remaining_bits_length == 0 {
        return Ok(base_fixed64 as f64 / scale);
    }

    if remaining_bits_length >= 8 {
        let subcolumn_chunks = bitpack
            .read_n_byte(number_of_records as usize)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        for (i, &subcolumn) in subcolumn_chunks.iter().enumerate() {
            let subcolumn = flip(subcolumn);
            if subcolumn > max_subcolumn {
                max_subcolumn = subcolumn;
                to_check.clear();
                to_check.push(i as u32 + logical_offset);
            } else if subcolumn == max_subcolumn {
                to_check.push(i as u32 + logical_offset);
            }
        }
    } else {
        let mut max_subcolumn = u8::MIN;
        for i in 0..number_of_records {
            let subcolumn = bitpack
                .read_bits(remaining_bits_length as usize)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            if subcolumn == max_subcolumn {
                to_check.insert(i + logical_offset);
            } else if subcolumn > max_subcolumn {
                to_check.clear();
                max_subcolumn = subcolumn;
                to_check.insert(i + logical_offset);
            }
        }
    }

    remaining_bits_length = remaining_bits_length.saturating_sub(8);

    // with bitmask
    while remaining_bits_length > 0 {
        let mut next_to_check = RoaringBitmap::new();
        // If the remaining bits are less than 8, it will be the last subcolumn
        // we treat it separately
        if remaining_bits_length < 8 {
            let mut expected_record_index_if_sequential = 0;
            let mut max_subcolumn = u8::MIN;
            for i in to_check.iter().map(|x| x - logical_offset) {
                let n_records_to_skip = i - expected_record_index_if_sequential;
                expected_record_index_if_sequential = i + 1;
                let n_bits_to_skip = n_records_to_skip * remaining_bits_length;
                bitpack
                    .skip(n_bits_to_skip as usize)
                    .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
                let subcolumn = bitpack
                    .read_bits(remaining_bits_length as usize)
                    .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
                if subcolumn == max_subcolumn {
                    next_to_check.insert(i + logical_offset);
                } else if subcolumn > max_subcolumn {
                    next_to_check.clear();
                    max_subcolumn = subcolumn;
                    next_to_check.insert(i + logical_offset);
                }
            }

            to_check = next_to_check;
            break;
        }

        // Process subcolumns(bytes)
        let subcolumn_chunk = bitpack
            .read_n_byte(number_of_records as usize)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        to_check = find_max_subcolumn(subcolumn_chunk, &to_check, logical_offset)?;

        remaining_bits_length -= 8;
    }

    let max_value_index = (to_check.iter().next().unwrap() - logical_offset) as usize;

    let mut remaining_bits_length = fixed_representation_bits_length;
    // Re-initialize the bitpack
    let mut bitpack = BitPack::new(&bytes[4 * 5..]);
    let mut delta_fixed_value = 0;
    while remaining_bits_length > 0 {
        if remaining_bits_length < 8 {
            let n_bits_to_skip = max_value_index * remaining_bits_length as usize;
            bitpack
                .skip(n_bits_to_skip)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            let last_subcolumn = bitpack
                .read_bits(remaining_bits_length as usize)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            delta_fixed_value =
                (delta_fixed_value << remaining_bits_length) | last_subcolumn as u64;
            break;
        }

        bitpack
            .skip_n_byte(max_value_index)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        let subcolumn = bitpack
            .read_byte()
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        let subcolumn = flip(subcolumn);
        bitpack
            .skip_n_byte(number_of_records as usize - max_value_index - 1)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;

        delta_fixed_value = (delta_fixed_value << 8) | subcolumn as u64;
        remaining_bits_length -= 8;
    }

    let fixed_value = base_fixed64 + delta_fixed_value as i64;
    let original_value = fixed_value as f64 / scale;

    Ok(original_value)
}
