use std::mem;

use crate::{
    compression_common::buff::{bit_packing::BitPack, flip},
    decoder::{self, error::DecoderError},
};

pub(super) fn sum_byte_subcolumn(
    chunk: &[u8],
    bitmask: Option<&roaring::RoaringBitmap>,
    logical_offset: u32,
) -> u64 {
    if let Some(bitmask) = bitmask {
        let mut sum = 0;
        for i in bitmask.iter().map(|x| x - logical_offset) {
            let subcolumn = chunk[i as usize];
            let subcolumn = flip(subcolumn);
            sum += subcolumn as u64;
        }
        sum
    } else {
        chunk.iter().map(|x| flip(*x) as u64).sum()
    }
}

pub(super) fn sum_bits_subcolumn(
    number_of_records: u32,
    bit_reader: &mut BitPack<&[u8]>,
    bitmask: Option<&roaring::RoaringBitmap>,
    logical_offset: u32,
    bits_length: usize,
) -> decoder::Result<u64> {
    if let Some(bitmask) = bitmask {
        let mut sum = 0;
        let mut expected_record_index_if_sequential = 0;
        for i in bitmask.iter().map(|x| x - logical_offset) {
            let n_bits_to_skip = (i - expected_record_index_if_sequential) as usize * bits_length;
            expected_record_index_if_sequential = i + 1;
            bit_reader
                .skip(n_bits_to_skip)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            let subcolumn = bit_reader
                .read_bits(bits_length)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            sum += subcolumn as u64;
        }
        Ok(sum)
    } else {
        let mut sum = 0;
        for _ in 0..number_of_records {
            let subcolumn = bit_reader
                .read_bits(bits_length)
                .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
            sum += subcolumn as u64;
        }
        Ok(sum)
    }
}

pub(super) fn sum_with_bitmask(
    bytes: &[u8],
    bitmask: Option<&roaring::RoaringBitmap>,
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

    // All values are the same
    if fixed_representation_bits_length == 0 {
        let fp = base_fixed64 as f64 / scale;
        return Ok(fp * number_of_records as f64);
    }

    let mut sum = 0;
    let n_byte_subcolumns = fixed_representation_bits_length / 8;
    let bits_subcolumn_length = fixed_representation_bits_length % 8;

    for i in (0..n_byte_subcolumns).rev() {
        let chunk = bitpack
            .read_n_byte(number_of_records as usize)
            .map_err(|_| DecoderError::UnexpectedEndOfChunk)?;
        let subcolumn_sum = sum_byte_subcolumn(chunk, bitmask, logical_offset);
        let subcolumn_sum_scaled = subcolumn_sum << ((i * 8 + bits_subcolumn_length) as u64);
        sum += subcolumn_sum_scaled;
    }
    if bits_subcolumn_length != 0 {
        let subcolumn_sum = sum_bits_subcolumn(
            number_of_records,
            &mut bitpack,
            bitmask,
            logical_offset,
            bits_subcolumn_length as usize,
        )?;
        // sum += subcolumn_sum as f64 / scale;
        // sum <<= bits_subcolumn_length;
        sum += subcolumn_sum;
    }

    let accumulated_len = if let Some(bitmask) = bitmask {
        bitmask.len() as i64
    } else {
        number_of_records as i64
    };
    let sum = (sum as i64 + base_fixed64 * accumulated_len) as f64 / scale;

    Ok(sum)
}
