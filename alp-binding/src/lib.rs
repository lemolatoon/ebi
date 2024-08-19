#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::missing_safety_doc)]
use autocxx::prelude::*; // use all the main autocxx functions

include_cpp! {
    #include "compressor_specialized.hpp"

    safety!(unsafe_ffi)

    generate!("alp::AlpCompressorDouble")
    generate!("alp::AlpDecompressorDouble")
}

pub use ffi::alp::AlpCompressorDouble;
pub use ffi::alp::AlpDecompressorDouble;

/// exp_p_t: exception position type
pub type ExpPT = u16;
/// exp_c_t: exception center type
pub type ExpCT = u16;
pub const VECTOR_SIZE: usize = 1024;

pub fn worst_case_compression_size_double(n_tuple: usize) -> usize {
    let n_vector = n_tuple.next_multiple_of(VECTOR_SIZE) / VECTOR_SIZE;
    // extra space for exception position and center
    let if_all_exception = n_vector * VECTOR_SIZE * size_of::<f64>()
        + size_of::<ExpPT>()
        + n_vector * size_of::<ExpCT>();
    n_tuple * size_of::<f64>() + if_all_exception
}

/// Compresses a slice of `f64` values and stores the result in the provided buffer.
///
/// # Arguments
///
/// * `input` - A slice of `f64` values to be compressed.
/// * `buffer` - A mutable reference to a `Vec<u8>` where the compressed data will be stored.
///
pub fn compress_double(input: &[f64], buffer: &mut Vec<u8>) {
    moveit! {
        let mut compressor = AlpCompressorDouble::new();
    }
    let buf_size = worst_case_compression_size_double(input.len());
    buffer.clear();
    buffer.reserve(buf_size);
    unsafe {
        compressor
            .as_mut()
            .compress(input.as_ptr() as *mut f64, input.len(), buffer.as_mut_ptr());
    }
    let compressed_size = compressor.as_mut().get_size();
    debug_assert!(compressed_size <= buf_size);
    unsafe {
        buffer.set_len(compressed_size);
    }
}

/// Decompresses a slice of `u8` values into a buffer of `f64` values.
///
/// # Arguments
///
/// * `input` - A slice of `u8` values representing the compressed data.
/// * `number_of_records` - The number of `f64` values expected after decompression.
/// * `buffer` - A mutable reference to a `Vec<f64>` where the decompressed data will be stored.
///
/// # Safety
/// * The number of records must be valid for the given compressed data.
pub unsafe fn decompress_double(input: &[u8], number_of_records: usize, buffer: &mut Vec<f64>) {
    moveit! {
        let mut decompressor = AlpDecompressorDouble::new();
    }
    let buf_size = number_of_records.next_multiple_of(VECTOR_SIZE);
    buffer.clear();
    buffer.reserve(buf_size);
    unsafe {
        decompressor.as_mut().decompress(
            input.as_ptr() as *mut u8,
            number_of_records,
            buffer.as_mut_ptr(),
        );
    }
    unsafe {
        buffer.set_len(number_of_records);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_double() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut buffer = Vec::new();
        compress_double(&input, &mut buffer);
    }

    #[test]
    fn test_decompress_double() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut buffer = Vec::new();
        compress_double(&input, &mut buffer);
        let mut decompressed = Vec::new();
        unsafe { decompress_double(&buffer, input.len(), &mut decompressed) };
        assert_eq!(input, decompressed);
    }
}
