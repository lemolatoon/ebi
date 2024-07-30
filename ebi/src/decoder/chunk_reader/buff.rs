use core::slice;
use std::{
    io::{self, Read},
    iter,
};

use either::Either;

use crate::decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFReader<R: Read> {
    reader: R,
    chunk_size: u64,
    decompressed: Option<Vec<f64>>,
}

impl<R: Read> BUFFReader<R> {
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, reader: R) -> Self {
        let chunk_size = handle.chunk_size();
        Self {
            reader,
            chunk_size,
            decompressed: None,
        }
    }
}

type F = fn(&f64) -> io::Result<f64>;
pub type BUFFIterator<'a> = Either<iter::Map<slice::Iter<'a, f64>, F>, iter::Once<io::Result<f64>>>;

impl<R: Read> Reader for BUFFReader<R> {
    type NativeHeader = ();

    type DecompressIterator<'a> = BUFFIterator<'a>
    where
        Self: 'a;

    fn decompress(&mut self) -> io::Result<&[f64]> {
        if self.decompressed.is_some() {
            return Ok(self.decompressed.as_ref().unwrap());
        }

        let mut buf = vec![0; self.chunk_size as usize];
        self.reader.read_exact(&mut buf)?;

        let result = internal::buff_simd256_decode(buf);

        self.decompressed = Some(result);

        Ok(self.decompressed.as_ref().unwrap())
    }

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        let decompressed = self.decompress();

        match decompressed {
            Ok(decompressed) => Either::Left(decompressed.iter().map(|f| Ok(*f))),
            Err(e) => Either::Right(iter::once(Err(e))),
        }
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed = Some(data);
        self.decompressed.as_ref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.decompressed.as_deref()
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
    }
}

// TODO: implement in-situ query execution
impl<R: Read> QueryExecutor for BUFFReader<R> {}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{bit_packing::BitPack, flip};

    pub fn buff_simd256_decode(bytes: Vec<u8>) -> Vec<f64> {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let base_fixed64_bits = (lower as u64) | ((higher as u64) << 32);
        let base_fixed64 = unsafe { mem::transmute::<u64, i64>(base_fixed64_bits) };

        let number_of_records = bitpack.read(32).unwrap();

        let fixed_representation_bits_length = bitpack.read(32).unwrap();

        let fractional_part_bits_length = bitpack.read(32).unwrap();

        let scale: f64 = 2.0f64.powi(fractional_part_bits_length as i32);

        let mut remaining_bits_length = fixed_representation_bits_length;

        let expected_datapoints: Vec<f64> = if remaining_bits_length < 8 {
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
}
