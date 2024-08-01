use core::slice;
use std::{
    io::{self, Read},
    iter,
};

use either::Either;

use crate::decoder::{
    query::{Predicate, QueryExecutor, Range, RangeValue},
    FileMetadataLike, GeneralChunkHandle,
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFReader<R: Read> {
    reader: R,
    chunk_size: u64,
    number_of_records: u64,
    bytes: Option<Vec<u8>>,
    decompressed: Option<Vec<f64>>,
}

impl<R: Read> BUFFReader<R> {
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, reader: R) -> Self {
        let chunk_size = handle.chunk_size();
        let number_of_records = handle.number_of_records();
        Self {
            reader,
            chunk_size,
            number_of_records,
            bytes: None,
            decompressed: None,
        }
    }

    fn bytes(&mut self) -> io::Result<&[u8]> {
        if self.bytes.is_none() {
            let mut buf = vec![0; self.chunk_size as usize];
            self.reader.read_exact(&mut buf)?;
            self.bytes = Some(buf);
        }
        Ok(self.bytes.as_deref().unwrap())
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

        let bytes = self.bytes()?;
        let result = internal::buff_simd256_decode(bytes);

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
impl<R: Read> QueryExecutor for BUFFReader<R> {
    fn filter(
        &mut self,
        predicate: Predicate,
        bitmask: Option<&roaring::RoaringBitmap>,
        logical_offset: usize,
    ) -> crate::decoder::Result<roaring::RoaringBitmap> {
        use internal::query::{buff_simd256_cmp_filter, buff_simd256_equal_filter};
        let logical_offset = logical_offset as u32;
        let number_of_records = self.number_of_records as u32;
        let bitmask: Option<roaring::RoaringBitmap> = bitmask.map(|x| {
            x.iter()
                .filter(|&x| x >= logical_offset && x < logical_offset + number_of_records)
                .collect()
        });
        let bytes = self.bytes()?;
        let all = || {
            let mut all = roaring::RoaringBitmap::new();
            all.insert_range(logical_offset..(number_of_records + logical_offset));
            all
        };
        Ok(match predicate {
            p @ (Predicate::Eq(pred) | Predicate::Ne(pred)) => {
                let result = internal::query::buff_simd256_equal_filter(
                    bytes,
                    pred,
                    bitmask.clone(),
                    logical_offset,
                );

                let result = if let Predicate::Eq(_) = p {
                    result
                } else {
                    // Predicate::Ne
                    #[allow(clippy::suspicious_operation_groupings)]
                    if let Some(bitmask) = bitmask {
                        bitmask - result
                    } else {
                        all() - result
                    }
                };

                result
            }
            Predicate::Range(r) => {
                let left = match r.start() {
                    RangeValue::Inclusive(pred) => Some(buff_simd256_cmp_filter::<true, true>(
                        bytes,
                        pred,
                        bitmask.clone(),
                        logical_offset,
                    )),
                    RangeValue::Exclusive(pred) => Some(buff_simd256_cmp_filter::<true, false>(
                        bytes,
                        pred,
                        bitmask.clone(),
                        logical_offset,
                    )),
                    RangeValue::None => None,
                };

                let right = match r.end() {
                    RangeValue::Inclusive(pred) => Some(buff_simd256_cmp_filter::<false, true>(
                        bytes,
                        pred,
                        bitmask,
                        logical_offset,
                    )),
                    RangeValue::Exclusive(pred) => Some(buff_simd256_cmp_filter::<false, false>(
                        bytes,
                        pred,
                        bitmask,
                        logical_offset,
                    )),
                    RangeValue::None => None,
                };

                match (left, right) {
                    (None, None) => all(),
                    (None, Some(right)) => right,
                    (Some(left), None) => left,
                    (Some(left), Some(right)) => left & right,
                }
            }
        })
    }
}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{bit_packing::BitPack, flip};

    pub(crate) mod query {
        use roaring::RoaringBitmap;
        use simd::{comp_simd, equal_simd};

        use crate::compression_common::buff::{
            bit_packing::BitPack, flip, precision_bound::PrecisionBound,
        };
        use cfg_if::cfg_if;

        const SIMD_THRESHOLD: f64 = 0.06;

        pub(crate) fn buff_simd256_cmp_filter<const IS_GREATER: bool, const INCLUSIVE: bool>(
            bytes: &[u8],
            pred: f64,
            bitmask: Option<RoaringBitmap>,
            logical_offset: u32,
        ) -> RoaringBitmap {
            // Read the header
            let mut bitpack = BitPack::<&[u8]>::new(bytes);

            let fixed_base64_lower = bitpack.read_u32().unwrap();
            let fixed_base64_higher = bitpack.read_u32().unwrap();
            let fixed_base64_bits =
                (fixed_base64_lower as u64) | ((fixed_base64_higher as u64) << 32);
            let fixed_base64 = unsafe { std::mem::transmute::<u64, i64>(fixed_base64_bits) };
            let number_of_records = bitpack.read_u32().unwrap();
            let fixed_representation_bits_length = bitpack.read_u32().unwrap();
            let fractional_part_bits_length = bitpack.read_u32().unwrap();

            let fixed_pred = PrecisionBound::into_fixed_representation_with_decimal_length(
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
                return if IS_GREATER { all() } else { none() };
            }
            if fixed_representation_bits_length == 0 {
                let all_if_or_none = |p: bool| if p { all() } else { none() };
                return match (IS_GREATER, INCLUSIVE) {
                    (true, true) => all_if_or_none(fixed_base64 >= fixed_pred),
                    (true, false) => all_if_or_none(fixed_base64 > fixed_pred),
                    (false, true) => all_if_or_none(fixed_base64 <= fixed_pred),
                    (false, false) => all_if_or_none(fixed_base64 < fixed_pred),
                };
            }

            let delta_fixed_pred = (fixed_pred - fixed_base64) as u64;
            {
                let pred_fixed_representation_bits_length = if delta_fixed_pred == 0 {
                    0
                } else {
                    64 - delta_fixed_pred.leading_zeros()
                };
                if fixed_representation_bits_length < pred_fixed_representation_bits_length {
                    return if IS_GREATER { none() } else { all() };
                }
            }

            let mut remaining_bits_length = fixed_representation_bits_length;

            let comp_simd_specialized = if IS_GREATER {
                comp_simd::<{ simd::GREATER }>
            } else {
                comp_simd::<{ simd::LESS }>
            };

            let mut qualified_bitmask = RoaringBitmap::new();
            let mut to_check_bitmask: Option<RoaringBitmap> = bitmask;
            while remaining_bits_length >= 8 {
                remaining_bits_length -= 8;

                let delta_fixed_pred_subcolumn = (delta_fixed_pred >> remaining_bits_length) as u8;
                let next_to_check_bitmask = if let Some(to_check_bitmask) = &to_check_bitmask {
                    let use_simd =
                        to_check_bitmask.len() as f64 / number_of_records as f64 > SIMD_THRESHOLD;
                    let (mut next_to_check_bitmask, mut expected_record_index_if_sequential) =
                        if use_simd {
                            let simd_chunk = bitpack
                                .read_n_byte(
                                    number_of_records as usize
                                        - number_of_records as usize % simd::VECTOR_SIZE,
                                )
                                .unwrap();
                            let mut temporary_qualified_bitmask = RoaringBitmap::new();
                            let mut next_to_check_bitmask = unsafe {
                                comp_simd_specialized(
                                    simd_chunk,
                                    delta_fixed_pred_subcolumn,
                                    &mut temporary_qualified_bitmask,
                                    logical_offset,
                                )
                            };
                            temporary_qualified_bitmask &= to_check_bitmask;
                            qualified_bitmask |= temporary_qualified_bitmask;

                            next_to_check_bitmask &= to_check_bitmask;

                            (
                                next_to_check_bitmask,
                                number_of_records - number_of_records % simd::VECTOR_SIZE as u32,
                            )
                        } else {
                            (RoaringBitmap::new(), 0)
                        };

                    for record_index in to_check_bitmask
                        .iter()
                        .map(|x| x - logical_offset)
                        .filter(move |&x| x >= expected_record_index_if_sequential)
                    {
                        bitpack
                            .skip_n_byte(
                                (record_index - expected_record_index_if_sequential) as usize,
                            )
                            .unwrap();
                        let subcolumn = bitpack.read_byte().unwrap();
                        let subcolumn = flip(subcolumn);
                        match subcolumn.cmp(&delta_fixed_pred_subcolumn) {
                            std::cmp::Ordering::Greater => {
                                if IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                            std::cmp::Ordering::Equal => {
                                next_to_check_bitmask.insert(logical_offset + record_index);
                            }
                            std::cmp::Ordering::Less => {
                                if !IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                        }

                        expected_record_index_if_sequential = record_index + 1;
                    }
                    bitpack
                        .skip_n_byte(
                            number_of_records as usize
                                - expected_record_index_if_sequential as usize,
                        )
                        .unwrap();

                    next_to_check_bitmask
                } else {
                    // TODO: use SIMD here
                    let chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();
                    cfg_if! {
                        if #[cfg(miri)] {
                            let mut to_check_bitmask = RoaringBitmap::new();
                            let remaining_chunk = chunk;
                            let record_index_offset = 0;
                        } else {
                            let (simd_chunk, scalar_chunk) =
                                chunk.split_at(chunk.len() - (chunk.len() % simd::VECTOR_SIZE));
                            let mut to_check_bitmask = unsafe {
                                comp_simd_specialized(
                                    simd_chunk,
                                    delta_fixed_pred_subcolumn,
                                    &mut qualified_bitmask,
                                    logical_offset,
                                )
                            };
                            let remaining_chunk = scalar_chunk;
                            let record_index_offset = number_of_records - number_of_records % simd::VECTOR_SIZE as u32;
                        }
                    };
                    for (record_index, subcolumn) in remaining_chunk
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| (i as u32 + record_index_offset, x))
                    {
                        let subcolumn = flip(subcolumn);
                        match subcolumn.cmp(&delta_fixed_pred_subcolumn) {
                            std::cmp::Ordering::Greater => {
                                if IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                            std::cmp::Ordering::Equal => {
                                to_check_bitmask.insert(logical_offset + record_index);
                            }
                            std::cmp::Ordering::Less => {
                                if !IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                        }
                    }

                    to_check_bitmask
                };

                if next_to_check_bitmask.is_empty() {
                    return qualified_bitmask;
                }

                to_check_bitmask.replace(next_to_check_bitmask);
            }

            // last subcolumn
            if remaining_bits_length > 0 {
                let mask_for_remaining_bits = (1 << remaining_bits_length) - 1;
                let delta_fixed_pred_subcolumn =
                    ((delta_fixed_pred) & mask_for_remaining_bits) as u8;
                if let Some(bitmask) = &to_check_bitmask {
                    let mut expected_record_index_if_sequential = 0;
                    for record_index in bitmask.iter().map(|x| x - logical_offset) {
                        let n_bits_skipped = (record_index - expected_record_index_if_sequential)
                            * remaining_bits_length;
                        bitpack.skip(n_bits_skipped as usize).unwrap();

                        let subcolumn = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                        // no flip for the last bits(less than byte) column
                        match subcolumn.cmp(&delta_fixed_pred_subcolumn) {
                            std::cmp::Ordering::Greater => {
                                if IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                            std::cmp::Ordering::Equal => {
                                if INCLUSIVE {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                            std::cmp::Ordering::Less => {
                                if !IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + record_index);
                                }
                            }
                        }

                        expected_record_index_if_sequential = record_index + 1;
                    }
                    let n_bits_skipped = (number_of_records - expected_record_index_if_sequential)
                        * remaining_bits_length;
                    bitpack.skip(n_bits_skipped as usize).unwrap();
                } else {
                    for i in 0..number_of_records {
                        let subcolumn = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                        match subcolumn.cmp(&delta_fixed_pred_subcolumn) {
                            std::cmp::Ordering::Greater => {
                                if IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + i);
                                }
                            }
                            std::cmp::Ordering::Equal => {
                                if INCLUSIVE {
                                    qualified_bitmask.insert(logical_offset + i);
                                }
                            }
                            std::cmp::Ordering::Less => {
                                if !IS_GREATER {
                                    qualified_bitmask.insert(logical_offset + i);
                                }
                            }
                        }
                    }
                }
            } else if INCLUSIVE {
                if let Some(to_check_bitmask) = to_check_bitmask {
                    qualified_bitmask |= to_check_bitmask
                }
            }
            return qualified_bitmask;
        }

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
            let fixed_base64_bits =
                (fixed_base64_lower as u64) | ((fixed_base64_higher as u64) << 32);
            let fixed_base64 = unsafe { std::mem::transmute::<u64, i64>(fixed_base64_bits) };
            let number_of_records = bitpack.read_u32().unwrap();
            let fixed_representation_bits_length = bitpack.read_u32().unwrap();
            let fractional_part_bits_length = bitpack.read_u32().unwrap();

            let fixed_pred = PrecisionBound::into_fixed_representation_with_decimal_length(
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
                            .skip_n_byte(
                                (record_index - expected_record_index_if_sequential) as usize,
                            )
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
                            number_of_records as usize
                                - expected_record_index_if_sequential as usize,
                        )
                        .unwrap();

                    next_result_bitmask
                } else {
                    let chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();

                    cfg_if! {
                        if #[cfg(miri)] {
                            let mut bitmask = RoaringBitmap::new();
                            let remaining_chunk = chunk;
                            let record_index_offset = 0;
                        } else {
                            let (simd_chunk, scalar_chunk) =
                                chunk.split_at(chunk.len() - (chunk.len() % simd::VECTOR_SIZE));
                            let mut bitmask = unsafe {
                                equal_simd(simd_chunk, delta_fixed_pred_subcolumn, logical_offset)
                            };

                            let remaining_chunk = scalar_chunk;
                            let record_index_offset = chunk.len() - chunk.len() % simd::VECTOR_SIZE;
                        }
                    }
                    for (record_index, subcolumn) in remaining_chunk
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| (i + record_index_offset, x))
                    {
                        let subcolumn = flip(subcolumn);
                        if subcolumn == delta_fixed_pred_subcolumn {
                            bitmask.insert(logical_offset + record_index as u32);
                        }
                    }

                    bitmask
                };

                if next_result_bitmask.is_empty() {
                    return next_result_bitmask;
                }

                result_bitmask.replace(next_result_bitmask);
            }

            // last subcolumn
            let r = if remaining_bits_length > 0 {
                if let Some(bitmask) = &result_bitmask {
                    let mask_for_remaining_bits = (1 << remaining_bits_length) - 1;
                    let delta_fixed_pred_subcolumn =
                        ((delta_fixed_pred) & mask_for_remaining_bits) as u8;
                    let mut next_result_bitmask = RoaringBitmap::new();
                    let mut expected_record_index_if_sequential = 0;
                    for record_index in bitmask.iter().map(|x| x - logical_offset) {
                        let n_bits_skipped = (record_index - expected_record_index_if_sequential)
                            * remaining_bits_length;
                        bitpack.skip(n_bits_skipped as usize).unwrap();

                        let subcolumn = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                        // no flip for the last bits(less than byte) column
                        if subcolumn == delta_fixed_pred_subcolumn {
                            next_result_bitmask.insert(logical_offset + record_index);
                        }

                        expected_record_index_if_sequential = record_index + 1;
                    }
                    let n_bits_skipped = (number_of_records - expected_record_index_if_sequential)
                        * remaining_bits_length;
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
            };
            return r;
        }

        mod simd {
            use std::arch::x86_64::{
                __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_lddqu_si256,
                _mm256_movemask_epi8, _mm256_set1_epi8, _popcnt32,
            };
            pub(super) const VECTOR_SIZE: usize = size_of::<__m256i>();

            use roaring::RoaringBitmap;

            use crate::compression_common::buff::flip;

            pub const GREATER: u8 = 1;
            pub const LESS: u8 = 1 << 1;
            pub unsafe fn comp_simd<const FLAGS: u8>(
                x: &[u8],
                pred: u8,
                qualified_bitmask: &mut RoaringBitmap,
                logical_offset: u32,
            ) -> RoaringBitmap {
                debug_assert!(x.len() % VECTOR_SIZE == 0);
                const {
                    assert!(FLAGS & (GREATER | LESS) != 0);
                };
                let pred_word = _mm256_set1_epi8(std::mem::transmute::<u8, i8>(flip(pred)));
                let mut to_check_bitmask = RoaringBitmap::new();
                for (word_index, word) in x
                    .chunks_exact(VECTOR_SIZE)
                    .map(|chunk| _mm256_lddqu_si256(chunk.as_ptr().cast::<__m256i>()))
                    .enumerate()
                {
                    let equal = _mm256_cmpeq_epi8(word, pred_word);
                    let eq_mask = _mm256_movemask_epi8(equal);

                    for i in 0..VECTOR_SIZE {
                        if (eq_mask & (1 << i)) != 0 {
                            to_check_bitmask
                                .insert((word_index * VECTOR_SIZE + i) as u32 + logical_offset);
                        }
                    }

                    let cmp_mask = if FLAGS & GREATER != 0 {
                        let greater = _mm256_cmpgt_epi8(word, pred_word);
                        _mm256_movemask_epi8(greater)
                    } else {
                        let less_or_eq = _mm256_cmpgt_epi8(pred_word, word);
                        let less_or_eq_mask = _mm256_movemask_epi8(less_or_eq);
                        less_or_eq_mask & (!eq_mask)
                    };

                    for i in 0..VECTOR_SIZE {
                        if (cmp_mask & (1 << i)) != 0 {
                            qualified_bitmask
                                .insert((word_index * VECTOR_SIZE + i) as u32 + logical_offset);
                        }
                    }
                }

                to_check_bitmask
            }

            pub unsafe fn equal_simd(x: &[u8], pred: u8, logical_offset: u32) -> RoaringBitmap {
                debug_assert_eq!(
                    x.len() % VECTOR_SIZE,
                    0,
                    "simd chunk size is not a multiple of 32"
                );
                let pred_word = _mm256_set1_epi8(std::mem::transmute::<u8, i8>(flip(pred)));
                let mut result = RoaringBitmap::new();
                for (word_index, word) in x
                    .chunks_exact(VECTOR_SIZE)
                    .map(|chunk| _mm256_lddqu_si256(chunk.as_ptr().cast::<__m256i>()))
                    .enumerate()
                {
                    let equal = _mm256_cmpeq_epi8(word, pred_word);
                    let eq_mask = _mm256_movemask_epi8(equal);

                    for i in 0..VECTOR_SIZE {
                        if (eq_mask & (1 << i)) != 0 {
                            result.insert((word_index * VECTOR_SIZE + i) as u32 + logical_offset);
                        }
                    }
                }

                result
            }
        }
    }

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
}
