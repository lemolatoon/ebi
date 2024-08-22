use core::slice;
use std::{io::Read, iter};

use crate::decoder::{
    self,
    query::{default_materialize, Predicate, QueryExecutor, RangeValue},
    FileMetadataLike, GeneralChunkHandle,
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFReader {
    chunk_size: u64,
    number_of_records: u64,
    bytes: Vec<u8>,
    decompressed: Option<Vec<f64>>,
}

impl BUFFReader {
    pub fn new<T: FileMetadataLike, R: Read>(
        handle: &GeneralChunkHandle<T>,
        mut reader: R,
    ) -> Self {
        let chunk_size = handle.chunk_size();
        let mut chunk_in_memory = vec![0; chunk_size as usize];
        reader.read_exact(&mut chunk_in_memory).unwrap();
        let number_of_records = handle.number_of_records();
        Self {
            chunk_size,
            number_of_records,
            bytes: chunk_in_memory,
            decompressed: None,
        }
    }

    fn bytes(&mut self) -> &[u8] {
        &self.bytes
    }
}

type F = fn(&f64) -> decoder::Result<f64>;
pub type BUFFIterator<'a> = iter::Map<slice::Iter<'a, f64>, F>;

impl Reader for BUFFReader {
    type NativeHeader = ();

    type DecompressIterator<'a> = BUFFIterator<'a>
    where
        Self: 'a;

    fn decompress(&mut self) -> decoder::Result<&[f64]> {
        if self.decompressed.is_some() {
            return Ok(self.decompressed.as_ref().unwrap());
        }

        let bytes = self.bytes();
        let result = internal::buff_simd256_decode(bytes);

        self.decompressed = Some(result);

        Ok(self.decompressed.as_ref().unwrap())
    }

    fn decompress_iter(&mut self) -> decoder::Result<Self::DecompressIterator<'_>> {
        let decompressed = self.decompress()?;

        Ok(decompressed.iter().map(|f| Ok(*f)))
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
impl QueryExecutor for BUFFReader {
    fn filter(
        &mut self,
        predicate: Predicate,
        bitmask: Option<&roaring::RoaringBitmap>,
        logical_offset: usize,
    ) -> crate::decoder::Result<roaring::RoaringBitmap> {
        use internal::query::buff_simd256_cmp_filter;
        let logical_offset = logical_offset as u32;
        let number_of_records = self.number_of_records as u32;
        let bitmask: Option<roaring::RoaringBitmap> = bitmask.map(|x| {
            x.iter()
                .filter(|&x| x >= logical_offset && x < logical_offset + number_of_records)
                .collect()
        });
        let bytes = self.bytes();
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

                if let Predicate::Eq(_) = p {
                    result
                } else {
                    // Predicate::Ne
                    #[allow(clippy::suspicious_operation_groupings)]
                    if let Some(bitmask) = bitmask {
                        bitmask - result
                    } else {
                        all() - result
                    }
                }
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
                        bitmask.clone(),
                        logical_offset,
                    )),
                    RangeValue::Exclusive(pred) => Some(buff_simd256_cmp_filter::<false, false>(
                        bytes,
                        pred,
                        bitmask.clone(),
                        logical_offset,
                    )),
                    RangeValue::None => None,
                };

                match (left, right) {
                    (None, None) => {
                        if let Some(bitmask) = bitmask {
                            bitmask
                        } else {
                            all()
                        }
                    }
                    (None, Some(right)) => right,
                    (Some(left), None) => left,
                    (Some(left), Some(right)) => left & right,
                }
            }
        })
    }

    fn filter_materialize<W: std::io::Write>(
        &mut self,
        output: &mut W,
        predicate: Predicate,
        bitmask: Option<&roaring::RoaringBitmap>,
        logical_offset: usize,
    ) -> decoder::Result<()> {
        let filtered = self.filter(predicate, bitmask, logical_offset)?;

        default_materialize(self, output, Some(&filtered), logical_offset)
    }
}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{bit_packing::BitPack, flip};

    pub(crate) mod query {

        use roaring::RoaringBitmap;
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        use simd_x86_64::{equal_simd, greater_simd, less_simd};

        use crate::compression_common::buff::{
            bit_packing::BitPack,
            flip,
            precision_bound::{self},
        };
        use cfg_if::cfg_if;

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        const SIMD_THRESHOLD: f64 = 0.06;

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        #[inline]
        fn split_for_simd_processing(chunk: &[u8], logical_offset: u32) -> (&[u8], &[u8], &[u8]) {
            let offset = logical_offset as usize;
            let total_len = chunk.len();

            // 1. Pre-processing: The portion before the offset is aligned to 8 bytes.
            let align_offset = (offset + 7) & !7;
            let initial_len = align_offset.saturating_sub(offset);
            let initial_part = &chunk[..initial_len.min(total_len)];

            // 2. SIMD processing: The portion that is processed in 32-byte units.
            let simd_start = initial_len;
            let simd_len = if simd_start < total_len {
                ((total_len - simd_start) / 32) * 32
            } else {
                0
            };
            let simd_part = if simd_start < total_len {
                &chunk[simd_start..simd_start + simd_len]
            } else {
                &[]
            };

            // 3. Post-processing: The remaining portion.
            let remaining_start = simd_start + simd_len;
            let remaining_part = if remaining_start < total_len {
                &chunk[remaining_start..]
            } else {
                &[]
            };

            (initial_part, simd_part, remaining_part)
        }

        /// Modify `to_check_bitmask` and `qualified_bitmask`
        #[inline]
        fn buff_simd256_cmp_filter_subcolumn_without_bitmask_nonsimd<
            const IS_GREATER: bool,
            const INCLUSIVE: bool,
        >(
            chunk: &[u8],
            logical_offset: u32,
            delta_fixed_pred_subcolumn: u8,
            to_check_bitmask: &mut RoaringBitmap,
            qualified_bitmask: &mut RoaringBitmap,
        ) {
            for (record_index, subcolumn) in chunk
                .iter()
                .enumerate()
                .map(|(i, &x)| (i as u32 + logical_offset, x))
            {
                let subcolumn = flip(subcolumn);
                match subcolumn.cmp(&delta_fixed_pred_subcolumn) {
                    std::cmp::Ordering::Greater => {
                        if IS_GREATER {
                            qualified_bitmask.insert(record_index);
                        }
                    }
                    std::cmp::Ordering::Equal => {
                        to_check_bitmask.insert(record_index);
                    }
                    std::cmp::Ordering::Less => {
                        if !IS_GREATER {
                            qualified_bitmask.insert(record_index);
                        }
                    }
                }
            }
        }

        /// Modify `to_check_bitmask` and `qualified_bitmask`
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        #[inline]
        fn buff_simd256_cmp_filter_subcolumn_without_bitmask_simd<
            const IS_GREATER: bool,
            const INCLUSIVE: bool,
        >(
            simd_chunk: &[u8],
            logical_offset: u32,
            delta_fixed_pred_subcolumn: u8,
            to_check_bitmask: &mut RoaringBitmap,
            qualified_bitmask: &mut RoaringBitmap,
        ) {
            if simd_chunk.is_empty() {
                return;
            }
            debug_assert_eq!(logical_offset % 8, 0);
            debug_assert_eq!(simd_chunk.len() % 32, 0);

            let to_check_bitmask_simd = unsafe {
                if IS_GREATER {
                    greater_simd(
                        simd_chunk,
                        delta_fixed_pred_subcolumn,
                        qualified_bitmask,
                        logical_offset,
                    )
                } else {
                    less_simd(
                        simd_chunk,
                        delta_fixed_pred_subcolumn,
                        qualified_bitmask,
                        logical_offset,
                    )
                }
            };
            *to_check_bitmask |= to_check_bitmask_simd;
        }

        /// Returns `next_to_check_bitmask`
        #[inline]
        fn buff_simd256_cmp_filter_subcolumn_without_bitmask<
            const IS_GREATER: bool,
            const INCLUSIVE: bool,
        >(
            chunk: &[u8],
            logical_offset: u32,
            delta_fixed_pred_subcolumn: u8,
            qualified_bitmask: &mut RoaringBitmap,
        ) -> RoaringBitmap {
            let mut to_check_bitmask = RoaringBitmap::new();
            cfg_if! {
                if #[cfg(all(target_arch = "x86_64", not(miri)))] {
                    // simd
                    let (initial_part, simd_part, remaining_part) =
                        split_for_simd_processing(chunk, logical_offset);
                    buff_simd256_cmp_filter_subcolumn_without_bitmask_nonsimd::<IS_GREATER, INCLUSIVE>(
                        initial_part,
                        logical_offset,
                        delta_fixed_pred_subcolumn,
                        &mut to_check_bitmask,
                        qualified_bitmask,
                    );
                    buff_simd256_cmp_filter_subcolumn_without_bitmask_simd::<IS_GREATER, INCLUSIVE>(
                        simd_part,
                        logical_offset + initial_part.len() as u32,
                        delta_fixed_pred_subcolumn,
                        &mut to_check_bitmask,
                        qualified_bitmask,
                    );
                    buff_simd256_cmp_filter_subcolumn_without_bitmask_nonsimd::<IS_GREATER, INCLUSIVE>(
                        remaining_part,
                        logical_offset + initial_part.len() as u32 + simd_part.len() as u32,
                        delta_fixed_pred_subcolumn,
                        &mut to_check_bitmask,
                        qualified_bitmask,
                    );
                } else {
                    // nonsimd
                    buff_simd256_cmp_filter_subcolumn_without_bitmask_nonsimd::<IS_GREATER, INCLUSIVE>(
                        chunk,
                        logical_offset,
                        delta_fixed_pred_subcolumn,
                        &mut to_check_bitmask,
                        qualified_bitmask,
                    );
                }
            }

            to_check_bitmask
        }

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        #[inline]
        fn buff_simd256_cmp_filter_subcolumn_with_bitmask_simd<
            const IS_GREATER: bool,
            const INCLUSIVE: bool,
        >(
            simd_chunk: &[u8],
            logical_offset: u32,
            delta_fixed_pred_subcolumn: u8,
            to_check_bitmask: &RoaringBitmap,
            qualified_bitmask: &mut RoaringBitmap,
        ) -> RoaringBitmap {
            if simd_chunk.is_empty() {
                return RoaringBitmap::new();
            }
            debug_assert_eq!(simd_chunk.len() % 32, 0);
            debug_assert_eq!(logical_offset % 8, 0);
            let mut next_to_check_bitmask = RoaringBitmap::new();
            let mut temporary_qualified_bitmask = RoaringBitmap::default();
            let next_to_check_bitmask_simd = unsafe {
                if IS_GREATER {
                    greater_simd(
                        simd_chunk,
                        delta_fixed_pred_subcolumn,
                        &mut temporary_qualified_bitmask,
                        logical_offset,
                    )
                } else {
                    less_simd(
                        simd_chunk,
                        delta_fixed_pred_subcolumn,
                        &mut temporary_qualified_bitmask,
                        logical_offset,
                    )
                }
            };
            next_to_check_bitmask |= next_to_check_bitmask_simd;
            temporary_qualified_bitmask &= to_check_bitmask;
            *qualified_bitmask |= temporary_qualified_bitmask;

            next_to_check_bitmask &= to_check_bitmask;

            next_to_check_bitmask
        }

        /// Modify `qualified_bitmask`, and returns `next_to_check_bitmask`
        #[inline]
        fn buff_simd256_cmp_filter_subcolumn_with_bitmask_nonsimd<
            const IS_GREATER: bool,
            const INCLUSIVE: bool,
        >(
            chunk: &[u8],
            logical_offset: u32,
            delta_fixed_pred_subcolumn: u8,
            to_check_bitmask_iter: impl Iterator<Item = u32>,
            qualified_bitmask: &mut RoaringBitmap,
        ) -> RoaringBitmap {
            let mut next_to_check_bitmask = RoaringBitmap::new();
            for record_index in to_check_bitmask_iter {
                let subcolumn = chunk[(record_index - logical_offset) as usize];
                let subcolumn = flip(subcolumn);
                match subcolumn.cmp(&delta_fixed_pred_subcolumn) {
                    std::cmp::Ordering::Greater => {
                        if IS_GREATER {
                            qualified_bitmask.insert(record_index);
                        }
                    }
                    std::cmp::Ordering::Equal => {
                        next_to_check_bitmask.insert(record_index);
                    }
                    std::cmp::Ordering::Less => {
                        if !IS_GREATER {
                            qualified_bitmask.insert(record_index);
                        }
                    }
                }
            }

            next_to_check_bitmask
        }

        // Returns `to_check_bitmask`
        #[inline]
        fn buff_simd256_cmp_filter_subcolumn_with_bitmask<
            const IS_GREATER: bool,
            const INCLUSIVE: bool,
        >(
            chunk: &[u8],
            #[allow(unused_variables)] number_of_records: u32,
            logical_offset: u32,
            delta_fixed_pred_subcolumn: u8,
            to_check_bitmask: &RoaringBitmap,
            qualified_bitmask: &mut RoaringBitmap,
        ) -> RoaringBitmap {
            cfg_if! {
                if #[cfg(all(target_arch = "x86_64", not(miri)))] {
                    let cardinality = to_check_bitmask.len() as f64 / number_of_records as f64;
                    let use_simd = cardinality > SIMD_THRESHOLD;
                    if use_simd {
                        let (initial_part, simd_part, remaining_part) =
                            split_for_simd_processing(chunk, logical_offset);
                        let next_to_check_bitmask0 = buff_simd256_cmp_filter_subcolumn_with_bitmask_nonsimd::<
                            IS_GREATER,
                            INCLUSIVE,
                        >(
                            initial_part,
                            logical_offset,
                            delta_fixed_pred_subcolumn,
                            to_check_bitmask
                                .iter()
                                .take_while(|&x| x < logical_offset + initial_part.len() as u32),
                            qualified_bitmask,
                        );
                        let next_to_check_bitmask1 =
                            buff_simd256_cmp_filter_subcolumn_with_bitmask_simd::<IS_GREATER, INCLUSIVE>(
                                simd_part,
                                logical_offset + initial_part.len() as u32,
                                delta_fixed_pred_subcolumn,
                                to_check_bitmask,
                                qualified_bitmask,
                            );
                        let remaining_logical_offset =
                            logical_offset + initial_part.len() as u32 + simd_part.len() as u32;
                        let next_to_check_bitmask2 = buff_simd256_cmp_filter_subcolumn_with_bitmask_nonsimd::<
                            IS_GREATER,
                            INCLUSIVE,
                        >(
                            remaining_part,
                            remaining_logical_offset,
                            delta_fixed_pred_subcolumn,
                            to_check_bitmask
                                .iter()
                                .skip_while(|&x| x < remaining_logical_offset),
                            qualified_bitmask,
                        );

                        next_to_check_bitmask0 | next_to_check_bitmask1 | next_to_check_bitmask2
                    } else {
                        buff_simd256_cmp_filter_subcolumn_with_bitmask_nonsimd::<IS_GREATER, INCLUSIVE>(
                            chunk,
                            logical_offset,
                            delta_fixed_pred_subcolumn,
                            to_check_bitmask.iter(),
                            qualified_bitmask,
                        )
                    }
                } else {
                    buff_simd256_cmp_filter_subcolumn_with_bitmask_nonsimd::<IS_GREATER, INCLUSIVE>(
                        chunk,
                        logical_offset,
                        delta_fixed_pred_subcolumn,
                        to_check_bitmask.iter(),
                        qualified_bitmask,
                    )
                }
            }
        }

        #[inline]
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

            let fixed_pred =
                precision_bound::into_fixed_representation_with_fractional_part_bits_length(
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

            let mut qualified_bitmask = RoaringBitmap::new();
            let mut to_check_bitmask: Option<RoaringBitmap> = bitmask;
            while remaining_bits_length >= 8 {
                remaining_bits_length -= 8;

                let delta_fixed_pred_subcolumn = (delta_fixed_pred >> remaining_bits_length) as u8;
                let chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();
                let next_to_check_bitmask = if let Some(to_check_bitmask) = &to_check_bitmask {
                    buff_simd256_cmp_filter_subcolumn_with_bitmask::<IS_GREATER, INCLUSIVE>(
                        chunk,
                        number_of_records,
                        logical_offset,
                        delta_fixed_pred_subcolumn,
                        to_check_bitmask,
                        &mut qualified_bitmask,
                    )
                } else {
                    buff_simd256_cmp_filter_subcolumn_without_bitmask::<IS_GREATER, INCLUSIVE>(
                        chunk,
                        logical_offset,
                        delta_fixed_pred_subcolumn,
                        &mut qualified_bitmask,
                    )
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
            qualified_bitmask
        }

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
                        split_for_simd_processing(chunk, logical_offset);
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
            let fixed_base64_bits =
                (fixed_base64_lower as u64) | ((fixed_base64_higher as u64) << 32);
            let fixed_base64 = unsafe { std::mem::transmute::<u64, i64>(fixed_base64_bits) };
            let number_of_records = bitpack.read_u32().unwrap();
            let fixed_representation_bits_length = bitpack.read_u32().unwrap();
            let fractional_part_bits_length = bitpack.read_u32().unwrap();

            let fixed_pred =
                precision_bound::into_fixed_representation_with_fractional_part_bits_length(
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
            }
        }

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        mod simd_x86_64 {
            use std::arch::x86_64::{
                __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_lddqu_si256,
                _mm256_movemask_epi8, _mm256_set1_epi8,
            };
            pub(super) const VECTOR_SIZE: usize = size_of::<__m256i>();

            use roaring::RoaringBitmap;

            use crate::compression_common::buff::flip;

            #[inline]
            pub unsafe fn greater_simd(
                x: &[u8],
                pred: u8,
                qualified_bitmask: &mut RoaringBitmap,
                logical_offset: u32,
            ) -> RoaringBitmap {
                comp_simd::<GREATER>(x, pred, qualified_bitmask, logical_offset)
            }

            #[inline]
            pub unsafe fn less_simd(
                x: &[u8],
                pred: u8,
                qualified_bitmask: &mut RoaringBitmap,
                logical_offset: u32,
            ) -> RoaringBitmap {
                comp_simd::<LESS>(x, pred, qualified_bitmask, logical_offset)
            }

            pub const GREATER: u8 = 1;
            pub const LESS: u8 = 1 << 1;
            #[allow(dead_code)]
            #[inline]
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
                let mut to_check_bitmasks =
                    Vec::with_capacity(x.len().next_multiple_of(VECTOR_SIZE) / VECTOR_SIZE);
                let mut qualified_bitmasks =
                    Vec::with_capacity(x.len().next_multiple_of(VECTOR_SIZE) / VECTOR_SIZE);
                for word in x
                    .chunks_exact(VECTOR_SIZE)
                    .map(|chunk| _mm256_lddqu_si256(chunk.as_ptr().cast::<__m256i>()))
                {
                    let equal = _mm256_cmpeq_epi8(word, pred_word);
                    let eq_mask = _mm256_movemask_epi8(equal);

                    to_check_bitmasks.push(eq_mask as u32);

                    let cmp_mask = if FLAGS & GREATER != 0 {
                        let greater = _mm256_cmpgt_epi8(word, pred_word);
                        _mm256_movemask_epi8(greater)
                    } else {
                        let less_or_eq = _mm256_cmpgt_epi8(pred_word, word);
                        let less_or_eq_mask = _mm256_movemask_epi8(less_or_eq);
                        less_or_eq_mask & (!eq_mask)
                    };

                    qualified_bitmasks.push(cmp_mask as u32);
                }

                debug_assert!(cfg!(target_endian = "little"));
                let qualified_bitmasks_bytes = unsafe {
                    std::slice::from_raw_parts(
                        qualified_bitmasks.as_ptr().cast::<u8>(),
                        size_of_val(qualified_bitmasks.as_slice()),
                    )
                };
                *qualified_bitmask =
                    RoaringBitmap::from_bitmap_bytes(logical_offset, qualified_bitmasks_bytes);

                let to_check_bitmasks_bytes = unsafe {
                    std::slice::from_raw_parts(
                        to_check_bitmasks.as_ptr().cast::<u8>(),
                        size_of_val(to_check_bitmasks.as_slice()),
                    )
                };
                RoaringBitmap::from_bitmap_bytes(logical_offset, to_check_bitmasks_bytes)
            }

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

                RoaringBitmap::from_bitmap_bytes(logical_offset, unsafe {
                    std::slice::from_raw_parts(
                        bitmasks.as_ptr().cast::<u8>(),
                        size_of_val(bitmasks.as_slice()),
                    )
                })
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
