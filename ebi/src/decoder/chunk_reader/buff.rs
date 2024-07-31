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
        let logical_offset = logical_offset as u32;
        let bytes = self.bytes()?;
        Ok(match predicate {
            p @ (Predicate::Eq(pred) | Predicate::Ne(pred)) => {
                let result = internal::query::buff_simd512_equal_filter::<
                    true,
                    false,
                    false,
                    false,
                    false,
                    false,
                >(bytes, pred, bitmask, logical_offset);

                let result = if let Predicate::Eq(_) = p {
                    result
                } else {
                    let mut all = roaring::RoaringBitmap::new();
                    all.insert_range(
                        logical_offset..(self.number_of_records as u32 + logical_offset),
                    );
                    all - result
                };

                result
            }
            Predicate::Range(r) => {
                let result = match (r.start, r.end) {
                    (RangeValue::Inclusive(left), RangeValue::Inclusive(right)) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            true,
                            false,
                            true,
                            false,
                        >(bytes, left, bitmask, logical_offset)
                            & internal::query::buff_simd512_equal_filter::<
                                false,
                                false,
                                false,
                                true,
                                false,
                                true,
                            >(bytes, right, bitmask, logical_offset)
                    }
                    (RangeValue::Inclusive(left), RangeValue::Exclusive(right)) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            true,
                            false,
                            true,
                            false,
                        >(bytes, left, bitmask, logical_offset)
                            & internal::query::buff_simd512_equal_filter::<
                                false,
                                false,
                                false,
                                true,
                                false,
                                false,
                            >(bytes, right, bitmask, logical_offset)
                    }
                    (RangeValue::Inclusive(left), RangeValue::None) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            true,
                            false,
                            true,
                            false,
                        >(bytes, left, bitmask, logical_offset)
                    }
                    (RangeValue::Exclusive(left), RangeValue::Inclusive(right)) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            true,
                            false,
                            false,
                            false,
                        >(bytes, left, bitmask, logical_offset)
                            & internal::query::buff_simd512_equal_filter::<
                                false,
                                false,
                                false,
                                true,
                                false,
                                true,
                            >(bytes, right, bitmask, logical_offset)
                    }
                    (RangeValue::Exclusive(left), RangeValue::Exclusive(right)) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            true,
                            false,
                            false,
                            false,
                        >(bytes, left, bitmask, logical_offset)
                            & internal::query::buff_simd512_equal_filter::<
                                false,
                                false,
                                false,
                                true,
                                false,
                                false,
                            >(bytes, right, bitmask, logical_offset)
                    }
                    (RangeValue::Exclusive(left), RangeValue::None) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            true,
                            false,
                            false,
                            false,
                        >(bytes, left, bitmask, logical_offset)
                    }
                    (RangeValue::None, RangeValue::Inclusive(right)) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            false,
                            true,
                            false,
                            true,
                        >(bytes, right, bitmask, logical_offset)
                    }
                    (RangeValue::None, RangeValue::Exclusive(right)) => {
                        internal::query::buff_simd512_equal_filter::<
                            false,
                            false,
                            false,
                            true,
                            false,
                            false,
                        >(bytes, right, bitmask, logical_offset)
                    }
                    (RangeValue::None, RangeValue::None) => {
                        let mut all = roaring::RoaringBitmap::new();
                        all.insert_range(
                            logical_offset..(self.number_of_records as u32 + logical_offset),
                        );
                        all
                    }
                };
                result
            }
        })
    }
}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{bit_packing::BitPack, flip};

    pub(crate) mod query {
        use roaring::RoaringBitmap;
        use simd::{equal_simd_myroaring, range_simd};

        use crate::compression_common::buff::{
            bit_packing::{BitPack, BYTE_BITS},
            flip,
            precision_bound::PrecisionBound,
        };

        #[inline]
        fn eval<
            const IS_EQ: bool,
            const IS_NE: bool,
            const LEFT_EXIST: bool,
            const RIGHT_EXISIT: bool,
            const LEFT_INCLUSIVE: bool,
            const RIGHT_INCLUSIVE: bool,
        >(
            x: u8,
            pred: u8,
        ) -> bool {
            println!("x: {:x?}({:x?}), pred: {:x?}", x, flip(x), pred);
            let r = if IS_EQ {
                x == pred
            } else if IS_NE {
                x != pred
            } else {
                // range
                let left = !LEFT_EXIST || if LEFT_INCLUSIVE { pred <= x } else { pred < x };
                let right = !RIGHT_EXISIT || if RIGHT_INCLUSIVE { x <= pred } else { x < pred };

                left && right
            };

            dbg!(r);
            r
        }

        pub(crate) fn buff_simd512_equal_filter<
            const IS_EQ: bool,
            const IS_NE: bool,
            const LEFT_EXIST: bool,
            const RIGHT_EXISIT: bool,
            const LEFT_INCLUSIVE: bool,
            const RIGHT_INCLUSIVE: bool,
        >(
            bytes: &[u8],
            pred: f64,
            bitmask: Option<&RoaringBitmap>,
            logical_offset: u32,
        ) -> RoaringBitmap {
            dbg!((
                IS_EQ,
                LEFT_EXIST,
                RIGHT_EXISIT,
                LEFT_INCLUSIVE,
                RIGHT_INCLUSIVE
            ));
            let eval_u8 =
                eval::<IS_EQ, IS_NE, LEFT_EXIST, RIGHT_EXISIT, LEFT_INCLUSIVE, RIGHT_INCLUSIVE>;
            const {
                assert!(
                    !(LEFT_EXIST && RIGHT_EXISIT),
                    "currently range should be one side"
                );
                assert!(!IS_NE, "currently IS_NE not implemented");
            };
            let simd_eval = if IS_EQ {
                equal_simd_myroaring
            } else if LEFT_EXIST {
                range_simd::<true, LEFT_INCLUSIVE>
            } else {
                range_simd::<false, RIGHT_INCLUSIVE>
            };

            // Read the header
            let mut bitpack = BitPack::<&[u8]>::new(bytes);
            // let fixed_base64_lower = bitpack.read(32).unwrap();
            let fixed_base64_lower = bitpack.read_u32().unwrap();
            // let fixed_base64_higher = bitpack.read(32).unwrap();
            let fixed_base64_higher = bitpack.read_u32().unwrap();
            let fixed_base64_bits =
                (fixed_base64_lower as u64) | ((fixed_base64_higher as u64) << 32);
            let fixed_base64 = unsafe { std::mem::transmute::<u64, i64>(fixed_base64_bits) };
            // println!("base integer: {}",base_int);
            // let number_of_records = bitpack.read(32).unwrap();
            let number_of_records = bitpack.read_u32().unwrap();
            // println!("total vector size:{}",len);
            // let fixed_representation_bits_length = bitpack.read(32).unwrap();
            let fixed_representation_bits_length = bitpack.read_u32().unwrap();
            // println!("bit packing length:{}",ilen);
            // let fractional_part_bits_length = bitpack.read(32).unwrap();
            let fractional_part_bits_length = bitpack.read_u32().unwrap();
            // check integer part and update bitmap;
            let fixed_pred = PrecisionBound::into_fixed_representation_with_decimal_length(
                pred,
                fractional_part_bits_length as i32,
            );

            dbg!(
                fixed_base64,
                fixed_pred,
                number_of_records,
                fixed_representation_bits_length,
                fractional_part_bits_length
            );
            let all = || {
                let mut all = RoaringBitmap::new();
                all.insert_range(logical_offset..(number_of_records + logical_offset));
                all
            };
            let none = RoaringBitmap::new;
            // First Skip all the evaluation based on delta base
            if fixed_pred < fixed_base64 {
                if IS_EQ || RIGHT_EXISIT {
                    return none();
                } else if IS_NE || LEFT_EXIST {
                    return all();
                }
            }

            let delta_fixed_pred = (fixed_pred - fixed_base64) as u64;

            let mut remaining_bits_length = fixed_representation_bits_length;

            let mut result_bitmask = bitmask.cloned();
            while remaining_bits_length >= 8 {
                dbg!(remaining_bits_length);
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
                        if eval_u8(subcolumn, delta_fixed_pred_subcolumn) {
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
                    let mut bitmask = RoaringBitmap::new();
                    let chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();
                    for (record_index, &subcolumn) in chunk.iter().enumerate() {
                        let subcolumn = flip(subcolumn);
                        if eval_u8(subcolumn, delta_fixed_pred_subcolumn) {
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

            dbg!(remaining_bits_length);
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

                        dbg!(
                            record_index,
                            expected_record_index_if_sequential,
                            n_bits_skipped,
                            remaining_bits_length
                        );
                        let subcolumn = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                        // no flip for the last bits(less than byte) column
                        if eval_u8(subcolumn, delta_fixed_pred_subcolumn) {
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
                        let subcolumn = flip(subcolumn);
                        if eval_u8(subcolumn, delta_fixed_pred as u8) {
                            next_result_bitmask.insert(logical_offset + i);
                        }
                    }

                    next_result_bitmask
                }
            } else {
                result_bitmask.unwrap_or_else(all)
            };
            dbg!(&r);
            return r;

            let mut cur;
            let mut rb1 = RoaringBitmap::new();
            let mut res = RoaringBitmap::new();
            let mut delta_fixed_pred_mostright_byte = delta_fixed_pred as u8;
            // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
            let mut byte_count = 0;
            // let start = Instant::now();

            if remaining_bits_length < 8 {
                for i in 0..number_of_records {
                    cur = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                    if eval_u8(cur, delta_fixed_pred as u8) {
                        res.insert(i + logical_offset);
                    }
                }
                remaining_bits_length = 0;
            } else {
                remaining_bits_length -= 8;
                byte_count += 1;
                let chunk = bitpack.read_n_byte(number_of_records as usize).unwrap();
                // let start = Instant::now();
                delta_fixed_pred_mostright_byte = (delta_fixed_pred >> remaining_bits_length) as u8;
                unsafe {
                    rb1 = simd_eval(chunk, delta_fixed_pred_mostright_byte, logical_offset);
                }
                // let duration = start.elapsed();
                // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
                // let mut i =0;
                // for &c in chunk {
                //     if c == dec_byte {
                //         rb1.insert(i);
                //     };
                //     i+=1;
                // }
            }
            // rb1.run_optimize();
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
            // println!("Number of qualified items for equal:{}", rb1.len());

            while (remaining_bits_length > 0) {
                // if we can read by byte
                if remaining_bits_length >= 8 {
                    remaining_bits_length -= 8;
                    byte_count += 1;
                    let mut cur_rb = RoaringBitmap::new();
                    if rb1.len() != 0 {
                        // let start = Instant::now();
                        let mut iterator = rb1.iter();
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre: u32 = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        delta_fixed_pred_mostright_byte =
                            (delta_fixed_pred >> remaining_bits_length as u64) as u8;
                        if it != None {
                            dec_cur = it.unwrap();
                            if dec_cur != 0 {
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if eval_u8(dec, delta_fixed_pred_mostright_byte) {
                                cur_rb.insert(dec_cur + logical_offset);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it != None {
                            dec_cur = it.unwrap();
                            delta = dec_cur - dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta - 1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if eval_u8(dec, delta_fixed_pred_mostright_byte) {
                                cur_rb.insert(dec_cur + logical_offset);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        if number_of_records - dec_pre > 1 {
                            bitpack.skip_n_byte((number_of_records - dec_pre - 1) as usize);
                        }
                    } else {
                        bitpack.skip_n_byte((number_of_records) as usize);
                        break;
                    }
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    delta_fixed_pred_mostright_byte = (((delta_fixed_pred as u8)
                        << ((BYTE_BITS - remaining_bits_length as usize) as u8))
                        >> ((BYTE_BITS - remaining_bits_length as usize) as u8));
                    bitpack.finish_read_byte();
                    if rb1.len() != 0 {
                        // let start = Instant::now();
                        let mut iterator = rb1.iter();
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;

                        let mut dec_pre: u32 = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        if it != None {
                            dec_cur = it.unwrap();
                            if dec_cur != 0 {
                                bitpack.skip(((dec_cur) * remaining_bits_length) as usize);
                            }
                            dec = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                            if eval_u8(dec, delta_fixed_pred_mostright_byte) {
                                res.insert(dec_cur + logical_offset);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it != None {
                            dec_cur = it.unwrap();
                            delta = dec_cur - dec_pre;
                            if delta != 1 {
                                bitpack.skip(((delta - 1) * remaining_bits_length) as usize);
                            }
                            dec = bitpack.read_bits(remaining_bits_length as usize).unwrap();
                            // if dec == delta_fixed_pred_mostright_byte {
                            if eval_u8(dec, delta_fixed_pred_mostright_byte) {
                                res.insert(dec_cur + logical_offset);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        // println!("read the remain {} bits of dec",remain);
                        remaining_bits_length = 0;
                    } else {
                        break;
                    }
                }
            }
            println!("Number of qualified int items:{}", res.len());
            return res;
        }

        mod simd {
            use std::arch::x86_64::{
                __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_lddqu_si256,
                _mm256_movemask_epi8, _mm256_set1_epi8, _popcnt32,
            };
            const VECTOR_SIZE: usize = size_of::<__m256i>();

            use roaring::RoaringBitmap;

            use crate::compression_common::buff::flip;

            pub unsafe fn range_simd<const IS_LEFT: bool, const IS_INCLUSIVE: bool>(
                x: &[u8],
                pred: u8,
                logical_offset: u32,
            ) -> RoaringBitmap {
                debug_assert!(x.len() % VECTOR_SIZE == 0);
                let haystack = x;
                let start_ptr = haystack.as_ptr();
                let end_ptr = haystack[haystack.len()..].as_ptr();
                let mut ptr = start_ptr;

                let predicate = std::mem::transmute::<u8, i8>(flip(pred));
                let pred_word = _mm256_set1_epi8(predicate);
                let ep = end_ptr.sub(VECTOR_SIZE);
                let mut middle = RoaringBitmap::new();

                let mut count = 0;
                while ptr <= ep {
                    let word1 = _mm256_lddqu_si256(ptr as *const __m256i);
                    let mask = if IS_LEFT {
                        let greater = _mm256_cmpgt_epi8(word1, pred_word);
                        let greater_mask = _mm256_movemask_epi8(greater);
                        if IS_INCLUSIVE {
                            let equal = _mm256_cmpeq_epi8(word1, pred_word);
                            let eq_mask = _mm256_movemask_epi8(equal);
                            greater_mask | eq_mask
                        } else {
                            greater_mask
                        }
                    } else {
                        let less_or_equal = _mm256_cmpgt_epi8(pred_word, word1);
                        let less_or_equal_mask = _mm256_movemask_epi8(less_or_equal);

                        if IS_INCLUSIVE {
                            less_or_equal_mask
                        } else {
                            let equal = _mm256_cmpeq_epi8(word1, pred_word);
                            let eq_mask = _mm256_movemask_epi8(equal);
                            less_or_equal_mask & !eq_mask
                        }
                    };

                    for i in 0..32 {
                        if (mask & (1 << i)) != 0 {
                            middle.insert(count + i + logical_offset);
                        }
                    }
                    count += 32;
                    ptr = ptr.add(VECTOR_SIZE);
                }
                println!("\tsimd greater than bitmap: {}", middle.len());
                middle
            }

            pub unsafe fn equal_simd_myroaring(
                x: &[u8],
                pred: u8,
                logical_offset: u32,
            ) -> RoaringBitmap {
                debug_assert!(x.len() % VECTOR_SIZE == 0);
                let haystack = x;
                let start_ptr = haystack.as_ptr();
                let end_ptr = haystack[haystack.len()..].as_ptr();
                let mut ptr = start_ptr;

                let predicate = std::mem::transmute::<u8, i8>(flip(pred));
                println!("current predicate:{}", pred);
                let mut pred_word = _mm256_set1_epi8(predicate);
                let ep = end_ptr.sub(VECTOR_SIZE);
                let mut middle = RoaringBitmap::new();
                // let mut gt = Vec::<i32>::new();
                let mut qualified = 0;
                let mut count = 0;
                while ptr <= ep {
                    let word1 = _mm256_lddqu_si256(ptr as *const __m256i);

                    let equal = _mm256_cmpeq_epi8(word1, pred_word);
                    let eq_mask = _mm256_movemask_epi8(equal);
                    // TODO: potentially, using custom `insert_direct_u32` could be faster
                    // middle.insert_direct_u32(
                    //     count + logical_offset,
                    //     std::mem::transmute::<i32, u32>(eq_mask),
                    // );
                    for i in 0..32 {
                        if (eq_mask & (1 << i)) != 0 {
                            middle.insert(count + i + logical_offset);
                        }
                    }

                    // println!("{:?}{:?}{:b}", word1, pred_word, mask);
                    // gt.push(mask);
                    qualified += _popcnt32(eq_mask);
                    count += 32;
                    ptr = ptr.add(VECTOR_SIZE);
                }
                println!("\tsimd equal bitmap: {}", qualified);
                middle
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
