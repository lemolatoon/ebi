mod controlled_precision;
mod decode;
mod filter;
mod filter_cmp;
mod filter_eq;
mod min_max;
mod sum;

use core::slice;
use std::{io::Read, iter};

use roaring::RoaringBitmap;

use crate::{
    decoder::{
        self,
        query::{default_materialize, Predicate, QueryExecutor, RangeValue},
        FileMetadataLike, GeneralChunkHandle,
    },
    time::{SegmentKind, SegmentedExecutionTimes},
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFReader {
    chunk_size: u64,
    number_of_records: u64,
    bytes: Vec<u8>,
    /// `None` means full precision, otherwise indicates the precision to be used when materializing
    precision: Option<u32>,
    decompressed: Option<Vec<f64>>,
}

impl BUFFReader {
    pub fn new<T: FileMetadataLike, R: Read>(
        handle: &GeneralChunkHandle<T>,
        mut reader: R,
        timer: &mut SegmentedExecutionTimes,
    ) -> Self {
        let chunk_size = handle.chunk_size();
        let mut chunk_in_memory = vec![0; chunk_size as usize];
        let io_read_timer = timer.start_addition_measurement(SegmentKind::IORead);
        reader.read_exact(&mut chunk_in_memory).unwrap();
        io_read_timer.stop();
        let number_of_records = handle.number_of_records();
        Self {
            chunk_size,
            number_of_records,
            precision: None,
            bytes: chunk_in_memory,
            decompressed: None,
        }
    }

    /// Set the precision to be used when materializing the data
    pub fn with_controlled_precision(&mut self, precision: u32) {
        // If the precision is higher than the current precision, we need to clear the cached decompressed data
        if self.precision.map_or(true, |old_p| old_p < precision) {
            self.decompressed = None;
        }
        self.precision = Some(precision);
    }

    pub fn is_full_precision(&self) -> bool {
        self.precision.is_none()
    }

    pub fn precision(&self) -> Option<u32> {
        self.precision
    }
}

type F = fn(&f64) -> decoder::Result<f64>;
pub type BUFFIterator<'a> = iter::Map<slice::Iter<'a, f64>, F>;

impl Reader for BUFFReader {
    type NativeHeader = ();

    type DecompressIterator<'a>
        = BUFFIterator<'a>
    where
        Self: 'a;

    fn decompress(&mut self, timer: &mut SegmentedExecutionTimes) -> decoder::Result<&[f64]> {
        if self.decompressed.is_some() {
            return Ok(self.decompressed.as_ref().unwrap());
        }

        let bitpacking_timer = timer.start_addition_measurement(SegmentKind::BitPacking);
        let result = match self.precision {
            Some(precision) => controlled_precision::decode_with_precision(&self.bytes, precision),
            None => decode::buff_simd256_decode(&self.bytes),
        };
        bitpacking_timer.stop();

        self.decompressed = Some(result);

        Ok(self.decompressed.as_ref().unwrap())
    }

    fn decompress_iter(&mut self) -> decoder::Result<Self::DecompressIterator<'_>> {
        let decompressed = self.decompress(&mut SegmentedExecutionTimes::new())?;

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
        timer: &mut SegmentedExecutionTimes,
    ) -> crate::decoder::Result<roaring::RoaringBitmap> {
        let logical_offset = logical_offset as u32;
        let number_of_records = self.number_of_records as u32;
        let bitmask: Option<roaring::RoaringBitmap> = bitmask.map(|x| {
            x.iter()
                .filter(|&x| x >= logical_offset && x < logical_offset + number_of_records)
                .collect()
        });
        let bytes = &self.bytes;
        let all = || {
            let mut all = roaring::RoaringBitmap::new();
            all.insert_range(logical_offset..(number_of_records + logical_offset));
            all
        };
        let comparison_timer = timer.start_addition_measurement(SegmentKind::CompareInsert);
        let result = match predicate {
            p @ (Predicate::Eq(pred) | Predicate::Ne(pred)) => {
                let result = filter_eq::buff_simd256_equal_filter(
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
                    RangeValue::Inclusive(pred) => {
                        Some(filter_cmp::buff_simd256_cmp_filter::<true, true>(
                            bytes,
                            pred,
                            bitmask.clone(),
                            logical_offset,
                        ))
                    }
                    RangeValue::Exclusive(pred) => {
                        Some(filter_cmp::buff_simd256_cmp_filter::<true, false>(
                            bytes,
                            pred,
                            bitmask.clone(),
                            logical_offset,
                        ))
                    }
                    RangeValue::None => None,
                };

                let right = match r.end() {
                    RangeValue::Inclusive(pred) => {
                        Some(filter_cmp::buff_simd256_cmp_filter::<false, true>(
                            bytes,
                            pred,
                            bitmask.clone(),
                            logical_offset,
                        ))
                    }
                    RangeValue::Exclusive(pred) => {
                        Some(filter_cmp::buff_simd256_cmp_filter::<false, false>(
                            bytes,
                            pred,
                            bitmask.clone(),
                            logical_offset,
                        ))
                    }
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
        };

        comparison_timer.stop();
        Ok(result)
    }

    fn filter_materialize<W: std::io::Write>(
        &mut self,
        output: &mut W,
        predicate: Predicate,
        bitmask: Option<&roaring::RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<()> {
        let filtered = self.filter(predicate, bitmask, logical_offset, timer)?;

        default_materialize(self, output, Some(&filtered), logical_offset, timer)
    }

    fn max(
        &mut self,
        bitmask: Option<&roaring::RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        let logical_offset = logical_offset as u32;
        let number_of_records = self.number_of_records as u32;
        let bitmask: Option<roaring::RoaringBitmap> = bitmask.map(|x| {
            RoaringBitmap::from_sorted_iter(
                x.iter()
                    .filter(|&x| x >= logical_offset && x < logical_offset + number_of_records),
            )
            .unwrap()
        });

        let bitpacking_timer = timer.start_addition_measurement(SegmentKind::CompareInsert);
        let max_fp = if let Some(bitmask) = bitmask {
            if bitmask.is_empty() {
                return Ok(f64::NAN);
            }

            min_max::max_with_bitmask::<false>(&self.bytes, bitmask, logical_offset)?
        } else {
            min_max::max_without_bitmask(&self.bytes, logical_offset)?
        };
        bitpacking_timer.stop();

        Ok(max_fp)
    }

    fn min(
        &mut self,
        bitmask: Option<&roaring::RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        let logical_offset = logical_offset as u32;
        let number_of_records = self.number_of_records as u32;
        let bitmask: Option<roaring::RoaringBitmap> = bitmask.map(|x| {
            RoaringBitmap::from_sorted_iter(
                x.iter()
                    .filter(|&x| x >= logical_offset && x < logical_offset + number_of_records),
            )
            .unwrap()
        });

        let bitpacking_timer = timer.start_addition_measurement(SegmentKind::CompareInsert);
        let min_fp = if let Some(bitmask) = bitmask {
            if bitmask.is_empty() {
                return Ok(f64::NAN);
            }

            min_max::max_with_bitmask::<true>(&self.bytes, bitmask, logical_offset)?
        } else {
            min_max::min_without_bitmask(&self.bytes)?
        };
        bitpacking_timer.stop();

        Ok(min_fp)
    }

    fn sum(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        let sum_timer = timer.start_addition_measurement(SegmentKind::Sum);
        let logical_offset = logical_offset as u32;
        let number_of_records = self.number_of_records as u32;
        let bitmask: Option<roaring::RoaringBitmap> = bitmask.map(|x| {
            RoaringBitmap::from_sorted_iter(
                x.iter()
                    .filter(|&x| x >= logical_offset && x < logical_offset + number_of_records),
            )
            .unwrap()
        });
        let result = sum::sum_with_bitmask(&self.bytes, bitmask.as_ref(), logical_offset);
        sum_timer.stop();
        result
    }
}
