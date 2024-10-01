use std::io::Write;

pub use roaring::RoaringBitmap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    decoder,
    time::{SegmentKind, SegmentedExecutionTimes},
};

use super::chunk_reader::Reader;

#[inline]
pub fn default_materialize<T: Reader + ?Sized, W: Write>(
    reader: &mut T,
    output: &mut W,
    bitmask: Option<&RoaringBitmap>,
    logical_offset: usize,
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<()> {
    let decompressed = reader.decompress(timer)?;
    let io_write_timer = timer.start_addition_measurement(SegmentKind::IOWrite);
    for (i, v) in decompressed.iter().enumerate() {
        let record_offset = logical_offset + i;
        if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
            continue;
        }
        output.write_all(&v.to_ne_bytes())?;
    }
    io_write_timer.stop();

    Ok(())
}

#[inline]
pub fn default_filter<T: Reader + ?Sized>(
    reader: &mut T,
    predicate: Predicate,
    bitmask: Option<&RoaringBitmap>,
    logical_offset: usize,
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<RoaringBitmap> {
    let mut result = RoaringBitmap::new();
    let decompressed = reader.decompress(timer)?;
    let comparison_timer = timer.start_addition_measurement(SegmentKind::CompareInsert);
    for (i, v) in decompressed.iter().enumerate() {
        let record_offset = logical_offset + i;

        if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
            continue;
        }

        if predicate.eval(*v) {
            result.insert(record_offset as u32);
        }
    }
    comparison_timer.stop();

    Ok(result)
}

#[inline]
pub fn default_filter_materialize<T: Reader + ?Sized, W: Write>(
    reader: &mut T,
    output: &mut W,
    predicate: Predicate,
    bitmask: Option<&RoaringBitmap>,
    logical_offset: usize,
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<()> {
    let decompressed = reader.decompress(timer)?;
    let io_write_timer = timer.start_addition_measurement(SegmentKind::IOWrite);
    for (i, v) in decompressed.iter().enumerate() {
        let record_offset = logical_offset + i;

        if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
            continue;
        }

        if predicate.eval(*v) {
            output.write_all(&v.to_ne_bytes())?;
        }
    }
    io_write_timer.stop();

    Ok(())
}

#[inline]
pub fn default_sum<T: Reader + ?Sized>(
    reader: &mut T,
    bitmask: Option<&RoaringBitmap>,
    logical_offset: usize,
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<f64> {
    let mut sum = 0.0;
    let decompressed = reader.decompress(timer)?;
    let sum_timer = timer.start_addition_measurement(SegmentKind::Sum);
    for (i, v) in decompressed.iter().enumerate() {
        let record_offset = logical_offset + i;
        if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
            continue;
        }
        sum += *v;
    }
    sum_timer.stop();

    Ok(sum)
}

#[inline]
pub fn default_min<T: Reader + ?Sized>(
    reader: &mut T,
    bitmask: Option<&RoaringBitmap>,
    logical_offset: usize,
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<f64> {
    let mut min = f64::INFINITY;
    let decompressed = reader.decompress(timer)?;
    let comparison_timer = timer.start_addition_measurement(SegmentKind::CompareInsert);
    for (i, v) in decompressed.iter().enumerate() {
        let record_offset = logical_offset + i;
        if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
            continue;
        }
        min = min.min(*v);
    }
    comparison_timer.stop();

    Ok(min)
}

#[inline]
pub fn default_max<T: Reader + ?Sized>(
    reader: &mut T,
    bitmask: Option<&RoaringBitmap>,
    logical_offset: usize,
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<f64> {
    let mut max = f64::NEG_INFINITY;
    let decompressed = reader.decompress(timer)?;
    let comparison_timer = timer.start_addition_measurement(SegmentKind::CompareInsert);
    for (i, v) in decompressed.iter().enumerate() {
        let record_offset = logical_offset + i;
        if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
            continue;
        }
        max = max.max(*v);
    }
    comparison_timer.stop();

    Ok(max)
}

#[inline]
pub fn default_distance_squared<T: Reader + ?Sized>(
    reader: &mut T,
    offset_in_chunk: usize,
    target: &[f64],
    timer: &mut SegmentedExecutionTimes,
) -> decoder::Result<f64> {
    let decompressed = reader.decompress(timer)?;
    let sum_timer = timer.start_addition_measurement(SegmentKind::Sum);
    let offset_in_chunk = std::cmp::min(decompressed.len(), offset_in_chunk);
    let sum = decompressed[offset_in_chunk..]
        .iter()
        .zip(target)
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    sum_timer.stop();
    Ok(sum)
}

/// A trait for query execution.
/// This trait provides default implementations based of [`Reader`] trait.
/// The each implementation of this trait can be specialized for each compression scheme.
pub trait QueryExecutor: Reader {
    /// Materialize the values filtered by the bitmask and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn materialize<W: Write>(
        &mut self,
        output: &mut W,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<()> {
        default_materialize(self, output, bitmask, logical_offset, timer)
    }

    /// Filter the values by the predicate and return the result as a bitmask.
    /// The result bitmask is global to the whole chunks.
    /// But the result bitmask is guaranteed to only contain the record offsets in the current chunk.
    ///
    /// `bitmask` is optional. If it is None, all values are evaluated by `predicate`.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn filter(
        &mut self,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<RoaringBitmap> {
        default_filter(self, predicate, bitmask, logical_offset, timer)
    }

    /// Filter the values by the predicate and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are filtered and then scaned.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn filter_materialize<W: Write>(
        &mut self,
        output: &mut W,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<()> {
        default_filter_materialize(self, output, predicate, bitmask, logical_offset, timer)
    }

    /// Calculate the sum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn sum(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        default_sum(self, bitmask, logical_offset, timer)
    }

    /// Calculate the minimum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn min(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        default_min(self, bitmask, logical_offset, timer)
    }

    /// Calculate the maximum of the values filtered by the bitmask.
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn max(
        &mut self,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        default_max(self, bitmask, logical_offset, timer)
    }

    /// Calculates the distance^2 between the data at the specified offset in the chunk and the target slice.
    ///
    /// Let `chunk` is the logical chunk of f64 array.
    /// The distance will be calculated with the vector of length: `min(chunk[offset_in_chunk..].len(), target.len())`.
    ///
    /// # Parameters
    ///
    /// - `offset_in_chunk`: The offset within the chunk where the data is located.
    /// - `target`: A slice of `f64` values representing the target data to compare against.
    /// - `timer`: A mutable reference to `SegmentedExecutionTimes` for recording execution times.
    ///
    /// # Returns
    ///
    /// A `Result` containing the calculated distance^2 as an `f64` value, or an error if the calculation fails.
    ///
    fn distance_squared(
        &mut self,
        offset_in_chunk: usize,
        target: &[f64],
        timer: &mut SegmentedExecutionTimes,
    ) -> decoder::Result<f64> {
        default_distance_squared(self, offset_in_chunk, target, timer)
    }
}

pub trait PredicateNumber: Copy + PartialEq + PartialOrd {}
impl<T: Copy + PartialEq + PartialOrd> PredicateNumber for T {}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RangeValueImpl<T: PredicateNumber> {
    Inclusive(T),
    Exclusive(T),
    None,
}

impl<T: PredicateNumber> RangeValueImpl<T> {
    pub fn map<U: PredicateNumber>(self, f: impl Fn(T) -> U) -> RangeValueImpl<U> {
        match self {
            RangeValueImpl::Inclusive(v) => RangeValueImpl::Inclusive(f(v)),
            RangeValueImpl::Exclusive(v) => RangeValueImpl::Exclusive(f(v)),
            RangeValueImpl::None => RangeValueImpl::None,
        }
    }
}

pub type RangeValue = RangeValueImpl<f64>;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RangeImpl<T: PredicateNumber> {
    pub start: RangeValueImpl<T>,
    pub end: RangeValueImpl<T>,
}

pub type Range = RangeImpl<f64>;

impl<T: PredicateNumber> RangeImpl<T> {
    pub fn new(start: RangeValueImpl<T>, end: RangeValueImpl<T>) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> RangeValueImpl<T> {
        self.start
    }

    pub fn end(&self) -> RangeValueImpl<T> {
        self.end
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.start, &mut self.end);
    }

    pub fn eval(&self, value: T) -> bool {
        let start_eval = match self.start {
            RangeValueImpl::Inclusive(v) => value >= v,
            RangeValueImpl::Exclusive(v) => value > v,
            RangeValueImpl::None => true,
        };
        let end_eval = match self.end {
            RangeValueImpl::Inclusive(v) => value <= v,
            RangeValueImpl::Exclusive(v) => value < v,
            RangeValueImpl::None => true,
        };

        start_eval && end_eval
    }

    pub fn map<U: PredicateNumber>(self, f: impl Fn(T) -> U) -> RangeImpl<U> {
        RangeImpl {
            start: self.start.map(&f),
            end: self.end.map(f),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", content = "predicate"))]
pub enum PredicateImpl<T: PredicateNumber> {
    Eq(T),
    Ne(T),
    Range(RangeImpl<T>),
}

pub type Predicate = PredicateImpl<f64>;

impl<T: PredicateNumber> PredicateImpl<T> {
    pub fn eval(&self, value: T) -> bool {
        match self {
            PredicateImpl::Eq(v) => value == *v,
            PredicateImpl::Ne(v) => value != *v,
            PredicateImpl::Range(r) => r.eval(value),
        }
    }

    pub fn map<U: PredicateNumber>(self, f: impl Fn(T) -> U) -> PredicateImpl<U> {
        match self {
            PredicateImpl::Eq(v) => PredicateImpl::Eq(f(v)),
            PredicateImpl::Ne(v) => PredicateImpl::Ne(f(v)),
            PredicateImpl::Range(r) => PredicateImpl::Range(r.map(f)),
        }
    }
}
