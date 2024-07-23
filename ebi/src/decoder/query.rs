use std::io::{self, Write};

use roaring::RoaringBitmap;

use super::chunk_reader::Reader;

/// A trait for query execution.
/// This trait provides default implementations based of [`Reader`] trait.
/// The each implementation of this trait can be specialized for each compression scheme.
pub trait QueryExecutor: Reader {
    /// Scan the values filtered by the bitmask and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are written.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn scan<W: Write>(
        &mut self,
        output: &mut W,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
    ) -> io::Result<()> {
        for (i, v) in self.decompress().iter().enumerate() {
            let record_offset = logical_offset + i;
            if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
                continue;
            }
            output.write_all(&v.to_ne_bytes())?;
        }

        Ok(())
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
    ) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for (i, v) in self.decompress().iter().enumerate() {
            let record_offset = logical_offset + i;

            if bitmask.is_some_and(|bm| !bm.contains(record_offset as u32)) {
                continue;
            }

            if predicate.eval(*v) {
                result.insert(record_offset as u32);
            }
        }

        result
    }

    /// Filter the values by the predicate and write the results as IEEE754 double array to the output.
    ///
    /// `bitmask` is optional. If it is None, all values are filtered and then scaned.
    ///
    /// `bitmask`'s index is global to the whole chunks.
    /// That is why `logical_offset` is necessary to access bitmask.
    fn filter_scan<W: Write>(
        &mut self,
        output: &mut W,
        predicate: Predicate,
        bitmask: Option<&RoaringBitmap>,
        logical_offset: usize,
    ) -> io::Result<()> {
        let filter_result = self.filter(predicate, bitmask, logical_offset);

        self.scan(output, Some(&filter_result), logical_offset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Predicate {
    Eq(f64),
    Ne(f64),
    Gt(f64),
    Ge(f64),
    Lt(f64),
    Le(f64),
}

impl Predicate {
    pub fn eval(&self, value: f64) -> bool {
        match self {
            Predicate::Eq(v) => value == *v,
            Predicate::Ne(v) => value != *v,
            Predicate::Gt(v) => value > *v,
            Predicate::Ge(v) => value >= *v,
            Predicate::Lt(v) => value < *v,
            Predicate::Le(v) => value <= *v,
        }
    }
}
