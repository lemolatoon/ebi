#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    ops::AddAssign,
    time::{Duration, Instant},
};

use crate::format::{deserialize::FromLeBytes, serialize};

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum SegmentKind {
    IORead = 0,
    IOWrite,
    Xor,
    Delta,
    Quantization,
    BitPacking,
    CompareInsert,
    Sum,
    Decompression,
}

pub trait KindIndexable: Copy {
    const N_SEGMENT_KINDS: usize;
    fn index(self) -> usize;
}

impl KindIndexable for SegmentKind {
    const N_SEGMENT_KINDS: usize = 9;
    #[inline]
    fn index(self) -> usize {
        let index = self as usize;
        debug_assert!(
            index < Self::N_SEGMENT_KINDS,
            "Invalid segment kind: {:?}",
            self
        );
        index
    }
}

#[derive(Eq, PartialEq)]
pub struct SegmentTimer<'a, T: KindIndexable, const N: usize> {
    times: &'a mut SegmentedExecutionTimesImpl<T, N>,
    is_addition: bool,
    stop_on_drop: bool,
    start: Instant,
    kind: T,
}

impl<T: KindIndexable, const N: usize> SegmentTimer<'_, T, N> {
    pub fn stop(mut self) {
        self.stop_without_drop();
    }

    fn stop_without_drop(&mut self) {
        let elapsed = self.start.elapsed();
        if self.is_addition {
            self.times.add_at(self.kind, elapsed);
        } else {
            self.times.update_at(self.kind, elapsed);
        }
    }

    pub fn stop_on_drop(&mut self) {
        self.stop_on_drop = true;
    }
}

impl<T: KindIndexable, const N: usize> Drop for SegmentTimer<'_, T, N> {
    fn drop(&mut self) {
        if self.stop_on_drop {
            self.stop_without_drop();
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub struct SegmentedExecutionTimesImpl<T: KindIndexable, const N: usize> {
    elapsed_times: [Duration; N],
    _kind_indexable: PhantomData<T>,
}

pub type SegmentedExecutionTimes =
    SegmentedExecutionTimesImpl<SegmentKind, { SegmentKind::N_SEGMENT_KINDS }>;

impl std::fmt::Debug for SegmentedExecutionTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentedExecutionTimes")
            .field("IORead", &self.elapsed_times[SegmentKind::IORead.index()])
            .field("IOWrite", &self.elapsed_times[SegmentKind::IOWrite.index()])
            .field("Xor", &self.elapsed_times[SegmentKind::Xor.index()])
            .field("Delta", &self.elapsed_times[SegmentKind::Delta.index()])
            .field(
                "Quantization",
                &self.elapsed_times[SegmentKind::Quantization.index()],
            )
            .field(
                "BitPacking",
                &self.elapsed_times[SegmentKind::BitPacking.index()],
            )
            .field(
                "CompareInsert",
                &self.elapsed_times[SegmentKind::CompareInsert.index()],
            )
            .field("Sum", &self.elapsed_times[SegmentKind::Sum.index()])
            .field(
                "Decompression",
                &self.elapsed_times[SegmentKind::Decompression.index()],
            )
            .finish()
    }
}

impl<T: KindIndexable, const N: usize> AddAssign for SegmentedExecutionTimesImpl<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        for (lhs, rhs) in self
            .elapsed_times
            .iter_mut()
            .zip(rhs.elapsed_times.into_iter())
        {
            *lhs += rhs;
        }
    }
}

#[repr(C, packed(1))]
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializableSegmentedExecutionTimes {
    pub io_read_nanos: u128,
    pub io_write_nanos: u128,
    pub xor_nanos: u128,
    pub delta_nanos: u128,
    pub quantization_nanos: u128,
    pub bit_packing_nanos: u128,
    pub compare_insert_nanos: u128,
    pub sum_nanos: u128,
    pub decompression_nanos: u128,
}
serialize::impl_to_le!(
    SerializableSegmentedExecutionTimes,
    io_read_nanos,
    io_write_nanos,
    xor_nanos,
    delta_nanos,
    quantization_nanos,
    bit_packing_nanos,
    compare_insert_nanos,
    sum_nanos,
    decompression_nanos
);
impl FromLeBytes for SerializableSegmentedExecutionTimes {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        let io_read_nanos = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
        let io_write_nanos = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
        let xor_nanos = u128::from_le_bytes(bytes[32..48].try_into().unwrap());
        let delta_nanos = u128::from_le_bytes(bytes[48..64].try_into().unwrap());
        let quantization_nanos = u128::from_le_bytes(bytes[64..80].try_into().unwrap());
        let bit_packing_nanos = u128::from_le_bytes(bytes[80..96].try_into().unwrap());
        let compare_insert_nanos = u128::from_le_bytes(bytes[96..112].try_into().unwrap());
        let sum_nanos = u128::from_le_bytes(bytes[112..128].try_into().unwrap());
        let decompression_nanos = u128::from_le_bytes(bytes[128..144].try_into().unwrap());

        Self {
            io_read_nanos,
            io_write_nanos,
            xor_nanos,
            delta_nanos,
            quantization_nanos,
            bit_packing_nanos,
            compare_insert_nanos,
            sum_nanos,
            decompression_nanos,
        }
    }
}

impl From<SegmentedExecutionTimes> for SerializableSegmentedExecutionTimes {
    fn from(times: SegmentedExecutionTimes) -> Self {
        Self {
            io_read_nanos: times.io_read().as_nanos(),
            io_write_nanos: times.io_write().as_nanos(),
            xor_nanos: times.xor().as_nanos(),
            delta_nanos: times.delta().as_nanos(),
            quantization_nanos: times.quantization().as_nanos(),
            bit_packing_nanos: times.bit_packing().as_nanos(),
            compare_insert_nanos: times.compare_insert().as_nanos(),
            sum_nanos: times.sum().as_nanos(),
            decompression_nanos: times.decompression().as_nanos(),
        }
    }
}

impl From<SerializableSegmentedExecutionTimes> for SegmentedExecutionTimes {
    fn from(times: SerializableSegmentedExecutionTimes) -> Self {
        let mut elapsed_times = [Duration::default(); SegmentKind::N_SEGMENT_KINDS];

        fn u128_nanos_to_duration(nanos: u128) -> Duration {
            let secs = (nanos / 1_000_000_000) as u64;
            let nanos = (nanos % 1_000_000_000) as u32;

            Duration::new(secs, nanos)
        }
        elapsed_times[SegmentKind::IORead.index()] = u128_nanos_to_duration(times.io_read_nanos);
        elapsed_times[SegmentKind::IOWrite.index()] = u128_nanos_to_duration(times.io_write_nanos);
        elapsed_times[SegmentKind::Xor.index()] = u128_nanos_to_duration(times.xor_nanos);
        elapsed_times[SegmentKind::Delta.index()] = u128_nanos_to_duration(times.delta_nanos);
        elapsed_times[SegmentKind::Quantization.index()] =
            u128_nanos_to_duration(times.quantization_nanos);
        elapsed_times[SegmentKind::BitPacking.index()] =
            u128_nanos_to_duration(times.bit_packing_nanos);
        elapsed_times[SegmentKind::CompareInsert.index()] =
            u128_nanos_to_duration(times.compare_insert_nanos);
        elapsed_times[SegmentKind::Sum.index()] = u128_nanos_to_duration(times.sum_nanos);
        elapsed_times[SegmentKind::Decompression.index()] =
            u128_nanos_to_duration(times.decompression_nanos);

        Self {
            elapsed_times,
            _kind_indexable: PhantomData,
        }
    }
}

impl<T: KindIndexable, const N: usize> Default for SegmentedExecutionTimesImpl<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: KindIndexable, const N: usize> SegmentedExecutionTimesImpl<T, N> {
    pub fn new() -> Self {
        Self {
            elapsed_times: [Duration::from_secs(0); N],
            _kind_indexable: PhantomData,
        }
    }

    pub fn start_measurement(&mut self, kind: T) -> SegmentTimer<T, N> {
        SegmentTimer {
            times: self,
            is_addition: false,
            stop_on_drop: false,
            start: Instant::now(),
            kind,
        }
    }

    pub fn start_addition_measurement(&mut self, kind: T) -> SegmentTimer<T, N> {
        SegmentTimer {
            times: self,
            is_addition: true,
            stop_on_drop: false,
            start: Instant::now(),
            kind,
        }
    }

    fn update_at(&mut self, kind: T, value: Duration) {
        self.elapsed_times[kind.index()] = value;
    }

    fn add_at(&mut self, kind: T, value: Duration) {
        let elapsed_time = &mut self.elapsed_times[kind.index()];

        *elapsed_time += value;
    }
}

impl SegmentedExecutionTimes {
    pub fn io_read(&self) -> Duration {
        self.elapsed_times[SegmentKind::IORead.index()]
    }

    pub fn io_write(&self) -> Duration {
        self.elapsed_times[SegmentKind::IOWrite.index()]
    }

    pub fn xor(&self) -> Duration {
        self.elapsed_times[SegmentKind::Xor.index()]
    }

    pub fn delta(&self) -> Duration {
        self.elapsed_times[SegmentKind::Delta.index()]
    }

    pub fn quantization(&self) -> Duration {
        self.elapsed_times[SegmentKind::Quantization.index()]
    }

    pub fn bit_packing(&self) -> Duration {
        self.elapsed_times[SegmentKind::BitPacking.index()]
    }

    pub fn compare_insert(&self) -> Duration {
        self.elapsed_times[SegmentKind::CompareInsert.index()]
    }

    pub fn sum(&self) -> Duration {
        self.elapsed_times[SegmentKind::Sum.index()]
    }

    pub fn decompression(&self) -> Duration {
        self.elapsed_times[SegmentKind::Decompression.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::SerializableSegmentedExecutionTimes;

    #[test]
    pub fn test_segmented_execution_times_consistency() {
        let mut times = super::SegmentedExecutionTimes::new();
        for (i, duration) in times.elapsed_times.iter_mut().enumerate() {
            *duration = std::time::Duration::from_secs(i as u64);
        }

        let serialized = SerializableSegmentedExecutionTimes::from(times);
        let deserialized = super::SegmentedExecutionTimes::from(serialized);

        assert_eq!(times, deserialized);
    }
}
