#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
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
}

const N_SEGMENT_KINDS: usize = 6;

impl SegmentKind {
    #[inline]
    pub fn index(self) -> usize {
        let index = self as usize;
        debug_assert!(index < N_SEGMENT_KINDS, "Invalid segment kind: {:?}", self);
        index
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct SegmentTimer<'a> {
    times: &'a mut SegmentedExecutionTimes,
    is_addition: bool,
    start: Instant,
    kind: SegmentKind,
}

impl SegmentTimer<'_> {
    pub fn stop(self) {
        let elapsed = self.start.elapsed();
        if self.is_addition {
            self.times.add_at(self.kind, elapsed);
        } else {
            self.times.update_at(self.kind, elapsed);
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct SegmentedExecutionTimes {
    elapsed_times: [Duration; N_SEGMENT_KINDS],
}

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
            .finish()
    }
}

impl AddAssign for SegmentedExecutionTimes {
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
}
serialize::impl_to_le!(
    SerializableSegmentedExecutionTimes,
    io_read_nanos,
    io_write_nanos,
    xor_nanos,
    delta_nanos,
    quantization_nanos,
    bit_packing_nanos
);
impl FromLeBytes for SerializableSegmentedExecutionTimes {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        let io_read_nanos = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
        let io_write_nanos = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
        let xor_nanos = u128::from_le_bytes(bytes[32..48].try_into().unwrap());
        let delta_nanos = u128::from_le_bytes(bytes[48..64].try_into().unwrap());
        let quantization_nanos = u128::from_le_bytes(bytes[64..80].try_into().unwrap());
        let bit_packing_nanos = u128::from_le_bytes(bytes[80..96].try_into().unwrap());

        Self {
            io_read_nanos,
            io_write_nanos,
            xor_nanos,
            delta_nanos,
            quantization_nanos,
            bit_packing_nanos,
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
        }
    }
}

impl From<SerializableSegmentedExecutionTimes> for SegmentedExecutionTimes {
    fn from(times: SerializableSegmentedExecutionTimes) -> Self {
        let mut elapsed_times = [Duration::default(); N_SEGMENT_KINDS];

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

        Self { elapsed_times }
    }
}

impl Default for SegmentedExecutionTimes {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentedExecutionTimes {
    pub fn new() -> Self {
        Self {
            elapsed_times: [Duration::from_secs(0); N_SEGMENT_KINDS],
        }
    }

    pub fn start_measurement(&mut self, kind: SegmentKind) -> SegmentTimer {
        SegmentTimer {
            times: self,
            is_addition: false,
            start: Instant::now(),
            kind,
        }
    }

    pub fn start_addition_measurement(&mut self, kind: SegmentKind) -> SegmentTimer {
        SegmentTimer {
            times: self,
            is_addition: true,
            start: Instant::now(),
            kind,
        }
    }

    fn update_at(&mut self, kind: SegmentKind, value: Duration) {
        self.elapsed_times[kind.index()] = value;
    }

    fn add_at(&mut self, kind: SegmentKind, value: Duration) {
        let elapsed_time = &mut self.elapsed_times[kind.index()];

        *elapsed_time += value;
    }

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
}
