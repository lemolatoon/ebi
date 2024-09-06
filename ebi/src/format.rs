//! Structs for interpreting file formats that ebi uses.
//! Structs below is *pseudo* structs for understanding the format.
//! Code blocks with syntax highlighting are valid as Rust's code, and defined in this module.
//! All bytes are laid as little-endian.
//!
//! ```text
//! // ================= File =====================
//! #[repr(C, packed(1))]
//! struct File {
//!     header: FileHeader, //  Sized
//!     chunks: [Chunk],    // ?Sized
//!     footer: FileFooter, // ?Sized
//! }
//! ```
//! ```rust
//!
//! // ============== FileHeader ==================
//!
//! #[repr(C, packed(1))]
//! pub struct FileHeader {
//!     /// EBI1
//!     pub magic_number: [u8; 4],
//!     /// unaligned
//!     ///
//!     /// major, minor, patch
//!     pub version: [u16; 3],
//!     /// unaligned
//!     pub footer_offset: u64,
//!     pub config: FileConfig,
//! }
//!
//! #[repr(C, packed(1))]
//! struct FileConfig {
//!     pub field_type: FieldType, // = u8
//!     pub chunk_option: ChunkOption,
//!     pub compression_scheme: CompressionScheme, // = u8
//! }
//!
//! #[repr(u8)]
//! enum FieldType {
//!     F64 = 0,
//! }
//!
//! #[repr(C, packed(1))]
//! struct ChunkOption {
//!     pub kind: ChunkOptionKind, // u8
//!     /// unaligned
//!     pub value: u64,
//! }
//!
//! #[repr(u8)]
//! enum ChunkOptionKind {
//!     Full,
//!     RecordCount,
//!     ByteSize,
//! }
//!
//! #[repr(u8)]
//! #[non_exhaustive]
//! #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
//! pub enum CompressionScheme {
//!     BUFF = 0,
//!     Gorilla,
//!     Uncompressed,
//! }
//!
//! // =============== Chunk ===================
//! #[repr(C, packed(1))]
//! pub struct Chunk<T> {
//!     pub chunk_header: ChunkHeader<T>,
//!     pub data: [u8], // ?Sized
//! }
//!
//! #[repr(C, packed(1))]
//! pub struct GeneralChunkHeader {
//!     /// The number of bytes of this chunk, including this `number_of_bytes` field
//!     pub number_of_bytes: u64,
//! }
//!
//! #[repr(C, packed(1))]
//! pub struct ChunkHeader<T> {
//!     pub n_bytes: GeneralChunkHeader,
//!     pub header: T,
//! }
//!
//!
//! // ============ FileFooter ==================
//!
//! #[repr(C, packed(1))]
//! pub struct FileFooter0 {
//!     /// unaligned
//!     ///
//!     /// The number of records included in this file.
//!     pub number_of_records: u64,
//!     /// unaligned
//!     ///
//!     /// The number of chunks included in this file.
//!     pub number_of_chunks: u64,
//! }
//!
//! #[repr(C, packed(1))]
//! pub struct FileFooter1 {
//!     pub chunk_footers: [ChunkFooter], // ?Sized
//! }
//! #[repr(C, packed(1))]
//! pub struct FileFooter2 {
//!     pub config_size: u8,
//!     compressor_config: [u8],
//! }
//!
//! use ebi::time::SerializableSegmentedExecutionTimes;
//! #[repr(C, packed(1))]
//! pub struct FileFooter3 {
//!     /// Compression Elapsed Time (nano seconds)
//!     pub compression_elapsed_time_nano_secs: u128,
//!     pub execution_elapsed_times_nano_secs: SerializableSegmentedExecutionTimes,
//!     pub crc: u32,
//! }
//!
//!
//! #[repr(C, packed(1))]
//! struct ChunkFooter {
//!     /// The byte offset of this chunk.
//!     pub physical_offset: u64,
//!     /// The number of tuples laid before this chunk.
//!     pub logical_offset: u64,
//!     // zone_map: ZoneMap
//! }
//! ```

use num_enum::TryFromPrimitive;

use crate::time::SerializableSegmentedExecutionTimes;

pub mod deserialize;
pub mod native;
pub mod serialize;

pub mod run_length;
// ============== FileHeader ==================

#[repr(C, packed(1))]
pub struct FileHeader {
    /// EBI1
    pub magic_number: [u8; 4],
    pub config: FileConfig,
    /// Bytes offset for FileFooter from the file start.
    pub footer_offset: u64,
    /// major, minor, patch
    pub version: [u16; 3],
}

#[repr(C, packed(1))]
pub struct FileConfig {
    pub field_type: FieldType,                 // = u8
    pub compression_scheme: CompressionScheme, // = u8
    pub chunk_option: ChunkOption,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FieldType {
    F64 = 0,
}

#[repr(C, packed(1))]
pub struct ChunkOption {
    pub kind: ChunkOptionKind, // u8
    pub reserved: u8,
    pub value: u64,
}

#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkOptionKind {
    Full,
    RecordCount,
    ByteSize,
}

#[derive(TryFromPrimitive, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum CompressionScheme {
    Uncompressed = 0,
    RLE = 1,
    BUFF = 2,
    Gorilla = 3,
    Chimp = 4,
    Chimp128 = 5,
    ElfOnChimp = 6,
    Elf = 7,
    DeltaSprintz = 8,
    Zstd = 9,
    Gzip = 10,
    Snappy = 11,
    FFIAlp = 12,
}

// =============== Chunk ===================
#[repr(C, packed(1))]
pub struct Chunk<T> {
    pub chunk_header: ChunkHeader<T>,
    pub data: [u8], // ?Sized
}

#[repr(C, packed(1))]
pub struct GeneralChunkHeader {}

#[repr(C, packed(1))]
pub struct ChunkHeader<T> {
    pub generic_header: GeneralChunkHeader,
    pub header: T,
}

// ============ FileFooter ==================

#[repr(C, packed(1))]
pub struct FileFooter0 {
    /// unaligned
    ///
    /// The number of records included in this file.
    pub number_of_records: u64,
    /// unaligned
    ///
    /// The number of chunks included in this file.
    pub number_of_chunks: u64,
}

#[repr(C, packed(1))]
pub struct FileFooter1 {
    pub chunk_footers: [ChunkFooter], // ?Sized
}

#[repr(C, packed(1))]
pub struct FileFooter2 {
    pub config_size: u8,
    compressor_config: [u8],
}

#[repr(C, packed(1))]
pub struct FileFooter3 {
    /// Compression Elapsed Time (nano seconds)
    pub compression_elapsed_time_nano_secs: u128,
    pub execution_elapsed_times_nano_secs: SerializableSegmentedExecutionTimes,
    pub crc: u32,
}

#[repr(C, packed(1))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkFooter {
    /// The byte offset of this chunk.
    pub physical_offset: u64,
    /// The number of tuples laid before this chunk.
    pub logical_offset: u64,
    // zone_map: ZoneMap
}
