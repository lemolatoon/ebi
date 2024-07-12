//! Structs for interpreting file formats that ebi uses.
//! Structs below is *pseudo* structs for understanding the format.
//! Code blocks with syntax highlighting are valid as Rust's code, and defined in this module.
//! All bytes are laid as little-endian.
//!
//! ```text
//! // ================= File =====================
//! #[repr(packed(1))]
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
//! #[repr(packed(1))]
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
//! #[repr(packed(1))]
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
//! #[repr(packed(1))]
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
//! #[repr(packed(1))]
//! pub struct Chunk<T> {
//!     pub chunk_header: ChunkHeader<T>,
//!     pub data: [u8], // ?Sized
//! }
//!
//! #[repr(packed(1))]
//! pub struct ChunkHeader<T> {
//!     /// unaligned
//!     ///
//!     /// The number of bytes of this chunk, including this `n_bytes` field
//!     pub n_bytes: u64,
//!     pub header: T,
//! }
//!
//!
//! // ============ FileFooter ==================
//!
//! #[repr(packed(1))]
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
//! #[repr(packed(1))]
//! pub struct FileFooter1 {
//!     pub chunk_footers: [ChunkFooter], // ?Sized
//! }
//!
//! #[repr(packed(1))]
//! pub struct FileFooter2 {
//!     pub crc: u32,
//! }
//!
//!
//! #[repr(packed(1))]
//! struct ChunkFooter {
//!     /// The byte offset of this chunk.
//!     pub physical_offset: u64,
//!     /// The number of tuples laid before this chunk.
//!     pub logical_offset: u64,
//!     // zone_map: ZoneMap
//! }
//! ```
pub mod deserialize;
// ============== FileHeader ==================

#[repr(packed(1))]
pub struct FileHeader {
    /// EBI1
    pub magic_number: [u8; 4],
    /// unaligned
    ///
    /// major, minor, patch
    pub version: [u16; 3],
    /// unaligned
    ///
    /// Bytes offset for FileFooter from the file start.
    pub footer_offset: u64,
    pub config: FileConfig,
}

#[repr(packed(1))]
pub struct FileConfig {
    pub field_type: FieldType, // = u8
    pub chunk_option: ChunkOption,
    pub compression_scheme: CompressionScheme, // = u8
}

#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FieldType {
    F64 = 0,
}

#[repr(packed(1))]
pub struct ChunkOption {
    pub kind: ChunkOptionKind, // u8
    /// unaligned
    pub value: u64,
}

#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkOptionKind {
    Full,
    RecordCount,
    ByteSize,
}

#[repr(u8)]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompressionScheme {
    BUFF = 0,
    Gorilla,
    Uncompressed,
}

// =============== Chunk ===================
#[repr(packed(1))]
pub struct Chunk<T> {
    pub chunk_header: ChunkHeader<T>,
    pub data: [u8], // ?Sized
}

#[repr(packed(1))]
pub struct ChunkHeader<T> {
    /// unaligned
    ///
    /// The number of bytes of this chunk, including this `n_bytes` field
    pub n_bytes: u64,
    pub header: T,
}

// ============ FileFooter ==================

#[repr(packed(1))]
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

#[repr(packed(1))]
pub struct FileFooter1 {
    pub chunk_footers: [ChunkFooter], // ?Sized
}

#[repr(packed(1))]
pub struct FileFooter2 {
    pub crc: u32,
}

#[repr(packed(1))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkFooter {
    /// The byte offset of this chunk.
    pub physical_offset: u64,
    /// The number of tuples laid before this chunk.
    pub logical_offset: u64,
    // zone_map: ZoneMap
}

/// A trait for obtaining a byte slice representation of a struct instance in little-endian byte order.
/// This trait should be implemented for structs that need to be represented as byte slices.
pub trait AsLeBytes {
    /// Returns a byte slice representation of the struct instance in little-endian byte order.
    ///
    /// # Panics
    ///
    /// Panics if the struct's alignment does not match the expected alignment for its fields.
    fn as_le_bytes(&self) -> &[u8];
}
