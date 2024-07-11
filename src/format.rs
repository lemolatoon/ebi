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

// ============== FileHeader ==================

use std::mem::size_of;

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

impl ChunkFooter {
    #[must_use]
    pub fn to_le(mut self) -> Self {
        self.physical_offset = self.physical_offset.to_le();
        self.logical_offset = self.logical_offset.to_le();

        self
    }
}

/// A trait for converting a byte slice into a struct instance, assuming little-endian byte order.
/// This trait should be implemented for structs that need to be constructed from byte slices.
pub trait FromLeBytes: Sized {
    /// Constructs a struct instance from a byte slice in little-endian byte order.
    ///
    /// # Panics
    ///
    /// Panics if the byte slice is smaller than the size of the struct.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

/// A trait for attempting to convert a byte slice into a struct instance, assuming little-endian byte order.
/// This trait should be implemented for structs that can potentially be constructed from byte slices.
pub trait TryFromLeBytes {
    type Error;
    /// Attempts to construct a struct instance from a byte slice in little-endian byte order.
    ///
    /// # Returns
    ///
    /// Returns `Some(instance)` of the struct if the byte slice is of adequate size and the conversion is successful,
    /// otherwise returns `None`.
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConversionError {
    FieldType,
    CompressionScheme,
}

impl<T: FromLeBytes + Sized> TryFromLeBytes for T {
    type Error = ();
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self::from_le_bytes(bytes))
    }
}

impl TryFromLeBytes for FileHeader {
    type Error = ConversionError;
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        let mut buf_cursor = 0;
        let magic_number = [
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
        ];
        buf_cursor += size_of::<[u8; 4]>();
        let version = [
            u16::from_le_bytes([bytes[buf_cursor], bytes[buf_cursor + 1]]),
            u16::from_le_bytes([bytes[buf_cursor + 2], bytes[buf_cursor + 3]]),
            u16::from_le_bytes([bytes[buf_cursor + 4], bytes[buf_cursor + 5]]),
        ];
        buf_cursor += size_of::<[u16; 3]>();
        let footer_offset = u64::from_le_bytes([
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
            bytes[buf_cursor + 4],
            bytes[buf_cursor + 5],
            bytes[buf_cursor + 6],
            bytes[buf_cursor + 7],
        ]);
        buf_cursor += size_of::<u64>();
        let config = FileConfig::try_from_le_bytes(&bytes[buf_cursor..])?;

        Ok(Self {
            magic_number,
            version,
            footer_offset,
            config,
        })
    }
}

impl TryFromLeBytes for FileConfig {
    type Error = ConversionError;
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        let mut buf_cursor = 0;
        let field_type = FieldType::try_from_le_bytes(&bytes[buf_cursor..])?;
        buf_cursor += size_of::<FieldType>();
        let chunk_option = ChunkOption::try_from_le_bytes(&bytes[buf_cursor..])?;
        buf_cursor += size_of::<ChunkOption>();
        let compression_scheme = CompressionScheme::try_from_le_bytes(&bytes[buf_cursor..])?;

        Ok(Self {
            field_type,
            chunk_option,
            compression_scheme,
        })
    }
}

impl TryFromLeBytes for FieldType {
    type Error = ConversionError;
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        const F64: u8 = FieldType::F64 as u8;
        match bytes[0] {
            F64 => Ok(Self::F64),
            _ => Err(ConversionError::FieldType),
        }
    }
}

impl TryFromLeBytes for CompressionScheme {
    type Error = ConversionError;
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        match bytes[0] {
            0 => Ok(Self::BUFF),
            1 => Ok(Self::Gorilla),
            2 => Ok(Self::Uncompressed),
            _ => Err(ConversionError::CompressionScheme),
        }
    }
}

impl TryFromLeBytes for ChunkOption {
    type Error = ConversionError;
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        let mut buf_cursor = 0;
        let kind = ChunkOptionKind::try_from_le_bytes(&bytes[buf_cursor..])?;
        buf_cursor += size_of::<ChunkOptionKind>();

        let value = u64::from_le_bytes([
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
            bytes[buf_cursor + 4],
            bytes[buf_cursor + 5],
            bytes[buf_cursor + 6],
            bytes[buf_cursor + 7],
        ]);

        Ok(Self { kind, value })
    }
}

impl TryFromLeBytes for ChunkOptionKind {
    type Error = ConversionError;
    fn try_from_le_bytes(bytes: &[u8]) -> Result<Self, Self::Error> {
        const FULL: u8 = ChunkOptionKind::Full as u8;
        const RECORD_COUNT: u8 = ChunkOptionKind::RecordCount as u8;
        const BYTE_SIZE: u8 = ChunkOptionKind::ByteSize as u8;
        match bytes[0] {
            FULL => Ok(Self::Full),
            RECORD_COUNT => Ok(Self::RecordCount),
            BYTE_SIZE => Ok(Self::ByteSize),
            _ => Err(ConversionError::CompressionScheme),
        }
    }
}

impl FromLeBytes for FileFooter0 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut buf_cursor = 0;
        let number_of_records = u64::from_le_bytes([
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
            bytes[buf_cursor + 4],
            bytes[buf_cursor + 5],
            bytes[buf_cursor + 6],
            bytes[buf_cursor + 7],
        ]);
        buf_cursor += size_of::<u64>();
        let number_of_chunks = u64::from_le_bytes([
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
            bytes[buf_cursor + 4],
            bytes[buf_cursor + 5],
            bytes[buf_cursor + 6],
            bytes[buf_cursor + 7],
        ]);

        Self {
            number_of_records,
            number_of_chunks,
        }
    }
}

impl FromLeBytes for FileFooter2 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        let crc = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        Self { crc }
    }
}

impl FromLeBytes for ChunkFooter {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut buf_cursor = 0;
        let physical_offset = u64::from_le_bytes([
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
            bytes[buf_cursor + 4],
            bytes[buf_cursor + 5],
            bytes[buf_cursor + 6],
            bytes[buf_cursor + 7],
        ]);
        buf_cursor += size_of::<u64>();
        let logical_offset = u64::from_le_bytes([
            bytes[buf_cursor],
            bytes[buf_cursor + 1],
            bytes[buf_cursor + 2],
            bytes[buf_cursor + 3],
            bytes[buf_cursor + 4],
            bytes[buf_cursor + 5],
            bytes[buf_cursor + 6],
            bytes[buf_cursor + 7],
        ]);

        Self {
            physical_offset,
            logical_offset,
        }
    }
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
