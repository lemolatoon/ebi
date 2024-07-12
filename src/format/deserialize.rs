use std::mem::size_of;

use thiserror::Error;

use super::{
    ChunkFooter, ChunkOption, ChunkOptionKind, CompressionScheme, FieldType, FileConfig,
    FileFooter0, FileFooter2, FileHeader,
};

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

#[derive(Error, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConversionError {
    #[error("Failed to convert FieldType")]
    FieldType,
    #[error("Failed to convert CompressionScheme")]
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
