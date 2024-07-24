use std::{
    io::{self, Read},
    mem::size_of,
};

use thiserror::Error;

use crate::decoder;

use super::{
    ChunkFooter, ChunkOption, ChunkOptionKind, CompressionScheme, FieldType, FileConfig,
    FileFooter0, FileFooter2, FileHeader,
};

/// A trait for converting a byte slice into a struct instance, assuming little-endian byte order.
/// This trait should be implemented for structs that need to be constructed from byte slices.
pub trait FromLeBytes: Sized {
    const SIZE: usize = size_of::<Self>();
    /// Constructs a struct instance from a byte slice in little-endian byte order.
    ///
    /// # Panics
    ///
    /// Panics if the byte slice is smaller than the size of the struct.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

pub trait FromLeBytesExt: FromLeBytes {
    /// Constructs a struct instance from the [`Read`] trait object in little-endian byte order.
    /// This function reads the byte slice from the reader and constructs a struct instance.
    /// The byte slice is read from the reader and passed to the `from_le_bytes` function.
    fn from_le_bytes_by_reader<R: std::io::Read>(reader: &mut R) -> io::Result<Self>;
}

macro_rules! impl_from_le_bytes_ext {
    ($($t:ty),*) => {
        $(
            impl crate::format::deserialize::FromLeBytesExt for $t {
                fn from_le_bytes_by_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                    let mut buf = [0u8; std::mem::size_of::<$t>()];
                    reader.read_exact(&mut buf)?;
                    Ok(<$t>::from_le_bytes(&buf))
                }
            }
        )*
    };
}

pub(crate) use impl_from_le_bytes_ext;

impl_from_le_bytes_ext!(ChunkFooter, FileFooter0, FileFooter2);

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

/// An extension trait for `TryFromLeBytes` that provides a method to read from a reader and attempt to construct an instance.
pub trait TryFromLeBytesExt {
    fn try_from_le_bytes_by_reader<R: Read>(reader: &mut R) -> decoder::Result<Self>
    where
        Self: Sized + TryFromLeBytes;
}

macro_rules! impl_try_from_le_bytes_ext {
    ($($t:ty),*) => {
        $(
            impl TryFromLeBytesExt for $t {
                fn try_from_le_bytes_by_reader<R: Read>(reader: &mut R) -> crate::decoder::Result<Self> {
                    let mut buf = vec![0u8; size_of::<$t>()];
                    reader.read_exact(&mut buf)?;
                    let constructed = <$t>::try_from_le_bytes(&buf)?;

                    Ok(constructed)
                }
            }
        )*
    };
}
pub(crate) use impl_try_from_le_bytes_ext;

impl_try_from_le_bytes_ext!(
    FileHeader,
    FileConfig,
    FieldType,
    CompressionScheme,
    ChunkOption,
    ChunkOptionKind
);

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

        let config = FileConfig::try_from_le_bytes(&bytes[buf_cursor..])?;
        buf_cursor += size_of::<FileConfig>();

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

        let version = [
            u16::from_le_bytes([bytes[buf_cursor], bytes[buf_cursor + 1]]),
            u16::from_le_bytes([bytes[buf_cursor + 2], bytes[buf_cursor + 3]]),
            u16::from_le_bytes([bytes[buf_cursor + 4], bytes[buf_cursor + 5]]),
        ];

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

        let compression_scheme = CompressionScheme::try_from_le_bytes(&bytes[buf_cursor..])?;
        buf_cursor += size_of::<CompressionScheme>();

        let chunk_option = ChunkOption::try_from_le_bytes(&bytes[buf_cursor..])?;

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
        const UNCOMPRESSED: u8 = CompressionScheme::Uncompressed as u8;
        const RLE: u8 = CompressionScheme::RLE as u8;
        const GORILLA: u8 = CompressionScheme::Gorilla as u8;
        const BUFF: u8 = CompressionScheme::BUFF as u8;

        match bytes[0] {
            UNCOMPRESSED => Ok(Self::Uncompressed),
            RLE => Ok(Self::RLE),
            GORILLA => Ok(Self::Gorilla),
            BUFF => Ok(Self::BUFF),
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

        // Skip reserved field
        buf_cursor += 1;

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

        Ok(Self {
            kind,
            reserved: 0,
            value,
        })
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
        let compression_elapsed_time_nano_secs = u128::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let crc = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        Self {
            compression_elapsed_time_nano_secs,
            crc,
        }
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
