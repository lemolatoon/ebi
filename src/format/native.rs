//! Native struct for the file format.
//! This module contains the native structs(=[repr(Rust)]).

use derive_getters::Getters;

use crate::encoder::ChunkOption;

use super::{
    ChunkFooter, CompressionScheme, FieldType, FileConfig, FileFooter0, FileFooter2, FileHeader,
};

#[derive(Getters, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeFileHeader {
    /// EBI1
    magic_number: [u8; 4],
    /// major, minor, patch
    version: [u16; 3],
    /// Bytes offset for FileFooter from the file start.
    footer_offset: u64,
    config: NativeFileConfig,
}

impl From<&FileHeader> for NativeFileHeader {
    fn from(header: &FileHeader) -> Self {
        let config = NativeFileConfig::from(&header.config);
        Self {
            magic_number: header.magic_number,
            version: header.version,
            footer_offset: header.footer_offset,
            config,
        }
    }
}

#[derive(Getters, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeFileConfig {
    pub field_type: FieldType, // = u8
    pub chunk_option: ChunkOption,
    pub compression_scheme: CompressionScheme, // = u8
}

impl From<&FileConfig> for NativeFileConfig {
    fn from(config: &FileConfig) -> Self {
        let value = config.chunk_option.value as usize;
        let chunk_option = match config.chunk_option.kind {
            super::ChunkOptionKind::Full => ChunkOption::Full,
            super::ChunkOptionKind::RecordCount => ChunkOption::RecordCount(value),
            super::ChunkOptionKind::ByteSize => ChunkOption::ByteSize(value),
        };
        let field_type = config.field_type;
        let compression_scheme = config.compression_scheme;
        Self {
            field_type,
            chunk_option,
            compression_scheme,
        }
    }
}

#[derive(Getters, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeFileFooter {
    /// The number of records included in this file.
    number_of_records: u64,
    /// The number of chunks included in this file.
    number_of_chunks: u64,
    chunk_footers: Vec<NativeChunkFooter>,
    crc: u32,
}

impl NativeFileFooter {
    pub fn new(
        footer0: &FileFooter0,
        chunk_footers: &[ChunkFooter],
        footer2: &FileFooter2,
    ) -> Self {
        let number_of_records = footer0.number_of_records;
        let number_of_chunks = footer0.number_of_chunks;
        let chunk_footers = chunk_footers.iter().map(NativeChunkFooter::from).collect();
        let crc = footer2.crc;
        Self {
            number_of_records,
            number_of_chunks,
            chunk_footers,
            crc,
        }
    }
}

#[derive(Getters, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeChunkFooter {
    /// The byte offset of this chunk.
    physical_offset: u64,
    /// The number of tuples laid before this chunk.
    logical_offset: u64,
    // zone_map: ZoneMap
}

impl From<&ChunkFooter> for NativeChunkFooter {
    fn from(footer: &ChunkFooter) -> Self {
        Self {
            physical_offset: footer.physical_offset,
            logical_offset: footer.logical_offset,
        }
    }
}
