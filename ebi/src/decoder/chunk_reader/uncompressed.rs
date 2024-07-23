use std::mem::align_of;

use crate::{
    decoder::{FileMetadataLike, GeneralChunkHandle},
    format::{
        deserialize::FromLeBytes,
        uncompressed::{NativeUncompressedHeader, UncompressedHeader0},
    },
};

use super::Reader;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UncompressedReader<'chunk> {
    header: NativeUncompressedHeader<'chunk>,
    chunk: &'chunk [u8],
}

impl<'chunk> UncompressedReader<'chunk> {
    /// Create a new UncompressedReader.
    /// Caller must guarantee that the input chunk is valid for Uncompressed Chunk.
    pub fn new<T: FileMetadataLike>(handle: &GeneralChunkHandle<T>, chunk: &'chunk [u8]) -> Self {
        let chunk_size = handle.chunk_size() as usize;
        let header_head = UncompressedHeader0::from_le_bytes(chunk);
        let header = NativeUncompressedHeader::new(header_head, chunk);
        let header_size = *header.header_size() as usize;
        let data_offset = header_size.next_multiple_of(align_of::<f64>());
        let chunk = &chunk[..(data_offset - header_size + chunk_size)];
        Self { header, chunk }
    }
}

impl<'a> Reader for UncompressedReader<'a> {
    type NativeHeader = NativeUncompressedHeader<'a>;

    fn decompress(&mut self) -> &'a [f64] {
        let header_size = *self.header.header_size() as usize;
        let data = &self.chunk[header_size.next_multiple_of(align_of::<f64>())..];
        let len = data.len() / std::mem::size_of::<f64>();

        // Safety:
        // `data` is valid for `len * size_of::<f64>()` -> `size_of::<f64> * len`.
        // `data` is non null because it is originated from slice.
        // `data`'s lifetime is the same as `self.chunk`.
        // `data` is properly aligned for f64.
        // Consecutive `len` f64 memory is properly initialized, the user of this struct guarantees this.
        // `len * size_of::<f64>()` -> `size_of::<f64> * len` is no larger than `isize::MAX`
        debug_assert!(std::mem::size_of::<f64>() * len <= isize::MAX as usize);
        debug_assert!(
            data.as_ptr().cast::<f64>().is_aligned(),
            "data is not aligned: {:p}, f64 alignment: {}",
            data.as_ptr(),
            std::mem::align_of::<f64>(),
        );
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f64>(), len) }
    }

    fn header_size(&self) -> usize {
        *self.header.header_size() as usize
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &self.header
    }
}
