//! Uncompressed Compression Scheme Header format
//! ```rust
//! #[repr(C, packed(1))]
//! pub struct UncompressedHeader0 {
//!     header_size: u8,
//! }
//!
//! #[repr(C, packed(1))]
//! pub struct UncompressedHeader1 {
//!     header: [u8],
//! }
//! ```

use std::mem::size_of;

use derive_getters::Getters;

use super::deserialize::FromLeBytes;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C, packed(1))]
pub struct UncompressedHeader0 {
    pub header_size: u8,
}

impl FromLeBytes for UncompressedHeader0 {
    fn from_le_bytes(buf: &[u8]) -> Self {
        let header_size = buf[0];
        Self { header_size }
    }
}

#[repr(C, packed(1))]
pub struct UncompressedHeader1 {
    pub header: [u8],
}

#[derive(Getters, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeUncompressedHeader<'a> {
    header_size: u8,
    header: &'a [u8],
}

impl<'a> NativeUncompressedHeader<'a> {
    pub fn new(header_head: UncompressedHeader0, buf: &'a [u8]) -> Self {
        let header_size = header_head.header_size;
        let header = &buf[size_of::<UncompressedHeader0>()..header_size as usize];
        Self {
            header_size,
            header,
        }
    }
}
