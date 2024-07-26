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

use derive_getters::Getters;

use super::deserialize::{self, FromLeBytes};

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

deserialize::impl_from_le_bytes_ext!(UncompressedHeader0);

#[repr(C, packed(1))]
pub struct UncompressedHeader1 {
    pub header: [u8],
}

#[derive(Getters, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeUncompressedHeader {
    header_size: u8,
    header: Box<[u8]>,
}

impl NativeUncompressedHeader {
    pub fn new(header_head: UncompressedHeader0, header: Box<[u8]>) -> Self {
        let header_size = header_head.header_size;
        Self {
            header_size,
            header,
        }
    }
}
