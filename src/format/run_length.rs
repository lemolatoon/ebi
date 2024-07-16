//! Run Length Compression Scheme Header format
//! ```rust
//! #[repr(packed(1))]
//! pub struct RunLengthHeader {
//!     /// number of fields in the chunk
//!     number_of_fields: u8,
//! }
//! ```

use super::{deserialize::FromLeBytes, serialize::ToLe};

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(packed(1))]
pub struct RunLengthHeader {
    /// number of fields in the chunk
    number_of_fields: u64,
}

impl ToLe for RunLengthHeader {
    fn to_le(&mut self) -> &mut Self {
        self.number_of_fields = self.number_of_fields.to_le();

        self
    }
}

impl FromLeBytes for RunLengthHeader {
    fn from_le_bytes(buf: &[u8]) -> Self {
        let number_of_fields = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        Self { number_of_fields }
    }
}

impl RunLengthHeader {
    pub fn number_of_fields(&self) -> u64 {
        self.number_of_fields
    }

    pub fn set_number_of_fields(&mut self, number_of_fields: u64) {
        self.number_of_fields = number_of_fields;
    }
}
