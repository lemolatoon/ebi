use super::GenericChunkHeader;

/// A trait for converting the elements of a struct into little-endian byte order.
pub trait ToLe {
    /// Converts the elements of the struct into little-endian byte order.
    fn to_le(&mut self) -> &mut Self;
}

impl ToLe for GenericChunkHeader {
    fn to_le(&mut self) -> &mut Self {
        self
    }
}

/// A trait for obtaining a byte slice representation of a struct instance.
pub trait AsBytes: private::Sealed {
    /// Returns a byte slice representation of the struct instance.
    ///
    /// # Panics
    ///
    /// Panics if the struct's alignment does not match the expected alignment for its fields.
    fn as_bytes(&self) -> &[u8];
}

mod private {
    pub(super) trait Sealed {}
    impl Sealed for super::GenericChunkHeader {}
}

impl<T: private::Sealed> AsBytes for T {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const T as *const u8, std::mem::size_of::<T>())
        }
    }
}
