pub trait AsBytesMut {
    /// Returns a mutable byte slice representation of the struct instance.
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

impl<T> AsBytesMut for [T] {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // Safety:
        // This range is valid by the end of the lifetime of the `self`.
        // Any bit pattern is valid for `u8`.
        unsafe {
            let ptr = self.as_mut_ptr().cast::<u8>();
            std::slice::from_raw_parts_mut(ptr, std::mem::size_of_val(self))
        }
    }
}
