use super::{
    run_length::RunLengthHeader, ChunkFooter, FileConfig, FileFooter0, FileFooter3, FileHeader,
    GeneralChunkHeader,
};

/// A trait for converting the elements of a struct into little-endian byte order.
pub trait ToLe {
    /// Converts the elements of the struct into little-endian byte order.
    fn to_le(&mut self) -> &mut Self;
}

impl ToLe for FileHeader {
    fn to_le(&mut self) -> &mut Self {
        self.magic_number = self.magic_number.map(|x| x.to_le());
        self.version = self.version.map(|x| x.to_le());
        self.footer_offset = self.footer_offset.to_le();
        self.config.to_le();

        self
    }
}

impl ToLe for FileConfig {
    fn to_le(&mut self) -> &mut Self {
        self.chunk_option.value = self.chunk_option.value.to_le();

        self
    }
}

impl ToLe for GeneralChunkHeader {
    fn to_le(&mut self) -> &mut Self {
        self
    }
}

impl ToLe for FileFooter0 {
    fn to_le(&mut self) -> &mut Self {
        self.number_of_records = self.number_of_records.to_le();
        self.number_of_chunks = self.number_of_chunks.to_le();

        self
    }
}

impl ToLe for FileFooter3 {
    fn to_le(&mut self) -> &mut Self {
        self.crc = self.crc.to_le();

        self
    }
}

impl ToLe for ChunkFooter {
    fn to_le(&mut self) -> &mut Self {
        self.physical_offset = self.physical_offset.to_le();
        self.logical_offset = self.logical_offset.to_le();

        self
    }
}

macro_rules! impl_to_le {
    ($struct_name:ident, $( $field:ident ),* ) => {
        impl $crate::format::serialize::ToLe for $struct_name {
            fn to_le(&mut self) -> &mut Self {
                $(
                    let $field = self.$field;
                    self.$field = $field.to_le();
                )*
                self
            }
        }
    };
    (< $( $declare:tt ),* >, $struct_name:ident< $( $gen:ident ),* >, $( $field:ident ),* ) => {
        impl< $( $declare ),* > ToLe for $struct_name< $( $gen ),* > {
            fn to_le(&mut self) -> &mut Self {
                $(
                    self.$field.to_le();
                )*
                self
            }
        }
    };
}
pub(crate) use impl_to_le;

/// A trait for obtaining a byte slice representation of a struct instance.
pub trait AsBytes {
    /// Returns a byte slice representation of the struct instance.
    fn as_bytes(&self) -> &[u8];
}

mod private {
    use crate::compressor::{
        buff::BUFFCompressorConfig, chimp_n::Chimp128CompressorConfig,
        general_xor::PackedGeneralXorCompressorConfig, gorilla::GorillaCompressorConfig,
        run_length::RunLengthCompressorConfig, sprintz::DeltaSprintzCompressorConfig,
        uncompressed::UncompressedCompressorConfig, zstd::ZstdCompressorConfig,
    };

    pub trait Sealed {}
    impl Sealed for super::FileHeader {}
    impl Sealed for super::FileConfig {}
    impl Sealed for super::GeneralChunkHeader {}
    impl Sealed for super::RunLengthHeader {}
    impl Sealed for super::FileFooter0 {}
    impl Sealed for super::FileFooter3 {}
    impl Sealed for super::ChunkFooter {}

    // CompressorConfigs
    impl Sealed for UncompressedCompressorConfig {}
    impl Sealed for RunLengthCompressorConfig {}
    impl Sealed for BUFFCompressorConfig {}
    impl Sealed for PackedGeneralXorCompressorConfig {}
    impl Sealed for Chimp128CompressorConfig {}
    impl Sealed for GorillaCompressorConfig {}
    impl Sealed for DeltaSprintzCompressorConfig {}
    impl Sealed for ZstdCompressorConfig {}
}

impl<T: private::Sealed> AsBytes for T {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const T as *const u8, std::mem::size_of::<T>())
        }
    }
}

impl<T: private::Sealed> AsBytes for [T] {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr() as *const u8, std::mem::size_of_val(self))
        }
    }
}
