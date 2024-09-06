use crate::{
    format::deserialize,
    io::bit_write::{BitWrite, BufferedBitWriter},
};

use super::{
    general_xor::{
        GeneralXorCompressor, GeneralXorCompressorConfig, GeneralXorCompressorConfigBuilder,
        XorEncoder,
    },
    Capacity,
};

pub type GorillaCompressor =
    GeneralXorCompressor<BufferedBitWriter, modified_tsz::GorillaFloatEncoder>;
pub type GorillaCompressorConfig = GeneralXorCompressorConfig<modified_tsz::GorillaFloatEncoder>;
pub type GorillaCompressorConfigBuilder =
    GeneralXorCompressorConfigBuilder<modified_tsz::GorillaFloatEncoder>;

deserialize::impl_from_le_bytes!(GorillaCompressorConfig, gorilla, (capacity, Capacity));

impl XorEncoder for modified_tsz::GorillaFloatEncoder {
    fn compress_float<W: BitWrite>(&mut self, mut w: W, bits: u64) -> usize {
        self.encode(&mut w, f64::from_bits(bits))
    }

    fn close<W: crate::io::bit_write::BitWrite>(&mut self, mut w: W) {
        self.encode(&mut w, f64::NAN);
        w.write_bit(false);
    }

    fn simulate_close(&self) -> usize {
        self.simulate_encode(f64::NAN.to_bits()) + 1
    }

    fn reset(&mut self) {
        self.reset();
    }
}

pub(crate) mod modified_tsz {
    //! This original implementation is from tsz crate. It is modified to just use the floating point values compression.
    //! Original tsz crate is licensed under MIT.
    //! Original LICENSE is below:
    //!
    //! Copyright (c) 2016 Jerome Froelich
    //! Permission is hereby granted, free of charge, to any person obtaining a copy
    //! of this software and associated documentation files (the "Software"), to deal
    //! in the Software without restriction, including without limitation the rights
    //! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    //! copies of the Software, and to permit persons to whom the Software is
    //! furnished to do so, subject to the following conditions:
    //!
    //! The above copyright notice and this permission notice shall be included in all
    //! copies or substantial portions of the Software.
    //!
    //! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    //! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    //! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    //! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    //! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    //! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    //! SOFTWARE.
    //!

    // END_MARKER relies on the fact that when we encode the delta of delta for a number that requires
    // more than 12 bits we write four control bits 1111 followed by the 32 bits of the value. Since
    // encoding assumes the value is greater than 12 bits, we can store the value 0 to signal the end
    // of the stream

    use crate::io::bit_write::BitWrite;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct GorillaFloatEncoderState {
        /// current float value as bits
        pub value_bits: u64,
        pub leading_zeroes: u32,
        // store the number of leading and trailing zeroes in the current xor as u32 so we
        // don't have to do any conversions after calling `leading_zeros` and `trailing_zeros`
        pub trailing_zeroes: u32,
        // will next DataPoint be the first DataPoint encoded
        pub first: bool,
    }
    impl GorillaFloatEncoderState {
        pub fn new() -> Self {
            Self {
                value_bits: 0,
                leading_zeroes: 64,  // 64 is an initial sentinel value
                trailing_zeroes: 64, // 64 is an initial sentinel value

                first: true,
            }
        }
    }

    /// StdEncoder
    ///
    /// StdEncoder is used to encode `DataPoint`s
    #[derive(Debug, Clone, PartialEq)]
    pub struct GorillaFloatEncoder {
        state: GorillaFloatEncoderState,
    }

    impl Default for GorillaFloatEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GorillaFloatEncoder {
        pub fn new() -> Self {
            Self {
                state: GorillaFloatEncoderState::new(),
            }
        }

        /// Returns the bits that have been written to the buffer
        fn write_first(&mut self, w: &mut impl BitWrite, value_bits: u64) -> usize {
            // store the first value exactly
            w.write_bits(value_bits, 64);
            self.state.value_bits = value_bits;

            self.state.first = true;

            64
        }

        /// Returns the bits that have been written to the buffer
        fn write_next_value(&mut self, w: &mut impl BitWrite, value_bits: u64) -> usize {
            let xor = value_bits ^ self.state.value_bits;
            self.state.value_bits = value_bits;

            let mut bits_buffered = 0;

            if xor == 0 {
                // if xor with previous value is zero just store single zero bit
                w.write_bit(false);
                bits_buffered += 1;
            } else {
                w.write_bit(true);

                bits_buffered += 1;

                let mut leading_zeroes = xor.leading_zeros();
                // we cannot encode leading_zeroes more than 32 bits in 5 bit.
                if leading_zeroes >= 32 {
                    leading_zeroes = 31;
                }
                let trailing_zeroes = xor.trailing_zeros();

                if leading_zeroes >= self.state.leading_zeroes
                    && trailing_zeroes >= self.state.trailing_zeroes
                {
                    // if the number of leading and trailing zeroes in this xor are >= the leading and
                    // trailing zeroes in the previous xor then we only need to store a control bit and
                    // the significant digits of this xor
                    w.write_bit(false);
                    w.write_bits(
                        xor.wrapping_shr(self.state.trailing_zeroes),
                        64 - self.state.leading_zeroes - self.state.trailing_zeroes,
                    );

                    bits_buffered +=
                        (1 + 64 - self.state.leading_zeroes - self.state.trailing_zeroes) as usize;
                } else {
                    // if the number of leading and trailing zeroes in this xor are not less than the
                    // leading and trailing zeroes in the previous xor then we store a control bit and
                    // use 6 bits to store the number of leading zeroes and 6 bits to store the number
                    // of significant digits before storing the significant digits themselves

                    w.write_bit(true);
                    w.write_bits(u64::from(leading_zeroes), 5); // not 6 bit, but 5 bit

                    // if significant_digits is 64 we cannot encode it using 6 bits, however since
                    // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
                    // significant_digits can always be expressed with 6 bits or less
                    let significant_digits = 64 - leading_zeroes - trailing_zeroes;
                    w.write_bits(u64::from(significant_digits - 1), 6);
                    w.write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);

                    bits_buffered += (1 + 5 + 6 + significant_digits) as usize;

                    // finally we need to update the number of leading and trailing zeroes
                    self.state.leading_zeroes = leading_zeroes;
                    self.state.trailing_zeroes = trailing_zeroes;
                }
            }

            bits_buffered
        }
    }

    impl GorillaFloatEncoder {
        /// Returns the bits that have been written to the buffer
        pub fn encode(&mut self, w: &mut impl BitWrite, value: f64) -> usize {
            let value_bits = value.to_bits();

            if self.state.first {
                let n_bits = self.write_first(w, value_bits);
                self.state.first = false;
                n_bits
            } else {
                self.write_next_value(w, value_bits)
            }
        }

        /// Returns the bits that would be written to the buffer if the value was encoded
        pub fn simulate_encode(&self, value_bits: u64) -> usize {
            if self.state.first {
                return 64;
            }

            let xor = value_bits ^ self.state.value_bits;

            let mut bits_buffered = 0;

            if xor == 0 {
                // if xor with previous value is zero just store single zero bit
                bits_buffered += 1;
            } else {
                bits_buffered += 1;

                let mut leading_zeroes = xor.leading_zeros();
                // we cannot encode leading_zeroes more than 32 bits in 5 bit.
                if leading_zeroes >= 32 {
                    leading_zeroes = 31;
                }
                let trailing_zeroes = xor.trailing_zeros();

                if leading_zeroes >= self.state.leading_zeroes
                    && trailing_zeroes >= self.state.trailing_zeroes
                {
                    // if the number of leading and trailing zeroes in this xor are >= the leading and
                    // trailing zeroes in the previous xor then we only need to store a control bit and
                    // the significant digits of this xor

                    bits_buffered +=
                        (1 + 64 - self.state.leading_zeroes - self.state.trailing_zeroes) as usize;
                } else {
                    // if the number of leading and trailing zeroes in this xor are not less than the
                    // leading and trailing zeroes in the previous xor then we store a control bit and
                    // use 6 bits to store the number of leading zeroes and 6 bits to store the number
                    // of significant digits before storing the significant digits themselves

                    // if significant_digits is 64 we cannot encode it using 6 bits, however since
                    // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
                    // significant_digits can always be expressed with 6 bits or less
                    let significant_digits = 64 - leading_zeroes - trailing_zeroes;

                    bits_buffered += (1 + 5 + 6 + significant_digits) as usize;

                    // finally we need to update the number of leading and trailing zeroes
                }
            }

            bits_buffered
        }

        /// Reset the encoder
        pub fn reset(&mut self) {
            self.state = GorillaFloatEncoderState::new();
        }
    }
}
