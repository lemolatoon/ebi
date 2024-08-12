use crate::io::bit_write::BufferedBitWriter;

use super::{AppendableCompressor, Compressor};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GorillaCompressor {
    encoder: modified_tsz::GorillaFloatEncoder,
}

const DEFAULT_BUF_SIZE: usize = 1024 * 8;

impl GorillaCompressor {
    pub fn new() -> Self {
        let w = BufferedBitWriter::with_capacity(DEFAULT_BUF_SIZE);
        let encoder = modified_tsz::GorillaFloatEncoder::new(w);
        Self { encoder }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let w = BufferedBitWriter::with_capacity(capacity);
        let encoder = modified_tsz::GorillaFloatEncoder::new(w);
        Self { encoder }
    }
}

impl Default for GorillaCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for GorillaCompressor {
    fn compress(&mut self, input: &[f64]) {
        self.reset();
        for value in input {
            self.encoder.encode(*value);
        }
    }

    fn total_bytes_in(&self) -> usize {
        self.encoder.total_bytes_in()
    }

    fn total_bytes_buffered(&self) -> usize {
        self.encoder.total_bytes_buffered()
    }

    fn prepare(&mut self) {
        self.encoder.prepare();
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const E: &[u8] = &[];

        [self.encoder.bytes(), E, E, E, E]
    }

    fn reset(&mut self) {
        self.encoder.reset();
    }
}

impl AppendableCompressor for GorillaCompressor {
    fn append_compress(&mut self, input: &[f64]) {
        for value in input {
            self.encoder.encode(*value);
        }
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

    use crate::io::bit_write::BitWrite as _;

    use super::BufferedBitWriter;

    /// END_MARKER is a special bit sequence used to indicate the end of the stream
    pub const END_MARKER: u64 = 0b1111_0000_0000_0000_0000_0000_0000_0000_0000;

    /// END_MARKER_LEN is the length, in bits, of END_MARKER
    pub const END_MARKER_LEN: u32 = 36;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct GorillaFloatEncoderState {
        /// current float value as bits
        pub value_bits: u64,
        pub leading_zeroes: u32,
        // store the number of leading and trailing zeroes in the current xor as u32 so we
        // don't have to do any conversions after calling `leading_zeros` and `trailing_zeros`
        pub trailing_zeroes: u32,
        pub total_bytes_in: usize,
        /// total_bits_buffered is the total number of bits that have been written to the buffer
        pub total_bits_buffered: usize,
        // will next DataPoint be the first DataPoint encoded
        pub first: bool,
    }
    impl GorillaFloatEncoderState {
        pub fn new() -> Self {
            Self {
                value_bits: 0,
                leading_zeroes: 64,  // 64 is an initial sentinel value
                trailing_zeroes: 64, // 64 is an initial sentinel value

                total_bytes_in: 0,
                total_bits_buffered: 0,

                first: true,
            }
        }
    }

    /// StdEncoder
    ///
    /// StdEncoder is used to encode `DataPoint`s
    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    pub struct GorillaFloatEncoder {
        state: GorillaFloatEncoderState,

        w: BufferedBitWriter,
    }

    impl GorillaFloatEncoder {
        /// new creates a new StdEncoder whose starting timestamp is `start` and writes its encoded
        /// bytes to `w`
        pub fn new(w: BufferedBitWriter) -> Self {
            Self {
                state: GorillaFloatEncoderState::new(),
                w,
            }
        }

        fn write_first(&mut self, value_bits: u64) {
            // store the first value exactly
            self.w.write_bits(value_bits, 64);
            self.state.value_bits = value_bits;
            self.state.total_bits_buffered += 64;

            self.state.first = true
        }

        fn write_next_value(&mut self, value_bits: u64) {
            let xor = value_bits ^ self.state.value_bits;
            self.state.value_bits = value_bits;

            if xor == 0 {
                // if xor with previous value is zero just store single zero bit
                self.w.write_bit(false);
                self.state.total_bits_buffered += 1;
            } else {
                self.w.write_bit(true);

                self.state.total_bits_buffered += 1;

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
                    self.w.write_bit(false);
                    self.w.write_bits(
                        xor.wrapping_shr(self.state.trailing_zeroes),
                        64 - self.state.leading_zeroes - self.state.trailing_zeroes,
                    );

                    self.state.total_bits_buffered +=
                        (1 + 64 - self.state.leading_zeroes - self.state.trailing_zeroes) as usize;
                } else {
                    // if the number of leading and trailing zeroes in this xor are not less than the
                    // leading and trailing zeroes in the previous xor then we store a control bit and
                    // use 6 bits to store the number of leading zeroes and 6 bits to store the number
                    // of significant digits before storing the significant digits themselves

                    self.w.write_bit(true);
                    self.w.write_bits(u64::from(leading_zeroes), 5); // not 6 bit, but 5 bit

                    // if significant_digits is 64 we cannot encode it using 6 bits, however since
                    // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
                    // significant_digits can always be expressed with 6 bits or less
                    let significant_digits = 64 - leading_zeroes - trailing_zeroes;
                    self.w.write_bits(u64::from(significant_digits - 1), 6);
                    self.w
                        .write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);

                    self.state.total_bits_buffered += (1 + 5 + 6 + significant_digits) as usize;

                    // finally we need to update the number of leading and trailing zeroes
                    self.state.leading_zeroes = leading_zeroes;
                    self.state.trailing_zeroes = trailing_zeroes;
                }
            }
        }
    }

    impl GorillaFloatEncoder {
        pub fn encode(&mut self, value: f64) {
            let value_bits = value.to_bits();

            self.state.total_bytes_in += 8;

            if self.state.first {
                self.write_first(value_bits);
                self.state.first = false;
            } else {
                self.write_next_value(value_bits)
            }
        }

        /// Returns the total number of bytes of the compressed data if the stream were to be closed
        pub fn total_bytes_buffered(&self) -> usize {
            (self.state.total_bits_buffered + END_MARKER_LEN as usize).next_multiple_of(8) / 8
        }

        /// Returns the total number of bytes of the encoded data
        pub fn total_bytes_in(&self) -> usize {
            self.state.total_bytes_in
        }

        /// Prepare for the call to `close` or `bytes`.
        /// This method writes the END_MARKER to the buffer
        pub fn prepare(&mut self) {
            self.w.write_bits(END_MARKER, 36);
        }

        /// Returns the compressed data as a slice of bytes
        /// If you have END_MARKER written, you should call `prepare` before calling this method
        pub fn bytes(&self) -> &[u8] {
            self.w.as_slice()
        }

        /// Reset the encoder
        pub fn reset(&mut self) {
            self.state = GorillaFloatEncoderState::new();

            self.w.reset();
        }
    }

    #[cfg(test)]
    mod tests {

        use super::BufferedBitWriter;

        use super::GorillaFloatEncoder;

        #[test]
        fn create_new_encoder() {
            let w = BufferedBitWriter::new();
            let mut e = GorillaFloatEncoder::new(w);

            e.prepare();

            assert_eq!(e.total_bytes_in(), 0);
            assert_eq!(e.total_bytes_buffered(), 5);

            let bytes = e.bytes();
            // Just the END_MARKER
            let expected_bytes: [u8; 5] =
                [0b1111_0000, 0b0000_0000, 0b0000_0000, 0b0000_0000, 0b0000];

            assert_eq!(bytes[..], expected_bytes[..]);
        }

        #[test]
        fn encode_datapoint() {
            let w = BufferedBitWriter::new();
            let mut e = GorillaFloatEncoder::new(w);

            let d1: f64 = 1.24;

            e.encode(d1);

            e.prepare();

            assert_eq!(e.total_bytes_in(), 8);
            assert_eq!(e.total_bytes_buffered(), 13);

            let bytes = e.bytes();
            let expected_bytes: [u8; 13] = [
                0x3f, 0xf3, 0xd7, 0x0a, 0x3d, 0x70, 0xa3, 0xd7, /* 1.24 as double */
                0xf0, 0, 0, 0, 0, /* END_MARKER */
            ];

            assert_eq!(bytes[..], expected_bytes[..]);
        }

        #[test]
        fn encode_multiple_datapoints() {
            let w = BufferedBitWriter::new();
            let mut e = GorillaFloatEncoder::new(w);

            let d1 = 12.0;

            e.encode(d1);

            let d2 = 24.0;

            e.encode(d2);

            println!("d1: {:016x}", d1.to_bits());
            println!(
                "d2: {:016x}, {:016x}",
                d2.to_bits(),
                d1.to_bits() ^ d2.to_bits()
            );

            e.prepare();

            assert_eq!(e.total_bytes_in(), 16);
            assert_eq!(e.total_bytes_buffered(), 15);

            let bytes = e.bytes();
            let expected_bytes: [u8; 15] = [
                0x40,
                0x28,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00, /* 12.0 as double */
                #[allow(clippy::unusual_byte_groupings)]
                0b11__0_1011__0, /* control bit 11, # of leading zeros (11 (5 bits)) */
                #[allow(clippy::unusual_byte_groupings)]
                0b0_0000__1__11, /* # of meaning bits (1 (6 bits)), actual meaning bits 1, END_MARKER starts (2 bits) */
                #[allow(clippy::unusual_byte_groupings)]
                0b11__00_0000,
                0,
                0,
                0,
                0, /* END_MARKER */
            ];

            assert_eq!(bytes[..], expected_bytes[..]);
        }
    }
}
