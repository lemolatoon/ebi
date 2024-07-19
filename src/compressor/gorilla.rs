use super::Compressor;

pub struct GorillaCompressor {}

impl GorillaCompressor {
    pub fn new() -> Self {
        Self {}
    }
}

impl Compressor for GorillaCompressor {
    fn compress(&mut self, _input: &[f64]) -> usize {
        unimplemented!()
    }

    fn total_bytes_in(&self) -> usize {
        todo!()
    }

    fn total_bytes_buffered(&self) -> usize {
        todo!()
    }

    fn prepare(&mut self) {
        todo!()
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        todo!()
    }

    fn reset(&mut self) {
        todo!()
    }
}

mod modified_tsz {
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

    use tsz::{stream::Write, Bit};

    /// END_MARKER is a special bit sequence used to indicate the end of the stream
    pub const END_MARKER: u64 = 0b1111_0000_0000_0000_0000_0000_0000_0000_0000;

    /// END_MARKER_LEN is the length, in bits, of END_MARKER
    pub const END_MARKER_LEN: u32 = 36;

    /// StdEncoder
    ///
    /// StdEncoder is used to encode `DataPoint`s
    #[derive(Debug)]
    pub struct GorillaFloatEncoder<T: Write> {
        /// current float value as bits
        value_bits: u64,

        // store the number of leading and trailing zeroes in the current xor as u32 so we
        // don't have to do any conversions after calling `leading_zeros` and `trailing_zeros`
        leading_zeroes: u32,
        trailing_zeroes: u32,

        first: bool, // will next DataPoint be the first DataPoint encoded

        w: T,
    }

    impl<T> GorillaFloatEncoder<T>
    where
        T: Write,
    {
        /// new creates a new StdEncoder whose starting timestamp is `start` and writes its encoded
        /// bytes to `w`
        pub fn new(w: T) -> Self {
            Self {
                value_bits: 0,
                leading_zeroes: 64,  // 64 is an initial sentinel value
                trailing_zeroes: 64, // 64 is an intitial sentinel value
                first: true,
                w,
            }
        }

        fn write_first(&mut self, value_bits: u64) {
            self.value_bits = value_bits;

            // store the first value exactly
            self.w.write_bits(self.value_bits, 64);

            self.first = true
        }

        fn write_next_value(&mut self, value_bits: u64) {
            let xor = value_bits ^ self.value_bits;
            self.value_bits = value_bits;

            if xor == 0 {
                // if xor with previous value is zero just store single zero bit
                self.w.write_bit(Bit::Zero);
            } else {
                self.w.write_bit(Bit::One);

                let leading_zeroes = xor.leading_zeros();
                let trailing_zeroes = xor.trailing_zeros();

                if leading_zeroes >= self.leading_zeroes && trailing_zeroes >= self.trailing_zeroes
                {
                    // if the number of leading and trailing zeroes in this xor are >= the leading and
                    // trailing zeroes in the previous xor then we only need to store a control bit and
                    // the significant digits of this xor
                    self.w.write_bit(Bit::Zero);
                    self.w.write_bits(
                        xor.wrapping_shr(self.trailing_zeroes),
                        64 - self.leading_zeroes - self.trailing_zeroes,
                    );
                } else {
                    // if the number of leading and trailing zeroes in this xor are not less than the
                    // leading and trailing zeroes in the previous xor then we store a control bit and
                    // use 6 bits to store the number of leading zeroes and 6 bits to store the number
                    // of significant digits before storing the significant digits themselves

                    self.w.write_bit(Bit::One);
                    self.w.write_bits(u64::from(leading_zeroes), 5); // not 6 bit, but 5 bit

                    // if significant_digits is 64 we cannot encode it using 6 bits, however since
                    // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
                    // significant_digits can always be expressed with 6 bits or less
                    let significant_digits = 64 - leading_zeroes - trailing_zeroes;
                    self.w.write_bits(u64::from(significant_digits - 1), 6);
                    self.w
                        .write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);

                    // finally we need to update the number of leading and trailing zeroes
                    self.leading_zeroes = leading_zeroes;
                    self.trailing_zeroes = trailing_zeroes;
                }
            }
        }
    }

    impl<T> GorillaFloatEncoder<T>
    where
        T: Write,
    {
        pub fn encode(&mut self, value: f64) {
            let value_bits = value.to_bits();

            if self.first {
                self.write_first(value_bits);
                self.first = false;
                return;
            }

            self.write_next_value(value_bits)
        }

        pub fn close(mut self) -> Box<[u8]> {
            self.w.write_bits(END_MARKER, 36);
            self.w.close()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::GorillaFloatEncoder;
        use tsz::stream::BufferedWriter;

        #[test]
        fn create_new_encoder() {
            let w = BufferedWriter::new();
            let e = GorillaFloatEncoder::new(w);

            let bytes = e.close();
            // Just the END_MARKER
            let expected_bytes: [u8; 5] =
                [0b1111_0000, 0b0000_0000, 0b0000_0000, 0b0000_0000, 0b0000];

            assert_eq!(bytes[..], expected_bytes[..]);
        }

        #[test]
        fn encode_datapoint() {
            let w = BufferedWriter::new();
            let mut e = GorillaFloatEncoder::new(w);

            let d1: f64 = 1.24;

            e.encode(d1);

            let bytes = e.close();
            let expected_bytes: [u8; 13] = [
                0x3f, 0xf3, 0xd7, 0x0a, 0x3d, 0x70, 0xa3, 0xd7, /* 1.24 as double */
                0xf0, 0, 0, 0, 0, /* END_MARKER */
            ];

            assert_eq!(bytes[..], expected_bytes[..]);
        }

        #[test]
        fn encode_multiple_datapoints() {
            let w = BufferedWriter::new();
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

            let bytes = e.close();
            for b in &bytes[..] {
                println!("{:#010b}", b);
            }
            let expected_bytes: [u8; 15] = [
                0x40,
                0x28,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,            /* 12.0 as double */
                0b11__0_1011__0, /* control bit 11, # of leading zeros (11 (5 bits)) */
                0b0_0000__1__11, /* # of meaning bits (1 (6 bits)), actual meaning bits 1, END_MARKER starts (2 bits) */
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
