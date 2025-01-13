#[inline]
pub(crate) fn flip(x: u8) -> u8 {
    let offset = 1u8 << 7;
    x ^ offset
}

const EXP_MASK: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
const FIRST_ONE: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;

#[inline]
fn into_fixed_representation(value: f64, decimal_length: i32) -> i64 {
    let v_bits = value.to_bits();
    let exp = ((v_bits & EXP_MASK) >> 52) as i32 - 1023;
    let sign = v_bits & FIRST_ONE;

    if (63 - exp - decimal_length) <= 0 || (63 - exp - decimal_length) >= 64 {
        dbg!(value, exp, decimal_length, 63 - exp - decimal_length);
    }
    let mut fixed = if (63 - exp - decimal_length) >= 64 {
        debug_assert!(63 - exp - decimal_length == 64);
        0
    } else {
        ((v_bits << (11)) | FIRST_ONE) >> (63 - (exp) - decimal_length) as u64
    };

    if sign != 0 {
        fixed = !(fixed - 1);
    }

    unsafe { std::mem::transmute::<u64, i64>(fixed) }
}

pub(crate) mod precision_bound {
    use super::{into_fixed_representation, EXP_MASK};

    pub const PRECISION_MAP: [u64; 13] = [0, 5, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 50];

    /// Use this function to convert a double to fixed representation with a given fractional part bits length
    #[inline]
    pub fn into_fixed_representation_with_fractional_part_bits_length(
        v: f64,
        fractional_part_bits_length: i32,
    ) -> i64 {
        let v_bits = v.to_bits();
        let exp = ((v_bits & EXP_MASK) >> 52) as i32 - 1023i32;
        if exp < -fractional_part_bits_length {
            0i64
        } else {
            into_fixed_representation(v, fractional_part_bits_length)
        }
    }
}

pub(crate) mod bit_packing {
    pub const MAX_BITS: usize = 32;
    pub const BYTE_BITS: usize = 8;
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct BitPack<B> {
        buff: B,
        cursor: usize,
        bits: usize,
    }

    impl<B> BitPack<B> {
        #[inline]
        pub fn new(buff: B) -> Self {
            BitPack {
                buff,
                cursor: 0,
                bits: 0,
            }
        }

        #[inline]
        pub fn sum_bits(&self) -> usize {
            self.cursor * BYTE_BITS + self.bits
        }
    }

    impl BitPack<&mut [u8]> {
        /// # Errors
        /// If bits > MAX_BITS, return Err(bits)
        ///
        /// If the buffer is full, in other words, the buffer does not have enough space to write bits, return Err(bits)
        pub fn write(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
            if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                return Err(bits);
            }
            if bits < MAX_BITS {
                value &= ((1 << bits) - 1) as u32;
            }

            loop {
                let bits_left = BYTE_BITS - self.bits;

                if bits <= bits_left {
                    self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                    self.bits += bits;

                    if self.bits >= BYTE_BITS {
                        self.cursor += 1;
                        self.bits = 0;
                    }

                    break;
                }

                let bb = value & ((1 << bits_left) - 1) as u32;
                self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
                self.cursor += 1;
                self.bits = 0;
                value >>= bits_left as u32;
                bits -= bits_left;
            }
            Ok(())
        }

        /***
        read bits less then BYTE_BITS
         */
        pub fn write_bits(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
            if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                return Err(bits);
            }
            value &= ((1 << bits) - 1) as u32;

            loop {
                let bits_left = BYTE_BITS - self.bits;

                if bits <= bits_left {
                    self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                    self.bits += bits;

                    if self.bits >= BYTE_BITS {
                        self.cursor += 1;
                        self.bits = 0;
                    }

                    break;
                }

                let bb = value & ((1 << bits_left) - 1) as u32;
                self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
                self.cursor += 1;
                self.bits = 0;
                value >>= bits_left as u32;
                bits -= bits_left;
            }
            Ok(())
        }
    }

    impl BitPack<&[u8]> {
        #[allow(dead_code)]
        pub fn read(&mut self, mut bits: usize) -> Result<u32, usize> {
            if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                return Err(bits);
            };

            let mut bits_left = 0;
            let mut output = 0;
            loop {
                let byte_left = BYTE_BITS - self.bits;

                if bits <= byte_left {
                    let mut bb = self.buff[self.cursor] as u32;
                    bb >>= self.bits as u32;
                    bb &= ((1 << bits) - 1) as u32;
                    output |= bb << bits_left;
                    self.bits += bits;
                    break;
                }

                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << byte_left) - 1) as u32;
                output |= bb << bits_left;
                self.bits += byte_left;
                bits_left += byte_left as u32;
                bits -= byte_left;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits -= BYTE_BITS;
                }
            }
            Ok(output)
        }

        pub fn read_u32(&mut self) -> Result<u32, usize> {
            if self.buff.len() * BYTE_BITS < self.sum_bits() + 32 {
                return Err(32);
            };

            let u32_slice = &self.buff[self.cursor..self.cursor + 4];
            self.cursor += 4;

            Ok(u32::from_le_bytes(u32_slice.try_into().unwrap()))
        }

        /***
        read bits less than BYTE_BITS
         */

        #[inline]
        pub fn read_bits(&mut self, mut bits: usize) -> Result<u8, usize> {
            if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                println!(
                    "buff length: {}, cursor: {}, bits: {}",
                    self.buff.len(),
                    self.cursor,
                    self.bits
                );
                return Err(bits);
            };

            let mut bits_left = 0;
            let mut output = 0;
            loop {
                let byte_left = BYTE_BITS - self.bits;

                if bits <= byte_left {
                    let mut bb = self.buff[self.cursor] as u32;
                    bb >>= self.bits as u32;
                    bb &= ((1 << bits) - 1) as u32;
                    output |= bb << bits_left;
                    self.bits += bits;
                    break;
                }

                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << byte_left) - 1) as u32;
                output |= bb << bits_left;
                self.bits += byte_left;
                bits_left += byte_left as u32;
                bits -= byte_left;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits -= BYTE_BITS;
                }
            }
            Ok(output as u8)
        }

        #[inline]
        pub fn read_byte(&mut self) -> Result<u8, usize> {
            let output = self.buff[self.cursor];
            self.cursor += 1;
            Ok(output)
        }

        #[inline]
        pub fn read_n_byte(&mut self, n: usize) -> Result<&[u8], usize> {
            let end = self.cursor + n;
            let output = &self.buff[self.cursor..end];
            self.cursor += n;
            Ok(output)
        }

        #[inline]
        pub fn skip_n_byte(&mut self, n: usize) -> Result<(), usize> {
            self.cursor += n;
            // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
            Ok(())
        }
        #[inline]
        pub fn skip(&mut self, bits: usize) -> Result<(), usize> {
            if bits == 0 {
                return Ok(());
            }
            if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
                return Err(bits);
            };
            // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
            // println!("try to skip {} bits",bits);
            let bytes = bits / BYTE_BITS;
            let left = bits % BYTE_BITS;

            let cur_bits = self.bits + left;
            self.cursor = self.cursor + bytes + cur_bits / BYTE_BITS;
            self.bits = cur_bits % BYTE_BITS;

            // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
            Ok(())
        }
    }

    impl Default for BitPack<Vec<u8>> {
        fn default() -> Self {
            Self::new(Vec::new())
        }
    }

    impl BitPack<Vec<u8>> {
        pub fn with_capacity(capacity: usize) -> Self {
            Self::new(Vec::with_capacity(capacity))
        }

        /// # Errors
        /// If bits > MAX_BITS, return Err(bits)
        #[inline]
        pub fn write(&mut self, value: u32, bits: usize) -> Result<(), usize> {
            let len = self.buff.len();

            if let Some(bits) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
                self.buff.resize(len + bits.div_ceil(BYTE_BITS), 0x0);
            }

            let mut bitpack = BitPack {
                buff: self.buff.as_mut_slice(),
                cursor: self.cursor,
                bits: self.bits,
            };

            bitpack.write(value, bits)?;

            self.bits = bitpack.bits;
            self.cursor = bitpack.cursor;

            Ok(())
        }

        /***
        read bits less then BYTE_BITS
         */
        #[inline]
        pub fn write_bits(&mut self, value: u32, bits: usize) -> Result<(), usize> {
            let len = self.buff.len();

            if let Some(bits) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
                self.buff.resize(len + bits.div_ceil(BYTE_BITS), 0x0);
            }

            let mut bitpack = BitPack {
                buff: self.buff.as_mut_slice(),
                cursor: self.cursor,
                bits: self.bits,
            };

            bitpack.write_bits(value, bits)?;

            self.bits = bitpack.bits;
            self.cursor = bitpack.cursor;

            Ok(())
        }

        #[inline]
        pub fn write_byte(&mut self, value: u8) {
            self.buff.push(value);
        }

        /// Update the state of the BitPack to reflect the cursor of the buffer
        /// [`Self::write_byte`] method does not update the cursor of the buffer
        /// This method should be called before [`Self::write_bits`] method to ensure the cursor is updated
        #[inline]
        pub fn finish_write_byte(&mut self) {
            let len = self.buff.len();
            self.buff.resize(len + 1, 0x0);
            self.bits = 0;
            self.cursor = len;
        }

        #[inline]
        pub fn into_vec(self) -> Vec<u8> {
            // println!("buff length: {}, cursor: {}, bits: {}", self.buff.len(), self.cursor,self.bits);
            self.buff
        }
    }
}
