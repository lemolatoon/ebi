pub fn get_precision_bound(precision: i32) -> f64 {
    let mut str = String::from("0.");
    for pos in 0..precision {
        str.push('0');
    }
    str.push_str("49");
    let error = str.parse().unwrap();
    error
}

#[inline]
pub fn flip(x: u8) -> u8 {
    let offset = 1u8 << 7;
    x ^ offset
}

const EXP_MASK: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
const FIRST_ONE: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;
const NEG_ONE: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111;

#[inline]
pub fn into_fixed_representation(value: f64, decimal_length: i32) -> i64 {
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

pub mod precision_bound {
    use std::mem;

    use crate::compression_common::buff::bit_packing::BitPack;

    use super::{get_precision_bound, into_fixed_representation, EXP_MASK, FIRST_ONE, NEG_ONE};

    pub const PRECISION_MAP: [u64; 13] = [0, 5, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 50];
    pub const INV_PRECISION_MAP: [Option<u64>; 51] = const {
        let mut arr = [None; 51];
        let mut i = 0;
        while i < PRECISION_MAP.len() {
            arr[PRECISION_MAP[i] as usize] = Some(i as u64);
            i += 1;
        }
        arr
    };

    pub struct PrecisionBound {
        position: u64,
        precision: f64,
        precision_exp: i32,
        int_length: u64,
        decimal_length: u64,
    }

    impl PrecisionBound {
        pub fn new(precision_delta: f64) -> Self {
            let precision_delta_bits = precision_delta.to_bits();
            let precision_exp = ((precision_delta_bits & EXP_MASK) >> 52) as i32 - 1023;
            PrecisionBound {
                position: 0,
                precision: precision_delta,
                precision_exp,
                int_length: 0,
                decimal_length: 0,
            }
        }

        pub fn precision_bound(&mut self, orig: f64) -> f64 {
            let a = 0u64;
            let mut mask = !a;
            let mut ret = 0f64;
            let mut pre = orig;
            let mut cur = 0f64;
            let origu = unsafe { mem::transmute::<f64, u64>(orig) };
            let mut curu = 0u64;
            curu = origu & (mask << self.position) | (1u64 << self.position);
            cur = unsafe { mem::transmute::<u64, f64>(curu) };
            pre = cur;
            let mut bounded = self.is_bounded(orig, cur);
            if bounded {
                // find first bit where is not bounded
                while bounded {
                    if self.position == 52 {
                        return pre;
                    }
                    self.position += 1;
                    curu = origu & (mask << self.position) | (1u64 << self.position);
                    cur = unsafe { mem::transmute::<u64, f64>(curu) };
                    if !self.is_bounded(orig, cur) {
                        bounded = false;
                        break;
                    }
                    pre = cur;
                }
            } else {
                // find the first bit where is bounded
                while !bounded {
                    if self.position == 0 {
                        break;
                    }
                    self.position -= 1;
                    curu = origu & (mask << self.position) | (1u64 << self.position);
                    cur = unsafe { mem::transmute::<u64, f64>(curu) };
                    if self.is_bounded(orig, cur) {
                        bounded = true;
                        pre = cur;
                        break;
                    }
                }
            }
            pre
        }

        //    iter over all bounded double and set the length for each one
        pub fn cal_length(&mut self, x: f64) {
            let xu = unsafe { mem::transmute::<f64, u64>(x) };
            let trailing_zeros = xu.trailing_zeros();
            let exp = ((xu & EXP_MASK) >> 52) as i32 - 1023 as i32;
            // println!("trailing_zeros:{}",trailing_zeros);
            // println!("exp:{}",exp);
            let mut dec_length = 0;
            if 52 <= trailing_zeros {
                if exp < 0 {
                    dec_length = (-exp) as u64;
                    if exp < self.precision_exp {
                        dec_length = 0;
                    }
                }
            } else if (52 - trailing_zeros as i32) > exp {
                dec_length = ((52 - trailing_zeros) as i32 - exp) as u64;
            }

            if exp >= 0 {
                if (exp + 1) as u64 > self.int_length {
                    self.int_length = (exp + 1) as u64;
                }
            }
            if dec_length > self.decimal_length {
                self.decimal_length = dec_length as u64;
                // let xu =  unsafe { mem::transmute::<f64, u64>(x)};
                // println!("{} with dec_length:{}, bounded => {:#066b}",x, dec_length, xu);
            }
            // println!("int len :{}, dec len:{}",self.int_length,self.decimal_length );
        }

        pub fn get_length(&self) -> (u64, u64) {
            (self.int_length, self.decimal_length)
        }

        #[inline]
        pub fn set_length(&mut self, ilen: u64, dlen: u64) {
            self.decimal_length = dlen;
            self.int_length = ilen;
        }

        // this is for dataset with same power of 2, power>1
        #[inline]
        pub fn fast_fetch_components_large(&self, bd: f64, exp: i32) -> (i64, u64) {
            let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
            // let sign = bdu&FIRST_ONE;
            let mut int_part = 0u64;
            let mut dec_part = 0u64;
            let dec_move = 64u64 - self.decimal_length;
            // if exp>=0{
            dec_part = bdu << (12 + exp) as u64;
            int_part = ((bdu << 11) | FIRST_ONE) >> (63 - exp) as u64;
            // dec_part = bdu << 18 as u64;
            // int_part = ((bdu << 11)| FIRST_ONE )>> 57 as u64;
            // }else if exp<self.precision_exp{
            //     dec_part=0u64;
            // }else{
            //     dec_part = (((bdu << (12)) >>1) | FIRST_ONE) >> ((-exp - 1) as u64);
            // }
            // int_part = int_part|sign;
            // let signed_int = unsafe { mem::transmute::<u64, i64>(int_part) };
            //let signed_int = bd.trunc() as i64;
            (int_part as i64, dec_part >> dec_move)
        }

        #[inline]
        pub fn fetch_components(&self, bd: f64) -> (i64, u64) {
            let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
            let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023 as i32;
            let sign = bdu & FIRST_ONE;
            let mut int_part = 0u64;
            let mut dec_part = 0u64;
            // todo: check whether we can remove those if branch
            if exp >= 0 {
                dec_part = bdu << (12 + exp) as u64;
                int_part = ((bdu << 11) | FIRST_ONE) >> (63 - exp) as u64;
                if sign != 0 {
                    int_part = !int_part;
                    // this is an approximation for simplification.
                    // dec_part = !dec_part;
                    // more accurate representation
                    dec_part = !dec_part + 1;
                }
            } else if exp < self.precision_exp {
                dec_part = 0u64;
                if sign != 0 {
                    int_part = NEG_ONE;
                    dec_part = !dec_part;
                }
            } else {
                dec_part = ((bdu << (11)) | FIRST_ONE) >> ((-exp - 1) as u64);
                if sign != 0 {
                    int_part = NEG_ONE;
                    dec_part = !dec_part;
                }
            }

            let signed_int = unsafe { mem::transmute::<u64, i64>(int_part) };
            //let signed_int = bd.trunc() as i64;
            (signed_int, dec_part >> 64u64 - self.decimal_length)
        }

        ///byte aligned version of spilt double
        #[inline]
        pub fn into_fixed_representation(&self, v: f64) -> i64 {
            let v_bits = v.to_bits();
            let exp = ((v_bits & EXP_MASK) >> 52) as i32 - 1023i32;
            if exp < self.precision_exp {
                0i64
            } else {
                into_fixed_representation(v, self.decimal_length as i32)
            }
        }

        /// Use this function to convert a double to fixed representation with a given decimal length
        /// This computation might be heavy, so if you convert a lot of double, you can use the `into_fixed_representation` function
        #[inline]
        pub fn into_fixed_representation_with_decimal_length(v: f64, decimal_length: i32) -> i64 {
            let precision = INV_PRECISION_MAP[decimal_length as usize].unwrap();
            let precision_bound = get_precision_bound(precision as i32);
            let precision_exp = ((precision_bound.to_bits() & EXP_MASK) >> 52) as i32 - 1023;
            let v_bits = v.to_bits();
            let exp = ((v_bits & EXP_MASK) >> 52) as i32 - 1023i32;
            if exp < precision_exp {
                0i64
            } else {
                into_fixed_representation(v, decimal_length)
            }
        }

        pub fn finer(&self, input: f64) -> Vec<u8> {
            print!("finer results:");
            let mut org = input.abs();
            let mut cur = 0.5f64;
            let mut pre = 0f64;
            let mut mid = 0f64;
            let mut low = 0.0;
            let mut high = 1.0;
            // check if input between 0 and 1;
            let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(1);
            if org == 0.0 {
                bitpack_vec.write(0, 1);
                print!("0");
                let vec = bitpack_vec.into_vec();
                return vec;
            } else if org == 0.5 {
                bitpack_vec.write(1, 1);
                print!("1");
                let vec = bitpack_vec.into_vec();
                return vec;
            } else if (org < 1.0 && org > 0.0) {
                mid = (low + high) / 2.0;
                while (!self.is_bounded(org, mid)) {
                    if org < mid {
                        bitpack_vec.write(0, 1);
                        print!("0");
                        high = mid;
                    } else {
                        bitpack_vec.write(1, 1);
                        print!("1");
                        low = mid;
                    }
                    mid = (low + high) / 2.0;
                }
            }
            if mid == org {
                bitpack_vec.write(0, 1);
                print!("0");
            } else {
                bitpack_vec.write(1, 1);
                print!("1");
            }
            let vec = bitpack_vec.into_vec();
            println!("{:?} after conversion: {:?}", vec, vec);
            vec
        }

        #[inline]
        pub fn is_bounded(&self, a: f64, b: f64) -> bool {
            let delta = a - b;
            if delta.abs() < self.precision {
                return true;
            }
            return false;
        }
    }
}

pub mod bit_packing {
    pub const MAX_BITS: usize = 32;
    pub const BYTE_BITS: usize = 8;
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct BitPack<B> {
        buff: B,
        cursor: usize,
        bits: usize,
    }

    impl<B> BitPack<B> {
        #[inline]
        pub fn new(buff: B) -> Self {
            BitPack {
                buff: buff,
                cursor: 0,
                bits: 0,
            }
        }

        #[inline]
        pub fn sum_bits(&self) -> usize {
            self.cursor * BYTE_BITS + self.bits
        }

        #[inline]
        pub fn with_cursor(&mut self, cursor: usize) -> &mut Self {
            self.cursor = cursor;
            self
        }

        #[inline]
        pub fn with_bits(&mut self, bits: usize) -> &mut Self {
            self.bits = bits;
            self
        }
    }

    impl<B: AsRef<[u8]>> BitPack<B> {
        #[inline]
        pub fn as_slice(&self) -> &[u8] {
            self.buff.as_ref()
        }
    }

    impl<'a> BitPack<&'a mut [u8]> {
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

    impl<'a> BitPack<&'a [u8]> {
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
        pub fn finish_read_byte(&mut self) {
            self.cursor += 1;
            self.bits = 0;
            // println!("cursor now at {}" , self.cursor)
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
        pub fn read_n_byte_unmut(&self, start: usize, n: usize) -> Result<&[u8], usize> {
            let s = start + self.cursor + 1;
            let end = s + n;
            let output = &self.buff[s..end];
            Ok(output)
        }

        #[inline]
        pub fn skip_n_byte(&mut self, mut n: usize) -> Result<(), usize> {
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

            let cur_bits = (self.bits + left);
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
                self.buff
                    .resize(len + (bits + BYTE_BITS - 1) / BYTE_BITS, 0x0);
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
                self.buff
                    .resize(len + (bits + BYTE_BITS - 1) / BYTE_BITS, 0x0);
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
        pub fn write_bytes(&mut self, value: &mut Vec<u8>) -> Result<(), usize> {
            self.buff.append(value);
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
