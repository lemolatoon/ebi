use super::Compressor;

pub struct BUFFCompressor {
    total_bytes_in: usize,
    data: Vec<f64>,
    scale: usize,
    compressed: Option<Vec<u8>>,
}

impl BUFFCompressor {
    pub fn new(scale: usize) -> Self {
        Self {
            total_bytes_in: 0,
            data: Vec::new(),
            scale,
            compressed: None,
        }
    }
}

impl Compressor for BUFFCompressor {
    fn compress(&mut self, data: &[f64]) -> usize {
        self.data.extend_from_slice(data);

        let n_bytes_compressed = size_of_val(data);
        self.total_bytes_in += n_bytes_compressed;

        n_bytes_compressed
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        self.compressed.as_ref().map_or(0, |v| v.len())
    }

    fn prepare(&mut self) {
        let comp = internal::SplitBDDoubleCompress::new(self.scale);
        let data = comp.buff_simd256_encode(&self.data);

        self.compressed = Some(data);
    }

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const EMPTY: &[u8] = &[];
        let data = self.compressed.as_ref().map_or(EMPTY, |v| v.as_slice());
        [data, EMPTY, EMPTY, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        todo!()
    }
}

mod internal {
    use bit_packing::BitPack;
    use precision_bound::{PrecisionBound, PRECISION_MAP};
    use std::mem;
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

    #[derive(Clone)]
    pub struct SplitBDDoubleCompress {
        pub(crate) scale: usize,
    }

    impl SplitBDDoubleCompress {
        pub fn new(scale: usize) -> Self {
            SplitBDDoubleCompress { scale }
        }

        pub fn buff_simd256_encode<'a>(&self, seg: &Vec<f64>) -> Vec<u8> {
            let mut fixed_vec = Vec::new();

            let mut t: u32 = seg.len() as u32;
            let mut prec = 0;
            if self.scale == 0 {
                prec = 0;
            } else {
                prec = (self.scale as f32).log10() as i32;
            }
            let prec_delta = get_precision_bound(prec);
            println!("precision {}, precision delta:{}", prec, prec_delta);

            let mut bound = PrecisionBound::new(prec_delta);
            // let start1 = Instant::now();
            let dec_len = *PRECISION_MAP.get(prec as usize).unwrap();
            bound.set_length(0, dec_len);
            let mut min = i64::max_value();
            let mut max = i64::min_value();

            for bd in seg {
                let fixed = bound.fetch_fixed_aligned(*bd);
                if fixed < min {
                    min = fixed;
                } else if fixed > max {
                    max = fixed;
                }
                fixed_vec.push(fixed);
            }
            let delta = max - min;
            let base_fixed = min;
            println!("base integer: {}, max:{}", base_fixed, max);
            let ubase_fixed = unsafe { mem::transmute::<i64, u64>(base_fixed) };
            let base_fixed64: i64 = base_fixed;
            let mut single_val = false;
            let mut cal_int_length = 0.0;
            if delta == 0 {
                single_val = true;
            } else {
                cal_int_length = (delta as f64).log2().ceil();
            }

            let fixed_len = cal_int_length as usize;
            bound.set_length((cal_int_length as u64 - dec_len), dec_len);
            let ilen = fixed_len - dec_len as usize;
            let dlen = dec_len as usize;
            println!("int_len:{},dec_len:{}", ilen as u64, dec_len);
            let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
            bitpack_vec.write(ubase_fixed as u32, 32);
            bitpack_vec.write((ubase_fixed >> 32) as u32, 32);
            bitpack_vec.write(t, 32);
            bitpack_vec.write(ilen as u32, 32);
            bitpack_vec.write(dlen as u32, 32);

            // let duration1 = start1.elapsed();
            // println!("Time elapsed in dividing double function() is: {:?}", duration1);

            // let start1 = Instant::now();
            let mut remain = fixed_len;
            let mut bytec = 0;

            if remain < 8 {
                for i in fixed_vec {
                    bitpack_vec
                        .write_bits((i - base_fixed64) as u32, remain)
                        .unwrap();
                }
                remain = 0;
            } else {
                bytec += 1;
                remain -= 8;
                let mut fixed_u64 = Vec::new();
                let mut cur_u64 = 0u64;
                if remain > 0 {
                    // let mut k = 0;
                    fixed_u64 = fixed_vec
                        .iter()
                        .map(|x| {
                            cur_u64 = (*x - base_fixed64) as u64;
                            bitpack_vec.write_byte(flip((cur_u64 >> remain) as u8));
                            cur_u64
                        })
                        .collect::<Vec<u64>>();
                } else {
                    fixed_u64 = fixed_vec
                        .iter()
                        .map(|x| {
                            cur_u64 = (*x - base_fixed64) as u64;
                            bitpack_vec.write_byte(flip(cur_u64 as u8));
                            cur_u64
                        })
                        .collect::<Vec<u64>>();
                }
                println!("write the {}th byte of dec", bytec);

                while (remain >= 8) {
                    bytec += 1;
                    remain -= 8;
                    if remain > 0 {
                        for d in &fixed_u64 {
                            bitpack_vec.write_byte(flip((*d >> remain) as u8)).unwrap();
                        }
                    } else {
                        for d in &fixed_u64 {
                            bitpack_vec.write_byte(flip(*d as u8)).unwrap();
                        }
                    }

                    println!("write the {}th byte of dec", bytec);
                }
                if (remain > 0) {
                    bitpack_vec.finish_write_byte();
                    for d in fixed_u64 {
                        bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
                    }
                    println!("write remaining {} bits of dec", remain);
                }
            }

            // println!("total number of dec is: {}", j);
            let vec = bitpack_vec.into_vec();

            // let duration1 = start1.elapsed();
            // println!("Time elapsed in writing double function() is: {:?}", duration1);

            let origin = t * mem::size_of::<f64>() as u32;
            println!("original size:{}", origin);
            println!("compressed size:{}", vec.len());
            let ratio = vec.len() as f64 / origin as f64;
            print!("{}", ratio);
            vec
            //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
            //println!("{}", decode_reader(bytes).unwrap());
        }
    }

    mod precision_bound {
        use std::mem;

        use crate::compressor::buff::internal::bit_packing::BitPack;
        pub const EXP_MASK: u64 =
            0b0111111111110000000000000000000000000000000000000000000000000000;
        pub const FIRST_ONE: u64 =
            0b1000000000000000000000000000000000000000000000000000000000000000;
        pub const NEG_ONE: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111;

        pub const PRECISION_MAP: [u64; 16] =
            [0, 5, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 50, 10, 10, 10];

        pub struct PrecisionBound {
            position: u64,
            precision: f64,
            precision_exp: i32,
            int_length: u64,
            decimal_length: u64,
        }

        impl PrecisionBound {
            pub fn new(precision: f64) -> Self {
                let mut e = PrecisionBound {
                    position: 0,
                    precision: precision,
                    precision_exp: 0,
                    int_length: 0,
                    decimal_length: 0,
                };
                let xu: u64 = precision.to_bits();
                let xu = unsafe { mem::transmute::<f64, u64>(precision) };
                e.precision_exp = ((xu & EXP_MASK) >> 52) as i32 - 1023 as i32;
                e
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
            pub fn fetch_fixed_aligned(&self, bd: f64) -> i64 {
                let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
                let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023 as i32;
                let sign = bdu & FIRST_ONE;
                let mut fixed = 0u64;
                if exp < self.precision_exp {
                    fixed = 0u64;
                } else {
                    fixed = ((bdu << (11)) | FIRST_ONE)
                        >> (63 - exp - self.decimal_length as i32) as u64;
                    if sign != 0 {
                        fixed = !(fixed - 1);
                    }
                }
                let signed_int = unsafe { mem::transmute::<u64, i64>(fixed) };
                signed_int
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

    mod bit_packing {
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
                self.cursor += 1;
                let output = self.buff[self.cursor] as u8;
                Ok(output)
            }

            #[inline]
            pub fn read_n_byte(&mut self, n: usize) -> Result<&[u8], usize> {
                self.cursor += 1;
                let end = self.cursor + n;
                let output = &self.buff[self.cursor..end];
                self.cursor += n - 1;
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
            pub fn skip(&mut self, mut bits: usize) -> Result<(), usize> {
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
            pub fn write_byte(&mut self, value: u8) -> Result<(), usize> {
                self.buff.push(value);
                Ok(())
            }

            #[inline]
            pub fn finish_write_byte(&mut self) {
                let len = self.buff.len();
                self.buff.resize(len + 1, 0x0);
                self.bits = 0;
                self.cursor = len;
                // println!("cursor now at {}" , self.cursor)
            }

            #[inline]
            pub fn into_vec(self) -> Vec<u8> {
                // println!("buff length: {}, cursor: {}, bits: {}", self.buff.len(), self.cursor,self.bits);
                self.buff
            }
        }
    }
}
