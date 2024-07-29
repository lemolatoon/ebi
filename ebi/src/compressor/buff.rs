use super::Compressor;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
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
        self.total_bytes_in = 0;
        self.data.clear();
        self.compressed = None;
    }
}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{
        bit_packing::BitPack,
        flip, get_precision_bound,
        precision_bound::{PrecisionBound, PRECISION_MAP},
    };

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
            dbg!(prec);
            let dec_len = *PRECISION_MAP.get(prec as usize).unwrap();
            dbg!(dec_len);
            bound.set_length(0, dec_len);
            let mut min = i64::max_value();
            let mut max = i64::min_value();

            for bd in seg {
                let fixed = bound.fetch_fixed_aligned(*bd);
                if fixed < min {
                    min = fixed;
                    dbg!((min, *bd));
                }
                if fixed > max {
                    max = fixed;
                    dbg!((max, *bd));
                }
                fixed_vec.push(fixed);
            }
            dbg!(min, max);
            let delta = max - min;
            let base_fixed = min;
            println!("base integer: {}, max:{}", base_fixed, max);
            let ubase_fixed = unsafe { mem::transmute::<i64, u64>(base_fixed) };
            let base_fixed64: i64 = base_fixed;
            let mut single_val = false;
            let mut cal_int_length = 0.0;
            // if delta == 0 {
            //     single_val = true;
            // } else {
            //     cal_int_length = (delta as f64).log2().ceil();
            // }
            cal_int_length = (max as f64).log2().ceil();

            let fixed_len = cal_int_length as usize;
            dbg!(min, max, cal_int_length, dec_len);
            println!("min:\t\t {:064b}", min);
            println!("min(delta):\t {:064b}", min - base_fixed64);
            println!("max:\t\t {:064b}", max);
            println!("max(delta):\t {:064b}", max - base_fixed64);
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
}
