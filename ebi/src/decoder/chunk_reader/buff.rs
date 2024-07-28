use core::slice;
use std::{
    io::{self, Read},
    iter,
};

use either::Either;

use crate::decoder::{query::QueryExecutor, FileMetadataLike, GeneralChunkHandle};

use super::Reader;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFReader<R: Read> {
    reader: R,
    chunk_size: u64,
    decompressed: Option<Vec<f64>>,
}

impl<R: Read> BUFFReader<R> {
    pub fn new<T: FileMetadataLike>(handle: GeneralChunkHandle<T>, reader: R) -> Self {
        let chunk_size = handle.chunk_size();
        Self {
            reader,
            chunk_size,
            decompressed: None,
        }
    }
}

type F = fn(&f64) -> io::Result<f64>;
pub type BUFFIterator<'a> = Either<iter::Map<slice::Iter<'a, f64>, F>, iter::Once<io::Result<f64>>>;

impl<R: Read> Reader for BUFFReader<R> {
    type NativeHeader = ();

    type DecompressIterator<'a> = BUFFIterator<'a>
    where
        Self: 'a;

    fn decompress(&mut self) -> io::Result<&[f64]> {
        let mut buf = vec![0; self.chunk_size as usize];
        self.reader.read_exact(&mut buf)?;

        // TODO: use the scale from the header
        let result = internal::buff_simd256_decode(1, buf);

        self.decompressed = Some(result);

        Ok(self.decompressed.as_ref().unwrap())
    }

    fn decompress_iter(&mut self) -> Self::DecompressIterator<'_> {
        let decompressed = self.decompress();

        match decompressed {
            Ok(decompressed) => Either::Left(decompressed.iter().map(|f| Ok(*f))),
            Err(e) => Either::Right(iter::once(Err(e))),
        }
    }

    fn set_decompress_result(&mut self, data: Vec<f64>) -> &[f64] {
        self.decompressed = Some(data);
        self.decompressed.as_ref().unwrap()
    }

    fn decompress_result(&mut self) -> Option<&[f64]> {
        self.decompressed.as_deref()
    }

    fn header_size(&self) -> usize {
        0
    }

    fn read_header(&mut self) -> &Self::NativeHeader {
        &()
    }
}

// TODO: implement in-situ query execution
impl<R: Read> QueryExecutor for BUFFReader<R> {}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{
        bit_packing::BitPack, flip, get_precision_bound, precision_bound::PrecisionBound,
    };

    pub fn buff_simd256_decode(scale: usize, bytes: Vec<u8>) -> Vec<f64> {
        let prec = (scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int = (lower as u64) | ((higher as u64) << 32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}", base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}", len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}", ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;

        let mut expected_datapoints: Vec<f64> = Vec::new();
        let mut fixed_vec: Vec<u64> = Vec::new();

        let mut dec_scl: f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen + ilen;
        let mut bytec = 0;
        let mut chunk;
        let mut f_cur = 0f64;
        let mut cur = 0;

        if remain < 8 {
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                expected_datapoints.push((base_int + cur as i64) as f64 / dec_scl);
            }
            remain = 0
        } else {
            bytec += 1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();

            if remain == 0 {
                for &x in chunk {
                    expected_datapoints.push((base_int + flip(x) as i64) as f64 / dec_scl);
                }
            } else {
                // dec_vec.push((bitpack.read_byte().unwrap() as u32) << remain);
                // let mut k = 0;
                for x in chunk {
                    // if k<10{
                    //     println!("write {}th value with first byte {}",k,(*x))
                    // }
                    // k+=1;
                    fixed_vec.push((flip(*x) as u64) << remain)
                }
            }
            println!("read the {}th byte of dec", bytec);

            while (remain >= 8) {
                bytec += 1;
                remain -= 8;
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                    // let mut iiter = int_vec.iter();
                    // let mut diter = dec_vec.iter();
                    // for cur_chunk in chunk.iter(){
                    //     expected_datapoints.push( *(iiter.next().unwrap()) as f64+ (((diter.next().unwrap())|((*cur_chunk) as u32)) as f64) / dec_scl);
                    // }

                    for (cur_fixed, cur_chunk) in fixed_vec.iter().zip(chunk.iter()) {
                        expected_datapoints.push(
                            (base_int + ((*cur_fixed) | (flip(*cur_chunk) as u64)) as i64) as f64
                                / dec_scl,
                        );
                    }
                } else {
                    let mut it = chunk.into_iter();
                    fixed_vec = fixed_vec
                        .into_iter()
                        .map(|x| x | ((flip(*(it.next().unwrap())) as u64) << remain))
                        .collect();
                }

                println!("read the {}th byte of dec", bytec);
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in leading bytes: {:?}", duration);

            // let start5 = Instant::now();
            if (remain > 0) {
                bitpack.finish_read_byte();
                println!("read remaining {} bits of dec", remain);
                println!("length for fixed:{}", fixed_vec.len());
                for cur_fixed in fixed_vec.into_iter() {
                    f_cur = (base_int
                        + ((cur_fixed) | (bitpack.read_bits(remain as usize).unwrap() as u64))
                            as i64) as f64
                        / dec_scl;
                    expected_datapoints.push(f_cur);
                }
            }
        }
        // for i in 0..10{
        //     println!("{}th item:{}",i,expected_datapoints.get(i).unwrap())
        // }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }
}
