use std::mem;

use crate::compression_common::buff::{bit_packing::BitPack, precision_bound::PRECISION_MAP};

pub fn decode_with_precision(bytes: &[u8], precision: u32) -> Vec<f64> {
    let mut bitpack = BitPack::<&[u8]>::new(bytes);

    let lower = bitpack.read_u32().unwrap();
    let higher = bitpack.read_u32().unwrap();
    let base_fixed64_bits = (lower as u64) | ((higher as u64) << 32);
    let base_int = unsafe { mem::transmute::<u64, i64>(base_fixed64_bits) };

    // Number of records
    let len = bitpack.read_u32().unwrap();

    // Fixed Repr bits length
    let fixed_bits_length = bitpack.read_u32().unwrap();

    // Fractional part bits length
    let dlen = bitpack.read_u32().unwrap();

    let ilen = fixed_bits_length - dlen;

    // check integer part and update bitmap;
    let mut cur;

    let mut expected_datapoints: Vec<f64> = Vec::new();
    let mut int_vec: Vec<i64> = Vec::new();
    let mut dec_vec = Vec::new();
    let mut cur_intf;
    let mut remain;
    let mut processed = 0;

    if precision == 0 {
        for _ in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            expected_datapoints.push(cur as f64);
        }
        expected_datapoints
    } else {
        let bits_needed = PRECISION_MAP[precision as usize] as u32;
        assert!(dlen >= bits_needed, "{} < {}", dlen, bits_needed);
        remain = bits_needed;
        let dec_byte = dlen / 8;
        let mut byte_needed = bits_needed / 8;
        let extra_bits = bits_needed % 8;
        if byte_needed < dec_byte && extra_bits > 0 {
            byte_needed += 1;
            remain = byte_needed * 8;
        }

        for _ in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            int_vec.push(cur as i64 + base_int);
        }

        let mut dec_scl: f64;

        let mut chunk;

        if remain >= 8 {
            remain -= 8;
            processed += 8;
            dec_scl = 2.0f64.powi(processed);
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            if remain == 0 {
                for (int_comp, dec_comp) in int_vec.iter().zip(chunk.iter()) {
                    cur_intf = *int_comp as f64;
                    //todo: this is problemetic.
                    expected_datapoints.push(cur_intf + ((*dec_comp) as f64) / dec_scl);
                }
            } else {
                // dec_vec.push((bitpack.read_byte().unwrap() as u32) << remain);
                // let mut k = 0;
                for x in chunk {
                    // if k<10{
                    //     println!("write {}th value with first byte {}",k,(*x))
                    // }
                    // k+=1;
                    dec_vec.push(((*x) as u32) << remain)
                }
            }
        }
        while remain >= 8 {
            remain -= 8;
            processed += 8;
            dec_scl = 2.0f64.powi(processed);
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            if remain == 0 {
                // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                for (int_comp, dec_comp, cur_chunk) in int_vec
                    .iter()
                    .zip(&dec_vec)
                    .zip(chunk)
                    .map(|((a, b), c)| (a, b, c))
                {
                    cur_intf = *int_comp as f64;
                    // if j<10{
                    //     println!("{}th item {}, decimal:{}",j, cur_f,*dec_comp);
                    // }
                    // j += 1;
                    expected_datapoints
                        .push(cur_intf + (((*dec_comp) | ((*cur_chunk) as u32)) as f64) / dec_scl);
                }
            } else {
                let mut it = chunk.iter();
                dec_vec = dec_vec
                    .into_iter()
                    .map(|x| x | ((*(it.next().unwrap()) as u32) << remain))
                    .collect();
            }
        }
        if remain > 0 {
            // let mut j =0;
            // bitpack.finish_read_byte();
            processed += remain as i32;
            dec_scl = 2.0f64.powi(processed);
            for (int_comp, dec_comp) in int_vec.iter().zip(dec_vec.iter()) {
                cur_intf = *int_comp as f64;
                // if j<10{
                //     println!("{}th item {}, decimal:{}",j, cur_f,*dec_comp);
                // }
                // j += 1;
                expected_datapoints.push(
                    cur_intf
                        + (((*dec_comp) | (bitpack.read_bits(remain as usize).unwrap() as u32))
                            as f64)
                            / dec_scl,
                );
            }
        }
        expected_datapoints
    }
}
