use std::mem;

use crate::compression_common::buff::{bit_packing::BitPack, precision_bound::PRECISION_MAP};

pub fn decode_with_precision(bytes: &[u8], precision: u32) -> Vec<f64> {
    let mut bitpack = BitPack::<&[u8]>::new(bytes);
    let ubase_int = bitpack.read(32).unwrap();
    let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    println!("base integer: {}", base_int);
    // Number of records
    let len = bitpack.read_u32().unwrap();
    println!("total vector size:{}", len);
    // Fixed Repr bits length
    let dlen = bitpack.read_u32().unwrap();
    // Fractional part bits length
    let ilen = bitpack.read_u32().unwrap();
    println!("bit packing length:{}", ilen);
    // check integer part and update bitmap;
    let mut cur;

    let mut expected_datapoints: Vec<f64> = Vec::new();
    let mut int_vec: Vec<i32> = Vec::new();
    let mut dec_vec = Vec::new();
    let mut cur_intf;
    let mut remain;
    let mut processed = 0;

    if precision == 0 {
        for _ in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            expected_datapoints.push(cur as f64);
        }
        println!(
            "Number of precision scan items:{}",
            expected_datapoints.len()
        );
        expected_datapoints
    } else {
        let bits_needed = PRECISION_MAP[precision as usize] as u32;
        assert!(dlen >= bits_needed);
        remain = bits_needed;
        let dec_byte = dlen / 8;
        let mut byte_needed = bits_needed / 8;
        let extra_bits = bits_needed % 8;
        if byte_needed < dec_byte && extra_bits > 0 {
            byte_needed += 1;
            remain = byte_needed * 8;
        }
        println!("adjusted dec bits to decode:{}", remain);

        for _ in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            int_vec.push(cur as i32 + base_int);
        }

        let mut dec_scl: f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut bytec = 0;
        let mut chunk;

        if remain >= 8 {
            bytec += 1;
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
            println!("read the {}th byte of dec", bytec);
        }
        while remain >= 8 {
            bytec += 1;
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

            println!("read the {}th byte of dec", bytec);
        }
        if remain > 0 {
            // let mut j =0;
            // bitpack.finish_read_byte();
            println!("read remaining {} bits of dec", remain);
            println!(
                "length for int:{} and length for dec: {}",
                int_vec.len(),
                dec_vec.len()
            );
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
        println!(
            "Number of precision scan items:{}",
            expected_datapoints.len()
        );
        expected_datapoints
    }
}
