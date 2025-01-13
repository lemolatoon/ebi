///////////////////////////////////////////////////////////////
//  共通定義・定数・型定義
///////////////////////////////////////////////////////////////

use std::cmp::{max, min};
use std::collections::HashMap;
use std::f32;
use std::f64;
use std::ops::{Add, Sub};

/// ALPがサポートする圧縮スキーム
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    INVALID,
    ALP_RD,
    ALP,
}

/// コンパイル時設定相当
pub mod config {
    // C++ の alp::config 内の定数と対応
    pub const VECTOR_SIZE: usize = 1024;
    pub const N_VECTORS_PER_ROWGROUP: usize = 100;
    pub const ROWGROUP_SIZE: usize = N_VECTORS_PER_ROWGROUP * VECTOR_SIZE;
    pub const ROWGROUP_VECTOR_SAMPLES: usize = 8;
    pub const ROWGROUP_SAMPLES_JUMP: usize =
        (ROWGROUP_SIZE / ROWGROUP_VECTOR_SAMPLES) / VECTOR_SIZE;
    pub const SAMPLES_PER_VECTOR: usize = 32;
    pub const MAX_K_COMBINATIONS: usize = 5;
    pub const CUTTING_LIMIT: usize = 16;

    /// ALP RD 用
    pub const MAX_RD_DICT_BIT_WIDTH: usize = 3;
    pub const MAX_RD_DICTIONARY_SIZE: usize = 1 << MAX_RD_DICT_BIT_WIDTH; // = 8
}

/// ALP 全体で使われる定数群
pub mod constants {
    use super::*;

    // 例外処理などで使う閾値相当
    pub const SAMPLING_EARLY_EXIT_THRESHOLD: u8 = 2;

    // C++ 側の "ENCODING_UPPER_LIMIT" / "ENCODING_LOWER_LIMIT" は非常に大きな値
    // f32, f64 の範囲内であれば下記を用いる（今回は簡易に i64::MAX / MIN の近傍を代用）
    pub const ENCODING_UPPER_LIMIT: f64 = 9_223_372_036_854_774_784.0; // 2^63 - 9000 程度を想定
    pub const ENCODING_LOWER_LIMIT: f64 = -ENCODING_UPPER_LIMIT;

    // 例外などのビット幅
    pub const DICTIONARY_ELEMENT_SIZE_BYTES: u8 = 2;

    // RD 用
    pub const RD_EXCEPTION_POSITION_SIZE: u8 = 16;
    pub const RD_EXCEPTION_POSITION_SIZE_BYTES: u8 = RD_EXCEPTION_POSITION_SIZE / 8; // 2
    pub const RD_EXCEPTION_SIZE: u8 = 16;
    pub const RD_EXCEPTION_SIZE_BYTES: u8 = RD_EXCEPTION_SIZE / 8; // 2

    // ALP で使用する例外関連
    pub const EXCEPTION_POSITION_SIZE: u8 = 16;
    pub const EXCEPTION_POSITION_SIZE_BYTES: u8 = EXCEPTION_POSITION_SIZE / 8; // 2

    /// ALP RD の閾値 (C++ 側 Constants<float>::RD_SIZE_THRESHOLD_LIMIT = 22 * SAMPLES_PER_VECTOR など)
    /// float/double それぞれ異なるが、ここではダブルに合わせる
    pub const RD_SIZE_THRESHOLD_LIMIT_FLOAT: usize = 22 * config::SAMPLES_PER_VECTOR;
    pub const RD_SIZE_THRESHOLD_LIMIT_DOUBLE: usize = 48 * config::SAMPLES_PER_VECTOR;

    // 浮動小数点をエンコード/デコードする際のマジックナンバー
    // C++ 側で float は 12582912.0, double は 0x0018000000000000 相当
    pub const MAGIC_NUMBER_FLOAT: f32 = 12582912.0;
    pub const MAGIC_NUMBER_DOUBLE: f64 = 0x0018000000000000u64 as f64;
    pub const MAX_EXPONENT_FLOAT: u8 = 10;
    pub const MAX_EXPONENT_DOUBLE: u8 = 18;

    // 10^k テーブル（C++ 側 Constants<T>::EXP_ARR, FRAC_ARR 等を合体）
    // float用 (指数, 逆数)
    pub const EXP_ARR_FLOAT: [f32; 11] = [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0,
    ];
    pub const FRAC_ARR_FLOAT: [f32; 11] = [
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001,
    ];
    /// factor = 10^k の整数版 (float 用)
    pub const FACT_ARR_FLOAT: [i32; 10] = [
        1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
    ];

    // double用
    pub const EXP_ARR_DOUBLE: [f64; 24] = [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0,
        10000000000.0,
        100000000000.0,
        1000000000000.0,
        10000000000000.0,
        100000000000000.0,
        1000000000000000.0,
        10000000000000000.0,
        100000000000000000.0,
        1000000000000000000.0,
        10000000000000000000.0,
        100000000000000000000.0,
        1000000000000000000000.0,
        10000000000000000000000.0,
        100000000000000000000000.0,
    ];
    pub const FRAC_ARR_DOUBLE: [f64; 22] = [
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
        0.0000000001,
        0.00000000001,
        0.000000000001,
        0.0000000000001,
        0.00000000000001,
        0.000000000000001,
        0.0000000000000001,
        0.00000000000000001,
        0.000000000000000001,
        0.0000000000000000001,
        0.00000000000000000001,
        0.000000000000000000001,
    ];
    /// factor = 10^k の整数版 (double 用)
    pub const FACT_ARR_DOUBLE: [i64; 19] = [
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
        10000000000,
        100000000000,
        1000000000000,
        10000000000000,
        100000000000000,
        1000000000000000,
        10000000000000000,
        100000000000000000,
        1000000000000000000,
    ];

    /// 通常 ALP での例外サイズ (float なら 32bit, double なら 64bit など)
    pub const EXCEPTION_SIZE_FLOAT: u8 = 32; // bits
    pub const EXCEPTION_SIZE_BYTES_FLOAT: u8 = EXCEPTION_SIZE_FLOAT / 8;
    pub const EXCEPTION_SIZE_DOUBLE: u8 = 64; // bits
    pub const EXCEPTION_SIZE_BYTES_DOUBLE: u8 = EXCEPTION_SIZE_DOUBLE / 8;
}

/// 簡易ユーティリティ関数: bit幅を数える (ビルトイン命令の模倣)
fn count_bits_u64(x: u64) -> u8 {
    if x == 0 {
        0
    } else {
        64 - x.leading_zeros() as u8
    }
}
fn count_bits_u32(x: u32) -> u8 {
    if x == 0 {
        0
    } else {
        32 - x.leading_zeros() as u8
    }
}

/// 特殊値 (NaN, Inf, -0.0) などをチェックする
/// エンコード不可能判定に使用
fn is_impossible_to_encode_f32(n: f32) -> bool {
    !n.is_finite()
        || n.is_nan()
        || n > constants::ENCODING_UPPER_LIMIT as f32
        || n < constants::ENCODING_LOWER_LIMIT as f32
        || (n == 0.0 && n.to_bits() & 0x80000000 != 0) // -0.0
}

fn is_impossible_to_encode_f64(n: f64) -> bool {
    !n.is_finite()
        || n.is_nan()
        || n > constants::ENCODING_UPPER_LIMIT
        || n < constants::ENCODING_LOWER_LIMIT
        || (n == 0.0 && n.to_bits() & 0x8000000000000000 != 0) // -0.0
}

/// 値域から min, max をとり、その差分のビット幅を求める
/// (Frame of Reference 用の bit幅推定に使用)
fn count_bits_for_for_i64(max_val: i64, min_val: i64) -> u8 {
    let delta = (max_val as u64).wrapping_sub(min_val as u64);
    count_bits_u64(delta)
}

///////////////////////////////////////////////////////////////
//  ALP用の状態管理 (C++ の state<PT> 相当)
///////////////////////////////////////////////////////////////

/// C++ の template <typename PT> state<PT> 相当
#[derive(Debug, Clone)]
pub struct AlpState<T> {
    pub scheme: Scheme,
    pub vector_size: u16,
    pub exceptions_count: u16,
    pub sampled_values_n: usize,

    // ALP
    pub k_combinations: u16,
    pub best_k_combinations: Vec<(i32, i32)>, // (exponent, factor)
    pub exp: u8,
    pub fac: u8,
    pub bit_width: u8,
    pub for_base: T, // Frame of reference の base

    // ALP RD
    pub right_bit_width: u8,
    pub left_bit_width: u8,
    pub right_for_base: u64, // ほぼ0
    pub left_for_base: u16,  // ほぼ0

    // dictionary
    pub left_parts_dict: Vec<u16>, // 実体は stt.left_parts_dict[8] など
    pub actual_dictionary_size: u8,
    pub actual_dictionary_size_bytes: u32,
    // left_parts_dict_map: left_part -> dict_idx
    pub left_parts_dict_map: HashMap<u16, u16>,
}

impl<T> Default for AlpState<T>
where
    T: Default,
{
    fn default() -> Self {
        AlpState {
            scheme: Scheme::INVALID,
            vector_size: config::VECTOR_SIZE as u16,
            exceptions_count: 0,
            sampled_values_n: 0,

            k_combinations: config::MAX_K_COMBINATIONS as u16,
            best_k_combinations: vec![],
            exp: 0,
            fac: 0,
            bit_width: 0,
            for_base: T::default(),

            right_bit_width: 0,
            left_bit_width: 0,
            right_for_base: 0,
            left_for_base: 0,

            left_parts_dict: vec![0; config::MAX_RD_DICTIONARY_SIZE],
            actual_dictionary_size: 0,
            actual_dictionary_size_bytes: 0,
            left_parts_dict_map: HashMap::new(),
        }
    }
}

///////////////////////////////////////////////////////////////
//  sampler 相当 (encoder の init() 内で使用)
///////////////////////////////////////////////////////////////

/// first_level_sample
/// C++ の alp::sampler::first_level_sample と同等
pub fn first_level_sample_f64(
    data: &[f64],
    data_offset: usize,
    data_size: usize,
    data_sample_out: &mut [f64],
) -> usize {
    let left_in_data = data_size - data_offset;
    let portion_to_sample = min(config::ROWGROUP_SIZE, left_in_data);
    let available_alp_vectors =
        (portion_to_sample as f64 / config::VECTOR_SIZE as f64).ceil() as usize;

    let mut sample_idx = 0;
    let mut data_idx = data_offset;

    for vector_idx in 0..available_alp_vectors {
        let current_vector_n_values = min(data_size - data_idx, config::VECTOR_SIZE);

        if (vector_idx % config::ROWGROUP_SAMPLES_JUMP) != 0 {
            data_idx += current_vector_n_values;
            continue;
        }

        let n_sampled_increments = max(
            1,
            (current_vector_n_values as f64 / config::SAMPLES_PER_VECTOR as f64).ceil() as i32,
        ) as usize;

        if current_vector_n_values < config::SAMPLES_PER_VECTOR && sample_idx != 0 {
            data_idx += current_vector_n_values;
            continue;
        }

        let end_idx = data_idx + current_vector_n_values;
        let mut i = data_idx;
        while i < end_idx {
            if sample_idx < data_sample_out.len() {
                data_sample_out[sample_idx] = data[i];
                sample_idx += 1;
            }
            i += n_sampled_increments;
        }
        data_idx += current_vector_n_values;
    }
    sample_idx
}

///////////////////////////////////////////////////////////////
//  ALP デコーダ (C++ の alp::decoder)
///////////////////////////////////////////////////////////////
pub struct AlpDecoder;

impl AlpDecoder {
    /// decode_value: 単体値を factor, exponent で復号する
    /// C++: decoder<PT>::decode_value
    pub fn decode_value_f64(encoded_value: i64, factor: u8, exponent: u8) -> f64 {
        let factor_val = if (factor as usize) < constants::FACT_ARR_DOUBLE.len() {
            constants::FACT_ARR_DOUBLE[factor as usize]
        } else {
            1
        } as f64;
        let frac_val = if (exponent as usize) < constants::FRAC_ARR_DOUBLE.len() {
            constants::FRAC_ARR_DOUBLE[exponent as usize]
        } else {
            1.0
        };
        encoded_value as f64 * factor_val * frac_val
    }

    /// ベクタ一括復号 (scalar バージョン)
    pub fn decode_vector_f64(
        encoded_integers: &[i64],
        fac_idx: u8,
        exp_idx: u8,
        output: &mut [f64],
    ) {
        for i in 0..encoded_integers.len() {
            output[i] = Self::decode_value_f64(encoded_integers[i], fac_idx, exp_idx);
        }
    }

    /// 例外パッチ
    /// out[例外位置] = exceptions[..]
    pub fn patch_exceptions_f64(
        out: &mut [f64],
        exceptions: &[f64],
        exceptions_positions: &[u16],
        exceptions_count: u16,
    ) {
        for i in 0..exceptions_count {
            let pos = exceptions_positions[i as usize] as usize;
            out[pos] = exceptions[i as usize];
        }
    }
}

///////////////////////////////////////////////////////////////
//  ALP エンコーダ (C++ の alp::encoder)
///////////////////////////////////////////////////////////////
pub struct AlpEncoder;

impl AlpEncoder {
    /// is_impossible_to_encode: ALP で扱えない浮動小数点を判定
    #[inline]
    fn is_impossible_to_encode_f64(n: f64) -> bool {
        is_impossible_to_encode_f64(n)
    }

    /// encode_value (C++: encoder<PT>::encode_value)
    /// `SAFE = true` の場合、特殊値チェックを行う
    pub fn encode_value_f64(value: f64, factor_idx: u8, exponent_idx: u8, safe: bool) -> i64 {
        let mut tmp = value;
        // multiply by 10^factor_idx
        let factor_val = if (factor_idx as usize) < constants::EXP_ARR_DOUBLE.len() {
            constants::EXP_ARR_DOUBLE[factor_idx as usize]
        } else {
            1.0
        };
        // multiply by 10^-exponent_idx
        let frac_val = if (exponent_idx as usize) < constants::FRAC_ARR_DOUBLE.len() {
            constants::FRAC_ARR_DOUBLE[exponent_idx as usize]
        } else {
            1.0
        };

        tmp = tmp * factor_val * frac_val;

        if safe {
            if Self::is_impossible_to_encode_f64(tmp) {
                return constants::ENCODING_UPPER_LIMIT as i64;
            }
        }

        let _magic = constants::MAGIC_NUMBER_DOUBLE;
        tmp as i64
    }

    /// Frame-of-Reference 用のビット幅 & min値を算出
    pub fn analyze_ffor_i32(input: &[i32]) -> (u8, i32) {
        let mut minv = i32::MAX;
        let mut maxv = i32::MIN;
        for &val in input {
            if val < minv {
                minv = val;
            }
            if val > maxv {
                maxv = val;
            }
        }
        let bw = count_bits_for_for_i32(maxv, minv);
        (bw, minv)
    }

    pub fn analyze_ffor_i64(input: &[i64]) -> (u8, i64) {
        let mut minv = i64::MAX;
        let mut maxv = i64::MIN;
        for &val in input {
            if val < minv {
                minv = val;
            }
            if val > maxv {
                maxv = val;
            }
        }
        let bw = count_bits_for_for_i64(maxv, minv);
        (bw, minv)
    }

    /// ALP 第一段階サンプリング結果から、rowgroup全体の「上位 k 組合せ (exponent, factor)」を求める
    /// C++: find_top_k_combinations
    fn find_top_k_combinations_f64(sample_arr: &[f64], stt: &mut AlpState<f64>) {
        let samples_size = std::cmp::min(stt.sampled_values_n, config::SAMPLES_PER_VECTOR);
        if samples_size == 0 {
            stt.scheme = Scheme::ALP_RD;
            return;
        }
        let n_vectors_to_sample =
            ((stt.sampled_values_n as f64) / (config::SAMPLES_PER_VECTOR as f64)).ceil() as usize;

        let mut global_combinations: HashMap<(i32, i32), i32> = HashMap::new();
        let mut best_estimated_compression_size: usize = (samples_size
            * (constants::EXCEPTION_SIZE_DOUBLE as usize
                + constants::EXCEPTION_POSITION_SIZE as usize))
            + (samples_size * (constants::EXCEPTION_SIZE_DOUBLE as usize));

        let mut smp_offset = 0;
        for _ in 0..n_vectors_to_sample {
            let mut found_exponent: i8 = 0;
            let mut found_factor: i8 = 0;
            let mut sample_estimated_compression_size = best_estimated_compression_size;

            for exp_ref in (0..=constants::MAX_EXPONENT_DOUBLE as i8).rev() {
                for factor_idx in (0..=exp_ref).rev() {
                    let mut exceptions_count = 0u16;
                    let mut non_exceptions_count = 0u16;
                    let mut max_encoded = i64::MIN;
                    let mut min_encoded = i64::MAX;

                    for i in 0..samples_size {
                        let actual_value = sample_arr[smp_offset + i];
                        let encoded_value = Self::encode_value_f64(
                            actual_value,
                            factor_idx as u8,
                            exp_ref as u8,
                            false,
                        );
                        let decoded_value = AlpDecoder::decode_value_f64(
                            encoded_value,
                            factor_idx as u8,
                            exp_ref as u8,
                        );
                        if (decoded_value - actual_value).abs() < f64::EPSILON {
                            non_exceptions_count += 1;
                            if encoded_value > max_encoded {
                                max_encoded = encoded_value;
                            }
                            if encoded_value < min_encoded {
                                min_encoded = encoded_value;
                            }
                        } else {
                            exceptions_count += 1;
                        }
                    }
                    if non_exceptions_count < 2 {
                        continue;
                    }
                    let bits_per_value = count_bits_for_for_i64(max_encoded, min_encoded);
                    let mut est_compression_size = (samples_size as u32) * (bits_per_value as u32);
                    est_compression_size += (exceptions_count as u32)
                        * (constants::EXCEPTION_SIZE_DOUBLE as u32
                            + constants::EXCEPTION_POSITION_SIZE as u32);

                    let est_compression_size = est_compression_size as usize;
                    if est_compression_size < sample_estimated_compression_size
                        || (est_compression_size == sample_estimated_compression_size
                            && (found_exponent < exp_ref))
                        || ((est_compression_size == sample_estimated_compression_size
                            && found_exponent == exp_ref)
                            && (found_factor < factor_idx))
                    {
                        sample_estimated_compression_size = est_compression_size;
                        found_exponent = exp_ref;
                        found_factor = factor_idx;
                        if sample_estimated_compression_size < best_estimated_compression_size {
                            best_estimated_compression_size = sample_estimated_compression_size;
                        }
                    }
                }
            }
            let cmb = (found_exponent as i32, found_factor as i32);
            *global_combinations.entry(cmb).or_insert(0) += 1;
            smp_offset += samples_size;
            if smp_offset + samples_size > sample_arr.len() {
                break;
            }
        }

        if best_estimated_compression_size >= constants::RD_SIZE_THRESHOLD_LIMIT_DOUBLE {
            stt.scheme = Scheme::ALP_RD;
            return;
        }

        let mut combo_vec: Vec<((i32, i32), i32)> = global_combinations.into_iter().collect();
        combo_vec.sort_by(|a, b| {
            let (ac, bc) = (a.1, b.1);
            if ac != bc {
                bc.cmp(&ac)
            } else {
                let (ae, af) = a.0;
                let (be, bf) = b.0;
                if ae != be {
                    be.cmp(&ae)
                } else {
                    bf.cmp(&af)
                }
            }
        });

        let top_k = std::cmp::min(combo_vec.len(), stt.k_combinations as usize);
        for i in 0..top_k {
            stt.best_k_combinations.push(combo_vec[i].0);
        }
    }

    /// ALP 第二段階: k候補から最適 exponent, factor を絞り込む
    fn find_best_exponent_factor_from_combinations_f64(
        top_combinations: &[(i32, i32)],
        input_vector: &[f64],
        stt: &mut AlpState<f64>,
    ) {
        let input_vector_size = stt.vector_size as usize;
        if top_combinations.is_empty() {
            stt.exp = 0;
            stt.fac = 0;
            return;
        }
        if top_combinations.len() == 1 {
            stt.exp = top_combinations[0].0 as u8;
            stt.fac = top_combinations[0].1 as u8;
            return;
        }

        let mut found_exponent = top_combinations[0].0 as i32;
        let mut found_factor = top_combinations[0].1 as i32;
        let mut best_estimated_compression_size: u64 = u64::MAX;
        let mut worse_threshold_count = 0u8;

        let sample_increments = std::cmp::max(
            1,
            (input_vector_size as f64 / config::SAMPLES_PER_VECTOR as f64).ceil() as i32,
        ) as usize;

        for (exp_i, fac_i) in top_combinations {
            let exp_i = *exp_i as u8;
            let fac_i = *fac_i as u8;
            let mut exceptions_count = 0u64;
            let mut max_encoded = i64::MIN;
            let mut min_encoded = i64::MAX;

            let mut idx = 0;
            while idx < input_vector_size {
                let actual_value = input_vector[idx];
                let encoded_value = Self::encode_value_f64(actual_value, fac_i, exp_i, false);
                let decoded_value = AlpDecoder::decode_value_f64(encoded_value, fac_i, exp_i);
                if (decoded_value - actual_value).abs() < f64::EPSILON {
                    if encoded_value > max_encoded {
                        max_encoded = encoded_value;
                    }
                    if encoded_value < min_encoded {
                        min_encoded = encoded_value;
                    }
                } else {
                    exceptions_count += 1;
                }
                idx += sample_increments;
            }

            let bits_per_value = count_bits_for_for_i64(max_encoded, min_encoded) as u64;
            let mut est_compression_size = (config::SAMPLES_PER_VECTOR as u64) * bits_per_value;
            est_compression_size += exceptions_count
                * ((constants::EXCEPTION_SIZE_DOUBLE + constants::EXCEPTION_POSITION_SIZE) as u64);

            if best_estimated_compression_size == u64::MAX {
                best_estimated_compression_size = est_compression_size;
                found_exponent = exp_i as i32;
                found_factor = fac_i as i32;
                continue;
            }
            if est_compression_size >= best_estimated_compression_size {
                worse_threshold_count += 1;
                if worse_threshold_count == constants::SAMPLING_EARLY_EXIT_THRESHOLD {
                    break;
                }
                continue;
            }
            best_estimated_compression_size = est_compression_size;
            found_exponent = exp_i as i32;
            found_factor = fac_i as i32;
            worse_threshold_count = 0;
        }
        stt.exp = found_exponent as u8;
        stt.fac = found_factor as u8;
    }

    /// ALP ベクタ本体のエンコード (例外配列を生成・書き換え)
    /// C++: encode_simdized (SIMD 化部分は省略して scalar のみ)
    fn encode_simdized_f64(
        input_vector: &[f64],
        exceptions: &mut [f64],
        exceptions_positions: &mut [u16],
        exceptions_count: &mut u16,
        encoded_integers: &mut [i64],
        factor_idx: u8,
        exponent_idx: u8,
    ) {
        let mut current_exceptions_count = 0;
        let mut value_arr_no_specials = vec![0f64; input_vector.len()];
        for (i, &val) in input_vector.iter().enumerate() {
            let is_special = !val.is_finite()
                || val.is_nan()
                || (val == 0.0 && val.to_bits() & 0x8000000000000000 != 0);
            if is_special {
                value_arr_no_specials[i] = constants::ENCODING_UPPER_LIMIT;
            } else {
                value_arr_no_specials[i] = val;
            }
        }

        let mut decoded_arr = vec![0f64; input_vector.len()];

        for i in 0..input_vector.len() {
            let encoded_val =
                Self::encode_value_f64(value_arr_no_specials[i], factor_idx, exponent_idx, false);
            encoded_integers[i] = encoded_val;
            let decoded_val = AlpDecoder::decode_value_f64(encoded_val, factor_idx, exponent_idx);
            decoded_arr[i] = decoded_val;
        }

        let mut tmp_index_arr = vec![0usize; input_vector.len()];
        let mut exceptions_idx = 0;
        for i in 0..input_vector.len() {
            let l = decoded_arr[i];
            let r = value_arr_no_specials[i];
            if (l - r).abs() > f64::EPSILON {
                tmp_index_arr[exceptions_idx] = i;
                exceptions_idx += 1;
            }
        }

        let mut a_non_exception_value = 0;
        for i in 0..input_vector.len() {
            if i >= tmp_index_arr.len() || tmp_index_arr[i] != i {
                a_non_exception_value = encoded_integers[i];
                break;
            }
        }

        for j in 0..exceptions_idx {
            let i = tmp_index_arr[j];
            let actual_value = input_vector[i];
            encoded_integers[i] = a_non_exception_value;
            exceptions[current_exceptions_count] = actual_value;
            exceptions_positions[current_exceptions_count] = i as u16;
            current_exceptions_count += 1;
        }
        *exceptions_count = current_exceptions_count as u16;
    }

    /// ALP の最終エンコード (C++: encode)
    /// ・もし k_combinations > 1 なら二段階で exponent, factor を検索し決定
    /// ・決定した exponent, factor で encode_simdized
    pub fn encode_f64(
        input_vector: &[f64],
        exceptions: &mut [f64],
        exceptions_positions: &mut [u16],
        exceptions_count: &mut [u16],
        encoded_integers: &mut [i64],
        stt: &mut AlpState<f64>,
    ) {
        if stt.k_combinations > 1 {
            // TODO: avoid clone
            let best_k_combinations = stt.best_k_combinations.clone();
            Self::find_best_exponent_factor_from_combinations_f64(
                &best_k_combinations,
                input_vector,
                stt,
            );
        } else if !stt.best_k_combinations.is_empty() {
            stt.exp = stt.best_k_combinations[0].0 as u8;
            stt.fac = stt.best_k_combinations[0].1 as u8;
        } else {
            stt.exp = 0;
            stt.fac = 0;
        }

        Self::encode_simdized_f64(
            input_vector,
            exceptions,
            exceptions_positions,
            &mut exceptions_count[0],
            encoded_integers,
            stt.fac,
            stt.exp,
        );
    }

    /// ALP 初期化 (C++: init)
    /// sampler::first_level_sample -> find_top_k_combinations
    pub fn init_f64(
        data_column: &[f64],
        column_offset: usize,
        tuples_count: usize,
        sample_arr: &mut [f64],
        stt: &mut AlpState<f64>,
    ) {
        stt.scheme = Scheme::ALP;
        let n_smp = first_level_sample_f64(data_column, column_offset, tuples_count, sample_arr);
        stt.sampled_values_n = n_smp;
        stt.k_combinations = config::MAX_K_COMBINATIONS as u16;
        stt.best_k_combinations.clear();
        Self::find_top_k_combinations_f64(sample_arr, stt);
    }
}

///////////////////////////////////////////////////////////////
//  ALP RD エンコーダ/デコーダ (C++ の alp::rd_encoder と decode 相当)
///////////////////////////////////////////////////////////////
pub struct AlpRdEncoder;

impl AlpRdEncoder {
    /// ALP RD における圧縮サイズの推定
    /// C++: estimate_compression_size
    fn estimate_compression_size(
        right_bit_width: u8,
        left_bit_width: u8,
        exceptions_count: u64,
        sample_count: usize,
    ) -> f64 {
        // exceptions_count * (RD_EXCEPTION_POSITION_SIZE + RD_EXCEPTION_SIZE)
        // + right_bit_width + left_bit_width
        let exceptions_bits_per_sample = (exceptions_count as f64)
            * ((constants::RD_EXCEPTION_POSITION_SIZE as f64)
                + (constants::RD_EXCEPTION_SIZE as f64));
        let estimated_size = (right_bit_width as f64)
            + (left_bit_width as f64)
            + (exceptions_bits_per_sample / sample_count as f64);
        estimated_size
    }

    /// 左側ビット列を辞書化する (C++: build_left_parts_dictionary)
    /// PERSIST_DICT=true のときに実際に stt へ辞書を格納
    fn build_left_parts_dictionary_f64(
        in_slice: &[f64],
        candidate_right_bit_width: u8,
        stt: &mut AlpState<f64>,
        persist_dict: bool,
    ) -> f64 {
        let mut left_parts_hash: HashMap<u64, u32> = HashMap::new();

        for &val in in_slice {
            let bits = val.to_bits();
            let left_tmp = bits >> candidate_right_bit_width;
            *left_parts_hash.entry(left_tmp).or_insert(0) += 1;
        }

        let mut left_parts_sorted_repetitions: Vec<(u32, u64)> =
            left_parts_hash.into_iter().map(|(k, v)| (v, k)).collect();
        left_parts_sorted_repetitions.sort_by(|a, b| b.0.cmp(&a.0)); // 出現数降順

        // exceptions_count は、MAX_RD_DICTIONARY_SIZE 以降に落ちるものの出現数の総和
        let mut exceptions_count = 0u32;
        for i in config::MAX_RD_DICTIONARY_SIZE..left_parts_sorted_repetitions.len() {
            exceptions_count += left_parts_sorted_repetitions[i].0;
        }

        let actual_dictionary_size = std::cmp::min(
            config::MAX_RD_DICTIONARY_SIZE,
            left_parts_sorted_repetitions.len(),
        );
        let left_bit_width = std::cmp::max(1, (actual_dictionary_size as f64).log2().ceil() as u8);

        let estimated_size = Self::estimate_compression_size(
            candidate_right_bit_width,
            left_bit_width,
            exceptions_count as u64,
            stt.sampled_values_n,
        );

        if persist_dict {
            stt.left_parts_dict_map.clear();
            for dict_idx in 0..actual_dictionary_size {
                let key = left_parts_sorted_repetitions[dict_idx].1;
                // stt.left_parts_dict[dict_idx] = key の下位16bit
                // ただし double は左側が 64 - right_bit_widthビットあるので
                stt.left_parts_dict[dict_idx] = key as u16;
                stt.left_parts_dict_map.insert(key as u16, dict_idx as u16);
            }
            for i in actual_dictionary_size..left_parts_sorted_repetitions.len() {
                let key = left_parts_sorted_repetitions[i].1;
                // 辞書外のものは actual_dictionary_size 番目 (= 例外枠) へ対応
                stt.left_parts_dict_map
                    .insert(key as u16, actual_dictionary_size as u16);
            }
            stt.left_bit_width = left_bit_width;
            stt.right_bit_width = candidate_right_bit_width;
            stt.actual_dictionary_size = actual_dictionary_size as u8;
            stt.actual_dictionary_size_bytes =
                (actual_dictionary_size as u32) * (constants::DICTIONARY_ELEMENT_SIZE_BYTES as u32);
        }

        estimated_size
    }

    /// best_dictionary を探す (C++: find_best_dictionary)
    fn find_best_dictionary_f64(sample_arr: &[f64], stt: &mut AlpState<f64>) {
        let mut right_bit_width_result = 0;
        let mut best_dict_size = f64::MAX;

        let exact_type_bit_size = 64;

        for i in 1..=config::CUTTING_LIMIT {
            let candidate_right_bit_width = (exact_type_bit_size - i) as u8;
            let estimated_size = Self::build_left_parts_dictionary_f64(
                sample_arr,
                candidate_right_bit_width,
                stt,
                false,
            );
            if estimated_size < best_dict_size {
                right_bit_width_result = candidate_right_bit_width;
                best_dict_size = estimated_size;
            }
        }
        // persist
        Self::build_left_parts_dictionary_f64(sample_arr, right_bit_width_result, stt, true);
    }

    /// ALP RD 用の初期化 (C++: rd_encoder::init)
    pub fn init_f64(
        data_column: &[f64],
        column_offset: usize,
        tuples_count: usize,
        sample_arr: &mut [f64],
        stt: &mut AlpState<f64>,
    ) {
        stt.scheme = Scheme::ALP_RD;
        let n_smp = first_level_sample_f64(data_column, column_offset, tuples_count, sample_arr);
        stt.sampled_values_n = n_smp;
        Self::find_best_dictionary_f64(sample_arr, stt);
    }

    /// ALP RD encode (C++: rd_encoder::encode)
    /// right_parts, left_parts を書き込み
    pub fn encode_f64(
        dbl_arr: &[f64],
        exceptions: &mut [u16],
        exception_positions: &mut [u16],
        exceptions_count_p: &mut [u16],
        right_parts: &mut [u64],
        left_parts: &mut [u16],
        stt: &mut AlpState<f64>,
    ) {
        let mut exceptions_count = 0u16;
        for i in 0..dbl_arr.len() {
            let bits = dbl_arr[i].to_bits();
            let r = bits & ((1u64 << stt.right_bit_width) - 1);
            let l = (bits >> stt.right_bit_width) as u16;
            right_parts[i] = r;
            left_parts[i] = l;
        }

        for i in 0..dbl_arr.len() {
            let dict_key = left_parts[i];
            match stt.left_parts_dict_map.get(&dict_key) {
                Some(&dict_idx) => {
                    // もし dict_idx >= actual_dictionary_size なら例外
                    if dict_idx as usize >= stt.actual_dictionary_size as usize {
                        left_parts[i] = stt.actual_dictionary_size as u16;
                        exceptions[exceptions_count as usize] = dict_key;
                        exception_positions[exceptions_count as usize] = i as u16;
                        exceptions_count += 1;
                    } else {
                        left_parts[i] = dict_idx;
                    }
                }
                None => {
                    // もし見つからなければ例外扱い
                    left_parts[i] = stt.actual_dictionary_size as u16;
                    exceptions[exceptions_count as usize] = dict_key;
                    exception_positions[exceptions_count as usize] = i as u16;
                    exceptions_count += 1;
                }
            }
        }

        stt.exceptions_count = exceptions_count;
        exceptions_count_p[0] = exceptions_count;
    }
}

pub struct AlpRdDecoder;

impl AlpRdDecoder {
    /// ALP RD decode (C++: rd_encoder::decode)
    /// unffor_right_arr, unffor_left_arr を復号して out に格納し、例外をパッチ
    pub fn decode_f64(
        out: &mut [f64],
        unffor_right_arr: &mut [u64],
        unffor_left_arr: &mut [u16],
        exceptions: &[u16],
        exception_positions: &[u16],
        exceptions_count: u16,
        stt: &AlpState<f64>,
    ) {
        // left_parts_dict からビット列を再合成
        for i in 0..out.len() {
            let dict_idx = unffor_left_arr[i];
            let left_val = if (dict_idx as usize) < stt.actual_dictionary_size as usize {
                stt.left_parts_dict[dict_idx as usize]
            } else {
                // 例外の場合、いったん 0 としておく
                // 後でパッチ
                0
            };
            let right_val = unffor_right_arr[i];
            let bits = ((left_val as u64) << stt.right_bit_width) | right_val;
            out[i] = f64::from_bits(bits);
        }

        // 例外をパッチ
        for i in 0..exceptions_count {
            let pos = exception_positions[i as usize] as usize;
            let left = exceptions[i as usize];
            let right = unffor_right_arr[pos];
            let bits = ((left as u64) << stt.right_bit_width) | right;
            out[pos] = f64::from_bits(bits);
        }
    }
}

fn ffor_u64_to_packed(input: &[u64; 1024], output: &mut Vec<u8>, bit_width: u8) -> u64 {
    // 1) min 値を求める
    let min_val = input.iter().copied().min().unwrap_or(0);
    // 2) min_val を引いた一時配列を作る
    let mut tmp = [0u64; 1024];
    for (i, &val) in input.iter().enumerate() {
        let diff = val.wrapping_sub(min_val);
        tmp[i] = diff as u64;
    }
    // 3) fastlanes で bit packing (unchecked_pack)
    //    事前に output のサイズを確保する: bit_width bits * 1024 / 8
    //    (bit_width <= 64 を仮定)
    let needed_bytes = (bit_width as usize * 1024 + 7) / 8;
    output.resize(needed_bytes, 0u8);
    unsafe {
        BitPacking::unchecked_pack(bit_width as usize, &tmp, output);
    }
    min_val
}

fn unffor_packed_to_u64(input: &[u8], output: &mut [u64; 1024], bit_width: u8, base: u64) {
    // 1) unpack 先として一時配列を作る
    let mut tmp = [0u64; 1024];
    unsafe {
        BitPacking::unchecked_unpack(bit_width as usize, input, &mut tmp);
    }
    // 2) unpackした値に base を足して u64 化
    for i in 0..1024 {
        let val = tmp[i] as u64;
        output[i] = val.wrapping_add(base);
    }
}

fn ffor_u16_to_packed(input: &[u16; 1024], output: &mut Vec<u8>, bit_width: u8) -> u16 {
    // 1) min 値を求める
    let min_val = input.iter().copied().min().unwrap_or(0);
    // 2) min_val を引いた一時配列を作る
    let mut tmp = [0u64; 1024];
    for (i, &val) in input.iter().enumerate() {
        let diff = val.wrapping_sub(min_val);
        tmp[i] = diff as u64;
    }
    // 3) fastlanes で bit packing (unchecked_pack)
    //    事前に output のサイズを確保する: bit_width bits * 1024 / 8
    //    (bit_width <= 64 を仮定)
    let needed_bytes = (bit_width as usize * 1024 + 7) / 8;
    output.resize(needed_bytes, 0u8);
    unsafe {
        BitPacking::unchecked_pack(bit_width as usize, &tmp, output);
    }
    min_val
}

fn unffor_packed_to_u16(input: &[u8], output: &mut [u16; 1024], bit_width: u8, base: u16) {
    // 1) unpack 先として一時配列を作る
    let mut tmp = [0u64; 1024];
    unsafe {
        BitPacking::unchecked_unpack(bit_width as usize, input, &mut tmp);
    }
    // 2) unpackした値に base を足して u16 化
    for i in 0..1024 {
        let val = tmp[i] as u16;
        output[i] = val.wrapping_add(base);
    }
}

pub fn alp_encode(data: &[f64]) -> Vec<u8> {
    let mut output = Vec::new();
    // 1) init buffer
    let mut sample_buf = vec![0.0; config::VECTOR_SIZE];
    let mut glue_buf = vec![1.0; config::VECTOR_SIZE];
    let mut sample_arr = vec![0.0; config::VECTOR_SIZE];

    let mut rd_exec_arr = vec![0; config::VECTOR_SIZE];
    let mut pos_arr = vec![0; config::VECTOR_SIZE];
    let mut exec_c_arr = vec![0; config::VECTOR_SIZE];
    let mut right_arr = vec![0; config::VECTOR_SIZE];
    let mut left_arr = vec![0; config::VECTOR_SIZE];

    let mut exec_arr = vec![0; config::VECTOR_SIZE];

    // 2) read data
    let n_tuples = data.len();
    if n_tuples < config::VECTOR_SIZE {
        panic!("// not enough data");
    }

    // 3) 速度計測
    // let bench_speed_result = self.typed_bench_speed_column_f64(&data);

    // 4) rowgroup 単位で圧縮伸張し、メタデータ収集
    let n_vecs = n_tuples / config::VECTOR_SIZE;
    let n_rowgroups = (n_tuples as f64 / config::ROWGROUP_SIZE as f64).ceil() as usize;

    let mut stt = crate::AlpState::<f64>::default();

    for rg_idx in 0..n_rowgroups {
        let n_vec_per_current_rg = if n_rowgroups == 1 {
            // Single row group: all vectors belong to it
            n_vecs
        } else if rg_idx == n_rowgroups - 1 {
            // Last row group: remainder vectors
            n_vecs % config::N_VECTORS_PER_ROWGROUP
        } else {
            // Regular row group
            config::N_VECTORS_PER_ROWGROUP
        };

        let n_values_per_current_rg = n_vec_per_current_rg * config::VECTOR_SIZE;
        let current_rg_slice = &data[rg_idx * config::ROWGROUP_SIZE..][..n_values_per_current_rg];
        AlpEncoder::init_f64(
            current_rg_slice,
            rg_idx,
            n_values_per_current_rg,
            &mut sample_arr,
            &mut stt,
        );

        match stt.scheme {
            Scheme::INVALID => panic!("// invalid scheme"),
            Scheme::ALP_RD => {
                AlpRdEncoder::init_f64(
                    current_rg_slice,
                    0,
                    n_values_per_current_rg,
                    &mut sample_arr,
                    &mut stt,
                );

                for vec_idx in 0..n_vec_per_current_rg {
                    let vec_offset = vec_idx * config::VECTOR_SIZE;
                    let vec_slice = &current_rg_slice[vec_offset..][..config::VECTOR_SIZE];
                    AlpRdEncoder::encode_f64(
                        vec_slice,
                        &mut rd_exec_arr,
                        &mut pos_arr,
                        &mut exec_c_arr,
                        &mut right_arr,
                        &mut left_arr,
                        &mut stt,
                    );

                    ffor_u64_to_packed(&right_arr, &mut output, stt.right_bit_width);
                    ffor_u16_to_packed(&left_arr, &mut output, stt.left_bit_width);
                }
            }
            Scheme::ALP => {
                for vec_idx in 0..n_vec_per_current_rg {
                    let vec_offset = vec_idx * config::VECTOR_SIZE;
                    let vec_slice = &current_rg_slice[vec_offset..][..config::VECTOR_SIZE];
                    AlpEncoder::encode_f64(
                        vec_slice,
                        &mut exec_arr,
                        &mut pos_arr,
                        &mut exec_c_arr,
                        &mut encoded_arr,
                        &mut stt,
                    );
                    let (bit_width, minv) = AlpEncoder::analyze_ffor_i64(encoded_arr);
                    stt.bit_width = bit_width;

                    ffor_u64_to_packed(&encoded_arr, &mut output, bit_width)
                }
            }
        }
    }

    unimplemented!()
}
