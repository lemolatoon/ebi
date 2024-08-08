// ceil(index * log_2(10))
const F: [u32; 21] = [
    0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67,
];

const MAP_10I_P: [f64; 21] = [
    1.0, 1.0E1, 1.0E2, 1.0E3, 1.0E4, 1.0E5, 1.0E6, 1.0E7, 1.0E8, 1.0E9, 1.0E10, 1.0E11, 1.0E12,
    1.0E13, 1.0E14, 1.0E15, 1.0E16, 1.0E17, 1.0E18, 1.0E19, 1.0E20,
];

const MAP_10I_N: [f64; 21] = [
    1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7, 1.0E-8, 1.0E-9, 1.0E-10, 1.0E-11,
    1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15, 1.0E-16, 1.0E-17, 1.0E-18, 1.0E-19, 1.0E-20,
];

const MAP_SP_GREATER_1: [u64; 10] = [
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
];

const MAP_SP_LESS_1: [f64; 11] = [
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

pub fn get_f_alpha(alpha: u32) -> u32 {
    if alpha as usize >= F.len() {
        (alpha as f64 * std::f64::consts::LOG2_10).ceil() as u32
    } else {
        F[alpha as usize]
    }
}

pub fn get_alpha_and_beta_star(v: f64, last_beta_star: i32) -> (u32, i32) {
    let abs_v = v.abs();
    let (sp, flag) = get_sp_and_10i_n_flag(abs_v);
    let beta = get_significant_count(abs_v, sp, last_beta_star);
    let alpha = (beta - sp - 1) as u32;
    let beta_star = if flag == 1 { 0 } else { beta };
    (alpha, beta_star)
}

pub fn round_up(v: f64, alpha: u32) -> f64 {
    let scale = get_10i_p(alpha);
    if v < 0.0 {
        (v * scale).floor() / scale
    } else {
        (v * scale).ceil() / scale
    }
}

fn get_significant_count(v: f64, sp: i32, last_beta_star: i32) -> i32 {
    let mut i = if last_beta_star != i32::MAX && last_beta_star != 0 {
        std::cmp::max(last_beta_star - sp - 1, 1)
    } else if last_beta_star == i32::MAX {
        17 - sp - 1
    } else if sp >= 0 {
        1
    } else {
        -sp
    };

    let mut temp = v * get_10i_p(i as u32);
    let mut temp_i64 = temp as i64;
    while (temp_i64 as f64) != temp {
        i += 1;
        temp = v * get_10i_p(i as u32);
        if temp.is_infinite() {
            return 17;
        }
        temp_i64 = temp as i64;
    }

    // There are some bugs for those with high significand, i.e., 0.23911204406033099
    // So we should further check
    if (temp / get_10i_p(i as u32)) != v {
        17
    } else {
        while i > 0 && temp_i64 % 10 == 0 {
            i -= 1;
            temp_i64 /= 10;
        }
        sp + i + 1
    }
}

fn get_10i_p(i: u32) -> f64 {
    if i as usize >= MAP_10I_P.len() {
        format!("1.0E{i}").parse::<f64>().unwrap()
    } else {
        MAP_10I_P[i as usize]
    }
}

pub fn get_10i_n(i: u32) -> f64 {
    if i as usize >= MAP_10I_N.len() {
        format!("1.0E-{i}").parse::<f64>().unwrap()
    } else {
        MAP_10I_N[i as usize]
    }
}

pub fn get_sp(v: f64) -> i32 {
    if v >= 1.0 {
        let mut i = 0;
        while i < MAP_SP_GREATER_1.len() - 1 {
            if v < MAP_SP_GREATER_1[i + 1] as f64 {
                return i as i32;
            }
            i += 1;
        }
    } else {
        let mut i = 1;
        while i < MAP_SP_LESS_1.len() {
            if v >= MAP_SP_LESS_1[i] {
                return -(i as i32);
            }
            i += 1;
        }
    }
    (v.log10()).floor() as i32
}

fn get_sp_and_10i_n_flag(v: f64) -> (i32, i32) {
    if v >= 1.0 {
        let mut i = 0;
        while i < MAP_SP_GREATER_1.len() - 1 {
            if v < MAP_SP_GREATER_1[i + 1] as f64 {
                return (i as i32, 0);
            }
            i += 1;
        }
    } else {
        let mut i = 1;
        while i < MAP_SP_LESS_1.len() {
            if v >= MAP_SP_LESS_1[i] {
                return (-(i as i32), if v == MAP_SP_LESS_1[i] { 1 } else { 0 });
            }
            i += 1;
        }
    }
    let log10v = v.log10();
    (
        log10v.floor() as i32,
        if log10v == log10v.floor() { 1 } else { 0 },
    )
}
