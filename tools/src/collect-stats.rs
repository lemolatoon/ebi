use anyhow::{Context, Result};
use glob::{glob, GlobError};
use serde::Serialize;
use std::{
    collections::HashMap,
    collections::HashSet,
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

fn get_decimal_precision(s: &str) -> usize {
    if let Some(e_pos) = s.find('e') {
        let (base, exponent) = s.split_at(e_pos);
        let exponent_value: i32 = exponent[1..].parse().unwrap_or(0);
        if let Some(dot_pos) = base.find('.') {
            let decimal_length = base.len() - dot_pos - 1;
            (decimal_length as i32 - exponent_value).max(0) as usize
        } else if exponent_value < 0 {
            (-exponent_value) as usize
        } else {
            0
        }
    } else if let Some(dot_pos) = s.find('.') {
        s.len() - dot_pos - 1
    } else {
        0
    }
}

#[derive(Debug, Serialize)]
struct Stats {
    // Decimal precision stats
    decimal_precision_max: usize,
    decimal_precision_min: usize,
    decimal_precision_avg: f64,
    decimal_precision_std: f64,

    // Value stats
    value_max: f64,
    value_min: f64,
    value_avg: f64,
    value_std: f64,

    // IEE754 exponent stats
    exponent_max: i32,
    exponent_min: i32,
    exponent_avg: f64,
    exponent_std: f64,

    // Average leading/trailing zeros in XOR of consecutive values
    xor_leading_zeros_avg: f64,
    xor_trailing_zeros_avg: f64,

    // Ratio of repeated values to total values
    non_unique_value_ratio: f64,

    num_values: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <glob-pattern>", args[0]);
        std::process::exit(1);
    }
    let pattern = &args[1];

    let mut results = HashMap::new();

    // 修正: 進捗を [x/total] の形式で表示するために enumerate() を使用
    let files: Vec<Result<PathBuf, GlobError>> = glob(pattern)?.collect();
    let total = files.len();
    for (i, entry) in files.into_iter().enumerate() {
        match entry {
            Ok(path) => {
                if let Some(stem_os) = path.file_stem() {
                    let stem = stem_os.to_string_lossy().to_string();
                    eprintln!(
                        "[{}/{}] Collecting stats for: {}",
                        i + 1,
                        total,
                        path.display()
                    );
                    match collect_stats_from_csv(&path) {
                        Ok(stats) => {
                            results.insert(stem, stats);
                            eprintln!(
                                "[{}/{}] Finished collecting stats for: {}",
                                i + 1,
                                total,
                                path.display()
                            );
                        }
                        Err(e) => {
                            eprintln!("Failed to collect stats from {}: {:?}", path.display(), e);
                        }
                    }
                }
            }
            Err(e) => eprintln!("Failed to read path: {e:?}"),
        }
    }

    let out_file = "stats.json";
    let json_str = serde_json::to_string_pretty(&results)?;
    std::fs::write(out_file, json_str)?;
    eprintln!("Wrote final JSON to {out_file}");

    Ok(())
}

fn collect_stats_from_csv(csv_path: &Path) -> Result<Stats> {
    let file = File::open(csv_path)
        .with_context(|| format!("Failed to open file: {}", csv_path.display()))?;
    let reader = BufReader::new(file);

    let mut decimal_precisions = Vec::new();
    let mut values = Vec::new();
    let mut exponents = Vec::new();

    // CSV行を読み込んで解析
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line #{}", lineno + 1))?;
        let items = line.split(',').filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        });

        for s in items {
            // precision を文字列から取得
            let prec = get_decimal_precision(&s);
            decimal_precisions.push(prec);

            // 数値値をパース
            let val = s.parse::<f64>().with_context(|| {
                format!("Failed to parse '{}' as f64 in line #{}", s, lineno + 1)
            })?;
            values.push(val);

            // Extract IEE754 exponent
            let exponent_bits = get_exponent_bits(val);
            exponents.push(exponent_bits);
        }
    }

    Ok(compute_stats(&decimal_precisions, &values, &exponents))
}

fn compute_stats(decimal_precisions: &[usize], values: &[f64], exponents: &[i32]) -> Stats {
    if values.is_empty() {
        return Stats {
            decimal_precision_max: 0,
            decimal_precision_min: 0,
            decimal_precision_avg: 0.0,
            decimal_precision_std: 0.0,
            value_max: 0.0,
            value_min: 0.0,
            value_avg: 0.0,
            value_std: 0.0,
            exponent_max: 0,
            exponent_min: 0,
            exponent_avg: 0.0,
            exponent_std: 0.0,
            xor_leading_zeros_avg: 64.0,
            xor_trailing_zeros_avg: 64.0,
            num_values: 0,
            non_unique_value_ratio: 0.0,
        };
    }

    let &dec_max = decimal_precisions.iter().max().unwrap();
    let &dec_min = decimal_precisions.iter().min().unwrap();
    let dec_mean = mean_usize(decimal_precisions);
    let dec_std_dev = std_dev_usize(decimal_precisions, dec_mean);

    let &value_max = values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let &value_min = values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let value_mean = mean_f64(values);
    let value_std_dev = std_dev_f64(values, value_mean);

    // IEE754 exponent stats
    let &exponent_max = exponents.iter().max().unwrap();
    let &exponent_min = exponents.iter().min().unwrap();
    let exponent_mean = mean_i32(exponents);
    let exponent_std_dev = std_dev_i32(exponents, exponent_mean);

    // XOR bits leading/trailing zeros
    let (leading_avg, trailing_avg) = xor_front_trail_stats(values);

    let non_unique_ratio = get_non_unique_value_ratio(values);

    Stats {
        decimal_precision_max: dec_max,
        decimal_precision_min: dec_min,
        decimal_precision_avg: dec_mean,
        decimal_precision_std: dec_std_dev,
        value_max,
        value_min,
        value_avg: value_mean,
        value_std: value_std_dev,
        exponent_max,
        exponent_min,
        exponent_avg: exponent_mean,
        exponent_std: exponent_std_dev,
        xor_leading_zeros_avg: leading_avg,
        xor_trailing_zeros_avg: trailing_avg,
        num_values: values.len(),
        non_unique_value_ratio: non_unique_ratio,
    }
}

fn mean_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        let sum: usize = values.iter().sum();
        sum as f64 / values.len() as f64
    }
}

fn std_dev_usize(values: &[usize], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let sum_sq_diff = values
        .iter()
        .map(|&v| {
            let diff = v as f64 - mean;
            diff * diff
        })
        .sum::<f64>();
    (sum_sq_diff / values.len() as f64).sqrt()
}

fn mean_f64(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    if values.is_empty() {
        0.0
    } else {
        sum / values.len() as f64
    }
}

fn std_dev_f64(values: &[f64], m: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mut sum_sq_diff = 0.0;
    for &v in values {
        let diff = v - m;
        sum_sq_diff += diff * diff;
    }
    (sum_sq_diff / values.len() as f64).sqrt()
}

fn xor_front_trail_stats(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (64.0, 64.0);
    }

    let mut leading_sum = 0u64;
    let mut trailing_sum = 0u64;
    for pair in values.windows(2) {
        let bits1 = pair[0].to_le_bytes();
        let bits2 = pair[1].to_le_bytes();
        let x1 = u64::from_le_bytes(bits1);
        let x2 = u64::from_le_bytes(bits2);
        let xored = x1 ^ x2;
        leading_sum += xored.leading_zeros() as u64;
        trailing_sum += xored.trailing_zeros() as u64;
    }
    let count = (values.len() - 1) as f64;
    (leading_sum as f64 / count, trailing_sum as f64 / count)
}

fn get_exponent_bits(value: f64) -> i32 {
    let bits = value.to_bits();
    let exponent_raw = ((bits >> 52) & 0x7ff) as i32;
    // Subtract the double-precision bias (1023)
    exponent_raw - 1023
}

fn mean_i32(values: &[i32]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        let sum: i32 = values.iter().sum();
        sum as f64 / values.len() as f64
    }
}

fn std_dev_i32(values: &[i32], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let sum_sq_diff = values
        .iter()
        .map(|&v| {
            let diff = v as f64 - mean;
            diff * diff
        })
        .sum::<f64>();
    (sum_sq_diff / values.len() as f64).sqrt()
}

fn get_non_unique_value_ratio(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let distinct_values: HashSet<u64> = values.iter().map(|v| v.to_bits()).collect();
    let distinct_count = distinct_values.len();
    for v in distinct_values.iter().take(10) {
        eprintln!("Distinct value: {}", f64::from_bits(*v));
    }
    let total_count = values.len();
    println!("Distinct count: {distinct_count}\t/\t{total_count}");
    (total_count as f64 - distinct_count as f64) / total_count as f64
}
