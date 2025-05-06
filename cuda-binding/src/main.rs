use cuda_binding::Context;

fn column_major(ctx: &mut Context) {
    println!("========= Column Major =========");
    let m = 2;
    let n = 4;
    let k = 3;

    let mut a = Vec::new(); // m x k: 2 x 3
    for i in 0..6 {
        a.push(i as f64);
    }
    let mut b = Vec::new(); // k x n: 3 x 4
    for i in 6..(6 + 12) {
        b.push(i as f64);
    }
    let mut c = vec![0.0; (m * n) as usize];

    ctx.gemm(&a, &b, &mut c, Context::config_matmul_column_major(m, n, k))
        .expect("gemm failed");
    println!("C: {:?}", c);

    let mut expected = vec![0.0; (m * n) as usize];
    for col in 0..n {
        for row in 0..m {
            for p in 0..k {
                expected[col * m + row] += a[p * m + row] * b[col * k + p];
            }
        }
    }
    println!("Expected: {:?}", expected);

    if c == expected {
        println!("GEMM result is correct!");
    } else {
        println!("GEMM result is incorrect!");
    }
}

fn transpose(a: &Vec<f64>, m: usize, n: usize) -> Vec<f64> {
    let mut transposed = vec![0.0; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            transposed[j * m + i] = a[i * n + j];
        }
    }
    transposed
}

fn row_major(ctx: &mut Context) {
    println!("========= Row Major =========");
    let m = 2;
    let n = 4;
    let k = 3;

    let mut a = Vec::new(); // m x k: 2 x 3
    for i in 0..6 {
        a.push(i as f64);
    }
    let a = transpose(&a, k, m); // column major to row major
    let mut b = Vec::new(); // k x n: 3 x 4
    for i in 6..(6 + 12) {
        b.push(i as f64);
    }
    let b = transpose(&b, n, k); // column major to row major
    let mut c = vec![0.0; (m * n) as usize];

    ctx.gemm(&b, &a, &mut c, Context::config_matmul_row_major(m, n, k))
        .expect("gemm failed");
    println!("C: {:?}", c);

    let mut expected = vec![0.0; (m * n) as usize];
    for row in 0..m {
        for col in 0..n {
            for p in 0..k {
                expected[row * n + col] += a[row * k + p] * b[p * n + col];
            }
        }
    }
    println!("Expected: {:?}", expected);

    if c == expected {
        println!("GEMM result is correct!");
    } else {
        println!("GEMM result is incorrect!");
    }
}

fn main() {
    println!("Hello, world!");

    let device_id = 0;
    let mut ctx = Context::new_at(device_id).expect("Failed to create context");
    column_major(&mut ctx);
    row_major(&mut ctx);
}
