use alp_binding::{compress_double, decompress_double};

fn main() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut buffer = Vec::new();
    compress_double(&input, &mut buffer);

    let mut decompressed = Vec::new();
    unsafe { decompress_double(&buffer, input.len(), &mut decompressed) };
    assert_eq!(input, decompressed);

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut buffer = Vec::new();
    compress_double(&input, &mut buffer);
    let mut decompressed = Vec::new();
    unsafe { decompress_double(&buffer, input.len(), &mut decompressed) };
    assert_eq!(input, decompressed);
}
