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

    println!("Compression and decompression functional!");

    #[cfg(feature = "speed-test")]
    {
        use core_affinity;
        use rand::Rng;
        use std::arch::x86_64::_rdtsc;
        use std::time::Instant;

        const N_GB: usize = 3;
        const ONE_GB_IN_BYTES: usize = 1_073_741_824 * N_GB;
        const ELEMENT_SIZE: usize = std::mem::size_of::<f64>();
        const NUM_ELEMENTS: usize = ONE_GB_IN_BYTES / ELEMENT_SIZE;

        println!("[speed-test] Generating {}GB of random f64 data...", N_GB);
        let mut rng = rand::rng();
        let input: Vec<f64> = (0..NUM_ELEMENTS).map(|_| rng.random()).collect();

        if let Some(core_id) = core_affinity::get_core_ids().and_then(|ids| ids.get(0).cloned()) {
            core_affinity::set_for_current(core_id);
        }

        let duration = Instant::now().elapsed(); // fake

        // Compression
        let mut compressed = Vec::new();
        let start_cycles = unsafe { _rdtsc() };
        let start = Instant::now();
        compress_double(&input, &mut compressed);
        let duration = start.elapsed();
        let end_cycles = unsafe { _rdtsc() };
        let cycles = end_cycles - start_cycles;

        let secs = duration.as_secs_f64();
        let throughput_gbps = (ONE_GB_IN_BYTES as f64) / secs / 1e9;
        let tuples_per_cycle = NUM_ELEMENTS as f64 / (cycles as f64);

        println!("Compression time: {:.3} sec", secs);
        println!("Compressed size: {} bytes", compressed.len());
        println!("Compression cycles: {}", cycles);
        println!("Compression throughput: {:.3} GB/s", throughput_gbps);
        println!("Compression tuples per cycle: {:.6}", tuples_per_cycle);

        // Decompression
        let mut decompressed = Vec::new();
        let start_cycles = unsafe { _rdtsc() };
        let start = Instant::now();
        unsafe { decompress_double(&compressed, input.len(), &mut decompressed) };
        let duration = start.elapsed();
        let end_cycles = unsafe { _rdtsc() };
        let cycles = end_cycles - start_cycles;

        let secs = duration.as_secs_f64();
        let throughput_gbps = (ONE_GB_IN_BYTES as f64) / secs / 1e9;
        let tuples_per_cycle = NUM_ELEMENTS as f64 / (cycles as f64);

        println!("Decompression time: {:.3} sec", secs);
        println!("Decompression cycles: {}", cycles);
        println!("Decompression throughput: {:.3} GB/s", throughput_gbps);
        println!("Decompression tuples per cycle: {:.6}", tuples_per_cycle);

        assert_eq!(input, decompressed);
        println!("[speed-test] Decompressed data matches original!");
    }
}
