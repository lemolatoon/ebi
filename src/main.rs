use std::{
    fs::File,
    io::{self, BufReader, Seek, SeekFrom, Write},
    path::Path,
};

use ebi::{
    compressor::{uncompressed::UncompressedCompressor, GenericCompressor},
    encoder::{ChunkOption, FileWriter},
};
use rand::Rng;

fn generate_and_write_random_f64(path: impl AsRef<Path>, n: usize) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    let random_values: Vec<f64> = (0..n).map(|_| rng.gen()).collect();

    let mut file = File::create(path)?;

    for value in random_values {
        let bytes = value.to_ne_bytes();
        file.write_all(&bytes)?;
    }

    Ok(())
}

fn main() {
    const RECORD_COUNT: usize = 100;
    generate_and_write_random_f64("uncompressed.bin", RECORD_COUNT * 3 + 3).unwrap();
    let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new());
    let in_f = File::open("uncompressed.bin").unwrap();
    let mut in_f = BufReader::new(in_f);
    let mut out_f = File::create("compressed.bin").unwrap();
    let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);
    let mut file_context = FileWriter::new(&mut in_f, compressor, chunk_option);

    let mut buf = Vec::<u8>::new();
    // write header leaving footer offset blank
    file_context.write_header(&mut out_f, &mut buf).unwrap();

    // loop until there are no data left
    while let Some(mut chunk_context) = file_context.chunk_writer(&mut buf) {
        chunk_context.write(&mut out_f).unwrap();
        // you can flush if you want
        out_f.flush().unwrap();
    }

    file_context.write_footer(&mut out_f, &mut buf).unwrap();
    let footer_offset_slot_offset = file_context.footer_offset_slot_offset();
    out_f
        .seek(SeekFrom::Start(footer_offset_slot_offset as u64))
        .unwrap();
    file_context.write_footer_offset(&mut out_f).unwrap();
    // you can flush if you want
    out_f.flush().unwrap();
}
