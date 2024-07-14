use std::{
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    mem::{align_of, size_of},
    path::Path,
};

use ebi::{
    compressor::{uncompressed::UncompressedCompressor, GenericCompressor},
    decoder::{
        chunk_reader::{GeneralChunkReaderInner, Reader},
        FileReader,
    },
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

    drop(in_f);
    drop(out_f);
    // ================== DECODING ==================
    let mut in_f = File::open("compressed.bin").unwrap(); // in_f: Read
    let mut file_reader = FileReader::new();
    let file_header = file_reader.fetch_header(&mut in_f).unwrap();
    // All fields can be gotten by getter
    println!(
        "magic_number: {:?}",
        file_header.magic_number().map(|x| x as char)
    );

    file_reader.seek_to_footer(&mut in_f).unwrap();

    let file_footer = file_reader.fetch_footer(&mut in_f).unwrap();

    // Also they are getters for footer.
    println!("number_of_records: {}", file_footer.number_of_records());
    let chunk_footers = file_footer.chunk_footers();
    println!("chunk_footer[1]: {:?}", chunk_footers[1]);

    // reading chunk in given buffer
    let chunk_handles = file_reader.chunks_iter().unwrap();
    let mut buf = Vec::new();
    for (i, mut chunk_handle) in chunk_handles.enumerate() {
        // you can skip this chunk if you like
        if i == 1 {
            continue;
        }
        let chunk_size = chunk_handle.chunk_size() as usize;
        if buf.len() * size_of::<u64>() < chunk_size {
            buf.resize(
                chunk_size.next_multiple_of(align_of::<u64>()) / size_of::<u64>() + 1,
                0,
            );
        }

        chunk_handle.seek_to_chunk(&mut in_f).unwrap();
        chunk_handle.fetch(&mut in_f, &mut buf[..]).unwrap();
        let mut chunk_reader = chunk_handle.make_chunk_reader(&buf[..]).unwrap();
        #[allow(irrefutable_let_patterns)]
        let GeneralChunkReaderInner::Uncompressed(ref mut chunk_reader) = chunk_reader.inner_mut() else {
            panic!("expect uncompressed chunk");
        };
        println!(
            "{i}: chunk_header: {:?}",
            chunk_reader
                .read_header()
                .header()
                .iter()
                .map(|x| *x as char)
                .collect::<String>()
        );
        println!(
            "logical offset of chunk {i}: {}",
            chunk_handle.chunk_footer().logical_offset()
        );
        println!("{i}th chunk data[3]: {:?}", chunk_reader.decompress()[3]);
        let mut in_f = File::open("uncompressed.bin").unwrap();
        in_f.seek(SeekFrom::Start(
            (chunk_handle.chunk_footer().logical_offset() + 3) * std::mem::size_of::<f64>() as u64,
        ))
        .unwrap();
        let mut buf: [u8; std::mem::size_of::<f64>()] = [0; std::mem::size_of::<f64>()];
        in_f.read_exact(&mut buf).unwrap();
        println!("expected data[3]: {:?}", f64::from_ne_bytes(buf));
        // Compression Method Specific Operations
    }
}
