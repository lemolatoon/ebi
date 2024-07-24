use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    mem::align_of,
    path::Path,
};

#[allow(unused_imports)]
use ebi::compressor::{
    gorilla::GorillaCompressor, run_length::RunLengthCompressor,
    uncompressed::UncompressedCompressor,
};

use ebi::{
    compressor::CompressorConfig,
    decoder::{
        chunk_reader::{GeneralChunkReaderInner, Reader},
        FileReader,
    },
    encoder::{ChunkOption, FileWriter},
    io::aligned_buf_reader::AlignedBufReader,
};
use rand::Rng;

fn generate_and_write_random_f64(path: impl AsRef<Path>, n: usize) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut random_values: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if rng.gen_bool(0.5) && random_values.last().is_some() {
            random_values.push(random_values[i - 1]);
        } else {
            random_values.push(rng.gen());
        }
    }

    let mut file = File::create(path)?;

    for value in random_values {
        let bytes = value.to_ne_bytes();
        file.write_all(&bytes)?;
    }

    Ok(())
}

fn main() {
    const RECORD_COUNT: usize = 100;
    generate_and_write_random_f64("uncompressed.bin", RECORD_COUNT * 300 + 3).unwrap();
    // let compressor = GenericCompressor::Uncompressed(UncompressedCompressor::new(100));
    // let compressor = GenericCompressor::RLE(RunLengthCompressor::new());
    let compressor_config = CompressorConfig::gorilla().build();
    let in_f = File::open("uncompressed.bin").unwrap();
    let mut in_f = AlignedBufReader::new(in_f);
    let mut out_f = File::create("compressed.bin").unwrap();
    let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);
    let mut file_context = FileWriter::new(&mut in_f, compressor_config.into(), chunk_option);

    // write header leaving footer offset blank
    file_context.write_header(&mut out_f).unwrap();

    // loop until there are no data left
    let mut compressor = None;
    while let Some(mut chunk_writer) = file_context.chunk_writer_with_compressor(compressor.take())
    {
        chunk_writer.write(&mut out_f).unwrap();
        // you can flush if you want
        out_f.flush().unwrap();

        compressor = Some(chunk_writer.into_compressor());
    }

    file_context.write_footer(&mut out_f).unwrap();
    let footer_offset_slot_offset = file_context.footer_offset_slot_offset();
    out_f
        .seek(SeekFrom::Start(footer_offset_slot_offset as u64))
        .unwrap();
    file_context.write_footer_offset(&mut out_f).unwrap();
    let elapsed_time_slot_offset = file_context.elapsed_time_slot_offset();
    out_f
        .seek(SeekFrom::Start(elapsed_time_slot_offset as u64))
        .unwrap();
    file_context.write_elapsed_time(&mut out_f).unwrap();
    // you can flush if you want
    out_f.flush().unwrap();

    drop(in_f);
    drop(out_f);
    // ================== DECODING ==================
    let mut in_f = File::open("compressed.bin").unwrap(); // in_f: Read
    println!(
        "compressed.bin: {} Bytes, {} KiB, {} MiB",
        in_f.metadata().unwrap().len(),
        in_f.metadata().unwrap().len() as f64 / 1024.0,
        in_f.metadata().unwrap().len() as f64 / 1024.0 / 1024.0
    );
    let mut file_reader = FileReader::new();
    let file_header = *file_reader.fetch_header(&mut in_f).unwrap();
    // All fields can be gotten by getter
    println!(
        "magic_number: {:?}",
        file_header.magic_number().map(|x| x as char)
    );

    file_reader.seek_to_footer(&mut in_f).unwrap();

    let file_footer = file_reader.fetch_footer(&mut in_f).unwrap();

    // Also they are getters for footer.
    println!("number_of_records: {}", file_footer.number_of_records());
    println!(
        "compression scheme: {:?}",
        file_header.config().compression_scheme()
    );
    println!(
        "compressed in {} ns, {} ms",
        file_footer.compression_elapsed_time_nano_secs(),
        file_footer.compression_elapsed_time_nano_secs() as f64 / 1_000_000.0
    );

    let file_metadata = file_reader.into_metadata().unwrap();

    // reading chunk in given buffer
    let mut chunk_handles = file_metadata.chunks_iter().peekable();
    chunk_handles
        .peek()
        .unwrap()
        .seek_to_chunk(&mut in_f)
        .unwrap();
    let mut buf = Vec::new();
    for (i, chunk_handle) in chunk_handles.enumerate() {
        // you can skip this chunk if you like
        if i == 1 {
            chunk_handle.seek_to_chunk_end(&mut in_f).unwrap();
            continue;
        }
        let chunk_size = chunk_handle.chunk_size() as usize;
        if buf.len() < chunk_size.next_multiple_of(align_of::<u64>()) + 1 {
            buf.resize(chunk_size.next_multiple_of(align_of::<u64>()) + 1, 0);
        }

        let mut chunk_reader = chunk_handle.make_chunk_reader(&mut in_f).unwrap();
        #[allow(irrefutable_let_patterns)]
        if let GeneralChunkReaderInner::Uncompressed(ref mut chunk_reader) =
            chunk_reader.inner_mut()
        {
            let mut in_f = File::open("uncompressed.bin").unwrap();
            in_f.seek(SeekFrom::Start(
                (chunk_handle.chunk_footer().logical_offset() + 2)
                    * std::mem::size_of::<f64>() as u64,
            ))
            .unwrap();
            let mut buf: [u8; std::mem::size_of::<f64>()] = [0; std::mem::size_of::<f64>()];
            in_f.read_exact(&mut buf).unwrap();
            assert_eq!(
                chunk_reader.decompress().unwrap()[2],
                f64::from_ne_bytes(buf)
            );
        };
        // Compression Method Specific Operations
    }
}
