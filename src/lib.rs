pub mod compressor;
pub mod core;
pub mod decoder;
pub mod encoder;
pub mod format;

#[cfg(test)]
mod tests {
    use std::{
        env,
        io::{self, BufRead, Seek, SeekFrom, Write},
        mem::{align_of, size_of},
    };

    use rand::Rng;

    use crate::{
        compressor::{uncompressed::UncompressedCompressor, GenericCompressor},
        decoder::{
            chunk_reader::{GeneralChunkReaderInner, Reader},
            FileReader,
        },
        encoder::{ChunkOption, FileWriter},
    };

    fn generate_and_write_random_f64<W: Write>(mut f: W, n: usize) -> io::Result<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let random_values: Vec<f64> = (0..n).map(|_| rng.gen()).collect();

        for value in random_values.clone() {
            let bytes = value.to_ne_bytes();
            f.write_all(&bytes)?;
        }

        Ok(random_values)
    }

    fn write_file<R: BufRead, W: Write + Seek>(
        mut file_writer: FileWriter<R>,
        mut out_f: W,
    ) -> io::Result<()> {
        // write header leaving footer offset blank
        file_writer.write_header(&mut out_f)?;

        let mut compressor = None;
        // loop until there are no data left
        while let Some(mut chunk_writer) =
            file_writer.chunk_writer_with_compressor(compressor.take())
        {
            chunk_writer.write(&mut out_f)?;
            out_f.flush()?;

            compressor = Some(chunk_writer.into_compressor());
        }

        file_writer.write_footer(&mut out_f)?;
        let footer_offset_slot_offset = file_writer.footer_offset_slot_offset();
        out_f.seek(SeekFrom::Start(footer_offset_slot_offset as u64))?;
        file_writer.write_footer_offset(&mut out_f)?;
        // you can flush if you want
        out_f.flush()?;

        Ok(())
    }

    #[test]
    fn test_round_trip() {
        let header = b"my_header".to_vec().into_boxed_slice();
        let compressor = GenericCompressor::Uncompressed(
            UncompressedCompressor::new(8000).header(header.clone()),
        );
        test_round_trip_for_compressor(compressor);

        // let compressor = GenericCompressor::RLE(RunLengthCompressor::new());
        // test_round_trip_for_compressor(compressor);
    }

    fn test_round_trip_for_compressor(compressor: GenericCompressor) {
        let record_count = 1003;

        let buf = Vec::<u8>::new();
        let mut in_f = io::Cursor::new(buf);

        let random_values = generate_and_write_random_f64(&mut in_f, record_count).unwrap();

        let mut in_f = io::Cursor::new(in_f.into_inner());
        let mut out_f = io::Cursor::new(Vec::<u8>::new());

        const RECORD_COUNT: usize = 100;
        let chunk_option = ChunkOption::RecordCount(RECORD_COUNT);
        let file_writer = FileWriter::new(&mut in_f, compressor, chunk_option);

        write_file(file_writer, &mut out_f).unwrap();

        let compressed_buf = out_f.into_inner();
        let mut in_f = io::Cursor::new(compressed_buf);

        let mut file_reader = FileReader::new();
        let file_header = file_reader.fetch_header(&mut in_f).unwrap();
        assert_eq!(
            file_header.magic_number(),
            b"EBI1",
            "magic number must be EBI1"
        );
        assert_eq!(
            file_header.version(),
            &[
                env!("CARGO_PKG_VERSION_MAJOR"),
                env!("CARGO_PKG_VERSION_MINOR"),
                env!("CARGO_PKG_VERSION_PATCH")
            ]
            .map(|x| x.parse::<u16>().unwrap()),
            "version must be equal to the version of this crate"
        );
        assert_ne!(
            file_header.footer_offset(),
            0,
            "footer offset must not be zero"
        );

        file_reader.seek_to_footer(&mut in_f).unwrap();

        let file_footer = file_reader.fetch_footer(&mut in_f).unwrap();
        assert_eq!(
            file_footer.number_of_records(),
            record_count as u64,
            "number of records must be equal to the number of records written"
        );

        let chunk_footers = file_footer.chunk_footers();
        dbg!(&chunk_footers);
        assert_eq!(
            chunk_footers.len(),
            (record_count + RECORD_COUNT - 1) / RECORD_COUNT,
            "number of chunk footers must be equal to the number of chunks, which is the number of records divided by the record count"
        );
        for (i, chunk_footer) in chunk_footers.iter().enumerate() {
            assert_eq!(
                chunk_footer.logical_offset(),
                (i * RECORD_COUNT) as u64,
                "logical offset of chunk must be equal to the offset of the chunk in the file"
            )
        }

        // reading chunk in given buffer
        let chunk_handles = file_reader.chunks_iter().unwrap();
        let mut buf = Vec::new();
        for (i, mut chunk_handle) in chunk_handles.enumerate() {
            let chunk_size = chunk_handle.chunk_size() as usize;
            let expected_chunk_size =
                chunk_size.next_multiple_of(align_of::<u64>()) / size_of::<u64>() + 1;
            if buf.len() * size_of::<u64>() < expected_chunk_size {
                buf.resize(expected_chunk_size, 0);
            }

            chunk_handle.seek_to_chunk(&mut in_f).unwrap();
            chunk_handle.fetch(&mut in_f, &mut buf[..]).unwrap();
            let mut chunk_reader = chunk_handle.make_chunk_reader(&buf[..]).unwrap();
            #[allow(irrefutable_let_patterns)]
            let GeneralChunkReaderInner::Uncompressed(ref mut chunk_reader) =
                chunk_reader.inner_mut()
            else {
                panic!("expect uncompressed chunk, but got: {:?}", chunk_reader);
            };
            let _chunk_header = chunk_reader.read_header();

            let written_data = if random_values.len() < (i + 1) * RECORD_COUNT {
                &random_values[i * RECORD_COUNT..]
            } else {
                &random_values[i * RECORD_COUNT..(i + 1) * RECORD_COUNT]
            };
            let decompressed_data = chunk_reader.decompress();

            assert_eq!(
                written_data, decompressed_data,
                "written data must be equal to decompressed data"
            );
        }
    }
}
