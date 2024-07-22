pub mod api;
pub mod compressor;
pub mod core;
pub mod decoder;
pub mod encoder;
pub mod format;
pub mod io;

#[cfg(test)]
mod tests {
    use std::{
        env,
        io::{self, Seek, SeekFrom, Write},
        mem::{align_of, size_of},
    };

    use rand::Rng;

    use crate::{
        compressor::CompressorConfig,
        decoder::FileReader,
        encoder::{ChunkOption, FileWriter},
        io::aligned_buf_reader::{AlignedBufRead, AlignedBufReader},
    };

    fn generate_and_write_random_f64(n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut random_values: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            if rng.gen_bool(0.5) && random_values.last().is_some() {
                random_values.push(random_values[i - 1]);
            } else {
                random_values.push(rng.gen());
            }
        }

        random_values
    }

    fn write_file<R: AlignedBufRead, W: Write + Seek>(
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

        let elapsed_time_slot_offset = file_writer.elapsed_time_slot_offset();
        out_f.seek(SeekFrom::Start(elapsed_time_slot_offset as u64))?;
        file_writer.write_elapsed_time(&mut out_f)?;
        // you can flush if you want
        out_f.flush()?;

        Ok(())
    }

    #[test]
    fn test_round_trip_uncompressed() {
        let header = b"my_header".to_vec().into_boxed_slice();
        let compressor_config = CompressorConfig::uncompressed()
            .capacity(8000)
            .header(header)
            .build();
        for n in [1003, 10003, 100004, 100005] {
            #[cfg(miri)] // miri is too slow
            if n > 1003 {
                continue;
            }

            let random_values = generate_and_write_random_f64(n);
            test_round_trip_for_compressor(&random_values, compressor_config.clone());
        }
    }

    #[test]
    fn test_round_trip_rle() {
        let compressor_config = CompressorConfig::rle().build();

        for n in [1003, 10003, 100004, 100005] {
            #[cfg(miri)] // miri is too slow
            if n > 1003 {
                continue;
            }
            let random_values = generate_and_write_random_f64(n);
            test_round_trip_for_compressor(&random_values, compressor_config);

            let one_value = vec![1.0; n];
            test_round_trip_for_compressor(&one_value, compressor_config);
        }
    }

    #[test]
    fn test_round_trip_gorilla() {
        let compressor_config = CompressorConfig::gorilla().build();

        for n in [1003, 10003, 100004, 100005] {
            #[cfg(miri)] // miri is too slow
            if n > 1003 {
                continue;
            }
            {
                let random_values = generate_and_write_random_f64(n);
                test_round_trip_for_compressor(&random_values, compressor_config);
            }

            {
                let one_value = vec![1.0; n];
                test_round_trip_for_compressor(&one_value, compressor_config);
            }

            {
                let mut doubleing_values = Vec::with_capacity(n);
                for i in 0..n {
                    if i == 0 {
                        doubleing_values.push(1.1);
                        continue;
                    }
                    doubleing_values.push(doubleing_values[i - 1] * 2.0);
                }
                test_round_trip_for_compressor(&doubleing_values, compressor_config);
            }
        }
    }

    fn test_round_trip_for_compressor(values: &[f64], compressor: impl Into<CompressorConfig>) {
        let record_count = values.len();
        const RECORD_COUNT_PER_CHUNK: usize = 1024;

        let buf = Vec::<u8>::new();
        let mut in_f = io::Cursor::new(buf);

        for v in values {
            let bytes = v.to_ne_bytes();
            in_f.write_all(&bytes).unwrap();
        }

        let mut in_f = AlignedBufReader::new(io::Cursor::new(in_f.into_inner()));
        let mut out_f = io::Cursor::new(Vec::<u8>::new());

        let chunk_option = ChunkOption::RecordCount(RECORD_COUNT_PER_CHUNK);
        let file_writer = FileWriter::new(&mut in_f, compressor.into(), chunk_option);

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
        assert_eq!(
            chunk_footers.len(),
            (record_count + RECORD_COUNT_PER_CHUNK - 1) / RECORD_COUNT_PER_CHUNK,
            "number of chunk footers must be equal to the number of chunks, which is the number of records divided by the record count"
        );
        for (i, chunk_footer) in chunk_footers.iter().enumerate() {
            assert_eq!(
                chunk_footer.logical_offset(),
                (i * RECORD_COUNT_PER_CHUNK) as u64,
                "logical offset of chunk must be equal to the offset of the chunk in the file"
            )
        }

        // reading chunk in given buffer
        let chunk_handles = file_reader.chunks_iter().unwrap();
        let mut buf = Vec::new();
        for (i, mut chunk_handle) in chunk_handles.enumerate() {
            let chunk_size = chunk_handle.chunk_size() as usize;
            let expected_chunk_size = chunk_size.next_multiple_of(align_of::<u64>()) + 1;
            if buf.len() * size_of::<u64>() < expected_chunk_size {
                buf.resize(expected_chunk_size.next_power_of_two(), 0);
            }

            chunk_handle.seek_to_chunk(&mut in_f).unwrap();
            chunk_handle.fetch(&mut in_f, &mut buf[..]).unwrap();
            let mut chunk_reader = chunk_handle.make_chunk_reader(&buf[..]).unwrap();

            let written_data = if values.len() < (i + 1) * RECORD_COUNT_PER_CHUNK {
                &values[i * RECORD_COUNT_PER_CHUNK..]
            } else {
                &values[i * RECORD_COUNT_PER_CHUNK..(i + 1) * RECORD_COUNT_PER_CHUNK]
            };

            let decompressed_data = chunk_reader.inner_mut().decompress();

            assert_eq!(
                written_data, decompressed_data,
                "written data must be equal to decompressed data"
            );
        }
    }
}
