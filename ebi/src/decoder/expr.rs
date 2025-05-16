use roaring::RoaringBitmap;
use std::io::{Cursor, Read};

use crate::decoder::{self, error::DecoderError};

use super::{chunk_reader::GeneralChunkReader, FileMetadataLike};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    #[inline]
    pub fn doit(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Op::Add => lhs + rhs,
            Op::Sub => lhs - rhs,
            Op::Mul => lhs * rhs,
            Op::Div => lhs / rhs,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(f64),
    Binary {
        op: Op,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    ChunkReader(usize),
}

impl Expr {
    pub fn max_chunk_reader_index(&self) -> Option<usize> {
        match self {
            Expr::Literal(_) => None,
            Expr::Binary { lhs, rhs, .. } => {
                let lhs_i = lhs.max_chunk_reader_index();
                let rhs_i = rhs.max_chunk_reader_index();
                match (lhs_i, rhs_i) {
                    (None, None) => None,
                    (None, Some(i)) => Some(i),
                    (Some(i), None) => Some(i),
                    (Some(l), Some(r)) => Some(l.max(r)),
                }
            }
            Expr::ChunkReader(index) => Some(*index),
        }
    }

    pub fn literal(&self) -> Option<f64> {
        if let Self::Literal(l) = self {
            return Some(*l);
        }
        None
    }

    pub fn calculate<T: FileMetadataLike, R: Read>(
        &self,
        readers: &mut Vec<GeneralChunkReader<'_, T, R>>,
        bitmask: &RoaringBitmap,
    ) -> decoder::Result<Vec<f64>> {
        match self {
            Expr::Literal(_) => Err(DecoderError::PreconditionsNotMet.into()),
            Expr::Binary { op, lhs, rhs } => match (lhs.literal(), rhs.literal()) {
                (None, None) => {
                    let lhs = lhs.calculate(readers, bitmask)?;
                    let rhs = rhs.calculate(readers, bitmask)?;
                    Ok(lhs
                        .into_iter()
                        .zip(rhs)
                        .map(|(l, r)| op.doit(l, r))
                        .collect())
                }
                (None, Some(rhs)) => {
                    let lhs = lhs.calculate(readers, bitmask)?;
                    Ok(lhs.into_iter().map(|lhs| op.doit(lhs, rhs)).collect())
                }
                (Some(lhs), None) => {
                    let rhs = rhs.calculate(readers, bitmask)?;
                    Ok(rhs.into_iter().map(|rhs| op.doit(lhs, rhs)).collect())
                }
                (Some(_), Some(_)) => Err(DecoderError::PreconditionsNotMet.into()),
            },
            Expr::ChunkReader(i) => {
                let number_of_records = readers[*i].number_of_records() as usize;
                let mut raw_buffer = Vec::<f64>::with_capacity(number_of_records);
                let raw_buffer_ptr = raw_buffer.as_mut_ptr();
                let raw_buffer_capacity = raw_buffer.capacity();
                std::mem::forget(raw_buffer);
                let len = { // Lifetime of `&mut [u8]`
                    let raw_buffer_byte_slice = unsafe {
                        std::slice::from_raw_parts_mut(raw_buffer_ptr as *mut u8, number_of_records * 8)
                    };

                    let mut writer = Cursor::new(raw_buffer_byte_slice);
                    readers[*i].materialize(&mut writer, Some(bitmask))?;

                   writer.position() as usize
                };

                let buffer_written =
                    unsafe { Vec::from_raw_parts(raw_buffer_ptr, len / 8, raw_buffer_capacity) };
                Ok(buffer_written)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::{
        api::{
            decoder::{ChunkId, Decoder, DecoderInput},
            encoder::{Encoder, EncoderInput, EncoderOutput},
        },
        compressor::CompressorConfig,
        decoder::Metadata,
        encoder::ChunkOption,
    };

    fn encode(values: &[f64]) -> Cursor<Vec<u8>> {
        let compressor_config: CompressorConfig = CompressorConfig::uncompressed().build().into();
        let encoded = {
            let encoder_input = EncoderInput::from_f64_slice(values);
            let encoder_output = EncoderOutput::from_vec(Vec::new());
            let mut encoder = Encoder::new(
                encoder_input,
                encoder_output,
                ChunkOption::RecordCount(100000),
                compressor_config,
            );
            encoder.encode().unwrap();
            encoder.into_output().into_vec()
        };

        Cursor::new(encoded)
    }

    #[test]
    fn test_add() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mut decoder = Decoder::new(DecoderInput::from_reader(encode(&values))).unwrap();
        let r1 = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        let values = vec![8.0, 7.0, 23.0, 6.0];
        let mut decoder = Decoder::new(DecoderInput::from_reader(encode(&values))).unwrap();
        let r2 = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        let expr = Expr::Binary {
            op: Op::Add,
            lhs: Box::new(Expr::ChunkReader(0)),
            rhs: Box::new(Expr::ChunkReader(1)),
        };
        let mut readers = vec![r1, r2];
        let bitmask = RoaringBitmap::full();
        let result = expr.calculate(&mut readers, &bitmask).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(&result[0..4], &[9.0, 9.0, 26.0, 10.0]);
    }

    // Test Expr::Literal: should immediately return an error.
    #[test]
    fn test_literal_expr() {
        #[allow(clippy::approx_constant)]
        let expr = Expr::Literal(3.14);
        let mut readers: Vec<GeneralChunkReader<Metadata, Cursor<Vec<u8>>>> = vec![];
        let bitmask = RoaringBitmap::full();
        let res = expr.calculate(&mut readers, &bitmask);
        assert!(res.is_err());
    }

    // Test Expr::Binary with both operands as literals.
    #[test]
    fn test_binary_with_literals() {
        let expr = Expr::Binary {
            op: Op::Mul,
            lhs: Box::new(Expr::Literal(2.0)),
            rhs: Box::new(Expr::Literal(3.0)),
        };
        let mut readers: Vec<GeneralChunkReader<Metadata, Cursor<Vec<u8>>>> = vec![];
        let bitmask = RoaringBitmap::full();
        let res = expr.calculate(&mut readers, &bitmask);
        assert!(res.is_err());
    }

    // Test Expr::Binary with left operand as literal and right operand as ChunkReader.
    #[test]
    fn test_binary_literal_and_chunkreader_rhs() {
        // Create a dummy chunk using an encoded vector.
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mut decoder = Decoder::new(DecoderInput::from_reader(encode(&values))).unwrap();
        let r = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        // Expression: Literal (10.0) op Mul with a ChunkReader.
        let expr = Expr::Binary {
            op: Op::Mul,
            lhs: Box::new(Expr::Literal(10.0)),
            rhs: Box::new(Expr::ChunkReader(0)),
        };
        let mut readers = vec![r];
        let bitmask = RoaringBitmap::full();
        let result = expr.calculate(&mut readers, &bitmask).unwrap();
        // Expect every value multiplied by 10.0.
        let expected: Vec<f64> = values.into_iter().map(|v| v * 10.0).collect();
        assert_eq!(result, expected);
    }

    // Test Expr::Binary with left operand as ChunkReader and right operand as literal.
    #[test]
    fn test_binary_chunkreader_and_literal_lhs() {
        // Create a dummy chunk using an encoded vector.
        let values = vec![10.0, 20.0, 30.0];
        let mut decoder = Decoder::new(DecoderInput::from_reader(encode(&values))).unwrap();
        let r = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        // Expression: ChunkReader subtracted by literal 2.0.
        let expr = Expr::Binary {
            op: Op::Sub,
            lhs: Box::new(Expr::ChunkReader(0)),
            rhs: Box::new(Expr::Literal(2.0)),
        };
        let mut readers = vec![r];
        let bitmask = RoaringBitmap::full();
        let result = expr.calculate(&mut readers, &bitmask).unwrap();
        let expected: Vec<f64> = values.into_iter().map(|v| v - 2.0).collect();
        assert_eq!(result, expected);
    }

    // Test Expr::ChunkReader directly.
    #[test]
    fn test_chunkreader_expr() {
        let values = vec![5.0, 6.0, 7.0, 8.0];
        let mut decoder = Decoder::new(DecoderInput::from_reader(encode(&values))).unwrap();
        let r = decoder.chunk_reader(ChunkId::new(0)).unwrap();

        let expr = Expr::ChunkReader(0);
        let mut readers = vec![r];
        // For this test, use a bitmask that only selects even-indexed values (indices: 0 and 2).
        let mut bm = RoaringBitmap::new();
        bm.insert(0);
        bm.insert(2);
        let result = expr.calculate(&mut readers, &bm).unwrap();

        // Expect only the values at indices 0 and 2.
        let expected = vec![5.0, 7.0];
        assert_eq!(result, expected);
    }
}
