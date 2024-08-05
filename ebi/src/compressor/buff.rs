use crate::{
    compression_common::buff::precision_bound::{self, PRECISION_MAP},
    compressor::size_estimater::SizeEstimatorError,
};

use super::{
    size_estimater::{self, EstimateOption, SizeEstimator},
    Compressor,
};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BUFFCompressor {
    total_bytes_in: usize,
    data: Vec<f64>,
    scale: usize,
    compressed: Option<Vec<u8>>,
}

impl BUFFCompressor {
    pub fn new(scale: usize) -> Self {
        Self {
            total_bytes_in: 0,
            data: Vec::new(),
            scale,
            compressed: None,
        }
    }

    fn compress_with_precalculated(&mut self, precalculated: Precalculated) {
        let number_of_records = precalculated.number_of_records();
        self.compressed = Some(internal::buff_simd256_encode(precalculated));

        let n_bytes_compressed = number_of_records * size_of::<f64>();
        self.total_bytes_in += n_bytes_compressed;
    }
}

impl Compressor for BUFFCompressor {
    type SizeEstimatorImpl<'comp, 'buf> = BUFFSizeEstimator<'comp, 'buf>;

    fn compress(&mut self, data: &[f64]) {
        self.compress_with_precalculated(Precalculated::precalculate(self.scale, data));
    }

    fn size_estimator<'comp, 'buf>(
        &'comp mut self,
        input: &'buf [f64],
        estimate_option: EstimateOption,
    ) -> Option<Self::SizeEstimatorImpl<'comp, 'buf>> {
        Some(BUFFSizeEstimator::new(estimate_option, input, self))
    }

    fn total_bytes_in(&self) -> usize {
        self.total_bytes_in
    }

    fn total_bytes_buffered(&self) -> usize {
        self.compressed.as_ref().map_or(0, |v| v.len())
    }

    fn prepare(&mut self) {}

    fn buffers(&self) -> [&[u8]; super::MAX_BUFFERS] {
        const EMPTY: &[u8] = &[];
        let data = self.compressed.as_ref().map_or(EMPTY, |v| v.as_slice());
        [data, EMPTY, EMPTY, EMPTY, EMPTY]
    }

    fn reset(&mut self) {
        self.total_bytes_in = 0;
        self.data.clear();
        self.compressed = None;
    }
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct BUFFSizeEstimator<'comp, 'buf> {
    comp: &'comp mut BUFFCompressor,
    buffer: &'buf [f64],
    estimate_option: EstimateOption,
    cursor: usize,
    prev_min_max: Option<(usize, usize)>,
    precalculated: Precalculated,
}

impl<'comp, 'buf> BUFFSizeEstimator<'comp, 'buf> {
    pub fn new(
        estimate_option: EstimateOption,
        buffer: &'buf [f64],
        comp: &'comp mut BUFFCompressor,
    ) -> Self {
        let scale = comp.scale;
        Self {
            comp,
            buffer,
            estimate_option,
            cursor: 0,
            prev_min_max: None,
            precalculated: Precalculated::new(scale),
        }
    }
}

impl<'comp, 'buf> SizeEstimator for BUFFSizeEstimator<'comp, 'buf> {
    fn size(&self) -> usize {
        self.precalculated.compressed_size()
    }

    fn advance_n(&mut self, n: usize) -> size_estimater::Result<()> {
        if self.buffer.len() < self.cursor + n {
            return Err(SizeEstimatorError::EndOfBuffer(
                self.buffer.len() - self.cursor,
            ));
        }
        let appending_values = &self.buffer[self.cursor..self.cursor + n];

        self.precalculated.append_values(&appending_values[..n - 1]);
        self.prev_min_max = Some((self.precalculated.min, self.precalculated.max));
        self.precalculated.append_values(&appending_values[n - 1..]);
        self.cursor += n;

        Ok(())
    }

    fn unload_value(&mut self) -> size_estimater::Result<()> {
        if self.cursor == 0 {
            return Err(SizeEstimatorError::EndOfBuffer(0));
        }

        let (min, max) = match self.prev_min_max {
            Some((min, max)) => (min, max),
            None => {
                return Err(SizeEstimatorError::EndOfBuffer(0));
            }
        };

        self.cursor -= 1;
        self.precalculated.rewind_values(1, min, max);

        self.prev_min_max = None;

        Ok(())
    }

    fn number_of_records_advanced(&self) -> usize {
        self.cursor
    }

    fn inner_buffer(&self) -> &[f64] {
        self.buffer
    }

    fn estimate_option(&self) -> EstimateOption {
        self.estimate_option
    }

    fn compress(self) -> usize {
        self.comp.compress_with_precalculated(self.precalculated);

        self.comp.total_bytes_buffered()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Precalculated {
    min: usize,
    max: usize,
    fixed_representation_values: Vec<i64>,
    precision: usize,
}

impl Precalculated {
    pub fn new(scale: usize) -> Self {
        let precision = if scale == 0 {
            0
        } else {
            (scale as f32).log10() as usize
        };
        Self {
            min: 0,
            max: 0,
            fixed_representation_values: Vec::new(),
            precision,
        }
    }

    pub fn rewind_values(&mut self, n: usize, prev_min: usize, prev_max: usize) {
        if n >= self.number_of_records() {
            self.min = 0;
            self.max = 0;
            self.fixed_representation_values.clear();
            return;
        }

        self.min = prev_min;
        self.max = prev_max;

        self.fixed_representation_values
            .truncate(self.number_of_records() - n);
    }

    pub fn append_values(&mut self, values: &[f64]) {
        let mut min = Precalculated::min_fixed(self).unwrap_or(i64::MAX);
        let mut min_index = self.min;
        let mut max = Precalculated::max_fixed(self).unwrap_or(i64::MIN);
        let mut max_index = self.max;

        let fractional_part_bits_length = self.fractional_part_bits_length() as i32;
        let number_of_records = self.number_of_records();

        for (i, f) in values
            .iter()
            .enumerate()
            .map(|(i, &f)| (i + number_of_records, f))
        {
            let fixed = precision_bound::into_fixed_representation_with_fractional_part_bits_length(
                f,
                fractional_part_bits_length,
            );
            if fixed < min {
                min = fixed;
                min_index = i;
            }
            if fixed > max {
                max = fixed;
                max_index = i;
            }
            self.fixed_representation_values.push(fixed);
        }

        self.min = min_index;
        self.max = max_index;
    }

    pub fn precalculate(scale: usize, floats: &[f64]) -> Self {
        let mut fixed_representation_values = Vec::new();

        let precision = if scale == 0 {
            0
        } else {
            (scale as f32).log10() as usize
        };

        let fractional_part_bits_length = *PRECISION_MAP.get(precision).unwrap();
        let mut min = i64::MAX;
        let mut min_index = 0;
        let mut max = i64::MIN;
        let mut max_index = 0;

        for (i, &f) in floats.iter().enumerate() {
            let fixed = precision_bound::into_fixed_representation_with_fractional_part_bits_length(
                f,
                fractional_part_bits_length as i32,
            );
            if fixed < min {
                min = fixed;
                min_index = i;
            }
            if fixed > max {
                max = fixed;
                max_index = i;
            }
            fixed_representation_values.push(fixed);
        }

        Self {
            min: min_index,
            max: max_index,
            fixed_representation_values,
            precision,
        }
    }

    #[inline]
    pub fn max_fixed(&self) -> Option<i64> {
        self.fixed_representation_values.get(self.max).copied()
    }

    #[inline]
    pub fn min_fixed(&self) -> Option<i64> {
        self.fixed_representation_values.get(self.min).copied()
    }

    #[inline]
    pub fn delta(&self) -> Option<i64> {
        match (self.min_fixed(), self.max_fixed()) {
            (Some(min), Some(max)) => Some(max - min),
            _ => None,
        }
    }

    pub fn fixed_representation_bits_length(&self) -> Option<u32> {
        let delta = self.delta()?;
        Some(if delta == 0 {
            0
        } else {
            64 - delta.leading_zeros()
        })
    }

    pub fn fixed_representation_values(&self) -> &[i64] {
        &self.fixed_representation_values[..]
    }

    pub fn fractional_part_bits_length(&self) -> u64 {
        *PRECISION_MAP.get(self.precision).unwrap()
    }

    pub fn number_of_records(&self) -> usize {
        self.fixed_representation_values.len()
    }

    pub fn compressed_size(&self) -> usize {
        let number_of_records = self.number_of_records();

        let header_size = size_of::<u32>() * 5;
        if number_of_records == 0 {
            return header_size;
        }

        let fixed_representation_bits_length =
            self.fixed_representation_bits_length().unwrap() as usize;

        let fixed_floats_size =
            (fixed_representation_bits_length * number_of_records).next_multiple_of(8) / 8;

        header_size + fixed_floats_size
    }
}

mod internal {
    use std::mem;

    use crate::compression_common::buff::{bit_packing::BitPack, flip};

    use super::Precalculated;

    pub fn buff_simd256_encode(precalculated: Precalculated) -> Vec<u8> {
        let fixed_representation_values = precalculated.fixed_representation_values();
        let base_fixed64 = precalculated.min_fixed().unwrap();

        let delta_fixed_representation_values = fixed_representation_values
            .iter()
            .map(|x| (x - base_fixed64) as u64)
            .collect::<Vec<u64>>();

        let fixed_representation_bits_length =
            precalculated.fixed_representation_bits_length().unwrap();
        let fractional_part_bits_length = precalculated.fractional_part_bits_length() as usize;
        let number_of_records = precalculated.number_of_records() as u32;

        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(precalculated.compressed_size());

        let base_fixed64_bits: u64 = unsafe { mem::transmute(base_fixed64) };
        let base_fixed64_low = base_fixed64_bits as u32;
        let base_fixed64_high = (base_fixed64_bits >> 32) as u32;

        // Safety:
        // The following unwraps are safe because bits length is always less than 32
        unsafe {
            bitpack_vec.write(base_fixed64_low, 32).unwrap_unchecked();
            bitpack_vec.write(base_fixed64_high, 32).unwrap_unchecked();
            bitpack_vec.write(number_of_records, 32).unwrap_unchecked();
            bitpack_vec
                .write(fixed_representation_bits_length, 32)
                .unwrap_unchecked();
            bitpack_vec
                .write(fractional_part_bits_length as u32, 32)
                .unwrap_unchecked();
        }

        let mut remaining_bits_length = fixed_representation_bits_length as usize;

        if 0 < remaining_bits_length && remaining_bits_length < 8 {
            for delta_fixed in delta_fixed_representation_values {
                bitpack_vec
                    .write_bits(delta_fixed as u32, remaining_bits_length)
                    .unwrap();
            }
        } else {
            while remaining_bits_length >= 8 {
                remaining_bits_length -= 8;
                if remaining_bits_length > 0 {
                    for &delta_fixed in &delta_fixed_representation_values {
                        bitpack_vec.write_byte(flip((delta_fixed >> remaining_bits_length) as u8));
                    }
                } else {
                    for &delta_fixed in &delta_fixed_representation_values {
                        bitpack_vec.write_byte(flip(delta_fixed as u8));
                    }
                }
            }
            if remaining_bits_length > 0 {
                bitpack_vec.finish_write_byte();
                for d in delta_fixed_representation_values {
                    bitpack_vec
                        .write_bits(d as u32, remaining_bits_length)
                        .unwrap();
                }
            }
        }
        bitpack_vec.into_vec()
    }
}
