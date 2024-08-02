use thiserror::Error;

use super::{AppendableCompressor, Compressor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EstimateOption {
    Exact = 0,
    Overestimate,
    BestEffort,
}

impl EstimateOption {
    pub fn is_stricter_than(&self, other: &Self) -> bool {
        let lhs = *self as u8;
        let rhs = *other as u8;

        lhs < rhs
    }

    pub fn is_stricter_or_equal(&self, other: &Self) -> bool {
        let lhs = *self as u8;
        let rhs = *other as u8;

        lhs <= rhs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SizeEstimatorError {
    #[error("End of buffer reached by advance/unload: cursor {{advance, load}}able: {0}")]
    EndOfBuffer(usize),
}

pub type Result<T> = std::result::Result<T, SizeEstimatorError>;

/// Interface for Estimater,
/// This estimater estimates the value over the buffer.
/// With advance, we can load the next value from the buffer,
/// With unload_value, we can rewind the loaded value, if possible usually, unload buffer is size 1
/// With compress method, leveraging the obtained statistics and internal Compressor object, do the actual compression
pub trait SizeEstimator {
    /// Estimate the size of the compressed bytes for the loaded uncompressed values
    fn size(&self) -> usize;
    /// Load the next n values from the buffer
    fn advance_n(&mut self, n: usize) -> Result<()>;
    /// Load the next value from the buffer
    fn advance(&mut self) -> Result<()> {
        self.advance_n(1)
    }
    /// Unload the last loaded value, typically unloaded buffer is size 1
    /// 2nd call of unload_value usually returns error
    fn unload_value(&mut self) -> Result<()>;
    /// Number of records advanced
    fn number_of_records_advanced(&self) -> usize;
    /// Return the inner buffer
    fn inner_buffer(&self) -> &[f64];
    /// Return the estimate option
    fn estimate_option(&self) -> EstimateOption;
    /// Compress the loaded values, and return the size of the compressed bytes
    fn compress(self) -> usize;
}

/// A naive size estimater that estimates the size by do the actual compression
#[derive(Debug, PartialEq, PartialOrd)]
pub struct NaiveSlowSizeEstimator<'comp, 'buf, C: Compressor + Clone> {
    compressor: &'comp mut C,
    buffer: &'buf [f64],
    estimate_option: EstimateOption,
    cursor: usize,
    number_of_records_advanced: usize,
}

impl<'comp, 'buf, C: Compressor + Clone> NaiveSlowSizeEstimator<'comp, 'buf, C> {
    pub fn new(
        compressor: &'comp mut C,
        buffer: &'buf [f64],
        estimate_option: EstimateOption,
    ) -> Self {
        Self {
            compressor,
            buffer,
            estimate_option,
            cursor: 0,
            number_of_records_advanced: 0,
        }
    }

    fn loaded_buffer(&self) -> &[f64] {
        &self.buffer[..self.cursor]
    }
}

impl<C: Compressor + Clone> SizeEstimator for NaiveSlowSizeEstimator<'_, '_, C> {
    fn size(&self) -> usize {
        let mut compressor = self.compressor.clone();
        compressor.compress(self.loaded_buffer());
        compressor.total_bytes_buffered()
    }

    fn advance_n(&mut self, n: usize) -> Result<()> {
        if self.cursor + n > self.buffer.len() {
            return Err(SizeEstimatorError::EndOfBuffer(
                self.buffer.len() - self.cursor,
            ));
        }

        self.cursor += n;
        self.number_of_records_advanced += n;

        Ok(())
    }

    fn unload_value(&mut self) -> Result<()> {
        if self.cursor == 0 {
            return Err(SizeEstimatorError::EndOfBuffer(0));
        }

        self.cursor -= 1;
        self.number_of_records_advanced -= 1;

        Ok(())
    }

    fn number_of_records_advanced(&self) -> usize {
        self.number_of_records_advanced
    }

    fn inner_buffer(&self) -> &[f64] {
        &self.buffer
    }

    fn estimate_option(&self) -> EstimateOption {
        self.estimate_option
    }

    fn compress(self) -> usize {
        let Self {
            compressor,
            buffer,
            cursor,
            ..
        } = self;
        let loaded_buffer = &buffer[..cursor];
        compressor.compress(loaded_buffer);
        compressor.total_bytes_buffered()
    }
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct StaticSizeEstimator<'comp, 'buf, C: Compressor + Clone> {
    compressor: &'comp mut C,
    buffer: &'buf [f64],
    estimate_option: EstimateOption,
    cursor: usize,
    number_of_records_advanced: usize,
}

impl<'comp, 'buf, C: Compressor + Clone> StaticSizeEstimator<'comp, 'buf, C> {
    pub fn new(
        compressor: &'comp mut C,
        buffer: &'buf [f64],
        estimate_option: EstimateOption,
    ) -> Self {
        Self {
            compressor,
            buffer,
            estimate_option,
            cursor: 0,
            number_of_records_advanced: 0,
        }
    }

    fn loaded_buffer(&self) -> &[f64] {
        &self.buffer[..self.cursor]
    }
}

impl<C: Compressor + Clone> SizeEstimator for StaticSizeEstimator<'_, '_, C> {
    fn size(&self) -> usize {
        self.compressor
            .estimate_size_static(self.number_of_records_advanced, self.estimate_option)
            .unwrap()
    }

    fn advance_n(&mut self, n: usize) -> Result<()> {
        if self.cursor + n > self.buffer.len() {
            return Err(SizeEstimatorError::EndOfBuffer(
                self.buffer.len() - self.cursor,
            ));
        }

        self.cursor += n;
        self.number_of_records_advanced += n;

        Ok(())
    }

    fn unload_value(&mut self) -> Result<()> {
        if self.cursor == 0 {
            return Err(SizeEstimatorError::EndOfBuffer(0));
        }

        self.cursor -= 1;
        self.number_of_records_advanced -= 1;

        Ok(())
    }

    fn number_of_records_advanced(&self) -> usize {
        self.number_of_records_advanced
    }

    fn inner_buffer(&self) -> &[f64] {
        &self.buffer
    }

    fn estimate_option(&self) -> EstimateOption {
        self.estimate_option
    }

    fn compress(self) -> usize {
        let Self {
            compressor,
            buffer,
            cursor,
            ..
        } = self;
        let loaded_buffer = &buffer[..cursor];
        compressor.compress(loaded_buffer);
        compressor.total_bytes_buffered()
    }
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct AppendCompressingSizeEstimator<'comp, 'buf, C: AppendableCompressor> {
    compressor: &'comp mut C,
    buffer: &'buf [f64],
    estimate_option: EstimateOption,
    cursor: usize,
    number_of_records_advanced: usize,
    size: usize,
}

impl<'comp, 'buf, C: AppendableCompressor> AppendCompressingSizeEstimator<'comp, 'buf, C> {
    pub fn new(
        compressor: &'comp mut C,
        buffer: &'buf [f64],
        estimate_option: EstimateOption,
    ) -> Self {
        let size = compressor.total_bytes_buffered();
        Self {
            compressor,
            buffer,
            estimate_option,
            cursor: 0,
            number_of_records_advanced: 0,
            size,
        }
    }

    fn loaded_buffer(&self) -> &[f64] {
        &self.buffer[..self.cursor]
    }
}

impl<C: AppendableCompressor> SizeEstimator for AppendCompressingSizeEstimator<'_, '_, C> {
    fn size(&self) -> usize {
        self.size
    }

    fn advance_n(&mut self, n: usize) -> Result<()> {
        if self.cursor + n > self.buffer.len() {
            return Err(SizeEstimatorError::EndOfBuffer(
                self.buffer.len() - self.cursor,
            ));
        }

        dbg!(self.cursor, n);
        self.compressor
            .append_compress(&self.buffer[self.cursor..self.cursor + n]);

        self.cursor += n;
        self.number_of_records_advanced += n;
        self.size = self.compressor.total_bytes_buffered();

        Ok(())
    }

    fn unload_value(&mut self) -> Result<()> {
        if self.cursor == 0 {
            return Err(SizeEstimatorError::EndOfBuffer(0));
        }
        if self.compressor.rewind(1) {
            self.cursor -= 1;
            self.number_of_records_advanced -= 1;
            self.size = self.compressor.total_bytes_buffered();
            Ok(())
        } else {
            Err(SizeEstimatorError::EndOfBuffer(0))
        }
    }

    fn number_of_records_advanced(&self) -> usize {
        self.number_of_records_advanced
    }

    fn inner_buffer(&self) -> &[f64] {
        self.buffer
    }

    fn estimate_option(&self) -> EstimateOption {
        self.estimate_option
    }

    fn compress(self) -> usize {
        // Already compressed by `advance_n`
        self.compressor.total_bytes_buffered()
    }
}

#[cfg(test)]
mod tests {
    use crate::compressor::{CompressorConfig, GenericCompressor, GenericSizeEstimator};

    use super::*;

    #[test]
    fn test_is_stricter_than() {
        assert!(EstimateOption::Exact.is_stricter_than(&EstimateOption::Overestimate));
        assert!(EstimateOption::Exact.is_stricter_than(&EstimateOption::BestEffort));
        assert!(!EstimateOption::Overestimate.is_stricter_than(&EstimateOption::Exact));
        assert!(EstimateOption::Overestimate.is_stricter_than(&EstimateOption::BestEffort));
        assert!(!EstimateOption::BestEffort.is_stricter_than(&EstimateOption::Exact));
        assert!(!EstimateOption::BestEffort.is_stricter_than(&EstimateOption::Overestimate));

        assert!(!EstimateOption::Exact.is_stricter_than(&EstimateOption::Exact));
        assert!(!EstimateOption::Overestimate.is_stricter_than(&EstimateOption::Overestimate));
        assert!(!EstimateOption::BestEffort.is_stricter_than(&EstimateOption::BestEffort));
    }

    #[test]
    fn test_is_stricter_or_equal() {
        assert!(EstimateOption::Exact.is_stricter_or_equal(&EstimateOption::Overestimate));
        assert!(EstimateOption::Exact.is_stricter_or_equal(&EstimateOption::BestEffort));
        assert!(!EstimateOption::Overestimate.is_stricter_or_equal(&EstimateOption::Exact));
        assert!(EstimateOption::Overestimate.is_stricter_or_equal(&EstimateOption::BestEffort));
        assert!(!EstimateOption::BestEffort.is_stricter_or_equal(&EstimateOption::Exact));
        assert!(!EstimateOption::BestEffort.is_stricter_or_equal(&EstimateOption::Overestimate));

        assert!(EstimateOption::Exact.is_stricter_or_equal(&EstimateOption::Exact));
        assert!(EstimateOption::Overestimate.is_stricter_or_equal(&EstimateOption::Overestimate));
        assert!(EstimateOption::BestEffort.is_stricter_or_equal(&EstimateOption::BestEffort));
    }

    fn test_size(mut comp: GenericCompressor, estimate_option: EstimateOption) {
        let mut se = comp
            .size_estimater(
                &[
                    0.5, 1.9, 1000.8, -2323.3, 33.7, 22.5, 907.8, 298.9, 298.9, 28.0, 0.9, 2182.2,
                ],
                estimate_option,
            )
            .unwrap();
        let mut size = 0;
        for _ in 0..5 {
            se.advance().unwrap();
            assert!(se.size() >= size, "size should be monotonically increasing");
            size = se.size();
        }
        let estimate_option = se.estimate_option();
        let compressed_size = se.compress();
        let actual_compressed_size = comp.total_bytes_buffered();
        assert_eq!(
            compressed_size, actual_compressed_size,
            "compressed size returned by SizeEstimator::compress should be equal to the actual compressed size"
        );

        match estimate_option {
            EstimateOption::Exact => {
                assert_eq!(size, compressed_size, "estimated size should be exact");
            }
            EstimateOption::Overestimate => {
                assert!(
                    size <= compressed_size,
                    "estimated size should be overestimated"
                );
            }
            EstimateOption::BestEffort => {}
        }
    }

    fn test_rewind(mut comp: GenericCompressor, estimate_option: EstimateOption) {
        let mut se = comp
            .size_estimater(
                &[
                    0.5, 1.9, 1000.8, -2323.3, 33.7, 22.5, 907.8, 298.9, 298.9, 28.0, 0.9, 2182.2,
                ],
                estimate_option,
            )
            .unwrap();
        let mut sizes = Vec::new();
        sizes.push(se.size());
        for _ in 0..5 {
            se.advance().unwrap();
            sizes.push(se.size());
        }

        for i in 0..5 {
            sizes.pop();
            if se.unload_value().is_err() {
                assert_ne!(
                    i, 0,
                    "unload_value should not return error for the first time"
                );
                break;
            };
            let size = se.size();
            let expected_size = *sizes.last().unwrap();
            println!("size: {}, expected_size: {}", size, expected_size);
            assert_eq!(size, expected_size, "size should be the same after rewind");
        }

        for _ in 0..5 {
            se.advance().unwrap();
            sizes.push(se.size());
        }

        for i in 0..5 {
            sizes.pop();
            if se.unload_value().is_err() {
                assert_ne!(
                    i, 0,
                    "unload_value should not return error for the first time"
                );
                break;
            };
            let size = se.size();
            let expected_size = *sizes.last().unwrap();
            assert_eq!(size, expected_size, "size should be the same after rewind");
        }

        let estimate_option = se.estimate_option();
        let estimated_size = se.size();
        let compressed_size = se.compress();

        match estimate_option {
            EstimateOption::Exact => {
                assert_eq!(
                    estimated_size, compressed_size,
                    "estimated size should be exact"
                );
            }
            EstimateOption::Overestimate => {
                assert!(
                    estimated_size <= compressed_size,
                    "estimated size should be overestimated"
                );
            }
            EstimateOption::BestEffort => {}
        }
    }

    #[test]
    fn test_uncompressed_estimator() {
        let comp: CompressorConfig = CompressorConfig::uncompressed()
            .header("hii".as_bytes().to_vec().into_boxed_slice())
            .build()
            .into();
        let options = [
            EstimateOption::Exact,
            EstimateOption::Overestimate,
            EstimateOption::BestEffort,
        ];
        for &estimate_option in &options {
            test_size(comp.clone().build(), estimate_option);
            test_rewind(comp.clone().build(), estimate_option);
        }
    }

    #[test]
    fn test_rle_estimator() {
        let comp: CompressorConfig = CompressorConfig::rle().build().into();
        let options = [
            EstimateOption::Exact,
            EstimateOption::Overestimate,
            EstimateOption::BestEffort,
        ];
        for &estimate_option in &options {
            test_size(comp.clone().build(), estimate_option);
            test_rewind(comp.clone().build(), estimate_option);
        }
    }

    #[test]
    fn test_gorilla_estimator() {
        let comp: CompressorConfig = CompressorConfig::gorilla().build().into();
        let options = [
            EstimateOption::Exact,
            EstimateOption::Overestimate,
            EstimateOption::BestEffort,
        ];
        for &estimate_option in &options {
            test_size(comp.clone().build(), estimate_option);
            test_rewind(comp.clone().build(), estimate_option);
        }
    }
}
