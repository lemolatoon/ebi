use thiserror::Error;

use super::Compressor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SizeEstimatorError {
    #[error("End of buffer reached by advance/unload: cursor {{advance, load}}able: {0}")]
    EndOfBuffer(usize),
}

type Result<T> = std::result::Result<T, SizeEstimatorError>;

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
        compressor.compress(self.loaded_buffer())
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
        compressor.compress(loaded_buffer)
    }
}

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
        compressor.compress(loaded_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_stricter_than() {
        assert!(EstimateOption::Exact.is_stricter_than(&EstimateOption::Overestimate));
        assert!(EstimateOption::Exact.is_stricter_than(&EstimateOption::BestEffort));
        assert!(!EstimateOption::Overestimate.is_stricter_than(&EstimateOption::Exact));
        assert!(EstimateOption::Overestimate.is_stricter_than(&EstimateOption::BestEffort));
        assert!(!EstimateOption::BestEffort.is_stricter_than(&EstimateOption::Exact));
        assert!(!EstimateOption::BestEffort.is_stricter_than(&EstimateOption::Overestimate));
    }
}
