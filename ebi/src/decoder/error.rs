use cfg_if::cfg_if;
use thiserror::Error;

use crate::format;

#[derive(Error, Debug)]
pub enum DecoderError {
    /// IO error.
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    /// This error occurs when the conversion from byte slice to struct fails.
    #[error("Conversion error: {0}")]
    ConversionError(#[from] format::deserialize::ConversionError),
    #[error("Unexpected end of chunk reached while parsing")]
    UnexpectedEndOfChunk,
    #[error("Preconditions not met")]
    PreconditionsNotMet,
}
cfg_if!(
    if #[cfg(debug_assertions)] {
        pub type Result<T> = anyhow::Result<T>;
    } else {

        pub type Result<T> = std::result::Result<T, DecoderError>;
    }
);
