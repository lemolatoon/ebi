use cfg_if::cfg_if;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EncoderError {
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}

cfg_if!(
    if #[cfg(debug_assertions)] {
        pub type Result<T> = anyhow::Result<T>;
    } else {

        pub type Result<T> = std::result::Result<T, EncoderError>;
    }
);
