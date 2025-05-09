#[cfg(not(feature = "cuda"))]
mod if_cuda {}

#[allow(unused_imports)]
pub use if_cuda::*;
#[cfg(feature = "cuda")]
mod if_cuda {
    use std::{error::Error, sync::Arc};

    pub use cudarc::cublas::safe::GemmConfig;
    use cudarc::{
        cublas::{self, CudaBlas, Gemm, result::CublasError},
        driver::{CudaContext, CudaSlice, CudaStream, DriverError},
    };
    use derive_more::{Display, From};

    #[derive(Debug, Clone, From, Display)]
    pub enum CudaError {
        Driver(DriverError),
        Cublas(CublasError),
        SizeError(String),
    }

    impl Error for CudaError {}

    impl CudaError {
        fn size(msg: impl ToString) -> Self {
            CudaError::SizeError(msg.to_string())
        }
    }

    pub struct Context {
        pub(crate) stream: Arc<CudaStream>,
        pub(crate) blas: CudaBlas,
        pub(crate) a_buf: Option<CudaSlice<f64>>,
        pub(crate) b_buf: Option<CudaSlice<f64>>,
        pub(crate) c_buf: Option<CudaSlice<f64>>,
    }

    impl Context {
        pub fn new_at(device_id: usize) -> Result<Self, CudaError> {
            let ctx = CudaContext::new(device_id)?;
            let stream = ctx.default_stream();
            let blas = CudaBlas::new(Arc::clone(&stream))?;

            Ok(Self {
                stream,
                blas,
                a_buf: None,
                b_buf: None,
                c_buf: None,
            })
        }

        // A: m x k
        // B: k x n
        // C: m x n
        // -> C = A * B
        pub fn gemm(
            &mut self,
            a: &[f64],
            b: &[f64],
            c: &mut [f64],
            cfg: GemmConfig<f64>,
        ) -> Result<(), CudaError> {
            // Size check
            if cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0 {
                return Err(CudaError::size(format!(
                    "Invalid size of m, n, k, should be positive: {} {} {}",
                    cfg.m, cfg.n, cfg.k
                )));
            }
            if a.len() != (cfg.m * cfg.k) as usize {
                return Err(CudaError::size(format!(
                    "Invalid length of A, expected {} but got {}",
                    cfg.m * cfg.k,
                    a.len()
                )));
            }
            if b.len() != (cfg.k * cfg.n) as usize {
                return Err(CudaError::size(format!(
                    "Invalid length of B, expected {} but got {}",
                    cfg.k * cfg.n,
                    b.len()
                )));
            }
            if c.len() != (cfg.m * cfg.n) as usize {
                return Err(CudaError::size(format!(
                    "Invalid length of C, expected {} but got {}",
                    cfg.m * cfg.n,
                    c.len()
                )));
            }
            match &mut self.a_buf {
                Some(slice) if slice.len() >= a.len() => {
                    self.stream.memcpy_htod(a, slice)?;
                }
                slice => {
                    slice.replace(self.stream.memcpy_stod(a)?);
                }
            };
            match &mut self.b_buf {
                Some(slice) if slice.len() >= b.len() => {
                    self.stream.memcpy_htod(b, slice)?;
                }
                slice => {
                    slice.replace(self.stream.memcpy_stod(b)?);
                }
            };
            match &mut self.c_buf {
                Some(slice) if slice.len() >= c.len() => {
                    self.stream.memcpy_htod(c, slice)?;
                }
                slice => {
                    if cfg.beta == 0.0 {
                        slice.replace(unsafe { self.stream.alloc(c.len()) }?);
                    } else {
                        slice.replace(self.stream.memcpy_stod(c)?);
                    }
                }
            }
            let Self {
                blas,
                a_buf,
                b_buf,
                c_buf,
                ..
            } = self;
            unsafe {
                blas.gemm(
                    cfg,
                    a_buf.as_ref().unwrap(),
                    b_buf.as_ref().unwrap(),
                    c_buf.as_mut().unwrap(),
                )?;
            }
            self.stream.memcpy_dtoh(c_buf.as_mut().unwrap(), c)?;

            Ok(())
        }

        pub fn synchronize(&self) -> Result<(), CudaError> {
            self.stream.synchronize()?;
            Ok(())
        }

        /// A: m x k
        /// B: k x n
        /// C: m x n
        pub fn config_matmul_column_major(m: usize, n: usize, k: usize) -> GemmConfig<f64> {
            GemmConfig {
                transa: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: m as i32,
                n: n as i32,
                k: k as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: m as i32,
                ldb: k as i32,
                ldc: m as i32,
            }
        }

        /// A: m x k, A^T: k x m
        /// B: k x n, B^T: n x k
        /// C: m x n, C^T: n x m
        pub fn config_matmul_row_major(m: usize, n: usize, k: usize) -> GemmConfig<f64> {
            GemmConfig {
                transa: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: n as i32, // B^T: n x k
                ldb: k as i32, // A^T: k x m
                ldc: n as i32, // C^T: n x m, but cuda recognizes it as m x n of column major (= n x m of row major)
            }
        }
    }
}
