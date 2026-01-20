#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::{CpuBackend, CpuTensor2D};

#[cfg(feature = "ndarray")]
mod ndarray_backend;
#[cfg(feature = "ndarray")]
pub use ndarray_backend::{NdarrayBackend, NdarrayTensor2D};
pub mod scalar;
pub mod tensorlike;
pub mod backend;
pub mod tensor1d;
pub mod tensor2d;
pub use scalar::{ScalarOps, Scalar};
pub use backend::Backend;

pub use self::tensor1d::Tensor1D;
pub use self::tensor2d::Tensor2D;
