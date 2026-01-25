#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::{CpuBackend, CpuTensor2D};

#[cfg(feature = "ndarray")]
mod ndarray_backend;
#[cfg(feature = "ndarray")]
pub use ndarray_backend::{NdarrayBackend, NdarrayTensor2D};
pub mod backend;
pub mod scalar;
pub mod tensor1d;
pub mod tensor2d;
pub mod tensorlike;
pub use backend::Backend;
pub use scalar::{Scalar, ScalarOps};

pub use self::tensor1d::Tensor1D;
pub use self::tensor2d::Tensor2D;
