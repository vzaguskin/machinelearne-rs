mod cpu;
pub mod scalar;
pub mod tensorlike;
pub mod backend;
pub mod tensor1d;
pub mod tensor2d;
use std::marker::PhantomData;

pub use self::cpu::CpuBackend;
pub use scalar::{ScalarOps};
pub use backend::Backend;

pub use self::tensor1d::Tensor1D;
pub use self::tensor2d::Tensor2D;

/// Backend trait: абстракция над вычислительным движком (CPU, CUDA, ndarray, candle и т.д.)

 
pub trait Backend_ {
    type Scalar: ScalarOps + Clone;
    type Tensor0D: Clone + Send + Sync;
    type Tensor1D: Clone + Send + Sync;
    type Tensor2D: Clone + Send + Sync;
    type Device: Clone + Send + Sync;

    fn default_device() -> Self::Device;

    // === Allocation ===
    fn zeros_1d(n: usize, device: &Self::Device) -> Self::Tensor1D;
    fn zeros_2d(rows: usize, cols: usize, device: &Self::Device) -> Self::Tensor2D;
    fn scalar(value: f64, device: &Self::Device) -> Self::Scalar;

    // === Shape queries ===
    fn len_1d(x: &Self::Tensor1D) -> usize;
    fn shape_2d(x: &Self::Tensor2D) -> (usize, usize);

    // === Element access ===
    fn get_1d(x: &Self::Tensor1D, i: usize) -> Self::Scalar;
    fn set_1d(x: &mut Self::Tensor1D, i: usize, v: Self::Scalar);
    fn get_2d(x: &Self::Tensor2D, i: usize, j: usize) -> Self::Scalar;
    fn set_2d(x: &mut Self::Tensor2D, i: usize, j: usize, v: Self::Scalar);

    // === Element-wise ops ===
    fn add_scalar_1d(x: &Self::Tensor1D, s: Self::Scalar) -> Self::Tensor1D;
    fn add_scalar_1d_inplace(x: &mut Self::Tensor1D, s: Self::Scalar);
    fn scale_1d(a: Self::Scalar, x: &Self::Tensor1D) -> Self::Tensor1D;
    fn scale_2d(a: Self::Scalar, x: &Self::Tensor2D) -> Self::Tensor2D;

    fn sum_1d(x: &Self::Tensor1D) -> Self::Scalar;
    fn sum_2d(x: &Self::Tensor2D) -> Self::Scalar;

    // === BLAS-like ops (with runtime asserts) ===
    fn dot(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Scalar {
        let nx = Self::len_1d(x);
        let ny = Self::len_1d(y);
        assert_eq!(nx, ny, "dot: vectors must have same length ({} != {})", nx, ny);
        Self::_dot_unchecked(x, y)
    }

    fn axpy(a: Self::Scalar, x: &Self::Tensor1D, y: &mut Self::Tensor1D) {
        let nx = Self::len_1d(x);
        let ny = Self::len_1d(y);
        assert_eq!(nx, ny, "axpy: vectors must have same length ({} != {})", nx, ny);
        Self::_axpy_unchecked(a, x, y);
    }

    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let (_rows, cols) = Self::shape_2d(a);
        let nx = Self::len_1d(x);
        assert_eq!(cols, nx, "matvec: A.cols ({}) != x.len ({})", cols, nx);
        Self::_matvec_unchecked(a, x)
    }

    fn matvec_transpose(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let (rows, _cols) = Self::shape_2d(a);
        let nx = Self::len_1d(x);
        assert_eq!(rows, nx, "matvec: A.rows ({}) != x.len ({})", rows, nx);
        Self::_matvec_transpose_unchecked(a, x)
    }

    fn matmul(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (_a_rows, a_cols) = Self::shape_2d(a);
        let (b_rows, _b_cols) = Self::shape_2d(b);
        assert_eq!(a_cols, b_rows, "matmul: A.cols ({}) != B.rows ({})", a_cols, b_rows);
        Self::_matmul_unchecked(a, b)
    }

    fn sub_1d(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D {
        let nx = Self::len_1d(x);
        let ny = Self::len_1d(y);
        assert_eq!(nx, ny, "axpy: vectors must have same length ({} != {})", nx, ny);
        Self::_minus_vec_vec_unchecked(x, y)
    }


    fn add_1d(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D {
        let nx = Self::len_1d(x);
        let ny = Self::len_1d(y);
        assert_eq!(nx, ny, "axpy: vectors must have same length ({} != {})", nx, ny);
        Self::_plus_vec_vec_unchecked(x, y)
    }

    fn transpose(x: &Self::Tensor2D) -> Self::Tensor2D;

    // === Norms ===
    fn l2_norm(x: &Self::Tensor1D) -> Self::Scalar {
        let s = Self::dot(x, x);
        s.sqrt()
    }

    fn abs_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D;

    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn maximum_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn neg_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn mul_1d(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D;

    // === Unsafe (unchecked) implementations — to be overridden efficiently ===
    #[doc(hidden)]
    fn _dot_unchecked(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Scalar;

    #[doc(hidden)]
    fn _axpy_unchecked(a: Self::Scalar, x: &Self::Tensor1D, y: &mut Self::Tensor1D);

    #[doc(hidden)]
    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;

    #[doc(hidden)]
    fn _matvec_transpose_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;


    #[doc(hidden)]
    fn _matmul_unchecked(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D;

    #[doc(hidden)]
    fn _minus_vec_vec_unchecked(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D;

    #[doc(hidden)]
    fn _plus_vec_vec_unchecked(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D;

}