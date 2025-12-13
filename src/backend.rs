#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(usize),   // device index
    Metal(usize),
    // можно расширить
}

/// Backend trait: абстракция над вычислительным движком (CPU, CUDA, ndarray, candle и т.д.)

pub trait ScalarOps:
    Clone
    + Copy
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

// === Реализации для f64 (и f32, если нужно) ===
impl ScalarOps for f64 {
    fn sqrt(self) -> Self { self.sqrt() }
    fn abs(self) -> Self { self.abs() }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    
}

pub trait Backend {
    type Scalar: ScalarOps;
    type Tensor0D: Clone + Send + Sync;
    type Tensor1D: Clone + Send + Sync;
    type Tensor2D: Clone + Send + Sync;
    type Device: Clone + Send + Sync;

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

    fn matmul(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (_a_rows, a_cols) = Self::shape_2d(a);
        let (b_rows, _b_cols) = Self::shape_2d(b);
        assert_eq!(a_cols, b_rows, "matmul: A.cols ({}) != B.rows ({})", a_cols, b_rows);
        Self::_matmul_unchecked(a, b)
    }

    fn minus_vec_vec(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D {
        let nx = Self::len_1d(x);
        let ny = Self::len_1d(y);
        assert_eq!(nx, ny, "axpy: vectors must have same length ({} != {})", nx, ny);
        Self::_minus_vec_vec_unchecked(x, y)
    }

    fn plus_vec_vec(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D {
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

    // === Unsafe (unchecked) implementations — to be overridden efficiently ===
    #[doc(hidden)]
    fn _dot_unchecked(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Scalar;

    #[doc(hidden)]
    fn _axpy_unchecked(a: Self::Scalar, x: &Self::Tensor1D, y: &mut Self::Tensor1D);

    #[doc(hidden)]
    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;

    #[doc(hidden)]
    fn _matmul_unchecked(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D;

    #[doc(hidden)]
    fn _minus_vec_vec_unchecked(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D;

    #[doc(hidden)]
    fn _plus_vec_vec_unchecked(x: &Self::Tensor1D, y: &Self::Tensor1D) -> Self::Tensor1D;

}