mod cpu;
pub mod tensorlike;
use std::str;
use std::marker::PhantomData;

pub use self::cpu::CpuBackend;

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


pub struct Scalar <B: Backend> {
    pub(crate) data: B::Scalar,
    pub(crate) backend: PhantomData<B>,
}

impl <B: Backend> Scalar<B>{
    pub fn new(f: f64) -> Self{
        Self{data: B::scalar_f64(f), backend: PhantomData}
    }
}

// Арифметические операции через std::ops
impl<B: Backend> std::ops::Add for Scalar<B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            data: self.data + rhs.data,
            backend: PhantomData,
        }
    }
}

impl<B: Backend> std::ops::Sub for Scalar<B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            data: self.data - rhs.data,
            backend: PhantomData,
        }
    }
}

impl<B: Backend> std::ops::Mul for Scalar<B> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            data: self.data * rhs.data,
            backend: PhantomData,
        }
    }
}

impl<B: Backend> std::ops::Div for Scalar<B> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            data: self.data / rhs.data,
            backend: PhantomData,
        }
    }
}


pub struct Tensor1D<B: Backend> {
    pub(crate) data: B::Tensor1D,
    pub(crate) backend: PhantomData<B>,
}

pub struct Tensor2D<B: Backend> {
    pub(crate) data: B::Tensor2D,
    pub(crate) backend: PhantomData<B>,
}

pub trait Backend: Clone + 'static {
    type Scalar: ScalarOps + Clone;
    type Tensor1D: Clone + Send + Sync;
    type Tensor2D: Clone + Send + Sync;
    type Device: Clone + Send + Sync;

    fn default_device() -> Self::Device;

    // --- Constructors ---
    fn zeros_1d(len: usize) -> Self::Tensor1D;
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D;
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D;
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D;

    // --- Element-wise ops ---
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D;

    // --- Reductions ---
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar;
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar;
    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar;

    // --- Scalar ops (for loss gradients, lr updates) ---
    fn scalar_f64(value: f64) -> Self::Scalar;
    fn scalar_mul_1d(s: &Self::Scalar, t: &Self::Tensor1D) -> Self::Tensor1D;

    // --- Access (for metrics / debug) ---
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64>;

    fn len_1d(t: &Self::Tensor1D) -> usize;

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D;
    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
}


impl<B: Backend> Tensor1D<B> {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: B::from_vec_1d(data),
            backend: PhantomData,
        }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            data: B::zeros_1d(len),
            backend: PhantomData,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: B::sub_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    pub fn mean(&self) -> Scalar<B> {
        Scalar {
            data: B::mean_all_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        B::to_vec_1d(&self.data)
    }

    pub fn dot(&self, other: &Self) -> Scalar<B> {
        let prod = B::mul_1d(&self.data,&other.data);
        let sum = B::sum_all_1d(&prod);
        Scalar {
            data: sum,
            backend: PhantomData,
        }
    
    }

    pub fn abs(&self) -> Self{
        Self {
            data: B::abs_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sign(&self) -> Self{
        Self {
            data: B::sign_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn len(&self) -> Scalar<B>{
        Scalar {
            data: B::scalar_f64(B::len_1d(&self.data) as f64),
            backend: PhantomData,
        }
    }
}
 
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