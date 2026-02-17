//! # BLAS Backend
//!
//! CPU backend using optimized BLAS libraries for linear algebra operations.
//! Uses ndarray for tensor storage and element-wise operations, with BLAS
//! acceleration for matrix operations.
//!
//! ## Features
//!
//! This backend requires one of the following BLAS implementations:
//! - `blas-openblas`: OpenBLAS (cross-platform, open source)
//! - `blas-netlib`: Netlib BLAS (reference implementation)
//! - `blas-accelerate`: Apple Accelerate framework (macOS only)
//!
//! ## Installation
//!
//! ### OpenBLAS (Linux/macOS)
//! ```bash
//! # Ubuntu/Debian
//! sudo apt install libopenblas-dev
//!
//! # macOS (via Homebrew)
//! brew install openblas
//! ```
//!
//! ### Accelerate (macOS)
//! No additional installation needed - Accelerate is included with macOS.
//! Use the `blas-accelerate` feature.
//!
//! ## Example
//!
//! ```ignore
//! use machinelearne_rs::backend::{BlasBackend, Tensor1D, Tensor2D, Backend};
//!
//! // Create tensors
//! let x: Tensor1D<BlasBackend> = Tensor1D::new(vec![1.0, 2.0, 3.0]);
//! let w: Tensor2D<BlasBackend> = Tensor2D::new(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 2, 3);
//!
//! // Matrix-vector multiplication uses BLAS GEMV
//! let y = BlasBackend::matvec(&w, &x);
//! ```
//!
//! ## Performance
//!
//! BLAS operations (matmul, matvec) are significantly faster than the pure-Rust
//! CpuBackend, especially for large matrices. Element-wise operations use
//! ndarray's optimized implementations.

use crate::preprocessing::PreprocessingError;
use ndarray::{Array1, Array2, Axis, Zip};

/// 2D tensor type for BLAS backend, wrapping ndarray::Array2.
#[derive(Clone, Debug)]
pub struct BlasTensor2D(pub Array2<f64>);

impl BlasTensor2D {
    /// Creates a new 2D tensor from row-major data.
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        BlasTensor2D(Array2::from_shape_vec((rows, cols), data_f64).unwrap())
    }
}

/// BLAS-accelerated backend using ndarray for tensor storage.
///
/// This backend combines:
/// - BLAS libraries (OpenBLAS, Accelerate, Netlib) for matrix operations
/// - ndarray for element-wise operations and tensor management
///
/// # Type Parameters
///
/// The backend uses f64 precision for numerical stability during training.
///
/// # Device
///
/// Currently only supports CPU (Device = ()).
#[derive(Clone, Copy, Debug)]
pub struct BlasBackend;

impl super::Backend for BlasBackend {
    type Scalar = f64;
    type Tensor1D = Array1<f64>;
    type Tensor2D = BlasTensor2D;
    type Device = ();

    fn default_device() -> Self::Device {}

    // --- Constructors ---

    fn zeros_1d(len: usize) -> Self::Tensor1D {
        Array1::zeros(len)
    }

    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        BlasTensor2D(Array2::zeros((rows, cols)))
    }

    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        Array1::from_vec(data.iter().map(|&x| x as f64).collect())
    }

    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        BlasTensor2D(
            Array2::from_shape_vec((rows, cols), data_f64).expect("Shape mismatch in from_vec_2d"),
        )
    }

    // --- Element-wise operations (1D) ---

    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a + b
    }

    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a - b
    }

    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a * b
    }

    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a / b
    }

    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t * *s
    }

    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t + *s
    }

    fn sub_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t - *s
    }

    fn div_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t / *s
    }

    // --- Element-wise operations (2D) ---

    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        BlasTensor2D(&t.0 * *s)
    }

    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        BlasTensor2D(&t.0 + *s)
    }

    fn sub_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        BlasTensor2D(&t.0 - *s)
    }

    fn div_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        BlasTensor2D(&t.0 / *s)
    }

    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(&a.0 + &b.0)
    }

    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(&a.0 - &b.0)
    }

    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(&a.0 * &b.0)
    }

    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(&a.0 / &b.0)
    }

    // --- Reduction operations ---

    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.mean().unwrap_or(0.0)
    }

    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.mean().unwrap_or(0.0)
    }

    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.sum()
    }

    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.sum()
    }

    // --- Scalar operations ---

    fn scalar_f64(value: f64) -> Self::Scalar {
        value
    }

    // --- Data access ---

    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
        t.to_vec()
    }

    fn len_1d(t: &Self::Tensor1D) -> usize {
        t.len()
    }

    fn len_2d(t: &Self::Tensor2D) -> usize {
        t.0.nrows()
    }

    // --- Mathematical functions (1D) ---

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        t.mapv(|x| x.abs())
    }

    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(|v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    fn maximum_1d(x: &Self::Tensor1D, other: &Self::Tensor1D) -> Self::Tensor1D {
        Zip::from(x).and(other).map_collect(|&a, &b| a.max(b))
    }

    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(|v| v.exp())
    }

    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(|v| {
            assert!(v > 0.0, "log of non-positive value: {}", v);
            v.ln()
        })
    }

    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(|v| {
            if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let exp_v = v.exp();
                exp_v / (1.0 + exp_v)
            }
        })
    }

    // --- Mathematical functions (2D) ---

    fn abs_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(t.0.mapv(|x| x.abs()))
    }

    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(x.0.mapv(|v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }))
    }

    fn maximum_2d(x: &Self::Tensor2D, other: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(Zip::from(&x.0).and(&other.0).map_collect(|&a, &b| a.max(b)))
    }

    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(x.0.mapv(|v| v.exp()))
    }

    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(x.0.mapv(|v| {
            assert!(v > 0.0, "log of non-positive value: {}", v);
            v.ln()
        }))
    }

    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(x.0.mapv(|v| {
            if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let exp_v = v.exp();
                exp_v / (1.0 + exp_v)
            }
        }))
    }

    // --- Linear algebra ---

    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.0.ncols(),
            x.len(),
            "matvec: matrix cols ({}) != vector len ({})",
            a.0.ncols(),
            x.len()
        );
        a.0.dot(x)
    }

    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.0.dot(x)
    }

    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.0.nrows(),
            x.len(),
            "matvec_transposed: matrix rows ({}) != vector len ({})",
            a.0.nrows(),
            x.len()
        );
        a.0.t().dot(x)
    }

    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.0.t().dot(x)
    }

    fn transpose(t: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(t.0.t().to_owned())
    }

    fn matmul(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        assert_eq!(
            a.0.ncols(),
            b.0.nrows(),
            "matmul: a.cols ({}) != b.rows ({})",
            a.0.ncols(),
            b.0.nrows()
        );
        BlasTensor2D(a.0.dot(&b.0))
    }

    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        (t.0.nrows(), t.0.ncols())
    }

    fn ravel_2d(x: &Self::Tensor2D) -> Self::Tensor1D {
        x.0.clone().into_shape_with_order((x.0.len(),)).unwrap()
    }

    // --- Column-wise operations ---

    fn col_mean_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        t.0.mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(t.0.ncols()))
    }

    fn col_std_2d(t: &Self::Tensor2D, ddof: usize) -> Self::Tensor1D {
        let mean = Self::col_mean_2d(t);
        let n = t.0.nrows() as f64;
        let var: Array1<f64> = Zip::from(t.0.axis_iter(Axis(1)))
            .and(&mean)
            .map_collect(|col, &m| col.mapv(|x| (x - m).powi(2)).sum() / (n - ddof as f64));
        var.mapv(|v| v.sqrt())
    }

    fn col_min_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        t.0.map_axis(Axis(0), |col| {
            col.iter().cloned().fold(f64::INFINITY, f64::min)
        })
    }

    fn col_max_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        t.0.map_axis(Axis(0), |col| {
            col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        })
    }

    fn col_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        t.0.sum_axis(Axis(0))
    }

    // --- Row-wise operations ---

    fn row_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        t.0.sum_axis(Axis(1))
    }

    // --- Broadcasting operations ---

    fn broadcast_sub_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        BlasTensor2D(&t.0 - v)
    }

    fn broadcast_div_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        BlasTensor2D(&t.0 / v)
    }

    fn broadcast_mul_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        BlasTensor2D(&t.0 * v)
    }

    fn broadcast_add_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        BlasTensor2D(&t.0 + v)
    }

    fn sqrt_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        t.mapv(|x| x.sqrt())
    }

    fn sqrt_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        BlasTensor2D(t.0.mapv(|x| x.sqrt()))
    }

    // --- Column manipulation operations ---

    fn hcat_2d(tensors: &[Self::Tensor2D]) -> Result<Self::Tensor2D, PreprocessingError> {
        if tensors.is_empty() {
            return Err(PreprocessingError::EmptyData(
                "Cannot concatenate empty tensor slice".to_string(),
            ));
        }

        let rows = tensors[0].0.nrows();
        for (i, t) in tensors.iter().enumerate() {
            if t.0.nrows() != rows {
                return Err(PreprocessingError::InvalidShape {
                    expected: format!("{} rows", rows),
                    got: format!("{} rows (tensor {})", t.0.nrows(), i),
                });
            }
        }

        let views: Vec<ndarray::ArrayView2<f64>> = tensors.iter().map(|t| t.0.view()).collect();
        let concatenated = ndarray::concatenate(Axis(1), &views)
            .map_err(|e| PreprocessingError::InvalidParameter(e.to_string()))?;
        Ok(BlasTensor2D(concatenated))
    }

    fn select_columns_2d(t: &Self::Tensor2D, columns: &[usize]) -> Self::Tensor2D {
        let ncols = t.0.ncols();
        for &col in columns {
            assert!(
                col < ncols,
                "Column index {} out of bounds (ncols={})",
                col,
                ncols
            );
        }

        let selected: Vec<ndarray::ArrayView1<f64>> =
            columns.iter().map(|&col| t.0.column(col)).collect();

        let mut result = Array2::zeros((t.0.nrows(), columns.len()));
        for (i, col_view) in selected.iter().enumerate() {
            result.column_mut(i).assign(col_view);
        }
        BlasTensor2D(result)
    }

    fn one_hot_from_indices(indices: &Self::Tensor1D, num_classes: usize) -> Self::Tensor2D {
        let n = indices.len();
        let mut result = Array2::zeros((n, num_classes));

        for (i, &idx) in indices.iter().enumerate() {
            let class_idx = idx as usize;
            assert!(
                class_idx < num_classes,
                "Index {} >= num_classes {}",
                class_idx,
                num_classes
            );
            result[[i, class_idx]] = 1.0;
        }

        BlasTensor2D(result)
    }
}
