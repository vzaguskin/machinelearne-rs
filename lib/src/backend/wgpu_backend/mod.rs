//! # WGPU Backend
//!
//! Cross-platform GPU backend using wgpu for tensor operations.
//! Targets Vulkan, Metal, D3D12, and WebGPU from a single codebase.
//!
//! ## Features
//!
//! This backend requires the `wgpu` feature to be enabled.
//!
//! ## Performance Optimization
//!
//! This backend uses several optimizations to improve GPU performance:
//!
//! - **Buffer Pooling**: Reuses GPU buffers to reduce allocation overhead
//! - **Command Batching**: Accumulates operations and submits in batches
//! - **Lazy Execution**: Operations are deferred until results are needed
//!
//! ## Example
//!
//! ```ignore
//! use machinelearne_rs::backend::{WgpuBackend, Tensor1D, Tensor2D, Backend};
//!
//! // Create tensors on GPU
//! let x: Tensor1D<WgpuBackend> = Tensor1D::new(vec![1.0, 2.0, 3.0]);
//! let w: Tensor2D<WgpuBackend> = Tensor2D::new(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 2, 3);
//!
//! // Matrix-vector multiplication on GPU
//! let y = WgpuBackend::matvec(&w, &x);
//! ```
//!
//! ## Platform Support
//!
//! - **Vulkan**: Linux, Windows, Android
//! - **Metal**: macOS, iOS
//! - **D3D12**: Windows
//! - **WebGPU**: Browsers (WASM)

mod accumulator;
mod buffer_pool;
mod device;
mod shaders;
mod tensor;

pub use accumulator::{
    flush_accumulator, with_accumulator, AccumulatorStats, CommandAccumulator, ExecutableCommand,
};
pub use buffer_pool::{BufferPool, PoolStats};
pub use device::WgpuDevice;
pub use tensor::{WgpuTensor1D, WgpuTensor2D};

use crate::backend::Backend;
use crate::preprocessing::PreprocessingError;

/// WGPU-based GPU backend for tensor operations.
///
/// This backend executes tensor operations on the GPU using wgpu compute shaders.
/// Operations are blocking from the caller's perspective but execute asynchronously
/// on the GPU internally.
#[derive(Clone, Copy, Debug)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    type Scalar = f64;
    type Tensor1D = WgpuTensor1D;
    type Tensor2D = WgpuTensor2D;
    type Device = WgpuDevice;

    fn default_device() -> Self::Device {
        WgpuDevice::global()
    }

    // --- Constructors ---

    fn zeros_1d(len: usize) -> Self::Tensor1D {
        let device = Self::default_device();
        WgpuTensor1D::zeros(&device, len)
    }

    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        let device = Self::default_device();
        WgpuTensor2D::zeros(&device, rows, cols)
    }

    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        let device = Self::default_device();
        WgpuTensor1D::from_vec(&device, data)
    }

    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        assert_eq!(
            data.len(),
            rows * cols,
            "from_vec_2d: data length {} != rows {} * cols {}",
            data.len(),
            rows,
            cols
        );
        let device = Self::default_device();
        WgpuTensor2D::from_vec(&device, data, rows, cols)
    }

    // --- Element-wise operations (1D) ---

    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.len(),
            b.len(),
            "add_1d: tensor lengths mismatch {} vs {}",
            a.len(),
            b.len()
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Add)
    }

    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.len(),
            b.len(),
            "sub_1d: tensor lengths mismatch {} vs {}",
            a.len(),
            b.len()
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Sub)
    }

    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.len(),
            b.len(),
            "mul_1d: tensor lengths mismatch {} vs {}",
            a.len(),
            b.len()
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Mul)
    }

    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.len(),
            b.len(),
            "div_1d: tensor lengths mismatch {} vs {}",
            a.len(),
            b.len()
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Div)
    }

    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Mul)
    }

    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Add)
    }

    fn sub_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Sub)
    }

    fn div_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Div)
    }

    // --- Element-wise operations (2D) ---

    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Mul)
    }

    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Add)
    }

    fn sub_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Sub)
    }

    fn div_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let device = Self::default_device();
        t.scalar_op(&device, *s, shaders::ScalarOp::Div)
    }

    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (ra, ca) = a.shape();
        let (rb, cb) = b.shape();
        assert_eq!(
            (ra, ca),
            (rb, cb),
            "add_2d: shape mismatch ({}, {}) vs ({}, {})",
            ra,
            ca,
            rb,
            cb
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Add)
    }

    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (ra, ca) = a.shape();
        let (rb, cb) = b.shape();
        assert_eq!(
            (ra, ca),
            (rb, cb),
            "sub_2d: shape mismatch ({}, {}) vs ({}, {})",
            ra,
            ca,
            rb,
            cb
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Sub)
    }

    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (ra, ca) = a.shape();
        let (rb, cb) = b.shape();
        assert_eq!(
            (ra, ca),
            (rb, cb),
            "mul_2d: shape mismatch ({}, {}) vs ({}, {})",
            ra,
            ca,
            rb,
            cb
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Mul)
    }

    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (ra, ca) = a.shape();
        let (rb, cb) = b.shape();
        assert_eq!(
            (ra, ca),
            (rb, cb),
            "div_2d: shape mismatch ({}, {}) vs ({}, {})",
            ra,
            ca,
            rb,
            cb
        );
        let device = Self::default_device();
        a.binary_op(&device, b, shaders::BinaryOp::Div)
    }

    // --- Reduction operations ---

    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        let sum = Self::sum_all_1d(t);
        sum / t.len() as f64
    }

    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        let sum = Self::sum_all_2d(t);
        let (rows, cols) = t.shape();
        sum / (rows * cols) as f64
    }

    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        let device = Self::default_device();
        pollster::block_on(t.sum(&device))
    }

    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        let device = Self::default_device();
        pollster::block_on(t.sum(&device))
    }

    // --- Scalar operations ---

    fn scalar_f64(value: f64) -> Self::Scalar {
        value
    }

    // --- Data access ---

    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
        pollster::block_on(t.to_vec())
    }

    fn len_1d(t: &Self::Tensor1D) -> usize {
        t.len()
    }

    fn len_2d(t: &Self::Tensor2D) -> usize {
        t.shape().0
    }

    // --- Mathematical functions (1D) ---

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.unary_op(&device, shaders::UnaryOp::Abs)
    }

    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Sign)
    }

    fn maximum_1d(x: &Self::Tensor1D, other: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            x.len(),
            other.len(),
            "maximum_1d: tensor lengths mismatch {} vs {}",
            x.len(),
            other.len()
        );
        let device = Self::default_device();
        x.binary_op(&device, other, shaders::BinaryOp::Max)
    }

    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Exp)
    }

    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Log)
    }

    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Sigmoid)
    }

    // --- Mathematical functions (2D) ---

    fn abs_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        t.unary_op(&device, shaders::UnaryOp::Abs)
    }

    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Sign)
    }

    fn maximum_2d(x: &Self::Tensor2D, other: &Self::Tensor2D) -> Self::Tensor2D {
        let (ra, ca) = x.shape();
        let (rb, cb) = other.shape();
        assert_eq!(
            (ra, ca),
            (rb, cb),
            "maximum_2d: shape mismatch ({}, {}) vs ({}, {})",
            ra,
            ca,
            rb,
            cb
        );
        let device = Self::default_device();
        x.binary_op(&device, other, shaders::BinaryOp::Max)
    }

    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Exp)
    }

    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Log)
    }

    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        x.unary_op(&device, shaders::UnaryOp::Sigmoid)
    }

    // --- Linear algebra ---

    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let (_rows, cols) = a.shape();
        assert_eq!(
            cols,
            x.len(),
            "matvec: matrix cols ({}) != vector len ({})",
            cols,
            x.len()
        );
        let device = Self::default_device();
        a.matvec(&device, x)
    }

    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        a.matvec(&device, x)
    }

    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let (rows, _cols) = a.shape();
        assert_eq!(
            rows,
            x.len(),
            "matvec_transposed: matrix rows ({}) != vector len ({})",
            rows,
            x.len()
        );
        let device = Self::default_device();
        a.matvec_transposed(&device, x)
    }

    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        a.matvec_transposed(&device, x)
    }

    fn transpose(t: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        t.transpose(&device)
    }

    fn matmul(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (_m, k1) = a.shape();
        let (k2, _n) = b.shape();
        assert_eq!(k1, k2, "matmul: a.cols ({}) != b.rows ({})", k1, k2);
        let device = Self::default_device();
        a.matmul(&device, b)
    }

    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        t.shape()
    }

    fn ravel_2d(x: &Self::Tensor2D) -> Self::Tensor1D {
        let device = Self::default_device();
        x.ravel(&device)
    }

    // --- Column-wise operations ---

    fn col_mean_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.col_mean(&device)
    }

    fn col_std_2d(t: &Self::Tensor2D, ddof: usize) -> Self::Tensor1D {
        let device = Self::default_device();
        t.col_std(&device, ddof)
    }

    fn col_min_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.col_min(&device)
    }

    fn col_max_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.col_max(&device)
    }

    fn col_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.col_sum(&device)
    }

    // --- Row-wise operations ---

    fn row_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.row_sum(&device)
    }

    // --- Broadcasting operations ---

    fn broadcast_sub_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (_, cols) = t.shape();
        assert_eq!(
            cols,
            v.len(),
            "broadcast_sub_1d_to_2d_rows: tensor cols ({}) != vector len ({})",
            cols,
            v.len()
        );
        let device = Self::default_device();
        t.broadcast_op(&device, v, shaders::BinaryOp::Sub)
    }

    fn broadcast_div_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (_, cols) = t.shape();
        assert_eq!(
            cols,
            v.len(),
            "broadcast_div_1d_to_2d_rows: tensor cols ({}) != vector len ({})",
            cols,
            v.len()
        );
        let device = Self::default_device();
        t.broadcast_op(&device, v, shaders::BinaryOp::Div)
    }

    fn broadcast_mul_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (_, cols) = t.shape();
        assert_eq!(
            cols,
            v.len(),
            "broadcast_mul_1d_to_2d_rows: tensor cols ({}) != vector len ({})",
            cols,
            v.len()
        );
        let device = Self::default_device();
        t.broadcast_op(&device, v, shaders::BinaryOp::Mul)
    }

    fn broadcast_add_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (_, cols) = t.shape();
        assert_eq!(
            cols,
            v.len(),
            "broadcast_add_1d_to_2d_rows: tensor cols ({}) != vector len ({})",
            cols,
            v.len()
        );
        let device = Self::default_device();
        t.broadcast_op(&device, v, shaders::BinaryOp::Add)
    }

    fn sqrt_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        let device = Self::default_device();
        t.unary_op(&device, shaders::UnaryOp::Sqrt)
    }

    fn sqrt_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        let device = Self::default_device();
        t.unary_op(&device, shaders::UnaryOp::Sqrt)
    }

    // --- Column manipulation operations ---

    fn hcat_2d(tensors: &[Self::Tensor2D]) -> Result<Self::Tensor2D, PreprocessingError> {
        if tensors.is_empty() {
            return Err(PreprocessingError::EmptyData(
                "Cannot concatenate empty tensor slice".to_string(),
            ));
        }

        let rows = tensors[0].shape().0;
        for (i, t) in tensors.iter().enumerate() {
            if t.shape().0 != rows {
                return Err(PreprocessingError::InvalidShape {
                    expected: format!("{} rows", rows),
                    got: format!("{} rows (tensor {})", t.shape().0, i),
                });
            }
        }

        // For now, use CPU fallback for hcat (less common operation)
        let total_cols: usize = tensors.iter().map(|t| t.shape().1).sum();
        let device = Self::default_device();
        WgpuTensor2D::hcat(&device, tensors, rows, total_cols)
    }

    fn select_columns_2d(t: &Self::Tensor2D, columns: &[usize]) -> Self::Tensor2D {
        let (rows, ncols) = t.shape();
        for &col in columns {
            assert!(
                col < ncols,
                "Column index {} out of bounds (ncols={})",
                col,
                ncols
            );
        }

        let device = Self::default_device();
        t.select_columns(&device, columns, rows)
    }

    fn one_hot_from_indices(indices: &Self::Tensor1D, num_classes: usize) -> Self::Tensor2D {
        let n = indices.len();
        let device = Self::default_device();
        pollster::block_on(WgpuTensor2D::one_hot(&device, indices, num_classes, n))
    }

    // --- Fused operations using optimized kernels ---

    fn matvec_bias(a: &Self::Tensor2D, x: &Self::Tensor1D, bias: &Self::Scalar) -> Self::Tensor1D {
        let device = Self::default_device();
        a.matvec_bias(&device, x, *bias)
    }

    fn sgd_step(
        param: &Self::Tensor1D,
        gradient: &Self::Tensor1D,
        learning_rate: &Self::Scalar,
    ) -> Self::Tensor1D {
        // For now, use the default implementation since sgd_step_inplace requires mutation
        // The tensor's sgd_step_inplace is for in-place updates
        let scaled_grad = Self::mul_scalar_1d(gradient, learning_rate);
        Self::sub_1d(param, &scaled_grad)
    }
}
