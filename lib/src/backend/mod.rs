//! # Backend Abstraction
//!
//! This module provides a trait-based abstraction over computation backends,
//! enabling models to run on different hardware (CPU, GPU) and tensor libraries
//! without code changes.
//!
//! ## Design Philosophy
//!
//! - **Minimal trait surface**: Only essential operations are exposed, keeping backend
//!   implementations simple and testable.
//! - **Zero-cost generics**: Backend selection happens at compile time via type parameters,
//!   avoiding runtime dispatch overhead.
//! - **Type-safe tensor handling**: Each backend defines its own tensor types (`Tensor1D`,
//!   `Tensor2D`) that encapsulate storage details while exposing a uniform API.
//! - **Feature-gated implementations**: Backends are enabled via Cargo features (`cpu`,
//!   `ndarray`, future `cuda`, etc.), allowing users to minimize dependencies.
//!
//! ## Available Backends
//!
//! | Backend      | Feature    | Use Case                          |
//! |--------------|------------|-----------------------------------|
//! | `CpuBackend` | `cpu`      | Default, pure-Rust implementation |
//! | `NdarrayBackend` | `ndarray` | Interop with `ndarray` ecosystem |
//!
//! ## Example
//!
//! ```rust
//! use machinelearne_rs::backend::{Backend, CpuBackend, Tensor1D, Tensor2D};
//!
//! // Backend selection via type parameter
//! let x: Tensor1D<CpuBackend> = Tensor1D::new(vec![1.0, 2.0]);
//! let w: Tensor2D<CpuBackend> = Tensor2D::new(vec![0.5, 0.5, 0.5, 0.5], 2, 2);
//!
//! // Operations work identically across backends
//! let y = w.dot(&x);
//! ```
//!
//! ## Implementing a New Backend
//!
//! To add a backend (e.g., CUDA):
//! 1. Create a module with feature gate (`#[cfg(feature = "cuda")]`)
//! 2. Implement concrete tensor types (`CudaTensor1D`, `CudaTensor2D`)
//! 3. Implement the `Backend` trait with GPU-accelerated operations
//! 4. Export types via `pub use` for user access
//!
//! See `cpu.rs` for a reference implementation.

use crate::preprocessing::PreprocessingError;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cpu")]
/// Pure-Rust CPU backend implementation with zero external dependencies.
pub use cpu::{CpuBackend, CpuTensor2D};

#[cfg(feature = "ndarray")]
mod ndarray_backend;
#[cfg(feature = "ndarray")]
/// Backend backed by the `ndarray` crate for ecosystem interoperability.
pub use ndarray_backend::{NdarrayBackend, NdarrayTensor2D};

/// Scalar value representation and arithmetic operations.
pub mod scalar;
/// One-dimensional tensor abstraction.
pub mod tensor1d;
/// Two-dimensional tensor abstraction.
pub mod tensor2d;
/// Shared tensor-like operations trait.
pub mod tensorlike;

pub use scalar::{Scalar, ScalarOps};
pub use tensor1d::Tensor1D;
pub use tensor2d::Tensor2D;

/// Abstraction over computation devices and tensor operations.
///
/// The `Backend` trait defines a minimal set of operations required for
/// training and inference in machine learning models. Implementations
/// provide concrete tensor types and device-specific optimizations while
/// maintaining a uniform API surface.
///
/// # Type Parameters
///
/// - `Scalar`: Primitive numeric type with arithmetic capabilities
/// - `Tensor1D`: One-dimensional array representation
/// - `Tensor2D`: Two-dimensional matrix representation
/// - `Device`: Hardware device identifier (CPU core, GPU ID, etc.)
///
/// # Safety Guarantees
///
/// - All checked operations (`matvec`, `matvec_transposed`) validate shapes
///   and panic on mismatch
/// - Unchecked variants (`_matvec_unchecked`) skip validation for performance;
///   caller must ensure correctness
/// - Tensor types are `Clone + Send + Sync` for safe concurrent usage
///
/// # Example Implementation Sketch
///
/// ```ignore
/// use machinelearne_rs::backend::{Backend, ScalarOps};
/// #[derive(Clone, Debug, Copy)]
/// struct MyBackend;
///
///  #[derive(Clone, Debug, Copy)]
/// struct MyTensor1D;
///
///  #[derive(Clone, Debug, Copy)]
/// struct MyTensor2D;
///
/// impl Backend for MyBackend {
///     type Scalar = f64;
///     type Tensor1D = MyTensor1D;
///     type Tensor2D = MyTensor2D;
///     type Device = ();
///
///     fn default_device() -> Self::Device { () }
///     // ... implement all required methods
/// }
/// ```
pub trait Backend: Clone + Copy + 'static {
    /// Scalar type supporting arithmetic operations.
    type Scalar: ScalarOps + Clone;

    /// One-dimensional tensor type.
    type Tensor1D: Clone + Send + Sync;

    /// Two-dimensional tensor type.
    type Tensor2D: Clone + Send + Sync;

    /// Device identifier type (CPU core index, GPU handle, etc.).
    type Device: Clone + Send + Sync;

    /// Returns the default device for this backend.
    ///
    /// For CPU backends, typically returns a unit type `()`.
    /// For GPU backends, may return a device handle or index.
    fn default_device() -> Self::Device;

    // --- Constructors ---

    /// Creates a 1D tensor filled with zeros of given length.
    fn zeros_1d(len: usize) -> Self::Tensor1D;

    /// Creates a 2D tensor filled with zeros of given dimensions.
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D;

    /// Constructs a 1D tensor from owned data.
    ///
    /// # Panics
    /// Implementation-defined (typically none for valid inputs).
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D;

    /// Constructs a 2D tensor from row-major ordered data.
    ///
    /// # Arguments
    /// * `data` - Flattened tensor values in row-major order
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D;

    // --- Element-wise operations (1D) ---

    /// Element-wise addition of two 1D tensors.
    ///
    /// # Panics
    /// If tensors have different lengths.
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise subtraction of two 1D tensors.
    ///
    /// # Panics
    /// If tensors have different lengths.
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise multiplication of two 1D tensors.
    ///
    /// # Panics
    /// If tensors have different lengths.
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise division of two 1D tensors.
    ///
    /// # Panics
    /// If tensors have different lengths or divisor contains zeros.
    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;

    /// Multiplies each element of tensor by a scalar.
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D;

    /// Adds a scalar to each element of tensor.
    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D;

    // --- Element-wise operations (2D) ---

    /// Multiplies each element of 2D tensor by a scalar.
    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D;

    /// Adds a scalar to each element of 2D tensor.
    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D;

    /// Element-wise addition of two 2D tensors.
    ///
    /// # Panics
    /// If tensors have different shapes.
    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise subtraction of two 2D tensors.
    ///
    /// # Panics
    /// If tensors have different shapes.
    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise multiplication of two 2D tensors.
    ///
    /// # Panics
    /// If tensors have different shapes.
    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise division of two 2D tensors.
    ///
    /// # Panics
    /// If tensors have different shapes or divisor contains zeros.
    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D;

    // --- Reduction operations ---

    /// Computes the arithmetic mean of all elements in a 1D tensor.
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar;

    /// Computes the arithmetic mean of all elements in a 2D tensor.
    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar;

    /// Computes the sum of all elements in a 2D tensor.
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar;

    /// Computes the sum of all elements in a 1D tensor.
    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar;

    // --- Scalar operations ---

    /// Creates a backend-specific scalar from an f64 value.
    ///
    /// Used for loss gradients, learning rate updates, and other scalar computations.
    fn scalar_f64(value: f64) -> Self::Scalar;

    // --- Data access ---

    /// Converts a 1D tensor to a Vec of f64 values.
    ///
    /// Primarily used for metrics computation and debugging.
    /// Not intended for hot paths due to allocation overhead.
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64>;

    /// Returns the number of elements in a 1D tensor.
    fn len_1d(t: &Self::Tensor1D) -> usize;

    /// Returns the total number of elements in a 2D tensor (rows × cols).
    fn len_2d(t: &Self::Tensor2D) -> usize;

    // --- Mathematical functions (1D) ---

    /// Element-wise absolute value.
    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise sign function: returns -1.0, 0.0, or 1.0.
    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise maximum between two tensors.
    ///
    /// # Panics
    /// If tensors have different lengths.
    fn maximum_1d(x: &Self::Tensor1D, other: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise exponential function (e^x).
    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise natural logarithm (ln(x)).
    ///
    /// # Panics
    /// If any element is ≤ 0.0.
    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Element-wise sigmoid function: 1 / (1 + e^(-x)).
    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D;

    // --- Mathematical functions (2D) ---

    /// Element-wise absolute value for 2D tensors.
    fn abs_2d(t: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise sign function for 2D tensors.
    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise maximum between two 2D tensors.
    ///
    /// # Panics
    /// If tensors have different shapes.
    fn maximum_2d(x: &Self::Tensor2D, other: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise exponential function for 2D tensors.
    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise natural logarithm for 2D tensors.
    ///
    /// # Panics
    /// If any element is ≤ 0.0.
    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D;

    /// Element-wise sigmoid function for 2D tensors.
    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D;

    // --- Linear algebra ---

    /// Matrix-vector multiplication with shape checking.
    ///
    /// Computes `y = A * x` where `A` is (m × n) and `x` is (n,).
    /// Returns a (m,) vector.
    ///
    /// # Panics
    /// If `A.cols() != x.len()`.
    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Matrix-vector multiplication without shape checking.
    ///
    /// # Safety
    /// Caller must ensure `A.cols() == x.len()`. Undefined behavior may occur
    /// if shapes are incompatible.
    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Transposed matrix-vector multiplication with shape checking.
    ///
    /// Computes `y = A^T * x` where `A` is (m × n) and `x` is (m,).
    /// Returns a (n,) vector.
    ///
    /// # Panics
    /// If `A.rows() != x.len()`.
    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Transposed matrix-vector multiplication without shape checking.
    ///
    /// # Safety
    /// Caller must ensure `A.rows() == x.len()`. Undefined behavior may occur
    /// if shapes are incompatible.
    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D;

    /// Returns the transpose of a 2D tensor.
    ///
    /// Converts an (m × n) matrix to (n × m) with elements at (i,j) ↔ (j,i).
    fn transpose(t: &Self::Tensor2D) -> Self::Tensor2D;

    /// Returns the shape of a 2D tensor as (rows, cols).
    fn shape(t: &Self::Tensor2D) -> (usize, usize);

    //Flattens 2d tensor into 1d tensor
    fn ravel_2d(x: &Self::Tensor2D) -> Self::Tensor1D;

    // --- Column-wise operations (for preprocessing) ---

    /// Computes the mean of each column in a 2D tensor.
    ///
    /// Returns a 1D tensor of length `cols` where each element is the mean
    /// of the corresponding column.
    ///
    /// For a tensor with shape (rows, cols), computes mean along axis 0.
    fn col_mean_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

    /// Computes the standard deviation of each column in a 2D tensor.
    ///
    /// # Arguments
    /// * `t` - Input 2D tensor
    /// * `ddof` - Delta degrees of freedom (1 for sample std, 0 for population std)
    ///
    /// Returns a 1D tensor of length `cols`.
    fn col_std_2d(t: &Self::Tensor2D, ddof: usize) -> Self::Tensor1D;

    /// Computes the minimum value of each column in a 2D tensor.
    ///
    /// Returns a 1D tensor of length `cols`.
    fn col_min_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

    /// Computes the maximum value of each column in a 2D tensor.
    ///
    /// Returns a 1D tensor of length `cols`.
    fn col_max_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

    /// Computes the sum of each column in a 2D tensor.
    ///
    /// Returns a 1D tensor of length `cols`.
    fn col_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

    // --- Row-wise operations ---

    /// Computes the sum of each row in a 2D tensor.
    ///
    /// Returns a 1D tensor of length `rows`.
    fn row_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D;

    // --- Broadcasting operations ---

    /// Broadcasts a 1D tensor and subtracts from each row of a 2D tensor.
    ///
    /// For a 2D tensor with shape (rows, cols) and a 1D tensor with shape (cols,),
    /// subtracts the 1D tensor from each row of the 2D tensor.
    ///
    /// Result[i, j] = t[i, j] - v[j]
    fn broadcast_sub_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;

    /// Broadcasts a 1D tensor and divides each row of a 2D tensor.
    ///
    /// For a 2D tensor with shape (rows, cols) and a 1D tensor with shape (cols,),
    /// divides each row of the 2D tensor by the 1D tensor element-wise.
    ///
    /// Result[i, j] = t[i, j] / v[j]
    fn broadcast_div_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;

    /// Broadcasts a 1D tensor and multiplies each row of a 2D tensor.
    ///
    /// For a 2D tensor with shape (rows, cols) and a 1D tensor with shape (cols,),
    /// multiplies each row of the 2D tensor by the 1D tensor element-wise.
    ///
    /// Result[i, j] = t[i, j] * v[j]
    fn broadcast_mul_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;

    /// Broadcasts a 1D tensor and adds to each row of a 2D tensor.
    ///
    /// For a 2D tensor with shape (rows, cols) and a 1D tensor with shape (cols,),
    /// adds the 1D tensor to each row of the 2D tensor.
    ///
    /// Result[i, j] = t[i, j] + v[j]
    fn broadcast_add_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D;

    /// Sqrt of all elements in a 1D tensor.
    fn sqrt_1d(t: &Self::Tensor1D) -> Self::Tensor1D;

    /// Sqrt of all elements in a 2D tensor.
    fn sqrt_2d(t: &Self::Tensor2D) -> Self::Tensor2D;

    // --- Column manipulation operations ---

    /// Horizontally concatenate 2D tensors (stack columns side by side).
    ///
    /// All input tensors must have the same number of rows.
    /// Returns a new tensor with shape (rows, sum of all cols).
    ///
    /// # Arguments
    /// * `tensors` - Slice of 2D tensors to concatenate
    ///
    /// # Panics
    /// Panics if tensors have different row counts or if the slice is empty.
    ///
    /// # Example
    /// ```ignore
    /// // [[1, 2]] + [[3]] -> [[1, 2, 3]]
    /// let a = Tensor2D::new(vec![1.0, 2.0], 1, 2);
    /// let b = Tensor2D::new(vec![3.0], 1, 1);
    /// let c = B::hcat_2d(&[a, b]); // shape (1, 3)
    /// ```
    fn hcat_2d(tensors: &[Self::Tensor2D]) -> Result<Self::Tensor2D, PreprocessingError>;

    /// Extract specific columns from a 2D tensor.
    ///
    /// Returns a new tensor with only the specified columns, preserving row order.
    ///
    /// # Arguments
    /// * `t` - Input 2D tensor
    /// * `columns` - Indices of columns to extract (in order)
    ///
    /// # Panics
    /// Panics if any column index is out of bounds.
    fn select_columns_2d(t: &Self::Tensor2D, columns: &[usize]) -> Self::Tensor2D;

    /// Create a one-hot encoded matrix from integer indices.
    ///
    /// Each index becomes a row with a 1 at the index position and 0 elsewhere.
    ///
    /// # Arguments
    /// * `indices` - 1D tensor of integer class indices (0 to num_classes-1)
    /// * `num_classes` - Total number of classes (determines output column count)
    ///
    /// # Returns
    /// A 2D tensor of shape (indices.len(), num_classes)
    ///
    /// # Panics
    /// Panics if any index >= num_classes.
    fn one_hot_from_indices(indices: &Self::Tensor1D, num_classes: usize) -> Self::Tensor2D;
}
