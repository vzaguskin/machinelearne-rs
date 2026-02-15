use super::scalar::Scalar;
use super::tensor1d::Tensor1D;
use crate::backend::Backend;
use std::marker::PhantomData;

/// Backend-typed 2D tensor (matrix) providing compile-time type safety and zero-cost abstractions.
///
/// Wraps a backend's native 2D tensor representation (`B::Tensor2D`) while carrying phantom
/// type information about its originating backend. This prevents accidental mixing of tensors
/// from different backends at compile time while maintaining performance through zero-sized
/// `PhantomData` overhead.
///
/// # Type safety guarantees
/// ```compile_fail
/// use machinelearne_rs::backend::{CpuBackend, NdarrayBackend};
/// use machinelearne_rs::backend::Tensor2D;
///
/// let cpu_mat: Tensor2D<CpuBackend> = Tensor2D::zeros(2, 2);
/// let ndarray_mat: Tensor2D<NdarrayBackend> = Tensor2D::zeros(2, 2);
/// let _ = cpu_mat.sub(&ndarray_mat); // COMPILE ERROR: mismatched backends
/// ```
///
/// # Precision semantics
/// - Constructors accept `Vec<f32>` for ergonomic data loading (row-major order)
/// - Values are immediately converted to backend's native precision (typically `f64`)
/// - All operations occur in native backend precision
/// - Row-major layout: `[a₁₁, a₁₂, ..., a₁ₙ, a₂₁, ..., aₘₙ]` for an (m×n) matrix
///
/// # Matrix-vector operations
/// This tensor provides two fundamental linear algebra operations:
/// - `dot(x)`: Computes `A @ x` where `A` is (m×n) and `x` is (n,) → result (m,)
/// - `tdot(x)`: Computes `Aᵀ @ x` where `A` is (m×n) and `x` is (m,) → result (n,)
///
/// # Zero-cost design
/// - `PhantomData<B>` adds no runtime memory overhead
/// - All operations delegate directly to backend implementations
/// - Implements `Clone` (but not `Copy`) due to potential heap allocation in underlying tensors
///
/// # Example
/// ```
/// use machinelearne_rs::backend::CpuBackend;
/// use machinelearne_rs::backend::{Tensor2D, Tensor1D};
///
/// // Create 2×2 matrix: [[1.0, 2.0], [3.0, 4.0]]
/// let a: Tensor2D<CpuBackend> = Tensor2D::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
/// assert_eq!(a.shape(), (2, 2));
///
/// // Matrix-vector multiplication: A @ [1, 0]ᵀ = [1, 3]ᵀ
/// let x = Tensor1D::<CpuBackend>::new(vec![1.0f32, 0.0]);
/// let y = a.dot(&x);
/// assert_eq!(y.to_vec(), vec![1.0, 3.0]);
/// ```
#[derive(Clone)]
pub struct Tensor2D<B: Backend> {
    pub(crate) data: B::Tensor2D,
    pub(crate) backend: PhantomData<B>,
}

impl<B: Backend> Tensor2D<B> {
    /// Creates a new 2D tensor from a flat vector of `f32` values in row-major order.
    ///
    /// The input vector must contain exactly `rows * cols` elements arranged as:
    /// `[row₀_col₀, row₀_col₁, ..., row₀_colₙ₋₁, row₁_col₀, ..., rowₘ₋₁_colₙ₋₁]`
    ///
    /// # Arguments
    /// * `data` - Flat vector containing matrix elements in row-major order
    /// * `rows` - Number of rows in the resulting matrix
    /// * `cols` - Number of columns in the resulting matrix
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols` (backend-dependent behavior).
    ///
    /// # Precision note
    /// Input values are converted from `f32` to backend precision (usually `f64`).
    /// Extremely large `f32` values near `f32::MAX` may lose precision during conversion.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// // Matrix: [[1.0, 2.0, 3.0],
    /// //          [4.0, 5.0, 6.0]]
    /// let t: Tensor2D<CpuBackend> = Tensor2D::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    /// assert_eq!(t.shape(), (2, 3));
    /// ```
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self {
            data: B::from_vec_2d(data, rows, cols),
            backend: PhantomData,
        }
    }

    /// Creates a 2D tensor filled with zeros of specified dimensions.
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let zeros: Tensor2D<CpuBackend> = Tensor2D::zeros(3, 4);
    /// assert_eq!(zeros.shape(), (3, 4));
    /// assert_eq!(zeros.mean().to_f64(), 0.0);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: B::zeros_2d(rows, cols),
            backend: PhantomData,
        }
    }

    /// Computes element-wise subtraction: `self - other`.
    ///
    /// # Panics
    /// Panics if tensors have different shapes (backend-dependent behavior).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let a = Tensor2D::<CpuBackend>::new(vec![5.0f32, 7.0, 9.0, 11.0], 2, 2);
    /// let b = Tensor2D::<CpuBackend>::new(vec![2.0f32, 3.0, 4.0, 5.0], 2, 2);
    /// let diff = a.sub(&b);
    /// // Result: [[3.0, 4.0], [5.0, 6.0]]
    /// assert_eq!(diff.mean().to_f64(), 4.5);
    /// ```
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: B::sub_2d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    /// Computes the arithmetic mean of all elements in the tensor.
    ///
    /// # Returns
    /// A `Scalar<B>` containing the mean value: `sum(elements) / (rows * cols)`
    ///
    /// # Panics
    /// Panics if the tensor is empty (0 rows or 0 columns).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// // Matrix: [[1.0, 2.0],
    /// //          [3.0, 4.0]]
    /// let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
    /// let mean = t.mean();
    /// assert!((mean.to_f64() - 2.5).abs() < 1e-12);
    /// ```
    pub fn mean(&self) -> Scalar<B> {
        Scalar {
            data: B::mean_all_2d(&self.data),
            backend: PhantomData,
        }
    }

    /// Matrix-vector multiplication: computes `A @ x` (no transpose).
    ///
    /// Multiplies this (m×n) matrix by a vector of length n to produce a vector of length m.
    ///
    /// # Formula
    /// `yᵢ = Σⱼ Aᵢⱼ * xⱼ` for i ∈ [0, m)
    ///
    /// # Arguments
    /// * `other` - Vector of length n (must match matrix columns)
    ///
    /// # Returns
    /// `Tensor1D<B>` of length m
    ///
    /// # Panics
    /// Panics if `self.shape().1 != other.len()` (columns ≠ vector length).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::{Tensor2D, Tensor1D};
    ///
    /// // A = [[1.0, 2.0],
    /// //      [3.0, 4.0]]
    /// let a = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
    ///
    /// // x = [1.0, 0.0]ᵀ
    /// let x = Tensor1D::<CpuBackend>::new(vec![1.0f32, 0.0]);
    ///
    /// // A @ x = [1.0, 3.0]ᵀ
    /// let y = a.dot(&x);
    /// assert_eq!(y.to_vec(), vec![1.0, 3.0]);
    /// ```
    pub fn dot(&self, other: &Tensor1D<B>) -> Tensor1D<B> {
        Tensor1D {
            data: B::matvec(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    /// Transposed matrix-vector multiplication: computes `Aᵀ @ x`.
    ///
    /// Multiplies the transpose of this (m×n) matrix by a vector of length m
    /// to produce a vector of length n.
    ///
    /// # Formula
    /// `yⱼ = Σᵢ Aᵢⱼ * xᵢ` for j ∈ [0, n)
    ///
    /// # Arguments
    /// * `other` - Vector of length m (must match matrix rows)
    ///
    /// # Returns
    /// `Tensor1D<B>` of length n
    ///
    /// # Panics
    /// Panics if `self.shape().0 != other.len()` (rows ≠ vector length).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::{Tensor2D, Tensor1D};
    ///
    /// // A = [[1.0, 2.0],
    /// //      [3.0, 4.0]]
    /// let a = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
    ///
    /// // x = [1.0, 0.0]ᵀ
    /// let x = Tensor1D::<CpuBackend>::new(vec![1.0f32, 0.0]);
    ///
    /// // Aᵀ @ x = [1.0*1 + 3.0*0, 2.0*1 + 4.0*0]ᵀ = [1.0, 2.0]ᵀ
    /// let y = a.tdot(&x);
    /// assert_eq!(y.to_vec(), vec![1.0, 2.0]);
    /// ```
    pub fn tdot(&self, other: &Tensor1D<B>) -> Tensor1D<B> {
        Tensor1D {
            data: B::matvec_transposed(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise absolute value.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![-1.0f32, 2.0, -3.0, 4.0], 2, 2);
    /// let abs_t = t.abs();
    /// // Result: [[1.0, 2.0], [3.0, 4.0]]
    /// assert_eq!(abs_t.mean().to_f64(), 2.5);
    /// ```
    pub fn abs(&self) -> Self {
        Self {
            data: B::abs_2d(&self.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise sign function.
    ///
    /// Returns:
    /// - `1.0` for positive values
    /// - `-1.0` for negative values
    /// - `0.0` for zero values
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![-2.0f32, 0.0, 3.0, -4.0], 2, 2);
    /// let sign_t = t.sign();
    /// // Result: [[-1.0, 0.0], [1.0, -1.0]]
    /// let vec = sign_t.ravel().to_vec(); // Note: to_vec() flattens the matrix
    /// // We can verify specific elements via mean or custom checks
    /// assert!((sign_t.mean().to_f64() + 0.25).abs() < 1e-12); // (-1+0+1-1)/4 = -0.25
    /// ```
    pub fn sign(&self) -> Self {
        Self {
            data: B::sign_2d(&self.data),
            backend: PhantomData,
        }
    }

    /// Returns the number of rows in the tensor as a scalar value.
    ///
    /// # Note
    /// This returns a `Scalar<B>` rather than `usize` to maintain a uniform
    /// tensor/scalar API for backend-agnostic generic code. For most use cases
    /// requiring integer dimensions, prefer the `shape()` method.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let t = Tensor2D::<CpuBackend>::zeros(5, 3);
    /// let rows = t.len();
    /// assert_eq!(rows.to_f64(), 5.0);
    /// ```
    pub fn len(&self) -> Scalar<B> {
        Scalar {
            data: B::scalar_f64(B::len_2d(&self.data) as f64),
            backend: PhantomData,
        }
    }

    /// Scales the tensor by multiplying each element by a scalar value.
    ///
    /// Equivalent to element-wise multiplication: `self * scalar`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    /// use machinelearne_rs::backend::Scalar;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
    /// let s = Scalar::<CpuBackend>::new(2.5);
    /// let scaled = t.scale(s);
    /// // Result: [[2.5, 5.0], [7.5, 10.0]]
    /// assert!((scaled.mean().to_f64() - 6.25).abs() < 1e-12);
    /// ```
    pub fn scale(&self, a: Scalar<B>) -> Self {
        Self {
            data: B::mul_scalar_2d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    /// Adds a scalar value to each element of the tensor.
    ///
    /// Equivalent to element-wise addition: `self + scalar`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    /// use machinelearne_rs::backend::Scalar;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
    /// let s = Scalar::<CpuBackend>::new(10.0);
    /// let shifted = t.add_scalar(s);
    /// // Result: [[11.0, 12.0], [13.0, 14.0]]
    /// assert_eq!(shifted.mean().to_f64(), 12.5);
    /// ```
    pub fn add_scalar(&self, a: Scalar<B>) -> Self {
        Self {
            data: B::add_scalar_2d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise maximum between two tensors.
    ///
    /// For each index `(i, j)`, returns `max(self[i,j], other[i,j])`.
    ///
    /// # Panics
    /// Panics if tensors have different shapes.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let a = Tensor2D::<CpuBackend>::new(vec![1.0f32, 5.0, 3.0, 2.0], 2, 2);
    /// let b = Tensor2D::<CpuBackend>::new(vec![2.0f32, 4.0, 6.0, 1.0], 2, 2);
    /// let max_ab = a.maximum(b);
    /// // Result: [[2.0, 5.0], [6.0, 2.0]]
    /// assert_eq!(max_ab.mean().to_f64(), 3.75);
    /// ```
    pub fn maximum(&self, other: Self) -> Self {
        Self {
            data: B::maximum_2d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise exponential function: `e^x`.
    ///
    /// # Numerical behavior
    /// Follows IEEE 754 semantics:
    /// - `exp(0.0)` = `1.0`
    /// - `exp(+∞)` = `+∞`
    /// - `exp(-∞)` = `0.0`
    /// - Large positive inputs may return `INFINITY` on overflow
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0], 1, 2);
    /// let exp_t = t.exp();
    /// let values = exp_t.ravel().to_vec();
    /// assert!((values[0] - 1.0).abs() < 1e-12);
    /// assert!((values[1] - std::f64::consts::E).abs() < 1e-6); // f32→f64 conversion error
    /// ```
    pub fn exp(&self) -> Self {
        Self {
            data: B::exp_2d(&self.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise natural logarithm: `ln(x)`.
    ///
    /// # Behavior for edge cases (IEEE 754 compliant)
    /// - `x > 0.0` → `ln(x)` (finite value)
    /// - `x == 0.0` → `-∞` (`f64::NEG_INFINITY`)
    /// - `x < 0.0` → `NaN` (`f64::NAN`)
    ///
    /// Does **not** panic — follows standard floating-point semantics used by
    /// NumPy, PyTorch, and TensorFlow.
    ///
    /// # Numerical stability in ML
    /// For loss functions involving logarithms (e.g., cross-entropy):
    /// - Avoid raw `log(x)` on unnormalized probabilities — use `log_softmax` instead
    /// - Clip inputs when necessary: `log(max(x, ε))` with `ε = 1e-12` to avoid `-inf`
    /// - `-inf` propagates through computations and will cause `NaN` in gradients
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{CpuBackend, Tensor2D};
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 0.0, -1.0], 1, 3);
    /// let log_t = t.log();
    /// let values = log_t.ravel().to_vec();
    ///
    /// assert!((values[0] - 0.0).abs() < 1e-12);      // ln(1) = 0
    /// assert!(values[1].is_infinite() && values[1] < 0.0); // ln(0) = -inf
    /// assert!(values[2].is_nan());                   // ln(-1) = NaN
    /// ```
    pub fn log(&self) -> Self {
        Self {
            data: B::log_2d(&self.data),
            backend: PhantomData,
        }
    }

    /// Computes numerically stable sigmoid activation element-wise.
    ///
    /// Implements the logistic function: `σ(x) = 1 / (1 + e^(-x))`
    ///
    /// # Numerical stability
    /// Uses a numerically stable implementation that avoids overflow/underflow
    /// for extreme input values (e.g., ±100):
    /// - For `x >= 0`: `1 / (1 + e^(-x))`
    /// - For `x < 0`: `e^x / (1 + e^x)`
    ///
    /// # Output range
    /// Always returns values in the open interval `(0, 1)`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![-100.0f32, 0.0, 100.0], 1, 3);
    /// let sig = t.sigmoid();
    /// let values = sig.ravel().to_vec();
    ///
    /// // Extreme negative → ≈0.0
    /// assert!(values[0] < 1e-10);
    /// // Zero → 0.5 exactly
    /// assert!((values[1] - 0.5).abs() < 1e-12);
    /// // Extreme positive → ≈1.0
    /// assert!(values[2] > 1.0 - 1e-10);
    /// ```
    pub fn sigmoid(&self) -> Self {
        Self {
            data: B::sigmoid_2d(&self.data),
            backend: PhantomData,
        }
    }

    /// Returns the shape of the tensor as `(rows, columns)`.
    ///
    /// # Returns
    /// Tuple `(m, n)` where `m` is the number of rows and `n` is the number of columns.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor2D;
    ///
    /// let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 3, 1);
    /// assert_eq!(t.shape(), (3, 1));
    ///
    /// let empty = Tensor2D::<CpuBackend>::zeros(0, 5);
    /// assert_eq!(empty.shape(), (0, 5));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        B::shape(&self.data)
    }

    pub fn ravel(&self) -> Tensor1D<B> {
        Tensor1D::<B> {
            data: B::ravel_2d(&self.data),
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CpuBackend, Tensor1D};

    #[test]
    fn test_new_constructor_valid() {
        // Standard 2x3 matrix
        let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(t.shape(), (2, 3));
        assert_eq!(t.ravel().to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Inconsistent shape")]
    fn test_new_constructor_invalid_shape() {
        // Should panic when data length != rows * cols
        let _ = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 2, 2); // 3 != 4
    }

    #[test]
    fn test_zeros_constructor() {
        let t = Tensor2D::<CpuBackend>::zeros(3, 4);
        assert_eq!(t.shape(), (3, 4));
        assert_eq!(t.ravel().to_vec(), vec![0.0; 12]);
        assert_eq!(t.mean().to_f64(), 0.0);
    }

    #[test]
    fn test_sub_shape_mismatch_panics() {
        let a = Tensor2D::<CpuBackend>::zeros(2, 3);
        let b = Tensor2D::<CpuBackend>::zeros(3, 2);

        // Should panic when shapes don't match
        std::panic::catch_unwind(|| {
            let _ = a.sub(&b);
        })
        .unwrap_err();
    }

    #[test]
    fn test_dot_shape_validation() {
        let mat = Tensor2D::<CpuBackend>::zeros(3, 4); // 3x4 matrix
        let vec_wrong = Tensor1D::<CpuBackend>::zeros(3); // Should be length 4

        // Should panic: columns (4) != vector length (3)
        std::panic::catch_unwind(|| {
            let _ = mat.dot(&vec_wrong);
        })
        .unwrap_err();
    }

    #[test]
    fn test_tdot_shape_validation() {
        let mat = Tensor2D::<CpuBackend>::zeros(3, 4); // 3x4 matrix
        let vec_wrong = Tensor1D::<CpuBackend>::zeros(4); // Should be length 3

        // Should panic: rows (3) != vector length (4)
        std::panic::catch_unwind(|| {
            let _ = mat.tdot(&vec_wrong);
        })
        .unwrap_err();
    }

    #[test]
    fn test_abs_negative_values() {
        let t = Tensor2D::<CpuBackend>::new(vec![-5.0f32, -0.0, 3.0, -2.5], 2, 2);
        let abs_t = t.abs();

        let values = abs_t.ravel().to_vec();
        assert_eq!(values[0], 5.0);
        assert_eq!(values[1], 0.0); // -0.0 becomes +0.0
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 2.5);
    }

    #[test]
    fn test_sign_edge_cases() {
        let t = Tensor2D::<CpuBackend>::new(vec![-3.0f32, -0.0, 0.0, 4.2], 2, 2);
        let sign_t = t.sign();

        let values = sign_t.ravel().to_vec();
        assert_eq!(values[0], -1.0);
        assert_eq!(values[1], -0.0); // sign of -0.0 is -0.0 in IEEE 754
        assert_eq!(values[2], 0.0);
        assert_eq!(values[3], 1.0);
    }

    #[test]
    fn test_len_returns_row_count() {
        let t = Tensor2D::<CpuBackend>::zeros(7, 3);
        let len_scalar = t.len();
        assert_eq!(len_scalar.to_f64(), 7.0); // Returns ROW count, not total elements
    }

    #[test]
    fn test_scale_negative_scalar() {
        let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let s = Scalar::<CpuBackend>::new(-2.0);
        let scaled = t.scale(s);

        let values = scaled.ravel().to_vec();
        assert_eq!(values, vec![-2.0, -4.0, -6.0, -8.0]);
    }

    #[test]
    fn test_add_scalar_negative() {
        let t = Tensor2D::<CpuBackend>::new(vec![5.0f32, -3.0, 0.0, 2.0], 2, 2);
        let s = Scalar::<CpuBackend>::new(-10.0);
        let shifted = t.add_scalar(s);

        let values = shifted.ravel().to_vec();
        assert_eq!(values, vec![-5.0, -13.0, -10.0, -8.0]);
    }

    #[test]
    fn test_maximum_elementwise() {
        let a = Tensor2D::<CpuBackend>::new(vec![1.0f32, 5.0, -2.0, 0.0], 2, 2);
        let b = Tensor2D::<CpuBackend>::new(vec![3.0f32, 2.0, -5.0, 4.0], 2, 2);
        let max_ab = a.maximum(b);

        let values = max_ab.ravel().to_vec();
        assert_eq!(values, vec![3.0, 5.0, -2.0, 4.0]); // Element-wise max
    }

    #[test]
    fn test_exp_numerical_behavior() {
        let t = Tensor2D::<CpuBackend>::new(vec![0.0f32, 1.0, -1.0, 1000.0], 2, 2);
        let exp_t = t.exp();

        let values = exp_t.ravel().to_vec();
        assert!((values[0] - 1.0).abs() < 1e-12); // e^0 = 1
        assert!((values[1] - std::f64::consts::E).abs() < 1e-6); // e^1 ≈ 2.718
        assert!((values[2] - 0.367879).abs() < 1e-6); // e^-1 ≈ 0.367879
        assert!(values[3].is_infinite()); // Overflow to INF
    }

    #[test]
    fn test_log_zero_returns_negative_infinity() {
        let t = Tensor2D::<CpuBackend>::new(vec![0.0f32], 1, 1);
        let result = t.log();
        let values = result.ravel().to_vec();
        assert!(
            values[0].is_infinite(),
            "log(0.0) should return -inf, got {}",
            values[0]
        );
        assert!(values[0] < 0.0, "log(0.0) should be negative infinity");
    }

    #[test]
    fn test_log_negative_returns_nan() {
        let t = Tensor2D::<CpuBackend>::new(vec![-1.0f32, -0.001, -100.0], 1, 3);
        let result = t.log();
        let values = result.ravel().to_vec();

        assert!(
            values[0].is_nan(),
            "log(-1.0) should return NaN, got {}",
            values[0]
        );
        assert!(values[1].is_nan(), "log(-0.001) should return NaN");
        assert!(values[2].is_nan(), "log(-100.0) should return NaN");
    }

    #[test]
    fn test_log_positive_values() {
        let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, std::f32::consts::E, 10.0], 1, 3);
        let result = t.log();
        let values = result.ravel().to_vec();

        assert!((values[0] - 0.0).abs() < 1e-12, "log(1.0) = 0.0");
        assert!((values[1] - 1.0).abs() < 1e-6, "log(e) = 1.0");
        assert!((values[2] - 2.302585).abs() < 1e-6, "log(10.0) ≈ 2.302585");
    }

    #[test]
    fn test_log_valid_inputs() {
        let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, std::f32::consts::E, 10.0], 1, 3);
        let result = t.log();
        let values = result.ravel().to_vec();

        assert!((values[0] - 0.0).abs() < 1e-12, "log(1.0) = 0.0");
        assert!((values[1] - 1.0).abs() < 1e-6, "log(e) = 1.0");
        assert!((values[2] - 2.302585).abs() < 1e-6, "log(10.0) ≈ 2.302585");
    }

    #[test]
    fn test_log_small_positive_values() {
        // Critical for numerical stability in ML (e.g., log-probabilities)
        let t = Tensor2D::<CpuBackend>::new(vec![1e-20f32, 1e-10, 1e-5], 1, 3);
        let result = t.log();
        let values = result.ravel().to_vec();

        // Should not panic or return NaN for tiny positive values
        assert!(!values[0].is_nan(), "log(1e-20) should not be NaN");
        assert!(!values[1].is_nan(), "log(1e-10) should not be NaN");
        assert!(!values[2].is_nan(), "log(1e-5) should not be NaN");

        // Verify approximate values
        assert!((values[0] + 46.0517).abs() < 0.1, "log(1e-20) ≈ -46.05");
    }

    #[test]
    fn test_sigmoid_numerical_stability() {
        let t = Tensor2D::<CpuBackend>::new(vec![-1000.0f32, 0.0, 1000.0], 1, 3);
        let sig = t.sigmoid();

        let values = sig.ravel().to_vec();
        assert!(values[0] < 1e-15); // ≈0.0 for large negative
        assert!((values[1] - 0.5).abs() < 1e-12); // Exactly 0.5 at zero
        assert!(values[2] > 1.0 - 1e-15); // ≈1.0 for large positive
    }

    #[test]
    fn test_ravel_row_major_order() {
        // 3x2 matrix: [[1,2],
        //              [3,4],
        //              [5,6]]
        let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let flat = t.ravel();

        // Row-major flattening: [1,2,3,4,5,6]
        assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_empty_tensor_constructors() {
        // Valid empty tensors (0 rows or 0 cols)
        let t1 = Tensor2D::<CpuBackend>::zeros(0, 5);
        assert_eq!(t1.shape(), (0, 5));

        let t2 = Tensor2D::<CpuBackend>::zeros(5, 0);
        assert_eq!(t2.shape(), (5, 0));

        let t3 = Tensor2D::<CpuBackend>::zeros(0, 0);
        assert_eq!(t3.shape(), (0, 0));
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn test_mean_empty_tensor_panics() {
        let t = Tensor2D::<CpuBackend>::zeros(0, 0);
        let _ = t.mean(); // Should panic on empty tensor
    }

    #[test]
    fn test_clone_semantics() {
        let original = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let clone = original.clone();

        // Modify clone via backend operations (simulated by creating new tensor)
        let modified = clone.sub(&Tensor2D::<CpuBackend>::zeros(2, 2));

        // Original should remain unchanged
        assert_eq!(original.ravel().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(modified.ravel().to_vec(), vec![1.0, 2.0, 3.0, 4.0]); // Still same values
    }

    #[test]
    fn test_dot_and_tdot_consistency() {
        // Verify that dot and tdot produce transposed results
        let a = Tensor2D::<CpuBackend>::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3 matrix
            2,
            3,
        );
        let x = Tensor1D::<CpuBackend>::new(vec![1.0f32, 0.0, 0.0]); // Length 3

        // A @ x = [1*1 + 2*0 + 3*0, 4*1 + 5*0 + 6*0] = [1, 4]
        let y = a.dot(&x);
        assert_eq!(y.to_vec(), vec![1.0, 4.0]);

        // Aᵀ @ y should reconstruct x scaled by row norms (not exact inverse, but dimensionally correct)
        let x_recon = a.tdot(&y);
        assert_eq!(x_recon.len(), 3); // Should be length 3 (columns of original)
    }
}
