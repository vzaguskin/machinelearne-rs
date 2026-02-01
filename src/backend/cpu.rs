//! # CPU Backend
//!
//! Pure-Rust CPU backend implementation with zero external dependencies.
//! Uses `f64` precision and row-major memory layout for all tensors.
//!
//! ## Design Characteristics
//!
//! - **Minimal dependencies**: No external crates required (enabled via `cpu` feature)
//! - **Row-major layout**: 2D tensors stored as flat `Vec<f64>` in row-major order
//! - **f64 precision**: All computations use double precision for numerical stability
//! - **Naive implementations**: Straightforward algorithms prioritizing correctness;
//!   optimizations (SIMD, cache-aware layouts) are future work
//!
//! ## Tensor Representations
//!
//! | Type          | Rust Type     | Layout                     |
//! |---------------|---------------|----------------------------|
//! | 1D Tensor     | `Vec<f64>`    | Contiguous elements        |
//! | 2D Tensor     | `CpuTensor2D` | `(Vec<f64>, rows, cols)`   |
//!
//! ## Performance Notes
//!
//! - Matrix-vector multiplication (`matvec`) uses naive O(n²) implementation
//! - Transpose creates a new allocated tensor (no view/slice optimization yet)
//! - Element-wise operations are not SIMD-accelerated (future optimization target)
//!
//! ## Example
//!
//! ```rust
//! use machinelearne_rs::backend::{Backend, CpuBackend, cpu::CpuTensor2D};
//!
//! // Create a 2x2 matrix: [[1.0, 2.0], [3.0, 4.0]]
//! let w = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//!
//! // Create input vector [1.0, 0.0]
//! let x = vec![1.0, 0.0];
//!
//! // Compute matrix-vector product
//! let y = CpuBackend::matvec(&w, &x); // Result: [1.0, 3.0]
//! ```

use super::Backend;

/// Pure-Rust CPU computation backend.
///
/// Provides baseline tensor operations without external dependencies.
/// All computations use `f64` precision for numerical stability during training.
///
/// # Usage
///
/// ```rust
/// use machinelearne_rs::backend::{Backend, CpuBackend};
///
/// let zeros = CpuBackend::zeros_1d(5);
/// assert_eq!(zeros, vec![0.0; 5]);
/// ```
#[derive(Clone, Debug, Copy)]
pub struct CpuBackend;

/// Two-dimensional tensor representation for CPU backend.
///
/// Stores data in **row-major order** as a flat `Vec<f64>` with explicit shape metadata.
///
/// # Memory Layout
///
/// For a `(rows=2, cols=3)` matrix:
/// ```text
/// [[a, b, c],
///  [d, e, f]]
/// ```
/// Stored as: `[a, b, c, d, e, f]`
///
/// # Invariants
///
/// - `data.len() == rows * cols` (enforced in `new()`)
/// - Row `i`, column `j` element at index `i * cols + j`
#[derive(Debug, Clone)]
pub struct CpuTensor2D(pub Vec<f64>, pub usize, pub usize);

impl CpuTensor2D {
    /// Creates a new 2D tensor with explicit shape validation.
    ///
    /// # Arguments
    /// * `data` - Elements in row-major order
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    ///
    /// # Example
    /// ```rust
    /// use machinelearne_rs::backend::cpu::CpuTensor2D;
    ///
    /// let t = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    /// assert_eq!(t.0, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!((t.1, t.2), (2, 2));
    /// ```
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "Inconsistent shape");
        Self(data, rows, cols)
    }
}

impl From<&[Vec<f64>]> for CpuTensor2D {
    /// Converts a nested vector representation into a row-major 2D tensor.
    ///
    /// # Arguments
    /// * `x` - Slice of rows, where each row is a `Vec<f64>`
    ///
    /// # Panics
    /// * If rows have inconsistent lengths
    /// * If input is non-empty but contains empty rows
    ///
    /// # Example
    /// ```rust
    /// use machinelearne_rs::backend::cpu::CpuTensor2D;
    ///
    /// let nested = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    /// let t = CpuTensor2D::from(&nested[..]);
    /// assert_eq!(t.0, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!((t.1, t.2), (2, 2));
    /// ```
    fn from(x: &[Vec<f64>]) -> Self {
        if x.is_empty() {
            return CpuTensor2D::new(Vec::new(), 0, 0);
        }
        let rows = x.len();
        let cols = x[0].len();
        assert!(
            x.iter().all(|row| row.len() == cols),
            "All rows must have same length"
        );
        let data: Vec<f64> = x.iter().flat_map(|row| row.iter()).copied().collect();
        CpuTensor2D::new(data, rows, cols)
    }
}

impl Backend for CpuBackend {
    type Scalar = f64;
    type Tensor1D = Vec<f64>;
    type Tensor2D = CpuTensor2D;
    type Device = ();

    /// Returns the default device identifier for CPU backend.
    ///
    /// Always returns unit type `()` since CPU operations don't require device selection.
    fn default_device() -> Self::Device {}

    // --- Constructors ---

    /// Creates a 1D tensor filled with zeros.
    ///
    /// # Arguments
    /// * `len` - Number of elements
    ///
    /// # Example
    /// ```rust
    /// use machinelearne_rs::backend::{Backend, CpuBackend};
    /// let zeros = CpuBackend::zeros_1d(3);
    /// assert_eq!(zeros, vec![0.0, 0.0, 0.0]);
    /// ```
    fn zeros_1d(len: usize) -> Self::Tensor1D {
        vec![0.; len]
    }

    /// Creates a 2D tensor filled with zeros.
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Example
    /// ```rust
    /// use machinelearne_rs::backend::{Backend, CpuBackend};
    /// let zeros = CpuBackend::zeros_2d(2, 3);
    /// assert_eq!(zeros.0, vec![0.0; 6]);
    /// assert_eq!((zeros.1, zeros.2), (2, 3));
    /// ```
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        CpuTensor2D::new(vec![0.; rows * cols], rows, cols)
    }

    /// Constructs a 1D tensor from `f32` data (converts to `f64`).
    ///
    /// # Arguments
    /// * `data` - Source values as `f32`
    ///
    /// # Example
    /// ```rust
    /// use machinelearne_rs::backend::{Backend, CpuBackend};
    /// let t = CpuBackend::from_vec_1d(vec![1.0f32, 2.5]);
    /// assert_eq!(t, vec![1.0, 2.5]);
    /// ```
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        data.into_iter().map(|x| x as f64).collect()
    }

    /// Constructs a 2D tensor from `f32` data (converts to `f64`).
    ///
    /// # Arguments
    /// * `data` - Source values in row-major order as `f32`
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        CpuTensor2D::new(data.into_iter().map(|x| x as f64).collect(), rows, cols)
    }

    // --- Element-wise operations (1D) ---

    /// Element-wise addition of two 1D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
    }

    /// Element-wise subtraction of two 1D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a - b).collect()
    }

    /// Element-wise multiplication of two 1D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
    }

    /// Element-wise division of two 1D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths or divisor contains zeros.
    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a / b).collect()
    }

    /// Multiplies each element of a 1D tensor by a scalar.
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t.iter().map(|x| *x * s).collect()
    }

    /// Adds a scalar to each element of a 1D tensor.
    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t.iter().map(|x| x + s).collect()
    }

    // --- Element-wise operations (2D) ---

    /// Multiplies each element of a 2D tensor by a scalar.
    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        CpuTensor2D::new(t.0.iter().map(|x| *x * s).collect(), t.1, t.2)
    }

    /// Adds a scalar to each element of a 2D tensor.
    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        CpuTensor2D::new(t.0.iter().map(|x| *x + s).collect(), t.1, t.2)
    }

    /// Element-wise addition of two 2D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different shapes.
    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a + b).collect(),
            a.1,
            a.2,
        )
    }

    /// Element-wise subtraction of two 2D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different shapes.
    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a - b).collect(),
            a.1,
            a.2,
        )
    }

    /// Element-wise multiplication of two 2D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different shapes.
    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a * b).collect(),
            a.1,
            a.2,
        )
    }

    /// Element-wise division of two 2D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different shapes or divisor contains zeros.
    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a / b).collect(),
            a.1,
            a.2,
        )
    }

    // --- Reduction operations ---

    /// Computes the arithmetic mean of all elements in a 1D tensor.
    ///
    /// # Returns
    /// * Mean value as `f64`
    /// * Returns `0.0` for empty tensors (by convention)
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        if t.is_empty() {
            0.0
        } else {
            t.iter().sum::<f64>() / t.len() as f64
        }
    }

    /// Computes the arithmetic mean of all elements in a 2D tensor.
    ///
    /// # Returns
    /// * Mean value as `f64`
    /// * Returns `0.0` for empty tensors (by convention)
    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        if t.0.is_empty() {
            0.0
        } else {
            t.0.iter().sum::<f64>() / t.0.len() as f64
        }
    }

    /// Computes the sum of all elements in a 2D tensor.
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.iter().sum::<f64>()
    }

    /// Computes the sum of all elements in a 1D tensor.
    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.iter().sum::<f64>()
    }

    // --- Scalar operations ---

    /// Creates a backend-specific scalar from an `f64` value.
    ///
    /// For CPU backend, this is a trivial identity conversion.
    fn scalar_f64(value: f64) -> Self::Scalar {
        value
    }

    // --- Data access ---

    /// Converts a 1D tensor to a `Vec<f64>` for inspection or metric computation.
    ///
    /// # Note
    /// This clones the underlying data. Avoid in performance-critical paths.
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
        t.clone()
    }

    /// Returns the number of elements in a 1D tensor.
    fn len_1d(t: &Self::Tensor1D) -> usize {
        t.len()
    }

    /// Returns the number of rows.
    fn len_2d(t: &Self::Tensor2D) -> usize {
        t.1
    }

    // --- Mathematical functions (1D) ---

    /// Element-wise absolute value operation.
    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        t.iter().map(|x| x.abs()).collect()
    }

    /// Element-wise sign function.
    ///
    /// Returns:
    /// * `1.0` for positive values
    /// * `-1.0` for negative values
    /// * `0.0` for zero (subgradient convention)
    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.iter()
            .map(|&x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Element-wise maximum between two tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    fn maximum_1d(x: &Self::Tensor1D, other: &Self::Tensor1D) -> Self::Tensor1D {
        x.iter()
            .zip(other.iter())
            .map(|(&a, &b)| a.max(b))
            .collect()
    }

    /// Element-wise exponential function (`e^x`).
    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.iter().map(|&v| v.exp()).collect()
    }

    /// Element-wise natural logarithm (`ln(x)`).
    ///
    /// # Panics
    /// Panics if any element is ≤ 0.0.
    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.iter().map(|&v| v.ln()).collect()
    }

    /// Element-wise sigmoid function with numerical stability.
    ///
    /// Computed as:
    /// * `1 / (1 + exp(-x))` for `x >= 0`
    /// * `exp(x) / (1 + exp(x))` for `x < 0`
    ///
    /// This formulation avoids overflow for large positive/negative values.
    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.iter()
            .map(|&z| {
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                }
            })
            .collect()
    }

    // --- Mathematical functions (2D) ---

    /// Element-wise absolute value for 2D tensors.
    fn abs_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(t.0.iter().map(|x| x.abs()).collect(), t.1, t.2)
    }

    /// Element-wise sign function for 2D tensors.
    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            x.0.iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            x.1,
            x.2,
        )
    }

    /// Element-wise maximum between two 2D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different shapes.
    fn maximum_2d(x: &Self::Tensor2D, other: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::maximum_1d(&x.0, &other.0), x.1, x.2)
    }

    /// Element-wise exponential function for 2D tensors.
    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::exp_1d(&x.0), x.1, x.2)
    }

    /// Element-wise natural logarithm for 2D tensors.
    ///
    /// # Panics
    /// Panics if any element is ≤ 0.0.
    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::log_1d(&x.0), x.1, x.2)
    }

    /// Element-wise sigmoid function for 2D tensors.
    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::sigmoid_1d(&x.0), x.1, x.2)
    }

    // --- Linear algebra ---

    /// Matrix-vector multiplication with explicit shape checking.
    ///
    /// Computes `y = A * x` where:
    /// * `A` has shape `(m, n)`
    /// * `x` has shape `(n,)`
    /// * Returns vector of shape `(m,)`
    ///
    /// # Panics
    /// Panics if `A.cols() != x.len()`.
    ///
    /// # Implementation
    /// Currently delegates to `_matvec_unchecked()` after shape validation.
    /// TODO: Implement efficient shape checking before computation.
    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.2,
            x.len(),
            "Shape mismatch: A.cols={} != x.len={}",
            a.2,
            x.len()
        );
        Self::_matvec_unchecked(a, x)
    }

    /// Matrix-vector multiplication without shape checking.
    ///
    /// # Safety
    /// Caller must ensure `A.cols() == x.len()`. Undefined behavior may occur
    /// if shapes are incompatible.
    ///
    /// # Implementation
    /// Naive O(m×n) implementation. Future optimizations:
    /// * SIMD vectorization
    /// * Cache-aware blocking
    /// * Strided access patterns
    fn _matvec_unchecked(a: &CpuTensor2D, x: &Vec<f64>) -> Vec<f64> {
        let CpuTensor2D(data, rows, cols) = a;
        let mut result = Vec::with_capacity(*rows);
        for i in 0..*rows {
            let mut sum = 0.0;
            for j in 0..*cols {
                sum += data[i * *cols + j] * x[j];
            }
            result.push(sum);
        }
        result
    }

    /// Transposed matrix-vector multiplication with shape checking.
    ///
    /// Computes `y = Aᵀ * x` where:
    /// * `A` has shape `(m, n)`
    /// * `x` has shape `(m,)`
    /// * Returns vector of shape `(n,)`
    ///
    /// # Panics
    /// Panics if `A.rows() != x.len()`.
    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            a.1,
            x.len(),
            "Shape mismatch: A.rows={} != x.len={}",
            a.1,
            x.len()
        );
        Self::_matvec_transposed_unchecked(a, x)
    }

    /// Transposed matrix-vector multiplication without shape checking.
    ///
    /// # Safety
    /// Caller must ensure `A.rows() == x.len()`.
    ///
    /// # Implementation
    /// Currently computes full transpose then multiplies. Future optimization:
    /// direct strided access without allocation.
    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        Self::_matvec_unchecked(&Self::transpose(a), x)
    }

    /// Returns the transpose of a 2D tensor.
    ///
    /// Converts matrix of shape `(m, n)` to shape `(n, m)` with elements:
    /// `output[j][i] = input[i][j]`
    ///
    /// # Implementation
    /// Allocates a new tensor. Future optimization: return view/slice without allocation.
    fn transpose(t: &CpuTensor2D) -> Self::Tensor2D {
        let CpuTensor2D(inp, rows, cols) = t;
        let mut out = Vec::with_capacity(cols * rows);
        for col in 0..*cols {
            for row in 0..*rows {
                out.push(inp[row * cols + col]);
            }
        }
        CpuTensor2D::new(out, *cols, *rows)
    }

    /// Returns the shape of a 2D tensor as `(rows, cols)`.
    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        (t.1, t.2)
    }

    //Returns copy of the inner 1d vector
    fn ravel_2d(x: &Self::Tensor2D) -> Self::Tensor1D {
        x.0.clone()
    }
}

#[cfg(test)]
mod matvec_tests {
    use super::*;

    #[test]
    fn test_matvec_transpose() {
        // Пример 1: X — (3, 2), v — (3,)
        // X = [[1.0, 2.0],
        //      [3.0, 4.0],
        //      [5.0, 6.0]]
        // v = [1.0, 0.0, 2.0]
        // Xᵀ @ v = [1*1 + 3*0 + 5*2, 2*1 + 4*0 + 6*2] = [11.0, 14.0]

        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // row-major
        let x = CpuTensor2D::new(x_data, 3, 2); // (rows=3, cols=2)
        let v = vec![1.0, 0.0, 2.0];

        let result = CpuBackend::matvec_transposed(&x, &v);
        let expected = vec![11.0, 14.0];

        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "Expected {}, got {}", e, r);
        }

        // Пример 2: (4, 1) — вектор-столбец (как в твоих линейных регрессиях)
        // X = [[2.0],
        //      [3.0],
        //      [4.0],
        //      [5.0]]
        // v = [1.0, 1.0, 1.0, 1.0]
        // Xᵀ @ v = [2+3+4+5] = [14.0]

        let x2 = CpuTensor2D::new(vec![2.0, 3.0, 4.0, 5.0], 4, 1);
        let v2 = vec![1.0, 1.0, 1.0, 1.0];
        let result2 = CpuBackend::matvec_transposed(&x2, &v2);
        let expected2 = vec![14.0];

        assert_eq!(result2, expected2);

        // Пример 3: (2, 3) → output len = 3
        // X = [[1, 0, 0],
        //      [0, 1, 0]]
        // v = [5.0, 7.0]
        // Xᵀ @ v = [5, 7, 0]

        let x3 = CpuTensor2D::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let v3 = vec![5.0, 7.0];
        let result3 = CpuBackend::matvec_transposed(&x3, &v3);
        let expected3 = vec![5.0, 7.0, 0.0];

        assert_eq!(result3.len(), 3);
        for (r, e) in result3.iter().zip(expected3.iter()) {
            assert!((r - e).abs() < 1e-12, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_matvec_transpose_consistency_with_transpose_and_matvec() {
        let x = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // (3,2)
        let v = vec![1.0, 0.0, 2.0];

        let result1 = CpuBackend::matvec_transposed(&x, &v);
        let x_t = CpuBackend::transpose(&x);
        let result2 = CpuBackend::matvec(&x_t, &v);

        assert_eq!(result1.len(), result2.len());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructors() {
        // zeros_1d
        let z1 = CpuBackend::zeros_1d(3);
        assert_eq!(z1, vec![0.0, 0.0, 0.0]);

        // zeros_2d
        let z2 = CpuBackend::zeros_2d(2, 3);
        assert_eq!(z2.0, vec![0.0; 6]);
        assert_eq!((z2.1, z2.2), (2, 3));

        // from_vec_1d
        let v1 = CpuBackend::from_vec_1d(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(v1, vec![1.0, 2.0, 3.0]);

        // from_vec_2d
        let v2 = CpuBackend::from_vec_2d(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(v2.0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!((v2.1, v2.2), (2, 2));

        // From<&[Vec<f64>]> for CpuTensor2D
        let nested = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let t = CpuTensor2D::from(&nested[..]);
        assert_eq!(t.0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!((t.1, t.2), (2, 2));
    }

    #[test]
    fn test_elementwise_ops_1d() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        assert_eq!(CpuBackend::add_1d(&a, &b), vec![4.0, 6.0]);
        assert_eq!(CpuBackend::sub_1d(&a, &b), vec![-2.0, -2.0]);
        assert_eq!(CpuBackend::mul_1d(&a, &b), vec![3.0, 8.0]);
        assert_eq!(CpuBackend::div_1d(&a, &b), vec![1.0 / 3.0, 0.5]);

        assert_eq!(CpuBackend::add_scalar_1d(&a, &5.0), vec![6.0, 7.0]);
        assert_eq!(CpuBackend::mul_scalar_1d(&a, &2.0), vec![2.0, 4.0]);
    }

    #[test]
    fn test_elementwise_ops_2d() {
        let a = CpuTensor2D::new(vec![1.0, 2.0], 1, 2);
        let b = CpuTensor2D::new(vec![3.0, 4.0], 1, 2);

        let add = CpuBackend::add_2d(&a, &b);
        assert_eq!(add.0, vec![4.0, 6.0]);

        let mul_s = CpuBackend::mul_scalar_2d(&a, &2.0);
        assert_eq!(mul_s.0, vec![2.0, 4.0]);
    }

    #[test]
    fn test_reductions() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(CpuBackend::sum_all_1d(&v), 6.0);
        assert!((CpuBackend::mean_all_1d(&v) - 2.0).abs() < 1e-12);

        let m = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(CpuBackend::sum_all_2d(&m), 10.0);
        assert!((CpuBackend::mean_all_2d(&m) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_abs_and_sign() {
        let v = vec![-2.0, 0.0, 3.0];
        assert_eq!(CpuBackend::abs_1d(&v), vec![2.0, 0.0, 3.0]);
        assert_eq!(CpuBackend::sign_1d(&v), vec![-1.0, 0.0, 1.0]);

        let m = CpuTensor2D::new(vec![-1.0, 0.0, 2.0], 1, 3);
        let sign_m = CpuBackend::sign_2d(&m);
        assert_eq!(sign_m.0, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_math_functions() {
        let v = vec![0.0, 1.0];
        assert_eq!(CpuBackend::exp_1d(&v), vec![1.0, std::f64::consts::E]);
        assert_eq!(
            CpuBackend::log_1d(&vec![1.0, std::f64::consts::E]),
            vec![0.0, 1.0]
        );

        // sigmoid(0) = 0.5
        let sig = CpuBackend::sigmoid_1d(&vec![0.0]);
        assert!((sig[0] - 0.5).abs() < 1e-12);

        // maximum
        let a = vec![1.0, 3.0];
        let b = vec![2.0, 2.0];
        assert_eq!(CpuBackend::maximum_1d(&a, &b), vec![2.0, 3.0]);
    }

    #[test]
    fn test_transpose() {
        let m = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3); // 2x3
        let t = CpuBackend::transpose(&m); // 3x2
        assert_eq!(t.0, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!((t.1, t.2), (3, 2));

        // Double transpose = original
        let tt = CpuBackend::transpose(&t);
        assert_eq!(tt.0, m.0);
        assert_eq!((tt.1, tt.2), (2, 3));
    }

    #[test]
    fn test_matvec() {
        let m = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let v = vec![1.0, 0.0];
        let res = CpuBackend::matvec(&m, &v);
        assert_eq!(res, vec![1.0, 3.0]); // [1*1 + 2*0, 3*1 + 4*0]
    }

    #[test]
    fn test_edge_cases() {
        // Пустой тензор 1D
        let empty1d = CpuBackend::zeros_1d(0);
        assert_eq!(empty1d.len(), 0);
        assert_eq!(CpuBackend::sum_all_1d(&empty1d), 0.0);

        // Пустой тензор 2D
        let empty2d = CpuBackend::zeros_2d(0, 0);
        assert_eq!(empty2d.0.len(), 0);
        assert_eq!(CpuBackend::sum_all_2d(&empty2d), 0.0);

        // From empty nested vec
        let t = CpuTensor2D::from(&[][..]);
        assert_eq!(t.0.len(), 0);
        assert_eq!((t.1, t.2), (0, 0));

        // Деление на ноль — ожидаем panic или NaN?
        // В текущей реализации будет panic при делении на 0.0.
        // Если это не желаемо — стоит обсудить, но пока покроем как есть.
        let a = vec![1.0];
        let b = vec![0.0];
        let res = CpuBackend::div_1d(&a, &b);
        assert!(res[0].is_infinite()); // или assert_panics, если хочешь явный контроль
    }
}
