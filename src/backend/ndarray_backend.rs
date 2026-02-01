use super::Backend;
use ndarray::{Array1, Array2, Ix1, Ix2};

/// CPU-based tensor backend implementation using the `ndarray` crate.
///
/// This backend provides efficient CPU tensor operations for machine learning workloads
/// using ndarray's optimized linear algebra routines. It supports both 1D and 2D tensors
/// with element-wise operations, reductions, and matrix-vector products.
///
/// # Type mappings
/// - `Scalar`: `f64` (double-precision floating point)
/// - `Tensor1D`: `ndarray::Array1<f64>` (1-dimensional array)
/// - `Tensor2D`: `NdarrayTensor2D` wrapper around `ndarray::Array2<f64>`
/// - `Device`: `()` (unit type, CPU-only execution)
///
/// # Numerical stability
/// Non-linear operations like `sigmoid` use numerically stable implementations
/// to avoid overflow/underflow for extreme input values (e.g., ±100).
#[derive(Clone, Debug, Copy)]
pub struct NdarrayBackend;

/// Wrapper type for 2D tensors using ndarray's `Array2<f64>`.
///
/// This wrapper enables trait implementation for external types while providing
/// convenient conversion from nested Vec representations commonly used in tests
/// and data loading.
///
/// # Example
/// ```
/// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
/// use ndarray::{Array1, Array2, Ix1, Ix2};
/// let tensor = NdarrayTensor2D::from(&[
///     vec![1.0f64, 2.0f64, 3.0f64],
///     vec![4.0f64, 5.0f64, 6.0f64],
/// ][..]);
/// assert_eq!(tensor.0.shape(), &[2, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct NdarrayTensor2D(pub Array2<f64>);

impl From<&[Vec<f64>]> for NdarrayTensor2D {
    /// Converts a slice of row vectors into a 2D tensor.
    ///
    /// # Panics
    /// Panics if rows have inconsistent lengths or if shape reconstruction fails.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let data = &[vec![1.0f64, 2.0f64], vec![3.0f64, 4.0f64]][..];
    /// let tensor = NdarrayTensor2D::from(data);
    /// assert_eq!(tensor.0[[0, 0]], 1.0);
    /// assert_eq!(tensor.0[[1, 1]], 4.0);
    /// ```
    fn from(x: &[Vec<f64>]) -> Self {
        let rows = x.len();
        if rows == 0 {
            return NdarrayTensor2D(Array2::from_shape_vec((0, 0), vec![]).unwrap());
        }
        let cols = x[0].len();
        assert!(x.iter().all(|r| r.len() == cols));
        let data: Vec<f64> = x.iter().flat_map(|r| r.iter()).copied().collect();
        NdarrayTensor2D(Array2::from_shape_vec((rows, cols), data).unwrap())
    }
}

impl super::Backend for NdarrayBackend {
    type Scalar = f64;
    type Tensor1D = Array1<f64>;
    type Tensor2D = NdarrayTensor2D;
    type Device = (); // CPU-only for now

    /// Returns the default device for this backend (unit type for CPU execution).
    fn default_device() -> Self::Device {
        ()
    }

    /// Creates a 1D tensor filled with zeros.
    ///
    /// # Arguments
    /// * `len` - Length of the resulting 1D tensor
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// let zeros = NdarrayBackend::zeros_1d(3);
    /// assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0]);
    /// ```
    fn zeros_1d(len: usize) -> Self::Tensor1D {
        Array1::zeros(len)
    }

    /// Creates a 2D tensor filled with zeros.
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// let zeros = NdarrayBackend::zeros_2d(2, 3);
    /// assert_eq!(NdarrayBackend::shape(&zeros), (2, 3));
    /// ```
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        NdarrayTensor2D(Array2::zeros((rows, cols)))
    }

    /// Converts a vector of f32 values to a 1D tensor (f64 precision).
    ///
    /// # Arguments
    /// * `data` - Input vector with f32 values
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// let tensor = NdarrayBackend::from_vec_1d(vec![1.0f32, 2.0, 3.0]);
    /// assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        Array1::from_iter(data.into_iter().map(|x| x as f64))
    }

    /// Converts a flat vector of f32 values to a 2D tensor with specified shape.
    ///
    /// # Arguments
    /// * `data` - Flat vector containing row-major ordered elements
    /// * `rows` - Number of rows in output tensor
    /// * `cols` - Number of columns in output tensor
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// let tensor = NdarrayBackend::from_vec_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    /// assert_eq!(tensor.0[[0, 0]], 1.0);
    /// assert_eq!(tensor.0[[1, 1]], 4.0);
    /// ```
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        let data_f64: Vec<f64> = data.into_iter().map(|x| x as f64).collect();
        NdarrayTensor2D(Array2::from_shape_vec((rows, cols), data_f64).unwrap())
    }

    /// Element-wise addition of two 1D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let a = Array1::from_vec(vec![1.0, 2.0]);
    /// let b = Array1::from_vec(vec![3.0, 4.0]);
    /// let c = NdarrayBackend::add_1d(&a, &b);
    /// assert_eq!(c.to_vec(), vec![4.0, 6.0]);
    /// ```
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a + b
    }

    /// Multiplies a 2D tensor by a scalar value.
    ///
    /// # Arguments
    /// * `t` - Input 2D tensor
    /// * `s` - Scalar multiplier
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let t = NdarrayTensor2D(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    /// let scaled = NdarrayBackend::mul_scalar_2d(&t, &2.0);
    /// assert_eq!(scaled.0[[0, 1]], 4.0);
    /// ```
    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        NdarrayTensor2D(&t.0 * *s)
    }

    /// Matrix-vector multiplication: `A @ x`.
    ///
    /// Computes the product of a 2D matrix and a 1D vector.
    ///
    /// # Arguments
    /// * `a` - Matrix of shape (m, n)
    /// * `x` - Vector of shape (n,)
    ///
    /// # Returns
    /// Vector of shape (m,)
    ///
    /// # Panics
    /// Panics if matrix columns != vector length.
    ///
    /// # Example
    /// ```
    /// // [[1, 2],    [1]   [5]
    /// //  [3, 4]] @ [2] = [11]
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let a = NdarrayTensor2D::from(&[vec![1.0, 2.0], vec![3.0, 4.0]][..]);
    /// let x = Array1::from_vec(vec![1.0, 2.0]);
    /// let y = NdarrayBackend::matvec(&a, &x);
    /// assert_eq!(y.to_vec(), vec![5.0, 11.0]);
    /// ```
    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.0.dot(x) // ndarray has efficient matvec
    }

    /// Transposed matrix-vector multiplication: `A^T @ x`.
    ///
    /// Computes the product of a transposed matrix and a vector.
    ///
    /// # Arguments
    /// * `a` - Matrix of shape (m, n)
    /// * `x` - Vector of shape (m,)
    ///
    /// # Returns
    /// Vector of shape (n,)
    ///
    /// # Example
    /// ```
    /// // [[1, 2],^T   [1]   [1]
    /// //  [3, 4]]  @ [0] = [2]
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let a = NdarrayTensor2D::from(&[vec![1.0, 2.0], vec![3.0, 4.0]][..]);
    /// let x = Array1::from_vec(vec![1.0, 0.0]);
    /// let y = NdarrayBackend::matvec_transposed(&a, &x);
    /// assert_eq!(y.to_vec(), vec![1.0, 2.0]);
    /// ```
    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.0.t().dot(x)
    }

    /// Returns the transpose of a 2D tensor.
    ///
    /// # Arguments
    /// * `a` - Input tensor of shape (m, n)
    ///
    /// # Returns
    /// Tensor of shape (n, m)
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// let a = NdarrayTensor2D::from(&[vec![1.0, 2.0], vec![3.0, 4.0]][..]);
    /// let at = NdarrayBackend::transpose(&a);
    /// assert_eq!(at.0[[0, 1]], 3.0); // Original [1,0] becomes [0,1]
    /// ```
    fn transpose(a: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(a.0.t().to_owned())
    }

    /// Returns the shape of a 2D tensor as (rows, columns).
    ///
    /// # Arguments
    /// * `t` - Input 2D tensor
    ///
    /// # Returns
    /// Tuple `(rows, cols)`
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let t = NdarrayTensor2D::from(&[vec![1.0, 2.0, 3.0]][..]);
    /// assert_eq!(NdarrayBackend::shape(&t), (1, 3));
    /// ```
    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        let shape = t.0.shape();
        (shape[0], shape[1])
    }

    // --- Element-wise non-linear ops (1D) ---

    /// Computes element-wise exponential: `e^x`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let x = Array1::from_vec(vec![0.0, 1.0]);
    /// let y = NdarrayBackend::exp_1d(&x);
    /// assert!((y[0] - 1.0).abs() < 1e-6);
    /// assert!((y[1] - std::f64::consts::E).abs() < 1e-6);
    /// ```
    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(f64::exp)
    }

    /// Computes element-wise natural logarithm: `ln(x)`.
    ///
    /// # Panics
    /// Panics for non-positive values (ndarray behavior).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let x = Array1::from_vec(vec![1.0, std::f64::consts::E]);
    /// let y = NdarrayBackend::log_1d(&x);
    /// assert!((y[0] - 0.0).abs() < 1e-6);
    /// assert!((y[1] - 1.0).abs() < 1e-6);
    /// ```
    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(f64::ln)
    }

    /// Computes numerically stable sigmoid activation: `1 / (1 + e^{-x})`.
    ///
    /// Uses a numerically stable implementation that avoids overflow for
    /// extreme values (±100+).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let x = Array1::from_vec(vec![-100.0, 0.0, 100.0]);
    /// let y = NdarrayBackend::sigmoid_1d(&x);
    /// assert!((y[0] - 0.0).abs() < 1e-6);   // ≈0
    /// assert!((y[1] - 0.5).abs() < 1e-6);   // =0.5
    /// assert!((y[2] - 1.0).abs() < 1e-6);   // ≈1
    /// ```
    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(|z| {
            if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            }
        })
    }

    /// Computes element-wise absolute value.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let x = Array1::from_vec(vec![-2.0, 3.0]);
    /// let y = NdarrayBackend::abs_1d(&x);
    /// assert_eq!(y.to_vec(), vec![2.0, 3.0]);
    /// ```
    fn abs_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(f64::abs)
    }

    /// Computes element-wise sign function.
    ///
    /// Returns -1.0 for negative values, 0.0 for zero, 1.0 for positive values.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let x = Array1::from_vec(vec![-2.0, 0.0, 3.0]);
    /// let y = NdarrayBackend::sign_1d(&x);
    /// assert_eq!(y.to_vec(), vec![-1.0, 0.0, 1.0]);
    /// ```
    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    /// Element-wise maximum of two 1D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::{NdarrayBackend, NdarrayTensor2D, Backend};
    /// use ndarray::{Array1, Array2, Ix1, Ix2};
    /// let a = Array1::from_vec(vec![1.0, 5.0, 3.0]);
    /// let b = Array1::from_vec(vec![2.0, 4.0, 6.0]);
    /// let m = NdarrayBackend::maximum_1d(&a, &b);
    /// assert_eq!(m.to_vec(), vec![2.0, 5.0, 6.0]);
    /// ```
    fn maximum_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(a.len(), b.len(), "Shapes must match");
        a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
    }

    // --- 2D versions (delegating to 1D via flat view or direct map) ---

    /// Computes element-wise exponential for 2D tensors.
    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(x.0.mapv(f64::exp))
    }

    /// Computes element-wise natural logarithm for 2D tensors.
    ///
    /// # Panics
    /// Panics for non-positive values.
    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(x.0.mapv(f64::ln))
    }

    /// Computes numerically stable sigmoid activation for 2D tensors.
    ///
    /// Uses the same stable implementation as `sigmoid_1d` applied element-wise.
    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(x.0.mapv(|z| {
            if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            }
        }))
    }

    /// Computes element-wise absolute value for 2D tensors.
    fn abs_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(x.0.mapv(f64::abs))
    }

    /// Computes element-wise sign function for 2D tensors.
    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(x.0.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }))
    }

    /// Element-wise maximum of two 2D tensors.
    ///
    /// # Panics
    /// Panics if tensors have different shapes.
    fn maximum_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = a.0.dim();
        assert_eq!(a.0.dim(), b.0.dim(), "Shapes must match");
        let data: Vec<f64> =
            a.0.iter()
                .zip(b.0.iter())
                .map(|(&x, &y)| x.max(y))
                .collect();
        NdarrayTensor2D(Array2::from_shape_vec((rows, cols), data).unwrap())
    }

    // --- Reductions (already partially covered, but for completeness) ---

    /// Sums all elements in a 1D tensor.
    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.sum()
    }

    /// Computes mean of all elements in a 1D tensor.
    ///
    /// # Panics
    /// Panics if tensor is empty (ndarray behavior).
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.mean().unwrap()
    }

    /// Sums all elements in a 2D tensor.
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.sum()
    }

    /// Computes mean of all elements in a 2D tensor.
    ///
    /// # Panics
    /// Panics if tensor is empty.
    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.mean().unwrap()
    }

    // --- Scalar and access helpers ---

    /// Creates a scalar value from f64.
    ///
    /// This is a trivial conversion since `Scalar = f64`.
    fn scalar_f64(value: f64) -> Self::Scalar {
        value
    }

    /// Returns the length (number of elements) of a 1D tensor.
    fn len_1d(t: &Self::Tensor1D) -> usize {
        t.len()
    }

    /// Returns the number of rows in a 2D tensor.
    ///
    /// Note: This returns `nrows()`, not total element count.
    fn len_2d(t: &Self::Tensor2D) -> usize {
        t.0.nrows()
    }

    /// Converts a 1D tensor to a standard Vec<f64>.
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
        t.to_vec()
    }

    // --- Element-wise binary ops (1D) ---

    /// Element-wise subtraction of 1D tensors.
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a - b
    }

    /// Element-wise multiplication of 1D tensors.
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a * b
    }

    /// Element-wise division of 1D tensors.
    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a / b
    }

    /// Multiplies each element of a 1D tensor by a scalar.
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t.mapv(|x| x * *s)
    }

    /// Adds a scalar to each element of a 1D tensor.
    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t.mapv(|x| x + *s)
    }

    // --- 2D scalar and binary ops ---

    /// Adds a scalar to each element of a 2D tensor.
    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        NdarrayTensor2D(t.0.mapv(|x| x + *s))
    }

    /// Element-wise addition of 2D tensors.
    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(&a.0 + &b.0)
    }

    /// Element-wise subtraction of 2D tensors.
    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(&a.0 - &b.0)
    }

    /// Element-wise multiplication of 2D tensors (Hadamard product).
    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(&a.0 * &b.0)
    }

    /// Element-wise division of 2D tensors.
    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(&a.0 / &b.0)
    }

    // --- "Unchecked" matvec helpers (для совместимости с CpuBackend) ---
    // В ndarray они не нужны, но трейт требует — делаем просто обёртки

    /// Unchecked matrix-vector multiplication (same as `matvec`).
    ///
    /// Provided for trait compatibility; delegates to `matvec`.
    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        Self::matvec(a, x)
    }

    /// Unchecked transposed matrix-vector multiplication (same as `matvec_transposed`).
    ///
    /// Provided for trait compatibility; delegates to `matvec_transposed`.
    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        Self::matvec_transposed(a, x)
    }

    //Returns copy of the inner 1d vector
    fn ravel_2d(x: &Self::Tensor2D) -> Self::Tensor1D {
        x.0.clone()
            .into_dimensionality::<Ix1>()
            .expect("Failed to ravel 2D tensor: shape conversion error")
    }
}

#[cfg(test)]
#[cfg(feature = "ndarray")]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    // Helper to create 2D tensor from nested vec
    fn tensor2d_from(data: &[Vec<f64>]) -> NdarrayTensor2D {
        NdarrayTensor2D::from(data)
    }

    #[test]
    fn test_zeros_1d() {
        let t = NdarrayBackend::zeros_1d(3);
        assert_eq!(t.to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zeros_2d() {
        let t = NdarrayBackend::zeros_2d(2, 3);
        assert_eq!(NdarrayBackend::shape(&t), (2, 3));
        assert_eq!(t.0.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn test_from_vec_1d() {
        let t = NdarrayBackend::from_vec_1d(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_vec_2d() {
        let t = NdarrayBackend::from_vec_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(NdarrayBackend::shape(&t), (2, 2));
        assert_eq!(
            t.0,
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap()
        );
    }

    #[test]
    fn test_from_nested_vec() {
        let data = &[vec![1.0, 2.0], vec![3.0, 4.0]];
        let t = tensor2d_from(data);
        assert_eq!(
            t.0,
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap()
        );
    }

    #[test]
    fn test_add_1d() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        let c = NdarrayBackend::add_1d(&a, &b);
        assert_eq!(c.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_add_2d() {
        let a = tensor2d_from(&[vec![1.0, 2.0]]);
        let b = tensor2d_from(&[vec![3.0, 4.0]]);
        let c = NdarrayBackend::add_2d(&a, &b);
        assert_eq!(c.0, Array2::from_shape_vec((1, 2), vec![4.0, 6.0]).unwrap());
    }

    #[test]
    fn test_sub_1d() {
        let a = Array1::from_vec(vec![5.0, 6.0]);
        let b = Array1::from_vec(vec![2.0, 1.0]);
        let c = NdarrayBackend::sub_1d(&a, &b);
        assert_eq!(c.to_vec(), vec![3.0, 5.0]);
    }

    #[test]
    fn test_sub_2d() {
        let a = tensor2d_from(&[vec![5.0, 6.0]]);
        let b = tensor2d_from(&[vec![2.0, 1.0]]);
        let c = NdarrayBackend::sub_2d(&a, &b);
        assert_eq!(c.0, Array2::from_shape_vec((1, 2), vec![3.0, 5.0]).unwrap());
    }

    #[test]
    fn test_mul_1d() {
        let a = Array1::from_vec(vec![2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0]);
        let c = NdarrayBackend::mul_1d(&a, &b);
        assert_eq!(c.to_vec(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_mul_2d() {
        let a = tensor2d_from(&[vec![2.0, 3.0]]);
        let b = tensor2d_from(&[vec![4.0, 5.0]]);
        let c = NdarrayBackend::mul_2d(&a, &b);
        assert_eq!(
            c.0,
            Array2::from_shape_vec((1, 2), vec![8.0, 15.0]).unwrap()
        );
    }

    #[test]
    fn test_div_1d() {
        let a = Array1::from_vec(vec![8.0, 15.0]);
        let b = Array1::from_vec(vec![2.0, 3.0]);
        let c = NdarrayBackend::div_1d(&a, &b);
        assert_eq!(c.to_vec(), vec![4.0, 5.0]);
    }

    #[test]
    fn test_div_2d() {
        let a = tensor2d_from(&[vec![8.0, 15.0]]);
        let b = tensor2d_from(&[vec![2.0, 3.0]]);
        let c = NdarrayBackend::div_2d(&a, &b);
        assert_eq!(c.0, Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap());
    }

    #[test]
    fn test_add_scalar_1d() {
        let t = Array1::from_vec(vec![1.0, 2.0]);
        let s = 10.0;
        let out = NdarrayBackend::add_scalar_1d(&t, &s);
        assert_eq!(out.to_vec(), vec![11.0, 12.0]);
    }

    #[test]
    fn test_add_scalar_2d() {
        let t = tensor2d_from(&[vec![1.0, 2.0]]);
        let s = 10.0;
        let out = NdarrayBackend::add_scalar_2d(&t, &s);
        assert_eq!(
            out.0,
            Array2::from_shape_vec((1, 2), vec![11.0, 12.0]).unwrap()
        );
    }

    #[test]
    fn test_mul_scalar_1d() {
        let t = Array1::from_vec(vec![2.0, 3.0]);
        let s = 5.0;
        let out = NdarrayBackend::mul_scalar_1d(&t, &s);
        assert_eq!(out.to_vec(), vec![10.0, 15.0]);
    }

    #[test]
    fn test_mul_scalar_2d() {
        let t = tensor2d_from(&[vec![2.0, 3.0]]);
        let s = 5.0;
        let out = NdarrayBackend::mul_scalar_2d(&t, &s);
        assert_eq!(
            out.0,
            Array2::from_shape_vec((1, 2), vec![10.0, 15.0]).unwrap()
        );
    }

    #[test]
    fn test_matvec() {
        let a = tensor2d_from(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = NdarrayBackend::matvec(&a, &x);
        assert_eq!(y.to_vec(), vec![5.0, 11.0]); // 1*1+2*2=5, 3*1+4*2=11
    }

    #[test]
    fn test_matvec_transposed() {
        let a = tensor2d_from(&[vec![1.0, 2.0], vec![3.0, 4.0]]); // 2x2
        let x = Array1::from_vec(vec![1.0, 0.0]); // shape (2,)
        let y = NdarrayBackend::matvec_transposed(&a, &x); // A^T @ x → (2,)
                                                           // A^T = [[1,3],[2,4]], so [1*1 + 3*0, 2*1 + 4*0] = [1, 2]
        assert_eq!(y.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_transpose() {
        let a = tensor2d_from(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let at = NdarrayBackend::transpose(&a);
        assert_eq!(NdarrayBackend::shape(&at), (2, 2));
        assert_eq!(
            at.0,
            Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 2.0, 4.0]).unwrap()
        );
    }

    #[test]
    fn test_shape() {
        let a = tensor2d_from(&[vec![1.0, 2.0, 3.0]]);
        assert_eq!(NdarrayBackend::shape(&a), (1, 3));
    }

    #[test]
    fn test_len_1d() {
        let t = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(NdarrayBackend::len_1d(&t), 3);
    }

    #[test]
    fn test_len_2d() {
        let t = tensor2d_from(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(NdarrayBackend::len_2d(&t), 2); // nrows
    }

    #[test]
    fn test_to_vec_1d() {
        let t = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(NdarrayBackend::to_vec_1d(&t), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_exp_1d() {
        let t = Array1::from_vec(vec![0.0, 1.0]);
        let out = NdarrayBackend::exp_1d(&t);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_log_1d() {
        let t = Array1::from_vec(vec![1.0, std::f64::consts::E]);
        let out = NdarrayBackend::log_1d(&t);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_stability() {
        let input = Array1::from_vec(vec![-100.0, 0.0, 100.0]);
        let out = NdarrayBackend::sigmoid_1d(&input);
        let expected = vec![0.0, 0.5, 1.0];
        for (o, e) in out.iter().zip(expected) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_abs_1d() {
        let t = Array1::from_vec(vec![-2.0, 3.0]);
        let out = NdarrayBackend::abs_1d(&t);
        assert_eq!(out.to_vec(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_sign_1d() {
        let t = Array1::from_vec(vec![-2.0, 0.0, 3.0]);
        let out = NdarrayBackend::sign_1d(&t);
        assert_eq!(out.to_vec(), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_maximum_1d() {
        let a = Array1::from_vec(vec![1.0, 5.0, 3.0]);
        let b = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let out = NdarrayBackend::maximum_1d(&a, &b);
        assert_eq!(out.to_vec(), vec![2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_exp_2d() {
        let t = tensor2d_from(&[vec![0.0, 1.0]]);
        let out = NdarrayBackend::exp_2d(&t);
        assert!((out.0[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((out.0[[0, 1]] - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_log_2d() {
        let t = tensor2d_from(&[vec![1.0, std::f64::consts::E]]);
        let out = NdarrayBackend::log_2d(&t);
        assert!((out.0[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((out.0[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_2d() {
        let t = tensor2d_from(&[vec![-100.0, 0.0, 100.0]]);
        let out = NdarrayBackend::sigmoid_2d(&t);
        let expected = vec![0.0, 0.5, 1.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!((out.0[[0, i]] - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_abs_2d() {
        let t = tensor2d_from(&[vec![-2.0, 3.0]]);
        let out = NdarrayBackend::abs_2d(&t);
        assert_eq!(
            out.0,
            Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).unwrap()
        );
    }

    #[test]
    fn test_sign_2d() {
        let t = tensor2d_from(&[vec![-2.0, 0.0, 3.0]]);
        let out = NdarrayBackend::sign_2d(&t);
        assert_eq!(
            out.0,
            Array2::from_shape_vec((1, 3), vec![-1.0, 0.0, 1.0]).unwrap()
        );
    }

    #[test]
    fn test_maximum_2d() {
        let a = tensor2d_from(&[vec![1.0, 5.0, 3.0]]);
        let b = tensor2d_from(&[vec![2.0, 4.0, 6.0]]);
        let out = NdarrayBackend::maximum_2d(&a, &b);
        assert_eq!(
            out.0,
            Array2::from_shape_vec((1, 3), vec![2.0, 5.0, 6.0]).unwrap()
        );
    }

    #[test]
    fn test_sum_all_1d() {
        let t = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(NdarrayBackend::sum_all_1d(&t), 6.0);
    }

    #[test]
    fn test_mean_all_1d() {
        let t = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((NdarrayBackend::mean_all_1d(&t) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_all_2d() {
        let t = tensor2d_from(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(NdarrayBackend::sum_all_2d(&t), 10.0);
    }

    #[test]
    fn test_mean_all_2d() {
        let t = tensor2d_from(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!((NdarrayBackend::mean_all_2d(&t) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_f64() {
        assert_eq!(NdarrayBackend::scalar_f64(42.0), 42.0);
    }

    #[test]
    fn test_empty_tensor_2d() {
        let empty: &[Vec<f64>] = &[];
        let t = tensor2d_from(empty);
        assert_eq!(NdarrayBackend::shape(&t), (0, 0));
    }

    #[test]
    #[should_panic(expected = "Shapes must match")]
    fn test_maximum_1d_mismatch() {
        let a = Array1::from_vec(vec![1.0]);
        let b = Array1::from_vec(vec![1.0, 2.0]);
        NdarrayBackend::maximum_1d(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Shapes must match")]
    fn test_maximum_2d_mismatch() {
        let a = tensor2d_from(&[vec![1.0]]);
        let b = tensor2d_from(&[vec![1.0, 2.0]]);
        NdarrayBackend::maximum_2d(&a, &b);
    }
}
