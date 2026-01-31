use super::scalar::Scalar;
use crate::backend::Backend;
use std::marker::PhantomData;

/// Backend-typed 1D tensor providing compile-time type safety and zero-cost abstractions.
///
/// Wraps a backend's native 1D tensor representation (`B::Tensor1D`) while carrying phantom
/// type information about its originating backend. This prevents accidental mixing of tensors
/// from different backends at compile time while maintaining performance through zero-sized
/// `PhantomData` overhead.
///
/// # Type safety guarantees
/// ```compile_fail
/// use machinelearne_rs::backend::{CpuBackend, NdarrayBackend};
/// use machinelearne_rs::backend::Tensor1D;
///
/// let cpu_tensor: Tensor1D<CpuBackend> = Tensor1D::zeros(3);
/// let ndarray_tensor: Tensor1D<NdarrayBackend> = Tensor1D::zeros(3);
/// let _ = cpu_tensor.sub(&ndarray_tensor); // COMPILE ERROR: mismatched backends
/// ```
///
/// # Precision semantics
/// - Constructors accept `Vec<f32>` for ergonomic data loading from common sources
/// - Values are immediately converted to backend's native precision (typically `f64`)
/// - All operations occur in native backend precision
/// - `to_vec()` returns `Vec<f64>` for host interoperability
///
/// # Zero-cost design
/// - `PhantomData<B>` adds no runtime memory overhead
/// - All operations delegate directly to backend implementations
/// - Implements `Clone` (but not `Copy`) due to potential heap allocation in underlying tensors
///
/// # Example
/// ```
/// use machinelearne_rs::backend::CpuBackend;
/// use machinelearne_rs::backend::{Tensor1D, Scalar};
///
/// // Create tensor from f32 data (converted to f64 internally)
/// let x: Tensor1D<CpuBackend> = Tensor1D::new(vec![1.0f32, 2.0, 3.0]);
/// assert_eq!(x.len(), 3);
///
/// // Element-wise operations
/// let y = x.scale(&Scalar::<CpuBackend>::new(2.0));
/// assert_eq!(y.to_vec(), vec![2.0, 4.0, 6.0]);
/// ```
#[derive(Clone)]
pub struct Tensor1D<B: Backend> {
    pub(crate) data: B::Tensor1D,
    pub(crate) backend: PhantomData<B>,
}

impl<B: Backend> Tensor1D<B> {
    /// Creates a new 1D tensor from a vector of `f32` values.
    ///
    /// Converts the input `Vec<f32>` to the backend's native scalar representation
    /// (typically `f64`) using the backend's `from_vec_1d` conversion function.
    ///
    /// # Precision note
    /// Input values are converted from `f32` to backend precision (usually `f64`).
    /// This conversion is lossless for values representable in both types, but extremely
    /// large `f32` values near `f32::MAX` may lose precision when converted to `f64`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t: Tensor1D<CpuBackend> = Tensor1D::new(vec![1.0f32, 2.5, 3.75]);
    /// assert_eq!(t.to_vec(), vec![1.0, 2.5, 3.75]);
    /// ```
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: B::from_vec_1d(data),
            backend: PhantomData,
        }
    }

    /// Creates a 1D tensor filled with zeros of specified length.
    ///
    /// # Arguments
    /// * `len` - Number of elements in the resulting tensor
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let zeros: Tensor1D<CpuBackend> = Tensor1D::zeros(4);
    /// assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
    /// assert_eq!(zeros.len(), 4);
    /// ```
    pub fn zeros(len: usize) -> Self {
        Self {
            data: B::zeros_1d(len),
            backend: PhantomData,
        }
    }

    /// Computes element-wise subtraction: `self - other`.
    ///
    /// # Panics
    /// Panics if tensors have different lengths (backend-dependent behavior).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let a = Tensor1D::<CpuBackend>::new(vec![5.0f32, 7.0, 9.0]);
    /// let b = Tensor1D::<CpuBackend>::new(vec![2.0f32, 3.0, 4.0]);
    /// let diff = a.sub(&b);
    /// assert_eq!(diff.to_vec(), vec![3.0, 4.0, 5.0]);
    /// ```
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: B::sub_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    /// Computes the arithmetic mean of all elements in the tensor.
    ///
    /// # Returns
    /// A `Scalar<B>` containing the mean value.
    ///
    /// # Panics
    /// Panics if the tensor is empty (backend-dependent behavior).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0]);
    /// let mean = t.mean();
    /// assert!((mean.to_f64() - 2.5).abs() < 1e-12);
    /// ```
    pub fn mean(&self) -> Scalar<B> {
        Scalar {
            data: B::mean_all_1d(&self.data),
            backend: PhantomData,
        }
    }

    /// Converts the tensor to a standard Rust `Vec<f64>` for host interoperability.
    ///
    /// # Use cases
    /// - Debugging and logging
    /// - Serialization to external formats
    /// - Interfacing with non-backend-aware code
    /// - Test assertions
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![1.5f32, -2.5, 3.5]);
    /// let host_vec = t.to_vec();
    /// assert_eq!(host_vec, vec![1.5, -2.5, 3.5]);
    /// ```
    pub fn to_vec(&self) -> Vec<f64> {
        B::to_vec_1d(&self.data)
    }

    /// Computes the dot product (inner product) between two tensors.
    ///
    /// Equivalent to `sum(self * other)` where `*` denotes element-wise multiplication.
    ///
    /// # Formula
    /// `dot(a, b) = Σᵢ aᵢ * bᵢ`
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let a = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
    /// let b = Tensor1D::<CpuBackend>::new(vec![4.0f32, 5.0, 6.0]);
    /// let dot = a.dot(&b);
    /// // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    /// assert_eq!(dot.to_f64(), 32.0);
    /// ```
    pub fn dot(&self, other: &Self) -> Scalar<B> {
        let prod = B::mul_1d(&self.data, &other.data);
        let sum = B::sum_all_1d(&prod);
        Scalar {
            data: sum,
            backend: PhantomData,
        }
    }

    /// Computes element-wise absolute value.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![-1.0f32, 2.0, -3.0]);
    /// let abs_t = t.abs();
    /// assert_eq!(abs_t.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn abs(&self) -> Self {
        Self {
            data: B::abs_1d(&self.data),
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
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![-2.0f32, 0.0, 3.0]);
    /// let sign_t = t.sign();
    /// assert_eq!(sign_t.to_vec(), vec![-1.0, 0.0, 1.0]);
    /// ```
    pub fn sign(&self) -> Self {
        Self {
            data: B::sign_1d(&self.data),
            backend: PhantomData,
        }
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
    /// assert_eq!(t.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        B::len_1d(&self.data)
    }

    /// Returns `true` if the tensor contains no elements.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let empty = Tensor1D::<CpuBackend>::zeros(0);
    /// assert!(empty.is_empty());
    ///
    /// let non_empty = Tensor1D::<CpuBackend>::zeros(1);
    /// assert!(!non_empty.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        B::len_1d(&self.data) == 0
    }

    /// Scales the tensor by multiplying each element by a scalar value.
    ///
    /// Equivalent to element-wise multiplication: `self * scalar`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    /// use machinelearne_rs::backend::Scalar;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
    /// let s = Scalar::<CpuBackend>::new(2.5);
    /// let scaled = t.scale(&s);
    /// assert_eq!(scaled.to_vec(), vec![2.5, 5.0, 7.5]);
    /// ```
    pub fn scale(&self, a: &Scalar<B>) -> Self {
        Self {
            data: B::mul_scalar_1d(&self.data, &a.data),
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
    /// use machinelearne_rs::backend::Tensor1D;
    /// use machinelearne_rs::backend::Scalar;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
    /// let s = Scalar::<CpuBackend>::new(10.0);
    /// let shifted = t.add_scalar(&s);
    /// assert_eq!(shifted.to_vec(), vec![11.0, 12.0, 13.0]);
    /// ```
    pub fn add_scalar(&self, a: &Scalar<B>) -> Self {
        Self {
            data: B::add_scalar_1d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise maximum between two tensors.
    ///
    /// For each index `i`, returns `max(self[i], other[i])`.
    ///
    /// # Panics
    /// Panics if tensors have different lengths.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let a = Tensor1D::<CpuBackend>::new(vec![1.0f32, 5.0, 3.0]);
    /// let b = Tensor1D::<CpuBackend>::new(vec![2.0f32, 4.0, 6.0]);
    /// let max_ab = a.maximum(b);
    /// assert_eq!(max_ab.to_vec(), vec![2.0, 5.0, 6.0]);
    /// ```
    pub fn maximum(&self, other: Self) -> Self {
        Self {
            data: B::maximum_1d(&self.data, &other.data),
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
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![0.0f32, 1.0]);
    /// let exp_t = t.exp();
    /// assert!((exp_t.to_vec()[0] - 1.0).abs() < 1e-12);
    /// assert!((exp_t.to_vec()[1] - std::f64::consts::E).abs() < 1e-12);
    /// ```
    pub fn exp(&self) -> Self {
        Self {
            data: B::exp_1d(&self.data),
            backend: PhantomData,
        }
    }

    /// Computes element-wise natural logarithm: `ln(x)`.
    ///
    /// # Panics
    /// Panics for non-positive values (backend-dependent behavior).
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![1.0f32, std::f64::consts::E as f32]);
    /// let log_t = t.log();
    /// assert!((log_t.to_vec()[0] - 0.0).abs() < 1e-6);
    /// assert!((log_t.to_vec()[1] - 1.0).abs() < 1e-6);
    /// ```
    pub fn log(&self) -> Self {
        Self {
            data: B::log_1d(&self.data),
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
    /// use machinelearne_rs::backend::Tensor1D;
    ///
    /// let t = Tensor1D::<CpuBackend>::new(vec![-100.0f32, 0.0, 100.0]);
    /// let sig = t.sigmoid();
    /// let output = sig.to_vec();
    ///
    /// // Extreme negative → ≈0.0
    /// assert!(output[0] < 1e-10);
    /// // Zero → 0.5 exactly
    /// assert!((output[1] - 0.5).abs() < 1e-12);
    /// // Extreme positive → ≈1.0
    /// assert!(output[2] > 1.0 - 1e-10);
    /// ```
    pub fn sigmoid(&self) -> Self {
        Self {
            data: B::sigmoid_1d(&self.data),
            backend: PhantomData,
        }
    }
}
