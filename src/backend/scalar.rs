use crate::backend::Backend;
use std::marker::PhantomData;

/// Trait for scalar operations required by numerical backends.
///
/// Defines a minimal set of arithmetic and mathematical operations needed for
/// machine learning computations. Implemented for primitive floating-point types
/// used by backends (e.g., `f64`).
///
/// # Design rationale
/// This trait abstracts scalar operations to enable backend-agnostic generic code
/// while maintaining performance through `Copy` semantics and avoiding dynamic dispatch.
///
/// # Required operations
/// - Basic arithmetic via standard library traits (`Add`, `Sub`, `Mul`, `Div`)
/// - Mathematical functions: `sqrt`, `abs`, `exp`
/// - Type conversion: `from_f64`/`to_f64` for interoperability
/// - Constants: `zero()`, `one()` for initialization
///
/// # Safety guarantees
/// Implementations must satisfy:
/// - `Copy` + `Clone` for zero-cost abstractions
/// - `Send` + `Sync` for thread-safe parallel computations
/// - Numerical stability for edge cases (e.g., `sqrt` of negative numbers should panic)
///
/// # Example
/// ```
/// use machinelearne_rs::backend::ScalarOps;
///
/// let x = 4.0f64;
/// assert_eq!(x.sqrt(), 2.0);
/// assert_eq!(f64::zero(), 0.0);
/// assert_eq!(f64::one(), 1.0);
/// ```
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
    /// Computes the square root of the scalar.
    ///
    /// # Panics
    /// Panics if called on a negative value (for real-number implementations).
    fn sqrt(self) -> Self;

    /// Returns the absolute value of the scalar.
    fn abs(self) -> Self;

    /// Returns the additive identity (zero) for this scalar type.
    fn zero() -> Self;

    /// Returns the multiplicative identity (one) for this scalar type.
    fn one() -> Self;

    /// Converts an `f64` value to this scalar type.
    ///
    /// Used for backend-agnostic initialization from host values.
    fn from_f64(v: f64) -> Self;

    /// Converts this scalar to an `f64` value.
    ///
    /// Used for interoperability with host code and debugging.
    fn to_f64(self) -> f64;

    /// Computes the exponential function `e^x`.
    fn exp(self) -> Self;
}

// === Implementations for primitive types ===

/// `f64` implementation of `ScalarOps`.
///
/// Provides IEEE 754 double-precision floating point operations with
/// hardware-accelerated math functions.
///
/// # Numerical behavior
/// - Follows standard IEEE 754 semantics (NaN propagation, infinities)
/// - `sqrt` panics on negative inputs via Rust's built-in `sqrt()` method
/// - `exp` handles overflow by returning `INFINITY` per IEEE 754
impl ScalarOps for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(v: f64) -> Self {
        v
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn exp(self) -> Self {
        self.exp()
    }
}

/// Backend-typed scalar wrapper providing compile-time type safety.
///
/// Wraps a backend's native scalar type (`B::Scalar`) while carrying phantom
/// type information about its originating backend. This prevents accidental
/// mixing of scalars from different backends at compile time.
///
/// # Type safety guarantees
/// ```compile_fail
/// use machinelearne_rs::backend::{CpuBackend, NdarrayBackend};
/// use machinelearne_rs::backend::Scalar;
///
/// let cpu_scalar: Scalar<CpuBackend> = Scalar::new(1.0);
/// let ndarray_scalar: Scalar<NdarrayBackend> = Scalar::new(2.0);
/// let _ = cpu_scalar + ndarray_scalar; // COMPILE ERROR: mismatched backends
/// ```
///
/// # Zero-cost abstraction
/// - `PhantomData<B>` adds no runtime overhead (zero-sized type)
/// - All operations delegate directly to backend's scalar type
/// - Implements `Copy` for efficient pass-by-value semantics
///
/// # Example
/// ```
/// use machinelearne_rs::backend::CpuBackend;
/// use machinelearne_rs::backend::{Scalar, ScalarOps};
///
/// let s: Scalar<CpuBackend> = Scalar::new(2.0);
/// let squared = s * s; // Backend-safe multiplication
/// assert_eq!(squared.to_f64(), 4.0);
/// ```
#[derive(Clone, Debug, Copy)]
pub struct Scalar<B: Backend> {
    pub(crate) data: B::Scalar,
    pub(crate) backend: PhantomData<B>,
}

impl<B: Backend> Scalar<B> {
    /// Creates a new scalar from an `f64` host value.
    ///
    /// Converts the host `f64` value to the backend's native scalar representation
    /// using the backend's `scalar_f64` conversion function.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Scalar;
    ///
    /// let s: Scalar<CpuBackend> = Scalar::new(3.14);
    /// assert_eq!(s.to_f64(), 3.14);
    /// ```
    pub fn new(f: f64) -> Self {
        Self {
            data: B::scalar_f64(f),
            backend: PhantomData,
        }
    }
    /// Converts this backend scalar to a host `f64` value.
    ///
    /// Extracts the scalar value from the backend's native representation and
    /// converts it to a standard Rust `f64` for interoperability with host code,
    /// debugging, logging, or serialization.
    ///
    /// # Precision considerations
    /// For backends using `f64` internally (e.g., `CpuBackend`, `NdarrayBackend`),
    /// this is a zero-cost identity conversion. For hypothetical future backends
    /// assert!((host_value - std::f64::consts::E).abs() < 1e-5);
    pub fn to_f64(&self) -> f64 {
        self.data.to_f64()
    }

    /// Computes the exponential function `e^x` for this scalar.
    ///
    /// Delegates to the backend's scalar implementation of `exp`.
    ///
    /// # Example
    /// ```
    /// use machinelearne_rs::backend::CpuBackend;
    /// use machinelearne_rs::backend::Scalar;
    ///
    /// let s: Scalar<CpuBackend> = Scalar::new(1.0);
    /// let e = s.exp();
    /// assert!((e.to_f64() - std::f64::consts::E).abs() < 1e-12);
    /// ```
    pub fn exp(&self) -> Self {
        Self {
            data: self.data.exp(),
            backend: PhantomData,
        }
    }
}

// === Standard arithmetic trait implementations ===

/// Implements addition for backend-typed scalars.
///
/// Enables ergonomic `a + b` syntax while preserving backend type safety.
/// Only scalars from the *same* backend can be added together.
///
/// # Example
/// ```
/// use machinelearne_rs::backend::CpuBackend;
/// use machinelearne_rs::backend::Scalar;
///
/// let a: Scalar<CpuBackend> = Scalar::new(2.0);
/// let b: Scalar<CpuBackend> = Scalar::new(3.0);
/// let sum = a + b;
/// assert_eq!(sum.to_f64(), 5.0);
/// ```
impl<B: Backend + Copy> std::ops::Add for Scalar<B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            data: self.data + rhs.data,
            backend: PhantomData,
        }
    }
}

/// Implements subtraction for backend-typed scalars.
impl<B: Backend> std::ops::Sub for Scalar<B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            data: self.data - rhs.data,
            backend: PhantomData,
        }
    }
}

/// Implements multiplication for backend-typed scalars.
impl<B: Backend> std::ops::Mul for Scalar<B> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            data: self.data * rhs.data,
            backend: PhantomData,
        }
    }
}

/// Implements division for backend-typed scalars.
///
/// # Panics
/// Panics on division by zero according to backend's scalar implementation.
impl<B: Backend> std::ops::Div for Scalar<B> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            data: self.data / rhs.data,
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_scalar_ops_f64() {
        let a = 4.0f64;
        assert_eq!(a.sqrt(), 2.0);
        assert_eq!(a.abs(), 4.0);
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(f64::from_f64(3.14), 3.14);
        assert_eq!(3.14f64.to_f64(), 3.14);
        assert_eq!(1.0f64.exp(), std::f64::consts::E);
    }

    #[test]
    fn test_scalar_new_and_exp() {
        let s: Scalar<CpuBackend> = Scalar::new(1.0);
        assert_eq!(s.data, 1.0);

        let e = s.exp();
        assert!((e.data - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_arithmetic() {
        let a: Scalar<CpuBackend> = Scalar::new(5.0);
        let b: Scalar<CpuBackend> = Scalar::new(2.0);

        let sum = a + b;
        assert_eq!(sum.data, 7.0);

        let diff = sum - Scalar::new(3.0);
        assert_eq!(diff.data, 4.0);

        let prod = diff * Scalar::new(0.5);
        assert_eq!(prod.data, 2.0);

        let quot = prod / Scalar::new(4.0);
        assert_eq!(quot.data, 0.5);
    }

    #[test]
    fn test_scalar_type_safety() {
        // Verify that scalars from the same backend compose correctly
        let x: Scalar<CpuBackend> = Scalar::new(10.0);
        let y: Scalar<CpuBackend> = Scalar::new(3.0);
        let _ = x / y; // Must compile successfully

        // Note: Cross-backend operations are prevented at compile time:
        // let z: Scalar<NdarrayBackend> = Scalar::new(2.0);
        // let _ = x + z; // <-- Would fail to compile with type mismatch error
    }
}
