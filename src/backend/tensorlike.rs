use crate::backend::scalar::Scalar;
use std::marker::PhantomData;

use super::tensor1d::Tensor1D;
use super::Backend;

/// Trait for tensor-like structures supporting element-wise arithmetic operations.
///
/// Provides a unified interface for performing element-wise operations, aggregations,
/// and scalar transformations over tensor implementations.
///
/// # Design Rationale
/// - All binary operations (`sub`, `add`, `mul`, `div`) are performed **element-wise**.
/// - Aggregation methods return scalar values wrapped in [`Scalar<B>`].
/// - Backend-generic: parameterized over `B: Backend` to abstract away concrete
///   computation backends (CPU, GPU, etc.).
/// - Enables generic algorithms: users can write functions that work with any tensor
///   dimensionality implementing this trait (e.g., `Tensor1D`, `Tensor2D`).
///
/// # Example
/// ```rust
/// # use machinelearne_rs::backend::{Scalar, Backend, tensorlike::TensorLike};
/// fn mean_squared_error<T: TensorLike<B>, B: Backend>(
///     pred: &T,
///     target: &T,
/// ) -> Scalar<B> {
///     let diff = pred.sub(target);
///     let sq = diff.mul(&diff);
///     sq.mean_all()  // Returns mean of squared errors
/// }
/// ```
pub trait TensorLike<B: Backend> {
    /// Element-wise subtraction: `self - other`.
    fn sub(&self, other: &Self) -> Self;

    /// Element-wise addition: `self + other`.
    fn add(&self, other: &Self) -> Self;

    /// Element-wise multiplication: `self * other`.
    fn mul(&self, other: &Self) -> Self;

    /// Element-wise division: `self / other`.
    fn div(&self, other: &Self) -> Self;

    /// Computes the arithmetic mean across all elements of the tensor.
    ///
    /// Returns a scalar value:
    /// ```text
    /// mean = (Σ x_i) / N
    /// ```
    fn mean_all(&self) -> Scalar<B>;

    /// Sums all elements of the tensor.
    ///
    /// Returns a scalar value:
    /// ```text
    /// sum = Σ x_i
    /// ```
    fn sum(&self) -> Scalar<B>;

    /// Scales all elements of the tensor by a scalar value.
    ///
    /// Equivalent to element-wise multiplication: `self * scalar`.
    fn scale(&self, other: Scalar<B>) -> Self;
}

impl<B: Backend> TensorLike<B> for Tensor1D<B> {
    fn sub(&self, other: &Self) -> Self {
        Self {
            data: B::sub_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            data: B::add_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            data: B::mul_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    fn div(&self, other: &Self) -> Self {
        Self {
            data: B::div_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    fn mean_all(&self) -> Scalar<B> {
        Scalar {
            data: B::mean_all_1d(&self.data),
            backend: PhantomData,
        }
    }

    fn scale(&self, other: Scalar<B>) -> Self {
        Self {
            data: B::mul_scalar_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    fn sum(&self) -> Scalar<B> {
        Scalar {
            data: B::sum_all_1d(&self.data),
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::tensor1d::Tensor1D;
    use crate::backend::CpuBackend;

    /// Tests basic `TensorLike` operations for one-dimensional tensors.
    ///
    /// Verifies correctness of:
    /// - Element-wise arithmetic operations (sub, add, mul, div)
    /// - Aggregation methods (mean_all, sum)
    /// - Scalar multiplication (scale)
    #[test]
    fn test_tensorlike_for_tensor1d() {
        let a: Tensor1D<CpuBackend> = Tensor1D::new(vec![4.0f32, 6.0]);
        let b: Tensor1D<CpuBackend> = Tensor1D::new(vec![1.0f32, 2.0]);

        // sub: [4-1, 6-2] = [3, 4]
        let diff = a.sub(&b);
        assert_eq!(diff.to_vec(), vec![3.0, 4.0]);

        // add: [4+1, 6+2] = [5, 8]
        let sum = a.add(&b);
        assert_eq!(sum.to_vec(), vec![5.0, 8.0]);

        // mul: [4*1, 6*2] = [4, 12]
        let prod = a.mul(&b);
        assert_eq!(prod.to_vec(), vec![4.0, 12.0]);

        // div: [4/1, 6/2] = [4, 3]
        let quot = a.div(&b);
        assert_eq!(quot.to_vec(), vec![4.0, 3.0]);

        // mean_all: (4 + 6) / 2 = 5.0
        let mean = a.mean_all();
        assert!((mean.data - 5.0).abs() < 1e-12);

        // sum: 4 + 6 = 10.0
        let total = a.sum();
        assert_eq!(total.data, 10.0);

        // scale: [4*0.5, 6*0.5] = [2, 3]
        let s = Scalar::<CpuBackend>::new(0.5);
        let scaled = a.scale(&s);
        assert_eq!(scaled.to_vec(), vec![2.0, 3.0]);
    }

    /// Tests a generic function operating on any type implementing `TensorLike`.
    ///
    /// Demonstrates the key advantage of the trait: writing algorithms agnostic
    /// to tensor dimensionality or concrete backend implementation.
    #[test]
    fn test_generic_function_over_tensorlike() {
        fn mean_squared_error<T: TensorLike<CpuBackend>>(
            pred: &T,
            target: &T,
        ) -> Scalar<CpuBackend> {
            let diff = pred.sub(target);
            let sq = diff.mul(&diff);
            sq.mean_all()
        }

        let pred = Tensor1D::<CpuBackend>::new(vec![3.0f32, 5.0]);
        let target = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0]);

        // MSE = ((3-1)² + (5-2)²) / 2 = (4 + 9) / 2 = 6.5
        let mse = mean_squared_error(&pred, &target);
        assert!((mse.data - 6.5).abs() < 1e-12);
    }
}
