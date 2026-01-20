use crate::backend::scalar::Scalar;
use std::marker::PhantomData;

use super::{Backend};
use super::tensor1d::Tensor1D;

pub trait TensorLike<B: Backend>{
    fn sub(&self, other: &Self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn mean_all(&self) -> Scalar<B>;      // агрегация до скаляра
    fn sum(&self) -> Scalar<B>;
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

    fn scale(&self, other: Scalar<B>) -> Self{
        Self {
            data: B::mul_scalar_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    fn sum(&self) -> Scalar<B>{
        Scalar {
            data: B::sum_all_1d(&self.data),
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::backend::tensor1d::Tensor1D;

    #[test]
    fn test_tensorlike_for_tensor1d() {
        let a: Tensor1D<CpuBackend> = Tensor1D::new(vec![4.0f32, 6.0]);
        let b: Tensor1D<CpuBackend> = Tensor1D::new(vec![1.0f32, 2.0]);

        // sub
        let diff = a.sub(&b);
        assert_eq!(diff.to_vec(), vec![3.0, 4.0]);

        // add
        let sum = a.add(&b);
        assert_eq!(sum.to_vec(), vec![5.0, 8.0]);

        // mul
        let prod = a.mul(&b);
        assert_eq!(prod.to_vec(), vec![4.0, 12.0]);

        // div
        let quot = a.div(&b);
        assert_eq!(quot.to_vec(), vec![4.0, 3.0]);

        // mean_all
        let mean = a.mean_all();
        assert!((mean.data - 5.0).abs() < 1e-12);

        // sum
        let total = a.sum();
        assert_eq!(total.data, 10.0);

        // scale
        let s = Scalar::<CpuBackend>::new(0.5);
        let scaled = a.scale(&s);
        assert_eq!(scaled.to_vec(), vec![2.0, 3.0]);
    }

    // Дополнительно: проверим, что generic-функция работает
    #[test]
    fn test_generic_function_over_tensorlike() {
        fn mean_squared_error<T: TensorLike<CpuBackend>>(pred: &T, target: &T) -> Scalar<CpuBackend> {
            let diff = pred.sub(target);
            let sq = diff.mul(&diff);
            sq.mean_all()
        }

        let pred = Tensor1D::<CpuBackend>::new(vec![3.0f32, 5.0]);
        let target = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0]);

        let mse = mean_squared_error(&pred, &target);
        // ((3-1)^2 + (5-2)^2) / 2 = (4 + 9) / 2 = 6.5
        assert!((mse.data - 6.5).abs() < 1e-12);
    }
}