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
}