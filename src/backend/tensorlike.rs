use crate::backend::Scalar;
use std::marker::PhantomData;

use super::{Backend, Tensor1D};

pub trait TensorLike<B: Backend>{
    fn sub(&self, other: &Self) -> Self;
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