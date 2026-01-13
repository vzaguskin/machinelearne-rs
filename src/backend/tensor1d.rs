use crate::backend::Backend;
use std::marker::PhantomData;
use super::scalar::Scalar;

#[derive(Clone)]
pub struct Tensor1D<B: Backend> {
    pub(crate) data: B::Tensor1D,
    pub(crate) backend: PhantomData<B>,
}

impl<B: Backend> Tensor1D<B> {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: B::from_vec_1d(data),
            backend: PhantomData,
        }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            data: B::zeros_1d(len),
            backend: PhantomData,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: B::sub_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    pub fn mean(&self) -> Scalar<B> {
        Scalar {
            data: B::mean_all_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        B::to_vec_1d(&self.data)
    }

    pub fn dot(&self, other: &Self) -> Scalar<B> {
        let prod = B::mul_1d(&self.data,&other.data);
        let sum = B::sum_all_1d(&prod);
        Scalar {
            data: sum,
            backend: PhantomData,
        }
    
    }

    pub fn abs(&self) -> Self{
        Self {
            data: B::abs_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sign(&self) -> Self{
        Self {
            data: B::sign_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn len(&self) -> Scalar<B>{
        Scalar {
            data: B::scalar_f64(B::len_1d(&self.data) as f64),
            backend: PhantomData,
        }
    }

    pub fn scale(&self, a: &Scalar<B>) -> Self{
        Self {
            data: B::mul_scalar_1d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    pub fn add_scalar(&self, a: &Scalar<B>) -> Self{
        Self {
            data: B::add_scalar_1d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    pub fn maximum(&self, other: Self) -> Self{
        Self {
            data: B::maximum_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    pub fn exp(&self) -> Self{
        Self {
            data: B::exp_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn log(&self) -> Self{
        Self {
            data: B::log_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sigmoid(&self) -> Self{
        Self {
            data: B::sigmoid_1d(&self.data),
            backend: PhantomData,
        }
    }
}