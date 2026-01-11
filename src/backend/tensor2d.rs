use crate::backend::Backend;
use std::marker::PhantomData;
use super::scalar::Scalar;
use super::tensor1d::Tensor1D;

#[derive(Clone)]
pub struct Tensor2D<B: Backend> {
    pub(crate) data: B::Tensor2D,
    pub(crate) backend: PhantomData<B>,
}

impl<B: Backend> Tensor2D<B> {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self {
            data: B::from_vec_2d(data, rows, cols),
            backend: PhantomData,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: B::zeros_2d(rows, cols),
            backend: PhantomData,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: B::sub_2d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    pub fn mean(&self) -> Scalar<B> {
        Scalar {
            data: B::mean_all_2d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn dot(&self, other: &Tensor1D<B>) -> Tensor1D<B> {
        Tensor1D {
            data: B::matvec(&self.data, &other.data),
            backend: PhantomData,
        }
    
    }

    pub fn tdot(&self, other: &Tensor1D<B>) -> Tensor1D<B> {
        Tensor1D {
            data: B::matvec_transposed(&self.data, &other.data),
            backend: PhantomData,
        }
    
    }

    pub fn abs(&self) -> Self{
        Self {
            data: B::abs_2d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sign(&self) -> Self{
        Self {
            data: B::sign_2d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn len(&self) -> Scalar<B>{
        Scalar {
            data: B::scalar_f64(B::len_2d(&self.data) as f64),
            backend: PhantomData,
        }
    }

    pub fn scale(&self, a: Scalar<B>) -> Self{
        Self {
            data: B::mul_scalar_2d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    pub fn add_scalar(&self, a: Scalar<B>) -> Self{
        Self {
            data: B::add_scalar_2d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    pub fn maximum(&self, other: Self) -> Self{
        Self {
            data: B::maximum_2d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    pub fn exp(&self) -> Self{
        Self {
            data: B::exp_2d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn log(&self) -> Self{
        Self {
            data: B::log_2d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sigmoid(&self) -> Self{
        Self {
            data: B::sigmoid_2d(&self.data),
            backend: PhantomData,
        }
    }
}