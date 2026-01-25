use super::scalar::Scalar;
use crate::backend::Backend;
use std::marker::PhantomData;

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
        let prod = B::mul_1d(&self.data, &other.data);
        let sum = B::sum_all_1d(&prod);
        Scalar {
            data: sum,
            backend: PhantomData,
        }
    }

    pub fn abs(&self) -> Self {
        Self {
            data: B::abs_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sign(&self) -> Self {
        Self {
            data: B::sign_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        B::len_1d(&self.data)
    }

    pub fn scale(&self, a: &Scalar<B>) -> Self {
        Self {
            data: B::mul_scalar_1d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    pub fn add_scalar(&self, a: &Scalar<B>) -> Self {
        Self {
            data: B::add_scalar_1d(&self.data, &a.data),
            backend: PhantomData,
        }
    }

    pub fn maximum(&self, other: Self) -> Self {
        Self {
            data: B::maximum_1d(&self.data, &other.data),
            backend: PhantomData,
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            data: B::exp_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn log(&self) -> Self {
        Self {
            data: B::log_1d(&self.data),
            backend: PhantomData,
        }
    }

    pub fn sigmoid(&self) -> Self {
        Self {
            data: B::sigmoid_1d(&self.data),
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend; // Убедись, что CpuBackend импортируется

    #[test]
    fn test_tensor1d_constructors() {
        // new from f32 vec
        let t = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);

        // zeros
        let z = Tensor1D::<CpuBackend>::zeros(3);
        assert_eq!(z.to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tensor1d_elementwise_ops() {
        let a = Tensor1D::<CpuBackend>::new(vec![1.0f32, -2.0, 3.0]);
        let b = Tensor1D::<CpuBackend>::new(vec![4.0f32, 5.0, 6.0]);

        // sub
        let diff = a.sub(&b);
        assert_eq!(diff.to_vec(), vec![-3.0, -7.0, -3.0]);

        // abs
        let abs_a = a.abs();
        assert_eq!(abs_a.to_vec(), vec![1.0, 2.0, 3.0]);

        // sign
        let sign_a = a.sign();
        assert_eq!(sign_a.to_vec(), vec![1.0, -1.0, 1.0]);

        // maximum
        let max_ab = a.maximum(b);
        assert_eq!(max_ab.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor1d_reductions() {
        let a = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);

        // mean
        let mean = a.mean();
        assert!((mean.data - 2.0).abs() < 1e-12);

        // dot
        let b = Tensor1D::<CpuBackend>::new(vec![4.0f32, 5.0, 6.0]);
        let dot = a.dot(&b);
        assert_eq!(dot.data, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0); // 32.0

        // len
        let len = a.len();
        assert_eq!(len, 3);
    }

    #[test]
    fn test_tensor1d_scalar_ops() {
        let a = Tensor1D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0]);
        let s = Scalar::<CpuBackend>::new(2.0);

        // scale
        let scaled = a.scale(&s);
        assert_eq!(scaled.to_vec(), vec![2.0, 4.0, 6.0]);

        // add_scalar
        let shifted = a.add_scalar(&s);
        assert_eq!(shifted.to_vec(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_tensor1d_math_functions() {
        let a = Tensor1D::<CpuBackend>::new(vec![0.0f32, 1.0]);

        // exp
        let exp_a = a.exp();
        assert_eq!(exp_a.to_vec(), vec![1.0, std::f64::consts::E]);

        // log
        let log_exp = exp_a.log();
        assert!((log_exp.to_vec()[0] - 0.0).abs() < 1e-12);
        assert!((log_exp.to_vec()[1] - 1.0).abs() < 1e-12);

        // sigmoid
        let sig = a.sigmoid();
        let expected = vec![0.5, 1.0 / (1.0 + (-1.0f64).exp())];
        for (got, exp) in sig.to_vec().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }
    }

    #[test]
    fn test_tensor1d_type_safety() {
        // Следующее не должно компилироваться:
        // let cpu_t: Tensor1D<CpuBackend> = Tensor1D::zeros(3);
        // let ndarray_t: Tensor1D<NdarrayBackend> = Tensor1D::zeros(3);
        // let _ = cpu_t.sub(&ndarray_t); // <-- ошибка типов

        // Проверим только корректность внутри одного бэкенда
        let x = Tensor1D::<CpuBackend>::new(vec![1.0f32]);
        let y = Tensor1D::<CpuBackend>::new(vec![2.0f32]);
        let _ = x.sub(&y); // должно работать
    }
}
