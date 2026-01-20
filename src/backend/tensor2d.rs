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

    pub fn shape(&self) -> (usize, usize){
        B::shape(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;


    #[test]
    fn test_tensor2d_matvec_ops() {
        // A = [[1.0, 2.0],
        //      [3.0, 4.0]]
        let a = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let x = Tensor1D::<CpuBackend>::new(vec![1.0f32, 0.0]);

        // A @ x = [1.0, 3.0]
        let ax = a.dot(&x);
        assert_eq!(ax.to_vec(), vec![1.0, 3.0]);

        // A^T @ x — но x длины 2, а A^T имеет 2 строки → OK
        // A^T = [[1, 3],
        //        [2, 4]]
        // A^T @ x = [1*1 + 3*0, 2*1 + 4*0] = [1.0, 2.0]
        let atx = a.tdot(&x);
        assert_eq!(atx.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_tensor2d_reductions() {
        let a = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);

        // mean = (1+2+3+4)/4 = 2.5
        let mean = a.mean();
        assert!((mean.data - 2.5).abs() < 1e-12);

        // len = rows = 2
        let len = a.len();
        assert_eq!(len.data, 2.0);
    }
    #[test]
    fn test_tensor2d_shape() {
        let t = Tensor2D::<CpuBackend>::new(vec![1.0f32, 2.0, 3.0], 3, 1);
        assert_eq!(t.shape(), (3, 1));

        let t2 = Tensor2D::<CpuBackend>::zeros(0, 5);
        assert_eq!(t2.shape(), (0, 5));
    }

    #[test]
    fn test_tensor2d_type_safety() {
        // Следующее не должно компилироваться:
        // let cpu_t: Tensor2D<CpuBackend> = Tensor2D::zeros(2, 2);
        // let ndarray_t: Tensor2D<NdarrayBackend> = Tensor2D::zeros(2, 2);
        // let _ = cpu_t.sub(&ndarray_t); // <-- ошибка типов

        // Проверим только корректность внутри одного бэкенда
        let x = Tensor2D::<CpuBackend>::new(vec![1.0f32], 1, 1);
        let y = Tensor2D::<CpuBackend>::new(vec![2.0f32], 1, 1);
        let _ = x.sub(&y); // должно работать
    }
}