use ndarray::{Array1, Array2, Ix1, Ix2};
use super::{Backend};

#[derive(Clone, Debug, Copy)]
pub struct NdarrayBackend;

#[derive(Debug, Clone)]
pub struct NdarrayTensor2D(pub Array2<f64>);

impl From<&[Vec<f64>]> for NdarrayTensor2D {
    fn from(x: &[Vec<f64>]) -> Self {
        let rows = x.len();
        if rows == 0 {
            return NdarrayTensor2D(Array2::from_shape_vec((0, 0), vec![]).unwrap());
        }
        let cols = x[0].len();
        assert!(x.iter().all(|r| r.len() == cols));
        let data: Vec<f64> = x.iter().flat_map(|r| r.iter()).copied().collect();
        NdarrayTensor2D(Array2::from_shape_vec((rows, cols), data).unwrap())
    }
}

impl super::Backend for NdarrayBackend {
    type Scalar = f64;
    type Tensor1D = Array1<f64>;
    type Tensor2D = NdarrayTensor2D;
    type Device = (); // CPU-only for now

    fn default_device() -> Self::Device { () }

    fn zeros_1d(len: usize) -> Self::Tensor1D {
        Array1::zeros(len)
    }

    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        NdarrayTensor2D(Array2::zeros((rows, cols)))
    }

    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        Array1::from_iter(data.into_iter().map(|x| x as f64))
    }

    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        let data_f64: Vec<f64> = data.into_iter().map(|x| x as f64).collect();
        NdarrayTensor2D(Array2::from_shape_vec((rows, cols), data_f64).unwrap())
    }

    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a + b
    }

    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        NdarrayTensor2D(&t.0 * *s)
    }

    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.0.dot(x) // ndarray has efficient matvec
    }

    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        // X^T @ v = (v^T @ X)^T, but easier: use .t().dot()
        a.0.t().dot(x)
    }

    fn transpose(a: &Self::Tensor2D) -> Self::Tensor2D {
        NdarrayTensor2D(a.0.t().to_owned())
    }

    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        let shape = t.0.shape();
        (shape[0], shape[1])
    }


   // --- Element-wise non-linear ops (1D) ---

fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
    x.mapv(f64::exp)
}

fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
    x.mapv(f64::ln)
}

fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
    // Численно стабильная сигмоида через ndarray
    x.mapv(|z| {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        }
    })
}

fn abs_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
    x.mapv(f64::abs)
}

fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
    x.mapv(|x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

fn maximum_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
    assert_eq!(a.len(), b.len(), "Shapes must match");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.max(y))
        .collect()
}

// --- 2D versions (delegating to 1D via flat view or direct map) ---

fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(x.0.mapv(f64::exp))
}

fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(x.0.mapv(f64::ln))
}

fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(x.0.mapv(|z| {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        }
    }))
}

fn abs_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(x.0.mapv(f64::abs))
}

fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(x.0.mapv(|x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }))
}

fn maximum_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
    let (rows, cols) = a.0.dim();
    assert_eq!(a.0.dim(), b.0.dim(), "Shapes must match");
    let data: Vec<f64> = a.0
        .iter()
        .zip(b.0.iter())
        .map(|(&x, &y)| x.max(y))
        .collect();
    NdarrayTensor2D(Array2::from_shape_vec((rows, cols), data).unwrap())
}

// --- Reductions (already partially covered, but for completeness) ---

fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
    t.sum()
}

fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
    t.mean().unwrap()
}

fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
    t.0.sum()
}

fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
    t.0.mean().unwrap()
}

// --- Scalar and access helpers ---

fn scalar_f64(value: f64) -> Self::Scalar {
    value
}

fn len_1d(t: &Self::Tensor1D) -> usize {
    t.len()
}

fn len_2d(t: &Self::Tensor2D) -> usize {
    t.0.nrows() // или .len() / ncols(), но nrows — прямой аналог твоему rows
}

fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
    t.to_vec()
}

// --- Element-wise binary ops (1D) ---

fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
    a - b
}

fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
    a * b
}

fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
    a / b
}

fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
    t.mapv(|x| x * *s)
}

fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
    t.mapv(|x| x + *s)
}

// --- 2D scalar and binary ops ---

fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
    NdarrayTensor2D(t.0.mapv(|x| x + *s))
}

fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(&a.0 + &b.0)
}

fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(&a.0 - &b.0)
}

fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(&a.0 * &b.0)
}

fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
    NdarrayTensor2D(&a.0 / &b.0)
}

// --- "Unchecked" matvec helpers (для совместимости с CpuBackend) ---
// В ndarray они не нужны, но трейт требует — делаем просто обёртки

fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
    Self::matvec(a, x)
}

fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
    Self::matvec_transposed(a, x)
}
}

#[cfg(test)]
#[cfg(feature = "ndarray")]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_sigmoid_stability() {
        let input = Array1::from_vec(vec![-100.0, 0.0, 100.0]);
        let out = NdarrayBackend::sigmoid_1d(&input);
        let expected = vec![0.0, 0.5, 1.0];
        for (o, e) in out.iter().zip(expected) {
            assert!((o - e).abs() < 1e-6);
        }
    }
}