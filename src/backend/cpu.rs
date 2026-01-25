use super::Backend;
#[derive(Clone, Debug, Copy)]
pub struct CpuBackend;

// В cpu_backend.rs или аналогичном
#[derive(Debug, Clone)]
pub struct CpuTensor2D(pub Vec<f64>, pub usize, pub usize);

impl CpuTensor2D {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "Inconsistent shape");
        Self(data, rows, cols)
    }
}

impl From<&[Vec<f64>]> for CpuTensor2D {
    fn from(x: &[Vec<f64>]) -> Self {
        if x.is_empty() {
            return CpuTensor2D::new(Vec::new(), 0, 0);
        }
        let rows = x.len();
        let cols = x[0].len();
        // Проверка одинаковой длины строк (опционально, но рекомендуется)
        assert!(
            x.iter().all(|row| row.len() == cols),
            "All rows must have same length"
        );
        let data: Vec<f64> = x.iter().flat_map(|row| row.iter()).copied().collect();
        CpuTensor2D::new(data, rows, cols)
    }
}

impl Backend for CpuBackend {
    type Scalar = f64;
    type Tensor1D = Vec<f64>;
    type Tensor2D = CpuTensor2D; // (data, rows, cols)
    type Device = ();

    fn default_device() -> Self::Device {}

    // --- Constructors ---
    fn zeros_1d(len: usize) -> Self::Tensor1D {
        vec![0.; len]
    }
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        CpuTensor2D::new(vec![0.; rows * cols], rows, cols)
    }
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        data.into_iter().map(|x| x as f64).collect()
    }
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        CpuTensor2D::new(data.into_iter().map(|x| x as f64).collect(), rows, cols)
    }

    // --- Element-wise ops ---
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
    }
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a - b).collect()
    }
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
    }
    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        a.iter().zip(b.iter()).map(|(a, b)| a / b).collect()
    }
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t.iter().map(|x| *x * s).collect()
    }

    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        t.iter().map(|x| x + s).collect()
    }

    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        CpuTensor2D::new(t.0.iter().map(|x| *x * s).collect(), t.1, t.2)
    }
    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        CpuTensor2D::new(t.0.iter().map(|x| *x + s).collect(), t.1, t.2)
    }

    // --- Reductions ---
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.iter().sum::<f64>() / t.len() as f64
    }
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.iter().sum::<f64>()
    }

    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.iter().sum::<f64>()
    }

    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.0.iter().sum::<f64>() / t.0.len() as f64
    }
    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a + b).collect(),
            a.1,
            a.2,
        )
    }
    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a - b).collect(),
            a.1,
            a.2,
        )
    }
    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a * b).collect(),
            a.1,
            a.2,
        )
    }
    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            a.0.iter().zip(b.0.iter()).map(|(a, b)| a / b).collect(),
            a.1,
            a.2,
        )
    }

    // --- Scalar ops (for loss gradients, lr updates) ---
    fn scalar_f64(value: f64) -> Self::Scalar {
        value
    }

    // --- Access (for metrics / debug) ---
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
        t.clone()
    }

    fn len_1d(t: &Self::Tensor1D) -> usize {
        t.len()
    }

    fn len_2d(t: &Self::Tensor2D) -> usize {
        t.1
    }

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        t.iter().map(|x| x.abs()).collect()
    }

    fn abs_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(t.0.iter().map(|x| x.abs()).collect(), t.1, t.2)
    }

    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        x.iter()
            .map(|&x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0 // субградиент в нуле — стандартный выбор
                }
            })
            .collect()
    }

    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(
            x.0.iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0 // субградиент в нуле — стандартный выбор
                    }
                })
                .collect(),
            x.1,
            x.2,
        )
    }

    fn maximum_1d(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        a.iter().zip(b).map(|(&x, &y)| x.max(y)).collect()
    }

    fn exp_1d(x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| v.exp()).collect()
    }

    fn log_1d(x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| v.ln()).collect()
    }

    fn sigmoid_1d(x: &Vec<f64>) -> Vec<f64> {
        // Численно стабильная сигмоида:
        // σ(z) = 1 / (1 + exp(-z))     if z >= 0
        // σ(z) = exp(z) / (1 + exp(z)) if z < 0
        x.iter()
            .map(|&z| {
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                }
            })
            .collect()
    }

    fn maximum_2d(x: &Self::Tensor2D, other: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::maximum_1d(&x.0, &other.0), x.1, x.2)
    }
    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::exp_1d(&x.0), x.1, x.2)
    }
    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::log_1d(&x.0), x.1, x.2)
    }
    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        CpuTensor2D::new(Self::sigmoid_1d(&x.0), x.1, x.2)
    }
    fn _matvec_unchecked(a: &CpuTensor2D, x: &Vec<f64>) -> Vec<f64> {
        let CpuTensor2D(data, rows, cols) = a;
        let mut result = Vec::with_capacity(*rows);
        for i in 0..*rows {
            let mut sum = 0.0;
            for j in 0..*cols {
                sum += data[i * *cols + j] * x[j];
            }
            result.push(sum);
        }
        result
    }

    fn matvec(a: &CpuTensor2D, x: &Vec<f64>) -> Vec<f64> {
        //TODO: check
        Self::_matvec_unchecked(a, x)
    }

    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        //todo - efficient implementation
        Self::_matvec_unchecked(&Self::transpose(a), x)
    }

    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        Self::_matvec_transposed_unchecked(a, x)
    }

    fn transpose(a: &CpuTensor2D) -> Self::Tensor2D {
        //TODO: return view slice
        let CpuTensor2D(inp, rows, cols) = a;
        let mut out = Vec::with_capacity(cols * rows);
        for col in 0..*cols {
            for row in 0..*rows {
                out.push(inp[row * cols + col]);
            }
        }

        CpuTensor2D::new(out, *cols, *rows)
    }

    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        (t.1, t.2)
    }
}

#[cfg(test)]
mod matvec_tests {
    use super::*;

    #[test]
    fn test_matvec_transpose() {
        // Пример 1: X — (3, 2), v — (3,)
        // X = [[1.0, 2.0],
        //      [3.0, 4.0],
        //      [5.0, 6.0]]
        // v = [1.0, 0.0, 2.0]
        // Xᵀ @ v = [1*1 + 3*0 + 5*2, 2*1 + 4*0 + 6*2] = [11.0, 14.0]

        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // row-major
        let x = CpuTensor2D::new(x_data, 3, 2); // (rows=3, cols=2)
        let v = vec![1.0, 0.0, 2.0];

        let result = CpuBackend::matvec_transposed(&x, &v);
        let expected = vec![11.0, 14.0];

        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "Expected {}, got {}", e, r);
        }

        // Пример 2: (4, 1) — вектор-столбец (как в твоих линейных регрессиях)
        // X = [[2.0],
        //      [3.0],
        //      [4.0],
        //      [5.0]]
        // v = [1.0, 1.0, 1.0, 1.0]
        // Xᵀ @ v = [2+3+4+5] = [14.0]

        let x2 = CpuTensor2D::new(vec![2.0, 3.0, 4.0, 5.0], 4, 1);
        let v2 = vec![1.0, 1.0, 1.0, 1.0];
        let result2 = CpuBackend::matvec_transposed(&x2, &v2);
        let expected2 = vec![14.0];

        assert_eq!(result2, expected2);

        // Пример 3: (2, 3) → output len = 3
        // X = [[1, 0, 0],
        //      [0, 1, 0]]
        // v = [5.0, 7.0]
        // Xᵀ @ v = [5, 7, 0]

        let x3 = CpuTensor2D::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let v3 = vec![5.0, 7.0];
        let result3 = CpuBackend::matvec_transposed(&x3, &v3);
        let expected3 = vec![5.0, 7.0, 0.0];

        assert_eq!(result3.len(), 3);
        for (r, e) in result3.iter().zip(expected3.iter()) {
            assert!((r - e).abs() < 1e-12, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_matvec_transpose_consistency_with_transpose_and_matvec() {
        let x = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // (3,2)
        let v = vec![1.0, 0.0, 2.0];

        let result1 = CpuBackend::matvec_transposed(&x, &v);
        let x_t = CpuBackend::transpose(&x);
        let result2 = CpuBackend::matvec(&x_t, &v);

        assert_eq!(result1.len(), result2.len());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructors() {
        // zeros_1d
        let z1 = CpuBackend::zeros_1d(3);
        assert_eq!(z1, vec![0.0, 0.0, 0.0]);

        // zeros_2d
        let z2 = CpuBackend::zeros_2d(2, 3);
        assert_eq!(z2.0, vec![0.0; 6]);
        assert_eq!((z2.1, z2.2), (2, 3));

        // from_vec_1d
        let v1 = CpuBackend::from_vec_1d(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(v1, vec![1.0, 2.0, 3.0]);

        // from_vec_2d
        let v2 = CpuBackend::from_vec_2d(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(v2.0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!((v2.1, v2.2), (2, 2));

        // From<&[Vec<f64>]> for CpuTensor2D
        let nested = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let t = CpuTensor2D::from(&nested[..]);
        assert_eq!(t.0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!((t.1, t.2), (2, 2));
    }

    #[test]
    fn test_elementwise_ops_1d() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        assert_eq!(CpuBackend::add_1d(&a, &b), vec![4.0, 6.0]);
        assert_eq!(CpuBackend::sub_1d(&a, &b), vec![-2.0, -2.0]);
        assert_eq!(CpuBackend::mul_1d(&a, &b), vec![3.0, 8.0]);
        assert_eq!(CpuBackend::div_1d(&a, &b), vec![1.0 / 3.0, 0.5]);

        assert_eq!(CpuBackend::add_scalar_1d(&a, &5.0), vec![6.0, 7.0]);
        assert_eq!(CpuBackend::mul_scalar_1d(&a, &2.0), vec![2.0, 4.0]);
    }

    #[test]
    fn test_elementwise_ops_2d() {
        let a = CpuTensor2D::new(vec![1.0, 2.0], 1, 2);
        let b = CpuTensor2D::new(vec![3.0, 4.0], 1, 2);

        let add = CpuBackend::add_2d(&a, &b);
        assert_eq!(add.0, vec![4.0, 6.0]);

        let mul_s = CpuBackend::mul_scalar_2d(&a, &2.0);
        assert_eq!(mul_s.0, vec![2.0, 4.0]);
    }

    #[test]
    fn test_reductions() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(CpuBackend::sum_all_1d(&v), 6.0);
        assert!((CpuBackend::mean_all_1d(&v) - 2.0).abs() < 1e-12);

        let m = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(CpuBackend::sum_all_2d(&m), 10.0);
        assert!((CpuBackend::mean_all_2d(&m) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_abs_and_sign() {
        let v = vec![-2.0, 0.0, 3.0];
        assert_eq!(CpuBackend::abs_1d(&v), vec![2.0, 0.0, 3.0]);
        assert_eq!(CpuBackend::sign_1d(&v), vec![-1.0, 0.0, 1.0]);

        let m = CpuTensor2D::new(vec![-1.0, 0.0, 2.0], 1, 3);
        let sign_m = CpuBackend::sign_2d(&m);
        assert_eq!(sign_m.0, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_math_functions() {
        let v = vec![0.0, 1.0];
        assert_eq!(CpuBackend::exp_1d(&v), vec![1.0, std::f64::consts::E]);
        assert_eq!(
            CpuBackend::log_1d(&vec![1.0, std::f64::consts::E]),
            vec![0.0, 1.0]
        );

        // sigmoid(0) = 0.5
        let sig = CpuBackend::sigmoid_1d(&vec![0.0]);
        assert!((sig[0] - 0.5).abs() < 1e-12);

        // maximum
        let a = vec![1.0, 3.0];
        let b = vec![2.0, 2.0];
        assert_eq!(CpuBackend::maximum_1d(&a, &b), vec![2.0, 3.0]);
    }

    #[test]
    fn test_transpose() {
        let m = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3); // 2x3
        let t = CpuBackend::transpose(&m); // 3x2
        assert_eq!(t.0, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!((t.1, t.2), (3, 2));

        // Double transpose = original
        let tt = CpuBackend::transpose(&t);
        assert_eq!(tt.0, m.0);
        assert_eq!((tt.1, tt.2), (2, 3));
    }

    #[test]
    fn test_matvec() {
        let m = CpuTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let v = vec![1.0, 0.0];
        let res = CpuBackend::matvec(&m, &v);
        assert_eq!(res, vec![1.0, 3.0]); // [1*1 + 2*0, 3*1 + 4*0]
    }

    #[test]
    fn test_edge_cases() {
        // Пустой тензор 1D
        let empty1d = CpuBackend::zeros_1d(0);
        assert_eq!(empty1d.len(), 0);
        assert_eq!(CpuBackend::sum_all_1d(&empty1d), 0.0);

        // Пустой тензор 2D
        let empty2d = CpuBackend::zeros_2d(0, 0);
        assert_eq!(empty2d.0.len(), 0);
        assert_eq!(CpuBackend::sum_all_2d(&empty2d), 0.0);

        // From empty nested vec
        let t = CpuTensor2D::from(&[][..]);
        assert_eq!(t.0.len(), 0);
        assert_eq!((t.1, t.2), (0, 0));

        // Деление на ноль — ожидаем panic или NaN?
        // В текущей реализации будет panic при делении на 0.0.
        // Если это не желаемо — стоит обсудить, но пока покроем как есть.
        let a = vec![1.0];
        let b = vec![0.0];
        let res = CpuBackend::div_1d(&a, &b);
        assert!(res[0].is_infinite()); // или assert_panics, если хочешь явный контроль
    }
}
