use super::{Backend, Backend_};
#[derive(Clone, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Scalar = f64;
    type Tensor1D = Vec<f64>;
    type Tensor2D = (Vec<f64>, usize, usize); // (data, rows, cols)
    type Device = ();

    fn default_device() -> Self::Device{
        ()
    }

    // --- Constructors ---
    fn zeros_1d(len: usize) -> Self::Tensor1D{
        vec![0.; len]
    }
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D{
        (vec![0.; rows * cols], rows, cols)
    }
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D{
        data.into_iter().map(|x| {x as f64}).collect()
    }
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D{
        (data.into_iter().map(|x| {x as f64}).collect(), rows, cols)
    }

    // --- Element-wise ops ---
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D{
        a.iter().zip(b.iter()).map(|(a, b)| {a + b} ).collect()
    }
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D{
        a.iter().zip(b.iter()).map(|(a, b)| {a - b} ).collect()
    }
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D{
        a.iter().zip(b.iter()).map(|(a, b)| {a * b} ).collect()
    }
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D{
        t.iter().map(|x| {*x * s}).collect()
    }

    // --- Reductions ---
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar{
        t.iter().sum::<f64>() / t.len() as f64
    }
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar{
        t.0.iter().sum::<f64>()
    }

    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar{
        t.iter().sum::<f64>()
    }

    // --- Scalar ops (for loss gradients, lr updates) ---
    fn scalar_f64(value: f64) -> Self::Scalar{
        value
    }
    fn scalar_mul_1d(s: &Self::Scalar, t: &Self::Tensor1D) -> Self::Tensor1D{
        t.iter().map(|x| x * s).collect()
    }

    // --- Access (for metrics / debug) ---
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64>{
        t.clone()
    }

    fn len_1d(t: &Self::Tensor1D) -> usize{
        t.len()
    }

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D{
        t.iter().map(|x| {x.abs()}).collect()
    }

    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D{
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
}

impl Backend_ for CpuBackend {
    type Scalar = f64;
    type Tensor0D = f64;
    type Tensor1D = Vec<f64>;
    type Tensor2D = (Vec<f64>, usize, usize); // (data, rows, cols)
    type Device = ();

    fn default_device() -> Self::Device{
        ()
    }

    // === Allocation ===
    fn zeros_1d(n: usize, _device: &Self::Device) -> Self::Tensor1D {
        vec![0.0; n]
    }

    fn zeros_2d(rows: usize, cols: usize, _device: &Self::Device) -> Self::Tensor2D {
        (vec![0.0; rows * cols], rows, cols)
    }

    fn scalar(value: f64, _device: &Self::Device) -> f64 {
        value
    }

    // === Shape queries ===
    fn len_1d(x: &Vec<f64>) -> usize {
        x.len()
    }

    fn shape_2d(x: &(Vec<f64>, usize, usize)) -> (usize, usize) {
        (x.1, x.2)
    }

    // === Element access ===
    fn get_1d(x: &Vec<f64>, i: usize) -> f64 {
        x[i]
    }

    fn set_1d(x: &mut Vec<f64>, i: usize, v: f64) {
        x[i] = v;
    }

    fn get_2d(x: &(Vec<f64>, usize, usize), i: usize, j: usize) -> f64 {
        let (_, rows, cols) = x;
        assert!(i < *rows && j < *cols, "get_2d: index out of bounds");
        x.0[i * cols + j]
    }

    fn set_2d(x: &mut (Vec<f64>, usize, usize), i: usize, j: usize, v: f64) {
        let (_, rows, cols) = x;
        assert!(i < *rows && j < *cols, "set_2d: index out of bounds");
        x.0[i * *cols + j] = v;
    }

    // === Element-wise ops ===
    fn add_scalar_1d(x: &Vec<f64>, s: f64) -> Vec<f64> {
        x.iter().map(|&xi| xi + s).collect()
    }

    fn add_scalar_1d_inplace(x: &mut Self::Tensor1D, s: Self::Scalar){
        for el  in x {
            *el += s;
        }
    }

    fn scale_1d(a: f64, x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&xi| a * xi).collect()
    }

    fn scale_2d(a: f64, x: &(Vec<f64>, usize, usize)) -> (Vec<f64>, usize, usize) {
        let (data, rows, cols) = x;
        let scaled = data.iter().map(|&v| a * v).collect();
        (scaled, *rows, *cols)
    }

    fn abs_1d(x: &Vec<f64>) -> Vec<f64>{
        x.iter().map(|x| {x.abs()}).collect()
    }

    fn sign_1d(x: &Vec<f64>) -> Vec<f64>{
        x
        .iter()
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

    fn sigmoid_1d(x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect()
    }

    fn maximum_1d(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        a.iter().zip(b).map(|(&x, &y)| x.max(y)).collect()
    }

    fn neg_1d(x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| -v).collect()
    }

    fn exp_1d(x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| v.exp()).collect()
    }

    fn log_1d(x: &Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| v.ln()).collect()
    }

    // === Unchecked BLAS-like implementations ===
    fn _dot_unchecked(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }

    fn _axpy_unchecked(a: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi += a * xi;
        }
    }

    fn _matvec_unchecked(a: &(Vec<f64>, usize, usize), x: &Vec<f64>) -> Vec<f64> {
        let (data, rows, cols) = a;
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

    fn _matvec_transpose_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D{
        //todo - efficient implementation
        Self::_matvec_unchecked(&Self::transpose(a), x)
    }

    fn _matmul_unchecked(a: &(Vec<f64>, usize, usize), b: &(Vec<f64>, usize, usize)) -> (Vec<f64>, usize, usize) {
        let (a_data, a_rows, a_cols) = a;
        let (b_data, b_rows, b_cols) = b;
        assert_eq!(*a_cols, *b_rows, "Internal error: matmul unchecked called with incompatible shapes");

        let mut c_data = vec![0.0; a_rows * b_cols];
        for i in 0..*a_rows {
            for j in 0..*b_cols {
                let mut sum = 0.0;
                for k in 0..*a_cols {
                    sum += a_data[i * *a_cols + k] * b_data[k * *b_cols + j];
                }
                c_data[i * b_cols + j] = sum;
            }
        }
        (c_data, *a_rows, *b_cols)
    }

    fn _minus_vec_vec_unchecked(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64>{
        let mut out = Vec::with_capacity(x.len());
        for (xi, yi) in x.iter().zip(y.iter()){
            out.push(xi - yi);
        }

        out

    }

    fn _plus_vec_vec_unchecked(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64>{
        let mut out = Vec::with_capacity(x.len());
        for (xi, yi) in x.iter().zip(y.iter()){
            out.push(xi + yi);
        }

        out

    }

    fn mul_1d(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64>{
        let mut out = Vec::with_capacity(x.len());
        for (xi, yi) in x.iter().zip(y.iter()){
            out.push(xi * yi);
        }

        out

    }

    fn sum_1d(x: &Vec<f64>) -> f64{

        x.iter().sum()

    }

    
    fn sum_2d(x: &Self::Tensor2D) -> f64{
        x.0.iter().sum()
    }

    fn transpose(a: &(Vec<f64>, usize, usize)) -> Self::Tensor2D{
        //TODO: return view slice
        let (inp, rows, cols ) = a;
        let mut out = Vec::with_capacity( cols * rows);
        for col in 0..*cols{
            for row in 0..*rows{
                out.push(inp[row * cols + col]);
            }
        }

        (out, *cols, *rows)

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
        let x = (x_data, 3, 2); // (rows=3, cols=2)
        let v = vec![1.0, 0.0, 2.0];

        let result = CpuBackend::matvec_transpose(&x, &v);
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

        let x2 = (vec![2.0, 3.0, 4.0, 5.0], 4, 1);
        let v2 = vec![1.0, 1.0, 1.0, 1.0];
        let result2 = CpuBackend::matvec_transpose(&x2, &v2);
        let expected2 = vec![14.0];

        assert_eq!(result2, expected2);

        // Пример 3: (2, 3) → output len = 3
        // X = [[1, 0, 0],
        //      [0, 1, 0]]
        // v = [5.0, 7.0]
        // Xᵀ @ v = [5, 7, 0]

        let x3 = (vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let v3 = vec![5.0, 7.0];
        let result3 = CpuBackend::matvec_transpose(&x3, &v3);
        let expected3 = vec![5.0, 7.0, 0.0];

        assert_eq!(result3.len(), 3);
        for (r, e) in result3.iter().zip(expected3.iter()) {
            assert!((r - e).abs() < 1e-12, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_matvec_transpose_consistency_with_transpose_and_matvec() {

        let x = (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // (3,2)
        let v = vec![1.0, 0.0, 2.0];

        let result1 = CpuBackend::matvec_transpose(&x, &v);
        let x_t = CpuBackend::transpose(&x);
        let result2 = CpuBackend::matvec(&x_t, &v);

        assert_eq!(result1.len(), result2.len());
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}