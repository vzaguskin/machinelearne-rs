use crate::backend::{Backend, Device};

#[derive(Clone, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Scalar = f64;
    type Tensor0D = f64;
    type Tensor1D = Vec<f64>;
    type Tensor2D = (Vec<f64>, usize, usize); // (data, rows, cols)
    type Device = Device;

    // === Allocation ===
    fn zeros_1d(n: usize, _device: &Device) -> Self::Tensor1D {
        vec![0.0; n]
    }

    fn zeros_2d(rows: usize, cols: usize, _device: &Device) -> Self::Tensor2D {
        (vec![0.0; rows * cols], rows, cols)
    }

    fn scalar(value: f64, _device: &Device) -> f64 {
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