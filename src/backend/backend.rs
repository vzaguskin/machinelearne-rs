use crate::backend::{ScalarOps};
pub trait Backend: Clone + 'static {
    type Scalar: ScalarOps + Clone;
    type Tensor1D: Clone + Send + Sync;
    type Tensor2D: Clone + Send + Sync;
    type Device: Clone + Send + Sync;

    fn default_device() -> Self::Device;

    // --- Constructors ---
    fn zeros_1d(len: usize) -> Self::Tensor1D;
    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D;
    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D;
    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D;

    // --- Element-wise ops ---
    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D;
    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D;
    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D;

    // --- Reductions ---
    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar;
    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar;
    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar;

    // --- Scalar ops (for loss gradients, lr updates) ---
    fn scalar_f64(value: f64) -> Self::Scalar;

    // --- Access (for metrics / debug) ---
    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64>;

    fn len_1d(t: &Self::Tensor1D) -> usize;

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D;
    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn maximum_1d(x: &Self::Tensor1D, other: &Self::Tensor1D) -> Self::Tensor1D;
    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D;
    fn scale_1d(a: Self::Scalar, x: &Self::Tensor1D) -> Self::Tensor1D;
}