//! # SPIR-V Backend
//!
//! Vulkan GPU backend using SPIR-V compute shaders for cross-platform GPU operations.
//! Uses rust-gpu to compile Rust kernels to SPIR-V.
//!
//! ## Design Characteristics
//!
//! - **Cross-platform**: Works on any Vulkan-capable GPU (NVIDIA, AMD, Intel, Mesa)
//! - **Rust kernels**: Write GPU kernels in Rust using spirv-std (future)
//! - **Vulkan API**: Uses vulkano for safe Vulkan bindings
//! - **f64 precision**: Uses double precision for numerical stability during training
//! - **Compute-only**: No graphics pipeline, pure compute shaders
//!
//! ## Requirements
//!
//! - Vulkan 1.0+ capable GPU and driver
//! - For rust-gpu: nightly Rust (future)
//! - Vulkan driver installed
//!
//! ## Feature Flag
//!
//! Enable with `spirv` feature in Cargo.toml:
//! ```toml
//! [dependencies]
//! machinelearne-rs = { features = ["spirv"] }
//! ```
//!
//! ## Platforms
//!
//! - Linux (NVIDIA, AMD, Intel, Mesa)
//! - Windows (NVIDIA, AMD, Intel)
//! - macOS via MoltenVK (requires separate setup)
//!
//! ## Example
//!
//! ```rust,ignore
//! use machinelearne_rs::backend::{Backend, spirv::{SpirvBackend, SpirvTensor2D}};
//!
//! // Create a 2x2 matrix on GPU
//! let w = SpirvTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//!
//! // Create input vector
//! let x = SpirvBackend::from_vec_1d(vec![1.0, 0.0]);
//!
//! // Compute matrix-vector product on GPU
//! let y = SpirvBackend::matvec(&w, &x);
//! ```

mod device;
mod tensor;

use super::Backend;
use crate::preprocessing::PreprocessingError;

pub use device::SpirvDevice;
pub use tensor::{SpirvTensor1D, SpirvTensor2D};

/// SPIR-V Vulkan GPU computation backend.
///
/// Provides GPU-accelerated tensor operations using Vulkan compute shaders.
/// All computations use f64 precision for numerical stability.
///
/// # Platform Support
///
/// Works on any platform with Vulkan support:
/// - Linux with NVIDIA, AMD, Intel, or Mesa drivers
/// - Windows with NVIDIA, AMD, or Intel drivers
/// - macOS via MoltenVK (requires separate setup)
#[derive(Clone, Debug, Copy)]
pub struct SpirvBackend;

impl Backend for SpirvBackend {
    type Scalar = f64;
    type Tensor1D = SpirvTensor1D;
    type Tensor2D = SpirvTensor2D;
    type Device = SpirvDevice;

    fn default_device() -> Self::Device {
        SpirvDevice::new(0).expect("Failed to initialize Vulkan device")
    }

    // --- Constructors ---

    fn zeros_1d(len: usize) -> Self::Tensor1D {
        SpirvTensor1D::zeros(len)
    }

    fn zeros_2d(rows: usize, cols: usize) -> Self::Tensor2D {
        SpirvTensor2D::zeros(rows, cols)
    }

    fn from_vec_1d(data: Vec<f32>) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(data.into_iter().map(|x| x as f64).collect())
    }

    fn from_vec_2d(data: Vec<f32>, rows: usize, cols: usize) -> Self::Tensor2D {
        SpirvTensor2D::new(data.into_iter().map(|x| x as f64).collect(), rows, cols)
    }

    // --- Element-wise operations (1D) ---

    fn add_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(a.len(), b.len(), "Tensor lengths must match for add_1d");
        SpirvTensor1D::from_vec(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x + y)
                .collect(),
        )
    }

    fn sub_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(a.len(), b.len(), "Tensor lengths must match for sub_1d");
        SpirvTensor1D::from_vec(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x - y)
                .collect(),
        )
    }

    fn mul_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(a.len(), b.len(), "Tensor lengths must match for mul_1d");
        SpirvTensor1D::from_vec(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x * y)
                .collect(),
        )
    }

    fn div_1d(a: &Self::Tensor1D, b: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(a.len(), b.len(), "Tensor lengths must match for div_1d");
        SpirvTensor1D::from_vec(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x / y)
                .collect(),
        )
    }

    fn mul_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(t.to_vec().iter().map(|x| x * s).collect())
    }

    fn add_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(t.to_vec().iter().map(|x| x + s).collect())
    }

    fn sub_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(t.to_vec().iter().map(|x| x - s).collect())
    }

    fn div_scalar_1d(t: &Self::Tensor1D, s: &Self::Scalar) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(t.to_vec().iter().map(|x| x / s).collect())
    }

    // --- Element-wise operations (2D) ---

    fn mul_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        SpirvTensor2D::new(t.to_vec().iter().map(|x| x * s).collect(), rows, cols)
    }

    fn add_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        SpirvTensor2D::new(t.to_vec().iter().map(|x| x + s).collect(), rows, cols)
    }

    fn sub_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        SpirvTensor2D::new(t.to_vec().iter().map(|x| x - s).collect(), rows, cols)
    }

    fn div_scalar_2d(t: &Self::Tensor2D, s: &Self::Scalar) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        SpirvTensor2D::new(t.to_vec().iter().map(|x| x / s).collect(), rows, cols)
    }

    fn add_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = a.shape();
        assert_eq!(a.shape(), b.shape(), "Tensor shapes must match for add_2d");
        SpirvTensor2D::new(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x + y)
                .collect(),
            rows,
            cols,
        )
    }

    fn sub_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = a.shape();
        assert_eq!(a.shape(), b.shape(), "Tensor shapes must match for sub_2d");
        SpirvTensor2D::new(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x - y)
                .collect(),
            rows,
            cols,
        )
    }

    fn mul_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = a.shape();
        assert_eq!(a.shape(), b.shape(), "Tensor shapes must match for mul_2d");
        SpirvTensor2D::new(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x * y)
                .collect(),
            rows,
            cols,
        )
    }

    fn div_2d(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = a.shape();
        assert_eq!(a.shape(), b.shape(), "Tensor shapes must match for div_2d");
        SpirvTensor2D::new(
            a.to_vec()
                .iter()
                .zip(b.to_vec().iter())
                .map(|(x, y)| x / y)
                .collect(),
            rows,
            cols,
        )
    }

    // --- Reduction operations ---

    fn mean_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        let data = t.to_vec();
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn mean_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        let data = t.to_vec();
        if data.is_empty() {
            panic!("mean_all_2d: cannot compute mean of empty tensor");
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn sum_all_2d(t: &Self::Tensor2D) -> Self::Scalar {
        t.to_vec().iter().sum()
    }

    fn sum_all_1d(t: &Self::Tensor1D) -> Self::Scalar {
        t.to_vec().iter().sum()
    }

    // --- Scalar operations ---

    fn scalar_f64(value: f64) -> Self::Scalar {
        value
    }

    // --- Data access ---

    fn to_vec_1d(t: &Self::Tensor1D) -> Vec<f64> {
        t.to_vec()
    }

    fn len_1d(t: &Self::Tensor1D) -> usize {
        t.len()
    }

    fn len_2d(t: &Self::Tensor2D) -> usize {
        t.rows()
    }

    // --- Mathematical functions (1D) ---

    fn abs_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(t.to_vec().iter().map(|x| x.abs()).collect())
    }

    fn sign_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(
            x.to_vec()
                .iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
    }

    fn maximum_1d(x: &Self::Tensor1D, other: &Self::Tensor1D) -> Self::Tensor1D {
        assert_eq!(
            x.len(),
            other.len(),
            "Tensor lengths must match for maximum_1d"
        );
        SpirvTensor1D::from_vec(
            x.to_vec()
                .iter()
                .zip(other.to_vec().iter())
                .map(|(&a, &b)| a.max(b))
                .collect(),
        )
    }

    fn exp_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(x.to_vec().iter().map(|x| x.exp()).collect())
    }

    fn log_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(x.to_vec().iter().map(|x| x.ln()).collect())
    }

    fn sigmoid_1d(x: &Self::Tensor1D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(
            x.to_vec()
                .iter()
                .map(|&z| {
                    if z >= 0.0 {
                        1.0 / (1.0 + (-z).exp())
                    } else {
                        let ez = z.exp();
                        ez / (1.0 + ez)
                    }
                })
                .collect(),
        )
    }

    // --- Mathematical functions (2D) ---

    fn abs_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        SpirvTensor2D::new(t.to_vec().iter().map(|x| x.abs()).collect(), rows, cols)
    }

    fn sign_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = x.shape();
        SpirvTensor2D::new(
            x.to_vec()
                .iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            rows,
            cols,
        )
    }

    fn maximum_2d(x: &Self::Tensor2D, other: &Self::Tensor2D) -> Self::Tensor2D {
        assert_eq!(
            x.shape(),
            other.shape(),
            "Tensor shapes must match for maximum_2d"
        );
        let (rows, cols) = x.shape();
        SpirvTensor2D::new(
            x.to_vec()
                .iter()
                .zip(other.to_vec().iter())
                .map(|(&a, &b)| a.max(b))
                .collect(),
            rows,
            cols,
        )
    }

    fn exp_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = x.shape();
        SpirvTensor2D::new(x.to_vec().iter().map(|x| x.exp()).collect(), rows, cols)
    }

    fn log_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = x.shape();
        SpirvTensor2D::new(x.to_vec().iter().map(|x| x.ln()).collect(), rows, cols)
    }

    fn sigmoid_2d(x: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = x.shape();
        SpirvTensor2D::new(
            x.to_vec()
                .iter()
                .map(|&z| {
                    if z >= 0.0 {
                        1.0 / (1.0 + (-z).exp())
                    } else {
                        let ez = z.exp();
                        ez / (1.0 + ez)
                    }
                })
                .collect(),
            rows,
            cols,
        )
    }

    // --- Linear algebra ---

    fn matvec(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let (_, cols) = a.shape();
        assert_eq!(cols, x.len(), "Shape mismatch: A.cols != x.len");
        Self::_matvec_unchecked(a, x)
    }

    fn _matvec_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.matvec(x)
    }

    fn matvec_transposed(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        let (rows, _) = a.shape();
        assert_eq!(rows, x.len(), "Shape mismatch: A.rows != x.len");
        Self::_matvec_transposed_unchecked(a, x)
    }

    fn _matvec_transposed_unchecked(a: &Self::Tensor2D, x: &Self::Tensor1D) -> Self::Tensor1D {
        a.matvec_transposed(x)
    }

    fn transpose(t: &Self::Tensor2D) -> Self::Tensor2D {
        t.transpose()
    }

    fn matmul(a: &Self::Tensor2D, b: &Self::Tensor2D) -> Self::Tensor2D {
        a.matmul(b)
    }

    fn shape(t: &Self::Tensor2D) -> (usize, usize) {
        t.shape()
    }

    fn ravel_2d(x: &Self::Tensor2D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(x.to_vec())
    }

    // --- Column-wise operations ---

    fn col_mean_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let (rows, cols) = t.shape();
        if cols == 0 || rows == 0 {
            return SpirvTensor1D::zeros(0);
        }

        let data = t.to_vec();
        let mut means = vec![0.0f64; cols];
        for col in 0..cols {
            let mut sum = 0.0f64;
            for row in 0..rows {
                sum += data[row * cols + col];
            }
            means[col] = sum / rows as f64;
        }
        SpirvTensor1D::from_vec(means)
    }

    fn col_std_2d(t: &Self::Tensor2D, ddof: usize) -> Self::Tensor1D {
        let (rows, cols) = t.shape();
        if cols == 0 || rows == 0 {
            return SpirvTensor1D::zeros(0);
        }

        let data = t.to_vec();
        let means = Self::col_mean_2d(t).to_vec();
        let mut stds = vec![0.0f64; cols];

        for col in 0..cols {
            let mut var_sum = 0.0f64;
            for row in 0..rows {
                let diff = data[row * cols + col] - means[col];
                var_sum += diff * diff;
            }
            let divisor = (rows - ddof) as f64;
            stds[col] = (var_sum / divisor).sqrt();
        }
        SpirvTensor1D::from_vec(stds)
    }

    fn col_min_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let (rows, cols) = t.shape();
        if cols == 0 || rows == 0 {
            return SpirvTensor1D::zeros(0);
        }

        let data = t.to_vec();
        let mut mins = vec![f64::INFINITY; cols];
        for col in 0..cols {
            for row in 0..rows {
                let val = data[row * cols + col];
                if val < mins[col] {
                    mins[col] = val;
                }
            }
        }
        SpirvTensor1D::from_vec(mins)
    }

    fn col_max_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let (rows, cols) = t.shape();
        if cols == 0 || rows == 0 {
            return SpirvTensor1D::zeros(0);
        }

        let data = t.to_vec();
        let mut maxs = vec![f64::NEG_INFINITY; cols];
        for col in 0..cols {
            for row in 0..rows {
                let val = data[row * cols + col];
                if val > maxs[col] {
                    maxs[col] = val;
                }
            }
        }
        SpirvTensor1D::from_vec(maxs)
    }

    fn col_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let (rows, cols) = t.shape();
        if cols == 0 || rows == 0 {
            return SpirvTensor1D::zeros(0);
        }

        let data = t.to_vec();
        let mut sums = vec![0.0f64; cols];
        for col in 0..cols {
            for row in 0..rows {
                sums[col] += data[row * cols + col];
            }
        }
        SpirvTensor1D::from_vec(sums)
    }

    // --- Row-wise operations ---

    fn row_sum_2d(t: &Self::Tensor2D) -> Self::Tensor1D {
        let (rows, cols) = t.shape();
        if rows == 0 || cols == 0 {
            return SpirvTensor1D::zeros(0);
        }

        let data = t.to_vec();
        let mut sums = vec![0.0f64; rows];
        for row in 0..rows {
            for col in 0..cols {
                sums[row] += data[row * cols + col];
            }
        }
        SpirvTensor1D::from_vec(sums)
    }

    // --- Broadcasting operations ---

    fn broadcast_sub_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        assert_eq!(v.len(), cols, "Vector length must match number of columns");

        let data = t.to_vec();
        let v_data = v.to_vec();
        let mut result = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                result.push(data[row * cols + col] - v_data[col]);
            }
        }
        SpirvTensor2D::new(result, rows, cols)
    }

    fn broadcast_div_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        assert_eq!(v.len(), cols, "Vector length must match number of columns");

        let data = t.to_vec();
        let v_data = v.to_vec();
        let mut result = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                result.push(data[row * cols + col] / v_data[col]);
            }
        }
        SpirvTensor2D::new(result, rows, cols)
    }

    fn broadcast_mul_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        assert_eq!(v.len(), cols, "Vector length must match number of columns");

        let data = t.to_vec();
        let v_data = v.to_vec();
        let mut result = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                result.push(data[row * cols + col] * v_data[col]);
            }
        }
        SpirvTensor2D::new(result, rows, cols)
    }

    fn broadcast_add_1d_to_2d_rows(t: &Self::Tensor2D, v: &Self::Tensor1D) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        assert_eq!(v.len(), cols, "Vector length must match number of columns");

        let data = t.to_vec();
        let v_data = v.to_vec();
        let mut result = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                result.push(data[row * cols + col] + v_data[col]);
            }
        }
        SpirvTensor2D::new(result, rows, cols)
    }

    fn sqrt_1d(t: &Self::Tensor1D) -> Self::Tensor1D {
        SpirvTensor1D::from_vec(t.to_vec().iter().map(|x| x.sqrt()).collect())
    }

    fn sqrt_2d(t: &Self::Tensor2D) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        SpirvTensor2D::new(t.to_vec().iter().map(|x| x.sqrt()).collect(), rows, cols)
    }

    // --- Column manipulation operations ---

    fn hcat_2d(tensors: &[Self::Tensor2D]) -> Result<Self::Tensor2D, PreprocessingError> {
        if tensors.is_empty() {
            return Err(PreprocessingError::InvalidParameter(
                "Cannot horizontally concatenate empty slice of tensors".to_string(),
            ));
        }

        let rows = tensors[0].rows();
        if rows == 0 {
            return Ok(SpirvTensor2D::zeros(0, 0));
        }

        // Verify all tensors have the same number of rows
        for t in tensors.iter() {
            if t.rows() != rows {
                return Err(PreprocessingError::InvalidShape {
                    expected: format!("({}, ?)", rows),
                    got: format!("({}, ?)", t.rows()),
                });
            }
        }

        // Calculate total columns
        let total_cols: usize = tensors.iter().map(|t| t.cols()).sum();

        // Concatenate row by row
        let mut result = Vec::with_capacity(rows * total_cols);
        for row in 0..rows {
            for t in tensors {
                let data = t.to_vec();
                let cols = t.cols();
                let start = row * cols;
                let end = start + cols;
                result.extend_from_slice(&data[start..end]);
            }
        }

        Ok(SpirvTensor2D::new(result, rows, total_cols))
    }

    fn select_columns_2d(t: &Self::Tensor2D, columns: &[usize]) -> Self::Tensor2D {
        let (rows, cols) = t.shape();
        if columns.is_empty() {
            return SpirvTensor2D::zeros(rows, 0);
        }

        // Validate column indices
        for &col in columns {
            assert!(
                col < cols,
                "Column index {} out of bounds (max {})",
                col,
                cols - 1
            );
        }

        let data = t.to_vec();
        let mut result = Vec::with_capacity(rows * columns.len());
        for row in 0..rows {
            for &col in columns {
                result.push(data[row * cols + col]);
            }
        }

        SpirvTensor2D::new(result, rows, columns.len())
    }

    fn one_hot_from_indices(indices: &Self::Tensor1D, num_classes: usize) -> Self::Tensor2D {
        let n_samples = indices.len();
        if n_samples == 0 || num_classes == 0 {
            return SpirvTensor2D::zeros(n_samples, num_classes);
        }

        let indices_data = indices.to_vec();

        // Validate indices
        for (i, &idx) in indices_data.iter().enumerate() {
            assert!(
                idx >= 0.0 && idx < num_classes as f64 && idx.fract() == 0.0,
                "Index {} at position {} is not a valid integer in range [0, {})",
                idx,
                i,
                num_classes
            );
        }

        let mut result = vec![0.0f64; n_samples * num_classes];
        for (i, &idx) in indices_data.iter().enumerate() {
            let col = idx as usize;
            result[i * num_classes + col] = 1.0;
        }

        SpirvTensor2D::new(result, n_samples, num_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructors() {
        let z1 = SpirvBackend::zeros_1d(3);
        assert_eq!(z1.len(), 3);

        let z2 = SpirvBackend::zeros_2d(2, 3);
        assert_eq!(z2.shape(), (2, 3));

        let v1 = SpirvBackend::from_vec_1d(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(v1.len(), 3);

        let v2 = SpirvBackend::from_vec_2d(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(v2.shape(), (2, 2));
    }

    #[test]
    fn test_elementwise_ops_1d() {
        let a = SpirvBackend::from_vec_1d(vec![1.0, 2.0]);
        let b = SpirvBackend::from_vec_1d(vec![3.0, 4.0]);

        let add = SpirvBackend::add_1d(&a, &b);
        assert_eq!(add.to_vec(), vec![4.0, 6.0]);

        let sub = SpirvBackend::sub_1d(&a, &b);
        assert_eq!(sub.to_vec(), vec![-2.0, -2.0]);

        let mul = SpirvBackend::mul_1d(&a, &b);
        assert_eq!(mul.to_vec(), vec![3.0, 8.0]);
    }

    #[test]
    fn test_elementwise_ops_2d() {
        let a = SpirvBackend::from_vec_2d(vec![1.0, 2.0], 1, 2);
        let b = SpirvBackend::from_vec_2d(vec![3.0, 4.0], 1, 2);

        let add = SpirvBackend::add_2d(&a, &b);
        assert_eq!(add.to_vec(), vec![4.0, 6.0]);

        let mul_s = SpirvBackend::mul_scalar_2d(&a, &2.0);
        assert_eq!(mul_s.to_vec(), vec![2.0, 4.0]);
    }

    #[test]
    fn test_reductions() {
        let v = SpirvBackend::from_vec_1d(vec![1.0, 2.0, 3.0]);
        assert!((SpirvBackend::sum_all_1d(&v) - 6.0).abs() < 1e-12);
        assert!((SpirvBackend::mean_all_1d(&v) - 2.0).abs() < 1e-12);

        let m = SpirvBackend::from_vec_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!((SpirvBackend::sum_all_2d(&m) - 10.0).abs() < 1e-12);
        assert!((SpirvBackend::mean_all_2d(&m) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_math_functions() {
        let v = SpirvBackend::from_vec_1d(vec![0.0, 1.0]);
        let exp_v = SpirvBackend::exp_1d(&v);
        assert!((exp_v.to_vec()[0] - 1.0).abs() < 1e-12);
        assert!((exp_v.to_vec()[1] - std::f64::consts::E).abs() < 1e-12);

        // sigmoid(0) = 0.5
        let sig = SpirvBackend::sigmoid_1d(&SpirvBackend::from_vec_1d(vec![0.0]));
        assert!((sig.to_vec()[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_transpose() {
        let m = SpirvBackend::from_vec_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = SpirvBackend::transpose(&m);
        assert_eq!(t.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(t.shape(), (3, 2));
    }

    #[test]
    fn test_matvec() {
        let m = SpirvBackend::from_vec_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let v = SpirvBackend::from_vec_1d(vec![1.0, 0.0]);
        let result = SpirvBackend::matvec(&m, &v);
        assert_eq!(result.to_vec(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_hcat_2d_basic() {
        let a = SpirvBackend::from_vec_2d(vec![1.0, 2.0], 1, 2);
        let b = SpirvBackend::from_vec_2d(vec![3.0, 4.0], 1, 2);
        let result = SpirvBackend::hcat_2d(&[a, b]).unwrap();
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape(), (1, 4));
    }

    #[test]
    fn test_select_columns_2d() {
        let t = SpirvBackend::from_vec_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = SpirvBackend::select_columns_2d(&t, &[0, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 4.0, 6.0]);
        assert_eq!(result.shape(), (2, 2));
    }

    #[test]
    fn test_one_hot_from_indices() {
        let indices = SpirvBackend::from_vec_1d(vec![0.0, 1.0, 2.0]);
        let result = SpirvBackend::one_hot_from_indices(&indices, 3);
        assert_eq!(
            result.to_vec(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(result.shape(), (3, 3));
    }
}
