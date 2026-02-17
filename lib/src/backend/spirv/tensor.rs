//! SPIR-V/Vulkan tensor types.
//!
//! Provides 1D and 2D tensor implementations for Vulkan GPU computation.
//! Uses f64 precision for numerical stability during training.

use std::sync::Arc;

/// 1D tensor stored in Vulkan GPU memory.
///
/// Uses Vulkan buffers for GPU storage. In a full implementation,
/// would use host-visible or device-local memory based on usage patterns.
#[derive(Clone, Debug)]
pub struct SpirvTensor1D {
    /// Host-side data cache
    /// In a full implementation, this would be a Vulkan buffer
    data: Arc<Vec<f64>>,
}

impl SpirvTensor1D {
    /// Creates a new 1D tensor from a vector.
    pub fn from_vec(data: Vec<f64>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Creates a 1D tensor filled with zeros.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: Arc::new(vec![0.0f64; len]),
        }
    }

    /// Returns the length of the tensor.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Copies data from GPU to host and returns as Vec.
    pub fn to_vec(&self) -> Vec<f64> {
        (*self.data).clone()
    }

    /// Returns a slice view of the data.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// 2D tensor stored in Vulkan GPU memory.
///
/// Data is stored in row-major order. In a full implementation,
/// would use Vulkan buffers with appropriate memory type.
#[derive(Clone, Debug)]
pub struct SpirvTensor2D {
    /// Host-side data cache
    /// In a full implementation, this would be a Vulkan buffer
    data: Arc<Vec<f64>>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
}

impl SpirvTensor2D {
    /// Creates a new 2D tensor from row-major data.
    ///
    /// # Arguments
    /// * `data` - Flattened data in row-major order
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} doesn't match shape ({}, {})",
            data.len(),
            rows,
            cols
        );
        Self {
            data: Arc::new(data),
            rows,
            cols,
        }
    }

    /// Creates a 2D tensor filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: Arc::new(vec![0.0f64; rows * cols]),
            rows,
            cols,
        }
    }

    /// Returns the shape as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Copies data from GPU to host and returns as Vec.
    pub fn to_vec(&self) -> Vec<f64> {
        (*self.data).clone()
    }

    /// Returns a slice view of the data.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns the transpose of this tensor.
    ///
    /// Converts (rows, cols) -> (cols, rows) with proper element reordering.
    pub fn transpose(&self) -> Self {
        let mut transposed = Vec::with_capacity(self.rows * self.cols);
        for col in 0..self.cols {
            for row in 0..self.rows {
                transposed.push(self.data[row * self.cols + col]);
            }
        }
        Self {
            data: Arc::new(transposed),
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Matrix-vector multiplication: y = A * x
    ///
    /// Where A is self (m × n), x is the input vector (n,), and result is (m,).
    ///
    /// In a full implementation, this would use a Vulkan compute shader.
    pub fn matvec(&self, x: &SpirvTensor1D) -> SpirvTensor1D {
        assert_eq!(
            self.cols,
            x.len(),
            "Matrix cols {} must match vector length {}",
            self.cols,
            x.len()
        );

        let x_data = x.as_slice();
        let mut result = vec![0.0f64; self.rows];

        for i in 0..self.rows {
            let mut sum = 0.0f64;
            for j in 0..self.cols {
                sum += self.data[i * self.cols + j] * x_data[j];
            }
            result[i] = sum;
        }

        SpirvTensor1D::from_vec(result)
    }

    /// Transposed matrix-vector multiplication: y = A^T * x
    ///
    /// Where A is self (m × n), x is the input vector (m,), and result is (n,).
    pub fn matvec_transposed(&self, x: &SpirvTensor1D) -> SpirvTensor1D {
        assert_eq!(
            self.rows,
            x.len(),
            "Matrix rows {} must match vector length {}",
            self.rows,
            x.len()
        );

        let x_data = x.as_slice();
        let mut result = vec![0.0f64; self.cols];

        for j in 0..self.cols {
            let mut sum = 0.0f64;
            for i in 0..self.rows {
                sum += self.data[i * self.cols + j] * x_data[i];
            }
            result[j] = sum;
        }

        SpirvTensor1D::from_vec(result)
    }

    /// Matrix-matrix multiplication: C = A * B
    ///
    /// Where A is self (m × k), B is (k × n), and result is (m × n).
    ///
    /// In a full implementation, this would use a tiled Vulkan compute shader.
    pub fn matmul(&self, b: &SpirvTensor2D) -> SpirvTensor2D {
        assert_eq!(
            self.cols, b.rows,
            "Matrix dimensions don't match: ({}, {}) x ({}, {})",
            self.rows, self.cols, b.rows, b.cols
        );

        let m = self.rows;
        let k = self.cols;
        let n = b.cols;

        let mut result = vec![0.0f64; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += self.data[i * k + p] * b.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        SpirvTensor2D::new(result, m, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor1d_basic() {
        let t = SpirvTensor1D::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.len(), 3);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);

        let z = SpirvTensor1D::zeros(5);
        assert_eq!(z.to_vec(), vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tensor2d_basic() {
        let t = SpirvTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(t.shape(), (2, 2));
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        let z = SpirvTensor2D::zeros(2, 3);
        assert_eq!(z.shape(), (2, 3));
        assert_eq!(z.to_vec(), vec![0.0; 6]);
    }

    #[test]
    fn test_transpose() {
        let t = SpirvTensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let tt = t.transpose();
        assert_eq!(tt.shape(), (3, 2));
        assert_eq!(tt.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matvec() {
        let m = SpirvTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let v = SpirvTensor1D::from_vec(vec![1.0, 0.0]);
        let result = m.matvec(&v);
        assert_eq!(result.to_vec(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_matvec_transposed() {
        // X = [[1, 2], [3, 4], [5, 6]], v = [1, 0, 2]
        // X^T @ v = [1 + 0 + 10, 2 + 0 + 12] = [11, 14]
        let m = SpirvTensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let v = SpirvTensor1D::from_vec(vec![1.0, 0.0, 2.0]);
        let result = m.matvec_transposed(&v);
        assert_eq!(result.to_vec(), vec![11.0, 14.0]);
    }

    #[test]
    fn test_matmul() {
        let a = SpirvTensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = SpirvTensor2D::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let result = a.matmul(&b);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
