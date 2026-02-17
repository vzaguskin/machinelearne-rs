//! Linear algebra utilities for closed-form solutions.
//!
//! This module provides matrix operations needed for analytical solutions
//! like the normal equation for linear regression.

use crate::backend::{Backend, Tensor1D, Tensor2D};

/// Error type for linear algebra operations.
#[derive(Debug, Clone)]
pub enum LinalgError {
    /// Matrix is singular and cannot be inverted.
    SingularMatrix(String),
    /// Dimensions don't match for the operation.
    DimensionMismatch(String),
    /// Empty input provided.
    EmptyInput(String),
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::SingularMatrix(msg) => write!(f, "Singular matrix: {}", msg),
            LinalgError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            LinalgError::EmptyInput(msg) => write!(f, "Empty input: {}", msg),
        }
    }
}

impl std::error::Error for LinalgError {}

/// Computes the inverse of a square matrix using Gauss-Jordan elimination.
///
/// This is a simple implementation suitable for small matrices (up to ~100x100).
/// For larger matrices, consider using a dedicated linear algebra library.
///
/// # Arguments
/// * `matrix` - A square matrix to invert
///
/// # Returns
/// The inverse matrix, or an error if the matrix is singular.
///
/// # Example
/// ```ignore
/// use machinelearne_rs::linalg::inverse;
/// use machinelearne_rs::backend::{CpuBackend, Tensor2D};
///
/// let a = Tensor2D::<CpuBackend>::new(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
/// let a_inv = inverse(&a).unwrap();
/// // a @ a_inv ≈ I
/// ```
#[allow(clippy::needless_range_loop)]
pub fn inverse<B: Backend>(matrix: &Tensor2D<B>) -> Result<Tensor2D<B>, LinalgError> {
    let (rows, cols) = matrix.shape();
    if rows == 0 || cols == 0 {
        return Err(LinalgError::EmptyInput("Cannot invert empty matrix".into()));
    }
    if rows != cols {
        return Err(LinalgError::DimensionMismatch(format!(
            "Matrix must be square for inversion, got {}x{}",
            rows, cols
        )));
    }

    let n = rows;

    // Create augmented matrix [A | I]
    // We'll work with f64 for numerical operations
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Fill with matrix values and identity
    let flat = matrix.ravel().to_vec();
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = flat[i * n + j];
        }
        aug[i][n + i] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        // Swap rows
        aug.swap(col, max_row);

        // Check for singularity
        if aug[col][col].abs() < 1e-12 {
            return Err(LinalgError::SingularMatrix(
                "Matrix is singular or nearly singular".into(),
            ));
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut result = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            result.push(aug[i][n + j] as f32);
        }
    }

    Ok(Tensor2D::new(result, n, n))
}

/// Solves the linear system Ax = b using the normal equation.
///
/// Computes x = (A^T A)^(-1) A^T b
///
/// This is used for linear regression closed-form solution.
///
/// # Arguments
/// * `x` - Design matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
///
/// # Returns
/// Solution vector (n_features + 1) including bias term, or an error.
pub fn solve_normal_equation<B: Backend>(
    x: &Tensor2D<B>,
    y: &Tensor1D<B>,
) -> Result<Tensor1D<B>, LinalgError> {
    let (n_samples, n_features) = x.shape();
    let y_len = y.len();

    if n_samples == 0 || n_features == 0 {
        return Err(LinalgError::EmptyInput("Empty design matrix".into()));
    }
    if y_len == 0 {
        return Err(LinalgError::EmptyInput("Empty target vector".into()));
    }
    if n_samples != y_len {
        return Err(LinalgError::DimensionMismatch(format!(
            "X has {} samples but y has {} elements",
            n_samples, y_len
        )));
    }

    // Build augmented matrix X_aug = [X | 1] for bias term
    let n_aug = n_features + 1;
    let mut x_aug_data = Vec::with_capacity(n_samples * n_aug);
    let x_flat = x.ravel().to_vec(); // Vec<f64>
    for i in 0..n_samples {
        for j in 0..n_features {
            x_aug_data.push(x_flat[i * n_features + j] as f32);
        }
        x_aug_data.push(1.0); // Bias column
    }
    let x_aug = Tensor2D::<B>::new(x_aug_data, n_samples, n_aug);

    // Compute X^T X
    let xt = x_aug.transpose();
    let xtx = xt.matmul(&x_aug);

    // Compute (X^T X)^(-1)
    let xtx_inv = inverse(&xtx)?;

    // Compute X^T y
    let y_vec = y.to_vec(); // Vec<f64>
    let mut xty_data = vec![0.0f32; n_aug];
    for j in 0..n_aug {
        let mut sum = 0.0f64;
        for i in 0..n_samples {
            let x_val = if j < n_features {
                x_flat[i * n_features + j]
            } else {
                1.0
            };
            sum += x_val * y_vec[i];
        }
        xty_data[j] = sum as f32;
    }
    let xty = Tensor1D::<B>::new(xty_data);

    // Compute w = (X^T X)^(-1) X^T y
    let xty_vec: Vec<f64> = xty.to_vec();
    let xty_col = Tensor2D::<B>::new(xty_vec.iter().map(|&v| v as f32).collect(), n_aug, 1);
    let w_matrix = xtx_inv.matmul(&xty_col);
    let w_flat: Vec<f64> = w_matrix.ravel().to_vec();

    Ok(Tensor1D::new(w_flat.iter().map(|&v| v as f32).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_inverse_identity() {
        // Identity matrix inverse is itself
        let i = Tensor2D::<CpuBackend>::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let i_inv = inverse(&i).unwrap();

        let result = i_inv.ravel().to_vec();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverse_2x2() {
        // [[4, 7], [2, 6]]^-1 = [[0.6, -0.7], [-0.2, 0.4]]
        let a = Tensor2D::<CpuBackend>::new(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
        let a_inv = inverse(&a).unwrap();

        // Verify A @ A^-1 ≈ I
        let product = a.matmul(&a_inv);
        let result = product.ravel().to_vec();

        assert!((result[0] - 1.0).abs() < 1e-6); // (0,0) ≈ 1
        assert!((result[1] - 0.0).abs() < 1e-6); // (0,1) ≈ 0
        assert!((result[2] - 0.0).abs() < 1e-6); // (1,0) ≈ 0
        assert!((result[3] - 1.0).abs() < 1e-6); // (1,1) ≈ 1
    }

    #[test]
    fn test_inverse_singular() {
        // Singular matrix (zero determinant)
        let a = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 2.0, 4.0], 2, 2);
        let result = inverse(&a);
        assert!(result.is_err());
    }

    #[test]
    fn test_inverse_non_square() {
        let a = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = inverse(&a);
        assert!(matches!(result, Err(LinalgError::DimensionMismatch(_))));
    }

    #[test]
    fn test_solve_normal_equation_simple() {
        // y = 2x + 1
        // X = [[1], [2], [3]], y = [3, 5, 7]
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0], 3, 1);
        let y = Tensor1D::<CpuBackend>::new(vec![3.0, 5.0, 7.0]);

        let w = solve_normal_equation(&x, &y).unwrap();
        let w_vec = w.to_vec();

        assert!((w_vec[0] - 2.0).abs() < 1e-5); // weight ≈ 2
        assert!((w_vec[1] - 1.0).abs() < 1e-5); // bias ≈ 1
    }

    #[test]
    fn test_solve_normal_equation_multifeature() {
        // y = 2*x1 + 3*x2 + 1
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0], 4, 2);
        let y = Tensor1D::<CpuBackend>::new(vec![3.0, 4.0, 6.0, 14.0]);

        let w = solve_normal_equation(&x, &y).unwrap();
        let w_vec = w.to_vec();

        assert!((w_vec[0] - 2.0).abs() < 1e-6); // weight1 ≈ 2
        assert!((w_vec[1] - 3.0).abs() < 1e-6); // weight2 ≈ 3
        assert!((w_vec[2] - 1.0).abs() < 1e-6); // bias ≈ 1
    }

    #[test]
    fn test_solve_dimension_mismatch() {
        let x = Tensor2D::<CpuBackend>::new(vec![1.0, 2.0, 3.0], 3, 1);
        let y = Tensor1D::<CpuBackend>::new(vec![1.0, 2.0]); // Wrong length

        let result = solve_normal_equation(&x, &y);
        assert!(matches!(result, Err(LinalgError::DimensionMismatch(_))));
    }
}
