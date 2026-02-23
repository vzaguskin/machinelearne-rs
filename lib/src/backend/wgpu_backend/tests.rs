//! Comprehensive tests for WGPU backend operations.
//!
//! These tests verify that WGPU backend operations produce results matching
//! the CPU backend within acceptable floating-point tolerances.
//!
//! **Note**: These tests require a GPU with Vulkan/Metal/D3D12 support.
//! They will be skipped on systems without a suitable GPU adapter.

use super::*;
use crate::backend::{Backend, CpuBackend};

/// Check if GPU is available, return true if tests should proceed.
/// Prints a message if GPU is not available.
fn gpu_available() -> bool {
    WgpuDevice::is_available()
}

/// Test sizes: 1 (edge), 4 (small), 16, 64 (medium), 256, 1024 (large)
fn test_sizes() -> Vec<usize> {
    vec![1, 4, 16, 64, 256, 1024]
}

/// Smaller test sizes for 2D operations (to keep matrix sizes reasonable)
fn test_sizes_2d() -> Vec<(usize, usize)> {
    vec![(1, 1), (2, 2), (4, 4), (8, 8), (16, 32), (32, 64)]
}

/// Generate test data for 1D tensors
fn generate_test_data_1d(size: usize) -> Vec<f32> {
    (0..size).map(|i| ((i as f32 % 10.0) + 1.0) / 5.0).collect()
}

/// Generate test data for 2D tensors (row-major)
fn generate_test_data_2d(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i as f32 % 20.0) + 1.0) / 10.0))
        .collect()
}

/// Compare two 1D vectors with relative tolerance
fn assert_close_1d(actual: &[f64], expected: &[f64], tolerance: f64, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", msg);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // Handle NaN case - both should be NaN
        if a.is_nan() && e.is_nan() {
            continue;
        }
        if a.is_nan() || e.is_nan() {
            panic!(
                "{} at index {}: one value is NaN (expected {:.6}, got {:.6})",
                msg, i, e, a
            );
        }
        let diff = (a - e).abs();
        let rel_diff = if e.abs() > 1e-10 {
            diff / e.abs()
        } else {
            diff
        };
        assert!(
            rel_diff < tolerance,
            "{} at index {}: expected {:.6}, got {:.6}, rel_diff={:.6}",
            msg,
            i,
            e,
            a,
            rel_diff
        );
    }
}

/// Compare two f64 scalars with relative tolerance
fn assert_close_scalar(actual: f64, expected: f64, tolerance: f64, msg: &str) {
    // Handle NaN case - both should be NaN
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    if actual.is_nan() || expected.is_nan() {
        panic!(
            "{}: one value is NaN (expected {:.6}, got {:.6})",
            msg, expected, actual
        );
    }
    let diff = (actual - expected).abs();
    let rel_diff = if expected.abs() > 1e-10 {
        diff / expected.abs()
    } else {
        diff
    };
    assert!(
        rel_diff < tolerance,
        "{}: expected {:.6}, got {:.6}, rel_diff={:.6}",
        msg,
        expected,
        actual,
        rel_diff
    );
}

// ============================================================================
// 1D Element-wise Operation Tests
// ============================================================================

#[test]
fn test_add_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);

        // CPU
        let cpu_a = CpuBackend::from_vec_1d(data.clone());
        let cpu_b = CpuBackend::from_vec_1d(data.iter().map(|x| x + 0.5).collect());
        let cpu_result = CpuBackend::add_1d(&cpu_a, &cpu_b);

        // WGPU
        let wgpu_a = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_b = WgpuBackend::from_vec_1d(data.iter().map(|x| x + 0.5).collect());
        let wgpu_result = WgpuBackend::add_1d(&wgpu_a, &wgpu_b);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("add_1d size {}", size));
    }
}

#[test]
fn test_sub_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);

        let cpu_a = CpuBackend::from_vec_1d(data.clone());
        let cpu_b = CpuBackend::from_vec_1d(data.iter().map(|x| x * 0.5).collect());
        let cpu_result = CpuBackend::sub_1d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_b = WgpuBackend::from_vec_1d(data.iter().map(|x| x * 0.5).collect());
        let wgpu_result = WgpuBackend::sub_1d(&wgpu_a, &wgpu_b);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("sub_1d size {}", size));
    }
}

#[test]
fn test_mul_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);

        let cpu_a = CpuBackend::from_vec_1d(data.clone());
        let cpu_b = CpuBackend::from_vec_1d(data.iter().map(|x| x + 0.1).collect());
        let cpu_result = CpuBackend::mul_1d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_b = WgpuBackend::from_vec_1d(data.iter().map(|x| x + 0.1).collect());
        let wgpu_result = WgpuBackend::mul_1d(&wgpu_a, &wgpu_b);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("mul_1d size {}", size));
    }
}

#[test]
fn test_div_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);

        let cpu_a = CpuBackend::from_vec_1d(data.clone());
        let cpu_b = CpuBackend::from_vec_1d(data.iter().map(|x| x + 0.5).collect()); // avoid division by zero
        let cpu_result = CpuBackend::div_1d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_b = WgpuBackend::from_vec_1d(data.iter().map(|x| x + 0.5).collect());
        let wgpu_result = WgpuBackend::div_1d(&wgpu_a, &wgpu_b);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("div_1d size {}", size));
    }
}

// ============================================================================
// 1D Scalar Operation Tests
// ============================================================================

#[test]
fn test_mul_scalar_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);
        let scalar = 2.5;

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::mul_scalar_1d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::mul_scalar_1d(&wgpu_t, &wgpu_s);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("mul_scalar_1d size {}", size),
        );
    }
}

#[test]
fn test_add_scalar_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);
        let scalar = 1.5;

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::add_scalar_1d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::add_scalar_1d(&wgpu_t, &wgpu_s);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("add_scalar_1d size {}", size),
        );
    }
}

#[test]
fn test_sub_scalar_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);
        let scalar = 0.5;

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::sub_scalar_1d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::sub_scalar_1d(&wgpu_t, &wgpu_s);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("sub_scalar_1d size {}", size),
        );
    }
}

#[test]
fn test_div_scalar_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);
        let scalar = 3.0;

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::div_scalar_1d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_1d(data.clone());
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::div_scalar_1d(&wgpu_t, &wgpu_s);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("div_scalar_1d size {}", size),
        );
    }
}

// ============================================================================
// 2D Element-wise Operation Tests
// ============================================================================

#[test]
fn test_add_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_a = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_b = CpuBackend::from_vec_2d(data.iter().map(|x| x + 0.3).collect(), rows, cols);
        let cpu_result = CpuBackend::add_2d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_b = WgpuBackend::from_vec_2d(data.iter().map(|x| x + 0.3).collect(), rows, cols);
        let wgpu_result = WgpuBackend::add_2d(&wgpu_a, &wgpu_b);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("add_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_sub_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_a = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_b = CpuBackend::from_vec_2d(data.iter().map(|x| x * 0.5).collect(), rows, cols);
        let cpu_result = CpuBackend::sub_2d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_b = WgpuBackend::from_vec_2d(data.iter().map(|x| x * 0.5).collect(), rows, cols);
        let wgpu_result = WgpuBackend::sub_2d(&wgpu_a, &wgpu_b);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("sub_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_mul_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_a = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_b = CpuBackend::from_vec_2d(data.iter().map(|x| x + 0.1).collect(), rows, cols);
        let cpu_result = CpuBackend::mul_2d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_b = WgpuBackend::from_vec_2d(data.iter().map(|x| x + 0.1).collect(), rows, cols);
        let wgpu_result = WgpuBackend::mul_2d(&wgpu_a, &wgpu_b);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("mul_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_div_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_a = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_b = CpuBackend::from_vec_2d(data.iter().map(|x| x + 0.5).collect(), rows, cols);
        let cpu_result = CpuBackend::div_2d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_b = WgpuBackend::from_vec_2d(data.iter().map(|x| x + 0.5).collect(), rows, cols);
        let wgpu_result = WgpuBackend::div_2d(&wgpu_a, &wgpu_b);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("div_2d size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// 2D Scalar Operation Tests
// ============================================================================

#[test]
fn test_mul_scalar_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);
        let scalar = 2.0;

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::mul_scalar_2d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::mul_scalar_2d(&wgpu_t, &wgpu_s);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("mul_scalar_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_add_scalar_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);
        let scalar = 1.5;

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::add_scalar_2d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::add_scalar_2d(&wgpu_t, &wgpu_s);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("add_scalar_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_sub_scalar_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);
        let scalar = 0.5;

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::sub_scalar_2d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::sub_scalar_2d(&wgpu_t, &wgpu_s);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("sub_scalar_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_div_scalar_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);
        let scalar = 4.0;

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_s = CpuBackend::scalar_f64(scalar);
        let cpu_result = CpuBackend::div_scalar_2d(&cpu_t, &cpu_s);

        let wgpu_t = WgpuBackend::from_vec_2d(data.clone(), rows, cols);
        let wgpu_s = WgpuBackend::scalar_f64(scalar);
        let wgpu_result = WgpuBackend::div_scalar_2d(&wgpu_t, &wgpu_s);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("div_scalar_2d size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// Linear Algebra Operation Tests
// ============================================================================

#[test]
fn test_matvec() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let mat_data = generate_test_data_2d(rows, cols);
        let vec_data: Vec<f32> = (0..cols).map(|i| ((i as f32 % 10.0) + 1.0) / 5.0).collect();

        let cpu_mat = CpuBackend::from_vec_2d(mat_data.clone(), rows, cols);
        let cpu_vec = CpuBackend::from_vec_1d(vec_data.clone());
        let cpu_result = CpuBackend::matvec(&cpu_mat, &cpu_vec);

        let wgpu_mat = WgpuBackend::from_vec_2d(mat_data, rows, cols);
        let wgpu_vec = WgpuBackend::from_vec_1d(vec_data);
        let wgpu_result = WgpuBackend::matvec(&wgpu_mat, &wgpu_vec);

        let cpu_result_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_result_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_result_vec,
            &cpu_result_vec,
            1e-3,
            &format!("matvec size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_matvec_transposed() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let mat_data = generate_test_data_2d(rows, cols);
        let vec_data: Vec<f32> = (0..rows).map(|i| ((i as f32 % 10.0) + 1.0) / 5.0).collect();

        let cpu_mat = CpuBackend::from_vec_2d(mat_data.clone(), rows, cols);
        let cpu_vec = CpuBackend::from_vec_1d(vec_data.clone());
        let cpu_result = CpuBackend::matvec_transposed(&cpu_mat, &cpu_vec);

        let wgpu_mat = WgpuBackend::from_vec_2d(mat_data, rows, cols);
        let wgpu_vec = WgpuBackend::from_vec_1d(vec_data);
        let wgpu_result = WgpuBackend::matvec_transposed(&wgpu_mat, &wgpu_vec);

        let cpu_result_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_result_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_result_vec,
            &cpu_result_vec,
            1e-3,
            &format!("matvec_transposed size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_matmul() {
    if !gpu_available() {
        return;
    }
    // Test (m, k) @ (k, n) -> (m, n)
    let test_cases = vec![(2, 2, 2), (4, 4, 4), (8, 16, 8), (16, 8, 16)];

    for (m, k, n) in test_cases {
        let a_data = generate_test_data_2d(m, k);
        let b_data = generate_test_data_2d(k, n);

        let cpu_a = CpuBackend::from_vec_2d(a_data.clone(), m, k);
        let cpu_b = CpuBackend::from_vec_2d(b_data.clone(), k, n);
        let cpu_result = CpuBackend::matmul(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_2d(a_data, m, k);
        let wgpu_b = WgpuBackend::from_vec_2d(b_data, k, n);
        let wgpu_result = WgpuBackend::matmul(&wgpu_a, &wgpu_b);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("matmul size {}x{} @ {}x{}", m, k, k, n),
        );
    }
}

#[test]
fn test_transpose() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_mat = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::transpose(&cpu_mat);

        let wgpu_mat = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::transpose(&wgpu_mat);

        // Check shape
        let (cpu_rows, cpu_cols) = CpuBackend::shape(&cpu_result);
        let (wgpu_rows, wgpu_cols) = WgpuBackend::shape(&wgpu_result);
        assert_eq!((cpu_rows, cpu_cols), (wgpu_rows, wgpu_cols));
        assert_eq!((cpu_rows, cpu_cols), (cols, rows));

        // Check values
        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        // Transpose should be exact (no FP variance)
        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-10,
            &format!("transpose size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// Reduction Operation Tests
// ============================================================================

#[test]
fn test_sum_all_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::sum_all_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::sum_all_1d(&wgpu_t);

        assert_close_scalar(
            wgpu_result,
            cpu_result,
            1e-3,
            &format!("sum_all_1d size {}", size),
        );
    }
}

#[test]
fn test_sum_all_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::sum_all_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::sum_all_2d(&wgpu_t);

        assert_close_scalar(
            wgpu_result,
            cpu_result,
            1e-3,
            &format!("sum_all_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_mean_all_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data = generate_test_data_1d(size);

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::mean_all_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::mean_all_1d(&wgpu_t);

        assert_close_scalar(
            wgpu_result,
            cpu_result,
            1e-3,
            &format!("mean_all_1d size {}", size),
        );
    }
}

#[test]
fn test_mean_all_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::mean_all_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::mean_all_2d(&wgpu_t);

        assert_close_scalar(
            wgpu_result,
            cpu_result,
            1e-3,
            &format!("mean_all_2d size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// Unary Math Operation Tests
// ============================================================================

#[test]
fn test_abs_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data: Vec<f32> = generate_test_data_1d(size)
            .iter()
            .map(|x| x - 0.5)
            .collect(); // Include negatives

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::abs_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::abs_1d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("abs_1d size {}", size));
    }
}

#[test]
fn test_abs_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data: Vec<f32> = generate_test_data_2d(rows, cols)
            .iter()
            .map(|x| x - 0.5)
            .collect();

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::abs_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::abs_2d(&wgpu_t);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("abs_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_sign_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data: Vec<f32> = generate_test_data_1d(size)
            .iter()
            .map(|x| x - 0.5)
            .collect();

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::sign_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::sign_1d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("sign_1d size {}", size));
    }
}

#[test]
fn test_sign_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data: Vec<f32> = generate_test_data_2d(rows, cols)
            .iter()
            .map(|x| x - 0.5)
            .collect();

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::sign_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::sign_2d(&wgpu_t);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("sign_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_exp_1d() {
    if !gpu_available() {
        return;
    }
    for size in vec![1, 4, 16, 64] {
        let data: Vec<f32> = generate_test_data_1d(size)
            .iter()
            .map(|x| x * 0.1)
            .collect(); // Small values to avoid overflow

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::exp_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::exp_1d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-3, &format!("exp_1d size {}", size));
    }
}

#[test]
fn test_exp_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in vec![(2, 2), (4, 4), (8, 8)] {
        let data: Vec<f32> = generate_test_data_2d(rows, cols)
            .iter()
            .map(|x| x * 0.1)
            .collect();

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::exp_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::exp_2d(&wgpu_t);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("exp_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_log_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data: Vec<f32> = generate_test_data_1d(size)
            .iter()
            .map(|x| x + 0.5)
            .collect(); // Ensure positive

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::log_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::log_1d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-3, &format!("log_1d size {}", size));
    }
}

#[test]
fn test_log_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data: Vec<f32> = generate_test_data_2d(rows, cols)
            .iter()
            .map(|x| x + 0.5)
            .collect();

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::log_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::log_2d(&wgpu_t);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("log_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_sigmoid_1d() {
    if !gpu_available() {
        return;
    }
    for size in vec![1, 4, 16, 64] {
        let data: Vec<f32> = generate_test_data_1d(size);

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::sigmoid_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::sigmoid_1d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("sigmoid_1d size {}", size),
        );
    }
}

#[test]
fn test_sigmoid_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in vec![(2, 2), (4, 4), (8, 8)] {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::sigmoid_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::sigmoid_2d(&wgpu_t);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("sigmoid_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_sqrt_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data: Vec<f32> = generate_test_data_1d(size)
            .iter()
            .map(|x| x + 0.5)
            .collect(); // Ensure positive

        let cpu_t = CpuBackend::from_vec_1d(data.clone());
        let cpu_result = CpuBackend::sqrt_1d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_1d(data);
        let wgpu_result = WgpuBackend::sqrt_1d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, &format!("sqrt_1d size {}", size));
    }
}

#[test]
fn test_sqrt_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data: Vec<f32> = generate_test_data_2d(rows, cols)
            .iter()
            .map(|x| x + 0.5)
            .collect();

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::sqrt_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::sqrt_2d(&wgpu_t);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("sqrt_2d size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// Column/Row Operation Tests
// ============================================================================

#[test]
fn test_col_mean_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::col_mean_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::col_mean_2d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("col_mean_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_col_std_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::col_std_2d(&cpu_t, 1);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::col_std_2d(&wgpu_t, 1);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("col_std_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_col_sum_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::col_sum_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::col_sum_2d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("col_sum_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_col_min_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::col_min_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::col_min_2d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("col_min_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_col_max_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::col_max_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::col_max_2d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("col_max_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_row_sum_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::row_sum_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::row_sum_2d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-3,
            &format!("row_sum_2d size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// Broadcasting Operation Tests
// ============================================================================

#[test]
fn test_broadcast_add_1d_to_2d_rows() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let mat_data = generate_test_data_2d(rows, cols);
        let vec_data: Vec<f32> = (0..cols).map(|i| ((i as f32 % 5.0) + 1.0) / 3.0).collect();

        let cpu_mat = CpuBackend::from_vec_2d(mat_data.clone(), rows, cols);
        let cpu_vec = CpuBackend::from_vec_1d(vec_data.clone());
        let cpu_result = CpuBackend::broadcast_add_1d_to_2d_rows(&cpu_mat, &cpu_vec);

        let wgpu_mat = WgpuBackend::from_vec_2d(mat_data, rows, cols);
        let wgpu_vec = WgpuBackend::from_vec_1d(vec_data);
        let wgpu_result = WgpuBackend::broadcast_add_1d_to_2d_rows(&wgpu_mat, &wgpu_vec);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec_result = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec_result = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec_result,
            &cpu_vec_result,
            1e-4,
            &format!("broadcast_add size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_broadcast_sub_1d_to_2d_rows() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let mat_data = generate_test_data_2d(rows, cols);
        let vec_data: Vec<f32> = (0..cols).map(|i| ((i as f32 % 5.0) + 1.0) / 5.0).collect();

        let cpu_mat = CpuBackend::from_vec_2d(mat_data.clone(), rows, cols);
        let cpu_vec = CpuBackend::from_vec_1d(vec_data.clone());
        let cpu_result = CpuBackend::broadcast_sub_1d_to_2d_rows(&cpu_mat, &cpu_vec);

        let wgpu_mat = WgpuBackend::from_vec_2d(mat_data, rows, cols);
        let wgpu_vec = WgpuBackend::from_vec_1d(vec_data);
        let wgpu_result = WgpuBackend::broadcast_sub_1d_to_2d_rows(&wgpu_mat, &wgpu_vec);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec_result = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec_result = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec_result,
            &cpu_vec_result,
            1e-4,
            &format!("broadcast_sub size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_broadcast_mul_1d_to_2d_rows() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let mat_data = generate_test_data_2d(rows, cols);
        let vec_data: Vec<f32> = (0..cols).map(|i| ((i as f32 % 5.0) + 1.0) / 3.0).collect();

        let cpu_mat = CpuBackend::from_vec_2d(mat_data.clone(), rows, cols);
        let cpu_vec = CpuBackend::from_vec_1d(vec_data.clone());
        let cpu_result = CpuBackend::broadcast_mul_1d_to_2d_rows(&cpu_mat, &cpu_vec);

        let wgpu_mat = WgpuBackend::from_vec_2d(mat_data, rows, cols);
        let wgpu_vec = WgpuBackend::from_vec_1d(vec_data);
        let wgpu_result = WgpuBackend::broadcast_mul_1d_to_2d_rows(&wgpu_mat, &wgpu_vec);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec_result = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec_result = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec_result,
            &cpu_vec_result,
            1e-4,
            &format!("broadcast_mul size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_broadcast_div_1d_to_2d_rows() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let mat_data = generate_test_data_2d(rows, cols);
        let vec_data: Vec<f32> = (0..cols).map(|i| ((i as f32 % 5.0) + 1.0) / 2.0).collect(); // Ensure non-zero

        let cpu_mat = CpuBackend::from_vec_2d(mat_data.clone(), rows, cols);
        let cpu_vec = CpuBackend::from_vec_1d(vec_data.clone());
        let cpu_result = CpuBackend::broadcast_div_1d_to_2d_rows(&cpu_mat, &cpu_vec);

        let wgpu_mat = WgpuBackend::from_vec_2d(mat_data, rows, cols);
        let wgpu_vec = WgpuBackend::from_vec_1d(vec_data);
        let wgpu_result = WgpuBackend::broadcast_div_1d_to_2d_rows(&wgpu_mat, &wgpu_vec);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec_result = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec_result = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec_result,
            &cpu_vec_result,
            1e-4,
            &format!("broadcast_div size {}x{}", rows, cols),
        );
    }
}

// ============================================================================
// Tensor Manipulation Operation Tests
// ============================================================================

#[test]
fn test_ravel_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data = generate_test_data_2d(rows, cols);

        let cpu_t = CpuBackend::from_vec_2d(data.clone(), rows, cols);
        let cpu_result = CpuBackend::ravel_2d(&cpu_t);

        let wgpu_t = WgpuBackend::from_vec_2d(data, rows, cols);
        let wgpu_result = WgpuBackend::ravel_2d(&wgpu_t);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        // Ravel should be exact
        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-10,
            &format!("ravel_2d size {}x{}", rows, cols),
        );
    }
}

#[test]
fn test_hcat_2d() {
    if !gpu_available() {
        return;
    }
    // Test concatenating matrices horizontally
    let data_a = generate_test_data_2d(4, 2);
    let data_b = generate_test_data_2d(4, 3);

    let cpu_a = CpuBackend::from_vec_2d(data_a.clone(), 4, 2);
    let cpu_b = CpuBackend::from_vec_2d(data_b.clone(), 4, 3);
    let cpu_result = CpuBackend::hcat_2d(&[cpu_a, cpu_b]).unwrap();

    let wgpu_a = WgpuBackend::from_vec_2d(data_a, 4, 2);
    let wgpu_b = WgpuBackend::from_vec_2d(data_b, 4, 3);
    let wgpu_result = WgpuBackend::hcat_2d(&[wgpu_a, wgpu_b]).unwrap();

    // Check shape
    let (cpu_rows, cpu_cols) = CpuBackend::shape(&cpu_result);
    let (wgpu_rows, wgpu_cols) = WgpuBackend::shape(&wgpu_result);
    assert_eq!((cpu_rows, cpu_cols), (wgpu_rows, wgpu_cols));
    assert_eq!((cpu_rows, cpu_cols), (4, 5));

    let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
    let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

    let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
    let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

    assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, "hcat_2d");
}

#[test]
fn test_select_columns_2d() {
    if !gpu_available() {
        return;
    }
    let data = generate_test_data_2d(4, 6);
    let columns = vec![0, 2, 5];

    let cpu_t = CpuBackend::from_vec_2d(data.clone(), 4, 6);
    let cpu_result = CpuBackend::select_columns_2d(&cpu_t, &columns);

    let wgpu_t = WgpuBackend::from_vec_2d(data, 4, 6);
    let wgpu_result = WgpuBackend::select_columns_2d(&wgpu_t, &columns);

    // Check shape
    let (cpu_rows, cpu_cols) = CpuBackend::shape(&cpu_result);
    let (wgpu_rows, wgpu_cols) = WgpuBackend::shape(&wgpu_result);
    assert_eq!((cpu_rows, cpu_cols), (wgpu_rows, wgpu_cols));
    assert_eq!((cpu_rows, cpu_cols), (4, 3));

    let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
    let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

    let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
    let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

    assert_close_1d(&wgpu_vec, &cpu_vec, 1e-4, "select_columns_2d");
}

#[test]
fn test_one_hot_from_indices() {
    if !gpu_available() {
        return;
    }
    let indices_data: Vec<f32> = vec![0.0, 1.0, 2.0, 0.0, 1.0];
    let num_classes = 3;

    let cpu_indices = CpuBackend::from_vec_1d(indices_data.clone());
    let cpu_result = CpuBackend::one_hot_from_indices(&cpu_indices, num_classes);

    let wgpu_indices = WgpuBackend::from_vec_1d(indices_data);
    let wgpu_result = WgpuBackend::one_hot_from_indices(&wgpu_indices, num_classes);

    // Check shape
    let (cpu_rows, cpu_cols) = CpuBackend::shape(&cpu_result);
    let (wgpu_rows, wgpu_cols) = WgpuBackend::shape(&wgpu_result);
    assert_eq!((cpu_rows, cpu_cols), (wgpu_rows, wgpu_cols));
    assert_eq!((cpu_rows, cpu_cols), (5, 3));

    let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
    let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

    let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
    let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

    // One-hot should be exact (0 or 1)
    assert_close_1d(&wgpu_vec, &cpu_vec, 1e-10, "one_hot_from_indices");
}

// ============================================================================
// Maximum Operation Tests
// ============================================================================

#[test]
fn test_maximum_1d() {
    if !gpu_available() {
        return;
    }
    for size in test_sizes() {
        let data_a = generate_test_data_1d(size);
        let data_b: Vec<f32> = data_a.iter().map(|x| x + 0.1).collect();

        let cpu_a = CpuBackend::from_vec_1d(data_a.clone());
        let cpu_b = CpuBackend::from_vec_1d(data_b.clone());
        let cpu_result = CpuBackend::maximum_1d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_1d(data_a);
        let wgpu_b = WgpuBackend::from_vec_1d(data_b);
        let wgpu_result = WgpuBackend::maximum_1d(&wgpu_a, &wgpu_b);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_result);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_result);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("maximum_1d size {}", size),
        );
    }
}

#[test]
fn test_maximum_2d() {
    if !gpu_available() {
        return;
    }
    for (rows, cols) in test_sizes_2d() {
        let data_a = generate_test_data_2d(rows, cols);
        let data_b: Vec<f32> = data_a.iter().map(|x| x + 0.1).collect();

        let cpu_a = CpuBackend::from_vec_2d(data_a.clone(), rows, cols);
        let cpu_b = CpuBackend::from_vec_2d(data_b.clone(), rows, cols);
        let cpu_result = CpuBackend::maximum_2d(&cpu_a, &cpu_b);

        let wgpu_a = WgpuBackend::from_vec_2d(data_a, rows, cols);
        let wgpu_b = WgpuBackend::from_vec_2d(data_b, rows, cols);
        let wgpu_result = WgpuBackend::maximum_2d(&wgpu_a, &wgpu_b);

        let cpu_flat = CpuBackend::ravel_2d(&cpu_result);
        let wgpu_flat = WgpuBackend::ravel_2d(&wgpu_result);

        let cpu_vec = CpuBackend::to_vec_1d(&cpu_flat);
        let wgpu_vec = WgpuBackend::to_vec_1d(&wgpu_flat);

        assert_close_1d(
            &wgpu_vec,
            &cpu_vec,
            1e-4,
            &format!("maximum_2d size {}x{}", rows, cols),
        );
    }
}
