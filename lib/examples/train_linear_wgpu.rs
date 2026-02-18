//! WGPU Backend tensor operations demonstration.
//!
//! This example demonstrates basic tensor operations on the GPU using WGPU
//! compute shaders.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example train_linear_wgpu --features wgpu
//! ```
//!
//! ## Features
//!
//! - Demonstrates GPU tensor creation and basic operations
//! - Cross-platform: Vulkan (Linux/Windows), Metal (macOS), D3D12 (Windows), WebGPU (browsers)
//!
//! ## Note
//!
//! This example demonstrates individual tensor operations on GPU.
//! Full training pipeline with gradient descent requires additional
//! buffer synchronization fixes in the WGPU backend.

#[cfg(feature = "wgpu")]
use machinelearne_rs::{
    backend::{tensorlike::TensorLike, Scalar, WgpuBackend},
    Tensor1D, Tensor2D,
};

#[cfg(feature = "wgpu")]
fn main() {
    println!("=== WGPU Backend Tensor Operations Demo ===\n");

    // Demo 1: Create 1D tensors on GPU
    println!("1. Creating 1D tensors on GPU...");
    let a: Tensor1D<WgpuBackend> = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b: Tensor1D<WgpuBackend> = Tensor1D::new(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    println!("   Created two 1D tensors of length {}", a.len());

    // Demo 2: Element-wise operations using tensor methods
    println!("\n2. Element-wise operations...");
    let sum = a.add(&b);
    println!("   a + b = {:?}", sum.to_vec());

    let product = a.mul(&b);
    println!("   a * b = {:?}", product.to_vec());

    // Demo 3: Scalar operations
    println!("\n3. Scalar operations...");
    let scalar = Scalar::<WgpuBackend>::new(2.0);
    let scaled = a.scale(&scalar);
    println!("   a * 2.0 = {:?}", scaled.to_vec());

    // Demo 4: Reductions
    println!("\n4. Reductions...");
    let sum_val = a.sum();
    let mean_val = a.mean();
    println!("   sum(a) = {:?}", sum_val);
    println!("   mean(a) = {:?}", mean_val);

    // Demo 5: 2D tensor operations
    println!("\n5. 2D tensor operations...");
    let matrix: Tensor2D<WgpuBackend> = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let (rows, cols) = matrix.shape();
    println!("   Created a {}x{} matrix", rows, cols);

    // Demo 6: Matrix-vector multiplication
    println!("\n6. Matrix-vector multiplication...");
    let vec: Tensor1D<WgpuBackend> = Tensor1D::new(vec![1.0, 2.0, 3.0]);
    let result = matrix.dot(&vec);
    println!("   matrix @ vec = {:?}", result.to_vec());

    println!("\n=== All GPU operations completed successfully! ===");
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("This example requires the 'wgpu' feature to be enabled.");
    println!("Run with: cargo run --example train_linear_wgpu --features wgpu");
}
