//! MLP Performance Benchmark: CPU vs WGPU
//!
//! This benchmark compares training and inference performance between
//! CPU and WGPU backends for MLP models.
//!
//! ## Results Summary (your machine may vary)
//!
//! Training performance depends heavily on:
//! - Dataset size and batch size
//! - Model architecture depth and width
//! - GPU/CPU capabilities
//!
//! Note: WGPU backend currently has training overhead due to GPU-CPU
//! synchronization for loss computation. This is expected to improve
//! with future optimizations.

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, TrainableModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

#[cfg(feature = "wgpu")]
use machinelearne_rs::{backend::WgpuBackend, optimizer::Optimizer};

/// Generate synthetic regression dataset
fn generate_dataset(n_samples: usize, n_features: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    use std::f64::consts::PI;

    let mut features = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut sample = Vec::with_capacity(n_features);
        for j in 0..n_features {
            // Create diverse feature patterns
            let val = ((i * (j + 1)) as f64 * PI / 100.0).sin() as f32;
            sample.push(val);
        }
        features.push(sample);

        // Target: non-linear combination of features
        let first_feature = features[i][0] as f64;
        let target = if n_features > 0 {
            features[i]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                / n_features as f64
                + first_feature.sin()
        } else {
            0.0
        };
        targets.push(target as f32);
    }

    (features, targets)
}

/// Benchmark training on CPU backend
fn benchmark_cpu_training(
    x: &[Vec<f32>],
    y: &[f32],
    architecture: &[usize],
    activations: &[Activation],
    epochs: usize,
    learning_rate: f64,
) -> (std::time::Duration, f32) {
    let dataset = InMemoryDataset::new(x.to_vec(), y.to_vec()).unwrap();
    let model = MLP::<CpuBackend>::new(architecture, activations);

    // Note: batch_size=1 is required for proper gradient computation in MLP
    let trainer = Trainer::builder(MSELoss, SGD::new(learning_rate), NoRegularizer)
        .batch_size(1)
        .max_epochs(epochs)
        .verbose(false)
        .build();

    let start = std::time::Instant::now();
    let fitted = trainer.fit(model, &dataset).unwrap();
    let duration = start.elapsed();

    // Compute final loss
    let mut total_loss = 0.0f32;
    for (inputs, target) in x.iter().zip(y.iter()) {
        let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
            inputs.iter().copied().collect(),
        );
        let pred = fitted.predict(&input_1d);
        let pred_val = pred.to_vec()[0] as f32;
        total_loss += (pred_val - target).powi(2);
    }
    let mse = total_loss / x.len() as f32;

    (duration, mse)
}

/// Benchmark inference on CPU backend
fn benchmark_cpu_inference(
    x: &[Vec<f32>],
    architecture: &[usize],
    activations: &[Activation],
    n_iterations: usize,
) -> std::time::Duration {
    // Create and fit a model first
    let model = MLP::<CpuBackend>::new(architecture, activations);
    let fitted = model.into_fitted();

    let start = std::time::Instant::now();

    for _ in 0..n_iterations {
        for inputs in x {
            let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
                inputs.iter().copied().collect(),
            );
            let _ = fitted.predict(&input_1d);
        }
    }

    start.elapsed()
}

#[cfg(feature = "wgpu")]
/// Benchmark training on WGPU backend
fn benchmark_wgpu_training(
    x: &[Vec<f32>],
    y: &[f32],
    architecture: &[usize],
    activations: &[Activation],
    epochs: usize,
    learning_rate: f64,
) -> (std::time::Duration, f32) {
    use machinelearne_rs::backend::{Tensor1D, Tensor2D};

    let n_features = architecture[0];
    let n_samples = x.len();

    // Create model
    let model = MLP::<WgpuBackend>::new(architecture, activations);
    let mut model = model;

    let optimizer = SGD::new(learning_rate);

    // Manual training loop with batch_size=1 (MLP limitation)
    let start = std::time::Instant::now();

    for _epoch in 0..epochs {
        for sample_idx in 0..n_samples {
            // Single sample as batch
            let input = Tensor2D::<WgpuBackend>::new(x[sample_idx].clone(), 1, n_features);
            let target = Tensor1D::<WgpuBackend>::new(vec![y[sample_idx]]);

            // Forward pass
            let prediction = model.forward(&input);

            // Compute loss gradient (MSE derivative: 2 * (pred - target) / n)
            // For single sample: grad = pred - target
            let grad_output = prediction.sub(&target);

            // Backward pass
            let gradients = model.backward(&input, &grad_output);

            // Update parameters
            let new_params = optimizer.step(model.params(), &gradients);
            model.update_params(&new_params);
        }
    }

    let duration = start.elapsed();

    // Compute final loss
    let mut total_loss = 0.0f32;
    let fitted = model.into_fitted();
    for (inputs, target) in x.iter().zip(y.iter()) {
        let input_1d = Tensor1D::<WgpuBackend>::new(inputs.iter().copied().collect());
        let pred = fitted.predict(&input_1d);
        let pred_val = pred.to_vec()[0] as f32;
        total_loss += (pred_val - target).powi(2);
    }
    let mse = total_loss / x.len() as f32;

    (duration, mse)
}

#[cfg(feature = "wgpu")]
/// Benchmark inference on WGPU backend
fn benchmark_wgpu_inference(
    x: &[Vec<f32>],
    architecture: &[usize],
    activations: &[Activation],
    n_iterations: usize,
) -> std::time::Duration {
    use machinelearne_rs::backend::Tensor1D;

    let model = MLP::<WgpuBackend>::new(architecture, activations);
    let fitted = model.into_fitted();

    let start = std::time::Instant::now();

    for _ in 0..n_iterations {
        for inputs in x {
            let input_1d = Tensor1D::<WgpuBackend>::new(inputs.iter().copied().collect());
            let _ = fitted.predict(&input_1d);
        }
    }

    start.elapsed()
}

fn main() {
    println!("=== MLP Performance Benchmark: CPU vs WGPU ===\n");

    // Dataset configuration - smaller for faster benchmark
    let n_samples = 50;
    let n_features = 8;
    let (x, y) = generate_dataset(n_samples, n_features);
    println!("Dataset: {} samples, {} features", n_samples, n_features);

    // Training configuration - reduced epochs for faster benchmark
    let epochs = 50;
    let learning_rate = 0.1_f64;

    // Model architectures to test
    let architectures = [
        (
            "Small",
            vec![n_features, 16, 1],
            vec![Activation::ReLU, Activation::Identity],
        ),
        (
            "Medium",
            vec![n_features, 32, 16, 1],
            vec![Activation::ReLU, Activation::ReLU, Activation::Identity],
        ),
    ];

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", epochs);
    println!("  Batch size: 1 (MLP gradient limitation)");
    println!("  Learning rate: {}", learning_rate);

    // ========================================
    // CPU Benchmarks
    // ========================================
    println!("\n{}", "=".repeat(60));
    println!("CPU Backend Benchmarks");
    println!("{}", "=".repeat(60));

    for (name, arch, acts) in &architectures {
        println!("\n--- Architecture: {} ({:?}) ---", name, arch);

        let (train_time, train_mse) =
            benchmark_cpu_training(&x, &y, arch, acts, epochs, learning_rate);
        println!("Training: {:?} (MSE: {:.6})", train_time, train_mse);

        let inference_time = benchmark_cpu_inference(&x, arch, acts, 10);
        let samples_per_sec = (x.len() * 10) as f64 / inference_time.as_secs_f64();
        println!("Inference (10 iterations): {:?}", inference_time);
        println!("Inference throughput: {:.0} samples/sec", samples_per_sec);
    }

    // ========================================
    // WGPU Benchmarks
    // ========================================
    #[cfg(feature = "wgpu")]
    {
        println!("\n{}", "=".repeat(60));
        println!("WGPU Backend Benchmarks");
        println!("{}", "=".repeat(60));
        println!("\nNote: WGPU training has overhead from GPU-CPU sync per batch.");
        println!("GPU acceleration is most beneficial for very large batches/models.\n");

        for (name, arch, acts) in &architectures {
            println!("\n--- Architecture: {} ({:?}) ---", name, arch);

            let (train_time, train_mse) =
                benchmark_wgpu_training(&x, &y, arch, acts, epochs, learning_rate);
            println!("Training: {:?} (MSE: {:.6})", train_time, train_mse);

            let inference_time = benchmark_wgpu_inference(&x, arch, acts, 10);
            let samples_per_sec = (x.len() * 10) as f64 / inference_time.as_secs_f64();
            println!("Inference (10 iterations): {:?}", inference_time);
            println!("Inference throughput: {:.0} samples/sec", samples_per_sec);
        }
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("\n{}", "=".repeat(60));
        println!("WGPU Backend: Not available (compile with --features wgpu)");
        println!("{}", "=".repeat(60));
    }

    // ========================================
    // Summary
    // ========================================
    println!("\n{}", "=".repeat(60));
    println!("Summary");
    println!("{}", "=".repeat(60));
    println!("\nFor typical ML workloads on this dataset size:");
    println!("  - CPU backend is efficient for training and inference");
    println!("  - WGPU backend benefits emerge with:");
    println!("    * Larger batch sizes (1000+)");
    println!("    * Larger models (100+ neurons per layer)");
    println!("    * Inference on pre-trained models (no sync overhead)");
    println!("\nTo enable WGPU benchmarks, compile with: --features wgpu");
}
