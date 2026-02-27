//! CPU vs GPU Training Comparison Example.
//!
//! This example compares training performance between CPU and WGPU backends
//! for Linear Regression and MLP models on the California Housing dataset.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example cpu_gpu_training_comparison --features wgpu
//! ```
//!
//! ## Expected Results (based on ADR-0009)
//!
//! The WGPU backend is **NOT recommended for training** due to GPU-CPU sync overhead:
//! - Linear Regression: CPU ~1000x faster
//! - MLP: CPU ~100-200x faster
//!
//! Root cause: Loss computation requires CPU access, forcing GPU sync every epoch.
//! See ADR-0009 for detailed analysis.
//!
//! **WGPU is suitable for inference only** (single forward pass, minimal sync).
//!
//! ## Note
//!
//! Gradient Boosting is CPU-only because tree algorithms require iterating
//! over data to find optimal splits, which is not GPU-friendly.

#[cfg(feature = "wgpu")]
use machinelearne_rs::{
    backend::{CpuBackend, WgpuBackend, WgpuDevice},
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegression, Activation, InferenceModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor2D,
};

#[cfg(feature = "wgpu")]
use std::fs::File;
#[cfg(feature = "wgpu")]
use std::io::{BufRead, BufReader};

#[cfg(feature = "wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CPU vs GPU Training Comparison ===\n");

    // Display GPU info
    println!("GPU Adapter Information:");
    let adapters = pollster::block_on(WgpuDevice::enumerate_adapters());
    for (i, adapter) in adapters.iter().enumerate() {
        println!(
            "  [{}] {} ({}, {})",
            i, adapter.name, adapter.backend, adapter.device_type
        );
    }
    println!();

    // Load dataset
    let (x_raw, y_raw) = load_california_housing()?;
    println!(
        "Dataset: {} samples, {} features",
        x_raw.len(),
        x_raw[0].len()
    );

    // Normalize and shuffle
    let (x_scaled, y) = normalize_data(&x_raw, &y_raw);

    // Shuffle with fixed seed
    println!("Shuffling data with seed 42...");
    let mut indices: Vec<usize> = (0..x_scaled.len()).collect();
    let mut rng_state: u64 = 42;
    for i in (1..indices.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state % ((i + 1) as u64)) as usize;
        indices.swap(i, j);
    }

    let x_shuffled: Vec<Vec<f64>> = indices.iter().map(|&i| x_scaled[i].clone()).collect();
    let y_shuffled: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    // Split train/test
    let split_idx = (x_shuffled.len() as f64 * 0.8) as usize;
    let (x_train, x_test) = x_shuffled.split_at(split_idx);
    let (y_train, y_test) = y_shuffled.split_at(split_idx);
    println!("Train: {}, Test: {}\n", x_train.len(), x_test.len());

    let n_features = x_train[0].len();

    // ============================================
    // Linear Regression: CPU vs GPU
    // ============================================
    println!("{}", "=".repeat(60));
    println!("LINEAR REGRESSION: CPU vs GPU");
    println!("{}", "=".repeat(60));

    // CPU Linear
    let cpu_start = std::time::Instant::now();
    let cpu_dataset = InMemoryDataset::new(
        x_train
            .iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect(),
        y_train.iter().map(|&v| v as f32).collect(),
    )?;
    let cpu_linear = LinearRegression::<CpuBackend>::new(n_features);
    let cpu_trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
        .batch_size(64)
        .max_epochs(100)
        .verbose(false)
        .build();
    let cpu_linear_fitted = cpu_trainer.fit(cpu_linear, &cpu_dataset)?;
    let cpu_linear_time = cpu_start.elapsed();

    // Evaluate CPU Linear
    let test_features_cpu: Tensor2D<CpuBackend> = Tensor2D::new(
        x_test
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect(),
        x_test.len(),
        n_features,
    );
    let cpu_preds = cpu_linear_fitted.predict_batch(&test_features_cpu);
    let cpu_preds_vec: Vec<f64> = cpu_preds.to_vec().iter().map(|&v| v as f64).collect();
    let cpu_metrics = compute_metrics(&cpu_preds_vec, &y_test);

    println!("CPU Training time: {:?}", cpu_linear_time);
    println!(
        "CPU R² = {:.4}, MSE = {:.6}",
        cpu_metrics.r2, cpu_metrics.mse
    );

    // GPU Linear
    let gpu_start = std::time::Instant::now();
    let gpu_dataset = InMemoryDataset::new(
        x_train
            .iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect(),
        y_train.iter().map(|&v| v as f32).collect(),
    )?;
    let gpu_linear = LinearRegression::<WgpuBackend>::new(n_features);
    let gpu_trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
        .batch_size(64)
        .max_epochs(100)
        .verbose(false)
        .build();
    let gpu_linear_fitted = gpu_trainer.fit(gpu_linear, &gpu_dataset)?;
    let gpu_linear_time = gpu_start.elapsed();

    // Evaluate GPU Linear
    let test_features_gpu: Tensor2D<WgpuBackend> = Tensor2D::new(
        x_test
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect(),
        x_test.len(),
        n_features,
    );
    let gpu_preds = gpu_linear_fitted.predict_batch(&test_features_gpu);
    let gpu_preds_vec: Vec<f64> = gpu_preds.to_vec().iter().map(|&v| v as f64).collect();
    let gpu_metrics = compute_metrics(&gpu_preds_vec, &y_test);

    println!("GPU Training time: {:?}", gpu_linear_time);
    println!(
        "GPU R² = {:.4}, MSE = {:.6}",
        gpu_metrics.r2, gpu_metrics.mse
    );

    let linear_speedup = cpu_linear_time.as_secs_f64() / gpu_linear_time.as_secs_f64();
    println!(
        "\nSpeedup: {:.2}x ({})",
        linear_speedup,
        if linear_speedup > 1.0 {
            "GPU faster"
        } else {
            "CPU faster"
        }
    );

    // ============================================
    // GPU Batch Size Impact
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("GPU BATCH SIZE IMPACT (Linear Regression)");
    println!("{}", "=".repeat(60));
    println!("Testing if larger batches reduce sync overhead...\n");

    let train_size = x_train.len();
    for batch_size in [64, 256, 1024, 4096, train_size] {
        let gpu_dataset = InMemoryDataset::new(
            x_train
                .iter()
                .map(|row| row.iter().map(|&v| v as f32).collect())
                .collect(),
            y_train.iter().map(|&v| v as f32).collect(),
        )?;
        let gpu_model = LinearRegression::<WgpuBackend>::new(n_features);
        let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
            .batch_size(batch_size)
            .max_epochs(100)
            .verbose(false)
            .build();

        let start = std::time::Instant::now();
        let _fitted = trainer.fit(gpu_model, &gpu_dataset)?;
        let time = start.elapsed();

        let batches_per_epoch = (train_size + batch_size - 1) / batch_size;
        let speedup = cpu_linear_time.as_secs_f64() / time.as_secs_f64();
        println!(
            "batch_size={:5} | batches/epoch={:3} | time={:7?} | speedup vs CPU: {:6.2}x",
            batch_size, batches_per_epoch, time, speedup
        );
    }

    // ============================================
    // MLP: CPU vs GPU (with full batch)
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("MLP (8->32->16->1): CPU vs GPU (full batch)");
    println!("{}", "=".repeat(60));
    println!("(Using full batch to minimize GPU-CPU sync overhead)");

    // CPU MLP
    let cpu_start = std::time::Instant::now();
    let cpu_mlp = MLP::<CpuBackend>::new(
        &[n_features, 32, 16, 1],
        &[Activation::ReLU, Activation::ReLU, Activation::Identity],
    );
    let cpu_mlp_trainer = Trainer::builder(MSELoss, SGD::new(0.005), NoRegularizer)
        .batch_size(64)
        .max_epochs(200)
        .verbose(false)
        .build();
    let cpu_mlp_fitted = cpu_mlp_trainer.fit(cpu_mlp, &cpu_dataset)?;
    let cpu_mlp_time = cpu_start.elapsed();

    // GPU MLP with full batch (batch_size = train_size)
    println!("\nTesting GPU MLP with full batch...");
    let gpu_mlp = MLP::<WgpuBackend>::new(
        &[n_features, 32, 16, 1],
        &[Activation::ReLU, Activation::ReLU, Activation::Identity],
    );
    let gpu_mlp_trainer = Trainer::builder(MSELoss, SGD::new(0.005), NoRegularizer)
        .batch_size(train_size) // Full batch - 1 sync per epoch
        .max_epochs(200)
        .verbose(false)
        .build();

    let gpu_start = std::time::Instant::now();
    let gpu_mlp_fitted = gpu_mlp_trainer.fit(gpu_mlp, &gpu_dataset)?;
    let gpu_mlp_time = gpu_start.elapsed();

    // Evaluate CPU MLP
    let cpu_mlp_preds = cpu_mlp_fitted.predict_batch(&test_features_cpu);
    let cpu_mlp_flat = cpu_mlp_preds.ravel().to_vec();
    let cpu_mlp_preds_vec: Vec<f64> = (0..x_test.len()).map(|i| cpu_mlp_flat[i] as f64).collect();
    let cpu_mlp_metrics = compute_metrics(&cpu_mlp_preds_vec, &y_test);

    println!("CPU Training time: {:?}", cpu_mlp_time);
    println!(
        "CPU R² = {:.4}, MSE = {:.6}",
        cpu_mlp_metrics.r2, cpu_mlp_metrics.mse
    );

    // Evaluate GPU MLP
    let gpu_mlp_preds = gpu_mlp_fitted.predict_batch(&test_features_gpu);
    let gpu_mlp_flat = gpu_mlp_preds.ravel().to_vec();
    let gpu_mlp_preds_vec: Vec<f64> = (0..x_test.len()).map(|i| gpu_mlp_flat[i] as f64).collect();
    let gpu_mlp_metrics = compute_metrics(&gpu_mlp_preds_vec, &y_test);

    println!("GPU Training time: {:?}", gpu_mlp_time);
    println!(
        "GPU R² = {:.4}, MSE = {:.6}",
        gpu_mlp_metrics.r2, gpu_mlp_metrics.mse
    );

    let mlp_speedup = cpu_mlp_time.as_secs_f64() / gpu_mlp_time.as_secs_f64();

    // ============================================
    // Summary
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!(
        "{:20} {:>15} {:>15} {:>15}",
        "Model", "CPU Time", "GPU Time", "Speedup"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:20} {:>15?} {:>15?} {:>15.2}x",
        "Linear Regression", cpu_linear_time, gpu_linear_time, linear_speedup
    );
    println!(
        "{:20} {:>15?} {:>15?} {:>15.2}x",
        "MLP (full batch)", cpu_mlp_time, gpu_mlp_time, mlp_speedup
    );

    let conclusion = if mlp_speedup > 1.0 {
        format!(
            "GPU IS faster for MLP training with full batch! ({:.1}x speedup)",
            mlp_speedup
        )
    } else {
        format!(
            "GPU still slower ({:.1}x). Sync overhead dominates.",
            1.0 / mlp_speedup
        )
    };

    println!("\n{}", "=".repeat(60));
    println!("CONCLUSION");
    println!("{}", "=".repeat(60));
    println!("{}", conclusion);
    println!();
    println!("Key finding: Larger batches reduce GPU-CPU syncs per epoch.");
    println!("With full batch (1 sync/epoch), GPU becomes more competitive.");
    println!();
    println!("For production GPU training, consider:");
    println!("  1. CUDA/cuBLAS backends (designed for ML workloads)");
    println!("  2. Async training without per-epoch loss logging");
    println!("See ADR-0009 for detailed analysis.");

    Ok(())
}

#[cfg(feature = "wgpu")]
struct Metrics {
    mse: f64,
    r2: f64,
}

#[cfg(feature = "wgpu")]
fn compute_metrics(predictions: &[f64], targets: &[f64]) -> Metrics {
    let n = predictions.len();
    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f64>()
        / n as f64;

    let target_mean: f64 = targets.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = targets.iter().map(|&t| (t - target_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Metrics { mse, r2 }
}

#[cfg(feature = "wgpu")]
fn load_california_housing() -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
    let possible_paths = [
        "lib/examples/data/california_housing.csv",
        "examples/data/california_housing.csv",
        "data/california_housing.csv",
    ];

    let mut file = None;
    for path in &possible_paths {
        if let Ok(f) = File::open(path) {
            file = Some(f);
            break;
        }
    }

    let file = file.ok_or(
        "Could not find california_housing.csv. Please download it from:\n  https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/california_housing.csv"
    )?;
    let reader = BufReader::new(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue;
        }

        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();

        if values.len() >= 9 {
            x.push(values[0..8].to_vec());
            y.push(values[8]);
        }
    }

    Ok((x, y))
}

#[cfg(feature = "wgpu")]
fn normalize_data(x: &[Vec<f64>], y: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_samples = x.len();
    let n_features = x[0].len();

    let mut mean = vec![0.0; n_features];
    for sample in x {
        for (j, &val) in sample.iter().enumerate() {
            mean[j] += val;
        }
    }
    for m in &mut mean {
        *m /= n_samples as f64;
    }

    let mut std = vec![0.0; n_features];
    for sample in x {
        for (j, &val) in sample.iter().enumerate() {
            std[j] += (val - mean[j]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / n_samples as f64).sqrt();
        if *s < 1e-10 {
            *s = 1.0;
        }
    }

    let normalized: Vec<Vec<f64>> = x
        .iter()
        .map(|sample| {
            sample
                .iter()
                .enumerate()
                .map(|(j, &val)| (val - mean[j]) / std[j])
                .collect()
        })
        .collect();

    (normalized, y.to_vec())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("This example requires the 'wgpu' feature.");
    println!("Run with: cargo run --example cpu_gpu_training_comparison --features wgpu");
}
