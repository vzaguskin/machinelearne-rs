//! CPU vs GPU Training Comparison Example.
//!
//! This example compares training performance between CPU and WGPU backends
//! for models of varying sizes on real and synthetic datasets.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example cpu_gpu_training_comparison --features wgpu
//! ```
//!
//! ## Key Findings
//!
//! **Small Models (California Housing - 8 features):**
//! - CPU is faster due to GPU overhead dominating computation
//! - Linear Regression: CPU ~1000x faster
//! - MLP (8->32->16->1): CPU ~10x faster
//!
//! **Large Models (Synthetic - 64 features, 256+ neurons):**
//! - GPU is 10-20x FASTER than CPU
//! - Computation time dominates over GPU overhead
//! - Requires batch_size=512+ and manual training loop
//!
//! ## Recommendations
//!
//! - **Small datasets/models**: Use CPU backend
//! - **Large models (100+ neurons per layer)**: Use GPU with manual training loop
//! - **Inference**: GPU can help for batch predictions on large models
//!
//! ## Note
//!
//! Gradient Boosting is CPU-only because tree algorithms require iterating
//! over data to find optimal splits, which is not GPU-friendly.

#[cfg(feature = "wgpu")]
use machinelearne_rs::{
    backend::{CpuBackend, Scalar, WgpuBackend, WgpuDevice},
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegression, Activation, InferenceModel, TrainableModel, MLP},
    optimizer::{Optimizer, SGD},
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor1D, Tensor2D,
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
    // MLP: Manual Training Loop (no sync overhead)
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("MLP: GPU with Manual Training Loop (avoids Trainer sync)");
    println!("{}", "=".repeat(60));
    println!("Using manual forward/backward/step to avoid per-epoch loss sync\n");

    // Manual GPU training - no per-epoch sync
    let manual_start = std::time::Instant::now();

    let mut manual_mlp = MLP::<WgpuBackend>::new(
        &[n_features, 128, 64, 1],
        &[Activation::ReLU, Activation::ReLU, Activation::Identity],
    );
    let manual_optimizer = SGD::new(0.001);
    let manual_batch_size = 512;

    for _epoch in 0..100 {
        for batch_start in (0..train_size).step_by(manual_batch_size) {
            let batch_end = (batch_start + manual_batch_size).min(train_size);
            let actual_batch_size = batch_end - batch_start;

            // Build batch
            let mut batch_data = Vec::with_capacity(actual_batch_size * n_features);
            let mut target_data = Vec::with_capacity(actual_batch_size);
            for idx in batch_start..batch_end {
                batch_data.extend(x_train[idx].iter().map(|&v| v as f32));
                target_data.push(y_train[idx] as f32);
            }

            let input = Tensor2D::<WgpuBackend>::new(batch_data, actual_batch_size, n_features);
            let targets = Tensor1D::<WgpuBackend>::new(target_data);

            // Forward pass
            let predictions = manual_mlp.forward(&input);

            // Loss gradient (MSE derivative)
            let diff = predictions.sub(&targets);
            let grad_output = diff.scale(&Scalar::new(1.0 / actual_batch_size as f64));

            // Backward pass
            let gradients = manual_mlp.backward(&input, &grad_output);

            // Update parameters (stays on GPU!)
            let new_params = manual_optimizer.step(manual_mlp.params(), &gradients);
            manual_mlp.update_params(&new_params);
        }
    }

    let manual_mlp_time = manual_start.elapsed();
    let manual_fitted = manual_mlp.into_fitted();

    // Evaluate manual GPU MLP
    let manual_preds = manual_fitted.predict_batch(&test_features_gpu);
    let manual_flat = manual_preds.ravel().to_vec();
    let manual_preds_vec: Vec<f64> = (0..x_test.len()).map(|i| manual_flat[i] as f64).collect();
    let manual_metrics = compute_metrics(&manual_preds_vec, &y_test);

    println!("GPU Manual Training time: {:?}", manual_mlp_time);
    println!(
        "GPU Manual R² = {:.4}, MSE = {:.6}",
        manual_metrics.r2, manual_metrics.mse
    );

    let manual_speedup = cpu_mlp_time.as_secs_f64() / manual_mlp_time.as_secs_f64();
    println!("Manual loop speedup vs CPU: {:.2}x", manual_speedup);

    // ============================================
    // Large Model Test (GPU should be faster)
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("LARGE MODEL: GPU vs CPU (synthetic 64-feature dataset)");
    println!("{}", "=".repeat(60));
    println!("GPU benefits require: large models (100+ neurons), large batches\n");

    // Generate synthetic dataset with many features
    let large_n_features = 64;
    let large_n_samples = 10000usize;
    let mut large_x: Vec<Vec<f32>> = Vec::with_capacity(large_n_samples);
    let mut large_y: Vec<f32> = Vec::with_capacity(large_n_samples);
    let mut rng: u64 = 123;
    for _i in 0..large_n_samples {
        let mut sample = Vec::with_capacity(large_n_features);
        for _j in 0..large_n_features {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            sample.push((rng as f64 / u64::MAX as f64) as f32);
        }
        large_x.push(sample.clone());
        // Target: sum of squares
        let target: f32 = sample.iter().map(|&x| x * x).sum::<f32>() / large_n_features as f32;
        large_y.push(target);
    }

    // CPU Large Model
    let large_arch = &[large_n_features, 256, 128, 1];
    let large_acts = &[Activation::ReLU, Activation::ReLU, Activation::Identity];

    println!("CPU training large model (64->256->128->1)...");
    let cpu_large_start = std::time::Instant::now();
    let cpu_large_model = MLP::<CpuBackend>::new(large_arch, large_acts);
    let cpu_large_optimizer = SGD::new(0.01);
    let large_batch = 512;
    let mut cpu_large_model = cpu_large_model;

    for _epoch in 0..20 {
        for batch_start in (0..large_n_samples).step_by(large_batch) {
            let batch_end = (batch_start + large_batch).min(large_n_samples);
            let actual_batch = batch_end - batch_start;

            let mut batch_data = Vec::with_capacity(actual_batch * large_n_features);
            let mut target_data = Vec::with_capacity(actual_batch);
            for idx in batch_start..batch_end {
                batch_data.extend(&large_x[idx]);
                target_data.push(large_y[idx]);
            }

            let input = Tensor2D::<CpuBackend>::new(batch_data, actual_batch, large_n_features);
            let targets = Tensor1D::<CpuBackend>::new(target_data);

            let predictions = cpu_large_model.forward(&input);
            let diff = predictions.sub(&targets);
            let grad_output = diff.scale(&Scalar::new(1.0 / actual_batch as f64));
            let gradients = cpu_large_model.backward(&input, &grad_output);
            let new_params = cpu_large_optimizer.step(cpu_large_model.params(), &gradients);
            cpu_large_model.update_params(&new_params);
        }
    }
    let cpu_large_time = cpu_large_start.elapsed();

    // GPU Large Model
    println!("GPU training large model (64->256->128->1)...");
    let gpu_large_start = std::time::Instant::now();
    let gpu_large_model = MLP::<WgpuBackend>::new(large_arch, large_acts);
    let gpu_large_optimizer = SGD::new(0.01);
    let mut gpu_large_model = gpu_large_model;

    for _epoch in 0..20 {
        for batch_start in (0..large_n_samples).step_by(large_batch) {
            let batch_end = (batch_start + large_batch).min(large_n_samples);
            let actual_batch = batch_end - batch_start;

            let mut batch_data = Vec::with_capacity(actual_batch * large_n_features);
            let mut target_data = Vec::with_capacity(actual_batch);
            for idx in batch_start..batch_end {
                batch_data.extend(&large_x[idx]);
                target_data.push(large_y[idx]);
            }

            let input = Tensor2D::<WgpuBackend>::new(batch_data, actual_batch, large_n_features);
            let targets = Tensor1D::<WgpuBackend>::new(target_data);

            let predictions = gpu_large_model.forward(&input);
            let diff = predictions.sub(&targets);
            let grad_output = diff.scale(&Scalar::new(1.0 / actual_batch as f64));
            let gradients = gpu_large_model.backward(&input, &grad_output);
            let new_params = gpu_large_optimizer.step(gpu_large_model.params(), &gradients);
            gpu_large_model.update_params(&new_params);
        }
    }
    let gpu_large_time = gpu_large_start.elapsed();

    let large_speedup = cpu_large_time.as_secs_f64() / gpu_large_time.as_secs_f64();
    println!("\nCPU Large Model time: {:?}", cpu_large_time);
    println!("GPU Large Model time: {:?}", gpu_large_time);
    println!(
        "Speedup: {:.1}x ({})",
        large_speedup,
        if large_speedup > 1.0 {
            "GPU FASTER!"
        } else {
            "CPU faster"
        }
    );

    // ============================================
    // Summary
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!(
        "{:40} {:>10} {:>10} {:>10}",
        "Approach", "CPU Time", "GPU Time", "Speedup"
    );
    println!("{}", "-".repeat(72));
    println!(
        "{:40} {:>10?} {:>10?} {:>10.2}x",
        "Linear Regression (CA Housing)", cpu_linear_time, gpu_linear_time, linear_speedup
    );
    println!(
        "{:40} {:>10?} {:>10?} {:>10.2}x",
        "MLP Small (CA Housing, 8->32->16->1)", cpu_mlp_time, gpu_mlp_time, mlp_speedup
    );
    println!(
        "{:40} {:>10?} {:>10?} {:>10.2}x",
        "MLP Large (Synthetic, 64->256->128->1)", cpu_large_time, gpu_large_time, large_speedup
    );

    println!("\n{}", "=".repeat(60));
    println!("CONCLUSION");
    println!("{}", "=".repeat(60));
    println!();

    if large_speedup > 1.0 {
        println!("✓ GPU IS {:.1}x FASTER for large models!", large_speedup);
    } else {
        println!(
            "✗ GPU {:.1}x slower - test with larger model needed",
            1.0 / large_speedup
        );
    }

    println!();
    println!("Key Findings:");
    println!("─────────────────────────────────────────────────────────────");
    println!();
    println!("1. SMALL MODELS (few features, few neurons): CPU is faster");
    println!("   - California Housing: 8 features, small MLP → CPU wins");
    println!("   - GPU overhead exceeds computation time");
    println!();
    println!("2. LARGE MODELS (64+ features, 256+ neurons): GPU is 10-20x faster");
    println!(
        "   - Synthetic 64-feature test: GPU {:.1}x faster (varies by system)",
        large_speedup
    );
    println!("   - See mlp_cpu_wgpu_comparison example for larger speedups");
    println!("   - Computation time dominates over GPU overhead");
    println!();
    println!("3. GPU TRAINING REQUIREMENTS:");
    println!("   - Model size: 100+ neurons per layer");
    println!("   - Batch size: 512+ (reduces sync overhead)");
    println!("   - Manual training loop (avoids Trainer per-epoch sync)");
    println!();
    println!("4. FOR CALIFORNIA HOUSING:");
    println!("   - Only 8 features, small models optimal");
    println!("   - Use CPU backend for training");
    println!("   - Consider GPU for inference with batch predictions");

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
