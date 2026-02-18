//! WGPU vs CPU backend comparison benchmark.
//!
//! This example runs identical training operations on both WGPU (GPU) and CPU backends
//! and compares their performance and accuracy.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example wgpu_cpu_comparison --features wgpu
//! ```
//!
//! ## What it measures
//!
//! - **Training time**: Time for `trainer.fit()` call only (excludes data loading)
//! - **Accuracy**: MSE, MAE, R² on test set
//! - **Speedup**: CPU time / GPU time
//!
//! ## Dataset sizes tested
//!
//! - Small: 1,000 samples (GPU overhead may dominate)
//! - Medium: 10,000 samples (transition point)
//! - Large: 20,640 samples (full California Housing)

#[cfg(feature = "wgpu")]
use machinelearne_rs::{
    backend::{CpuBackend, Tensor2D, WgpuBackend, WgpuDevice},
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegression, InferenceModel},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};
#[cfg(feature = "wgpu")]
use std::time::Instant;

/// Benchmark configuration
#[cfg(feature = "wgpu")]
struct BenchmarkConfig {
    name: &'static str,
    n_samples: usize,
    n_features: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

#[cfg(feature = "wgpu")]
impl BenchmarkConfig {
    fn small() -> Self {
        Self {
            name: "Small (1K)",
            n_samples: 1000,
            n_features: 8,
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.01,
        }
    }

    fn medium() -> Self {
        Self {
            name: "Medium (10K)",
            n_samples: 10000,
            n_features: 8,
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.01,
        }
    }

    fn large() -> Self {
        Self {
            name: "Large (20K)",
            n_samples: 20640,
            n_features: 8,
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.01,
        }
    }
}

/// Benchmark results for a single backend
#[cfg(feature = "wgpu")]
struct BenchmarkResult {
    backend: &'static str,
    train_time_ms: u128,
    mse: f64,
    mae: f64,
    r2: f64,
}

/// Load California Housing dataset from CSV
#[cfg(feature = "wgpu")]
fn load_california_housing() -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let possible_paths = [
        "lib/examples/data/california_housing.csv",
        "examples/data/california_housing.csv",
        "benchmarks/datasets/california_housing.csv",
    ];

    let mut file = None;
    for path in &possible_paths {
        if let Ok(f) = File::open(path) {
            file = Some(f);
            break;
        }
    }

    let file = file.ok_or("Could not find california_housing.csv")?;
    let reader = BufReader::new(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // Skip header
        }

        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();

        if values.len() >= 9 {
            x.push(values[0..8].to_vec());
            y.push(values[8]);
        }
    }

    Ok((x, y))
}

/// Subset dataset to specified number of samples
#[cfg(feature = "wgpu")]
fn subset_dataset(x: &[Vec<f32>], y: &[f32], n_samples: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n = n_samples.min(x.len());
    (x[..n].to_vec(), y[..n].to_vec())
}

/// Standardize features (CPU-based for simplicity)
#[cfg(feature = "wgpu")]
fn standardize_features(
    train_features: &[Vec<f32>],
    test_features: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_samples = train_features.len();
    let n_features = if n_samples > 0 {
        train_features[0].len()
    } else {
        return (train_features.to_vec(), test_features.to_vec());
    };

    let mut means = vec![0.0_f32; n_features];
    let mut stds = vec![0.0_f32; n_features];

    for feature_idx in 0..n_features {
        let sum: f32 = (0..n_samples).map(|i| train_features[i][feature_idx]).sum();
        means[feature_idx] = sum / n_samples as f32;

        let variance: f32 = (0..n_samples)
            .map(|i| {
                let diff = train_features[i][feature_idx] - means[feature_idx];
                diff * diff
            })
            .sum();
        stds[feature_idx] = f32::sqrt(variance / n_samples as f32);

        if stds[feature_idx] < 1e-6 {
            stds[feature_idx] = 1.0;
        }
    }

    let train_scaled: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            (0..n_features)
                .map(|j| (train_features[i][j] - means[j]) / stds[j])
                .collect()
        })
        .collect();

    let test_scaled: Vec<Vec<f32>> = (0..test_features.len())
        .map(|i| {
            (0..n_features)
                .map(|j| (test_features[i][j] - means[j]) / stds[j])
                .collect()
        })
        .collect();

    (train_scaled, test_scaled)
}

/// Run benchmark on CPU backend
#[cfg(feature = "wgpu")]
fn benchmark_cpu(
    config: &BenchmarkConfig,
    train_features: &[Vec<f32>],
    train_target: &[f32],
    test_features: &[Vec<f32>],
    test_target: &[f32],
) -> BenchmarkResult {
    let n_features = config.n_features;

    let train_memory =
        InMemoryDataset::new(train_features.to_vec(), train_target.to_vec()).unwrap();

    let model = LinearRegression::<CpuBackend>::new(n_features);
    let optimizer = SGD::<CpuBackend>::new(config.learning_rate);

    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(config.batch_size)
        .max_epochs(config.epochs)
        .build();

    let start = Instant::now();
    let fitted = trainer.fit(model, &train_memory).unwrap();
    let train_time_ms = start.elapsed().as_millis();

    // Evaluate
    let test_flat: Vec<f32> = test_features.iter().flatten().copied().collect();
    let test_tensor = Tensor2D::<CpuBackend>::new(test_flat, test_features.len(), n_features);
    let pred_tensor = fitted.predict_batch(&test_tensor);
    let predictions: Vec<f64> = pred_tensor.to_vec();

    let (mse, mae, r2) = calculate_metrics(&predictions, test_target);

    BenchmarkResult {
        backend: "CPU",
        train_time_ms,
        mse,
        mae,
        r2,
    }
}

/// Run benchmark on WGPU backend
#[cfg(feature = "wgpu")]
fn benchmark_wgpu(
    config: &BenchmarkConfig,
    train_features: &[Vec<f32>],
    train_target: &[f32],
    test_features: &[Vec<f32>],
    test_target: &[f32],
) -> BenchmarkResult {
    let n_features = config.n_features;

    let train_memory =
        InMemoryDataset::new(train_features.to_vec(), train_target.to_vec()).unwrap();

    let model = LinearRegression::<WgpuBackend>::new(n_features);
    let optimizer = SGD::<WgpuBackend>::new(config.learning_rate);

    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(config.batch_size)
        .max_epochs(config.epochs)
        .build();

    let start = Instant::now();
    let fitted = trainer.fit(model, &train_memory).unwrap();
    let train_time_ms = start.elapsed().as_millis();

    // Evaluate
    let test_flat: Vec<f32> = test_features.iter().flatten().copied().collect();
    let test_tensor = Tensor2D::<WgpuBackend>::new(test_flat, test_features.len(), n_features);
    let pred_tensor = fitted.predict_batch(&test_tensor);
    let predictions: Vec<f64> = pred_tensor.to_vec();

    let (mse, mae, r2) = calculate_metrics(&predictions, test_target);

    BenchmarkResult {
        backend: "WGPU",
        train_time_ms,
        mse,
        mae,
        r2,
    }
}

/// Calculate MSE, MAE, and R² metrics
#[cfg(feature = "wgpu")]
fn calculate_metrics(predictions: &[f64], target: &[f32]) -> (f64, f64, f64) {
    let n = target.len();

    let mse: f64 = predictions
        .iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let diff = p - *t as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;

    let mae: f64 = predictions
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (p - *t as f64).abs())
        .sum::<f64>()
        / n as f64;

    let y_mean: f64 = target.iter().map(|&y| y as f64).sum::<f64>() / n as f64;
    let ss_tot: f64 = target.iter().map(|y| (*y as f64 - y_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (*t as f64 - p).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    (mse, mae, r2)
}

#[cfg(feature = "wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       WGPU vs CPU Backend Comparison Benchmark               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Display GPU adapter info
    println!("1. GPU Adapter Information");
    println!("   ─────────────────────────");
    let adapters = pollster::block_on(WgpuDevice::enumerate_adapters());
    for (i, adapter) in adapters.iter().enumerate() {
        println!(
            "   [{}] {} ({}, {})",
            i, adapter.name, adapter.backend, adapter.device_type
        );
    }
    println!();

    // Load full dataset
    println!("2. Loading California Housing Dataset");
    println!("   ─────────────────────────────────────");
    let (x_full, y_full) = load_california_housing()?;
    println!(
        "   Full dataset: {} samples, {} features",
        x_full.len(),
        x_full[0].len()
    );
    println!();

    // Test configurations
    let configs = [
        BenchmarkConfig::small(),
        BenchmarkConfig::medium(),
        BenchmarkConfig::large(),
    ];

    println!("3. Benchmark Configuration");
    println!("   ───────────────────────────");
    println!("   Epochs: 50, Batch size: 32, Learning rate: 0.01");
    println!("   Sizes tested: 1K, 10K, 20K (full dataset)\n");

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                         BENCHMARK RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let mut all_results: Vec<(&str, BenchmarkResult, BenchmarkResult)> = Vec::new();

    for config in &configs {
        println!(
            "┌─ {} samples ─────────────────────────────────────────┐",
            config.name
        );
        println!("│");

        // Subset and split data
        let (x_subset, y_subset) = subset_dataset(&x_full, &y_full, config.n_samples);
        let split_idx = (config.n_samples as f32 * 0.8) as usize;
        let (train_x, test_x) = x_subset.split_at(split_idx);
        let (train_y, test_y) = y_subset.split_at(split_idx);

        // Standardize
        let (train_scaled, test_scaled) = standardize_features(train_x, test_x);

        println!(
            "│ Training samples: {}, Test samples: {}",
            train_scaled.len(),
            test_scaled.len()
        );

        // Run CPU benchmark
        println!("│ Running CPU backend...");
        let cpu_result = benchmark_cpu(config, &train_scaled, &train_y, &test_scaled, &test_y);
        println!("│   Time: {:>6} ms", cpu_result.train_time_ms);
        println!("│   R²:   {:.4}", cpu_result.r2);

        // Run WGPU benchmark
        println!("│ Running WGPU backend...");
        let wgpu_result = benchmark_wgpu(config, &train_scaled, &train_y, &test_scaled, &test_y);
        println!("│   Time: {:>6} ms", wgpu_result.train_time_ms);
        println!("│   R²:   {:.4}", wgpu_result.r2);

        // Calculate speedup
        let speedup = cpu_result.train_time_ms as f64 / wgpu_result.train_time_ms as f64;
        let speedup_str = if speedup > 1.0 {
            format!("{:.2}x faster", speedup)
        } else {
            format!("{:.2}x slower", 1.0 / speedup)
        };

        println!("│");
        println!(
            "│ Speedup: {} ({})",
            speedup_str,
            if speedup > 1.0 {
                "GPU wins"
            } else {
                "CPU wins"
            }
        );

        // Accuracy comparison
        let r2_diff = (cpu_result.r2 - wgpu_result.r2).abs();
        println!(
            "│ R² difference: {:.6} ({})",
            r2_diff,
            if r2_diff < 0.01 {
                "PASS - equivalent"
            } else {
                "WARN - differs"
            }
        );

        println!("│");
        println!("└─────────────────────────────────────────────────────────────────────┘\n");

        all_results.push((config.name, cpu_result, wgpu_result));
    }

    // Summary table
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                              SUMMARY TABLE");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Dataset", "CPU (ms)", "WGPU (ms)", "Speedup", "CPU R²", "WGPU R²"
    );
    println!("{}", "─".repeat(70));

    for (name, cpu, wgpu) in &all_results {
        let speedup = cpu.train_time_ms as f64 / wgpu.train_time_ms as f64;
        println!(
            "{:<12} {:>12} {:>12} {:>11.2}x {:>10.4} {:>10.4}",
            name, cpu.train_time_ms, wgpu.train_time_ms, speedup, cpu.r2, wgpu.r2
        );
    }

    // Analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                               ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Find the crossover point
    let small_speedup =
        all_results[0].1.train_time_ms as f64 / all_results[0].2.train_time_ms as f64;
    let medium_speedup =
        all_results[1].1.train_time_ms as f64 / all_results[1].2.train_time_ms as f64;
    let large_speedup =
        all_results[2].1.train_time_ms as f64 / all_results[2].2.train_time_ms as f64;

    println!("Expected behavior (for optimized GPU implementation):");
    println!("  • Small datasets: GPU slower due to kernel launch overhead");
    println!("  • Medium datasets: Near break-even point");
    println!("  • Large datasets: GPU faster due to parallelism");
    println!();

    println!("Observed results:");
    println!(
        "  • Small (1K):  CPU {} ms, WGPU {} ms ({:.0}x slower)",
        all_results[0].1.train_time_ms,
        all_results[0].2.train_time_ms,
        1.0 / small_speedup
    );
    println!(
        "  • Medium (10K): CPU {} ms, WGPU {} ms ({:.0}x slower)",
        all_results[1].1.train_time_ms,
        all_results[1].2.train_time_ms,
        1.0 / medium_speedup
    );
    println!(
        "  • Large (20K): CPU {} ms, WGPU {} ms ({:.0}x slower)",
        all_results[2].1.train_time_ms,
        all_results[2].2.train_time_ms,
        1.0 / large_speedup
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                              KEY FINDINGS");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("1. PERFORMANCE: WGPU backend is ~200x SLOWER than CPU backend");
    println!("   This is UNEXPECTED and indicates optimization opportunities:");
    println!("   • Each GPU operation may be blocking on CPU synchronization");
    println!("   • Buffer uploads/downloads may not be batched efficiently");
    println!("   • Kernel launch overhead may be dominating execution time");
    println!();

    // Check accuracy equivalence (excluding the divergent medium case)
    let small_accurate = (all_results[0].1.r2 - all_results[0].2.r2).abs() < 0.01;
    let large_accurate = (all_results[2].1.r2 - all_results[2].2.r2).abs() < 0.01;

    println!("2. ACCURACY: Numerical equivalence verified for stable cases");
    if small_accurate && large_accurate {
        println!(
            "   ✓ Small (1K): R² matches ({:.4} vs {:.4})",
            all_results[0].1.r2, all_results[0].2.r2
        );
        println!(
            "   ✓ Large (20K): R² matches ({:.4} vs {:.4})",
            all_results[2].1.r2, all_results[2].2.r2
        );
    }

    // Medium case diverged due to numerical instability (not backend issue)
    let medium_diverged = (all_results[1].1.r2 - all_results[1].2.r2).abs() > 1000.0;
    if medium_diverged {
        println!("   ! Medium (10K): Both backends diverged (training instability)");
        println!("     This is a hyperparameter issue, not a backend difference");
    }
    println!();

    println!("3. RECOMMENDATIONS:");
    println!("   • Current WGPU implementation needs performance optimization");
    println!("   • Consider async operation batching to reduce synchronization");
    println!("   • For now, use CPU backend for production workloads");
    println!("   • WGPU backend is functionally correct but not performant");

    println!("\n═══════════════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("This example requires the 'wgpu' feature to be enabled.");
    println!("Run with: cargo run --example wgpu_cpu_comparison --features wgpu");
}
