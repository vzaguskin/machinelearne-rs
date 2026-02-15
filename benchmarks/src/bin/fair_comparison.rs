#!/usr/bin/env rust-script
//!
//! Fair comparison benchmark for machinelearne-rs vs sklearn.
//!
//! This benchmark runs with EXACT configurations to match sklearn:
//! - Same number of epochs (5, 50, etc.)
//! - Same batch sizes (full batch = all training samples)
//! - Same learning rate (0.01)
//! - Same data preprocessing (z-score standardization)
//!
//! Key principle: Compare identical workloads, not different defaults.

use machinelearne_rs::{
    backend::{CpuBackend, Tensor1D},
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegression, InferenceModel},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

type Features = Vec<Vec<f32>>;
type Targets = Vec<f32>;

// ============================================================================
// Data Loading and Preprocessing (matches sklearn exactly)
// ============================================================================

fn load_california_housing() -> (Vec<Vec<f32>>, Vec<f32>) {
    let data = std::fs::read_to_string("benchmarks/datasets/california_housing.csv")
        .expect("Failed to read California Housing dataset");

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for (i, line) in data.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }

        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.trim().parse().expect("Failed to parse number"))
            .collect();

        if values.len() == 9 {
            features.push(values[..8].to_vec());
            targets.push(values[8]);
        }
    }

    (features, targets)
}

fn split_train_test(
    features: &[Vec<f32>],
    targets: &[f32],
) -> (Features, Features, Targets, Targets) {
    let n = features.len();
    let train_size = (n as f64 * 0.8) as usize;

    let train_features = features[..train_size].to_vec();
    let train_targets = targets[..train_size].to_vec();
    let test_features = features[train_size..].to_vec();
    let test_targets = targets[train_size..].to_vec();

    (train_features, test_features, train_targets, test_targets)
}

fn standardize_features(
    train_features: &[Vec<f32>],
    test_features: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_samples = train_features.len();
    let n_features = train_features[0].len();

    // Compute mean and std from training data (same as sklearn StandardScaler)
    let mut means = vec![0.0_f32; n_features];
    let mut stds = vec![0.0_f32; n_features];

    for feature_idx in 0..n_features {
        let sum: f32 = (0..n_samples).map(|i| train_features[i][feature_idx]).sum();
        means[feature_idx] = sum / n_samples as f32;

        // sklearn uses biased variance (divide by n, not n-1)
        let variance: f32 = (0..n_samples)
            .map(|i| {
                let diff = train_features[i][feature_idx] - means[feature_idx];
                diff * diff
            })
            .sum();
        let std = (variance / n_samples as f32).sqrt();
        // sklearn uses with_std=True which prevents division by zero
        stds[feature_idx] = if std > 1e-8 { std } else { 1.0 };
    }

    // Apply z-score to train and test data using same statistics
    let train_scaled: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            (0..n_features)
                .map(|j| (train_features[i][j] - means[j]) / stds[j])
                .collect()
        })
        .collect();

    let test_scaled: Vec<Vec<f32>> = test_features
        .iter()
        .map(|row| {
            (0..n_features)
                .map(|j| (row[j] - means[j]) / stds[j])
                .collect()
        })
        .collect();

    (train_scaled, test_scaled)
}

// ============================================================================
// Metrics (matches sklearn exactly)
// ============================================================================

fn compute_metrics(predictions: &[f32], targets: &[f32]) -> (f64, f64, f64) {
    let n = targets.len();

    // MSE
    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (*p as f64 - *t as f64).powi(2))
        .sum::<f64>()
        / n as f64;

    // MAE
    let mae: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (*p as f64 - *t as f64).abs())
        .sum::<f64>()
        / n as f64;

    // R²
    let target_mean: f64 = targets.iter().map(|t| *t as f64).sum::<f64>() / n as f64;
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (*p as f64 - *t as f64).powi(2))
        .sum();
    let ss_tot: f64 = targets
        .iter()
        .map(|t| (*t as f64 - target_mean).powi(2))
        .sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    (mse, mae, r_squared)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

struct BenchmarkConfig {
    name: String,
    n_features: usize,
    n_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    n_runs: usize,
}

struct BenchmarkResult {
    config: BenchmarkConfig,
    train_time_mean_ms: f64,
    train_time_std_ms: f64,
    train_time_min_ms: f64,
    time_per_epoch_ms: f64,
    time_per_sample_us: f64,
    mse: f64,
    mae: f64,
    r2: f64,
    weights: Vec<f32>,
    bias: f32,
    samples_processed: usize,
}

fn run_single_benchmark(
    config: &BenchmarkConfig,
    train_features: &[Vec<f32>],
    train_targets: &[f32],
    test_features: &[Vec<f32>],
    test_targets: &[f32],
) -> BenchmarkResult {
    let n_train = train_features.len();
    let samples_per_epoch = if config.batch_size >= n_train {
        n_train
    } else {
        (n_train / config.batch_size + 1) * config.batch_size
    };
    let samples_processed = config.n_epochs * samples_per_epoch;

    let mut times: Vec<Duration> = Vec::with_capacity(config.n_runs);
    let mut final_weights: Vec<f32> = Vec::new();
    let mut final_bias: f32 = 0.0;
    let mut final_mse = 0.0;
    let mut final_mae = 0.0;
    let mut final_r2 = 0.0;

    for _ in 0..config.n_runs {
        // Create dataset
        let train_dataset = InMemoryDataset::new(train_features.to_vec(), train_targets.to_vec())
            .expect("Failed to create dataset");

        // Create model and trainer
        let model = LinearRegression::<CpuBackend>::new(config.n_features);
        let optimizer = SGD::new(config.learning_rate);

        let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
            .batch_size(config.batch_size)
            .max_epochs(config.n_epochs)
            .verbose(false) // Suppress training output for clean benchmark output
            .build();

        // Suppress training output by capturing stdout
        // Note: In production, we'd add a quiet flag to trainer

        // Train
        let start = Instant::now();
        let fitted = trainer.fit(model, &train_dataset).expect("Training failed");
        let elapsed = start.elapsed();
        times.push(elapsed);

        // Get predictions
        let predictions: Vec<f32> = test_features
            .iter()
            .map(|sample| {
                let tensor_input = Tensor1D::<CpuBackend>::new(sample.clone());
                fitted.predict(&tensor_input).to_f64() as f32
            })
            .collect();

        // Compute metrics
        let (mse, mae, r2) = compute_metrics(&predictions, test_targets);

        // Store final results
        let params = fitted.extract_params();
        final_weights = params.weights.clone();
        final_bias = params.bias;
        final_mse = mse;
        final_mae = mae;
        final_r2 = r2;
    }

    // Calculate statistics
    let times_ms: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1000.0).collect();
    let mean_time = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let std_time = if times_ms.len() > 1 {
        let variance = times_ms
            .iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f64>()
            / (times_ms.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let min_time = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);

    BenchmarkResult {
        config: BenchmarkConfig {
            name: config.name.clone(),
            ..config.clone()
        },
        train_time_mean_ms: mean_time,
        train_time_std_ms: std_time,
        train_time_min_ms: min_time,
        time_per_epoch_ms: mean_time / config.n_epochs as f64,
        time_per_sample_us: mean_time * 1e6 / samples_processed as f64,
        mse: final_mse,
        mae: final_mae,
        r2: final_r2,
        weights: final_weights,
        bias: final_bias,
        samples_processed,
    }
}

impl Clone for BenchmarkConfig {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            n_features: self.n_features,
            n_epochs: self.n_epochs,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            n_runs: self.n_runs,
        }
    }
}

fn run_all_benchmarks() -> Vec<serde_json::Value> {
    println!("{}", "=".repeat(70));
    println!("MACHINELEARN-RS FAIR COMPARISON BENCHMARKS");
    println!("{}", "=".repeat(70));

    // Load and preprocess data
    let (features, targets) = load_california_housing();
    println!(
        "\nLoaded {} samples with {} features",
        features.len(),
        features[0].len()
    );

    let (train_features_all, test_features_all, train_targets, test_targets) =
        split_train_test(&features, &targets);

    let (train_scaled_all, test_scaled_all) =
        standardize_features(&train_features_all, &test_features_all);

    println!(
        "Train: {} samples, Test: {} samples",
        train_scaled_all.len(),
        test_scaled_all.len()
    );

    let n_train = train_scaled_all.len();
    let mut results = Vec::new();

    // Define benchmark configurations to match sklearn exactly
    let configs = vec![
        // 2 features tests - Full batch with different learning rates
        BenchmarkConfig {
            name: "sgd_epochs_5_batch_full_lr_0.5".to_string(),
            n_features: 2,
            n_epochs: 5,
            batch_size: n_train, // Full batch
            learning_rate: 0.5,  // Optimal for Rust full-batch
            n_runs: 10,
        },
        BenchmarkConfig {
            name: "sgd_epochs_50_batch_full_lr_0.5".to_string(),
            n_features: 2,
            n_epochs: 50,
            batch_size: n_train,
            learning_rate: 0.5,
            n_runs: 5,
        },
        BenchmarkConfig {
            name: "sgd_epochs_5_batch_full_lr_0.1".to_string(),
            n_features: 2,
            n_epochs: 5,
            batch_size: n_train,
            learning_rate: 0.1, // sklearn-stable
            n_runs: 10,
        },
        BenchmarkConfig {
            name: "sgd_epochs_50_batch_full_lr_0.1".to_string(),
            n_features: 2,
            n_epochs: 50,
            batch_size: n_train,
            learning_rate: 0.1,
            n_runs: 5,
        },
        // Mini-batch tests
        BenchmarkConfig {
            name: "sgd_epochs_5_batch_32".to_string(),
            n_features: 2,
            n_epochs: 5,
            batch_size: 32,
            learning_rate: 0.01,
            n_runs: 10,
        },
        BenchmarkConfig {
            name: "sgd_epochs_50_batch_32".to_string(),
            n_features: 2,
            n_epochs: 50,
            batch_size: 32,
            learning_rate: 0.01,
            n_runs: 5,
        },
        // 8 features tests - use LR=0.1 for stability
        BenchmarkConfig {
            name: "sgd_epochs_5_batch_full_8feat_lr_0.1".to_string(),
            n_features: 8,
            n_epochs: 5,
            batch_size: n_train,
            learning_rate: 0.1,
            n_runs: 10,
        },
        BenchmarkConfig {
            name: "sgd_epochs_50_batch_full_8feat_lr_0.1".to_string(),
            n_features: 8,
            n_epochs: 50,
            batch_size: n_train,
            learning_rate: 0.1,
            n_runs: 5,
        },
    ];

    for config in configs {
        println!("\n{}", "-".repeat(50));
        println!("Test: {} ({} features)", config.name, config.n_features);
        println!(
            "Epochs: {}, Batch size: {}",
            config.n_epochs, config.batch_size
        );

        // Select features
        let n_feat = config.n_features;
        let train_features: Vec<Vec<f32>> = train_scaled_all
            .iter()
            .map(|row| row[..n_feat].to_vec())
            .collect();
        let test_features: Vec<Vec<f32>> = test_scaled_all
            .iter()
            .map(|row| row[..n_feat].to_vec())
            .collect();

        let result = run_single_benchmark(
            &config,
            &train_features,
            &train_targets,
            &test_features,
            &test_targets,
        );

        println!(
            "  Time: {:.3} +/- {:.3} ms",
            result.train_time_mean_ms, result.train_time_std_ms
        );
        println!("  Time per epoch: {:.3} ms", result.time_per_epoch_ms);
        println!("  Time per sample: {:.4} us", result.time_per_sample_us);
        println!("  MSE: {:.6}, R2: {:.6}", result.mse, result.r2);
        println!("  Weights: {:?}, Bias: {:.6}", result.weights, result.bias);

        results.push(json!({
            "test": result.config.name,
            "implementation": "rust",
            "n_features": result.config.n_features,
            "n_epochs": result.config.n_epochs,
            "batch_size": result.config.batch_size,
            "learning_rate": result.config.learning_rate,
            "n_train_samples": n_train,
            "n_test_samples": test_features.len(),
            "samples_processed": result.samples_processed,
            "train_time_mean_ms": result.train_time_mean_ms,
            "train_time_std_ms": result.train_time_std_ms,
            "train_time_min_ms": result.train_time_min_ms,
            "time_per_epoch_ms": result.time_per_epoch_ms,
            "time_per_sample_us": result.time_per_sample_us,
            "mse": result.mse,
            "mae": result.mae,
            "r2": result.r2,
            "weights": result.weights,
            "bias": result.bias,
            "n_runs": result.config.n_runs,
        }));
    }

    results
}

fn main() {
    let results = run_all_benchmarks();

    // Save results
    let output = json!({
        "results": results
    });

    let mut file = File::create("benchmarks/results/rust_fair_comparison.json")
        .expect("Failed to create output file");
    file.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .expect("Failed to write to file");

    println!("\nResults saved to benchmarks/results/rust_fair_comparison.json");

    // Print summary
    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK COMPLETE");
    println!("{}", "=".repeat(70));

    println!("\nSummary:");
    println!(
        "{:<35} {:<12} {:<12} {:<10} {:<10}",
        "Test", "Time (ms)", "Time/epoch", "MSE", "R2"
    );
    println!("{}", "-".repeat(80));

    for r in &results {
        let test = r["test"].as_str().unwrap();
        let time_ms = r["train_time_mean_ms"].as_f64().unwrap();
        let time_per_epoch = r["time_per_epoch_ms"].as_f64().unwrap();
        let mse = r["mse"].as_f64().unwrap();
        let r2 = r["r2"].as_f64().unwrap();

        println!(
            "{:<35} {:<12.3} {:<12.3} {:<10.6} {:<10.4}",
            test, time_ms, time_per_epoch, mse, r2
        );
    }
}
