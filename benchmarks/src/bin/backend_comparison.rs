#!/usr/bin/env rust-script
//!
//! Compare CPU backend vs ndarray backend performance

use benchmarks::data::CaliforniaHousingDataset;
use benchmarks::metrics::Metrics;
use machinelearne_rs::{
    backend::{CpuBackend, Tensor2D},
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
use std::time::Instant;

/// Helper function to convert Vec<Vec<f32>> to Tensor2D<CpuBackend>
fn vec_to_tensor2d_cpu(data: &[Vec<f32>]) -> Tensor2D<CpuBackend> {
    let flat: Vec<f32> = data.iter().flatten().copied().collect();
    Tensor2D::new(flat, data.len(), data.first().map(|v| v.len()).unwrap_or(0))
}

#[cfg(feature = "ndarray")]
use machinelearne_rs::backend::NdarrayBackend;

/// Helper function to convert Vec<Vec<f32>> to Tensor2D for ndarray backend
#[cfg(feature = "ndarray")]
fn vec_to_tensor2d_ndarray(data: &[Vec<f32>]) -> Tensor2D<NdarrayBackend> {
    let flat: Vec<f32> = data.iter().flatten().copied().collect();
    Tensor2D::new(flat, data.len(), data.first().map(|v| v.len()).unwrap_or(0))
}

/// Apply z-score standardization to features
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

/// Benchmark CPU backend
fn benchmark_cpu_backend(n_features: usize) -> serde_json::Value {
    let feature_indices: Vec<usize> = (0..n_features).collect();

    let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&feature_indices);
    let (train_dataset, _, test_dataset) = subset.split_train_val_test(0.8, 0.0);

    let train_features = train_dataset.features();
    let train_target = train_dataset.target();
    let test_features = test_dataset.features();
    let test_target = test_dataset.target();

    let (train_features_scaled, test_features_scaled) =
        standardize_features(train_features, test_features);

    let train_memory = InMemoryDataset::new(train_features_scaled, train_target.to_vec())
        .expect("Failed to create training dataset");

    let lr = 0.01;
    let model = LinearRegression::<CpuBackend>::new(n_features);
    let optimizer = SGD::<CpuBackend>::new(lr);

    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(32)
        .max_epochs(50)
        .build();

    let start = Instant::now();
    let fitted = trainer
        .fit(model, &train_memory)
        .expect("Failed to fit model");
    let train_time_ms = start.elapsed().as_millis();

    let test_tensor = vec_to_tensor2d_cpu(&test_features_scaled);
    let pred_tensor = fitted.predict_batch(&test_tensor);

    let predictions: Vec<f32> = pred_tensor.to_vec().into_iter().map(|v| v as f32).collect();

    let mse = Metrics::mse(test_target, &predictions);
    let mae = Metrics::mae(test_target, &predictions);
    let r2 = Metrics::r_squared(test_target, &predictions);

    json!({
        "backend": "CpuBackend",
        "n_features": n_features,
        "train_time_ms": train_time_ms,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    })
}

/// Benchmark ndarray backend (only available with ndarray feature)
#[cfg(feature = "ndarray")]
fn benchmark_ndarray_backend(n_features: usize) -> serde_json::Value {
    let feature_indices: Vec<usize> = (0..n_features).collect();

    let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&feature_indices);
    let (train_dataset, _, test_dataset) = subset.split_train_val_test(0.8, 0.0);

    let train_features = train_dataset.features();
    let train_target = train_dataset.target();
    let test_features = test_dataset.features();
    let test_target = test_dataset.target();

    let (train_features_scaled, test_features_scaled) =
        standardize_features(train_features, test_features);

    let train_memory = InMemoryDataset::new(train_features_scaled, train_target.to_vec())
        .expect("Failed to create training dataset");

    let lr = 0.01;
    let model = LinearRegression::<NdarrayBackend>::new(n_features);
    let optimizer = SGD::<NdarrayBackend>::new(lr);

    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(32)
        .max_epochs(50)
        .build();

    let start = Instant::now();
    let fitted = trainer
        .fit(model, &train_memory)
        .expect("Failed to fit model");
    let train_time_ms = start.elapsed().as_millis();

    let test_tensor = vec_to_tensor2d_ndarray(&test_features_scaled);
    let pred_tensor = fitted.predict_batch(&test_tensor);

    let predictions: Vec<f32> = pred_tensor.to_vec().into_iter().map(|v| v as f32).collect();

    let mse = Metrics::mse(test_target, &predictions);
    let mae = Metrics::mae(test_target, &predictions);
    let r2 = Metrics::r_squared(test_target, &predictions);

    json!({
        "backend": "NdarrayBackend",
        "n_features": n_features,
        "train_time_ms": train_time_ms,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    })
}

/// Placeholder for when ndarray feature is not enabled
#[cfg(not(feature = "ndarray"))]
fn benchmark_ndarray_backend(n_features: usize) -> serde_json::Value {
    json!({
        "backend": "NdarrayBackend",
        "n_features": n_features,
        "train_time_ms": null,
        "mse": null,
        "mae": null,
        "r2": null,
        "disabled_reason": "ndarray feature not enabled"
    })
}

fn main() {
    println!("Running backend comparison benchmarks...\n");

    let mut results = Vec::new();

    // Test with different feature counts
    for n_features in [1, 2, 4, 8] {
        println!("Testing {} features...", n_features);

        // CPU Backend
        println!("  CpuBackend...");
        results.push(benchmark_cpu_backend(n_features));

        // Ndarray Backend (if available)
        println!("  NdarrayBackend...");
        results.push(benchmark_ndarray_backend(n_features));
    }

    let output = json!({
        "results": results
    });

    // Write to file
    let mut file = File::create("benchmarks/results/backend_comparison.json")
        .expect("Failed to create output file");
    file.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .expect("Failed to write to file");

    println!("\nResults saved to benchmarks/results/backend_comparison.json");

    // Print comparison table
    println!("\nBackend Comparison (50 epochs, lr=0.01, bs=32):");
    println!(
        "{:<15} {:<12} {:>12} {:>10} {:>10} {:>10}",
        "Backend", "Features", "Time (ms)", "MSE", "MAE", "R²"
    );
    println!("{}", "-".repeat(70));

    #[cfg(feature = "ndarray")]
    for n_features in [1, 2, 4, 8] {
        let cpu_result = results
            .iter()
            .find(|r| r["backend"] == "CpuBackend" && r["n_features"] == n_features)
            .unwrap();

        let ndarray_result = results
            .iter()
            .find(|r| r["backend"] == "NdarrayBackend" && r["n_features"] == n_features)
            .unwrap();

        let cpu_time = cpu_result["train_time_ms"].as_f64().unwrap();
        let ndarray_time = ndarray_result["train_time_ms"].as_f64().unwrap();
        let speedup = if ndarray_time > 0.0 {
            cpu_time / ndarray_time
        } else {
            0.0
        };

        println!(
            "{:<15} {:<12} {:>12.2} {:>10.4} {:>10.4} {:>10.4}",
            "CpuBackend",
            n_features,
            cpu_time,
            cpu_result["mse"].as_f64().unwrap(),
            cpu_result["mae"].as_f64().unwrap(),
            cpu_result["r2"].as_f64().unwrap(),
        );

        if ndarray_time > 0.0 {
            println!(
                "{:<15} {:<12} {:>12.2} {:>10.4} {:>10.4} {:>10.4} [{:>6.1}x]",
                "NdarrayBackend",
                n_features,
                ndarray_time,
                ndarray_result["mse"].as_f64().unwrap(),
                ndarray_result["mae"].as_f64().unwrap(),
                ndarray_result["r2"].as_f64().unwrap(),
                speedup,
            );
        } else {
            println!(
                "{:<15} {:<12} {:>12.2} {:>10.4} {:>10.4} {:>10.4} [N/A]",
                "NdarrayBackend",
                n_features,
                ndarray_time,
                ndarray_result["mse"].as_f64().unwrap(),
                ndarray_result["mae"].as_f64().unwrap(),
                ndarray_result["r2"].as_f64().unwrap(),
            );
        }

        println!();
    }

    #[cfg(not(feature = "ndarray"))]
    {
        println!("\nNote: Run with --features ndarray to benchmark ndarray backend");
    }

    // Print sklearn comparison
    println!("Sklearn SGDRegressor (for reference, 1000 iterations):");
    println!(
        "{:<30} {:>12} {:>10} {:>10} {:>10}",
        "Config", "Features", "Time (ms)", "MSE", "R²"
    );
    println!("{}", "-".repeat(75));
    println!(
        "{:<30} {:<12} {:>12.2} {:>10.4} {:>10.4}",
        "lr_constant", 2, 5.09, 0.6697, 0.4889
    );
    println!(
        "{:<30} {:<12} {:>12.2} {:>10.4} {:>10.4}",
        "lr_adaptive", 2, 25.38, 0.6630, 0.4941
    );
}
