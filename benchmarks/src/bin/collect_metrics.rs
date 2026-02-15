#!/usr/bin/env rust-script
//!
//! Collect metrics from machinelearne-rs benchmarks
//!
//! This runs each training configuration once and records:
//! - Training time
//! - MSE
//! - MAE
//! - R²

use benchmarks::data::CaliforniaHousingDataset;
use benchmarks::metrics::Metrics;
use machinelearne_rs::{
    backend::Tensor2D,
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{linear::LinearRegression, InferenceModel},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
    CpuBackend,
};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Simple black_box function to prevent optimization
#[inline(never)]
fn black_box<T>(dummy: T) -> T {
    std::hint::black_box(dummy)
}

/// Helper function to convert Vec<Vec<f32>> to Tensor2D<CpuBackend>
fn vec_to_tensor2d(data: &[Vec<f32>]) -> Tensor2D<CpuBackend> {
    let flat: Vec<f32> = data.iter().flatten().copied().collect();
    Tensor2D::new(flat, data.len(), data.first().map(|v| v.len()).unwrap_or(0))
}

/// Apply z-score standardization to features
///
/// This computes mean and std from training data, then applies:
/// z-score = (x - mean) / std
/// to both train and test data using the same statistics.
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

    // Compute mean and std for each feature from training data
    let mut means = vec![0.0_f32; n_features];
    let mut stds = vec![0.0_f32; n_features];

    for feature_idx in 0..n_features {
        // Compute mean
        let sum: f32 = (0..n_samples).map(|i| train_features[i][feature_idx]).sum();
        means[feature_idx] = sum / n_samples as f32;

        // Compute variance
        let variance: f32 = (0..n_samples)
            .map(|i| {
                let diff = train_features[i][feature_idx] - means[feature_idx];
                diff * diff
            })
            .sum();
        stds[feature_idx] = f32::sqrt(variance / n_samples as f32);

        // Prevent division by zero
        if stds[feature_idx] < 1e-6 {
            stds[feature_idx] = 1.0;
        }
    }

    // Apply z-score to training data
    let train_scaled: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            (0..n_features)
                .map(|j| (train_features[i][j] - means[j]) / stds[j])
                .collect()
        })
        .collect();

    // Apply z-score to test data (using train's mean/std)
    let test_scaled: Vec<Vec<f32>> = (0..test_features.len())
        .map(|i| {
            (0..n_features)
                .map(|j| (test_features[i][j] - means[j]) / stds[j])
                .collect()
        })
        .collect();

    (train_scaled, test_scaled)
}

/// Collect metrics for a given feature count
fn collect_metrics(n_features: usize) -> serde_json::Value {
    let feature_indices: Vec<usize> = (0..n_features).collect();

    let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&feature_indices);

    // Split into train and test (function returns train, val, test)
    let (train_dataset, _, test_dataset) = subset.split_train_val_test(0.8, 0.0);

    // Get raw features and targets
    let train_features = train_dataset.features();
    let train_target = train_dataset.target();
    let test_features = test_dataset.features();
    let test_target = test_dataset.target();

    // Apply z-score standardization
    let (train_features_scaled, test_features_scaled) =
        standardize_features(train_features, test_features);

    // Create in-memory datasets with scaled features
    let train_memory = InMemoryDataset::new(train_features_scaled, train_target.to_vec())
        .expect("Failed to create training dataset");

    // Use learning rate appropriate for standardized features
    let lr = 0.01;
    let model = LinearRegression::<CpuBackend>::new(n_features);
    let optimizer = SGD::<CpuBackend>::new(lr);

    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(32)
        .max_epochs(50)
        .build();

    // Time the training
    let start = Instant::now();
    let fitted = trainer
        .fit(model, &train_memory)
        .expect("Failed to fit model");
    let train_time_ms = start.elapsed().as_millis();

    // Predict on test set
    let test_tensor = vec_to_tensor2d(&test_features_scaled);
    let pred_tensor = fitted.predict_batch(black_box(&test_tensor));

    let predictions: Vec<f32> = pred_tensor.to_vec().into_iter().map(|v| v as f32).collect();

    let mse = Metrics::mse(test_target, &predictions);
    let mae = Metrics::mae(test_target, &predictions);
    let r2 = Metrics::r_squared(test_target, &predictions);

    json!({
        "n_features": n_features,
        "model": "LinearRegression",
        "train_time_ms": train_time_ms,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    })
}

fn main() {
    let mut results = Vec::new();

    // Collect metrics for different feature counts
    for n_features in [1, 2, 4, 8] {
        println!("Collecting metrics for {} features...", n_features);
        results.push(collect_metrics(n_features));
    }

    let output = json!({
        "results": results
    });

    // Write to file
    let mut file =
        File::create("benchmarks/results/rust_metrics.json").expect("Failed to create output file");
    file.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .expect("Failed to write to file");

    println!("\nMetrics collected and saved to benchmarks/results/rust_metrics.json");
}
