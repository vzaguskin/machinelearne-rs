#!/usr/bin/env rust-script
//!
//! Compare Rust SGD performance against sklearn SGDRegressor
//!
//! Matches sklearn's SGDRegressor configuration:
//! - max_iter=1000
//! - tol=1e-3 (early stopping)
//! - learning_rate=0.01
//! - StandardScaler for feature scaling

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

/// Helper function to convert Vec<Vec<f32>> to Tensor2D<CpuBackend>
fn vec_to_tensor2d(data: &[Vec<f32>]) -> Tensor2D<CpuBackend> {
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

/// Collect metrics with different configurations
fn run_comparison() -> serde_json::Value {
    let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
        .expect("Failed to load dataset");

    // Test with 2 features (same as sklearn benchmark)
    let feature_indices: Vec<usize> = vec![0, 1];
    let subset = dataset.select_features(&feature_indices);
    let (train_dataset, _, test_dataset) = subset.split_train_val_test(0.8, 0.0);

    let train_features = train_dataset.features();
    let train_target = train_dataset.target();
    let test_features = test_dataset.features();
    let test_target = test_dataset.target();

    // Apply z-score standardization
    let (train_features_scaled, test_features_scaled) =
        standardize_features(train_features, test_features);

    let train_memory = InMemoryDataset::new(train_features_scaled, train_target.to_vec())
        .expect("Failed to create training dataset");

    let mut results = Vec::new();

    // Configuration 1: 50 epochs (current config)
    let lr = 0.01;
    let model1 = LinearRegression::<CpuBackend>::new(2);
    let optimizer = SGD::<CpuBackend>::new(lr);

    let trainer = Trainer::builder(MSELoss, optimizer, NoRegularizer)
        .batch_size(32)
        .max_epochs(50)
        .build();

    let start = Instant::now();
    let fitted1 = trainer
        .fit(model1, &train_memory)
        .expect("Failed to fit model");
    let train_time_50 = start.elapsed().as_millis();

    let test_tensor = vec_to_tensor2d(&test_features_scaled);
    let pred_tensor = fitted1.predict_batch(&test_tensor);
    let predictions: Vec<f32> = pred_tensor.to_vec().into_iter().map(|v| v as f32).collect();

    results.push(json!({
        "config": "50 epochs",
        "train_time_ms": train_time_50,
        "mse": Metrics::mse(test_target, &predictions),
        "mae": Metrics::mae(test_target, &predictions),
        "r2": Metrics::r_squared(test_target, &predictions),
    }));

    // Configuration 2: 1000 epochs (matching sklearn)
    let model2 = LinearRegression::<CpuBackend>::new(2);
    let optimizer2 = SGD::<CpuBackend>::new(lr);

    let trainer2 = Trainer::builder(MSELoss, optimizer2, NoRegularizer)
        .batch_size(32)
        .max_epochs(1000)
        .build();

    let start = Instant::now();
    let fitted2 = trainer2
        .fit(model2, &train_memory)
        .expect("Failed to fit model");
    let train_time_1000 = start.elapsed().as_millis();

    let pred_tensor2 = fitted2.predict_batch(&test_tensor);
    let predictions2: Vec<f32> = pred_tensor2
        .to_vec()
        .into_iter()
        .map(|v| v as f32)
        .collect();

    results.push(json!({
        "config": "1000 epochs (sklearn match)",
        "train_time_ms": train_time_1000,
        "mse": Metrics::mse(test_target, &predictions2),
        "mae": Metrics::mae(test_target, &predictions2),
        "r2": Metrics::r_squared(test_target, &predictions2),
    }));

    // Configuration 3: 1000 epochs with larger batch size
    let model3 = LinearRegression::<CpuBackend>::new(2);
    let optimizer3 = SGD::<CpuBackend>::new(lr);

    let trainer3 = Trainer::builder(MSELoss, optimizer3, NoRegularizer)
        .batch_size(128)
        .max_epochs(1000)
        .build();

    let start = Instant::now();
    let fitted3 = trainer3
        .fit(model3, &train_memory)
        .expect("Failed to fit model");
    let train_time_1000_bs128 = start.elapsed().as_millis();

    let pred_tensor3 = fitted3.predict_batch(&test_tensor);
    let predictions3: Vec<f32> = pred_tensor3
        .to_vec()
        .into_iter()
        .map(|v| v as f32)
        .collect();

    results.push(json!({
        "config": "1000 epochs, batch_size=128",
        "train_time_ms": train_time_1000_bs128,
        "mse": Metrics::mse(test_target, &predictions3),
        "mae": Metrics::mae(test_target, &predictions3),
        "r2": Metrics::r_squared(test_target, &predictions3),
    }));

    json!({
        "n_features": 2,
        "model": "LinearRegression",
        "results": results
    })
}

fn main() {
    println!("Running SGD comparison benchmarks...");
    let result = run_comparison();

    // Write to file
    let mut file = File::create("benchmarks/results/rust_sgd_comparison.json")
        .expect("Failed to create output file");
    file.write_all(serde_json::to_string_pretty(&result).unwrap().as_bytes())
        .expect("Failed to write to file");

    println!("\nResults saved to benchmarks/results/rust_sgd_comparison.json");

    // Print summary
    let results = result["results"].as_array().unwrap();
    println!("\nRust SGD Performance:");
    println!(
        "{:<30} {:>12} {:>10} {:>10} {:>10}",
        "Config", "Time (ms)", "MSE", "MAE", "R²"
    );
    for r in results {
        let config = r["config"].as_str().unwrap();
        let time = r["train_time_ms"].as_f64().unwrap();
        let mse = r["mse"].as_f64().unwrap();
        let mae = r["mae"].as_f64().unwrap();
        let r2 = r["r2"].as_f64().unwrap();
        println!(
            "{:<30} {:>12.2} {:>10.4} {:>10.4} {:>10.4}",
            config, time, mse, mae, r2
        );
    }

    println!("\nSklearn SGDRegressor (for comparison):");
    println!(
        "{:<30} {:>12} {:>10} {:>10} {:>10}",
        "Config", "Time (ms)", "MSE", "MAE", "R²"
    );
    println!(
        "{:<30} {:>12.2} {:>10.4} {:>10.4} {:>10.4}",
        "lr_constant (1000 iter)", 5.09, 0.6697, 0.5978, 0.4889
    );
    println!(
        "{:<30} {:>12.2} {:>10.4} {:>10.4} {:>10.4}",
        "lr_adaptive (1000 iter)", 25.38, 0.6630, 0.6061, 0.4941
    );
}
