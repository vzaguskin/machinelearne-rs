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

/// Collect metrics for a given feature count
fn collect_metrics(n_features: usize) -> serde_json::Value {
    let feature_indices: Vec<usize> = (0..n_features).collect();

    let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
        .expect("Failed to load dataset");
    let subset = dataset.select_features(&feature_indices);
    let (train_dataset, test_dataset) = subset.split(0.8);
    let train_memory = train_dataset
        .to_in_memory_dataset()
        .expect("Failed to create dataset");

    let model = LinearRegression::<CpuBackend>::new(n_features);
    // Use lower learning rate for multi-feature models to avoid NaN
    let lr = if n_features == 1 { 0.01 } else { 0.001 };
    let optimizer = SGD::new(lr);
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
    let test_x = test_dataset.features().to_vec();
    let test_y = test_dataset.target().to_vec();
    let pred_tensor = fitted.predict_batch(black_box(&vec_to_tensor2d(&test_x)));

    let predictions: Vec<f32> = pred_tensor.to_vec().into_iter().map(|v| v as f32).collect();

    let mse = Metrics::mse(&test_y, &predictions);
    let mae = Metrics::mae(&test_y, &predictions);
    let r2 = Metrics::r_squared(&test_y, &predictions);

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
