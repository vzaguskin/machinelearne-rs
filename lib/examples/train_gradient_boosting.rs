//! Example: Training a gradient boosting regressor.
//!
//! Demonstrates the gradient boosting API on synthetic data.

use machinelearne_rs::{
    backend::CpuBackend,
    ensemble::{GradientBoostedModel, GradientBoostedModelParams, GradientBoostingRegressor},
    model::InferenceModel,
    serialization::SerializableParams,
    Tensor1D, Tensor2D,
};
use std::fs;

fn main() {
    println!("=== Gradient Boosting Regression Example ===\n");

    // Generate synthetic data: y = 3*x + noise
    let n_samples = 50;
    let features_data: Vec<f32> = (0..n_samples).map(|i| i as f32 * 0.1).collect();
    let targets_data: Vec<f32> = features_data
        .iter()
        .map(|&x| 3.0 * x + (x * 10.0).sin() * 0.5) // y = 3x + sin(10x)*0.5
        .collect();

    let features = Tensor2D::<CpuBackend>::new(features_data.clone(), n_samples, 1);
    let targets = Tensor1D::<CpuBackend>::new(targets_data.clone());

    println!("Training data: {} samples", n_samples);
    println!("True function: y = 3*x + sin(10*x)*0.5\n");

    // Train gradient boosting model
    let trainer = GradientBoostingRegressor::default()
        .n_estimators(100)
        .learning_rate(0.1)
        .verbose(1);

    println!("Training gradient boosting model...\n");
    let model = trainer.fit(&features, &targets);

    println!(
        "\nModel trained with {} weak learners",
        model.n_estimators()
    );
    println!("Learning rate: {}", model.learning_rate());
    println!("Initial prediction: {:.4}", model.initial_prediction());

    // Evaluate on training data
    let predictions = model.predict_batch(&features);
    let mse: f64 = targets_data
        .iter()
        .zip(predictions.to_vec().iter())
        .map(|(&t, &p)| ((t as f64) - p).powi(2))
        .sum::<f64>()
        / n_samples as f64;
    println!("\nTraining MSE: {:.6}", mse);

    // Test predictions
    println!("\n=== Test Predictions ===");
    let test_values = vec![0.5, 1.0, 2.0, 3.0, 4.0];
    for &x in &test_values {
        let input = Tensor1D::<CpuBackend>::new(vec![x]);
        let pred = model.predict(&input);
        let true_val = 3.0 * x + (x * 10.0).sin() * 0.5;
        println!(
            "x = {:.1}: predicted = {:.3}, true = {:.3}, error = {:.3}",
            x,
            pred.to_f64(),
            true_val,
            (pred.to_f64() - true_val as f64).abs()
        );
    }

    // Save model
    let model_path = "/tmp/gradient_boosting_model.bin";
    match model.extract_params().to_bytes() {
        Ok(bytes) => {
            fs::write(model_path, &bytes).expect("Failed to write model");
            println!("\nModel saved to: {}", model_path);

            // Load and verify
            let loaded_bytes = fs::read(model_path).expect("Failed to read model");
            let params = GradientBoostedModelParams::from_bytes(&loaded_bytes)
                .expect("Failed to deserialize");
            let loaded_model: GradientBoostedModel<CpuBackend> =
                GradientBoostedModel::from_params(params).expect("Failed to create model");

            let test_input = Tensor1D::<CpuBackend>::new(vec![2.5]);
            let orig_pred = model.predict(&test_input);
            let loaded_pred = loaded_model.predict(&test_input);

            println!("Original prediction: {:.4}", orig_pred.to_f64());
            println!("Loaded prediction:    {:.4}", loaded_pred.to_f64());
            println!(
                "Serialization OK: {}",
                (orig_pred.to_f64() - loaded_pred.to_f64()).abs() < 1e-10
            );
        }
        Err(e) => {
            eprintln!("Failed to serialize model: {:?}", e);
        }
    }

    println!("\n=== Example Complete ===");
}
