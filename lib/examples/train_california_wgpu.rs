//! California Housing linear regression training on GPU using WGPU backend.
//!
//! This example demonstrates full ML training pipeline on GPU:
//! - Loads California Housing dataset
//! - Displays GPU adapter information to verify GPU usage
//! - Standardizes features using GPU compute shaders
//! - Trains linear regression model on GPU
//! - Reports training time and evaluation metrics
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example train_california_wgpu --features wgpu
//! ```
//!
//! ## Requirements
//!
//! - GPU with Vulkan (Linux/Windows), Metal (macOS), or D3D12 (Windows) support
//! - The `wgpu` feature enabled
//!
//! ## Dataset: California Housing
//!
//! - 20,640 samples, 8 features
//! - Target: Median house value (in $100k)
//! - Expected R² ≈ 0.55-0.60 for linear regression

#[cfg(feature = "wgpu")]
use machinelearne_rs::{
    backend::{Tensor2D, WgpuBackend, WgpuDevice},
    dataset::InMemoryDataset,
    loss::MSELoss,
    model::{
        linear::{LinearModel, LinearRegression, Unfitted},
        InferenceModel,
    },
    optimizer::SGD,
    preprocessing::{FittedTransformer, StandardScaler, Transformer},
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor1D,
};
#[cfg(feature = "wgpu")]
use std::fs::File;
#[cfg(feature = "wgpu")]
use std::io::{BufRead, BufReader};
#[cfg(feature = "wgpu")]
use std::time::Instant;

#[cfg(feature = "wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== California Housing Linear Regression on GPU (WGPU) ===\n");

    // Display GPU adapter information
    println!("1. GPU Adapter Information:");
    println!("   Checking available GPU adapters...\n");

    // Note: enumerate_adapters is async, use pollster to block
    let adapters = pollster::block_on(WgpuDevice::enumerate_adapters());
    for (i, adapter) in adapters.iter().enumerate() {
        println!(
            "   [{}] {} ({}, {})",
            i, adapter.name, adapter.backend, adapter.device_type
        );
    }
    println!();

    // Load the dataset
    println!("2. Loading California Housing dataset...");
    let (x_raw, y) = load_california_housing()?;

    let n_samples = x_raw.len();
    let n_features = x_raw[0].len();
    println!(
        "   Loaded {} samples with {} features",
        n_samples, n_features
    );

    // Split into train/test (80/20)
    let split_idx = (n_samples as f32 * 0.8) as usize;
    let (train_x, test_x) = x_raw.split_at(split_idx);
    let (train_y, test_y) = y.split_at(split_idx);

    println!(
        "   Train: {} samples, Test: {} samples",
        train_x.len(),
        test_x.len()
    );
    println!();

    // Create WGPU tensors
    println!("3. Creating GPU tensors...");

    // Flatten training data
    let train_flat: Vec<f32> = train_x.iter().flatten().copied().collect();
    let test_flat: Vec<f32> = test_x.iter().flatten().copied().collect();

    let x_train: Tensor2D<WgpuBackend> = Tensor2D::new(train_flat, train_x.len(), n_features);
    let _y_train: Tensor1D<WgpuBackend> = Tensor1D::new(train_y.to_vec());

    let x_test: Tensor2D<WgpuBackend> = Tensor2D::new(test_flat, test_x.len(), n_features);
    let y_test: Tensor1D<WgpuBackend> = Tensor1D::new(test_y.to_vec());

    println!(
        "   Created {}x{} training tensor",
        train_x.len(),
        n_features
    );
    println!("   Created {}x{} test tensor", test_x.len(), n_features);
    println!();

    // Standardize features on GPU
    println!("4. Standardizing features on GPU...");
    let scaler = StandardScaler::<WgpuBackend>::new();
    let fitted_scaler = scaler.fit(&x_train)?;
    let x_train_scaled = fitted_scaler.transform(&x_train)?;
    let x_test_scaled = fitted_scaler.transform(&x_test)?;

    // Print mean/std to verify GPU computation
    let mean = fitted_scaler.mean().to_vec();
    let std = fitted_scaler.std().to_vec();
    println!(
        "   Feature means: {:?}",
        mean.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>()
    );
    println!(
        "   Feature stds:  {:?}",
        std.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>()
    );
    println!();

    // Create dataset
    let (n_train, _) = x_train_scaled.shape();
    let train_flat_scaled = x_train_scaled.ravel().to_vec();
    let train_x_scaled: Vec<Vec<f32>> = (0..n_train)
        .map(|r| {
            (0..n_features)
                .map(|c| train_flat_scaled[r * n_features + c] as f32)
                .collect()
        })
        .collect();

    let dataset = InMemoryDataset::new(
        train_x_scaled,
        train_y.to_vec().iter().map(|&v| v as f32).collect(),
    )?;

    // Create model and training components
    println!("5. Setting up linear regression on GPU...");
    let model: LinearModel<WgpuBackend, Unfitted> =
        LinearRegression::<WgpuBackend>::new(n_features);
    let loss = MSELoss;
    let optimizer = SGD::new(0.01);
    let regularizer = NoRegularizer;

    let trainer = Trainer::builder(loss, optimizer, regularizer)
        .batch_size(128)
        .max_epochs(100)
        .build();

    println!("   Learning rate: 0.01");
    println!("   Batch size: 128");
    println!("   Max epochs: 100");
    println!();

    // Train on GPU
    println!("6. Training on GPU...");
    let start = Instant::now();
    let fitted_model = trainer.fit(model, &dataset)?;
    let training_time = start.elapsed();

    println!("   Training completed in {:?}", training_time);
    println!();

    // Evaluate on test set
    println!("7. Evaluating on test set...");
    let predictions = fitted_model.predict_batch(&x_test_scaled);
    let pred_vec = predictions.to_vec();
    let y_test_vec = y_test.to_vec();

    // Calculate metrics
    let n_test = test_y.len();
    let mse: f64 = pred_vec
        .iter()
        .zip(y_test_vec.iter())
        .map(|(p, t)| {
            let diff = p - t;
            diff * diff
        })
        .sum::<f64>()
        / n_test as f64;

    let mae: f64 = pred_vec
        .iter()
        .zip(y_test_vec.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n_test as f64;

    // R² = 1 - SS_res / SS_tot
    let y_mean: f64 = y_test_vec.iter().sum::<f64>() / n_test as f64;
    let ss_tot: f64 = y_test_vec.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = pred_vec
        .iter()
        .zip(y_test_vec.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    println!("   MSE:  {:.4}", mse);
    println!("   MAE:  {:.4}", mae);
    println!("   R²:   {:.4}", r2);
    println!();

    // Display learned parameters
    println!("8. Learned Parameters:");
    let params = fitted_model.extract_params();
    println!(
        "   Weights: {:?}",
        params
            .weights
            .iter()
            .map(|w| format!("{:.4}", w))
            .collect::<Vec<_>>()
    );
    println!("   Bias:    {:.4}", params.bias);
    println!();

    // Sample predictions
    println!("9. Sample Predictions (first 5 test samples):");
    println!(
        "   {:<8} {:>12} {:>12} {:>10}",
        "Sample", "Actual ($k)", "Predicted ($k)", "Error ($k)"
    );
    println!("   {}", "-".repeat(48));
    for i in 0..5 {
        let actual = test_y[i] * 100.0;
        let predicted = pred_vec[i] as f32 * 100.0;
        let error = (predicted - actual).abs();
        println!(
            "   {:<8} {:>12.0} {:>12.0} {:>10.0}",
            i + 1,
            actual as i32,
            predicted as i32,
            error as i32
        );
    }

    println!();
    println!("=== GPU Training Complete ===");

    // Performance summary
    println!();
    println!("Performance Summary:");
    println!("   Total training time: {:?}", training_time);
    println!(
        "   Samples/second:      {:.0}",
        n_train as f64 / training_time.as_secs_f64()
    );
    println!("   R² score:            {:.4} (expected ~0.55-0.60)", r2);

    if r2 >= 0.50 {
        println!("   Status:              OK - Model trained successfully on GPU");
    } else {
        println!("   Status:              WARNING - Low R², consider more epochs");
    }

    Ok(())
}

#[cfg(feature = "wgpu")]
fn load_california_housing() -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
    // Try multiple possible paths
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

    let file = file.ok_or("Could not find california_housing.csv")?;
    let reader = BufReader::new(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            // Skip header
            continue;
        }

        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();

        if values.len() >= 9 {
            // Last column is the target (MedHouseVal)
            x.push(values[0..8].to_vec());
            y.push(values[8]);
        }
    }

    Ok((x, y))
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("This example requires the 'wgpu' feature to be enabled.");
    println!("Run with: cargo run --example train_california_wgpu --features wgpu");
}
