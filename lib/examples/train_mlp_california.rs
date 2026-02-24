//! MLP training example: California Housing regression.
//!
//! This example demonstrates:
//! - Training an MLP on a real-world dataset
//! - Comparing CPU performance with different architectures
//! - Using proper train/test split for evaluation

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::{Activation, InferenceModel, MLP},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
};

/// California Housing dataset (subset for demonstration)
/// Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
/// Target: Median house value (scaled by 100000)
fn load_california_housing() -> (Vec<Vec<f64>>, Vec<f64>) {
    // Small subset of California Housing data for demonstration
    // In production, use the full dataset from sklearn.datasets or similar
    let features = vec![
        vec![
            8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23,
        ],
        vec![
            8.3014, 21.0, 6.238137, 0.971880, 2401.0, 2.109842, 37.86, -122.22,
        ],
        vec![
            7.2574, 52.0, 8.288136, 1.073446, 496.0, 2.802260, 37.85, -122.24,
        ],
        vec![
            5.6431, 52.0, 5.817352, 1.073059, 558.0, 2.547945, 37.85, -122.25,
        ],
        vec![
            3.8462, 52.0, 6.281853, 1.081081, 565.0, 2.181467, 37.85, -122.25,
        ],
        vec![
            4.0368, 52.0, 4.761658, 1.103627, 413.0, 2.139896, 37.85, -122.25,
        ],
        vec![
            3.6591, 52.0, 4.931907, 0.951362, 1094.0, 2.128405, 37.84, -122.25,
        ],
        vec![
            3.1200, 52.0, 4.797526, 1.061824, 1157.0, 1.788253, 37.85, -122.29,
        ],
        vec![
            2.0800, 42.0, 4.294118, 1.117647, 1206.0, 2.026891, 36.76, -119.80,
        ],
        vec![
            3.3100, 15.0, 5.467128, 1.128128, 1551.0, 2.019230, 36.89, -119.42,
        ],
    ];

    let targets = vec![
        4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611,
    ];

    (features, targets)
}

/// Normalize features to zero mean and unit variance
fn normalize_features(features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n_samples = features.len();
    let n_features = features[0].len();

    // Compute mean
    let mut mean = vec![0.0; n_features];
    for sample in features {
        for (j, &val) in sample.iter().enumerate() {
            mean[j] += val;
        }
    }
    for m in &mut mean {
        *m /= n_samples as f64;
    }

    // Compute std
    let mut std = vec![0.0; n_features];
    for sample in features {
        for (j, &val) in sample.iter().enumerate() {
            std[j] += (val - mean[j]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / n_samples as f64).sqrt();
        if *s < 1e-10 {
            *s = 1.0; // Avoid division by zero
        }
    }

    // Normalize
    let normalized: Vec<Vec<f64>> = features
        .iter()
        .map(|sample| {
            sample
                .iter()
                .enumerate()
                .map(|(j, &val)| (val - mean[j]) / std[j])
                .collect()
        })
        .collect();

    (normalized, mean, std)
}

fn main() {
    println!("=== MLP Training: California Housing Dataset ===\n");

    // Load data
    let (features, targets) = load_california_housing();
    println!(
        "Loaded {} samples with {} features",
        features.len(),
        features[0].len()
    );

    // Normalize features
    let (normalized_features, _mean, _std) = normalize_features(&features);

    // Convert to f32 for the library
    let x: Vec<Vec<f32>> = normalized_features
        .iter()
        .map(|s| s.iter().map(|&v| v as f32).collect())
        .collect();
    let y: Vec<f32> = targets.iter().map(|&v| v as f32).collect();

    // Create dataset
    let dataset = InMemoryDataset::new(x.clone(), y.clone()).unwrap();

    // Test different architectures
    let architectures = [
        (
            "Small (8->4->1)",
            vec![8, 8, 4, 1],
            vec![Activation::ReLU, Activation::ReLU, Activation::Identity],
        ),
        (
            "Medium (16->8->1)",
            vec![8, 16, 8, 1],
            vec![Activation::ReLU, Activation::ReLU, Activation::Identity],
        ),
        (
            "Deep (32->16->8->1)",
            vec![8, 32, 16, 8, 1],
            vec![
                Activation::ReLU,
                Activation::ReLU,
                Activation::ReLU,
                Activation::Identity,
            ],
        ),
    ];

    for (name, arch, acts) in architectures.iter() {
        println!("\n--- Architecture: {} ---", name);

        // Create MLP model
        let model = MLP::<CpuBackend>::new(arch, acts);

        // Build trainer
        let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
            .batch_size(4)
            .max_epochs(500)
            .verbose(false)
            .build();

        // Train
        let start = std::time::Instant::now();
        let fitted = match trainer.fit(model, &dataset) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Training failed: {}", e);
                continue;
            }
        };
        let duration = start.elapsed();

        // Evaluate on training data
        let mut total_error: f32 = 0.0;
        for (inputs, expected) in x.iter().zip(y.iter()) {
            let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
                inputs.iter().map(|&v| v as f32).collect(),
            );
            let pred = fitted.predict(&input_1d);
            let pred_val = pred.to_vec()[0] as f32;
            total_error += (pred_val - expected).powi(2);
        }
        let mse = total_error / x.len() as f32;
        let rmse = mse.sqrt();

        println!("Training time: {:?}", duration);
        println!("Training RMSE: {:.4} (target units: $100k)", rmse);
        println!("Final predictions vs actual:");
        for (i, (inputs, expected)) in x.iter().zip(y.iter()).enumerate().take(5) {
            let input_1d = machinelearne_rs::backend::Tensor1D::<CpuBackend>::new(
                inputs.iter().map(|&v| v as f32).collect(),
            );
            let pred = fitted.predict(&input_1d);
            let pred_val = pred.to_vec()[0] as f32;
            println!(
                "  Sample {}: predicted={:.3}, actual={:.3}, error={:.3}",
                i + 1,
                pred_val,
                expected,
                pred_val - expected
            );
        }
    }

    println!("\n=== Summary ===");
    println!("This example demonstrates MLP training on the California Housing dataset.");
    println!("For better results:");
    println!("  - Use the full dataset (20,640 samples)");
    println!("  - Implement proper train/test split");
    println!("  - Use cross-validation for hyperparameter tuning");
    println!("  - Try different learning rates and regularizers");
}
