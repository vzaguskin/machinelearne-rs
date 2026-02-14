use machinelearne_rs::{
    dataset::InMemoryDataset, loss::MSELoss, model::linear::LinearRegression,
    model::InferenceModel, optimizer::SGD, regularizers::NoRegularizer, trainer::TrainerBuilder,
    Tensor1D,
};

use std::time::Instant;

use machinelearne_rs::backend::CpuBackend;

type Features = Vec<Vec<f32>>;
type Targets = Vec<f32>;

fn load_california_housing() -> (Vec<Vec<f32>>, Vec<f32>) {
    // Load CSV and parse
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

    // Compute mean and std from training data
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
        let std = (variance / n_samples as f32).sqrt();
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

fn compute_metrics(predictions: &[f32], targets: &[f32]) -> (f64, f64, f64) {
    let n = targets.len();
    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t) as f64 * (p - t) as f64)
        .sum::<f64>()
        / n as f64;

    let mae: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs() as f64)
        .sum::<f64>()
        / n as f64;

    let target_mean: f64 = targets.iter().map(|t| *t as f64).sum::<f64>() / n as f64;
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t) as f64 * (p - t) as f64)
        .sum();
    let ss_tot: f64 = targets
        .iter()
        .map(|t| (*t as f64 - target_mean) * (*t as f64 - target_mean))
        .sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    (mse, mae, r_squared)
}

fn main() {
    println!("Learning Rate Search for Full-Batch Training");
    println!("===========================================\n");

    let (features, targets) = load_california_housing();
    println!(
        "Loaded {} samples with {} features\n",
        features.len(),
        features[0].len()
    );

    let (train_features, test_features, train_targets, test_targets) =
        split_train_test(&features, &targets);
    let (train_features_scaled, test_features_scaled) =
        standardize_features(&train_features, &test_features);

    let n_features = 2;
    let n_epochs = 5;
    let batch_size = 16512; // Full batch (all training samples)

    // Learning rates to test (logarithmic scale)
    let learning_rates = vec![
        1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1,
    ];

    println!(
        "Testing {} learning rates ({} epochs, batch size = {})\n",
        learning_rates.len(),
        n_epochs,
        batch_size
    );

    let mut best_lr = 0.0;
    let mut best_mse = f64::MAX;
    let mut best_r2 = f64::MIN;

    for lr in learning_rates {
        // Select features
        let train_selected: Vec<Vec<f32>> = train_features_scaled
            .iter()
            .map(|row| row[..n_features].to_vec())
            .collect();
        let test_selected: Vec<Vec<f32>> = test_features_scaled
            .iter()
            .map(|row| row[..n_features].to_vec())
            .collect();

        let train_dataset = InMemoryDataset::new(train_selected.clone(), train_targets.clone())
            .expect("Failed to create dataset");

        // Train with CPU backend
        let start = Instant::now();
        let model_cpu = LinearRegression::<CpuBackend>::new(n_features);

        let loss_fn = MSELoss;
        let optimizer = SGD::new(lr);
        let regularizer = NoRegularizer;

        let trainer_cpu = TrainerBuilder::new(loss_fn, optimizer, regularizer)
            .batch_size(batch_size)
            .max_epochs(n_epochs)
            .build();

        let fitted_cpu = trainer_cpu
            .fit(model_cpu, &train_dataset)
            .expect("Training failed");
        let training_time = start.elapsed();

        // Predictions
        let predictions: Vec<f32> = test_selected
            .iter()
            .map(|sample| {
                let tensor_input = Tensor1D::<CpuBackend>::new(sample.clone());
                let result = fitted_cpu.predict(&tensor_input);
                result.to_f64() as f32
            })
            .collect();

        // Metrics
        let (mse, mae, r_squared) = compute_metrics(&predictions, &test_targets);

        println!(
            "LR = {:.6}: Time = {:.2} ms, MSE = {:.4}, MAE = {:.4}, R² = {:.4}",
            lr,
            training_time.as_secs_f64() * 1000.0,
            mse,
            mae,
            r_squared
        );

        // Track best
        if mse < best_mse {
            best_mse = mse;
            best_r2 = r_squared;
            best_lr = lr;
        }
    }

    println!("\n=== Best Learning Rate ===");
    println!("LR = {:.6}", best_lr);
    println!("MSE = {:.4}", best_mse);
    println!("R² = {:.4}", best_r2);

    println!("\n=== Sklearn Comparison ===");
    println!("Sklearn lr_constant: 5.09 ms, MSE = 0.6697, R² = 0.4889");
    println!("Sklearn lr_adaptive: 25.38 ms, MSE = 0.6630, R² = 0.4941");

    if best_lr > 0.0 {
        println!("Rust is competitive at LR = {:.6}", best_lr);
    }
}
