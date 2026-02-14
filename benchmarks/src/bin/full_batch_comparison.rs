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
    println!("Full-Batch vs Mini-Batch Comparison for Rust");
    println!("==============================================\n");

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

    // Test configurations
    let test_cases = vec![
        (2, 5, 16512, "Full-batch, 5 epochs (like sklearn)"),
        (2, 50, 16512, "Full-batch, 50 epochs"),
        (2, 50, 32, "Mini-batch (bs=32), 50 epochs (current default)"),
        (2, 1000, 32, "Mini-batch (bs=32), 1000 epochs"),
    ];

    for (n_features, n_epochs, batch_size, desc) in test_cases {
        println!("Test: {}", desc);
        println!(
            "Features: {}, Epochs: {}, Batch size: {}",
            n_features, n_epochs, batch_size
        );

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
        let optimizer = SGD::new(0.01);
        let regularizer = NoRegularizer;

        let trainer_cpu = TrainerBuilder::new(loss_fn, optimizer, regularizer)
            .batch_size(batch_size)
            .max_epochs(n_epochs)
            .build();

        let fitted_cpu = trainer_cpu
            .fit(model_cpu, &train_dataset)
            .expect("Training failed");
        let training_time_cpu = start.elapsed();

        // Predictions
        let predictions_cpu: Vec<f32> = test_selected
            .iter()
            .map(|sample| {
                let tensor_input = Tensor1D::<CpuBackend>::new(sample.clone());
                let result = fitted_cpu.predict(&tensor_input);
                result.to_f64() as f32
            })
            .collect();

        // Metrics
        let (mse, mae, r_squared) = compute_metrics(&predictions_cpu, &test_targets);

        println!("CPU Backend:");
        println!(
            "  Training time: {:.2} ms",
            training_time_cpu.as_secs_f64() * 1000.0
        );
        println!("  MSE: {:.4}", mse);
        println!("  MAE: {:.4}", mae);
        println!("  R²: {:.4}", r_squared);
        println!();
    }

    // Compare with sklearn numbers
    println!("Sklearn SGDRegressor (for reference):");
    println!("  lr_constant: 5.09 ms (2 features, full-batch, converges in 5 epochs)");
    println!("  lr_adaptive: 25.38 ms (2 features, full-batch, converges in 5 epochs)");
    println!("  Training details (verbose output):");
    println!("    -- Epoch 1, T: 100, Avg. loss: 0.584");
    println!("    -- Epoch 2, T: 200, Avg. loss: 0.574");
    println!("    -- Epoch 3, T: 300, Avg. loss: 0.571");
    println!("    -- Epoch 4, T: 400, Avg. loss: 0.570");
    println!("    -- Epoch 5, T: 500, Avg. loss: 0.566");
    println!("  Converges in 5 epochs (not full 1000)");
}
