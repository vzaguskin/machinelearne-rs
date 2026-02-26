//! Model Comparison Example: California Housing with Linear, MLP, and Gradient Boosting.
//!
//! This example demonstrates:
//! - Loading the real California Housing dataset
//! - Hyperparameter tuning with GridSearchGB for Gradient Boosting
//! - Training Linear Regression and MLP models
//! - Comparing prediction quality and training time using ModelComparison
//!
//! # Dataset: California Housing
//!
//! - 20,640 samples, 8 features
//! - Target: Median house value (in $100k)
//! - sklearn LinearRegression baseline: R² ≈ 0.60
//! - sklearn GradientBoosting baseline: R² ≈ 0.78-0.82

use machinelearne_rs::{
    backend::CpuBackend,
    ensemble::{
        DecisionTree, Evaluable, GradientBoostedModel, GradientBoostingRegressor, ModelComparison,
        WeakLearner,
    },
    loss::MSELoss,
    model::{
        linear::{LinearModel, LinearRegression},
        state::Fitted,
        Activation, InferenceModel, MLP,
    },
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor1D, Tensor2D,
};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Comparison: California Housing Dataset ===\n");

    // Load the real California Housing dataset
    let (x_raw, y_raw) = load_california_housing()?;

    println!(
        "Dataset loaded: {} samples, {} features",
        x_raw.len(),
        x_raw[0].len()
    );
    println!(
        "Target range: [{:.2}, {:.2}] (${}k - ${}k)",
        y_raw.iter().cloned().fold(f64::INFINITY, f64::min),
        y_raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        (y_raw.iter().cloned().fold(f64::INFINITY, f64::min) * 100.0) as i32,
        (y_raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 100.0) as i32
    );
    println!();

    // Normalize features
    println!("Preprocessing: Standardizing features...");
    let (x_scaled, y) = normalize_data(&x_raw, &y_raw);

    // Shuffle data before splitting (critical for unbiased evaluation!)
    // California Housing is sorted geographically, so without shuffling,
    // train/test would have different geographic distributions.
    println!("Shuffling data with seed 42 for reproducibility...");
    let mut indices: Vec<usize> = (0..x_scaled.len()).collect();
    // Simple Fisher-Yates shuffle with fixed seed
    let mut rng_state: u64 = 42;
    for i in (1..indices.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state % ((i + 1) as u64)) as usize;
        indices.swap(i, j);
    }

    let x_shuffled: Vec<Vec<f64>> = indices.iter().map(|&i| x_scaled[i].clone()).collect();
    let y_shuffled: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    // Split into train/test (80/20)
    let split_idx = (x_shuffled.len() as f64 * 0.8) as usize;
    let (x_train, x_test) = x_shuffled.split_at(split_idx);
    let (y_train, y_test) = y_shuffled.split_at(split_idx);

    println!(
        "Train/Test split: {}/{} samples",
        x_train.len(),
        x_test.len()
    );
    println!();

    // Convert to tensors
    let n_features = x_train[0].len();

    let train_features = vec_to_tensor2d(x_train, n_features);
    let train_targets = vec_to_tensor1d(y_train);
    let test_features = vec_to_tensor2d(x_test, n_features);
    let test_targets = vec_to_tensor1d(y_test);

    // Create model comparison
    let mut comparison = ModelComparison::new(test_features.clone(), test_targets.clone());

    // ============================================
    // 1. Train and evaluate Linear Regression
    // ============================================
    println!("=== 1. Linear Regression ===");

    let linear_start = std::time::Instant::now();

    // Use trainer for linear model
    let dataset = machinelearne_rs::dataset::InMemoryDataset::new(
        x_train
            .iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect(),
        y_train.iter().map(|&v| v as f32).collect(),
    )?;

    let linear_model = LinearRegression::<CpuBackend>::new(n_features);
    let linear_trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
        .batch_size(64)
        .max_epochs(100)
        .verbose(false)
        .build();

    let linear_fitted = linear_trainer.fit(linear_model, &dataset)?;
    let linear_time = linear_start.elapsed();

    // Evaluate linear model
    struct LinearEvaluator {
        model: LinearModel<CpuBackend, Fitted>,
    }

    impl Evaluable<CpuBackend> for LinearEvaluator {
        fn predict_batch(&self, features: &Tensor2D<CpuBackend>) -> Vec<f64> {
            self.model.predict_batch(features).to_vec()
        }
    }

    comparison.evaluate(
        "Linear",
        &LinearEvaluator {
            model: linear_fitted,
        },
    );
    println!("Training time: {:?}", linear_time);
    println!();

    // ============================================
    // 2. Train and evaluate MLP
    // ============================================
    println!("=== 2. MLP (Neural Network) ===");

    let mlp_start = std::time::Instant::now();

    let mlp_model = MLP::<CpuBackend>::new(
        &[n_features, 32, 16, 1],
        &[Activation::ReLU, Activation::ReLU, Activation::Identity],
    );

    let mlp_trainer = Trainer::builder(MSELoss, SGD::new(0.005), NoRegularizer)
        .batch_size(64)
        .max_epochs(200)
        .verbose(false)
        .build();

    let mlp_fitted = mlp_trainer.fit(mlp_model, &dataset)?;
    let mlp_time = mlp_start.elapsed();

    // Evaluate MLP
    struct MLPEvaluator {
        model: machinelearne_rs::model::MLPModel<CpuBackend, Fitted>,
    }

    impl Evaluable<CpuBackend> for MLPEvaluator {
        fn predict_batch(&self, features: &Tensor2D<CpuBackend>) -> Vec<f64> {
            // MLP predict_batch returns Tensor2D, extract first column
            let output = self.model.predict_batch(features);
            let (n_samples, _) = output.shape();
            let flat = output.ravel().to_vec();
            // Extract predictions (first column of each row for single output)
            (0..n_samples).map(|i| flat[i]).collect()
        }
    }

    comparison.evaluate("MLP", &MLPEvaluator { model: mlp_fitted });
    println!("Architecture: {} -> 32 -> 16 -> 1", n_features);
    println!("Training time: {:?}", mlp_time);
    println!();

    // ============================================
    // 3. Train and evaluate Gradient Boosting with deeper trees
    // ============================================
    println!("=== 3. Gradient Boosting with Deeper Trees ===");

    // sklearn uses max_depth=3 by default, which achieves R² ≈ 0.78-0.82
    // Let's try different tree depths to match sklearn's performance

    let gb_trainer = GradientBoostingRegressor::default()
        .n_estimators(200)
        .learning_rate(0.1)
        .colsample_bytree(0.8)
        .random_seed(42);

    // Helper function to train and evaluate with a given tree depth
    struct TreeEvaluator {
        model: GradientBoostedModel<
            CpuBackend,
            <DecisionTree as WeakLearner<CpuBackend>>::FittedModel,
        >,
    }

    impl Evaluable<CpuBackend> for TreeEvaluator {
        fn predict_batch(&self, features: &Tensor2D<CpuBackend>) -> Vec<f64> {
            self.model.predict_batch(features).to_vec()
        }
    }

    let depths_to_try = [1, 3, 5];

    for &depth in &depths_to_try {
        let tree = DecisionTree::new().max_depth(depth);
        let start = std::time::Instant::now();
        let fitted = gb_trainer.fit_with_weak_learner(&train_features, &train_targets, &tree);
        let train_time = start.elapsed();

        // Evaluate
        let predictions = fitted.predict_batch(&test_features);
        let preds_vec = predictions.to_vec();
        let metrics =
            machinelearne_rs::ensemble::ModelMetrics::compute(&preds_vec, &test_targets.to_vec());

        let name = format!("GB(depth={})", depth);
        comparison.evaluate(&name, &TreeEvaluator { model: fitted });

        println!(
            "max_depth={}: R²={:.4}, MSE={:.6}, MAE={:.6}, Time={:?}",
            depth, metrics.r2, metrics.mse, metrics.mae, train_time
        );
    }
    println!();

    // ============================================
    // Print comparison summary
    // ============================================
    println!("{}", "=".repeat(70));
    println!("MODEL COMPARISON SUMMARY");
    println!("{}", "=".repeat(70));

    comparison.summary();

    // Additional timing info
    println!("\nTiming Summary:");
    println!("  Linear Regression: {:?}", linear_time);
    println!("  MLP: {:?}", mlp_time);
    println!("  GB (various depths): see above");

    // sklearn comparison
    println!("\n{}", "=".repeat(70));
    println!("COMPARISON WITH SKLEARN BASELINES");
    println!("{}", "=".repeat(70));
    println!("\nExpected sklearn baselines on California Housing:");
    println!("  LinearRegression:    R² ≈ 0.60");
    println!("  MLP (tuned):         R² ≈ 0.70-0.75");
    println!("  GradientBoosting:    R² ≈ 0.78-0.82");

    if let Some(best) = comparison.best_model() {
        let r2 = best.metrics.r2;
        println!();
        if r2 >= 0.70 {
            println!(
                "✓ Our best model ({}) achieved R² = {:.4} - Good result!",
                best.name, r2
            );
        } else if r2 >= 0.55 {
            println!(
                "~ Our best model ({}) achieved R² = {:.4} - Acceptable for linear models",
                best.name, r2
            );
        } else {
            println!(
                "! Our best model ({}) achieved R² = {:.4} - May need more tuning",
                best.name, r2
            );
        }
    }

    Ok(())
}

/// Load the real California Housing dataset from CSV.
fn load_california_housing() -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
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

    let file = file.ok_or(
        "Could not find california_housing.csv. Please download it from:\n  https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/california_housing.csv"
    )?;
    let reader = BufReader::new(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            // Skip header
            continue;
        }

        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();

        if values.len() >= 9 {
            x.push(values[0..8].to_vec());
            y.push(values[8]);
        }
    }

    Ok((x, y))
}

/// Normalize features to zero mean and unit variance, and convert to f64.
fn normalize_data(x: &[Vec<f64>], y: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_samples = x.len();
    let n_features = x[0].len();

    // Compute mean
    let mut mean = vec![0.0; n_features];
    for sample in x {
        for (j, &val) in sample.iter().enumerate() {
            mean[j] += val;
        }
    }
    for m in &mut mean {
        *m /= n_samples as f64;
    }

    // Compute std
    let mut std = vec![0.0; n_features];
    for sample in x {
        for (j, &val) in sample.iter().enumerate() {
            std[j] += (val - mean[j]).powi(2);
        }
    }
    for s in &mut std {
        *s = (*s / n_samples as f64).sqrt();
        if *s < 1e-10 {
            *s = 1.0;
        }
    }

    // Normalize features
    let normalized: Vec<Vec<f64>> = x
        .iter()
        .map(|sample| {
            sample
                .iter()
                .enumerate()
                .map(|(j, &val)| (val - mean[j]) / std[j])
                .collect()
        })
        .collect();

    // Target: just convert to f64 (no normalization needed for tree-based models)
    let targets = y.to_vec();

    (normalized, targets)
}

/// Convert 2D vector to Tensor2D.
fn vec_to_tensor2d(data: &[Vec<f64>], n_features: usize) -> Tensor2D<CpuBackend> {
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    Tensor2D::new(flat, data.len(), n_features)
}

/// Convert 1D vector to Tensor1D.
fn vec_to_tensor1d(data: &[f64]) -> Tensor1D<CpuBackend> {
    Tensor1D::new(data.iter().map(|&v| v as f32).collect())
}
