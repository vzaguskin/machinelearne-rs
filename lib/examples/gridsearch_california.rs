//! GridSearchCV example with real California Housing dataset.
//!
//! This example demonstrates hyperparameter tuning using GridSearchCV
//! on the real California Housing dataset, a standard ML benchmark.
//!
//! # Dataset: California Housing
//!
//! - 20,640 samples, 8 features
//! - Target: Median house value (in $100k)
//! - sklearn LinearRegression baseline: R² ≈ 0.60
//! - sklearn Ridge (alpha=1.0): R² ≈ 0.60
//!
//! # sklearn Equivalent
//!
//! ```python
//! from sklearn.datasets import fetch_california_housing
//! from sklearn.linear_model import Ridge
//! from sklearn.model_selection import cross_val_score
//! from sklearn.preprocessing import StandardScaler
//!
//! X, y = fetch_california_housing(return_X_y=True)
//! scaler = StandardScaler()
//! X_scaled = scaler.fit_transform(X)
//!
//! ridge = Ridge(alpha=0.01)
//! scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
//! print(f"Ridge R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
//! # Expected: R² ≈ 0.60
//! ```

use machinelearne_rs::{
    backend::{CpuBackend, Tensor2D},
    dataset::InMemoryDataset,
    metrics::RegressionMetric,
    model::InferenceModel,
    model_selection::{GridSearchCV, KFold, LinearRegressionGrid, RegularizerGrid, TrainerGrid},
    preprocessing::{FittedTransformer, StandardScaler, Transformer},
};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GridSearchCV with Real California Housing Dataset ===\n");

    // Load the real California Housing dataset
    let (x_raw, y) = load_california_housing()?;

    println!(
        "Dataset loaded: {} samples, {} features",
        x_raw.len(),
        x_raw[0].len()
    );
    println!(
        "Target range: [{:.2}, {:.2}] (${}k - ${}k)",
        y.iter().cloned().fold(f32::INFINITY, f32::min),
        y.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        (y.iter().cloned().fold(f32::INFINITY, f32::min) * 100.0) as i32,
        (y.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 100.0) as i32
    );
    println!();

    // Feature names
    let feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];

    println!("Feature statistics:");
    for (i, name) in feature_names.iter().enumerate() {
        let col: Vec<f32> = x_raw.iter().map(|row| row[i]).collect();
        let mean = col.iter().sum::<f32>() / col.len() as f32;
        let std = {
            let variance: f32 =
                col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / col.len() as f32;
            variance.sqrt()
        };
        println!("  {:12}: mean={:>10.2}, std={:>10.2}", name, mean, std);
    }
    println!();

    // Standardize features (critical for SGD convergence)
    println!("Preprocessing: StandardScaler");
    let scaler = StandardScaler::<CpuBackend>::new();

    let flat_x: Vec<f32> = x_raw.iter().flatten().copied().collect();
    let x_tensor = Tensor2D::<CpuBackend>::new(flat_x, x_raw.len(), 8);

    let fitted_scaler = scaler.fit(&x_tensor)?;
    let x_scaled_tensor = fitted_scaler.transform(&x_tensor)?;

    // Convert back to Vec<Vec<f32>> for InMemoryDataset
    let (n_samples, n_features) = x_scaled_tensor.shape();
    let scaled_flat = x_scaled_tensor.ravel().to_vec();
    let x_scaled: Vec<Vec<f32>> = (0..n_samples)
        .map(|r| {
            (0..n_features)
                .map(|c| scaled_flat[r * n_features + c] as f32)
                .collect()
        })
        .collect();

    let dataset = InMemoryDataset::new(x_scaled.clone(), y.clone())?;

    // Quick timing test with 1 epoch
    println!("=== Quick Timing Test (1 epoch) ===");
    let quick_grid = LinearRegressionGrid::new()
        .with_learning_rates(vec![0.01])
        .with_lambdas(vec![0.0])
        .with_trainer(TrainerGrid::new(vec![128], vec![1]));

    let start = std::time::Instant::now();
    let quick_search = GridSearchCV::<CpuBackend, _>::new(quick_grid, RegressionMetric::R2)
        .with_cv(KFold::new(3))
        .verbose(0);
    let _ = quick_search.fit(&dataset, n_features)?;
    let elapsed = start.elapsed();
    println!("1 epoch with 3-fold CV took: {:?}", elapsed);
    println!();

    // Define parameter grid for full search
    // Using smaller epochs for faster execution
    let param_grid = LinearRegressionGrid::new()
        .with_learning_rates(vec![0.005, 0.01, 0.05])
        .with_regularizer(RegularizerGrid::L2 {
            lambdas: vec![0.0, 0.01, 0.1],
        })
        .with_trainer(TrainerGrid::new(vec![64, 128], vec![50, 100]));

    println!("=== Full GridSearch ===");
    println!("Parameter grid:");
    println!(
        "  Learning rates: {:?}",
        param_grid.optimizer.learning_rates
    );
    if let RegularizerGrid::L2 { lambdas } = &param_grid.regularizer {
        println!("  L2 lambdas (alpha): {:?}", lambdas);
    }
    println!("  Batch sizes: {:?}", param_grid.trainer.batch_sizes);
    println!("  Max epochs: {:?}", param_grid.trainer.max_epochs);
    println!("  Total combinations: {}", param_grid.n_combinations());
    println!();

    // Run GridSearchCV with 5-fold cross-validation
    println!("Running 5-fold cross-validation...\n");

    let start = std::time::Instant::now();
    let grid_search = GridSearchCV::<CpuBackend, _>::new(param_grid, RegressionMetric::R2)
        .with_cv(KFold::new(5).with_random_state(42))
        .verbose(1);

    let result = grid_search.fit(&dataset, n_features)?;
    let elapsed = start.elapsed();

    // Print results
    println!("\n{}", "=".repeat(60));
    println!("RESULTS (completed in {:?})", elapsed);
    println!("{}", "=".repeat(60));

    println!("\nBest parameters found:");
    println!(
        "  learning_rate (eta0): {}",
        result.best_params.learning_rate
    );
    println!("  lambda (alpha): {}", result.best_params.lambda);
    println!("  batch_size: {}", result.best_params.batch_size);
    println!("  max_epochs: {}", result.best_params.max_epochs);

    println!(
        "\nBest CV R² score: {:.4} (+/- {:.4})",
        result.best_params.mean_score, result.best_params.std_score
    );

    // Print top 10 results
    println!("\nTop 10 parameter combinations:");
    println!(
        "{:<10} {:<10} {:<10} {:<10} {:<10}",
        "LR", "Lambda", "Batch", "Epochs", "R²"
    );
    println!("{}", "-".repeat(50));
    for res in result.all_results.iter().take(10) {
        println!(
            "{:<10.4} {:<10.3} {:<10} {:<10} {:<10.4}",
            res.learning_rate, res.lambda, res.batch_size, res.max_epochs, res.mean_score
        );
    }

    // sklearn comparison
    println!("\n{}", "=".repeat(60));
    println!("COMPARISON WITH SKLEARN");
    println!("{}", "=".repeat(60));
    println!("\nExpected sklearn baselines on California Housing:");
    println!("  LinearRegression:    R² ≈ 0.60");
    println!("  Ridge (alpha=0.01):  R² ≈ 0.60");
    println!("  SGDRegressor (tuned): R² ≈ 0.58-0.62");
    println!();
    println!(
        "Our result:           R² = {:.4}",
        result.best_params.mean_score
    );

    if result.best_params.mean_score >= 0.55 {
        println!("✓ Within expected range for linear models!");
    } else if result.best_params.mean_score >= 0.50 {
        println!("~ Close to expected range - more epochs may help");
    } else {
        println!("! Below expected range - need more epochs or different parameters");
    }

    // Sample predictions
    println!("\n{}", "=".repeat(60));
    println!("SAMPLE PREDICTIONS (first 5 samples)");
    println!("{}", "=".repeat(60));

    let test_x: Vec<Vec<f32>> = (0..5).map(|i| x_scaled[i].clone()).collect();
    let test_y: Vec<f32> = (0..5).map(|i| y[i]).collect();

    let test_tensor = Tensor2D::<CpuBackend>::new(
        test_x.iter().flat_map(|r| r.iter().copied()).collect(),
        5,
        8,
    );

    let predictions = result.best_model.predict_batch(&test_tensor);
    let pred_vec = predictions.to_vec();

    println!(
        "\n{:<8} {:<12} {:<12} {:<12}",
        "Sample", "Actual ($k)", "Predicted ($k)", "Error ($k)"
    );
    println!("{}", "-".repeat(48));
    for (i, (&actual, &pred)) in test_y.iter().zip(pred_vec.iter()).enumerate() {
        let error = (pred as f32 - actual).abs();
        println!(
            "{:<8} {:<12.0} {:<12.0} {:<12.0}",
            i + 1,
            actual * 100.0,
            (pred as f32) * 100.0,
            error * 100.0
        );
    }

    Ok(())
}

/// Load the real California Housing dataset from CSV.
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
