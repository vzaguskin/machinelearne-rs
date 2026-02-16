//! Real-World Pipeline Selection Example
//!
//! Compares three approaches with holdout test set:
//! 1. Naive baseline: LinearRegression on raw data
//! 2. Default baseline: StandardScaler + LinearRegression
//! 3. Optimized: PipelineGridSearchCV over scalers + L2 + polynomial features
//!
//! After finding best params via CV, retrains on full train, saves, loads,
//! and evaluates on the holdout test set.

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    metrics::RegressionMetric,
    model::linear::{InferenceModel, LinearRegression},
    model_selection::{
        KFold, LinearRegressionGrid, PipelineGrid, PipelineGridSearchCV, PolynomialGrid,
        PreprocessingGrid, ScalerGrid, ScalerType, TrainerGrid,
    },
    optimizer::SGD,
    pipeline::FittedPipeline,
    preprocessing::scaling::StandardScaler,
    preprocessing::traits::{FittedTransformer, Transformer},
    regularizers::{NoRegularizer, L2},
    trainer::Trainer,
    Tensor1D, Tensor2D,
};
use std::fs;
use std::path::Path;
use std::time::Instant;

mod dataset_loader;

/// Compute regression metrics: R², MSE, MAE, RMSE
fn compute_metrics(
    y_true: &Tensor1D<CpuBackend>,
    y_pred: &Tensor1D<CpuBackend>,
) -> (f64, f64, f64, f64) {
    let y_true_vec = y_true.to_vec();
    let y_pred_vec = y_pred.to_vec();
    let n = y_true_vec.len() as f64;

    // MSE
    let mse: f64 = y_true_vec
        .iter()
        .zip(y_pred_vec.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n;

    // MAE
    let mae: f64 = y_true_vec
        .iter()
        .zip(y_pred_vec.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / n;

    // R²
    let y_mean: f64 = y_true_vec.iter().sum::<f64>() / n;
    let ss_tot: f64 = y_true_vec.iter().map(|x| (x - y_mean).powi(2)).sum();
    let ss_res: f64 = y_true_vec
        .iter()
        .zip(y_pred_vec.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    // RMSE
    let rmse = mse.sqrt();

    (r2, mse, mae, rmse)
}

/// Split data into train and test sets.
fn split_data(
    features: &[Vec<f32>],
    targets: &[f32],
    test_ratio: f32,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let n_samples = features.len();
    let split_idx = (n_samples as f32 * (1.0 - test_ratio)) as usize;

    let x_train = features[..split_idx].to_vec();
    let y_train = targets[..split_idx].to_vec();
    let x_test = features[split_idx..].to_vec();
    let y_test = targets[split_idx..].to_vec();

    (x_train, y_train, x_test, y_test)
}

/// Convert features and targets to tensors.
fn to_tensors(
    features: &[Vec<f32>],
    targets: &[f32],
) -> (Tensor2D<CpuBackend>, Tensor1D<CpuBackend>) {
    let n_samples = features.len();
    let n_features = features[0].len();
    let x_flat: Vec<f32> = features.iter().flatten().copied().collect();
    (
        Tensor2D::new(x_flat, n_samples, n_features),
        Tensor1D::new(targets.to_vec()),
    )
}

/// Convert a 2D tensor to Vec<Vec<f32>>.
fn tensor2d_to_vec(tensor: &Tensor2D<CpuBackend>) -> Vec<Vec<f32>> {
    let (rows, cols) = tensor.shape();
    let flat = tensor.ravel().to_vec();
    (0..rows)
        .map(|i| (0..cols).map(|j| flat[i * cols + j] as f32).collect())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Real-World Pipeline Selection ===\n");

    // Load California Housing dataset
    let (features, targets) = dataset_loader::load_california_housing()?;
    let n_samples = features.len();
    let n_features = features[0].len();

    println!("Dataset: California Housing");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);

    // HOLDOUT TEST SET: 80/20 split (no shuffle for reproducibility)
    let (x_train_feat, y_train, x_test_feat, y_test) = split_data(&features, &targets, 0.2);

    println!(
        "  Train: {}, Test: {}",
        x_train_feat.len(),
        x_test_feat.len()
    );

    // Convert to tensors for GridSearchCV
    let (x_train_tensor, y_train_tensor) = to_tensors(&x_train_feat, &y_train);
    let (x_test_tensor, y_test_tensor) = to_tensors(&x_test_feat, &y_test);

    // Results storage
    let mut results = serde_json::json!({
        "train_size": x_train_feat.len(),
        "test_size": x_test_feat.len(),
    });

    // ========================================
    // 1. NAIVE BASELINE (with StandardScaler, matching sklearn approach)
    // ========================================
    println!("\n--- 1. Naive Baseline (StandardScaler + LinearRegression) ---");
    let start = Instant::now();

    // Fit scaler on train only (proper ML practice)
    let scaler = StandardScaler::new();
    let fitted_scaler = scaler.fit(&x_train_tensor)?;
    let x_train_scaled = fitted_scaler.transform(&x_train_tensor)?;
    let x_test_scaled = fitted_scaler.transform(&x_test_tensor)?;

    // Convert scaled data back to dataset format
    let x_train_scaled_vec = tensor2d_to_vec(&x_train_scaled);

    let dataset = InMemoryDataset::new(x_train_scaled_vec, y_train.clone())?;

    // Use LR=0.5, 100 epochs based on fair comparison benchmarks
    let trainer = Trainer::builder(MSELoss, SGD::new(0.5), NoRegularizer)
        .batch_size(x_train_feat.len()) // Full batch
        .max_epochs(100)
        .verbose(false)
        .build();

    let model = LinearRegression::<CpuBackend>::new(n_features);
    let fitted_model = trainer.fit(model, &dataset)?;

    let y_pred = fitted_model.predict_batch(&x_test_scaled);
    let naive_time = start.elapsed().as_secs_f64() * 1000.0;
    let (r2, mse, mae, rmse) = compute_metrics(&y_test_tensor, &y_pred);

    println!("  Test R2: {:.4}, RMSE: {:.4}, MAE: {:.4}", r2, rmse, mae);
    results["naive_baseline"] = serde_json::json!({
        "test_r2": r2, "test_mse": mse, "test_mae": mae, "test_rmse": rmse,
        "train_time_ms": naive_time,
        "config": {"scaler": "StandardScaler", "lr": 0.1, "epochs": 100}
    });

    // ========================================
    // 2. DEFAULT BASELINE (StandardScaler + L2 regularization)
    // ========================================
    println!("\n--- 2. Default Baseline (StandardScaler + L2=0.01) ---");
    let start = Instant::now();

    let dataset = InMemoryDataset::new(tensor2d_to_vec(&x_train_scaled), y_train.clone())?;

    // Add L2 regularization
    let trainer = Trainer::builder(MSELoss, SGD::new(0.5), L2::new(0.01))
        .batch_size(x_train_feat.len())
        .max_epochs(100)
        .verbose(false)
        .build();

    let model = LinearRegression::<CpuBackend>::new(n_features);
    let fitted_model = trainer.fit(model, &dataset)?;

    let y_pred = fitted_model.predict_batch(&x_test_scaled);
    let default_time = start.elapsed().as_secs_f64() * 1000.0;
    let (r2, mse, mae, rmse) = compute_metrics(&y_test_tensor, &y_pred);

    println!("  Test R2: {:.4}, RMSE: {:.4}, MAE: {:.4}", r2, rmse, mae);
    results["default_baseline"] = serde_json::json!({
        "test_r2": r2, "test_mse": mse, "test_mae": mae, "test_rmse": rmse,
        "train_time_ms": default_time,
        "config": {"scaler": "StandardScaler", "lr": 0.1, "epochs": 100, "l2": 0.01}
    });

    // ========================================
    // 3. OPTIMIZED PIPELINE (GridSearchCV)
    // ========================================
    println!("\n--- 3. Optimized Pipeline (GridSearchCV) ---");
    println!(
        "   Search space: 2 scalers x 2 poly degrees x 3 LR x 4 L2 x 2 epochs = 96 combinations"
    );
    println!("   Note: SGD needs more epochs for polynomial features (44 features vs 8)");

    // Define search space - match sklearn's winning configuration
    let preprocessing = PreprocessingGrid::new()
        .with_scaler(ScalerGrid::new(vec![
            ScalerType::MinMax,   // sklearn found this best
            ScalerType::Standard, // Also good, common default
        ]))
        .with_polynomial(PolynomialGrid::new(vec![1, 2])); // 1=no poly, 2=quadratic (sklearn found degree 2 best)

    let model_grid = LinearRegressionGrid::new()
        .with_learning_rates(vec![0.1, 0.5, 1.0]) // Higher LR for faster convergence with poly features
        .with_lambdas(vec![0.0, 0.01, 0.1, 1.0]) // Match sklearn's alpha range
        .with_trainer(TrainerGrid::new(
            vec![x_train_feat.len()], // Full batch
            vec![500, 1000],          // More epochs for polynomial feature convergence
        ));

    let pipeline_grid = PipelineGrid::new(preprocessing, model_grid);

    // Run grid search with 5-fold CV on TRAIN only
    // Use SHUFFLE for better CV estimates (California Housing is spatially ordered)
    let start = Instant::now();
    let cv = KFold::new(5).with_random_state(42); // Shuffle with fixed seed for reproducibility
    let search = PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
        .with_cv(cv)
        .verbose(1);

    let result = search.fit(&x_train_tensor, &y_train_tensor)?;
    let cv_time = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "\n   Best CV R2: {:.4} (+/- {:.4})",
        result.best_params.mean_score, result.best_params.std_score
    );
    println!("   Best params:");
    println!(
        "     - Scaler: {:?}",
        result.best_params.preprocessing.scaler
    );
    println!("     - L2 lambda: {}", result.best_params.model.lambda);
    println!(
        "     - Poly degree: {}",
        result.best_params.preprocessing.poly_degree
    );

    // The best pipeline is already fitted on full train data
    let best_pipeline = &result.best_pipeline;

    // SAVE pipeline
    fs::create_dir_all("saved_models")?;
    let model_path = Path::new("saved_models/rust_best_pipeline.bin");
    best_pipeline.save_to_file(model_path)?;
    println!("\n   Saved pipeline to {}", model_path.display());

    // LOAD pipeline
    let loaded_pipeline = FittedPipeline::<CpuBackend>::load_from_file(model_path)?;

    // PREDICT on test set using loaded pipeline
    let y_pred = loaded_pipeline.predict(&x_test_tensor)?;
    let (r2, mse, mae, rmse) = compute_metrics(&y_test_tensor, &y_pred);

    println!(
        "\n   Test R2: {:.4}, RMSE: {:.4}, MAE: {:.4}",
        r2, rmse, mae
    );

    // Count total combinations
    let n_combinations = result.all_results.len();

    results["optimized"] = serde_json::json!({
        "test_r2": r2, "test_mse": mse, "test_mae": mae, "test_rmse": rmse,
        "cv_time_ms": cv_time,
        "best_cv_score": result.best_params.mean_score,
        "best_params": {
            "scaler": format!("{:?}", result.best_params.preprocessing.scaler),
            "l2_lambda": result.best_params.model.lambda,
            "poly_degree": result.best_params.preprocessing.poly_degree,
        },
        "n_combinations": n_combinations,
    });

    // ========================================
    // SUMMARY
    // ========================================
    println!("\n{}", "=".repeat(60));
    println!("RUST RESULTS (evaluated on holdout test set)");
    println!("{}", "=".repeat(60));
    println!("{:<25} {:>8} {:>10} {:>8}", "Method", "R2", "RMSE", "MAE");
    println!("{}", "-".repeat(60));

    for name in ["naive_baseline", "default_baseline", "optimized"] {
        let r = &results[name];
        let r2 = r["test_r2"].as_f64().unwrap_or(f64::NAN);
        let rmse = r["test_rmse"].as_f64().unwrap_or(f64::NAN);
        let mae = r["test_mae"].as_f64().unwrap_or(f64::NAN);
        if r2.is_nan() {
            println!("{:<25} {:>8} {:>10} {:>8}", name, "NaN", "NaN", "NaN");
        } else {
            println!("{:<25} {:>8.4} {:>10.4} {:>8.4}", name, r2, rmse, mae);
        }
    }

    // Improvement
    let naive_r2 = results["naive_baseline"]["test_r2"]
        .as_f64()
        .unwrap_or(f64::NAN);
    let opt_r2 = results["optimized"]["test_r2"].as_f64().unwrap_or(f64::NAN);
    if !naive_r2.is_nan() && !opt_r2.is_nan() && naive_r2.abs() > 0.001 {
        let improvement = (opt_r2 - naive_r2) / naive_r2.abs() * 100.0;
        if improvement >= 0.0 {
            println!("\nImprovement over naive: +{:.1}%", improvement);
        } else {
            println!("\nChange from naive: {:.1}%", improvement);
        }
    }

    // Save results
    fs::create_dir_all("benchmarks/results")?;
    fs::write(
        "benchmarks/results/pipeline_comparison_rust.json",
        serde_json::to_string_pretty(&results)?,
    )?;

    println!("\nResults saved to benchmarks/results/pipeline_comparison_rust.json");

    Ok(())
}
