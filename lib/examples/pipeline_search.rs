//! Full ML Pipeline Example: End-to-End Training, CV, and Deployment
//!
//! This example demonstrates:
//! 1. Loading raw data
//! 2. Defining preprocessing + model search space
//! 3. Running PipelineGridSearchCV with proper CV (no data leakage)
//! 4. Saving the best pipeline for deployment
//! 5. Loading and using for inference

use machinelearne_rs::{
    backend::{CpuBackend, Tensor1D, Tensor2D},
    metrics::RegressionMetric,
    model_selection::{
        KFold, LinearRegressionGrid, PipelineGrid, PipelineGridSearchCV, PreprocessingGrid,
        ScalerGrid, ScalerType, TrainerGrid,
    },
    pipeline::FittedPipeline,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Full ML Pipeline Example ===\n");

    // 1. Create synthetic data: y = 2*x1 + 3*x2 + 1 + noise
    let n_samples = 100;
    let n_features = 2;

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    // Use a simple pattern for reproducibility
    for i in 0..n_samples {
        let x1 = (i % 10) as f32 / 10.0;
        let x2 = ((i / 10) % 10) as f32 / 10.0;
        x_data.push(x1);
        x_data.push(x2);
        // y = 2*x1 + 3*x2 + 1 + small noise
        let noise = ((i as f32 * 0.1) % 0.2) - 0.1;
        y_data.push(2.0 * x1 + 3.0 * x2 + 1.0 + noise);
    }

    let raw_x = Tensor2D::<CpuBackend>::new(x_data, n_samples, n_features);
    let y = Tensor1D::<CpuBackend>::new(y_data);

    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!("True relationship: y = 2*x1 + 3*x2 + 1 + noise\n");

    // 2. Define search space
    let preprocessing = PreprocessingGrid::new().with_scaler(ScalerGrid::new(vec![
        ScalerType::Standard,
        ScalerType::MinMax,
        ScalerType::None,
    ]));

    let model = LinearRegressionGrid::new()
        .with_learning_rates(vec![0.01, 0.1])
        .with_lambdas(vec![0.0, 0.01])
        .with_trainer(TrainerGrid::new(vec![32], vec![500]));

    let pipeline_grid = PipelineGrid::new(preprocessing, model);

    // Total: 3 * 2 * 2 * 1 = 12 combinations
    println!(
        "Search space: {} combinations",
        pipeline_grid.n_combinations()
    );
    println!(
        "  - Scalers: Standard, MinMax, None\n  - Learning rates: 0.01, 0.1\n  - Lambda: 0.0, 0.01\n"
    );

    // 3. Run grid search with proper CV
    println!("Running PipelineGridSearchCV with 5-fold CV...\n");

    let search = PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
        .with_cv(KFold::new(5))
        .verbose(1);

    let result = search.fit(&raw_x, &y)?;

    // 4. Display results
    println!("\n=== Best Parameters ===");
    println!("  Scaler: {:?}", result.best_params.preprocessing.scaler);
    println!(
        "  Learning rate: {}",
        result.best_params.model.learning_rate
    );
    println!("  Lambda: {}", result.best_params.model.lambda);
    println!(
        "  R² score: {:.4} (+/- {:.4})",
        result.best_params.mean_score, result.best_params.std_score
    );

    // 5. Save the best pipeline for deployment
    let model_path = "best_pipeline.bin";
    result.best_pipeline.save_to_file(model_path)?;
    println!("\nPipeline saved to: {}", model_path);

    // 6. Load and use for inference
    println!("\n=== Inference Demo ===");
    let loaded_pipeline = FittedPipeline::<CpuBackend>::load_from_file(model_path)?;

    // Test predictions
    let test_x = Tensor2D::<CpuBackend>::new(
        vec![
            0.5, 0.5, // y ≈ 2*0.5 + 3*0.5 + 1 = 4.5
            1.0, 0.0, // y ≈ 2*1.0 + 3*0.0 + 1 = 3.0
            0.0, 1.0, // y ≈ 2*0.0 + 3*1.0 + 1 = 4.0
        ],
        3,
        2,
    );

    let predictions = loaded_pipeline.predict(&test_x)?;

    println!("Test predictions:");
    let test_data = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]];
    let expected = [4.5, 3.0, 4.0];
    for (i, (&input, expected)) in test_data.iter().zip(expected.iter()).enumerate() {
        let pred = predictions.to_vec()[i];
        println!(
            "  [{:.1}, {:.1}] -> predicted: {:.3}, expected: {:.1}, error: {:.3}",
            input[0],
            input[1],
            pred,
            expected,
            (pred - expected).abs()
        );
    }

    // 7. Display all results
    println!("\n=== Top 5 Results ===");
    for (i, res) in result.all_results.iter().take(5).enumerate() {
        println!(
            "{}. scaler={:?}, lr={}, lambda={:.3}, R²={:.4}",
            i + 1,
            res.preprocessing.scaler,
            res.model.learning_rate,
            res.model.lambda,
            res.mean_score
        );
    }

    // Clean up
    std::fs::remove_file(model_path).ok();

    println!("\n=== Example Complete ===");
    Ok(())
}
