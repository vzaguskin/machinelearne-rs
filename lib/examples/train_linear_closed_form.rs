//! Example demonstrating closed-form (normal equation) solution for linear regression.
//!
//! This compares the closed-form solution with iterative SGD training:
//! - Closed-form: One-step solution, no hyperparameters, exact (within numerical precision)
//! - SGD: Iterative, requires learning rate and epochs tuning, approximate
//!
//! Run with: cargo run --example train_linear_closed_form

use machinelearne_rs::backend::{CpuBackend, Tensor1D, Tensor2D};
use machinelearne_rs::dataset::memory::InMemoryDataset;
use machinelearne_rs::loss::MSELoss;
use machinelearne_rs::model::linear::{InferenceModel, LinearRegression};
use machinelearne_rs::optimizer::SGD;
use machinelearne_rs::regularizers::NoRegularizer;
use machinelearne_rs::trainer::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Linear Regression: Closed-Form vs SGD ===\n");

    // Generate synthetic data: y = 2x + 1 + noise
    // True parameters: weight = 2.0, bias = 1.0
    let n_samples = 100;
    let n_features = 1;

    let mut x_data = Vec::with_capacity(n_samples);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x = (i as f32) / 10.0; // x = 0, 0.1, 0.2, ..., 9.9
        let noise = ((i % 7) as f32 - 3.0) * 0.1; // Small noise
        let y = 2.0 * x + 1.0 + noise;
        x_data.push(vec![x]);
        y_data.push(y);
    }

    // Create tensors
    let x_flat: Vec<f32> = x_data.iter().flat_map(|v| v.iter().copied()).collect();
    let x_tensor = Tensor2D::<CpuBackend>::new(x_flat.clone(), n_samples, n_features);
    let y_tensor = Tensor1D::<CpuBackend>::new(y_data.clone());

    // === Method 1: Closed-Form Solution ===
    println!("Method 1: Closed-Form Solution (Normal Equation)");
    println!("---------------------------------------------------");

    let start = std::time::Instant::now();
    let model_cf = LinearRegression::<CpuBackend>::new(n_features);
    let fitted_cf = model_cf.fit_closed_form(&x_tensor, &y_tensor)?;
    let cf_duration = start.elapsed();

    let params_cf = fitted_cf.extract_params();
    println!("  Time: {:?}", cf_duration);
    println!("  Weight: {:.6} (true: 2.0)", params_cf.weights[0]);
    println!("  Bias:   {:.6} (true: 1.0)", params_cf.bias);

    // === Method 2: SGD Training ===
    println!("\nMethod 2: SGD Training");
    println!("----------------------");

    let dataset = InMemoryDataset::new(x_data.clone(), y_data.clone())?;

    let start = std::time::Instant::now();
    let model_sgd = LinearRegression::<CpuBackend>::new(n_features);
    let trainer = Trainer::builder(MSELoss, SGD::<CpuBackend>::new(0.01), NoRegularizer)
        .batch_size(32)
        .max_epochs(500)
        .verbose(false)
        .build();

    let fitted_sgd = trainer.fit(model_sgd, &dataset)?;
    let sgd_duration = start.elapsed();

    let params_sgd = fitted_sgd.extract_params();
    println!("  Time: {:?}", sgd_duration);
    println!("  Weight: {:.6} (true: 2.0)", params_sgd.weights[0]);
    println!("  Bias:   {:.6} (true: 1.0)", params_sgd.bias);

    // === Comparison ===
    println!("\n=== Comparison ===");
    println!(
        "Closed-form is {:.1}x {} than SGD",
        if cf_duration < sgd_duration {
            sgd_duration.as_micros() as f64 / cf_duration.as_micros().max(1) as f64
        } else {
            cf_duration.as_micros() as f64 / sgd_duration.as_micros().max(1) as f64
        },
        if cf_duration < sgd_duration {
            "faster"
        } else {
            "slower"
        }
    );

    // Compare predictions
    let test_x = Tensor1D::<CpuBackend>::new(vec![5.0]);
    let pred_cf = fitted_cf.predict(&test_x);
    let pred_sgd = fitted_sgd.predict(&test_x);
    let true_value = 2.0 * 5.0 + 1.0; // 11.0

    println!("\nPrediction for x=5.0:");
    println!("  Closed-form: {:.6}", pred_cf.to_f64());
    println!("  SGD:         {:.6}", pred_sgd.to_f64());
    println!("  True value:  {:.6}", true_value);

    println!(
        "\nClosed-form error: {:.6}",
        (pred_cf.to_f64() - true_value).abs()
    );
    println!(
        "SGD error:         {:.6}",
        (pred_sgd.to_f64() - true_value).abs()
    );

    println!("\n=== Conclusion ===");
    println!("Closed-form solution:");
    println!("  + No hyperparameter tuning needed");
    println!("  + Exact solution (within numerical precision)");
    println!("  + Faster for small-to-medium datasets");
    println!("  - Requires matrix inversion (O(n³))");
    println!("  - Not suitable for very large datasets");

    println!("\nSGD:");
    println!("  + Scales to large datasets");
    println!("  + Works with any loss function");
    println!("  + Supports online learning");
    println!("  - Requires hyperparameter tuning");
    println!("  - Approximate solution");

    Ok(())
}
