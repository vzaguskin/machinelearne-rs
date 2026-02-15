//! Titanic survival prediction using Pipeline V2 with type-state enforcement.
//!
//! This example demonstrates the experimental Pipeline V2 which uses:
//! - Trait objects for transformer dispatch (no enum matching)
//! - Type-state pattern to enforce preprocessing order at compile time
//!
//! The type-state pattern ensures:
//! - Imputer can only be added at Raw state
//! - Scaler can only be added at Imputed state (after imputer)
//! - Encoder can only be added at Scaled state (after scaler)
//! - Model prediction requires complete preprocessing

use machinelearne_rs::backend::{CpuBackend, Tensor1D, Tensor2D};
use machinelearne_rs::dataset::InMemoryDataset;
use machinelearne_rs::loss::BCEWithLogitsLoss;
use machinelearne_rs::model::linear::LinearRegressor;
use machinelearne_rs::model::InferenceModel;
use machinelearne_rs::optimizer::SGD;
use machinelearne_rs::preprocessing::imputation::{ImputeStrategy, SimpleImputer};
use machinelearne_rs::preprocessing::pipeline_v2::{FittedPipelineV2, PipelineV2, Scaled};
use machinelearne_rs::preprocessing::scaling::StandardScaler;
use machinelearne_rs::preprocessing::traits::{FittedTransformer, Transformer};
use machinelearne_rs::regularizers::NoRegularizer;
use machinelearne_rs::trainer::Trainer;
use std::error::Error;

type Backend = CpuBackend;

/// Create synthetic Titanic-like dataset.
///
/// Features (7 total):
/// - Pclass: Passenger class (1, 2, 3) - normalized
/// - Sex: 0 = male, 1 = female
/// - Age: Age in years (may have missing values as f32::NAN)
/// - SibSp: Number of siblings/spouses aboard
/// - Parch: Number of parents/children aboard
/// - Fare: Ticket fare
/// - Embarked: 0 = C, 1 = Q, 2 = S
fn create_titanic_data() -> (Vec<Vec<f32>>, Vec<f32>) {
    let features = vec![
        // [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
        vec![3.0, 0.0, 22.0, 1.0, 0.0, 7.25, 2.0], // Male, 3rd class
        vec![1.0, 1.0, 38.0, 1.0, 0.0, 71.28, 0.0], // Female, 1st class
        vec![3.0, 1.0, 26.0, 0.0, 0.0, 7.92, 2.0], // Female, 3rd class
        vec![1.0, 0.0, 35.0, 1.0, 0.0, 53.10, 2.0], // Male, 1st class
        vec![3.0, 0.0, 35.0, 0.0, 0.0, 8.05, 2.0], // Male, 3rd class
        vec![3.0, 1.0, f32::NAN, 0.0, 0.0, 8.46, 1.0], // Female, missing age
        vec![1.0, 1.0, 54.0, 0.0, 0.0, 51.86, 2.0], // Female, 1st class
        vec![3.0, 0.0, 2.0, 3.0, 1.0, 21.07, 2.0], // Male, 3rd class, child
        vec![2.0, 1.0, 27.0, 0.0, 2.0, 11.13, 2.0], // Female, 2nd class
        vec![3.0, 1.0, 14.0, 1.0, 0.0, 30.07, 0.0], // Female, 3rd class, child
        vec![1.0, 0.0, 4.0, 1.0, 1.0, 16.70, 2.0], // Male, 1st class, child
        vec![3.0, 0.0, 20.0, 0.0, 0.0, 8.05, 2.0], // Male, 3rd class
        vec![3.0, 0.0, f32::NAN, 0.0, 0.0, 7.22, 2.0], // Male, missing age
        vec![1.0, 1.0, 58.0, 0.0, 0.0, 26.55, 0.0], // Female, 1st class
        vec![2.0, 0.0, 44.0, 0.0, 0.0, 13.00, 2.0], // Male, 2nd class
        vec![3.0, 1.0, 34.0, 0.0, 0.0, 7.88, 2.0], // Female, 3rd class
        vec![2.0, 1.0, 28.0, 0.0, 0.0, 13.00, 2.0], // Female, 2nd class
        vec![3.0, 1.0, 19.0, 1.0, 0.0, 7.88, 0.0], // Female, 3rd class
        vec![1.0, 0.0, 49.0, 1.0, 0.0, 89.10, 2.0], // Male, 1st class
        vec![3.0, 0.0, 25.0, 0.0, 0.0, 7.22, 2.0], // Male, 3rd class
    ];

    // Survival (1 = survived, 0 = died)
    // Higher survival for: females, 1st class, children
    let targets = vec![
        0.0, // Male, 3rd class - died
        1.0, // Female, 1st class - survived
        1.0, // Female, 3rd class - survived
        0.0, // Male, 1st class - died (actually survived in real data)
        0.0, // Male, 3rd class - died
        1.0, // Female - survived
        1.0, // Female, 1st class - survived
        0.0, // Male, 3rd class, child - died (actually survived in real data)
        1.0, // Female, 2nd class - survived
        1.0, // Female, 3rd class, child - survived
        1.0, // Male, 1st class, child - survived
        0.0, // Male, 3rd class - died
        0.0, // Male, missing age - died
        1.0, // Female, 1st class - survived
        0.0, // Male, 2nd class - died
        1.0, // Female, 3rd class - survived
        1.0, // Female, 2nd class - survived
        1.0, // Female, 3rd class - survived
        0.0, // Male, 1st class - died
        0.0, // Male, 3rd class - died
    ];

    (features, targets)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Titanic Pipeline V2 Example ===\n");
    println!("Demonstrating type-state enforcement with trait object dispatch\n");

    // 1. Create dataset
    let (features, targets) = create_titanic_data();
    let n_samples = features.len();
    let n_features = features[0].len();
    println!("Dataset: {} samples, {} features", n_samples, n_features);

    // Convert to tensors
    let flat_features: Vec<f32> = features.iter().flatten().copied().collect();
    let x = Tensor2D::<Backend>::new(flat_features, n_samples, n_features);
    let y = Tensor1D::<Backend>::new(targets);

    // 2. Create train/test split (simple: 80/20)
    let train_size = (n_samples as f32 * 0.8) as usize;
    let x_flat: Vec<f32> = x.ravel().to_vec().iter().map(|&v| v as f32).collect();
    let y_flat: Vec<f32> = y.to_vec().iter().map(|&v| v as f32).collect();

    let x_train = Tensor2D::<Backend>::new(
        x_flat[..train_size * n_features].to_vec(),
        train_size,
        n_features,
    );
    let y_train = Tensor1D::<Backend>::new(y_flat[..train_size].to_vec());
    let x_test = Tensor2D::<Backend>::new(
        x_flat[train_size * n_features..].to_vec(),
        n_samples - train_size,
        n_features,
    );
    let y_test = Tensor1D::<Backend>::new(y_flat[train_size..].to_vec());

    println!(
        "Train: {} samples, Test: {} samples\n",
        train_size,
        n_samples - train_size
    );

    // ============================================================
    // PIPELINE V2: Type-state enforced preprocessing pipeline
    // ============================================================

    println!("--- Building Pipeline V2 with Type-State Enforcement ---\n");

    // Step 1: Fit imputer on training data (handles missing Age values)
    let imputer = SimpleImputer::<Backend>::new(ImputeStrategy::Mean);
    let fitted_imputer = imputer.fit(&x_train)?;
    println!("1. Fitted SimpleImputer (strategy: Mean)");

    // Step 2: Fit scaler on training data
    // Note: We fit on imputed data for better statistics
    let x_imputed = fitted_imputer.transform(&x_train)?;
    let scaler = StandardScaler::<Backend>::new();
    let fitted_scaler = scaler.fit(&x_imputed)?;
    println!("2. Fitted StandardScaler");

    // Step 3: Build pipeline with type-state enforcement
    // The compiler ensures correct order:
    // - add_imputer() can only be called on PipelineV2<Raw>
    // - add_scaler() can only be called on PipelineV2<Imputed>
    // - into_fitted() finalizes the pipeline

    let pipeline: PipelineV2<Backend, _> = PipelineV2::new()
        .add_imputer(fitted_imputer) // Raw -> Imputed (type changes!)
        .add_standard_scaler(fitted_scaler.clone()); // Imputed -> Scaled (type changes!)

    println!("\nPipeline steps: {:?}", pipeline.step_names());
    println!("Pipeline length: {}", pipeline.len());

    // Convert to fitted pipeline
    let fitted_pipeline: FittedPipelineV2<Backend, Scaled> = pipeline.into_fitted();
    println!("Pipeline is now fitted and ready for inference\n");

    // ============================================================
    // TYPE-STATE ENFORCEMENT DEMONSTRATION
    // ============================================================

    println!("--- Type-State Enforcement ---\n");
    println!("The following would NOT compile (uncomment to verify):");
    println!("  // let bad = PipelineV2::new()");
    println!("  //     .add_scaler(fitted_scaler);  // ERROR: Scaler requires Imputed state");
    println!("  //     .add_imputer(fitted_imputer);");
    println!();
    println!("  // let bad2 = PipelineV2::new()");
    println!("  //     .add_imputer(fitted_imputer);");
    println!("  //     .add_imputer(fitted_imputer2);  // ERROR: No add_imputer on Imputed state");
    println!();

    // ============================================================
    // TRANSFORM DATA
    // ============================================================

    println!("--- Transforming Data ---\n");

    let x_train_preprocessed = fitted_pipeline.transform(&x_train)?;
    let x_test_preprocessed = fitted_pipeline.transform(&x_test)?;

    println!("Train shape: {:?}", x_train_preprocessed.shape());
    println!("Test shape: {:?}", x_test_preprocessed.shape());

    // ============================================================
    // TRAIN MODEL
    // ============================================================

    println!("\n--- Training Logistic Regression ---\n");

    // Create in-memory dataset for training
    let train_data: Vec<Vec<f32>> = x_train_preprocessed
        .ravel()
        .to_vec()
        .iter()
        .map(|&v| v as f32)
        .collect::<Vec<_>>()
        .chunks(n_features)
        .map(|chunk| chunk.to_vec())
        .collect();
    let train_targets: Vec<f32> = y_train.to_vec().iter().map(|&v| v as f32).collect();
    let dataset = InMemoryDataset::new(train_data, train_targets)?;

    // Build model and trainer
    let model = LinearRegressor::new(n_features);
    let trainer = Trainer::builder(BCEWithLogitsLoss, SGD::new(0.1), NoRegularizer)
        .batch_size(4)
        .max_epochs(500)
        .build();

    // Train
    let fitted_model = trainer.fit(model, &dataset)?;
    println!("Training complete!");

    // ============================================================
    // EVALUATE
    // ============================================================

    println!("\n--- Evaluation ---\n");

    // Predict on test set
    let logits = fitted_model.predict_batch(&x_test_preprocessed);

    // Convert logits to probabilities and predictions
    let predictions: Vec<u8> = logits
        .to_vec()
        .iter()
        .map(|&l| if l > 0.0 { 1 } else { 0 })
        .collect();

    // Calculate accuracy
    let y_test_vec: Vec<f32> = y_test.to_vec().iter().map(|&v| v as f32).collect();
    let correct = predictions
        .iter()
        .zip(y_test_vec.iter())
        .filter(|(&pred, &actual)| pred as f32 == actual)
        .count();
    let accuracy = correct as f32 / predictions.len() as f32 * 100.0;

    println!(
        "Test Accuracy: {:.1}% ({}/{})",
        accuracy,
        correct,
        predictions.len()
    );
    println!("Predictions: {:?}", predictions);
    println!(
        "Actual:      {:?}",
        y_test_vec.iter().map(|&v| v as u8).collect::<Vec<_>>()
    );

    // ============================================================
    // INVERSE TRANSFORM DEMONSTRATION
    // ============================================================

    println!("\n--- Inverse Transform ---\n");

    // Note: SimpleImputer does not support inverse_transform because
    // missing value information is lost during imputation.
    // We can demonstrate inverse_transform with a scaler-only pipeline:

    let scaler_only_pipeline: FittedPipelineV2<Backend, Scaled> = PipelineV2::new()
        .add_scaler_direct(fitted_scaler.clone())
        .into_fitted();

    // Transform and inverse transform
    let scaled_data = scaler_only_pipeline.transform(&x_test)?;
    let original_scale = scaler_only_pipeline.inverse_transform(&scaled_data)?;

    let orig_flat: Vec<f32> = original_scale
        .ravel()
        .to_vec()
        .iter()
        .map(|&v| v as f32)
        .collect();
    let x_test_flat: Vec<f32> = x_test.ravel().to_vec().iter().map(|&v| v as f32).collect();

    println!("Scaler-only pipeline demonstrates inverse transform:");
    println!(
        "First test sample after inverse transform: {:.2?}",
        &orig_flat[..n_features]
    );
    println!(
        "Original first test sample:               {:.2?}",
        &x_test_flat[..n_features]
    );
    println!("\nNote: Values match because inverse_transform reverses the scaling.");

    // ============================================================
    // PREDICTIVE PIPELINE (optional)
    // ============================================================

    println!("\n--- Predictive Pipeline ---\n");

    // Combine preprocessing with model for end-to-end prediction
    let predictive = fitted_pipeline.with_model(fitted_model);

    // Predict on raw data (preprocessing happens automatically)
    let predictions = predictive.predict(&x_test)?;
    let preds: Vec<f32> = predictions.to_vec().iter().map(|&v| v as f32).collect();
    println!("Predictions from raw data: {:.3?}", preds);

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("1. Type-state pattern enforces correct preprocessing order at compile time");
    println!("2. Trait objects provide flexible transformer dispatch without enums");
    println!("3. Pipeline transformations are automatically applied during prediction");
    println!("4. Inverse transforms work in reverse order through the pipeline");

    Ok(())
}
