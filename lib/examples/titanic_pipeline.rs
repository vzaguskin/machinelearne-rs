//! Titanic Survival Prediction Pipeline
//!
//! This example demonstrates a complete ML workflow on the classic Titanic dataset,
//! showcasing all preprocessing capabilities:
//! - Mixed feature types (numerical, categorical, ordinal)
//! - Missing value imputation
//! - Feature scaling and encoding
//! - Binary classification with accuracy metrics
//! - Pipeline serialization
//!
//! Run with: cargo run --example titanic_pipeline

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::BCEWithLogitsLoss,
    model::linear::{LinearModel, LinearParams, SerializableLinearParams},
    model::state::Fitted,
    model::InferenceModel,
    optimizer::SGD,
    preprocessing::{
        ColumnSpec, ColumnTransformer, FittedTransformer, ImputeStrategy, OneHotEncoder, Pipeline,
        PredictivePipeline, SimpleImputer, StandardScaler, Transformer,
    },
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor2D,
};
use std::error::Error;

// Feature indices for the raw data
// [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
const PCLASS: usize = 0;
const SEX: usize = 1;
const AGE: usize = 2;
const SIBSP: usize = 3;
const PARCH: usize = 4;
const FARE: usize = 5;
const EMBARKED: usize = 6;

/// Titanic passenger data (subset of the classic dataset).
/// Features: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
/// - Pclass: 1=1st, 2=2nd, 3=3rd (ordinal)
/// - Sex: 0=male, 1=female (will be one-hot encoded)
/// - Age: in years (has missing values as f32::NAN)
/// - SibSp: number of siblings/spouses aboard
/// - Parch: number of parents/children aboard
/// - Fare: ticket fare
/// - Embarked: 0=C, 1=Q, 2=S (Cherbourg, Queenstown, Southampton - will be one-hot encoded)
/// Target: Survived (0=no, 1=yes)
fn get_titanic_data() -> (Vec<Vec<f32>>, Vec<f32>) {
    let features: Vec<Vec<f32>> = vec![
        // First class passengers
        vec![1.0, 1.0, 29.0, 0.0, 0.0, 211.3375, 0.0], // Female, survived
        vec![1.0, 0.0, 0.9167, 1.0, 2.0, 151.55, 0.0], // Male infant, survived
        vec![1.0, 1.0, 2.0, 1.0, 2.0, 151.55, 0.0],    // Female child, survived
        vec![1.0, 0.0, 30.0, 1.0, 0.0, 164.8667, 0.0], // Male, survived
        vec![1.0, 1.0, 25.0, 1.0, 0.0, 151.55, 0.0],   // Female, survived
        vec![1.0, 0.0, 48.0, 0.0, 0.0, 26.55, 1.0],    // Male, didn't survive
        vec![1.0, 0.0, 36.0, 1.0, 0.0, 135.6333, 1.0], // Male, survived
        vec![1.0, 1.0, 27.0, 1.0, 0.0, 153.4625, 1.0], // Female, survived
        vec![1.0, 0.0, 22.0, 0.0, 0.0, 135.6333, 1.0], // Male, didn't survive
        vec![1.0, 1.0, 38.0, 0.0, 0.0, 80.0, 2.0],     // Female, survived
        // Second class passengers
        vec![2.0, 1.0, 29.0, 0.0, 2.0, 23.0, 2.0], // Female, survived
        vec![2.0, 0.0, 32.0, 0.0, 0.0, 10.5, 2.0], // Male, didn't survive
        vec![2.0, 1.0, 24.0, 0.0, 0.0, 13.0, 2.0], // Female, survived
        vec![2.0, 0.0, 36.0, 0.0, 0.0, 13.0, 2.0], // Male, didn't survive
        vec![2.0, 1.0, 28.0, 0.0, 0.0, 12.65, 2.0], // Female, survived
        vec![2.0, 0.0, 25.0, 0.0, 0.0, 13.0, 2.0], // Male, didn't survive
        vec![2.0, 1.0, 18.0, 0.0, 1.0, 33.0, 2.0], // Female, survived
        vec![2.0, 0.0, 19.0, 1.0, 0.0, 26.0, 2.0], // Male, didn't survive
        vec![2.0, 1.0, 23.0, 0.0, 0.0, 10.5, 2.0], // Female, survived
        vec![2.0, 0.0, 34.0, 0.0, 0.0, 13.0, 2.0], // Male, didn't survive
        // Third class passengers
        vec![3.0, 0.0, 22.0, 0.0, 0.0, 7.25, 2.0], // Male, didn't survive
        vec![3.0, 1.0, 26.0, 0.0, 0.0, 7.925, 2.0], // Female, survived
        vec![3.0, 0.0, 24.0, 0.0, 0.0, 8.4583, 0.0], // Male, didn't survive
        vec![3.0, 1.0, 21.0, 0.0, 0.0, 7.75, 1.0], // Female, survived
        vec![3.0, 0.0, 22.0, 0.0, 0.0, 7.8958, 2.0], // Male, didn't survive
        vec![3.0, 1.0, 27.0, 0.0, 2.0, 21.075, 2.0], // Female, survived
        vec![3.0, 0.0, 30.0, 0.0, 0.0, 7.8958, 2.0], // Male, didn't survive
        vec![3.0, 1.0, 18.0, 0.0, 0.0, 7.775, 2.0], // Female, survived
        vec![3.0, 0.0, 19.0, 0.0, 0.0, 7.8958, 2.0], // Male, didn't survive
        vec![3.0, 1.0, 15.0, 0.0, 0.0, 8.0292, 1.0], // Female, survived
        // More diverse examples with missing ages
        vec![1.0, 1.0, f32::NAN, 1.0, 0.0, 78.85, 2.0], // Female, missing age, survived
        vec![3.0, 0.0, f32::NAN, 0.0, 0.0, 7.75, 1.0],  // Male, missing age, didn't survive
        vec![2.0, 1.0, f32::NAN, 0.0, 0.0, 10.5, 2.0],  // Female, missing age, survived
        vec![3.0, 0.0, f32::NAN, 1.0, 0.0, 15.5, 2.0],  // Male, missing age, didn't survive
        vec![1.0, 0.0, 45.0, 0.0, 0.0, 28.7125, 2.0],   // Male, didn't survive
        vec![3.0, 1.0, 31.0, 1.0, 0.0, 18.0, 2.0],      // Female, survived
        vec![1.0, 0.0, 54.0, 0.0, 0.0, 51.8625, 2.0],   // Male, didn't survive
        vec![3.0, 1.0, 4.0, 3.0, 1.0, 31.275, 2.0],     // Female child, survived
        vec![2.0, 0.0, 29.0, 0.0, 0.0, 13.0, 2.0],      // Male, didn't survive
        vec![3.0, 0.0, 25.0, 1.0, 0.0, 7.775, 2.0],     // Male, didn't survive
        // Additional samples for better training
        vec![1.0, 1.0, 35.0, 0.0, 0.0, 128.0, 2.0], // Female, survived
        vec![3.0, 0.0, 28.0, 0.0, 0.0, 7.05, 2.0],  // Male, didn't survive
        vec![2.0, 1.0, 30.0, 0.0, 0.0, 26.0, 2.0],  // Female, survived
        vec![3.0, 0.0, 20.0, 0.0, 0.0, 8.05, 2.0],  // Male, didn't survive
        vec![1.0, 1.0, 49.0, 0.0, 0.0, 110.8833, 0.0], // Female, survived
        vec![3.0, 0.0, 21.0, 2.0, 0.0, 11.5, 2.0],  // Male, didn't survive
        vec![2.0, 0.0, 39.0, 0.0, 0.0, 26.0, 2.0],  // Male, didn't survive
        vec![3.0, 1.0, 16.0, 1.0, 1.0, 20.2125, 2.0], // Female, survived
        vec![1.0, 0.0, 80.0, 0.0, 0.0, 30.0, 2.0],  // Male, survived (oldest passenger)
        vec![3.0, 0.0, 33.0, 0.0, 0.0, 7.8958, 2.0], // Male, didn't survive
    ];

    let targets: Vec<f32> = vec![
        // First class
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, // Second class
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // Third class
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // Missing ages
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, // Additional
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    ];

    (features, targets)
}

/// Calculate sigmoid for binary classification probability.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Calculate classification metrics.
fn calculate_metrics(predictions: &[bool], targets: &[bool]) -> (f64, f64, f64, f64) {
    let mut tp = 0usize; // True positives
    let mut tn = 0usize; // True negatives
    let mut fp = 0usize; // False positives
    let mut fn_ = 0usize; // False negatives

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        match (pred, target) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
        }
    }

    let accuracy = (tp + tn) as f64 / predictions.len() as f64;
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    (accuracy, precision, recall, f1)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Titanic Survival Prediction Pipeline ===\n");

    // 1. Load data
    println!("Loading Titanic dataset...");
    let (features, targets) = get_titanic_data();
    let n_samples = features.len();
    println!("  {} passengers loaded", n_samples);
    println!(
        "  {} survivors ({}%)",
        targets.iter().filter(|&&t| t == 1.0).count(),
        100.0 * targets.iter().filter(|&&t| t == 1.0).count() as f64 / n_samples as f64
    );

    // 2. Train/test split (80/20)
    let split_idx = (n_samples as f64 * 0.8) as usize;
    let (train_features, test_features) = features.split_at(split_idx);
    let (train_targets, test_targets) = targets.split_at(split_idx);

    println!(
        "\nTrain/test split: {}/{}",
        train_features.len(),
        test_features.len()
    );

    // 3. Convert to tensors
    let flat_train: Vec<f32> = train_features.iter().flatten().copied().collect();
    let x_train = Tensor2D::<CpuBackend>::new(flat_train, train_features.len(), 7);
    let y_train = train_targets.to_vec();

    let flat_test: Vec<f32> = test_features.iter().flatten().copied().collect();
    let x_test = Tensor2D::<CpuBackend>::new(flat_test, test_features.len(), 7);

    // 4. Build preprocessing pipeline
    println!("\nBuilding preprocessing pipeline...");

    // Numerical features (Age, SibSp, Parch, Fare): impute + scale
    let numerical_pipeline = Pipeline::<CpuBackend>::new()
        .add_simple_imputer(SimpleImputer::new(ImputeStrategy::Mean))
        .add_standard_scaler(StandardScaler::new());

    // Categorical features (Sex, Embarked): one-hot encoding
    // Ordinal feature (Pclass): scale (treat as continuous for simplicity)

    let column_transformer = ColumnTransformer::<CpuBackend>::new()
        // Pclass (ordinal, column 0) - scale
        .add_standard_scaler(StandardScaler::new(), ColumnSpec::Indices(vec![PCLASS]))
        // Sex (categorical, column 1) - one-hot encode
        .add_one_hot_encoder(OneHotEncoder::new(), ColumnSpec::Indices(vec![SEX]))
        // Age, SibSp, Parch, Fare (numerical, columns 2,3,4,5) - impute + scale
        .add_pipeline(
            numerical_pipeline,
            ColumnSpec::Indices(vec![AGE, SIBSP, PARCH, FARE]),
        )
        // Embarked (categorical, column 6) - one-hot encode
        .add_one_hot_encoder(OneHotEncoder::new(), ColumnSpec::Indices(vec![EMBARKED]));

    // 5. Fit preprocessor
    println!("Fitting preprocessor...");
    let fitted_ct = column_transformer.fit(&x_train)?;
    println!("  Input features: {}", fitted_ct.n_features_in());
    println!("  Output features: {}", fitted_ct.n_features_out());

    // 6. Transform data
    let x_train_processed = fitted_ct.transform(&x_train)?;
    let x_test_processed = fitted_ct.transform(&x_test)?;

    // 7. Optionally add polynomial features (comment out for simpler model)
    // Uncomment the following block to use polynomial features:
    /*
    println!("\nAdding polynomial features...");
    let poly = PolynomialFeatures::<CpuBackend>::new()
        .with_degree(2)
        .with_include_bias(false)
        .with_interaction_only(true);
    let fitted_poly = poly.fit(&x_train_processed)?;
    println!("  Features after polynomial: {}", fitted_poly.n_features_out());
    let x_train_final = fitted_poly.transform(&x_train_processed)?;
    let x_test_final = fitted_poly.transform(&x_test_processed)?;
    */
    // For now, use linear features
    let fitted_poly = None;
    let x_train_final = x_train_processed;
    let x_test_final = x_test_processed;

    // 8. Create dataset for training
    let (n_rows, n_features) = x_train_final.shape();
    let x_train_vec: Vec<Vec<f32>> = {
        let flat = x_train_final.ravel().to_vec();
        (0..n_rows)
            .map(|r| {
                (0..n_features)
                    .map(|c| flat[r * n_features + c] as f32)
                    .collect()
            })
            .collect()
    };

    let dataset = InMemoryDataset::new(x_train_vec, y_train.clone())?;

    // 9. Train model (Logistic Regression using BCEWithLogitsLoss)
    println!("\nTraining logistic regression model...");
    let model = machinelearne_rs::model::linear::LinearRegression::<CpuBackend>::new(n_features);
    let loss = BCEWithLogitsLoss;
    let optimizer = SGD::new(0.5);
    let regularizer = NoRegularizer;

    let trainer = Trainer::builder(loss, optimizer, regularizer)
        .batch_size(10)
        .max_epochs(1000)
        .build();

    let fitted_model = trainer.fit(model, &dataset)?;

    // 10. Evaluate on test set
    println!("\n=== Evaluation on Test Set ===\n");

    // Get predictions
    let test_logits = fitted_model.predict_batch(&x_test_final);
    let logits_vec = test_logits.to_vec();

    // Convert to binary predictions (threshold at 0.5 probability)
    let predictions: Vec<bool> = logits_vec
        .iter()
        .map(|&logit| sigmoid(logit as f64) >= 0.5)
        .collect();
    let actual: Vec<bool> = test_targets.iter().map(|&t| t == 1.0).collect();

    // Calculate metrics
    let (accuracy, precision, recall, f1) = calculate_metrics(&predictions, &actual);

    println!("Predictions vs Actual:");
    println!(
        "{:<5} {:<10} {:<10} {:<10}",
        "Idx", "Prob", "Pred", "Actual"
    );
    println!("{}", "-".repeat(40));
    for (i, (logit, &actual_val)) in logits_vec.iter().zip(test_targets.iter()).enumerate() {
        let prob = sigmoid(*logit as f64);
        let pred = if prob >= 0.5 { "Survived" } else { "Died" };
        let act = if actual_val == 1.0 {
            "Survived"
        } else {
            "Died"
        };
        println!("{:<5} {:<10.3} {:<10} {:<10}", i, prob, pred, act);
    }

    println!("\n=== Classification Metrics ===");
    println!("Accuracy:  {:.1}%", accuracy * 100.0);
    println!("Precision: {:.1}%", precision * 100.0);
    println!("Recall:    {:.1}%", recall * 100.0);
    println!("F1 Score:  {:.1}%", f1 * 100.0);

    // 11. Save complete pipeline
    println!("\n=== Saving Pipeline ===");
    let pipeline = PredictivePipeline::new(fitted_ct, fitted_poly, fitted_model);

    let temp_file = std::env::temp_dir().join("titanic_pipeline.bin");
    pipeline.save_to_file(&temp_file)?;
    println!("Pipeline saved to: {:?}", temp_file);

    // 12. Load and verify
    println!("\nLoading pipeline...");
    let loaded_pipeline =
        PredictivePipeline::<CpuBackend, LinearModel<CpuBackend, Fitted>>::load_from_file(
            &temp_file,
            |bytes| {
                let serial_params: SerializableLinearParams =
                    bincode::deserialize(bytes).map_err(|e| {
                        machinelearne_rs::preprocessing::PreprocessingError::SerializationError(
                            e.to_string(),
                        )
                    })?;
                let params = LinearParams::try_from(serial_params).map_err(|e| {
                    machinelearne_rs::preprocessing::PreprocessingError::SerializationError(
                        e.to_string(),
                    )
                })?;
                Ok(<LinearModel<CpuBackend, Fitted>>::new(params))
            },
        )?;
    println!("Pipeline loaded successfully!");
    println!(
        "Expected input features: {}",
        loaded_pipeline.n_features_in()
    );

    // 13. Demo prediction for a new passenger
    println!("\n=== Demo: Predict for New Passenger ===");
    // New passenger: 1st class, female, 25 years old, no family, $100 fare, embarked at Southampton
    let new_passenger =
        Tensor2D::<CpuBackend>::new(vec![1.0, 1.0, 25.0, 0.0, 0.0, 100.0, 2.0], 1, 7);
    let logit = loaded_pipeline.predict(&new_passenger)?;
    let prob = sigmoid(logit.to_vec()[0] as f64);
    println!("New passenger: 1st class, female, 25yo, no family, $100 fare, Southampton");
    println!("Survival probability: {:.1}%", prob * 100.0);

    // Another passenger: 3rd class, male, 30 years old, no family, $10 fare, Southampton
    let another_passenger =
        Tensor2D::<CpuBackend>::new(vec![3.0, 0.0, 30.0, 0.0, 0.0, 10.0, 2.0], 1, 7);
    let logit2 = loaded_pipeline.predict(&another_passenger)?;
    let prob2 = sigmoid(logit2.to_vec()[0] as f64);
    println!("\nAnother passenger: 3rd class, male, 30yo, no family, $10 fare, Southampton");
    println!("Survival probability: {:.1}%", prob2 * 100.0);

    // Cleanup
    std::fs::remove_file(&temp_file).ok();

    println!("\n=== Pipeline Complete ===");
    Ok(())
}
