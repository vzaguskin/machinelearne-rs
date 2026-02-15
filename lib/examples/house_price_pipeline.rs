//! Comprehensive ML Pipeline Example: House Price Prediction
//!
//! This example demonstrates a complete end-to-end ML workflow including:
//! - Mixed-type dataset (numerical + categorical features)
//! - Different preprocessing per column type using ColumnTransformer
//! - Missing value imputation
//! - Feature engineering with polynomial features
//! - Model training
//! - Pipeline serialization and loading
//! - Inference on new data
//!
//! # Features:
//! - Column 0: sqft (numerical, needs scaling)
//! - Column 1: bedrooms (numerical, may have missing values)
//! - Column 2: bathrooms (numerical, needs scaling)
//! - Column 3: neighborhood (categorical, one-hot encoded)
//! - Column 4: condition (ordinal, 0-3 scale, passed through)
//!
//! Run with: cargo run --example house_price_pipeline

use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::linear::{LinearModel, LinearParams, SerializableLinearParams},
    model::state::Fitted,
    optimizer::SGD,
    preprocessing::{
        ColumnSpec, ColumnTransformer, FittedTransformer, ImputeStrategy, OneHotEncoder, Pipeline,
        PolynomialFeatures, PredictivePipeline, SimpleImputer, StandardScaler, Transformer,
    },
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor2D,
};
use std::error::Error;

// Feature indices
const SQFT: usize = 0;
const BEDROOMS: usize = 1;
const BATHROOMS: usize = 2;
const NEIGHBORHOOD: usize = 3;
const CONDITION: usize = 4;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== House Price Prediction Pipeline ===\n");

    // 1. Create synthetic training data
    // Features: [sqft, bedrooms, bathrooms, neighborhood, condition]
    // neighborhood: 0=downtown, 1=suburban, 2=rural
    // condition: 0=poor, 1=fair, 2=good, 3=excellent
    let x_train_raw: Vec<Vec<f32>> = vec![
        vec![1500.0, 3.0, 2.0, 0.0, 2.0], // downtown, good
        vec![2000.0, 4.0, 3.0, 1.0, 3.0], // suburban, excellent
        vec![1200.0, 2.0, 1.0, 2.0, 1.0], // rural, fair
        vec![1800.0, 3.0, 2.0, 0.0, 3.0], // downtown, excellent
        vec![2200.0, 4.0, 3.0, 1.0, 2.0], // suburban, good
        vec![1100.0, 2.0, 1.0, 2.0, 0.0], // rural, poor
        vec![2500.0, 5.0, 4.0, 0.0, 3.0], // downtown, excellent
        vec![1400.0, 3.0, 2.0, 1.0, 1.0], // suburban, fair
        // Some with missing values (NaN for bedrooms)
        vec![1600.0, f32::NAN, 2.0, 0.0, 2.0], // missing bedrooms
        vec![1900.0, f32::NAN, 3.0, 1.0, 3.0], // missing bedrooms
    ];

    // Target: house prices (in thousands)
    let y_train: Vec<f32> = vec![
        350.0, 450.0, 180.0, 420.0, 480.0, 150.0, 550.0, 280.0, 360.0, 410.0,
    ];

    println!(
        "Training data: {} samples with {} features each",
        x_train_raw.len(),
        x_train_raw[0].len()
    );

    // 2. Convert to tensor
    let flat_x: Vec<f32> = x_train_raw.iter().flatten().copied().collect();
    let x_train = Tensor2D::<CpuBackend>::new(flat_x, x_train_raw.len(), 5);

    // 3. Build ColumnTransformer for heterogeneous preprocessing
    println!("\nBuilding ColumnTransformer...");

    // Numerical columns (sqft, bedrooms, bathrooms): impute + scale
    let numerical_pipeline = Pipeline::<CpuBackend>::new()
        .add_simple_imputer(SimpleImputer::new(ImputeStrategy::Mean))
        .add_standard_scaler(StandardScaler::new());

    // Categorical column (neighborhood): one-hot encoding
    let categorical_encoder = OneHotEncoder::<CpuBackend>::new();

    // Ordinal column (condition): scale (for ordinal features, scaling is acceptable)
    let ordinal_scaler = StandardScaler::<CpuBackend>::new();

    // Build column transformer
    let column_transformer = ColumnTransformer::<CpuBackend>::new()
        // Numerical: sqft, bedrooms, bathrooms (columns 0, 1, 2)
        .add_pipeline(
            numerical_pipeline,
            ColumnSpec::Indices(vec![SQFT, BEDROOMS, BATHROOMS]),
        )
        // Categorical: neighborhood (column 3) - one-hot encoding
        .add_one_hot_encoder(categorical_encoder, ColumnSpec::Indices(vec![NEIGHBORHOOD]))
        // Ordinal: condition (column 4) - scale
        .add_standard_scaler(ordinal_scaler, ColumnSpec::Indices(vec![CONDITION]));

    // 4. Fit the column transformer
    println!("Fitting column transformer...");
    let fitted_ct = column_transformer.fit(&x_train)?;
    println!("  Input features: {}", fitted_ct.n_features_in());
    println!("  Output features: {}", fitted_ct.n_features_out());

    // 5. Transform training data
    let x_preprocessed = fitted_ct.transform(&x_train)?;

    // 6. Add polynomial features (degree 2, interaction only to avoid too many features)
    println!("\nGenerating polynomial features...");
    let poly = PolynomialFeatures::<CpuBackend>::new()
        .with_degree(2)
        .with_include_bias(false)
        .with_interaction_only(true);

    let fitted_poly = poly.fit(&x_preprocessed)?;
    println!(
        "  Polynomial features: {} -> {}",
        fitted_poly.n_features_in(),
        fitted_poly.n_features_out()
    );

    let x_final = fitted_poly.transform(&x_preprocessed)?;

    // 7. Train the model
    println!("\nTraining linear regression model...");
    let n_features = fitted_poly.n_features_out();

    // Create dataset from preprocessed data
    let x_final_vec: Vec<Vec<f32>> = {
        let (rows, cols) = x_final.shape();
        let flat = x_final.ravel().to_vec();
        (0..rows)
            .map(|r| (0..cols).map(|c| flat[r * cols + c] as f32).collect())
            .collect()
    };
    let dataset = InMemoryDataset::new(x_final_vec, y_train.clone())?;

    // Build and train
    let model = machinelearne_rs::model::linear::LinearRegression::<CpuBackend>::new(n_features);
    let loss = MSELoss;
    let optimizer = SGD::new(0.001);
    let regularizer = NoRegularizer;

    let trainer = Trainer::builder(loss, optimizer, regularizer)
        .batch_size(5)
        .max_epochs(2000)
        .build();

    let fitted_model = trainer.fit(model, &dataset)?;
    println!("  Model trained successfully!");

    // 8. Create and save the complete pipeline
    println!("\nSaving complete pipeline...");
    let pipeline = PredictivePipeline::new(fitted_ct, Some(fitted_poly), fitted_model);

    let temp_file = std::env::temp_dir().join("house_price_pipeline.bin");
    pipeline.save_to_file(&temp_file)?;
    println!("  Pipeline saved to: {:?}", temp_file);

    // 9. Load the pipeline (simulating deployment)
    println!("\nLoading pipeline for inference...");
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
    println!("  Pipeline loaded successfully!");
    println!(
        "  Loaded pipeline expects {} input features",
        loaded_pipeline.n_features_in()
    );

    // 10. Make predictions on new data
    println!("\n=== Predictions on New Houses ===\n");

    // New house 1: 1700 sqft, 3 bed, 2 bath, suburban, excellent condition
    let house1 = Tensor2D::<CpuBackend>::new(vec![1700.0, 3.0, 2.0, 1.0, 3.0], 1, 5);
    let price1 = loaded_pipeline.predict(&house1)?;
    println!(
        "House 1 (suburban, 3bed/2bath, 1700sqft, excellent): ${:.0}k",
        price1.to_vec()[0]
    );

    // New house 2: 1300 sqft, 2 bed, 1 bath, downtown, good condition
    let house2 = Tensor2D::<CpuBackend>::new(vec![1300.0, 2.0, 1.0, 0.0, 2.0], 1, 5);
    let price2 = loaded_pipeline.predict(&house2)?;
    println!(
        "House 2 (downtown, 2bed/1bath, 1300sqft, good): ${:.0}k",
        price2.to_vec()[0]
    );

    // New house 3: 2400 sqft, with missing bedrooms, rural, fair condition
    let house3 = Tensor2D::<CpuBackend>::new(vec![2400.0, f32::NAN, 3.0, 2.0, 1.0], 1, 5);
    let price3 = loaded_pipeline.predict(&house3)?;
    println!(
        "House 3 (rural, ?bed/3bath, 2400sqft, fair, missing data): ${:.0}k",
        price3.to_vec()[0]
    );

    // Batch prediction: multiple houses at once
    let houses_batch = Tensor2D::<CpuBackend>::new(
        vec![
            1500.0, 3.0, 2.0, 0.0, 2.0, // house A
            2000.0, 4.0, 3.0, 1.0, 3.0, // house B
            1200.0, 2.0, 1.0, 2.0, 1.0, // house C
        ],
        3,
        5,
    );
    let prices_batch = loaded_pipeline.predict(&houses_batch)?;
    println!("\nBatch predictions:");
    let prices = prices_batch.to_vec();
    for (i, price) in prices.iter().enumerate() {
        println!("  House {}: ${:.0}k", i + 1, price);
    }

    // Cleanup
    std::fs::remove_file(&temp_file).ok();

    println!("\n=== Pipeline Complete ===");
    Ok(())
}
