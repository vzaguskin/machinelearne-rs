# machinelearne-rs

> A type-safe, zero-overhead machine learning library in Rust - built for learners, by learners.

`machinelearne-rs` is a composable ML library designed around **explicit separation of concerns**:

- **Models** know how to predict and compute gradients
- **Losses** know how to score predictions and emit per-sample gradients
- **Optimizers** update parameters based on gradients
- **Regularizers** penalize model complexity independently
- **Trainers** orchestrate the loop - nothing more
- **Preprocessing** transforms data before training
- **Pipelines** combine everything for end-to-end workflows

No hidden state. No dynamic dispatch. No runtime surprises. Just pure, generic Rust.

---

## Features

### Core ML
- **Models**: Linear Regression (with L1/L2 regularization)
- **Losses**: MSE, MAE, BCE with Logits
- **Optimizers**: SGD
- **Regularizers**: L1, L2
- **Metrics**: R2, MSE, MAE, RMSE

### Preprocessing
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
- **Imputation**: SimpleImputer (mean, median, most_frequent, constant)
- **Encoding**: OneHotEncoder, OrdinalEncoder, LabelEncoder
- **Feature Engineering**: PolynomialFeatures (degree 2+)
- **Pipeline**: Chain multiple transformers
- **ColumnTransformer**: Apply different transformers to different columns

### Model Selection
- **Cross-Validation**: KFold with optional shuffling
- **Train/Test Split**: Stratified or random splitting
- **GridSearchCV**: Hyperparameter tuning with CV
- **PipelineGridSearchCV**: End-to-end pipeline optimization

### Deployment
- **Serialization**: Save/load models and pipelines with bincode
- **FittedPipeline**: Combined preprocessing + model for inference

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
machinelearne-rs = "0.1"
```

### Basic Training

```rust
use machinelearne_rs::{
    backend::CpuBackend,
    dataset::memory::InMemoryDataset,
    loss::MSELoss,
    model::linear::{InferenceModel, LinearRegression},
    optimizer::SGD,
    regularizers::NoRegularizer,
    trainer::Trainer,
    Tensor1D,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create model and training components
    let model = LinearRegression::<CpuBackend>::new(2);
    let trainer = Trainer::builder(MSELoss, SGD::new(0.01), NoRegularizer)
        .batch_size(4)
        .max_epochs(1000)
        .build();

    // Training data: y = x0 + 2*x1
    let x = vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![1.0, 2.0], vec![2.0, 2.0]];
    let y = vec![3.0, 4.0, 5.0, 6.0];
    let dataset = InMemoryDataset::new(x, y)?;

    // Train and convert to inference model
    let fitted = trainer.fit(model, &dataset)?;

    // Predict
    let input = Tensor1D::<CpuBackend>::new(vec![3.0, 4.0]);
    let pred = fitted.predict(&input);
    println!("Prediction: {:.2}", pred.data()[0]); // ≈ 11.0

    Ok(())
}
```

### Preprocessing Pipeline

```rust
use machinelearne_rs::{
    backend::CpuBackend,
    preprocessing::{
        StandardScaler, PolynomialFeatures,
        Transformer, FittedTransformer, Pipeline,
    },
    Tensor2D,
};

// Build a preprocessing pipeline
let pipeline = Pipeline::<CpuBackend>::new()
    .add(StandardScaler::new())
    .add(PolynomialFeatures::new(2).include_bias(false));

// Fit on training data
let fitted = pipeline.fit(&train_data)?;

// Transform data
let transformed = fitted.transform(&train_data)?;

// Save for later
fitted.save_to_file("preprocessing.bin")?;
```

### Grid Search with Cross-Validation

```rust
use machinelearne_rs::{
    backend::CpuBackend,
    model_selection::{
        KFold, LinearRegressionGrid,
        PreprocessingGrid, PipelineGridSearchCV, PolynomialGrid,
        ScalerGrid, ScalerType,
    },
    metrics::RegressionMetric,
};

// Define hyperparameter grid
let model_grid = LinearRegressionGrid::new()
    .with_learning_rates(vec![0.01, 0.1, 0.5])
    .with_lambdas(vec![0.0, 0.01, 0.1, 1.0]);

let preprocessing = PreprocessingGrid::new()
    .with_scaler(ScalerGrid::new(vec![ScalerType::Standard, ScalerType::MinMax]))
    .with_polynomial(PolynomialGrid::new(vec![1, 2]));

let pipeline_grid = PipelineGrid::new(preprocessing, model_grid);

// Run grid search with 5-fold CV
let cv = KFold::new(5).with_random_state(42);
let search = PipelineGridSearchCV::<CpuBackend, _>::new(pipeline_grid, RegressionMetric::R2)
    .with_cv(cv);

let result = search.fit(&x_train, &y_train)?;
println!("Best CV R2: {:.4}", result.best_params.mean_score);
```

### Complete ML Pipeline

```rust
use machinelearne_rs::pipeline::FittedPipeline;

// Load a trained pipeline (preprocessing + model)
let pipeline = FittedPipeline::<CpuBackend>::load_from_file("model.bin")?;

// Predict on raw data - preprocessing is applied automatically
let predictions = pipeline.predict(&raw_data)?;

// Save predictions or use in production
```

---

## Examples

Run the examples to see full workflows:

```bash
# Basic linear regression
cargo run --example train_linear

# With L2 regularization
cargo run --example train_linear_l2

# Binary classification
cargo run --example train_logistic

# Titanic survival prediction
cargo run --example titanic_pipeline

# California Housing with GridSearchCV
cargo run --example real_world_pipeline
```

---

## Architecture

The library follows a modular design with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline Layer                         │
│  (FittedPipeline: preprocessing + model for deployment)     │
├─────────────────────────────────────────────────────────────┤
│                   Model Selection Layer                     │
│  (Cross-validation, GridSearchCV, PipelineGridSearchCV)     │
├──────────────────────┬──────────────────────────────────────┤
│    Preprocessing     │              Training                │
│  ┌────────────────┐  │  ┌────────────────────────────────┐  │
│  │ Scaling        │  │  │ Trainer (orchestrates loop)    │  │
│  │ Imputation     │  │  │   ├── Loss (MSE, MAE, BCE)     │  │
│  │ Encoding       │  │  │   ├── Optimizer (SGD)          │  │
│  │ Feature Eng.   │  │  │   └── Regularizer (L1, L2)     │  │
│  │ Pipeline       │  │  └────────────────────────────────┘  │
│  └────────────────┘  │                                      │
├──────────────────────┴──────────────────────────────────────┤
│                       Model Layer                           │
│  (LinearRegression: TrainableModel → InferenceModel)        │
├─────────────────────────────────────────────────────────────┤
│                       Backend Layer                         │
│  (CpuBackend: Tensor1D, Tensor2D, scalar operations)        │
└─────────────────────────────────────────────────────────────┘
```

### Type Safety

Models use phantom types to enforce valid operations at compile time:

```rust
// Unfitted model - can train, cannot predict
let model: LinearModel<CpuBackend, Unfitted> = LinearRegression::new(2);

// After training - can predict, cannot train again
let fitted: LinearModel<CpuBackend, Fitted> = trainer.fit(model, &dataset)?;
```

---

## Design Principles

1. **Type Safety First**: Leverage Rust's type system to prevent shape mismatches and invalid operations at compile time.

2. **Separation of Concerns**: Following [ADR-0001](docs/adr/0001-separate-trainer-losses.md), training logic is separated from model parameters.

3. **Backend Agnostic**: Abstract `Backend` trait enables CPU/GPU implementations without changing model code.

4. **Zero-Cost Abstractions**: Generics and traits provide flexibility without runtime overhead.

5. **No Magic**: What you write is what you get - no implicit graph building or lazy evaluation.

---

## Documentation

- [Architecture Decision Records](docs/adr/) - Design rationale
- [CLAUDE.md](CLAUDE.md) - Guidelines for AI-assisted development
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## Status

This is a **learning-focused library**, not a production framework. It demonstrates:

- Clean ML library architecture in Rust
- Type-safe training/inference separation
- sklearn-compatible preprocessing API
- End-to-end pipeline serialization

Contributions and experiments welcome!

---

## License

MIT
