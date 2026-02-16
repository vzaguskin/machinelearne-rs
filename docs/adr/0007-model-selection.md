# ADR-0007: Model Selection and Hyperparameter Tuning

## Status

Accepted

## Context

The library previously lacked utilities for:
1. **Train/test split** - Users had to manually partition data
2. **Cross-validation** - No K-fold or similar splitting strategies
3. **Hyperparameter search** - No systematic way to tune learning rate, regularization, batch size, or epochs
4. **Scoring metrics** - Metrics existed only in the benchmarks crate, unavailable to library users

Users need a sklearn-style `GridSearchCV` to:
- Search over learning rates: `[0.001, 0.01, 0.1]`
- Search over regularization: `[0.0, 0.01, 0.1, 1.0]`
- Search over batch sizes: `[16, 32, 64]`
- Search over epochs: `[100, 500, 1000]`
- Use K-fold cross-validation for robust evaluation

## Decision

We introduce a new `model_selection` module with the following components:

### 1. Metrics Module (`lib/src/metrics/`)

A `Scorer` trait that computes metrics from predictions and targets:

```rust
pub trait Scorer<B: Backend>: Clone {
    type Prediction;
    type Target;
    fn score(&self, prediction: &Self::Prediction, target: &Self::Target) -> f64;
    fn name(&self) -> &'static str;
}
```

`RegressionMetric` enum implements `Scorer` with:
- `NegMSE` - Negative Mean Squared Error (higher is better)
- `NegRMSE` - Negative Root Mean Squared Error
- `NegMAE` - Negative Mean Absolute Error
- `R2` - Coefficient of determination

All metrics follow sklearn's convention: **higher is better**.

### 2. Train/Test Split (`lib/src/model_selection/split.rs`)

```rust
pub fn train_test_split(
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(InMemoryDataset, InMemoryDataset), String>
```

### 3. Cross-Validation (`lib/src/model_selection/cross_validation.rs`)

```rust
pub trait CVSplit {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
    fn n_splits(&self) -> usize;
}

pub struct KFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}
```

### 4. Parameter Grids (`lib/src/model_selection/param_grid.rs`)

Type-safe parameter grid builders:

```rust
let grid = LinearRegressionGrid::new()
    .with_learning_rates(vec![0.001, 0.01, 0.1])
    .with_lambdas(vec![0.0, 0.1])
    .with_trainer(TrainerGrid::new(vec![16, 32], vec![100, 500]));
```

The grid provides:
- `n_combinations()` - Total number of parameter combinations
- `iter()` - Iterator over all `ParamCombination` values

### 5. GridSearchCV (`lib/src/model_selection/grid_search.rs`)

```rust
let grid_search = GridSearchCV::<CpuBackend, _>::new(
    param_grid,
    RegressionMetric::R2,
)
.with_cv(KFold::new(5).with_random_state(42))
.verbose(1);

let result = grid_search.fit(&dataset, n_features)?;
```

Returns:
- `best_model` - Fitted model with best parameters, retrained on full dataset
- `best_params` - Parameter combination and cross-validation score
- `all_results` - All combinations sorted by score

## Design Rationale

### Type-Safe Parameter Grids

Instead of using `HashMap<String, Vec<f64>>` like sklearn, we use strongly-typed structs:
- Compile-time validation of parameter names
- Better IDE support and autocompletion
- Clear documentation of available parameters

### Generic Backend

`GridSearchCV<B, S, C>` is generic over:
- `B: Backend` - CPU, GPU, or other compute backend
- `S: Scorer<B>` - Scoring metric
- `C: CVSplit` - Cross-validation strategy

### Reuse Existing Components

GridSearchCV reuses existing components:
- `Trainer` for fitting models
- `InMemoryDataset::subset()` for creating folds
- `InferenceModel::predict_batch()` for scoring

## Consequences

### Positive

- Systematic hyperparameter tuning is now possible
- Cross-validation provides robust model evaluation
- Metrics are available to all library users
- Type-safe API prevents configuration errors

### Negative

- Grid search can be slow for large parameter spaces
- No parallel evaluation (could be added in future)
- Currently only supports linear regression models

### Neutral

- `rand` dependency added for shuffling
- Metrics module exposes `Tensor1D` in public API

## Alternatives Considered

1. **Random Search** - Would require fewer evaluations, but grid search is more predictable and easier to debug.

2. **Bayesian Optimization** - More complex to implement, better suited for very large search spaces.

3. **sklearn-style dict-based grids** - Less type-safe, harder to document.

## Example Usage

```rust
use machinelearne_rs::{
    backend::CpuBackend,
    dataset::InMemoryDataset,
    model_selection::{
        GridSearchCV, KFold, LinearRegressionGrid,
        SGDGrid, RegularizerGrid, TrainerGrid,
    },
    metrics::RegressionMetric,
};

// Create dataset
let dataset = InMemoryDataset::new(x, y)?;

// Define parameter grid
let param_grid = LinearRegressionGrid::new()
    .with_learning_rates(vec![0.001, 0.01, 0.1])
    .with_regularizer(RegularizerGrid::L2 {
        lambdas: vec![0.0, 0.01, 0.1, 1.0]
    })
    .with_trainer(TrainerGrid::new(vec![32, 64], vec![500, 1000]));

// Run grid search
let grid_search = GridSearchCV::<CpuBackend, _>::new(
    param_grid,
    RegressionMetric::R2,
)
.with_cv(KFold::new(5).with_random_state(42))
.verbose(1);

let result = grid_search.fit(&dataset, n_features)?;

// Use best model
let predictions = result.best_model.predict_batch(&test_data);
```

## Future Work

1. **Parallel execution** - Add `rayon` for parallel fold evaluation
2. **Random search** - `RandomizedSearchCV` for large parameter spaces
3. **More models** - Extend to logistic regression and other models
4. **Early stopping** - Stop training when score plateaus
5. **Progress callbacks** - Allow custom progress reporting
