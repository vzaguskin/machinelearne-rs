# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

#### Regularization
- `L1<B>`: L1 (Lasso) regularizer for encouraging sparsity in model weights

#### Backend
- `sub_scalar_1d`, `div_scalar_1d`: Subtract/divide scalar from 1D tensor elements
- `sub_scalar_2d`, `div_scalar_2d`: Subtract/divide scalar from 2D tensor elements
- `Tensor1D::sub_scalar`, `Tensor1D::mul_scalar`, `Tensor1D::div_scalar`: Wrapper methods for scalar operations
- `Tensor2D::sub_scalar`, `Tensor2D::mul_scalar`, `Tensor2D::div_scalar`: Wrapper methods for scalar operations

### Fixed
- `len_2d` documentation now correctly states it returns number of rows (not total elements)

## [0.1.0] - 2025-02-17

### Added

#### Core ML
- Linear Regression model with type-state safety (`Unfitted` → `Fitted`)
- Loss functions: MSE, MAE, BCE with Logits
- SGD optimizer with configurable learning rate
- L1 and L2 regularizers
- Trainer builder pattern for orchestrating training loops
- Regression metrics: R2, MSE, MAE, RMSE

#### Preprocessing Module
- **Scaling transformers**:
  - StandardScaler (Z-score normalization)
  - MinMaxScaler (scale to [0, 1] or custom range)
  - RobustScaler (median/IQR-based, robust to outliers)
  - MaxAbsScaler (scale by maximum absolute value)
  - Normalizer (per-sample L1/L2/Max normalization)
- **Imputation**:
  - SimpleImputer with mean, median, most_frequent, constant strategies
- **Encoding**:
  - OneHotEncoder for categorical features
  - OrdinalEncoder for ordinal categorical features
  - LabelEncoder for target labels
- **Feature Engineering**:
  - PolynomialFeatures for polynomial and interaction terms
- **Pipeline**:
  - Chain multiple transformers with unified fit/transform API
- **ColumnTransformer**:
  - Apply different transformers to different feature columns

#### Model Selection
- KFold cross-validation with optional shuffling and random state
- Train/test split utility
- GridSearchCV for hyperparameter tuning with CV
- PipelineGridSearchCV for end-to-end pipeline optimization including:
  - Scaler selection (Standard, MinMax, Robust, None)
  - Polynomial degree (1, 2, etc.)
  - Imputation strategies
  - Learning rate, L2 lambda, batch size, epochs

#### Pipeline & Deployment
- FittedPipeline: combines preprocessing + trained model
- Serialization for models, transformers, and pipelines via bincode
- Pipeline metadata for versioning and model info

#### Examples
- `train_linear`: Basic linear regression
- `train_linear_l2`: L2 regularization
- `train_linear_mae`: MAE loss function
- `train_linear_ndarray`: Using ndarray backend
- `train_logistic`: Binary classification with BCE
- `titanic_pipeline`: End-to-end classification pipeline
- `house_price_pipeline`: Regression pipeline example
- `gridsearch_california`: GridSearchCV on California Housing
- `pipeline_search`: Pipeline hyperparameter search
- `real_world_pipeline`: Complete ML workflow with holdout test set

#### Infrastructure
- Backend abstraction trait (CpuBackend, NdarrayBackend)
- 85%+ test coverage with cargo-tarpaulin
- CI pipeline with fmt, clippy, and test checks
- Pre-commit hooks for code quality
- Benchmark suite for sklearn comparison

### Architecture Decisions
- [ADR-0001](docs/adr/0001-separate-trainer-losses.md): Separate Trainer, Losses, and Regularizers
- [ADR-0002](docs/adr/0002-method-oriented-tensor-api.md): Method-Oriented Tensor API
- [ADR-0003](docs/adr/0003-semantic-axis-labeling.md): Semantic Axis Labeling
- [ADR-0004](docs/adr/0004-model-and-pipeline-serialization.md): Model and Pipeline Serialization
- [ADR-0005](docs/adr/0005-backend_invalid_values_handling.md): Backend Invalid Values Handling
- [ADR-0006](docs/adr/0006-preprocessing-api.md): Preprocessing API Design
- [ADR-0007](docs/adr/0007-model-selection.md): Model Selection Module Design

### Performance
- Zero-cost abstractions via generics and traits
- Type-level state safety prevents runtime checks
- Efficient tensor operations with backend abstraction

## [0.0.1] - Initial Release

### Added
- Basic linear regression model
- MSE loss function
- SGD optimizer
- CpuBackend with Tensor1D/Tensor2D
- InMemoryDataset
- Basic trainer loop
