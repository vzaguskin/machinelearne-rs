# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

#### MLP Model Selection and Pipeline Support
- `MLPGridSearchCV`: Hyperparameter grid search for MLP models with cross-validation
  - Search over architectures (hidden layer configurations)
  - Search over activation functions
  - Search over learning rates, batch sizes, epochs, and L2 regularization
- `MLPGrid`: Parameter grid for defining MLP hyperparameter search space
- `MLPArchitecture`: Configuration for hidden layer sizes
- `MLPActivations`: Configuration for per-layer activation functions
- `MLPFittedPipeline`: End-to-end pipeline combining preprocessing and MLP model
  - Supports preprocessing transformers (scalers, imputers, encoders)
  - Supports polynomial feature expansion
  - Full serialization/deserialization with "MLPM" magic bytes
  - Multi-output support (2D predictions)
  - `predict()` for 2D output, `predict_flattened()` for 1D output
- `mlp_inference_benchmark` example: Benchmark comparing CPU inference performance across different MLP architectures and batch sizes

#### MLP (Multi-Layer Perceptron) Neural Network
- `MLPModel<B, S>`: Configurable feedforward neural network with type-state pattern
  - Variable number of hidden layers
  - Configurable activation functions per layer (ReLU, Sigmoid, Tanh, Identity)
  - Full gradient computation for backpropagation
  - Xavier/Glorot weight initialization
- `Activation` enum: Supported activation functions with forward and backward methods
- `LayerParams<B>`: Parameters for a single dense layer
- `MLPParams<B>`: Container for all layer parameters
- `TrainableModel<B>` implementation for `MLPModel<B, Unfitted>` with forward/backward passes
- `InferenceModel<B>` implementation for `MLPModel<B, Fitted>` with predict/predict_batch
- `OnnxExportable` implementation for exporting trained MLP models to ONNX format
- `Optimizer<B, MLPParams<B>>` implementation for `SGD<B>`
- `Regularizer<B, MLP<B>>` implementations for L1, L2, and NoRegularizer
- Backend extensions: `relu_1d`, `relu_2d`, `tanh_1d`, `tanh_2d` methods
- WGPU shader support for ReLU and Tanh activations
- Examples: `train_mlp.rs` (XOR), `train_mlp_california.rs`, `export_mlp_onnx.rs`

#### WGPU Performance Optimizations (Phase 2)

- **Reduced GPU-CPU synchronization in training loop**: Loss computation now happens once per epoch instead of per batch
  - Previous: ~`num_batches × epochs` GPU syncs (e.g., 32,000 for large dataset)
  - Now: ~`epochs` GPU syncs (e.g., 50 for large dataset)
  - `compute_epoch_loss()`: New method that samples first batch for loss estimation
  - Training loop no longer calls `loss_fn.loss()` per batch - only `grad_wrt_prediction()` which stays on GPU

#### WGPU Performance Optimizations (Phase 1)
- `StagingBufferPool`: Pool staging buffers for efficient CPU readback
  - Size-based bucketing for efficient buffer reuse
  - LRU eviction when pool exceeds max size (64MB default)
  - Reduces allocation overhead on `to_vec()` calls
- `PooledStagingBuffer`: RAII wrapper that returns buffers to pool on drop
- `StagingPoolStats`: Statistics for monitoring pool usage
- Debug mode for eager flushing: `device.set_debug_mode(true)` for debugging
- Configurable flush threshold: `device.set_flush_threshold(n)` (default 500 operations)
- `AccumulatorStats.debug_mode`: Track debug mode status

### Changed
- Increased default command accumulator flush threshold from 50 to 500 operations
- `to_vec()` and `sum()` methods now use pooled staging buffers
- Command accumulator supports debug mode for eager flushing

#### Composable ONNX Export API
- `OnnxNodeBuilder` trait: Allows preprocessing transformers to contribute nodes to ONNX graphs
  - Implement `build_onnx_nodes()` to add transformer-specific operations
  - Enables custom transformer export without modifying library code
- `OnnxExportable` trait (refactored): New composable export API
  - `build_onnx_graph()`: Core method for building ONNX graphs
  - `to_onnx(model_name)`: Export to bytes with model name
  - `to_onnx_default()`: Convenience method for quick exports
  - `save_onnx(path, model_name)`: Save to file with optional model name
  - `save_onnx_to_path(path)`: Convenience method for saving to file
- Trait-based pipeline export: `FittedPipeline` now uses trait dispatch instead of hardcoded match statements
- All preprocessing transformers implement `OnnxNodeBuilder`:
  - StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
  - Normalizer
  - SimpleImputer
  - OneHotEncoder, OrdinalEncoder
  - PolynomialFeatures (degree 1 only)
- `GraphConstruction` error variant for graph building failures
- Extensibility: Users can implement `OnnxNodeBuilder` for custom transformers

#### ONNX Export and Inference
- `onnx` feature: Export trained models and pipelines to ONNX format for portable deployment
- `onnx-inference` feature: Load ONNX models and run inference using ONNX Runtime
- `onnx-server` feature: HTTP inference server for deploying models as microservices
- `onnx-cuda` feature: GPU acceleration via CUDA execution provider
- `OnnxExportable` trait: Export models to ONNX format with metadata support
- `OnnxInferenceSession`: Load and run ONNX model inference with batch support
- `OnnxServer`: HTTP server with `/predict`, `/predict/batch`, `/health`, `/ready` endpoints
- Full pipeline export: `FittedPipeline` exports preprocessing + model in single ONNX file
  - StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
  - Normalizer (L1, L2, Max norms)
  - SimpleImputer (mean, median, most_frequent, constant strategies)
  - OneHotEncoder, OrdinalEncoder
  - LinearRegressor, LinearClassifier
- Execution provider selection (CPU, CUDA with graceful fallback)
- `onnx_deployment` example: Complete end-to-end deployment workflow
  - Train model with preprocessing pipeline
  - Export full pipeline to ONNX
  - Start HTTP server
  - Make predictions via REST API
  - Compare native vs ONNX predictions

#### Training Stability Features
- `gradient_clipping(max_norm)`: Clips gradients by global L2 norm to prevent gradient explosion
  - Configured via `TrainerBuilder::gradient_clipping()`
  - Applied per-batch before optimizer step
- `early_stopping(patience, min_delta)`: Stops training when loss plateaus
  - Configured via `TrainerBuilder::early_stopping()`
  - Restores model parameters from best epoch
- `divergence_threshold(threshold)`: Detects and stops training when loss diverges
  - Configured via `TrainerBuilder::divergence_threshold()`
  - Returns error when loss exceeds `best_loss * threshold`
- `Scalar::sqrt()`: Square root method for scalar values
- `ParamOps::l2_norm()`: Computes L2 norm of parameters
- `ParamOps::clip_by_norm()`: Clips parameters to maximum L2 norm

#### WGPU Backend Examples and Benchmarks
- `train_linear_wgpu` example: Demonstrates GPU tensor operations using WGPU backend
- `train_california_wgpu` example: Full ML training pipeline on GPU with California Housing dataset
- Comprehensive test suite for WGPU backend operations
  - 52 tests verifying all tensor operations match CPU backend results
  - Tests cover: element-wise ops, scalar ops, linear algebra, reductions, unary math, broadcasting, column/row operations, tensor manipulation
  - Tolerance-based comparisons for GPU floating-point precision
  - Multiple tensor sizes tested (1, 4, 16, 64, 256, 1024 elements)
  - Displays GPU adapter information to verify GPU usage
  - Feature standardization using GPU compute shaders
  - Linear regression training on GPU
  - Training time, MSE, MAE, and R² metrics reporting
- WGPU backend benchmark in `backend_comparison` binary
  - Benchmarks WGPU against CPU and ndarray backends
  - Speedup comparison table with all backends
- `wgpu_cpu_comparison` example: Direct head-to-head comparison of WGPU vs CPU backends
  - Tests multiple dataset sizes (1K, 10K, 20K samples)
  - Measures training time and accuracy for identical operations
  - Analysis of GPU overhead vs benefit (currently shows WGPU ~200x slower - needs optimization)

#### Linear Algebra
- `linalg` module with matrix inverse (`inverse`) and normal equation solver (`solve_normal_equation`)
- `Tensor2D::matmul`: Matrix-matrix multiplication
- `Tensor2D::add`: Element-wise addition for 2D tensors
- `Tensor2D::transpose`: Matrix transpose method
- `Backend::matmul`: Matrix multiplication trait method

#### Linear Regression
- `LinearRegression::fit_closed_form`: One-step solution using normal equation (no hyperparameters needed)
- Example `train_linear_closed_form`: Compares closed-form vs SGD performance and accuracy

#### Regularization
- `L1<B>`: L1 (Lasso) regularizer for encouraging sparsity in model weights

#### Backend
- `sub_scalar_1d`, `div_scalar_1d`: Subtract/divide scalar from 1D tensor elements
- `sub_scalar_2d`, `div_scalar_2d`: Subtract/divide scalar from 2D tensor elements
- `Tensor1D::sub_scalar`, `Tensor1D::mul_scalar`, `Tensor1D::div_scalar`: Wrapper methods for scalar operations
- `Tensor2D::sub_scalar`, `Tensor2D::mul_scalar`, `Tensor2D::div_scalar`: Wrapper methods for scalar operations

#### WGPU Performance Optimizations
- `UniformBufferPool`: Thread-local pool for reusing uniform buffers in GPU operations
  - Reduces buffer allocation overhead for operation parameters
  - Organized by size categories (16, 32, 64, 128 bytes)
  - Replaces per-operation `create_buffer_init` with pooled buffers
- Fused kernels: Combined operations to reduce GPU dispatch overhead
  - `matvec_bias`: Single kernel for y = W @ x + b (combines matvec + add_scalar)
  - `sgd_step_inplace`: Single kernel for param -= lr * grad (combines mul + sub)
- `Tensor2D::dot_add_scalar()`: Fused matvec + bias operation
- `Backend::matvec_bias()`: Backend trait method with fused implementation
- `Backend::sgd_step()`: Backend trait method for optimizer step on GPU
- Command batching: Operations queued to accumulator for lazy execution
  - Single command encoder for batched operations
  - Auto-flush on `to_vec()` and `sum()` calls

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
