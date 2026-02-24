# Examples

This directory contains runnable examples demonstrating the machinelearne-rs library capabilities.

## Quick Start

```bash
# Run a basic example
cargo run --example train_linear

# Run with specific features
cargo run --example train_linear_wgpu --features wgpu
cargo run --example export_onnx --features onnx
```

---

## 1. Basic Training

### train_linear
Basic linear regression with synthetic data.

```bash
cargo run --example train_linear
```

**Demonstrates:**
- Creating and training a linear regression model
- MSE loss with SGD optimizer
- Basic prediction workflow

---

### train_linear_closed_form
Compares closed-form (normal equation) vs iterative SGD solutions.

```bash
cargo run --example train_linear_closed_form
```

**Demonstrates:**
- Exact closed-form solution for linear regression
- Performance comparison with SGD
- When to use each approach

---

## 2. Regularization & Loss Functions

### train_linear_l2
Linear regression with L2 (ridge) regularization.

```bash
cargo run --example train_linear_l2
```

**Demonstrates:**
- L2 regularization to prevent overfitting
- Training on data with outliers
- Regularization strength tuning

---

### train_linear_mae
Linear regression using Mean Absolute Error (MAE) loss.

```bash
cargo run --example train_linear_mae
```

**Demonstrates:**
- MAE loss function (robust to outliers)
- Comparison with MSE loss
- When to use MAE vs MSE

---

## 3. Classification

### train_logistic
Binary classification using logistic regression.

```bash
cargo run --example train_logistic
```

**Demonstrates:**
- Binary classification with BCEWithLogitsLoss
- Probability outputs
- Decision boundary interpretation

---

## 4. MLP Neural Networks

### train_mlp
MLP training on the XOR problem - demonstrates non-linear learning.

```bash
cargo run --example train_mlp
```

**Demonstrates:**
- Multi-layer perceptron architecture
- Learning non-linear patterns (XOR)
- Activation functions (ReLU, Tanh, Sigmoid)

---

### train_mlp_california
MLP regression on California Housing dataset.

```bash
cargo run --example train_mlp_california
```

**Demonstrates:**
- MLP on real-world regression data
- Feature normalization
- Architecture comparison (different hidden layer sizes)

---

### mlp_cpu_wgpu_comparison
Performance benchmark comparing CPU vs WGPU backends for MLP training.

```bash
cargo run --example mlp_cpu_wgpu_comparison --features wgpu
```

**Demonstrates:**
- GPU acceleration for neural networks
- Performance comparison (WGPU 10-20x faster for training, 6-16x for inference)
- Large batch and model benefits

---

## 5. Preprocessing Pipelines

### titanic_pipeline
Complete ML pipeline for Titanic survival prediction.

```bash
cargo run --example titanic_pipeline
```

**Demonstrates:**
- ColumnTransformer for mixed feature types
- Missing value imputation
- One-hot encoding for categorical features
- Pipeline serialization

---

### house_price_pipeline
House price prediction with feature engineering.

```bash
cargo run --example house_price_pipeline
```

**Demonstrates:**
- Heterogeneous preprocessing pipeline
- Polynomial feature engineering
- End-to-end ML workflow
- Model persistence

---

### real_world_pipeline
Pipeline selection and hyperparameter tuning.

```bash
cargo run --example real_world_pipeline
```

**Demonstrates:**
- Comparing naive vs optimized pipelines
- PipelineGridSearchCV for hyperparameter tuning
- Holdout test set evaluation

---

## 6. Model Selection

### gridsearch_california
Hyperparameter tuning on California Housing dataset.

```bash
cargo run --example gridsearch_california
```

**Demonstrates:**
- GridSearchCV with real dataset
- Parameter grid definition
- Cross-validation with KFold
- Best parameter selection

---

### pipeline_search
Full ML pipeline with GridSearchCV including preprocessing.

```bash
cargo run --example pipeline_search
```

**Demonstrates:**
- Grid search over preprocessing + model parameters
- Pipeline serialization for deployment
- Multiple preprocessing strategy comparison

---

## 7. GPU/WGPU Backend

### train_linear_wgpu
Basic tensor operations on GPU using WGPU.

```bash
cargo run --example train_linear_wgpu --features wgpu
```

**Demonstrates:**
- GPU tensor creation and operations
- Cross-platform support (Vulkan, Metal, D3D12, WebGPU)
- Backend initialization

---

### train_california_wgpu
California Housing regression on GPU.

```bash
cargo run --example train_california_wgpu --features wgpu
```

**Demonstrates:**
- Full training pipeline on GPU
- GPU preprocessing (StandardScaler)
- GPU adapter information display

---

### wgpu_cpu_comparison
Comprehensive WGPU vs CPU performance benchmark.

```bash
cargo run --example wgpu_cpu_comparison --features wgpu
```

**Demonstrates:**
- Performance across different dataset sizes
- Training stability features (gradient clipping, early stopping)
- Detailed accuracy and timing comparisons
- GPU vs CPU trade-off analysis

---

## 8. ONNX Export

### export_onnx
Export trained models to ONNX format.

```bash
cargo run --example export_onnx --features onnx
```

**Demonstrates:**
- ONNX model export
- Cross-platform model deployment
- ONNX model verification

---

### export_mlp_onnx
Export MLP neural networks to ONNX.

```bash
cargo run --example export_mlp_onnx --features onnx
```

**Demonstrates:**
- Neural network ONNX export
- Multi-layer model serialization
- ONNX structure verification

---

## 9. Deployment

### onnx_deployment
Complete ONNX deployment workflow with HTTP inference server.

```bash
cargo run --example onnx_deployment --features onnx-server
```

**Demonstrates:**
- Full pipeline preprocessing + model export to ONNX
- HTTP inference server setup
- Raw input handling (preprocessing included in ONNX)
- Native vs ONNX prediction comparison
- Production deployment workflow

---

## 10. Utilities

### dataset_loader
California Housing dataset loading utility.

```bash
cargo run --example dataset_loader
```

**Demonstrates:**
- CSV dataset loading
- Feature name mapping
- Dataset utility functions

---

### train_linear_ndarray
Linear regression using the ndarray backend.

```bash
cargo run --example train_linear_ndarray --features ndarray
```

**Demonstrates:**
- ndarray backend for tensor operations
- Integration with ndarray ecosystem
- Alternative backend selection

---

## Feature Requirements Matrix

| Example | Default | `ndarray` | `wgpu` | `onnx` | `onnx-server` |
|---------|:-------:|:---------:|:------:|:------:|:-------------:|
| train_linear | ✓ | | | | |
| train_linear_closed_form | ✓ | | | | |
| train_linear_l2 | ✓ | | | | |
| train_linear_mae | ✓ | | | | |
| train_logistic | ✓ | | | | |
| train_mlp | ✓ | | | | |
| train_mlp_california | ✓ | | | | |
| titanic_pipeline | ✓ | | | | |
| house_price_pipeline | ✓ | | | | |
| real_world_pipeline | ✓ | | | | |
| gridsearch_california | ✓ | | | | |
| pipeline_search | ✓ | | | | |
| dataset_loader | ✓ | | | | |
| train_linear_ndarray | | ✓ | | | |
| train_linear_wgpu | | | ✓ | | |
| train_california_wgpu | | | ✓ | | |
| wgpu_cpu_comparison | | | ✓ | | |
| mlp_cpu_wgpu_comparison | | | ✓ | | |
| export_onnx | | | | ✓ | |
| export_mlp_onnx | | | | ✓ | |
| onnx_deployment | | | | | ✓ |

---

## Learning Path

### Beginner
1. `train_linear` - Basic training workflow
2. `train_linear_l2` - Adding regularization
3. `train_logistic` - Classification basics

### Intermediate
4. `train_mlp` - Neural networks
5. `titanic_pipeline` - Preprocessing pipelines
6. `gridsearch_california` - Hyperparameter tuning

### Advanced
7. `real_world_pipeline` - Complete ML workflow
8. `export_onnx` - Model export
9. `wgpu_cpu_comparison` - GPU acceleration

### Production
10. `onnx_deployment` - Full deployment pipeline

---

## Running Examples

### Default Examples
Most examples work with default features:
```bash
cargo run --example <name>
```

### With Features
Some examples require additional features:
```bash
cargo run --example <name> --features <feature>
```

### All Features
To run with all features:
```bash
cargo run --example <name> --all-features
```

### Release Mode
For performance examples, use release mode:
```bash
cargo run --release --example mlp_cpu_wgpu_comparison --features wgpu
```
