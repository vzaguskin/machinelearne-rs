# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`machinelearne-rs` is a type-safe, zero-overhead machine learning library in Rust built around **explicit separation of concerns**:
- **Models** know how to predict and compute gradients
- **Losses** know how to score predictions and emit per-sample gradients
- **Optimizers** update parameters based on gradients
- **Regularizers** penalize model complexity independently
- **Trainers** orchestrate the loop — nothing more

The design follows ADR-0001: a fitted model contains **only** inference parameters, with no optimizer state, loss functions, or training hyperparameters.

## Common Commands

### Building and Testing

```bash
# Build entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace --all-features

# Run tests for library only
cargo test -p machinelearne-rs --all-features

# Run coverage (requires cargo-tarpaulin)
cargo tarpaulin --workspace --exclude benchmarks-sklearn --all-features --out Xml --fail-under 85

# Run a specific example
cargo run --example train_linear
cargo run --example train_linear_ndarray --features ndarray
```

### Running Examples

Examples are in `lib/examples/`:
```bash
cargo run --example train_linear          # Basic linear regression
cargo run --example train_linear_l2       # L2 regularization
cargo run --example train_linear_mae      # MAE loss
cargo run --example train_linear_ndarray   # Using ndarray backend (requires --features ndarray)
cargo run --example train_logistic        # Binary classification with BCE
cargo run --example train_mlp             # MLP learning XOR function
cargo run --example train_mlp_california  # MLP on California Housing dataset
cargo run --example export_mlp_onnx --features onnx  # Export MLP to ONNX
```

## Git Workflow

1. **Always run `cargo fmt` before committing** to ensure consistent code formatting.
2. **Always work in a feature branch**, never commit directly to `main`. Create a branch like `feature/your-feature-name`.
3. **Pre-commit hook enforces branch protection**: The repository has a pre-commit hook that blocks direct commits to `main`. If you attempt to commit on `main`, you'll see an error directing you to create a feature branch first.

### Initial Setup

After cloning the repository, run the hook setup script:

```bash
./scripts/setup-hooks.sh
```

This installs the pre-commit hook that prevents direct commits to `main`.

## Feature Development Process

**CRITICAL**: Follow this process for EVERY feature. No exceptions.

**Important**: This workflow applies to ALL development, including when using OpenSpec commands like `/opsx:ff`, `/opsx:apply`, `/opsx:continue`, etc. These commands create artifacts and implement code, but you must still:
1. Create a feature branch BEFORE running them
2. Commit all changes to the feature branch
3. Create a PR from the feature branch
4. Merge via PR (not direct commit to main)

### 1. Planning Phase

Before writing any code, complete these steps in order:

```bash
# 1. Create OpenSpec change request
openspec new change "feature-name-in-kebab-case"

# 2. Create GitHub issue (reference the OpenSpec change)
gh issue create --title "feat: short description" --body "Description linking to OpenSpec change"

# 3. Create feature branch from main
git checkout main && git pull
git checkout -b feature/feature-name-in-kebab-case
```

**OpenSpec artifacts** (all required before implementation):
- `proposal.md` - WHY this change is needed
- `specs/<capability>/spec.md` - WHAT the system should do
- `design.md` - HOW to implement it
- `tasks.md` - Implementation checklist with `- [ ]` format

### 2. Implementation Phase

```bash
# Work through tasks in tasks.md
# Mark each task complete as you go: - [ ] → - [x]

# Run tests frequently
cargo test -p machinelearne-rs --all-features

# Format before committing
cargo fmt
```

### 3. Pre-Merge Phase (Single PR)

Before creating the PR, archive the OpenSpec change **in the feature branch**:

```bash
# Archive OpenSpec (creates spec files, moves change to archive/)
openspec archive "feature-name-in-kebab-case" -y

# Commit the archive
git add openspec/
git commit -m "chore(openspec): archive feature-name change"
```

This ensures the feature implementation AND OpenSpec archive are in one PR.

### 4. Pull Request Phase

```bash
# Push feature branch
git push -u origin feature/feature-name-in-kebab-case

# Create PR
gh pr create --title "feat: description" --body "Summary, changes, test plan"
```

**PR must include:**
- Feature implementation
- OpenSpec archive commit
- CHANGELOG.md update
- All tests passing

### 5. Post-Merge Phase

After PR is merged:

```bash
# Switch back to main
git checkout main && git pull

# Close the GitHub issue
gh issue close <issue-number> --comment "Implemented in #<pr-number>"
```

### Summary Checklist

For every feature, ensure:
- [ ] Feature branch created FIRST (`git checkout -b feature/name`) - enforced by pre-commit hook
- [ ] OpenSpec change created (`openspec new change "name"`)
- [ ] GitHub issue created (`gh issue create`)
- [ ] All OpenSpec artifacts complete (proposal, specs, design, tasks)
- [ ] Implementation complete with tests
- [ ] CHANGELOG.md updated
- [ ] OpenSpec archived in feature branch (`openspec archive "name"`)
- [ ] PR created and CI passing
- [ ] PR merged
- [ ] GitHub issue closed

### Quick Reference Commands

```bash
# OpenSpec workflow
openspec new change "feature-name"           # Create new change
openspec status --change "feature-name"      # Check progress
openspec instructions apply --change "name"  # Get implementation instructions
openspec archive "feature-name" -y           # Archive completed change

# GitHub workflow
gh issue create --title "..." --body "..."   # Create issue
gh issue list                                # List open issues
gh issue close <num> --comment "..."         # Close issue
gh pr create --title "..." --body "..."      # Create PR
gh pr checks <num> --watch                   # Watch CI
gh pr merge <num> --squash --delete-branch   # Merge PR
```

## Documentation Workflow

After implementing any feature, follow these steps:

1. **Update CHANGELOG.md**: Add the new feature under `[Unreleased]` or the appropriate version section. Include:
   - What was added/changed/fixed
   - Any new public APIs or types
   - New examples if applicable

2. **Update README.md** if necessary:
   - Add new features to the Features section
   - Add new code examples if the feature changes the API
   - Update the architecture diagram if the structure changes
   - Add new examples to the Examples section

3. **Run doc tests** to verify all code examples compile:
   ```bash
   cargo test -p machinelearne-rs --all-features --doc
   ```

4. **Run all tests** before committing:
   ```bash
   cargo test -p machinelearne-rs --all-features
   ```

## Architecture Overview

### Core Design Principles

1. **Stateful Type Safety**: Models carry their training state in the type system (`Unfitted` vs `Fitted`), preventing invalid operations at compile time.

2. **Training/Inference Separation**: Trained models contain only prediction parameters; training logic lives in separate components (losses, optimizers, trainers).

3. **Backend Agnosticism**: Abstract `Backend` trait enables CPU/GPU implementations without changing model code.

4. **Zero-Cost Abstractions**: Generics and traits provide flexibility without runtime overhead.

### Key Traits and Types

#### Backend (`lib/src/backend/mod.rs`)

The `Backend` trait abstracts computation devices. Backend selection happens at compile time via type parameters.

- `CpuBackend` (default, pure-Rust, `cpu` feature)
- `NdarrayBackend` (interop with ndarray ecosystem, `ndarray` feature)

Tensor types are backend-specific: `Tensor1D<B>`, `Tensor2D<B>`, `Scalar<B>`.

The trait defines:
- Constructors: `from_vec_1d`, `from_vec_2d`, `zeros_1d`, `zeros_2d`
- Element-wise ops: `add_1d`, `sub_1d`, `mul_1d`, `mul_scalar_1d`, etc.
- Linear algebra: `matvec`, `matvec_transposed`, `transpose`
- Reductions: `mean_all_1d`, `sum_all_1d`, etc.

**Note**: There are both checked (`matvec`) and unchecked (`_matvec_unchecked`) variants. Use unchecked only when you're certain shapes are compatible.

#### Model System (`lib/src/model/mod.rs`)

Models are split into two phases via marker types in `state.rs`:

- `TrainableModel<B>`: Used during training with state `Unfitted`
  - Methods: `forward`, `backward`, `params`, `update_params`, `into_fitted`
- `InferenceModel<B>`: Lightweight, serializable predictor with state `Fitted`
  - Methods: `predict`, `predict_batch`, `extract_params`, `from_params`, `save_to_file`, `load_from_file`

`LinearModel<B, S>` is the main implementation:
- `LinearRegression<B>` = `LinearModel<B, Unfitted>` for training
- `LinearModel<B, Fitted>` for inference
- `LinearRegressor` = `LinearRegression<CpuBackend>` (convenience alias)

**Key invariant**: You cannot call `predict()` on an `Unfitted` model — this is enforced at compile time.

#### MLP Model (`lib/src/model/mlp.rs`)

`MLPModel<B, S>` implements a Multi-Layer Perceptron neural network:
- Configurable architecture with variable hidden layers
- Per-layer activation functions (ReLU, Sigmoid, Tanh, Identity)
- Xavier/Glorot weight initialization
- Full backpropagation support

Type aliases:
- `MLP<B>` = `MLPModel<B, Unfitted>` for training
- `MLPModel<B, Fitted>` for inference
- `MLPRegressor` = `MLP<CpuBackend>` (convenience alias)

Usage:
```rust
use machinelearne_rs::model::{MLP, Activation, InferenceModel};

// Create: 2 inputs -> 8 hidden (Tanh) -> 1 output (Sigmoid)
let model = MLP::<CpuBackend>::new(&[2, 8, 1], &[Activation::Tanh, Activation::Sigmoid]);

// Train using existing Trainer infrastructure
let trainer = Trainer::builder(MSELoss, SGD::new(0.5), NoRegularizer)
    .batch_size(1)
    .max_epochs(5000)
    .build();
let fitted = trainer.fit(model, &dataset).unwrap();

// Inference
let prediction = fitted.predict(&input);

// Export to ONNX
fitted.save_onnx(&path, Some("mlp_model")).unwrap();
```

#### Loss Functions (`lib/src/loss/mod.rs`)

`Loss<B>` trait requires:
- `loss(&prediction, &target)` → scalar loss value
- `grad_wrt_prediction(&prediction, &target)` → gradient w.r.t. prediction

Available losses:
- `MSELoss`: Mean squared error
- `MAELoss`: Mean absolute error
- `BCEWithLogitsLoss`: Binary cross-entropy with logits (numerically stable)

Loss gradients are passed to `model.backward()` to compute parameter gradients.

#### Optimizers (`lib/src/optimizer/mod.rs`)

`Optimizer<B, P>` trait:
- `step(&params, &gradients)` → new parameters (immutable return)

Current optimizers:
- `SGD<B>`: Stochastic gradient descent with learning rate

The step rule is typically: `params_new = params - lr * gradients`.

#### Regularizers (`lib/src/regularizers/mod.rs`)

`Regularizer<B, M>` trait:
- `regularizer_penalty_grad(&model)` → `(penalty_scalar, gradients)`

Available regularizers:
- `L2<B>`: Ridge regularization (λ||w||²) — only applies to weights, not bias
- `NoRegularizer`: Zero penalty and zero gradient (default)

#### Trainer (`lib/src/trainer/mod.rs`)

`Trainer` orchestrates the training loop. Built via `TrainerBuilder`:

```rust
Trainer::builder(loss_fn, optimizer, regularizer)
    .batch_size(32)
    .max_epochs(1000)
    .build()
```

Then call `trainer.fit(model, dataset)` to train.

The trainer:
- Iterates batches via `dataset.batches::<B>(batch_size)`
- Computes predictions via `model.forward()`
- Computes loss gradient via `loss_fn.grad_wrt_prediction()`
- Computes model gradients via `model.backward()`
- Adds regularizer gradients
- Updates parameters via `optimizer.step()`
- Returns a `Fitted` model for inference

### Dataset Trait (`lib/src/dataset/mod.rs`)

`Dataset` trait provides uniform access to training data:
- `len()` → `Option<usize>` (may be unknown for streaming data)
- `get_batch<B>(range)` → `Result<(Tensor2D<B>, Tensor1D<B>)>`
- `batches<B>(batch_size)` → `DatasetBatchIter` for iteration

`InMemoryDataset` is the primary implementation, storing `Vec<Vec<f32>>` for features and `Vec<f32>` for targets.

## Important Implementation Details

### Parameter Operations

`ParamOps<B>` trait is implemented by parameter structures (e.g., `LinearParams<B>`):
- `add(&self, other)` → element-wise addition
- `scale(&self, scalar)` → element-wise multiplication by scalar

This is used by optimizers to compute parameter updates.

### Serialization

Models implement `InferenceModel` with `ParamsRepr` type for serialization:

```rust
// Save
fitted_model.save_to_file("model.bin")?;

// Load
let loaded = LinearModel::<CpuBackend, Fitted>::load_from_file("model.bin")?;
```

Serialization uses `bincode` under the hood (requires `serde` feature).

### Type-Level State

The `Unfitted` and `Fitted` marker types are phantom types used to encode state:

```rust
pub struct LinearModel<B: Backend, S> {
    params: LinearParams<B>,
    _state: PhantomData<S>,  // S is Unfitted or Fitted
}
```

This enables:
```rust
impl<B> TrainableModel<B> for LinearModel<B, Unfitted> { ... }
impl<B> InferenceModel<B> for LinearModel<B, Fitted> { ... }
```

Only `TrainableModel` implements `into_fitted()`, which consumes the `Unfitted` model and returns a `Fitted` one.

### Gradient Flow

1. Forward pass: `pred = model.forward(&x)`
2. Loss gradient: `grad_pred = loss_fn.grad_wrt_prediction(&pred, &y)`
3. Parameter gradients: `grads = model.backward(&x, &grad_pred)`
4. Regularizer gradients: `(reg_penalty, reg_grads) = regularizer.regularizer_penalty_grad(&model)`
5. Combined: `total_grads = grads + reg_grads`
6. Update: `new_params = optimizer.step(model.params(), &total_grads)`
7. Apply: `model.update_params(&new_params)`

For linear regression:
- `forward`: `x @ w + b`
- `backward`: `∇w = x^T @ grad_pred`, `∇b = sum(grad_pred)`

### Backend Feature Gates

The library uses Cargo features to enable different backends:
- `cpu` (default): Pure-Rust CPU backend
- `ndarray`: Backend backed by ndarray crate
- `wgpu`: GPU backend using wgpu (Vulkan/Metal/D3D12)
- `serde` (default): Model serialization support

When adding backend-specific code, always wrap in `#[cfg(feature = "...")]`.

### WGPU Backend Performance Notes

The WGPU backend provides GPU acceleration but has important performance limitations:

**Current Status:**
- WGPU backend is functionally correct but **not viable for training**
- Expect ~100-200x slower than CPU for typical training workloads
- Root cause: Training loops trigger GPU synchronization per batch (for loss computation)
- See ADR-0009 for detailed analysis

**Optimizations Implemented:**
- Staging buffer pooling: Reuses buffers for CPU readback
- Command accumulation: Batches GPU operations (default 500 ops before flush)
- Debug mode: `device.set_debug_mode(true)` for eager flushing during debugging

**When to Use WGPU:**
- **Recommended**: Inference on pre-trained models (single forward pass, minimal sync)
- **Recommended**: Large-batch predictions
- **Not recommended**: Training loops (sync overhead negates GPU benefits)
- **Future**: Consider CUDA/cuBLAS backend for GPU training (designed for ML workloads)

**Why Async Backend Won't Help:**
The sync points are inherent to training, not the API. Loss computation requires the scalar value on CPU for metrics, early stopping, and LR scheduling. Even with async, training would still need to wait for loss values. See ADR-0009 for detailed analysis.

**Debugging WGPU:**
```rust
let device = WgpuDevice::global();
device.set_debug_mode(true);  // Flush after each operation for debugging
device.set_flush_threshold(50);  // Adjust threshold as needed
```

## Working with the Codebase

### Adding a New Model

1. Create a struct with a state parameter: `struct MyModel<B, S> { params: MyParams<B>, _state: PhantomData<S> }`
2. Implement `TrainableModel<B>` for `MyModel<B, Unfitted>`:
   - Define `forward`, `backward`, `params`, `update_params`, `into_fitted`
3. Implement `InferenceModel<B>` for `MyModel<B, Fitted>`:
   - Define `predict`, `predict_batch`, `extract_params`, `from_params`
4. Implement `ParamOps<B>` for your parameter type
5. Add a convenience type alias: `type MyRegressor = MyModel<CpuBackend, Unfitted>`

### Adding a New Backend

1. Create module with feature gate: `#[cfg(feature = "mybackend")]`
2. Implement `Backend` trait with concrete tensor types
3. Export types: `pub use self::mybackend::{MyBackend, MyTensor1D, MyTensor2D};`

See `lib/src/backend/cpu.rs` for reference implementation.

### Adding a New Loss Function

1. Create a zero-sized struct: `pub struct MyLoss;`
2. Implement `Loss<B>` for your struct:
   - Define `Prediction` and `Target` associated types
   - Implement `loss()` method (scalar value)
   - Implement `grad_wrt_prediction()` method (gradient)

See `lib/src/loss/mod.rs` for examples.

### Testing Strategy

- Unit tests are co-located with implementation files
- Integration tests for training pipelines exist in `lib/src/lib.rs`
- CI requires 85% code coverage via `cargo-tarpaulin`
- Test both forward and backward passes
- Test edge cases (empty datasets, single samples, numerical stability)

## File Structure Notes

- `lib/src/` — Main library source
  - `backend/` — Tensor operations and Backend trait
  - `model/` — Model implementations with state tracking
  - `loss/` — Differentiable loss functions
  - `optimizer/` — Gradient-based optimizers
  - `trainer/` — Training loop orchestration
  - `regularizers/` — Weight regularization
  - `dataset/` — Data loading abstractions
  - `serialization.rs` — Model persistence utilities
- `lib/examples/` — Runnable training examples
- `docs/adr/` — Architecture Decision Records
- `benchmarks/` — Performance benchmarks (separate workspace member)
