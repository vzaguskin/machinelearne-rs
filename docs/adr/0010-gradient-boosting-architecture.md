# ADR-0010: Gradient Boosting Architecture

## Status

Proposed

## Context

The machinelearne-rs library currently supports gradient descent-based models (LinearModel, MLPModel) trained via the `Trainer` abstraction. Users want gradient boosting for tabular data, which has fundamentally different training dynamics.

### Current Architecture

**Model System** (`lib/src/model/mod.rs`):
- `TrainableModel<B>`: For training with `forward()`, `backward()`, `params()`, `update_params()`
- `InferenceModel<B>`: For prediction with `predict()`, `predict_batch()`, serialization
- State machine: `Unfitted` → `Fitted` via `into_fitted()`

**Training System** (`lib/src/trainer/mod.rs`):
- Designed for **single-model gradient descent**
- Orchestrates: forward → loss gradient → backward → optimizer step
- Supports callbacks, early stopping, LR scheduling

**ADR-0001 Constraint**: Fitted models contain only inference parameters (no optimizer state)

### The Gradient Boosting Challenge

Gradient Boosting (GBDT) has **fundamentally different training dynamics**:

| Aspect | Gradient Descent | Gradient Boosting |
|--------|------------------|-------------------|
| Training | Iterative parameter updates | Sequential ensemble building |
| Target | Fixed targets | Residuals change each iteration |
| Update Rule | `params -= lr * gradient` | `F(x) += lr * h(x)` |
| Model Count | Single model | Multiple weak learners |

**Key Insight**: Boosting does NOT fit the existing Trainer pattern because:
1. Trainer expects gradient-based parameter updates
2. Boosting fits new models to residuals, not parameter updates
3. Each iteration produces a new weak learner, not updated parameters

### GPU Friendliness

Contrary to initial assumption, gradient boosting CAN be GPU-accelerated via **histogram-based algorithms**:

**Traditional exact split finding**: O(#data) per feature - requires sorting, not GPU-friendly

**Histogram-based (LightGBM, XGBoost gpu_hist, CatBoost)**:
1. Discretize continuous features into k bins (typically 256)
2. Build histograms: aggregate gradients/hessians per bin
3. Find best split by scanning bins: O(#bins) vs O(#data)

**Why GPU works**:
- Histogram building: Massive parallelism (all features, all samples simultaneously)
- Split finding: Fast parallel reduction on bin statistics
- Memory efficient: Only store bin counts, not sorted data

**Performance gains**: XGBoost reports 10-200x speedup with gpu_hist vs CPU

## Decision

Create a **separate ensemble module** that follows existing conventions but doesn't force-fit into the gradient descent Trainer pattern.

### Module Structure

```
lib/src/ensemble/
    mod.rs           - Public API and traits
    boosting.rs      - GradientBoostingTrainer and config
    decision_tree.rs - Decision tree implementation (Phase 2)
    decision_stump.rs - Simplified stump (Phase 1)
    ensemble_model.rs - GradientBoostedModel
    loss.rs          - BoostingLoss trait
```

### Core Traits

```rust
/// Weak learner that can be fit on residuals
pub trait WeakLearner<B: Backend> {
    type FittedModel: PredictSingle<B>;
    fn fit(&self, features: &Tensor2D<B>, targets: &Tensor1D<B>) -> Self::FittedModel;
}

/// Boosting-specific loss (different from gradient descent losses)
pub trait BoostingLoss: Clone {
    fn initial_prediction(&self, targets: &[f64]) -> f64;
    fn negative_gradient(&self, target: f64, prediction: f64) -> f64;
}
```

### GradientBoostedModel

```rust
pub struct GradientBoostedModel<B: Backend, H: PredictSingle<B>> {
    initial_prediction: Scalar<B>,
    learning_rate: f64,
    weak_learners: Vec<H>,
}

impl<B, H> InferenceModel<B> for GradientBoostedModel<B, H> {
    // Implements predict, predict_batch, serialization
    // Does NOT implement TrainableModel (training is different)
}
```

### GradientBoostingTrainer

```rust
pub struct GradientBoostingTrainer<L: BoostingLoss> {
    config: GradientBoostingConfig,
    loss: L,
}

impl<L> GradientBoostingTrainer<L> {
    pub fn fit<B, W>(&self, learner: &W, features: &Tensor2D<B>, targets: &Tensor1D<B>)
        -> GradientBoostedModel<B, W::FittedModel>
    where W: WeakLearner<B>
    {
        // 1. Initialize: F_0 = mean(targets)
        // 2. For each iteration:
        //    a. residuals = negative_gradient(targets, predictions)
        //    b. weak_learner = learner.fit(features, residuals)
        //    c. predictions += learning_rate * weak_learner.predict(features)
        // 3. Return fitted ensemble
    }
}
```

## Consequences

### What Works Well
- `InferenceModel` trait for serialization and prediction
- Backend abstraction (CPU, WGPU, ndarray)
- State separation (training config not in fitted model)

### What's Different
- **No TrainableModel implementation** - boosting doesn't use gradient descent
- **Separate trainer** - can't reuse existing Trainer
- **New loss trait** - BoostingLoss needs `negative_gradient()` not `grad_wrt_prediction()`

### Tradeoffs Accepted
1. Not reusing Trainer (paradigm mismatch)
2. Separate loss trait (boosting semantics differ)

## GPU Acceleration Path (Phase 3+)

**Key insight**: Modern GBDT (LightGBM, XGBoost, CatBoost) use **histogram-based algorithms** which ARE GPU-friendly:

| Algorithm | How it works | GPU Benefit |
|-----------|--------------|-------------|
| **Histogram binning** | Discretize continuous features into k bins (typically 256) | Parallel bin creation for all features |
| **Split finding** | Find best split by scanning bins, not data points | Parallel aggregation of gradients/hessians per bin |
| **Complexity** | O(#bins × #features) instead of O(#data × #features) | Massive parallelism with thousands of GPU cores |

**GPU Implementation Strategy**:
1. **Phase 1-2**: CPU-only with exact split finding (simpler, correct)
2. **Phase 3**: Add histogram-based trees (CPU)
3. **Phase 4**: GPU histogram kernel for WGPU backend

```rust
// GPU histogram kernel concept
@compute @workgroup_size(64)
fn build_histogram(
    features: array<f32>,
    gradients: array<f32>,
    bins: array<u32>,  // Pre-computed bin assignments
) {
    // Each thread processes samples, atomically adds to bin counters
    // Highly parallel - ideal for GPU
}
```

## Alternatives Considered

### Alternative 1: Extend Existing Trainer
**Rejected**: Trainer expects gradient descent. Boosting needs sequential ensemble building. Mixing violates single responsibility.

### Alternative 2: Generic EnsembleModel<B, M>
**Rejected**: Too generic. Gradient boosting has specific semantics (learning rate, sequential addition).

### Alternative 3: External Tree Library
**Rejected**: Would break backend abstraction and zero-overhead principle.

## Phased Implementation

### Phase 1: Foundation (MVP)
- [x] Create `ensemble/` module
- [ ] `BoostingLoss` trait with `LeastSquaresLoss`
- [ ] `DecisionStump` weak learner (exact split finding)
- [ ] `GradientBoostingTrainer` for regression
- [ ] `GradientBoostedModel` with `InferenceModel`
- [ ] Serialization and tests

### Phase 2: Full Trees
- [ ] `DecisionTree` with configurable depth
- [ ] Feature subsampling
- [ ] `LogisticLoss` for binary classification
- [ ] Early stopping

### Phase 3: Histogram-Based Trees (CPU)
- [ ] Implement histogram binning for continuous features
- [ ] Histogram-based split finding (O(#bins) vs O(#data))
- [ ] Gradient/hessian aggregation per bin
- [ ] Bin subtraction trick for faster child histogram computation

### Phase 4: GPU Acceleration
- [ ] WGPU compute shader for histogram building
- [ ] Parallel histogram aggregation
- [ ] GPU-based split finding
- [ ] Benchmark: CPU vs GPU histogram method

### Phase 5: Advanced
- [ ] Multi-class classification
- [ ] Feature importance
- [ ] GOSS (Gradient-based One-Side Sampling)
- [ ] EFB (Exclusive Feature Bundling)

## References

- ADR-0001: Separate trainer losses (stateless fitted models)
- ADR-0009: WGPU backend limitations (sync overhead in training loops)
- XGBoost: https://xgboost.readthedocs.io/en/stable/gpu/
- LightGBM: https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html
- CatBoost: https://catboost.ai/docs/features/gpu-training
