## Context

Phase 1 of gradient boosting (ADR-0010) is complete with:
- `DecisionStump` weak learner (single split)
- `GradientBoostingTrainer` with basic configuration
- `BoostingLoss` trait with `LeastSquaresLoss`, `LogisticLoss`, `QuantileLoss`

This phase extends the implementation with deeper trees, early stopping, and feature subsampling to improve model quality and training efficiency.

## Goals / Non-Goals

**Goals:**
- Implement `DecisionTree` with configurable depth (1-10 levels)
- Add early stopping to prevent overfitting and reduce training time
- Add feature subsampling for regularization
- Maintain backward compatibility with existing API

**Non-Goals:**
- GPU acceleration (Phase 4)
- Histogram-based split finding (Phase 3)
- Multi-class classification (Phase 5)
- Feature importance extraction (Phase 5)

## Decisions

### D1: DecisionTree Architecture

**Decision**: Implement `DecisionTree` as a recursive tree structure with `TreeNode` enum.

**Rationale**:
- Recursive structure naturally maps to tree traversal
- Enum-based design supports both leaf and internal nodes cleanly
- Similar to `FittedStump` but with recursive nesting

**Alternatives considered**:
- Array-based tree (index-based navigation) - harder to implement, no clear benefit
- External tree library - would break backend abstraction

```rust
pub struct DecisionTreeConfig {
    pub max_depth: usize,           // Maximum tree depth (default: 3)
    pub min_samples_split: usize,   // Minimum samples to split (default: 2)
    pub min_samples_leaf: usize,    // Minimum samples in leaf (default: 1)
    pub max_features: Option<usize>,// Features to consider per split (None = all)
}

pub enum TreeNode {
    Leaf { value: f64 },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

pub struct FittedTree {
    root: TreeNode,
    n_features: usize,
}
```

### D2: Early Stopping Design

**Decision**: Implement early stopping via a callback-based approach.

**Rationale**:
- Consistent with existing `Trainer` callback pattern
- Allows flexible stopping criteria (loss, metric, patience)
- Non-invasive to existing training loop

```rust
pub struct EarlyStoppingConfig {
    pub validation_fraction: f64,   // Fraction of data for validation (default: 0.1)
    pub n_iter_no_change: usize,    // Iterations without improvement (default: 10)
    pub tol: f64,                   // Minimum improvement threshold (default: 1e-4)
}

// Integration in GradientBoostingTrainer
impl<L: BoostingLoss> GradientBoostingTrainer<L> {
    pub fn early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self
    }
}
```

### D3: Feature Subsampling

**Decision**: Add `colsample_bytree` parameter to `GradientBoostingConfig`.

**Rationale**:
- Standard technique in XGBoost/LightGBM
- Reduces overfitting by introducing randomness
- Simple to implement with existing infrastructure

```rust
pub struct GradientBoostingConfig {
    // Existing fields...
    pub colsample_bytree: f64,  // Fraction of features per tree (0.0-1.0, default: 1.0)
}
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| DecisionTree depth too high causes overfitting | Default max_depth=3, require explicit config for deeper trees |
| Feature subsampling with few features may fail | Clamp max_features to at least 1, warn if features < 5 |
| Early stopping may stop too early | Default n_iter_no_change=10 provides reasonable patience |

## Migration Plan

1. Add new types and configurations (backward compatible)
2. Update `GradientBoostingTrainer` with new methods
3. Add integration tests
4. Update documentation and examples

No breaking changes - all additions are additive.
