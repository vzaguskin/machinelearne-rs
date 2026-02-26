## Context

Phase 1 and 2 of gradient boosting are complete with:
- `GradientBoostingTrainer` with configurable weak learners
- `DecisionStump` and `DecisionTree` weak learners
- `BoostingLoss` trait with `LeastSquaresLoss`, `LogisticLoss`, `QuantileLoss`
- Feature subsampling (`colsample_bytree`) and early stopping
- Full serialization via `InferenceModel` trait

The existing ensemble module follows ADR-0010 architecture with a separation between training and inference.

## Goals / Non-Goals

**Goals:**
- Add model comparison utilities for evaluating different gradient boosting configurations
- Add stacking ensemble meta-learner for combining multiple models
- Add hyperparameter tuning utilities (grid search)
- Provide comprehensive examples demonstrating all ensemble features

**Non-Goals:**
- GPU-accelerated training (Phase 4 in ADR-0010)
- Multi-class classification (Phase 5 in ADR-0010)
- Feature importance extraction (Phase 5 in ADR-0010)
- GOSS/EFB advanced sampling (Phase 5 in ADR-0010)

## Decisions

### Decision 1: Model Comparison via Metrics
**Choice**: Create `ModelComparison` struct that evaluates models on test data using standard metrics (MSE, MAE, R²).

**Rationale**: Simple and reusable. Users can compare any models implementing `StumpPredictor` or `InferenceModel`.

**Alternatives considered**:
- Cross-validation built-in: Rejected - adds complexity, users can implement their own CV loops
- Statistical significance tests: Rejected - overkill for typical ML workflows

### Decision 2: Stacking via Hold-out Validation Set
**Choice**: Stacking ensemble requires user to provide a validation set for training meta-learner.

**Rationale**: Keeps API simple and explicit. Avoids hiding complexity in internal cross-validation.

**Alternatives considered**:
- Built-in cross-validation: Rejected - significantly increases training time
- Pre-fitted models only: Rejected - limits flexibility

### Decision 3: Grid Search Implementation
**Choice**: Implement simple grid search that returns all results for user analysis.

**Rationale**: Transparent and allows users to analyze trade-offs. Matches sklearn's GridSearchCV pattern without CV.

**Alternatives considered**:
- Random search: Rejected - less deterministic, harder to reproduce
- Bayesian optimization: Rejected - adds dependencies, overkill for typical use cases

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Grid search can be slow with many hyperparameters | Document recommended parameter ranges, start small |
| Stacking can overfit with small datasets | Document minimum dataset sizes, encourage hold-out validation |
| Model comparison depends on test set quality | Document that comparison is only as good as the test data |
