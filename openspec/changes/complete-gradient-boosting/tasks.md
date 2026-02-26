# Implementation Tasks

## 1. Model Comparison

- [x] 1.1 Create `ModelComparison` struct with models and test data, metrics storage
- [x] 1.2 Implement MSE, MAE, and R² metric computations
- [x] 1.3 Implement `best_model()` returning model with lowest MSE
- [x] 1.4 Implement `summary()` for formatted output
- [x] 1.5 Add unit tests for model comparison

## 2. Stacking Ensemble

- [x] 2.1 Create `StackingEnsemble` struct with base models and meta-learner
- [ ] 2.2 Implement `fit()` method using validation set for meta-learner training
- [x] 2.3 Implement `predict()` and `predict_batch()` methods
- [x] 2.4 Support heterogeneous base models (Linear, MLP, GB)
- [x] 2.5 Add unit tests for stacking ensemble

## 3. Hyperparameter Tuning

- [x] 3.1 Create `GridSearchGB` struct with parameter grid configuration
- [x] 3.2 Implement `search()` method returning all parameter combinations and metrics
- [x] 3.3 Implement `best()` and `top_n()` result methods
- [x] 3.4 Add configurable validation split and verbose output
- [x] 3.5 Add unit tests for grid search

## 4. Integration and Examples

- [x] 4.1 Export new types from `ensemble/mod.rs`
- [ ] 4.2 Create example: model comparison on California Housing
- [ ] 4.3 Create example: stacking ensemble with multiple model types
- [ ] 4.4 Create example: hyperparameter tuning workflow
- [ ] 4.5 Run cargo fmt and fix clippy warnings

## 5. Documentation

- [ ] 5.1 Update ADR-0010 with Phase 3 completion status
- [x] 5.2 Add documentation comments to all new public APIs
