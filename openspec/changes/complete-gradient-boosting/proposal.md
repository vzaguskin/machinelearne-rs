## Why

Phase 1 and Phase 2 of gradient boosting have been implemented, providing:
- Basic boosting with DecisionStump
- DecisionTree with configurable depth
- Feature subsampling (colsample_bytree)
- Early stopping with validation loss

The library now needs advanced ensemble capabilities to complete the gradient boosting implementation:
- Model comparison and selection tools
- Stacking ensemble support
- Hyperparameter tuning utilities
- Complete examples demonstrating all features

## What Changes

- Add model comparison framework for comparing different ensemble configurations
- Add stacking ensemble meta-learner
- Add hyperparameter grid search for gradient boosting
- Add comprehensive examples demonstrating complete gradient boosting pipeline
- Archive the OpenSpec change after implementation

## Capabilities

### New Capabilities
- `model-comparison`: Framework for comparing gradient boosting models with different configurations
- `stacking-ensemble`: Meta-learner that combines multiple base models via a meta-learner
- `gb-hyperparameter-tuning`: Grid search for finding optimal hyperparameters

### Modified Capabilities
- None - this is a new functionality

## Impact

- **Affected modules**: `lib/src/ensemble/` - new modules for comparison and stacking
- **New examples**: Demonstrate complete gradient boosting workflow
- **Tests**: Unit and integration tests for new functionality
