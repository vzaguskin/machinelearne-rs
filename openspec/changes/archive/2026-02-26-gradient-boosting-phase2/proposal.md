## Why

Phase 1 (MVP) of gradient boosting is complete with DecisionStump as the weak learner. However, single-split stumps have limited expressiveness and require many iterations to learn complex patterns. Real-world gradient boosting implementations use deeper decision trees (depth 3-10) which capture more complex interactions per iteration, requiring fewer total estimators.

Additionally, the current implementation lacks early stopping and feature subsampling - essential techniques for preventing overfitting and improving generalization on real datasets.

## What Changes

- Add `DecisionTree` weak learner with configurable depth (1-10 levels)
- Add feature subsampling (random feature selection per tree) for regularization
- Integrate `LogisticLoss` with early stopping support for binary classification
- Add early stopping callback support to `GradientBoostingTrainer`
- Add tree-specific configuration: `max_depth`, `min_samples_split`, `min_samples_leaf`
- Add `colsample_bytree` parameter for feature subsampling ratio

## Capabilities

### New Capabilities

- `decision-tree`: Full decision tree implementation with configurable depth, supporting multi-way splits and recursive tree building
- `gradient-boosting-early-stopping`: Early stopping mechanism that monitors validation loss and stops training when improvement plateaus
- `feature-subsampling`: Random feature selection per tree iteration to reduce overfitting and improve generalization

### Modified Capabilities

- `gradient-boosting-regression`: Extend existing boosting trainer with early stopping, feature subsampling, and decision tree support

## Impact

- **New files**: `lib/src/ensemble/decision_tree.rs`
- **Modified files**: `lib/src/ensemble/boosting.rs`, `lib/src/ensemble/mod.rs`
- **API additions**: `DecisionTree` struct, `DecisionTreeConfig`, early stopping callbacks
- **Backward compatible**: All changes are additive, no breaking changes to existing API
