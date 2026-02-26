# decision-tree Specification

## Purpose
TBD - created by archiving change gradient-boosting-phase2. Update Purpose after archive.
## Requirements
### Requirement: DecisionTree configurable depth
The system SHALL provide a `DecisionTree` weak learner with configurable maximum depth from 1 to 10 levels.

#### Scenario: Create decision tree with depth 3
- **WHEN** user creates `DecisionTree::new().max_depth(3)`
- **THEN** the tree SHALL build at most 3 levels of splits

#### Scenario: Default depth is 3
- **WHEN** user creates `DecisionTree::new()` without specifying depth
- **THEN** the tree SHALL use max_depth=3 as default

### Requirement: DecisionTree split criteria
The system SHALL use mean-squared-error reduction as the split criteria for regression.

#### Scenario: Best split selection
- **WHEN** fitting a decision tree node
- **THEN** the system SHALL select the split that maximizes MSE reduction

### Requirement: DecisionTree minimum samples
The system SHALL support `min_samples_split` and `min_samples_leaf` parameters to control split eligibility.

#### Scenario: Min samples split prevents early split
- **WHEN** a node has fewer than `min_samples_split` samples
- **THEN** the node SHALL become a leaf without splitting

#### Scenario: Min samples leaf threshold
- **WHEN** a potential split would create a child with fewer than `min_samples_leaf` samples
- **THEN** the split SHALL be rejected

### Requirement: DecisionTree implements WeakLearner
The system SHALL implement the `WeakLearner` trait for `DecisionTree` to enable use in gradient boosting.

#### Scenario: Fit returns FittedTree
- **WHEN** calling `tree.fit(features, targets)`
- **THEN** the system SHALL return a `FittedTree` implementing `StumpPredictor`

### Requirement: FittedTree prediction
The system SHALL support both single-sample and batch prediction for `FittedTree`.

#### Scenario: Single sample prediction traverses tree
- **WHEN** predicting with `tree.predict_one(&features)`
- **THEN** the system SHALL traverse the tree from root to leaf and return the leaf value

#### Scenario: Batch prediction
- **WHEN** predicting with `tree.predict_batch(&features)`
- **THEN** the system SHALL return predictions for all samples in the batch

