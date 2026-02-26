# feature-subsampling Specification

## Purpose
TBD - created by archiving change gradient-boosting-phase2. Update Purpose after archive.
## Requirements
### Requirement: Column subsampling per tree
The system SHALL support `colsample_bytree` parameter to randomly select features for each tree.

#### Scenario: Use 80% of features per tree
- **WHEN** user sets `config.colsample_bytree = 0.8` with 10 total features
- **THEN** each tree SHALL use approximately 8 randomly selected features

#### Scenario: Default uses all features
- **WHEN** user does not specify `colsample_bytree`
- **THEN** the system SHALL use all available features (colsample_bytree = 1.0)

### Requirement: Deterministic feature sampling
The system SHALL produce reproducible results when given the same random seed.

#### Scenario: Reproducible with seed
- **WHEN** training twice with the same seed and colsample_bytree
- **THEN** both models SHALL produce identical predictions

### Requirement: Feature sampling at tree level
The system SHALL select features once per tree (not per split).

#### Scenario: Features fixed per tree
- **WHEN** building a tree with 10 features and colsample_bytree=0.5
- **THEN** all splits in that tree SHALL only consider the same 5 sampled features

### Requirement: Minimum features guarantee
The system SHALL ensure at least 1 feature is selected.

#### Scenario: Few features edge case
- **WHEN** colsample_bytree would select 0 features
- **THEN** the system SHALL select at least 1 feature

### Requirement: Column subsampling integration with WeakLearner
The system SHALL pass feature mask to WeakLearner during fit.

#### Scenario: DecisionTree uses feature mask
- **WHEN** fitting DecisionTree with feature subsampling
- **THEN** only the sampled features SHALL be considered for splits

