## ADDED Requirements

### Requirement: Stacking ensemble meta-learner
The system SHALL provide a `StackingEnsemble` for combining predictions from multiple base models via a meta-learner.

#### Scenario: Train stacking ensemble
- **WHEN** user trains a `StackingEnsemble` with base models and meta-learner on validation data
- **THEN** the system SHALL fit the meta-learner on base model predictions

#### Scenario: Predict with stacking ensemble
- **WHEN** user calls `ensemble.predict(&features)`
- **THEN** the system SHALL get predictions from all base models and combine via meta-learner

### Requirement: Stacking configuration
The system SHALL allow configurable base models and meta-learner.

#### Scenario: Custom meta-learner
- **WHEN** user specifies a custom meta-learner (e.g., GradientBoostingRegressor)
- **THEN** the system SHALL use that model for combining base predictions

#### Scenario: Heterogeneous base models
- **WHEN** user provides different types of base models (Linear, MLP, GB)
- **THEN** the system SHALL combine their predictions via the meta-learner
