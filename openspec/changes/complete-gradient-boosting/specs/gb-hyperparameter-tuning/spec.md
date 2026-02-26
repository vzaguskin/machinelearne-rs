## ADDED Requirements

### Requirement: Grid search hyperparameter tuning
The system SHALL provide `GridSearchGB` for finding optimal hyperparameters via exhaustive search.

#### Scenario: Grid search returns all results
- **WHEN** user runs `GridSearchGB::search(&features, &targets)`
- **THEN** the system SHALL return all parameter combinations and their MSE scores

#### Scenario: Grid search respects parameter ranges
- **WHEN** user specifies `n_estimators: vec![50, 100, 200]`
- **THEN** the system SHALL test all combinations with those values

### Requirement: Grid search configuration
The system SHALL allow configurable search parameters.

#### Scenario: Custom validation split
- **WHEN** user sets `validation_fraction = 0.2`
- **THEN** the system SHALL use 20% of data for evaluation

#### Scenario: Verbose output
- **WHEN** user sets `verbose = true`
- **THEN** the system SHALL print progress for each parameter combination

### Requirement: Best parameter selection
The system SHALL provide utilities for finding optimal parameters from search results.

#### Scenario: Get best parameters
- **WHEN** user calls `results.best()`
- **THEN** the system SHALL return parameters with lowest validation MSE

#### Scenario: Top-N results
- **WHEN** user calls `results.top_n(3)`
- **THEN** the system SHALL return the 3 best parameter configurations
