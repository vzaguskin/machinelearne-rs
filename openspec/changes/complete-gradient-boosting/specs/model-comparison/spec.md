## ADDED Requirements

### Requirement: Model comparison with metrics
The system SHALL provide a `ModelComparison` struct for evaluating and comparing multiple fitted models on test data.

#### Scenario: Compare two models on test data
- **WHEN** user creates a `ModelComparison` with test features and targets
- **THEN** the system SHALL compute MSE, MAE, and R² metrics for each model

#### Scenario: Best model selection
- **WHEN** user calls `comparison.best_model()`
- **THEN** the system SHALL return the model with lowest MSE

### Requirement: Comparison result display
The system SHALL provide formatted output for model comparison results.

#### Scenario: Display comparison table
- **WHEN** user calls `comparison.summary()`
- **THEN** the system SHALL print a formatted table showing all models and their metrics
