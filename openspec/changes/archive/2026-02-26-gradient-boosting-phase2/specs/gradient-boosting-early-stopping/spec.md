## ADDED Requirements

### Requirement: Early stopping configuration
The system SHALL provide `EarlyStoppingConfig` to configure early stopping behavior in `GradientBoostingTrainer`.

#### Scenario: Configure early stopping
- **WHEN** user calls `trainer.early_stopping(EarlyStoppingConfig::default())`
- **THEN** the trainer SHALL monitor validation loss and stop when improvement plateaus

### Requirement: Validation split for early stopping
The system SHALL split training data into train/validation sets when early stopping is enabled.

#### Scenario: Default validation fraction
- **WHEN** early stopping is enabled without specifying validation_fraction
- **THEN** the system SHALL use 10% of training data for validation

#### Scenario: Custom validation fraction
- **WHEN** user sets `config.validation_fraction = 0.2`
- **THEN** the system SHALL use 20% of training data for validation

### Requirement: Patience-based stopping
The system SHALL stop training when validation loss doesn't improve for `n_iter_no_change` iterations.

#### Scenario: Stop after patience exhausted
- **WHEN** validation loss has not improved by at least `tol` for `n_iter_no_change` iterations
- **THEN** training SHALL stop early

#### Scenario: Continue if improving
- **WHEN** validation loss improves by at least `tol` at iteration N
- **THEN** patience counter SHALL reset to 0

### Requirement: Best model restoration
The system SHALL restore the model to the best iteration when early stopping triggers.

#### Scenario: Restore best iteration
- **WHEN** early stopping triggers at iteration 50 with best model at iteration 35
- **THEN** the returned model SHALL contain only the weak learners from iteration 35
