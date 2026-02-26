## ADDED Requirements

### Requirement: Validation callback configuration
The system SHALL provide a `ValidationCallback` that evaluates on a separate dataset.

#### Scenario: Create validation callback
- **WHEN** user creates `ValidationCallback` with dataset, loss function, and frequency
- **THEN** validation SHALL run every N epochs as specified

#### Scenario: Validation batch size
- **WHEN** user specifies batch size for validation
- **THEN** validation dataset SHALL be processed in batches of that size

### Requirement: Validation metrics
The validation callback SHALL compute and store validation metrics.

#### Scenario: Validation loss computed
- **WHEN** validation runs
- **THEN** `val_loss` metric SHALL be computed and added to training state

#### Scenario: Custom validation metrics
- **WHEN** user provides custom metric functions
- **THEN** those metrics SHALL be computed and stored

### Requirement: Validation results accessible
Validation results SHALL be accessible to other callbacks.

#### Scenario: Metrics in training state
- **WHEN** validation callback completes
- **THEN** metrics SHALL be available in `TrainingState.metrics` hashmap

### Requirement: Validation frequency control
Validation SHALL only run at specified epoch intervals.

#### Scenario: Every N epochs
- **WHEN** frequency is set to 5
- **THEN** validation SHALL run on epochs 5, 10, 15, etc.

#### Scenario: First epoch validation
- **WHEN** frequency allows validation on epoch 1
- **THEN** validation SHALL run after first epoch completes
