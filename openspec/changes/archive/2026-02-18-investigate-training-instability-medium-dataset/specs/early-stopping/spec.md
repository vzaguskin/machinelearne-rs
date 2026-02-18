## ADDED Requirements

### Requirement: Early Stopping Configuration

The trainer SHALL support optional early stopping based on loss improvement via the builder pattern.

#### Scenario: Enable early stopping

- **WHEN** `TrainerBuilder::early_stopping(5, 0.001)` is called
- **THEN** the trainer monitors loss improvement with patience of 5 epochs
- **AND** minimum delta of 0.001 is required to count as improvement

#### Scenario: Default no early stopping

- **WHEN** a trainer is built without calling `early_stopping()`
- **THEN** training continues for all `max_epochs`
- **AND** no early termination occurs

### Requirement: Early Stopping Triggers on Plateau

Training SHALL stop early when loss does not improve for the configured patience.

#### Scenario: Loss plateaus

- **WHEN** loss does not decrease by at least `min_delta` for `patience` consecutive epochs
- **THEN** training stops before reaching `max_epochs`
- **AND** the fitted model from the best epoch is returned

#### Scenario: Loss continues improving

- **WHEN** loss decreases by at least `min_delta` within the patience window
- **THEN** the patience counter resets
- **AND** training continues

### Requirement: Divergence Detection

The trainer SHALL detect and stop training when loss diverges significantly.

#### Scenario: Loss divergence

- **WHEN** `TrainerBuilder::divergence_threshold(10.0)` is set
- **AND** current loss exceeds best loss by more than 10x
- **THEN** training stops immediately
- **AND** an error or warning is returned indicating divergence

#### Scenario: No divergence threshold configured

- **WHEN** divergence threshold is not configured
- **THEN** training continues even if loss increases
- **AND** early stopping may still trigger based on patience

### Requirement: Best Model Restoration

When early stopping triggers, the trainer SHALL return the model from the epoch with best loss.

#### Scenario: Best model returned

- **WHEN** early stopping triggers at epoch 20 with best loss at epoch 15
- **THEN** the returned fitted model has parameters from epoch 15
- **AND** not the parameters from epoch 20
