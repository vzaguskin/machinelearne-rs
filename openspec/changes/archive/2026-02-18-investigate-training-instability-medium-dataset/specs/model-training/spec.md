## MODIFIED Requirements

### Requirement: Type-State Model Safety

Models SHALL use phantom type-state to enforce compile-time guarantees about training status.

#### Scenario: Unfitted model cannot predict

- **WHEN** a model is in `Unfitted` state
- **THEN** `predict()` method is not available at compile time
- **AND** only `forward()`, `backward()`, and `update_params()` are available

#### Scenario: Fitted model cannot train

- **WHEN** a model is in `Fitted` state
- **THEN** `forward()`, `backward()`, and `update_params()` are not available
- **AND** only `predict()` and `predict_batch()` are available

## ADDED Requirements

### Requirement: Trainer Stability Features

The trainer SHALL support optional stability features that are disabled by default for backward compatibility.

#### Scenario: Stability features are opt-in

- **WHEN** a trainer is built using default configuration
- **THEN** no gradient clipping is applied
- **AND** no early stopping is active
- **AND** behavior is identical to previous versions

#### Scenario: Combined stability features

- **WHEN** gradient clipping and early stopping are both enabled
- **THEN** clipping is applied per-batch
- **AND** early stopping is evaluated per-epoch
- **AND** both features work together correctly

### Requirement: Trainer Builder Extensibility

The TrainerBuilder SHALL support new configuration methods without breaking existing code.

#### Scenario: New methods are additive

- **WHEN** `gradient_clipping()`, `early_stopping()`, or `divergence_threshold()` are called
- **THEN** they return `Self` for method chaining
- **AND** existing builder methods continue to work
- **AND** method call order does not matter
