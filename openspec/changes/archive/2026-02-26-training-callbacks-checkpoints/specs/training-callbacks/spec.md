## ADDED Requirements

### Requirement: Callback trait definition
The system SHALL provide a `Callback` trait with hooks for training lifecycle events.

#### Scenario: Callback with all hooks
- **WHEN** a callback implements all hook methods
- **THEN** each method SHALL be called at the appropriate training event

#### Scenario: Callback with no hooks
- **WHEN** a callback implements no methods (uses defaults)
- **THEN** training proceeds normally with no callback behavior

### Requirement: Training state access
Callbacks SHALL receive a `TrainingState` struct containing current training information.

#### Scenario: Read training progress
- **WHEN** callback accesses `TrainingState`
- **THEN** system SHALL provide current epoch, batch, loss, learning rate, and metrics

#### Scenario: Access model parameters
- **WHEN** callback needs to inspect model
- **THEN** system SHALL provide immutable reference to current parameters

### Requirement: Callback registration
The system SHALL allow registering multiple callbacks via `TrainerBuilder`.

#### Scenario: Register single callback
- **WHEN** user calls `.with_callback(callback)` on builder
- **THEN** callback SHALL be invoked during training

#### Scenario: Register multiple callbacks
- **WHEN** user registers multiple callbacks
- **THEN** callbacks SHALL be invoked in registration order

### Requirement: Early stopping via callback
Callbacks SHALL be able to request training termination.

#### Scenario: Callback stops training
- **WHEN** callback sets `stop_requested = true` in TrainingState
- **THEN** training SHALL terminate after current epoch completes

### Requirement: Callback lifecycle hooks
The system SHALL invoke callbacks at specific training events.

#### Scenario: on_train_start
- **WHEN** training begins before first epoch
- **THEN** `on_train_start` SHALL be called once

#### Scenario: on_train_end
- **WHEN** training completes (naturally or stopped)
- **THEN** `on_train_end` SHALL be called once

#### Scenario: on_epoch_start
- **WHEN** each epoch begins
- **THEN** `on_epoch_start` SHALL be called with current epoch number

#### Scenario: on_epoch_end
- **WHEN** each epoch completes
- **THEN** `on_epoch_end` SHALL be called with epoch results

#### Scenario: on_batch_end
- **WHEN** each batch completes
- **THEN** `on_batch_end` SHALL be called with batch loss
