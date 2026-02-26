## MODIFIED Requirements

### Requirement: Trainer Builder Extensibility

The TrainerBuilder SHALL support new configuration methods without breaking existing code.

#### Scenario: New methods are additive

- **WHEN** `gradient_clipping()`, `early_stopping()`, `divergence_threshold()`, `with_callback()`, or `with_lr_scheduler()` are called
- **THEN** they return `Self` for method chaining
- **AND** existing builder methods continue to work
- **AND** method call order does not matter

#### Scenario: Register callback via builder
- **WHEN** `.with_callback(callback)` is called on builder
- **THEN** callback SHALL be registered for training events

#### Scenario: Register scheduler via builder
- **WHEN** `.with_lr_scheduler(scheduler)` is called on builder
- **THEN** scheduler SHALL adjust learning rate during training

### Requirement: Trainer Stability Features

The trainer SHALL support optional stability features that are disabled by default for backward compatibility.

#### Scenario: Stability features are opt-in

- **WHEN** a trainer is built using default configuration
- **THEN** no gradient clipping is applied
- **AND** no early stopping is active
- **AND** no callbacks are registered
- **AND** no learning rate scheduler is active
- **AND** behavior is identical to previous versions

#### Scenario: Combined stability features with callbacks

- **WHEN** gradient clipping, early stopping, and callbacks are all enabled
- **THEN** clipping is applied per-batch
- **AND** early stopping is evaluated per-epoch
- **AND** callbacks are invoked at appropriate events
- **AND** all features work together correctly
