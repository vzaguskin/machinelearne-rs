## ADDED Requirements

### Requirement: Gradient Clipping Configuration

The trainer SHALL support optional gradient clipping by global L2 norm via the builder pattern.

#### Scenario: Enable gradient clipping

- **WHEN** `TrainerBuilder::gradient_clipping(1.0)` is called
- **THEN** the resulting trainer clips gradients to a maximum L2 norm of 1.0
- **AND** clipping is applied before the optimizer step

#### Scenario: Default no clipping

- **WHEN** a trainer is built without calling `gradient_clipping()`
- **THEN** no gradient clipping is applied
- **AND** gradients are passed unchanged to the optimizer

### Requirement: Gradient Clipping Preserves Direction

Gradient clipping SHALL rescale gradients while preserving their direction.

#### Scenario: Gradients exceed max norm

- **WHEN** gradients have L2 norm of 5.0 and max_norm is 1.0
- **THEN** gradients are scaled to have L2 norm of 1.0
- **AND** the direction (unit vector) is unchanged

#### Scenario: Gradients within max norm

- **WHEN** gradients have L2 norm of 0.5 and max_norm is 1.0
- **THEN** gradients are unchanged
- **AND** no scaling is applied

### Requirement: Gradient Clipping Applied Per Batch

Gradient clipping SHALL be applied to each batch before the optimizer step.

#### Scenario: Clipping in training loop

- **WHEN** training with gradient clipping enabled
- **THEN** each batch's gradients are clipped independently
- **AND** clipping happens after combining loss and regularizer gradients
- **AND** clipping happens before calling `optimizer.step()`
