## ADDED Requirements

### Requirement: Adam optimizer construction
The system SHALL provide an `Adam<B>` optimizer that can be constructed with configurable hyperparameters.

#### Scenario: Default hyperparameters
- **WHEN** user creates Adam optimizer with only learning rate
- **THEN** system SHALL use default values beta1=0.9, beta2=0.999, epsilon=1e-8

#### Scenario: Custom hyperparameters
- **WHEN** user creates Adam optimizer with custom beta1, beta2, epsilon
- **THEN** system SHALL store and use the provided values

#### Scenario: Builder pattern
- **WHEN** user wants to customize individual hyperparameters
- **THEN** system SHALL provide builder methods for fluent configuration

### Requirement: Adam optimization step
The system SHALL implement the `Optimizer<B, P>` trait for Adam, updating parameters using the Adam algorithm with bias correction.

#### Scenario: First optimization step
- **WHEN** Adam performs the first step with non-zero gradients
- **THEN** system SHALL apply bias correction using t=1

#### Scenario: Multiple optimization steps
- **WHEN** Adam performs consecutive steps
- **THEN** system SHALL accumulate momentum and velocity across steps with incrementing time step

#### Scenario: Zero gradients
- **WHEN** gradients are all zeros
- **THEN** system SHALL maintain existing momentum/velocity but not change parameters significantly

### Requirement: Parameter type support
The system SHALL support Adam optimization for all model parameter types that support `ParamOps<B>`.

#### Scenario: LinearParams optimization
- **WHEN** Adam is used with LinearParams
- **THEN** system SHALL correctly update weights and bias with adaptive learning rates

#### Scenario: MLPParams optimization
- **WHEN** Adam is used with MLPParams
- **THEN** system SHALL correctly update all layer parameters with shared optimizer state

### Requirement: Hyperparameter accessors
The system SHALL provide read access to Adam hyperparameters.

#### Scenario: Learning rate accessor
- **WHEN** user calls learning_rate() on Adam
- **THEN** system SHALL return the current learning rate value

#### Scenario: Momentum decay accessor
- **WHEN** user calls beta1() on Adam
- **THEN** system SHALL return the first moment decay rate

### Requirement: Backend compatibility
The system SHALL work with any backend implementing the `Backend` trait.

#### Scenario: CPU backend
- **WHEN** Adam is instantiated with CpuBackend
- **THEN** system SHALL perform all operations using CPU tensors

#### Scenario: WGPU backend
- **WHEN** Adam is instantiated with WgpuBackend
- **THEN** system SHALL perform all operations using GPU tensors

### Requirement: Numerical stability
The system SHALL handle numerical edge cases gracefully.

#### Scenario: Small gradient values
- **WHEN** gradients contain very small values near zero
- **THEN** system SHALL not produce NaN or Inf due to epsilon in denominator

#### Scenario: Large gradient values
- **WHEN** gradients contain large values
- **THEN** system SHALL clip second moment appropriately to prevent overflow
