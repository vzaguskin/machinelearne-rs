## EXISTING Requirements

### Requirement: SGD Update Rule

SGD (Stochastic Gradient Descent) SHALL update parameters using `params_new = params - lr * gradients`.

#### Scenario: SGD step with positive gradients

- **WHEN** params are `[2.0, 3.0]`, gradients are `[1.0, -1.0]`, and learning rate is `0.1`
- **THEN** new params are `[2.0 - 0.1*1.0, 3.0 - 0.1*(-1.0)] = [1.9, 3.1]`

#### Scenario: SGD step with zero learning rate

- **WHEN** learning rate is `0.0`
- **THEN** parameters remain unchanged

### Requirement: SGD Immutability

SGD step SHALL NOT mutate input parameters.

#### Scenario: Original params preserved

- **WHEN** `step()` is called with params and gradients
- **THEN** original params and gradients are unchanged
- **AND** new params are returned as a new value

### Requirement: SGD Learning Rate Access

SGD SHALL provide access to the current learning rate.

#### Scenario: Learning rate getter

- **WHEN** `learning_rate()` is called on SGD with lr=0.01
- **THEN** return value is `0.01`

### Requirement: SGD Clone

SGD SHALL be clonable with identical behavior.

#### Scenario: Cloned optimizer produces same updates

- **WHEN** SGD is cloned
- **THEN** both optimizers produce identical parameter updates
