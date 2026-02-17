# l1-regularization Specification

## Purpose
TBD - created by archiving change l1-regularizer. Update Purpose after archive.
## Requirements
### Requirement: L1 Regularization Penalty

The L1 regularizer SHALL compute the L1 norm penalty as the sum of absolute weight values multiplied by the regularization strength lambda.

#### Scenario: Compute L1 penalty for positive weights

- **WHEN** model weights are `[3.0, 4.0]` and lambda is `0.5`
- **THEN** penalty equals `0.5 * (|3.0| + |4.0|) = 3.5`

#### Scenario: Compute L1 penalty for mixed sign weights

- **WHEN** model weights are `[-2.0, 3.0]` and lambda is `1.0`
- **THEN** penalty equals `1.0 * (|-2.0| + |3.0|) = 5.0`

### Requirement: L1 Regularization Gradient

The L1 regularizer SHALL compute the gradient as lambda times the sign of each weight (subgradient at zero SHALL be zero).

#### Scenario: Compute L1 gradient for non-zero weights

- **WHEN** model weights are `[3.0, -4.0]` and lambda is `0.5`
- **THEN** weight gradient equals `0.5 * [1.0, -1.0] = [0.5, -0.5]`
- **AND** bias gradient is `0.0`

#### Scenario: Compute L1 gradient with zero weights

- **WHEN** model weights are `[0.0, 2.0]` and lambda is `1.0`
- **THEN** weight gradient equals `1.0 * [0.0, 1.0] = [0.0, 1.0]`

### Requirement: Bias Not Regularized

The L1 regularizer MUST NOT apply penalty or gradient to the bias term, matching L2 behavior.

#### Scenario: Bias gradient is always zero

- **WHEN** L1 regularization is applied to any model
- **THEN** the bias gradient component is `0.0`

