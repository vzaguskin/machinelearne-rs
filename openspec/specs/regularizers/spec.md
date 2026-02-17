## EXISTING Requirements

### Requirement: L2 Regularization

L2 (Ridge) regularizer SHALL compute penalty as `λ * ||w||²` with gradient `2λw`.

#### Scenario: L2 penalty computation

- **WHEN** weights are `[3.0, 4.0]` and lambda is `0.5`
- **THEN** penalty equals `0.5 * (3² + 4²) = 12.5`

#### Scenario: L2 gradient computation

- **WHEN** weights are `[3.0, 4.0]` and lambda is `0.5`
- **THEN** weight gradient equals `2 * 0.5 * [3.0, 4.0] = [3.0, 4.0]`

### Requirement: L1 Regularization

L1 (Lasso) regularizer SHALL compute penalty as `λ * ||w||₁` with gradient `λ * sign(w)`.

#### Scenario: L1 penalty computation

- **WHEN** weights are `[3.0, 4.0]` and lambda is `0.5`
- **THEN** penalty equals `0.5 * (|3| + |4|) = 3.5`

#### Scenario: L1 gradient computation

- **WHEN** weights are `[3.0, -4.0]` and lambda is `0.5`
- **THEN** weight gradient equals `0.5 * [1.0, -1.0] = [0.5, -0.5]`

#### Scenario: L1 gradient at zero

- **WHEN** a weight is `0.0`
- **THEN** its gradient is `0.0` (subgradient at zero)

### Requirement: Bias Not Regularized

Regularizers SHALL NOT apply penalty or gradient to bias term.

#### Scenario: Bias gradient is always zero

- **WHEN** any regularizer is applied
- **THEN** the bias gradient component is `0.0`

### Requirement: No Regularizer

NoRegularizer SHALL return zero penalty and zero gradients.

#### Scenario: No regularization

- **WHEN** NoRegularizer is used
- **THEN** penalty is `0.0`
- **AND** all gradients are zero

### Requirement: Regularizer Trait Interface

All regularizers SHALL implement `Regularizer<B, M>` trait returning `(Scalar<B>, M::Gradients)`.

#### Scenario: Regularizer returns penalty and gradients

- **WHEN** `regularizer_penalty_grad()` is called
- **THEN** first element is scalar penalty value
- **AND** second element is gradient structure matching model's gradient type
