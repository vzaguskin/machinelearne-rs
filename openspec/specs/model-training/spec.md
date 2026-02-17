## EXISTING Requirements

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

### Requirement: Linear Model Forward Pass

The linear model SHALL compute predictions as `y = X @ w + b`.

#### Scenario: Forward pass with 2D input

- **WHEN** input `X` has shape `(n_samples, n_features)` and weights have shape `(n_features,)`
- **THEN** output has shape `(n_samples,)`
- **AND** output values are `X @ w + b`

### Requirement: Linear Model Backward Pass

The linear model SHALL compute gradients for weights and bias.

#### Scenario: Backward pass gradient shapes

- **WHEN** backward is called with gradient of shape `(n_samples,)` and input of shape `(n_samples, n_features)`
- **THEN** weight gradient has shape `(n_features,)`
- **AND** bias gradient is a scalar

### Requirement: Model Serialization

Fitted models SHALL be serializable to and from files.

#### Scenario: Save and load round-trip

- **WHEN** a fitted model is saved to file and loaded back
- **THEN** the loaded model produces identical predictions

### Requirement: Inference Model Separation

A fitted model SHALL contain only inference parameters with no optimizer state or loss functions.

#### Scenario: Fitted model is lightweight

- **WHEN** a model is converted to fitted state via `into_fitted()`
- **THEN** the resulting model contains only weights and bias
- **AND** no learning rate, loss function, or training metadata is retained
