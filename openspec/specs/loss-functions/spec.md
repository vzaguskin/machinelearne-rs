## EXISTING Requirements

### Requirement: MSE Loss

MSE (Mean Squared Error) loss SHALL compute the average squared difference between predictions and targets.

#### Scenario: MSE loss computation

- **WHEN** predictions are `[3.0, 5.0]` and targets are `[1.0, 2.0]`
- **THEN** loss equals `((3-1)² + (5-2)²) / 2 = 6.5`

#### Scenario: MSE gradient computation

- **WHEN** computing gradient w.r.t. predictions
- **THEN** gradient equals `(prediction - target) / n`

### Requirement: MAE Loss

MAE (Mean Absolute Error) loss SHALL compute the average absolute difference between predictions and targets.

#### Scenario: MAE loss computation

- **WHEN** predictions are `[3.0, -1.0]` and targets are `[1.0, 2.0]`
- **THEN** loss equals `(|3-1| + |-1-2|) / 2 = 2.5`

#### Scenario: MAE gradient computation

- **WHEN** computing gradient w.r.t. predictions
- **THEN** gradient equals `sign(prediction - target) / n`

### Requirement: BCE With Logits Loss

BCEWithLogitsLoss SHALL compute binary cross-entropy with numerical stability using the log-sum-exp trick.

#### Scenario: BCE loss is numerically stable

- **WHEN** logits are very large (`100.0`) or very small (`-100.0`)
- **THEN** loss value is finite (no overflow/underflow)
- **AND** gradients are finite

#### Scenario: BCE gradient computation

- **WHEN** computing gradient w.r.t. logits
- **THEN** gradient equals `(sigmoid(logits) - targets) / n`

### Requirement: Loss Trait Interface

All loss functions SHALL implement the `Loss<B>` trait with `loss()` and `grad_wrt_prediction()` methods.

#### Scenario: Loss returns scalar

- **WHEN** `loss()` is called
- **THEN** return type is `Scalar<B>`

#### Scenario: Gradient returns prediction type

- **WHEN** `grad_wrt_prediction()` is called
- **THEN** return type matches the prediction type
