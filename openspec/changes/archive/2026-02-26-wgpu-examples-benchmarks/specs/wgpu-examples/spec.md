## ADDED Requirements

### Requirement: Synthetic data example with WGPU backend
The library SHALL provide a `train_linear_wgpu.rs` example that demonstrates linear regression training using the WGPU backend on synthetic data.

#### Scenario: Example runs successfully with wgpu feature
- **WHEN** user runs `cargo run --example train_linear_wgpu --features wgpu`
- **THEN** the example trains a model on GPU and prints predictions and learned parameters

#### Scenario: Example shows helpful message without wgpu feature
- **WHEN** user runs `cargo run --example train_linear_wgpu` without wgpu feature
- **THEN** the example prints instructions to enable the wgpu feature

### Requirement: Real dataset example with WGPU backend
The library SHALL provide a `train_california_wgpu.rs` example that demonstrates linear regression training using the WGPU backend on the California Housing dataset.

#### Scenario: California Housing example trains on GPU
- **WHEN** user runs `cargo run --example train_california_wgpu --features wgpu`
- **THEN** the example loads California Housing data, trains on GPU, and reports metrics (MSE, MAE, R²)

#### Scenario: Example reports training time and accuracy
- **WHEN** the California Housing example completes training
- **THEN** the example prints training time, test metrics, and learned model parameters
