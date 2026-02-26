## ADDED Requirements

### Requirement: Performance parity with CPU
The WGPU backend SHALL outperform the CPU backend for large datasets.

#### Scenario: Large dataset speedup
- **WHEN** training on datasets with >10K samples
- **THEN** WGPU backend completes training faster than CPU backend

#### Scenario: Performance scaling
- **WHEN** dataset size increases
- **THEN** WGPU backend shows increasing speedup relative to CPU

### Requirement: Numerical accuracy
The WGPU backend SHALL maintain numerical accuracy comparable to CPU backend.

#### Scenario: Prediction accuracy
- **WHEN** comparing predictions from WGPU-trained model to CPU-trained model
- **THEN** R² difference is less than 0.01

#### Scenario: Gradient accuracy
- **WHEN** comparing gradients computed on WGPU vs CPU
- **THEN** relative difference is less than 1e-5

### Requirement: Transparent API
The WGPU backend SHALL provide the same synchronous API as other backends.

#### Scenario: Drop-in replacement
- **WHEN** user switches from CpuBackend to WgpuBackend
- **THEN** existing code compiles and runs without modification
