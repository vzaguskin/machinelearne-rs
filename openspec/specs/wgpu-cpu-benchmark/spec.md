# wgpu-cpu-benchmark Specification

## Purpose
TBD - created by archiving change wgpu-cpu-comparison-benchmark. Update Purpose after archive.
## Requirements
### Requirement: WGPU vs CPU backend comparison benchmark
The library SHALL provide a `wgpu_cpu_comparison.rs` example that runs identical training operations on both WGPU and CPU backends and compares their performance and accuracy.

#### Scenario: Example runs successfully with wgpu feature
- **WHEN** user runs `cargo run --example wgpu_cpu_comparison --features wgpu`
- **THEN** the example trains the same model on both backends and prints comparison results

#### Scenario: Example shows helpful message without wgpu feature
- **WHEN** user runs `cargo run --example wgpu_cpu_comparison` without wgpu feature
- **THEN** the example prints instructions to enable the wgpu feature

### Requirement: Training time comparison
The benchmark SHALL measure and compare training time between WGPU and CPU backends for the same model, dataset, and hyperparameters.

#### Scenario: Training time is reported for both backends
- **WHEN** the benchmark completes training on both backends
- **THEN** the example prints training time in milliseconds for both WGPU and CPU backends

#### Scenario: Speedup ratio is calculated
- **WHEN** both training runs complete
- **THEN** the example calculates and displays the speedup ratio (CPU time / GPU time)

### Requirement: Accuracy comparison
The benchmark SHALL compare accuracy metrics between both backends to verify numerical equivalence.

#### Scenario: MSE, MAE, and R² are reported for both backends
- **WHEN** evaluation completes on both backends
- **THEN** the example prints MSE, MAE, and R² for both WGPU and CPU backends

#### Scenario: Accuracy difference is within acceptable tolerance
- **WHEN** comparing accuracy metrics between backends
- **THEN** the difference in R² is less than 0.01 (both backends produce equivalent results)

### Requirement: Multiple dataset sizes tested
The benchmark SHALL test with different dataset sizes to show where GPU acceleration provides benefit.

#### Scenario: Multiple sizes are tested
- **WHEN** the benchmark runs
- **THEN** it tests with small (1,000), medium (10,000), and large (20,640 - full California Housing) sample sizes

#### Scenario: Results include analysis of GPU overhead vs benefit
- **WHEN** all tests complete
- **THEN** the example prints analysis of when GPU is faster vs slower than CPU

### Requirement: GPU adapter information displayed
The benchmark SHALL display GPU adapter information to verify which GPU is being used.

#### Scenario: GPU adapter info is shown
- **WHEN** the benchmark starts
- **THEN** the example displays the name, backend type, and device type of the GPU being used

