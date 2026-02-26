## ADDED Requirements

### Requirement: WGPU backend in backend comparison benchmark
The backend comparison benchmark SHALL include WGPU backend performance measurements alongside CPU and ndarray backends.

#### Scenario: WGPU benchmark runs with feature flag
- **WHEN** user runs `cargo run --bin backend_comparison --features wgpu`
- **THEN** the benchmark includes WGPU backend timing and metrics in the output

#### Scenario: WGPU benchmark produces comparable metrics
- **WHEN** WGPU backend benchmark completes
- **THEN** output includes MSE, MAE, R², and training time for comparison with other backends

### Requirement: Benchmark output includes WGPU comparison
The benchmark output SHALL display WGPU results in the comparison table format.

#### Scenario: Comparison table shows WGPU row
- **WHEN** benchmark output is printed
- **THEN** a row for WgpuBackend is shown with timing and accuracy metrics

#### Scenario: Speedup calculation against CPU
- **WHEN** both CPU and WGPU benchmarks complete
- **THEN** the output includes relative speedup (CPU time / WGPU time)
