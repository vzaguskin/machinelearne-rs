## MODIFIED Requirements

### Requirement: Benchmark Performance Comparison

The benchmark SHALL compare WGPU and CPU backend performance with accurate expectations.

#### Scenario: Large dataset shows GPU advantage

- **WHEN** training on datasets with 20K+ samples
- **THEN** WGPU backend shows measurable speedup over CPU
- **AND** speedup is at least 2x for large datasets

#### Scenario: Small dataset shows CPU advantage

- **WHEN** training on datasets with 1K or fewer samples
- **THEN** CPU backend is faster due to GPU overhead
- **AND** benchmark reports this as expected behavior

#### Scenario: Medium dataset shows competitive performance

- **WHEN** training on datasets with 5K-15K samples
- **THEN** WGPU and CPU performance are within 2x of each other
- **AND** neither backend is drastically slower

## ADDED Requirements

### Requirement: Performance Regression Detection

The benchmark SHALL detect performance regressions in the WGPU backend.

#### Scenario: Performance threshold check

- **WHEN** benchmark runs in CI
- **THEN** WGPU performance is compared against baseline
- **AND** significant regressions (>50% slower) fail the check
