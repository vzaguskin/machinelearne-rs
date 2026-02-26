## Why

The WGPU backend was recently completed with full GPU compute shader support, but there are no examples or benchmarks demonstrating its usage on real datasets. Users cannot see the WGPU backend in action or understand its performance characteristics compared to other backends. This makes it difficult to validate GPU acceleration benefits and provides no reference for how to use WGPU in practice.

## What Changes

- Add `train_linear_wgpu.rs` example demonstrating linear regression training on synthetic data using WGPU backend
- Add `train_california_wgpu.rs` example demonstrating training on the California Housing dataset with WGPU backend
- Add WGPU backend to `backend_comparison.rs` benchmark for performance comparison against CPU and ndarray backends
- Update documentation to reference WGPU examples

## Capabilities

### New Capabilities

- `wgpu-examples`: Example code demonstrating WGPU backend usage with both synthetic and real datasets (California Housing)
- `wgpu-benchmarks`: Performance benchmarks comparing WGPU backend against CPU and ndarray backends on real ML workloads

### Modified Capabilities

- None (additive only - examples and benchmarks do not change existing behavior)

## Impact

- `lib/examples/train_linear_wgpu.rs`: New example for synthetic data training with WGPU
- `lib/examples/train_california_wgpu.rs`: New example for California Housing dataset with WGPU
- `benchmarks/src/bin/backend_comparison.rs`: Add WGPU backend benchmark function
- `benchmarks/src/data/california_housing.rs`: May need minor updates for WGPU tensor conversion
- `CHANGELOG.md`: Document new examples and benchmarks
