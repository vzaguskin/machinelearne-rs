## Why

We need to validate and demonstrate the GPU acceleration benefits of the WGPU backend compared to the CPU backend. Currently, we have separate examples for each backend but no direct head-to-head comparison showing the same operations on both backends with timing and accuracy metrics. This benchmark will help users understand when GPU acceleration provides value and validate that both backends produce equivalent results.

## What Changes

- Add a new benchmark example that runs identical operations on both WGPU and CPU backends
- Compare training time for the same model, dataset, and hyperparameters
- Compare accuracy metrics (MSE, MAE, R²) to verify numerical equivalence
- Test with different dataset sizes to show where GPU overhead vs. benefit kicks in
- Provide analysis and reflection on whether results match expectations

## Capabilities

### New Capabilities
- `wgpu-cpu-benchmark`: Direct comparison benchmark between WGPU and CPU backends with timing and accuracy metrics

### Modified Capabilities
- None (this is a new example, no existing specs are modified)

## Impact

- **New file**: `lib/examples/wgpu_cpu_comparison.rs` - the benchmark example
- **Documentation**: CHANGELOG.md update
- No API changes - purely additive example
