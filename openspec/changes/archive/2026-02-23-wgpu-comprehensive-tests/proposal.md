## Why

The WGPU backend lacks comprehensive tests to verify that GPU operations produce correct results compared to the CPU backend. This makes it difficult to:
1. Detect bugs in GPU compute shaders
2. Ensure numerical accuracy across operations
3. Have confidence when making changes to the WGPU backend
4. Prepare for performance optimization work (issue #83)

## What Changes

- Add comprehensive test module for WGPU backend operations
- Test all Backend trait methods against CPU backend results
- Test with different tensor sizes (small, medium, large)
- Test edge cases (empty tensors, single elements, power-of-2 sizes)
- Add tolerance-based comparisons for floating-point GPU operations

## Capabilities

### New Capabilities

- `wgpu-backend-tests`: Comprehensive test suite for WGPU backend verifying all tensor operations match CPU backend results within acceptable tolerances

### Modified Capabilities

(None - this is purely additive testing)

## Impact

- **New files**: `lib/src/backend/wgpu_backend/tests.rs` - comprehensive test module
- **Modified files**: `lib/src/backend/wgpu_backend/mod.rs` - include test module
- **Dependencies**: No new dependencies required
- **CI**: Tests will run automatically with `cargo test --features wgpu`
