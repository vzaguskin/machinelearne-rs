## Context

The WGPU backend implements the `Backend` trait using GPU compute shaders via the wgpu crate. Currently, there are some unit tests in individual modules (buffer_pool, uniform_pool, etc.) but no comprehensive tests that verify tensor operations produce correct results compared to the reference CPU backend.

GPU operations can introduce subtle numerical differences due to:
- Floating-point precision differences between CPU and GPU
- Parallel reduction ordering differences
- Shader implementation bugs

## Goals / Non-Goals

**Goals:**
- Test all Backend trait methods for WGPU backend
- Compare GPU results against CPU backend (reference implementation)
- Test various tensor sizes (small: <10, medium: 10-100, large: >100)
- Establish acceptable tolerance thresholds for floating-point comparisons
- Provide clear test failure messages showing expected vs actual values

**Non-Goals:**
- Performance benchmarking (covered by existing wgpu-cpu-benchmark spec)
- Testing error handling (OOM, device loss) - infrastructure not ready
- Multi-GPU testing - requires additional infrastructure

## Decisions

### 1. Test Organization
**Decision**: Create a single `tests.rs` module in `lib/src/backend/wgpu_backend/`

**Rationale**: Keeps tests co-located with implementation, follows existing pattern in other backend modules. Easier to maintain than separate test files.

### 2. Comparison Strategy
**Decision**: Use relative tolerance for comparisons with CPU backend

**Rationale**:
- Absolute tolerance doesn't scale with value magnitude
- Relative tolerance (`|a - b| / |b| < epsilon`) handles varying magnitudes
- Use `epsilon = 1e-4` for most operations, `1e-3` for reductions (more parallel variance)

### 3. Test Size Categories
**Decision**: Test with sizes: 1, 4, 16, 64, 256, 1024

**Rationale**:
- 1: Edge case (single element)
- 4: Small (fits in single workgroup)
- 16: Medium small
- 64: Medium (multiple workgroups likely)
- 256: Large enough to show differences
- 1024: Stress test

### 4. Test Structure
**Decision**: Use test matrix pattern with helper functions

```rust
fn test_sizes() -> Vec<usize> { vec![1, 4, 16, 64, 256, 1024] }

#[test]
fn test_add_1d() {
    for size in test_sizes() {
        let cpu_a = ...;
        let wgpu_a = ...;
        let cpu_result = CpuBackend::add_1d(&cpu_a, &cpu_b);
        let wgpu_result = WgpuBackend::add_1d(&wgpu_a, &wgpu_b);
        assert_tensors_close(&cpu_result, &wgpu_result, 1e-4);
    }
}
```

**Rationale**: Reduces boilerplate, ensures all operations tested at all sizes.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Tests may be slow with large sizes | Use only sizes up to 1024, run in parallel |
| GPU timing variance may cause flaky tests | Use generous tolerances, avoid timing-dependent tests |
| Different GPU backends (Vulkan/Metal/D3D12) may behave differently | Document that tests should pass on all platforms |
