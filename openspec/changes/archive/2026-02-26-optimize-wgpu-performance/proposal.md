## Why

The WGPU backend is currently **~200x slower** than the CPU backend, defeating the purpose of GPU acceleration. This makes the WGPU backend unusable for real workloads despite having comprehensive test coverage and a well-structured implementation with buffer pooling and command batching.

Benchmark results show:
- Small (1K samples): 57ms CPU vs 10,870ms WGPU (190x slower)
- Medium (10K samples): 613ms CPU vs 123,935ms WGPU (202x slower)
- Large (20K samples): 1,359ms CPU vs 260,969ms WGPU (192x slower)

The root cause is **excessive CPU↔GPU synchronization**: each tensor operation blocks until GPU completion, eliminating any parallelism benefits.

## What Changes

### Asynchronous Operation Execution
- Replace blocking `pollster::block_on()` calls with deferred execution
- Queue operations without forcing synchronization
- Only synchronize when user explicitly requests data (`to_vec()`) or at controlled flush points

### Batched Command Submission
- Accumulate multiple operations before submitting to GPU
- Implement compute pass bundling for related operations
- Reduce kernel launch overhead by combining operations where possible

### Optimized Data Transfer
- Implement staged readback with async buffer mapping
- Support batched readback of multiple tensors in single sync
- Reuse staging buffers instead of creating new ones per readback

### Improved Buffer Management
- Suballocation within large buffers to reduce allocation overhead
- Persistent storage for frequently accessed tensors
- Better buffer pool sizing based on actual workload patterns

## Capabilities

### New Capabilities
- `wgpu-async-operations`: Non-blocking GPU operation execution with deferred synchronization
- `wgpu-batched-transfers`: Batched CPU↔GPU data transfers to minimize sync overhead

### Modified Capabilities
- `wgpu-backend`: Performance requirements added - must be faster than CPU for large datasets

## Impact

**Affected Files:**
- `lib/src/backend/wgpu_backend/tensor.rs` - Tensor operations (add async patterns)
- `lib/src/backend/wgpu_backend/device.rs` - Device/queue handling (async submission)
- `lib/src/backend/wgpu_backend/accumulator.rs` - Command batching improvements
- `lib/src/backend/wgpu_backend/buffer_pool.rs` - Suballocation support

**API Changes:**
- No breaking changes to public API
- Internal refactoring to support async patterns

**Dependencies:**
- No new external dependencies
- Leverages existing wgpu async capabilities

**Acceptance Criteria:**
- WGPU backend faster than CPU for datasets >10K samples
- Performance scales with dataset size (larger = more benefit)
- Numerical accuracy maintained (R² difference < 0.01)
