## Why

The WGPU backend is ~200x slower than the CPU backend, making GPU acceleration effectively useless for ML training. Every tensor operation blocks on GPU completion, creating massive overhead that negates any parallelism benefits. For ML training to benefit from GPU acceleration, the backend must minimize CPU-GPU synchronization and maximize kernel batching.

## What Changes

### Phase 1: Operation Batching & Deferred Execution
- Add command buffer accumulation to batch multiple operations before GPU submission
- Implement lazy evaluation - operations queue commands without immediate execution
- Add explicit `flush()` or automatic flush at synchronization points

### Phase 2: Kernel Fusion
- Create fused kernels for common operation sequences (matvec + bias, forward pass, backward pass)
- Combine element-wise operations into single kernel launches where possible

### Phase 3: Memory Optimization
- Implement buffer pooling to reuse GPU memory across operations
- Keep model parameters on GPU throughout training epochs
- Minimize CPU-GPU data transfers

### Phase 4: Async API (Optional/Future)
- Consider async variants of Backend trait methods for non-blocking execution
- Allow overlapping computation and data transfer

## Capabilities

### New Capabilities

- `wgpu-command-batching`: Capability to batch multiple GPU operations into single command buffer submissions, reducing kernel launch overhead
- `wgpu-buffer-pooling`: Capability to reuse GPU buffers across operations, reducing allocation overhead
- `wgpu-kernel-fusion`: Capability to combine multiple operations into single kernel launches for common patterns

### Modified Capabilities

- `wgpu-cpu-benchmark`: Update to reflect improved GPU performance and new optimal use cases

## Impact

**Affected Code:**
- `lib/src/backend/wgpu_backend/mod.rs` - Core backend with batching support
- `lib/src/backend/wgpu_backend/tensor.rs` - Tensor operations with deferred execution
- `lib/src/backend/wgpu_backend/shaders.rs` - Fused kernel shaders
- `lib/src/backend/wgpu_backend/device.rs` - Buffer pool management

**API Changes:**
- Internal changes only - Backend trait remains unchanged
- Optional: Add `flush()` method to WgpuBackend for explicit synchronization control

**Dependencies:**
- No new external dependencies

**Performance Expectations:**
- Small datasets (1K): Still CPU-favored due to GPU overhead
- Medium datasets (10K): Near break-even or slight GPU advantage
- Large datasets (20K+): Clear GPU advantage (2-10x faster)

**Examples:**
- `wgpu_cpu_comparison.rs` should show improved GPU performance
- May enable new GPU-focused examples for large-scale training
