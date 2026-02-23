## Context

The WGPU backend has a well-structured implementation with buffer pooling, command accumulation, and bind group caching. However, it suffers from severe performance issues (~200x slower than CPU) due to synchronous operation patterns.

**Current State:**
- Operations block via `pollster::block_on()` waiting for GPU completion
- Each `to_vec()` creates a new staging buffer and forces full sync
- Command accumulator batches operations but flushes too frequently
- Buffer pool works well but doesn't support suballocation

**Key Constraint:** Must maintain the existing synchronous public API - users shouldn't need to handle async/await for basic tensor operations.

## Goals / Non-Goals

**Goals:**
- Achieve GPU speedup over CPU for datasets >10K samples
- Minimize CPU↔GPU synchronization points
- Batch operations effectively without user intervention
- Maintain numerical accuracy (R² difference < 0.01)

**Non-Goals:**
- Async public API (keep synchronous interface)
- Multi-GPU support
- Custom shader optimization (use existing shaders)
- Memory-mapped buffers (too platform-specific)

## Decisions

### 1. Lazy Execution Model

**Decision:** Operations queue commands without executing until data is needed.

```rust
// Current: Each operation blocks
pub fn add_1d(&self, a: &Tensor1D, b: &Tensor1D) -> Tensor1D {
    pollster::block_on(async { ... });  // BLOCKS HERE
}

// New: Queue command, return immediately
pub fn add_1d(&self, a: &Tensor1D, b: &Tensor1D) -> Tensor1D {
    self.queue_add_command(a, b);  // Just queues
    Tensor1D { buffer: output_buffer }
}
```

**Rationale:** Decouples operation submission from execution. GPU can process commands while CPU continues queuing more work.

**Alternatives considered:**
- Async API: Would require users to handle futures, breaking existing code
- Explicit flush: Forces users to understand GPU execution model

### 2. Single Flush Point

**Decision:** Only flush when user reads data back (`to_vec()`, `sum()`, etc.).

```rust
impl Tensor1D {
    pub fn to_vec(&self) -> Vec<f32> {
        self.device.flush_if_needed();  // Only sync here
        self.device.read_buffer(&self.buffer)
    }
}
```

**Rationale:** Minimizes sync points to exactly when needed. For training loops, this means one sync per epoch (when computing loss) instead of per operation.

### 3. Staging Buffer Pool

**Decision:** Pool staging buffers for readback, reuse across calls.

```rust
struct StagingBufferPool {
    buffers: HashMap<usize, Vec<Buffer>>,  // Size -> available buffers
    max_size: usize,
}
```

**Rationale:** Eliminates staging buffer allocation overhead on every `to_vec()` call.

### 4. Compute Pass Bundling

**Decision:** Bundle multiple operations into single compute pass where possible.

```rust
// Instead of N dispatches with N pipeline changes:
for op in operations {
    encoder.run_compute_pass(|pass| {
        pass.set_pipeline(&op.pipeline);
        pass.dispatch(...);
    });
}

// Bundle into fewer passes:
encoder.run_compute_pass(|pass| {
    for op in compatible_operations {
        pass.set_pipeline(&op.pipeline);
        pass.dispatch(...);
    }
});
```

**Rationale:** Reduces GPU driver overhead from frequent pipeline switches.

### 5. Increased Flush Threshold

**Decision:** Increase default flush threshold from 50 to 500 operations.

**Rationale:** Current threshold is too low, causing premature syncs. Training loops typically have 10-20 operations per batch, so 50 operations flushes every 2-5 batches.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Memory exhaustion from queued operations | Cap queued command memory; flush when exceeded |
| Increased latency for single operations | Acceptable - GPU overhead dominates small ops anyway |
| Breaking existing timing assumptions | No API changes; behavior is internal optimization |
| Debugging harder with lazy execution | Add debug mode that flushes after each operation |

## Migration Plan

1. Implement staging buffer pool (no behavior change)
2. Refactor operations to queue instead of execute
3. Update flush logic to be lazy
4. Increase flush threshold
5. Benchmark and tune thresholds

No rollback needed - each step is independently deployable and revertible.
