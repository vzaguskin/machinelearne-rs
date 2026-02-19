## Context

The WGPU backend was designed for API compatibility with the synchronous Backend trait. Each operation calls `pollster::block_on()` to wait for GPU completion before returning. This design choice, while maintaining API simplicity, creates massive overhead:

- **Current pattern**: op → create encoder → submit → block until done → return
- **Overhead per operation**: Pipeline creation, command encoding, queue submission, GPU-CPU sync
- **Training impact**: ~200x slower than CPU because training requires hundreds of operations per epoch

The Backend trait is synchronous by design, so we need to optimize within that constraint while potentially adding optional async extensions.

## Goals / Non-Goals

**Goals:**
- Reduce GPU-CPU synchronization points by 90%+
- Batch multiple operations into single command buffer submissions
- Reuse GPU buffers across operations to eliminate allocation overhead
- Achieve GPU speedup on large datasets (20K+ samples)
- Maintain backward compatibility with existing Backend trait API

**Non-Goals:**
- Rewriting the Backend trait to be async (breaking change)
- Implementing automatic differentiation on GPU
- Supporting multi-GPU setups
- Optimizing for small datasets (<1K samples) where CPU is inherently faster

## Decisions

### Decision 1: Command Buffer Accumulation (Deferred Execution)

**Choice:** Implement a per-thread command accumulator that batches operations

**Rationale:**
- Minimizes changes to Backend trait (keeps synchronous API)
- Allows multiple operations to share single submission
- Automatic flush at synchronization points (to_vec, loss computation)

**Implementation:**
```rust
// In WgpuDevice, add command accumulator
struct CommandAccumulator {
    encoder: Option<CommandEncoder>,
    pending_buffers: Vec<Buffer>,
    operation_count: usize,
}

// Operations add to accumulator instead of immediate submit
fn add_1d(a: &Tensor1D, b: &Tensor1D) -> Tensor1D {
    device.accumulator.add_binary_op(a, b, Op::Add);
    // Returns immediately, actual GPU work deferred
}

// Explicit flush or auto-flush on read
fn to_vec(&self) -> Vec<f32> {
    device.accumulator.flush(); // Submit all pending ops
    // Now read back
}
```

**Alternatives considered:**
- **Full async API**: Would require Backend trait redesign. Rejected as breaking change.
- **Manual flush only**: Error-prone for users. Rejected.

### Decision 2: Buffer Pooling

**Choice:** Implement a buffer pool in WgpuDevice for reusable GPU memory

**Rationale:**
- Eliminates constant buffer creation/destruction
- Reduces memory fragmentation
- Keeps frequently-used buffers (model parameters) on GPU

**Implementation:**
```rust
struct BufferPool {
    available_1d: HashMap<usize, Vec<Buffer>>,
    available_2d: HashMap<(usize, usize), Vec<Buffer>>,
    in_use: Vec<Buffer>,
}

impl BufferPool {
    fn acquire_1d(&mut self, len: usize) -> Buffer {
        // Reuse existing buffer or create new
    }

    fn release(&mut self, buffer: Buffer) {
        // Return to pool for reuse
    }
}
```

**Alternatives considered:**
- **Global buffer pool**: Thread-safety complexity. Rejected.
- **No pooling**: Current approach, causes 200x slowdown. Rejected.

### Decision 3: Keep Parameters on GPU

**Choice:** During training, model parameters stay on GPU between epochs

**Rationale:**
- Currently: params are read back after each batch for optimizer step
- Optimization: do optimizer update on GPU, only sync when training complete

**Implementation:**
- Add GPU-based optimizer step kernel
- Parameter updates happen entirely on GPU
- Only sync to CPU when `into_fitted()` is called

**Alternatives considered:**
- **CPU optimizer with frequent syncs**: Current approach, main bottleneck. Rejected.
- **Hybrid approach**: More complex without significant benefit. Rejected.

### Decision 4: Fused Kernels for Common Patterns

**Choice:** Create fused shaders for training-critical operation sequences

**Rationale:**
- Forward pass: matvec + bias add → single kernel
- Backward pass: transpose matvec + gradient computation → single kernel
- Reduces kernel launch overhead by 50%+

**Implementation:**
```rust
// Fused forward pass shader
@compute @workgroup_size(64)
fn forward_fused(...) {
    // 1. Matrix-vector multiply
    // 2. Add bias
    // All in single kernel
}

// Usage
fn forward_fused(&self, x: &Tensor2D, weights: &Tensor1D, bias: &Scalar) -> Tensor1D {
    device.accumulator.add_fused_forward(x, weights, bias);
}
```

**Alternatives considered:**
- **No fusion**: Current approach, each op is separate. Rejected.
- **Full graph compilation**: Over-engineered for current needs. Rejected.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Increased memory usage from buffer pool | Add pool size limits and LRU eviction |
| Deferred execution makes debugging harder | Add debug mode with immediate execution |
| Fused kernels may not cover all cases | Fall back to individual ops gracefully |
| Thread-local accumulator complexity | Clear documentation and scoped flush |
| GPU optimizer numerical differences vs CPU | Validate against CPU optimizer results |

## Migration Plan

This is an internal optimization - no migration required for users.

**Deployment phases:**
1. **Phase 1**: Add buffer pooling (transparent, immediate benefit)
2. **Phase 2**: Add command accumulation with auto-flush
3. **Phase 3**: Add fused kernels for forward/backward passes
4. **Phase 4**: Add GPU optimizer step

**Validation:**
- Existing tests continue to pass
- Benchmark shows progressive improvement
- Numerical accuracy validated against CPU

## Open Questions

1. **Auto-flush threshold**: How many operations before auto-flush? (Suggestion: 10-50 ops)
2. **Buffer pool eviction policy**: LRU or size-based? (Suggestion: LRU with max pool size)
3. **Debug mode**: Environment variable or compile-time flag? (Suggestion: Compile-time feature)
