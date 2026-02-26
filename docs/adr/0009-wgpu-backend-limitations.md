# ADR-0009: WGPU Backend Limitations and Async Analysis

## Status

**Accepted** - 2026-02-26

## Context

Issue #62 tracks WGPU backend implementation. After significant optimization work (command accumulation, buffer pooling, fused kernels), the WGPU backend remains **~2000-4000x slower than CPU** for training workloads.

### Benchmark Results (Post-Optimization)

| Dataset Size | CPU | WGPU | Slowdown |
|-------------|-----|------|----------|
| Small (1K) | 57ms | 7,430ms | ~130x |
| Medium (10K) | 613ms | 36,517ms | ~60x |
| Large (20K) | 1,359ms | 169,422ms | ~125x |

### Root Cause Analysis

The fundamental bottleneck is **GPU-CPU synchronization**, not the API design:

```
Training Loop Per Batch:
1. Forward pass    → GPU ops (async, accumulated)
2. Loss compute    → sum_all_1d() → GPU-CPU SYNC!
3. Backward pass   → GPU ops (async, accumulated)
4. Param update    → GPU ops (async, accumulated)
5. Next batch...   → Repeat with sync per batch
```

Even with command accumulation (batching 500 ops before GPU submit), we MUST sync when computing loss because the CPU needs the scalar value for:
- Loss metrics logging
- Early stopping decisions
- Learning rate scheduling
- Gradient clipping decisions

### Why MLP Training is "OK"

MLP has more operations between sync points:
- Multiple layers with activations
- More GPU work per sync
- Better amortization of sync overhead

### Why Linear/Inference is Poor

Linear models have minimal operations:
- Single matrix-vector multiply
- Immediate loss computation
- Sync overhead dominates

## Decision

### 1. Do NOT Pursue Async Backend Trait

After thorough analysis, an asynchronous Backend trait would NOT solve the core performance problem.

**Why async doesn't help:**

| Issue | Async Solution? | Reality |
|-------|-----------------|---------|
| Per-batch loss sync | No | CPU needs scalar value for metrics |
| Training monitoring | No | Requires sync by design |
| Early stopping | No | Needs loss value to decide |
| LR scheduling | No | Needs metric to adjust rate |

**The sync point is inherent to the algorithm, not the API.** Even with async:
- Loss computation still requires waiting for GPU result
- Training loops need scalar values for metrics, early stopping, LR scheduling
- The fundamental pattern (forward → loss sync → backward → update) remains

### 2. Document WGPU as Inference-Only for Now

The WGPU backend is recommended for:
- **Batch inference on large datasets**: Single forward pass, minimal sync points
- **Pre-trained model prediction**: No training loop, no per-batch loss sync

Not recommended for:
- **Training loops**: Per-batch loss sync causes 100-200x slowdown
- **Small batch operations**: CPU is inherently faster

### 3. Future Direction: GPU-Native Training

For GPU training to be viable, we need:

1. **GPU-Native Optimizers**: Implement SGD/Adam as GPU kernels that operate entirely on GPU memory
2. **Fused Loss Computation**: Keep loss on GPU, only sync for periodic logging
3. **Different Framework**: CUDA/cuBLAS designed for ML workloads with native tensor reductions

## Consequences

### Positive

- Clear guidance for users on WGPU backend use cases
- No breaking changes to Backend trait
- Development effort focused on proven improvements
- Documentation accurately reflects performance characteristics

### Negative

- WGPU backend not viable for training workloads
- GPU training requires CUDA backend (future work)
- Users must switch backends for training vs inference

### Neutral

- ADR-0008 backend prioritization remains valid
- WGPU backend remains functionally correct
- Command accumulation and buffer pooling still useful for inference

## Alternatives Considered

### Alternative 1: Async Backend Trait

```rust
// Current (synchronous)
fn add_1d(a: &Tensor1D, b: &Tensor1D) -> Tensor1D;

// Async variant
async fn add_1d(a: &Tensor1D, b: &Tensor1D) -> Tensor1D;
```

**Rejected**: Does not solve the fundamental problem. Loss computation requires waiting for GPU result regardless of API. Breaking change with minimal benefit.

### Alternative 2: Internal Async with Lazy Evaluation

Keep synchronous API, but internally queue entire training runs.

**Rejected**: Breaks important training features:
- Early stopping
- Learning rate scheduling based on metrics
- Logging/metrics per epoch
- Any form of training monitoring

### Alternative 3: GPU-Native Training Loop

Move entire training loop to GPU with single sync at end.

**Rejected for now**: Requires significant architectural changes:
- GPU-specific training code
- No visibility during training
- Can't implement callbacks, early stopping, LR scheduling
- Major API redesign

**May revisit** for specialized use cases in the future.

## References

- Issue #62 (WGPU Backend Implementation)
- `openspec/changes/archive/2026-02-26-optimize-wgpu-backend-performance/` - Buffer pooling, command batching
- `openspec/changes/archive/2026-02-26-optimize-wgpu-performance/` - Async operations analysis
- `lib/src/backend/wgpu_backend/` - Implementation with optimizations
- `lib/src/trainer/mod.rs` - Training loop (per-epoch loss optimization)
