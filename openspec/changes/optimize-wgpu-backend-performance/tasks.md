## 1. Buffer Pooling Infrastructure

- [x] 1.1 Create `BufferPool` struct in `wgpu_backend/buffer_pool.rs`
- [x] 1.2 Implement `acquire_1d(len)` and `acquire_2d(rows, cols)` methods
- [x] 1.3 Implement `release(buffer)` method to return buffers to pool
- [x] 1.4 Add pool size limits with LRU eviction
- [x] 1.5 Integrate `BufferPool` into `WgpuDevice` as thread-local storage
- [ ] 1.6 Modify tensor creation to use buffer pool
- [x] 1.7 Add unit tests for buffer pool behavior
- [ ] 1.8 Test: buffers are reused for same-size allocations
- [ ] 1.9 Test: pool evicts buffers when size limit exceeded

## 2. Command Buffer Accumulation

- [x] 2.1 Create `CommandAccumulator` struct in `wgpu_backend/accumulator.rs`
- [x] 2.2 Implement operation queue for deferred execution
- [x] 2.3 Add `add_binary_op()`, `add_unary_op()`, `add_matvec()` methods
- [x] 2.4 Implement `flush()` method to submit accumulated commands
- [x] 2.5 Add auto-flush on `to_vec()` calls
- [x] 2.6 Add configurable flush threshold (default 50 operations)
- [x] 2.7 Integrate accumulator into `WgpuDevice`
- [x] 2.8 Modify tensor operations to use accumulator (queue commands instead of immediate submit)
- [x] 2.9 Add unit tests for command accumulation
- [x] 2.10 Test: multiple operations batch into single submission
- [ ] 2.11 Test: auto-flush triggers at threshold
- [x] 2.12 Test: results identical to immediate execution (verified by benchmark)

## 3. Kernel Fusion

- [x] 3.1 Create fused forward pass shader (matvec + bias) in `shaders.rs`
- [ ] 3.2 Create fused backward pass shader (gradient computation) in `shaders.rs`
- [x] 3.3 Add `matvec_bias()` fused method to tensor operations
- [x] 3.4 Add `sgd_step_inplace()` fused method to tensor operations
- [x] 3.5 Implement `matvec_bias` and `sgd_step` in Backend trait with default fallback
- [x] 3.6 Add fused `dot_add_scalar()` method to Tensor2D
- [x] 3.7 Integrate fused forward in LinearModel
- [x] 3.8 Test: fused forward produces same results as separate ops (verified by benchmark)

## 4. GPU Optimizer Step (Optional Enhancement)

- [x] 4.1 Create optimizer step shader (SGD: param = param - lr * grad)
- [x] 4.2 Add `sgd_step_inplace()` method to WgpuTensor1D
- [ ] 4.3 Integrate with trainer for GPU-native training (requires mutable params)
- [x] 4.4 Add `sgd_step` to Backend trait with default implementation

## 4. GPU Optimizer Step (Optional Enhancement)

- [ ] 4.1 Create optimizer step shader (SGD: param = param - lr * grad)
- [ ] 4.2 Add `optimizer_step_gpu()` method
- [ ] 4.3 Integrate with trainer for GPU-native training
- [ ] 4.4 Test: GPU optimizer produces same results as CPU optimizer

## 5. Benchmark Updates

- [ ] 5.1 Update `wgpu_cpu_comparison.rs` with new performance expectations
- [ ] 5.2 Add performance regression detection thresholds
- [ ] 5.3 Document optimal dataset sizes for GPU vs CPU
- [ ] 5.4 Run benchmark and verify GPU shows speedup on large datasets
- [ ] 5.5 Update CHANGELOG.md with performance improvements

## 6. Documentation

- [x] 6.1 Add doc comments to new `BufferPool` API
- [x] 6.2 Add doc comments to `CommandAccumulator` API
- [ ] 6.3 Document flush threshold configuration
- [x] 6.4 Add performance tuning guide to module docs

## 7. Verification

- [x] 7.1 Run all existing tests: `cargo test -p machinelearne-rs --features wgpu`
- [x] 7.2 Verify no performance regression on CPU backend
- [ ] 7.3 Run coverage: `cargo tarpaulin` (maintain 85%+ coverage)
- [x] 7.4 Format code: `cargo fmt`
- [x] 7.5 Run clippy: `cargo clippy` (no warnings)
- [ ] 7.6 Manual benchmark: verify GPU speedup on 20K+ dataset

## Notes

### Current Performance State
Benchmark results (release mode, 2026-02-19 after kernel fusion):
- CPU: Small 3ms, Medium 9ms, Large 59ms
- WGPU: Small 7430ms, Medium 36517ms, Large 169422ms
- WGPU is ~2500-4000x slower than CPU

### Kernel Fusion Results
After implementing fused matvec_bias kernel:
- Forward pass now uses single fused kernel (matvec + bias) instead of 2 operations
- Results are numerically identical to separate operations
- No significant performance improvement (~5% regression observed)
- Conclusion: Per-operation bind group creation overhead dominates, not kernel count

### Batching Implementation Results
After implementing command batching with lazy execution:
- Operations are queued to accumulator instead of immediate submission
- Single command encoder for all pending operations on flush
- Flush triggered by to_vec(), sum() calls
- ~5-10% improvement over previous implementation
- Still significantly slower than CPU due to per-operation bind group creation

### Root Cause Analysis
The bottleneck is now:
1. **Per-operation bind group creation**: Each operation still creates bind groups (can't be batched)
2. **Per-operation buffer creation**: Params buffers created for each op
3. **Many flushes per epoch**: ~52 flushes/epoch (516 batches × 5 ops / 50 threshold)
4. **GPU-CPU sync overhead**: Each flush has synchronization cost

### Architecture Challenge
The `Backend` trait is synchronous, but GPU performance requires:
1. Reducing bind group creation overhead (cache bind groups)
2. Kernel fusion helps but not enough (bind groups still created)
3. Higher flush threshold or batch across epochs
4. Consider async Backend trait variant or internal async runtime

### Next Steps for Performance
1. ~~Implement kernel fusion for forward/backward pass~~ (done - limited impact)
2. Cache bind groups to reduce creation overhead
3. Increase flush threshold or remove intermediate flushes
4. Consider GPU-native optimizer step to reduce operations
