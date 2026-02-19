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

- [ ] 3.1 Create fused forward pass shader (matvec + bias) in `shaders.rs`
- [ ] 3.2 Create fused backward pass shader (gradient computation) in `shaders.rs`
- [ ] 3.3 Add `forward_fused()` method to tensor operations
- [ ] 3.4 Add `backward_fused()` method to tensor operations
- [ ] 3.5 Implement fusion detection logic (use fused kernel when dimensions match)
- [ ] 3.6 Add fallback to individual ops when fusion not applicable
- [ ] 3.7 Add unit tests for fused kernels
- [ ] 3.8 Test: fused forward produces same results as separate ops
- [ ] 3.9 Test: fused backward produces same gradients as separate ops

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
Benchmark results (release mode, 2026-02-19):
- CPU: Small 2ms, Medium 7ms, Large 54ms
- WGPU: Small 6590ms, Medium 34515ms, Large 159970ms
- WGPU is ~3000-5000x slower than CPU

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
2. Kernel fusion (combine multiple ops into single shader)
3. Higher flush threshold or batch across epochs
4. Consider async Backend trait variant or internal async runtime

### Next Steps for Performance
1. Implement kernel fusion for forward/backward pass
2. Cache bind groups to reduce creation overhead
3. Increase flush threshold or remove intermediate flushes
4. Consider GPU-native optimizer step to reduce operations
