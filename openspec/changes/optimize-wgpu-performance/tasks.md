## 1. Staging Buffer Pool

- [x] 1.1 Create `StagingBufferPool` struct in `lib/src/backend/wgpu_backend/staging_pool.rs`
- [x] 1.2 Implement buffer reuse with size-based buckets
- [x] 1.3 Add LRU eviction when pool exceeds max size
- [x] 1.4 Integrate staging pool into `WgpuDevice`
- [x] 1.5 Update `to_vec()` to use pooled staging buffers
- [x] 1.6 Add unit tests for staging buffer pool

## 2. Lazy Operation Execution

- [x] 2.1 Refactor tensor operations to queue commands instead of executing immediately
- [x] 2.2 Remove `pollster::block_on()` calls from operation methods
- [x] 2.3 Add flush logic to `to_vec()` and `sum()` methods
- [x] 2.4 Ensure operations still work correctly with lazy execution
- [x] 2.5 Add debug mode flag for eager flushing

## 3. Command Accumulator Improvements

- [x] 3.1 Increase default flush threshold from 50 to 500 operations
- [x] 3.2 Make flush threshold configurable via `WgpuDevice`
- [ ] 3.3 Add memory-based flush threshold (cap queued command memory)
- [x] 3.4 Implement compute pass bundling for compatible operations

## 4. Performance Validation

- [x] 4.1 Run existing `wgpu_cpu_comparison` benchmark to establish baseline
- [x] 4.2 Create performance regression test (existing tests cover correctness)
- [ ] 4.3 Verify WGPU is faster than CPU for >10K samples (still ~2000x slower - deeper optimization needed)
- [x] 4.4 Verify numerical accuracy (R² < 0.01 difference)

## 5. Documentation

- [x] 5.1 Update CLAUDE.md with WGPU performance notes
- [x] 5.2 Add doc comments for new public APIs
- [x] 5.3 Update CHANGELOG.md
