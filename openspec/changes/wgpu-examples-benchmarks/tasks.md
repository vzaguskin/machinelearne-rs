## 0. WGPU Backend Fixes (Prerequisite)

- [x] 0.1 Fix global device singleton pattern for buffer compatibility
- [x] 0.2 Update `to_vec()` to use global device
- [x] 0.3 Verify tensor operations work without buffer conflicts

## 1. Tensor Operations Example

- [x] 1.1 Create `lib/examples/train_linear_wgpu.rs` with feature-gated WGPU imports
- [x] 1.2 Demonstrate 1D tensor creation and operations
- [x] 1.3 Demonstrate 2D tensor operations and matrix-vector multiplication
- [x] 1.4 Demonstrate reductions (sum, mean)
- [x] 1.5 Add fallback message when wgpu feature is not enabled

## 2. California Housing Example (Deferred)

- [ ] 2.1 Create `lib/examples/train_california_wgpu.rs` example file
- [ ] 2.2 Load California Housing dataset from benchmarks directory
- [ ] 2.3 Implement feature standardization using WGPU tensor operations
- [ ] 2.4 Train linear regression model on GPU with WgpuBackend
- [ ] 2.5 Report training time, MSE, MAE, and R² metrics
- [ ] 2.6 Display learned weights and bias

## 3. Backend Comparison Benchmark (Deferred)

- [ ] 3.1 Add `benchmark_wgpu_backend()` function to `backend_comparison.rs`
- [ ] 3.2 Implement WGPU tensor conversion helper (vec to WgpuTensor2D)
- [ ] 3.3 Add conditional compilation for WGPU benchmark with `#[cfg(feature = "wgpu")]`
- [ ] 3.4 Update benchmark main loop to include WGPU in feature count iterations
- [ ] 3.5 Add WGPU row to comparison table output with speedup calculation

## 4. Documentation and Testing

- [x] 4.1 Verify all examples compile with `cargo build --examples --features wgpu`
- [x] 4.2 Run examples manually to validate GPU execution
- [ ] 4.3 Update CHANGELOG.md with new examples and benchmarks
- [ ] 4.4 Run `cargo fmt` and `cargo clippy` on all changes
