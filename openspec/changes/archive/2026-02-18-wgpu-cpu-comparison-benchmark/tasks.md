## 1. Example Setup

- [x] 1.1 Create `lib/examples/wgpu_cpu_comparison.rs` with feature-gated imports
- [x] 1.2 Add fallback message when wgpu feature is not enabled

## 2. GPU Adapter Display

- [x] 2.1 Display available GPU adapters at startup
- [x] 2.2 Show which adapter is being used for WGPU backend

## 3. Dataset Preparation

- [x] 3.1 Load California Housing dataset
- [x] 3.2 Create helper to subset dataset by sample count
- [x] 3.3 Prepare standardization for both backends

## 4. CPU Backend Benchmark

- [x] 4.1 Implement CPU backend training function with timing
- [x] 4.2 Calculate MSE, MAE, R² metrics for CPU backend

## 5. WGPU Backend Benchmark

- [x] 5.1 Implement WGPU backend training function with timing
- [x] 5.2 Calculate MSE, MAE, R² metrics for WGPU backend

## 6. Comparison and Analysis

- [x] 6.1 Compare training times and calculate speedup ratio
- [x] 6.2 Compare accuracy metrics between backends
- [x] 6.3 Test with multiple dataset sizes (1K, 10K, 20K)
- [x] 6.4 Print analysis of when GPU is faster vs slower

## 7. Documentation and Testing

- [x] 7.1 Verify example compiles with `cargo build --example wgpu_cpu_comparison --features wgpu`
- [x] 7.2 Run example manually to validate output
- [x] 7.3 Update CHANGELOG.md
- [x] 7.4 Run `cargo fmt` and `cargo clippy`
