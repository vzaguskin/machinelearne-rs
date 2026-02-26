## Context

The WGPU backend was implemented in a previous change with full GPU compute shader support for tensor operations. However, the library lacks:
1. Examples showing how to use WGPU backend for ML training
2. Benchmarks comparing GPU performance against CPU backends

This change adds examples and benchmarks to demonstrate and validate the WGPU backend on real ML workloads.

## Goals / Non-Goals

**Goals:**
- Provide working examples of WGPU backend usage with both synthetic and real datasets
- Add WGPU to the existing backend comparison benchmark
- Enable users to validate GPU acceleration on their hardware

**Non-Goals:**
- Optimizing WGPU shader performance (separate concern)
- Adding new WGPU features or operations
- Creating automated CI benchmarks (requires GPU runners)

## Decisions

### Example Structure
Follow the existing example pattern (`train_linear.rs`, `train_linear_ndarray.rs`):
- Use feature-gated `#[cfg(feature = "wgpu")]` for WGPU-specific code
- Provide fallback message when feature is not enabled
- Keep examples focused on single concepts

**Alternatives considered:**
- Single unified example with backend selection flag - rejected to keep examples simple and explicit

### Benchmark Integration
Extend existing `backend_comparison.rs` with WGPU backend benchmark function:
- Reuse existing California Housing dataset loader
- Use same standardization and training parameters for fair comparison
- Report MSE, MAE, R² metrics alongside timing

**Alternatives considered:**
- Separate WGPU benchmark file - rejected to enable direct comparison in single output

### Dataset Choice
Use California Housing dataset for real-world examples:
- Already available in benchmarks module
- 20,640 samples with 8 features (sufficient for GPU to show benefit)
- Regression task works well with linear models

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| GPU may be slower than CPU for small datasets | Use batch sizes and feature counts where GPU overhead is justified |
| Different GPU drivers may produce different results | Use tolerance-based assertions in tests |
| WGPU may not initialize on headless systems | Document GPU requirements in example comments |
