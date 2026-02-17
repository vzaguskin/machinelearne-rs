## Context

BLAS is the industry-standard API for basic linear algebra operations. Multiple optimized implementations exist:
- **OpenBLAS**: Open-source, cross-platform, good performance
- **Intel MKL**: Best performance on Intel CPUs, proprietary
- **Apple Accelerate**: Native on macOS, excellent on Apple Silicon
- **Netlib BLAS**: Reference implementation, slow but always available

The `blas-src` crate provides a unified interface to these implementations via feature flags.

## Goals / Non-Goals

**Goals:**
- Implement `BlasBackend` satisfying the `Backend` trait
- Use BLAS for matrix operations (GEMM, GEMV, etc.)
- Support multiple BLAS implementations via feature flags
- Delegate non-BLAS operations to ndarray

**Non-Goals:**
- Custom BLAS kernels (use existing libraries)
- GPU BLAS (that's cuBLAS, separate backend)
- Automatic BLAS implementation selection

## Decisions

### Decision 1: Use ndarray for tensor storage

Store tensors as ndarray::Array1/Array2 and use ndarray-blas for BLAS operations.

**Rationale:** ndarray already integrates with blas-src, avoiding redundant wrapper code. Element-wise ops use ndarray's optimized implementations.

### Decision 2: Feature flags for BLAS implementations

Provide separate feature flags for each BLAS implementation.

```toml
[features]
blas = ["dep:blas-src", "dep:ndarray"]
blas-openblas = ["blas", "blas-src/openblas"]
blas-netlib = ["blas", "blas-src/netlib"]
blas-accelerate = ["blas", "blas-src/accelerate"]
```

**Rationale:** Users choose based on their platform and licensing needs.

### Decision 3: Fall back to ndarray matmul if BLAS unavailable

If no BLAS feature is selected, use ndarray's built-in matmul.

**Rationale:** Ensures the backend always works, even if configuration is incomplete.

### Decision 4: Require external BLAS installation

Document that users must install a BLAS library on their system.

**Rationale:** blas-src links to system libraries rather than bundling them.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Missing system BLAS library | Clear error message at compile/link time |
| Different BLAS implementations give different results | Document acceptable numerical differences |
| Windows BLAS installation complexity | Recommend WSL or vcpkg |

## Open Questions

- Should we vendor OpenBLAS for easier installation? (Increases binary size)
- Detect and use the best available BLAS automatically? (Complexity vs convenience)
