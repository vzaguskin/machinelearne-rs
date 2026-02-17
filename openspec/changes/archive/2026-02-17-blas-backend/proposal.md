## Why

The pure-Rust CpuBackend uses naive O(n³) matrix multiplication and O(n²) matrix-vector operations. BLAS (Basic Linear Algebra Subprograms) libraries provide highly optimized implementations of these operations that can be 10-100x faster, especially for larger matrices. This is a quick win for CPU performance.

## What Changes

- Add `BlasBackend` implementing the `Backend` trait using BLAS for linear algebra
- Use `blas-src` crate to abstract over different BLAS implementations
- Support multiple BLAS backends: OpenBLAS, Netlib BLAS, Apple Accelerate
- Delegate element-wise operations to ndarray (already optimized)

## Capabilities

### New Capabilities

- `blas-backend`: CPU backend with optimized BLAS linear algebra

### Modified Capabilities

- None (additive only)

## Impact

- `lib/Cargo.toml`: Add blas-src dependency under `blas`, `blas-openblas`, `blas-netlib`, `blas-accelerate` features
- `lib/src/backend/mod.rs`: Add blas module and exports
- `lib/src/backend/blas.rs`: New BlasBackend implementation
- `CHANGELOG.md`: Document new capability
