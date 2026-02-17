## Why

NVIDIA GPUs dominate ML training due to CUDA's mature ecosystem and cuDNN's optimized kernels. Rust-CUDA allows writing CUDA kernels in pure Rust (compiled to PTX), giving us:
1. Native NVIDIA GPU performance
2. Access to cuDNN for ML primitives
3. Pure Rust kernels without CUDA C++
4. Full control over kernel optimization

## What Changes

- Add `CudaBackend` implementing the `Backend` trait using CUDA
- Use `cust` crate for CUDA driver API
- Use `cuda_std` for writing kernels in Rust
- Integrate `cudnn` for optimized ML primitives (matmul, etc.)
- Add feature flag `cuda` to enable the backend

## Capabilities

### New Capabilities

- `cuda-backend`: NVIDIA GPU-accelerated tensor operations with Rust kernels

### Modified Capabilities

- None (additive only)

## Impact

- `lib/Cargo.toml`: Add cust, cudarc dependencies under `cuda` feature
- `lib/src/backend/mod.rs`: Add cuda module and exports
- `lib/src/backend/cuda.rs`: CudaBackend implementation
- `lib/src/backend/cuda/kernels/`: Rust CUDA kernels
- `CHANGELOG.md`: Document new capability
