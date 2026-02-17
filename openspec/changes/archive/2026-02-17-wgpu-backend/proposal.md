## Why

The library currently only supports CPU backends (CpuBackend, NdarrayBackend). To support neural network training and large-scale ML workloads, we need GPU acceleration. WGPU provides cross-platform GPU compute that works on Vulkan, Metal, D3D12, and WebGPU from a single codebase - covering virtually all modern hardware.

## What Changes

- Add `WgpuBackend` implementing the `Backend` trait using wgpu compute shaders
- Add WGSL compute shaders for tensor operations (matmul, element-wise, reductions)
- Add async runtime support for GPU operations
- Add feature flag `wgpu` to enable the backend
- Add `WgpuDevice` type for device selection and management

## Capabilities

### New Capabilities

- `wgpu-backend`: GPU-accelerated tensor operations using wgpu compute shaders

### Modified Capabilities

- None (additive only)

## Impact

- `lib/Cargo.toml`: Add wgpu, futures, pollster dependencies under `wgpu` feature
- `lib/src/backend/mod.rs`: Add wgpu module and exports
- `lib/src/backend/wgpu.rs`: New WgpuBackend implementation
- `lib/src/backend/wgpu/shaders/`: WGSL compute shader files
- `CHANGELOG.md`: Document new capability
