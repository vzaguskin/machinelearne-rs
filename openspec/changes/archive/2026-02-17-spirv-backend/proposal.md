## Why

rust-gpu compiles Rust directly to SPIR-V for Vulkan compute shaders. This enables:
1. Cross-platform GPU compute via Vulkan
2. Writing GPU kernels in pure Rust (no GLSL/SPIR-V by hand)
3. Leverage Rust's type system and safety for GPU code
4. Alternative to wgpu for users who prefer Vulkan directly

## What Changes

- Add `SpirvBackend` implementing the `Backend` trait using Vulkan
- Use rust-gpu to compile Rust kernels to SPIR-V
- Use vulkano or ash for Vulkan API bindings
- Add feature flag `spirv` to enable the backend

## Capabilities

### New Capabilities

- `spirv-backend`: Vulkan GPU compute with Rust-compiled SPIR-V kernels

### Modified Capabilities

- None (additive only)

## Impact

- `lib/Cargo.toml`: Add spirv-std, vulkano/ash dependencies under `spirv` feature
- `lib/src/backend/mod.rs`: Add spirv module and exports
- `lib/src/backend/spirv.rs`: SpirvBackend implementation
- `lib/src/backend/spirv/kernels/`: Rust kernel code compiled to SPIR-V
- `CHANGELOG.md`: Document new capability
