## Why

Apple Silicon (M1/M2/M3/M4) provides excellent ML performance through the Metal framework. With unified memory architecture, there's no CPU-GPU copy overhead. Metal Performance Shaders (MPS) provides optimized ML primitives. This backend gives native performance on macOS and iOS.

## What Changes

- Add `MetalBackend` implementing the `Backend` trait using Metal
- Use `metal-rs` crate for Rust bindings to Metal
- Leverage Metal Performance Shaders for ML operations
- Write custom Metal compute shaders for non-MPS operations
- Add feature flag `metal` to enable the backend (macOS only)

## Capabilities

### New Capabilities

- `metal-backend`: Apple Silicon GPU-accelerated tensor operations

### Modified Capabilities

- None (additive only)

## Impact

- `lib/Cargo.toml`: Add metal dependency under `metal` feature
- `lib/src/backend/mod.rs`: Add metal module and exports
- `lib/src/backend/metal.rs`: MetalBackend implementation
- `lib/src/backend/metal/shaders/`: Metal Shading Language (.metal) files
- `CHANGELOG.md`: Document new capability
