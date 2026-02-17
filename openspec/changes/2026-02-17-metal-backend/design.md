## Context

Metal is Apple's low-level GPU API for macOS and iOS. Key advantages for ML:
- **Unified Memory**: CPU and GPU share the same memory on Apple Silicon
- **Metal Performance Shaders (MPS)**: Pre-built optimized ML kernels
- **Low overhead**: Direct GPU access with minimal API overhead
- **Wide availability**: All Macs since 2012 support Metal

The `metal-rs` crate provides Rust bindings to the Metal framework.

## Goals / Non-Goals

**Goals:**
- Implement `MetalBackend` satisfying the `Backend` trait
- Use MPS for matrix operations where available
- Write custom Metal shaders for other operations
- Support unified memory efficiently (avoid unnecessary copies)
- Multi-GPU support for Mac Pro configurations

**Non-Goals:**
- iOS support (focus on macOS first)
- Older macOS versions (require macOS 12+ for best MPS)
- Vulkan/MoltenVK fallback (use wgpu backend for that)

## Decisions

### Decision 1: Use MPS for linear algebra

Use Metal Performance Shaders for matmul, matvec, and reductions.

**Rationale:** MPS provides highly optimized kernels maintained by Apple. Best performance on Apple Silicon.

### Decision 2: Unified memory optimization

Use `MTLStorageModeShared` for tensors to enable zero-copy CPU access.

**Rationale:** Apple Silicon has unified memory. No need for explicit CPU-GPU copies.

### Decision 3: Metal Shading Language for custom shaders

Write custom shaders in MSL (Metal Shading Language), not compiled separately.

**Rationale:** Metal shaders compile at runtime from MSL source, avoiding complex build setup.

### Decision 4: Compile-time macOS guard

The `metal` feature only compiles on macOS targets.

**Rationale:** Metal is macOS/iOS only. Clear compile error on other platforms.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| macOS-only | Document platform requirement clearly |
| MPS API changes | Pin to stable MPS interfaces |
| Non-Apple Silicon Macs | Test on Intel Macs with discrete GPU |
| Build system complexity | Ship shaders as embedded strings |

## Open Questions

- Support for older Intel Macs with AMD GPUs?
- iOS support timing (same backend or separate)?
- How to handle MPS unavailability gracefully?
