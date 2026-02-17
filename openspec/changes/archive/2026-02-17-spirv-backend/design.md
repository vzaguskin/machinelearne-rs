## Context

rust-gpu (now maintained by the Rust-GPU organization after Embark Studios) compiles Rust to SPIR-V, the intermediate representation for Vulkan shaders. This allows writing GPU compute shaders in Rust instead of GLSL or HLSL.

Key components:
- `spirv-std`: Standard library for GPU Rust (thread indexing, etc.)
- `rustc_codegen_spirv`: rustc backend producing SPIR-V
- Vulkan runtime: vulkano (safe Rust) or ash (raw bindings)

## Goals / Non-Goals

**Goals:**
- Implement `SpirvBackend` satisfying the `Backend` trait
- Write compute kernels in Rust using spirv-std
- Support Vulkan devices via vulkano or ash
- Multi-GPU support via physical device selection

**Non-Goals:**
- Graphics rendering (compute only)
- WebGPU support (use wgpu backend)
- CUDA/Metal support (use those backends)
- Optimize for small tensors (CPU is better)

## Decisions

### Decision 1: Use vulkano for Vulkan API

Use vulkano crate for safe Vulkan bindings.

**Rationale:** Safer API than raw ash bindings, better documentation. Can switch to ash later if performance-critical.

### Decision 2: Compile kernels at build time

Use build.rs to compile Rust kernels to SPIR-V.

**Rationale:** Avoids runtime compilation overhead. Simpler deployment.

### Decision 3: Compute-only queues

Use compute-only Vulkan queues, no graphics.

**Rationale:** Simpler setup, no need for swapchain or rendering pipeline.

### Decision 4: Shader specialization constants

Use Vulkan specialization constants for tensor dimensions.

**Rationale:** Allows shader optimization for specific tensor sizes.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| rust-gpu is still maturing | Test extensively; document limitations |
| Complex build setup | Provide detailed build documentation |
| Vulkan driver availability | Document minimum driver versions |
| SPIR-V compatibility | Target Vulkan 1.0 for wide compatibility |

## Open Questions

- Use vulkano or ash for Vulkan bindings? (Start with vulkano)
- Minimum Vulkan version? (1.0 or 1.1?)
- How to handle shader compilation failures?
- Support for WebGPU via SPIR-V? (wgpu handles this already)
