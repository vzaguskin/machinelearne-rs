## Context

WGPU (WebGPU) is a cross-platform GPU API that targets Vulkan, Metal, D3D12, OpenGL, and WebGPU. It's written in pure Rust and used in production by Firefox, Bevy, and Servo. This makes it ideal for providing GPU acceleration across all major platforms from a single codebase.

The current Backend trait is synchronous, but GPU operations are inherently asynchronous. This requires careful design to maintain API compatibility.

## Goals / Non-Goals

**Goals:**
- Implement `WgpuBackend` satisfying the `Backend` trait
- Support all Backend trait methods with GPU acceleration
- Write compute shaders in WGSL for tensor operations
- Support device selection (multiple GPUs)
- Enable async operation where beneficial

**Non-Goals:**
- Auto-diff / gradient computation (future work)
- Multi-GPU distribution (future work)
- Training-specific optimizations (keep it general tensor ops)

## Decisions

### Decision 1: Blocking API with internal async

Keep the Backend trait synchronous by blocking on async GPU operations internally.

**Rationale:** This maintains compatibility with existing models and training code. Users don't need to change their code to use GPU acceleration.

```
// Backend::matvec blocks internally
fn matvec(a: &Tensor2D, x: &Tensor1D) -> Tensor1D {
    pollster::block_on(async {
        // Submit compute pass, wait for completion
    })
}
```

### Decision 2: WGSL compute shaders

Write tensor operations as WGSL compute shaders, not using wgpu's raw SPIR-V.

**Rationale:** WGSL is the portable shading language for wgpu. It's well-documented and the compiler handles optimization.

### Decision 3: Staging buffers for CPU-GPU transfer

Use staging buffers for all data transfer between CPU and GPU.

**Rationale:** Required by wgpu for proper memory management and async transfer.

### Decision 4: Shader specialization

Create specialized shaders for common operations rather than a single generic kernel.

**Rationale:** Better performance through optimized workgroup sizes and memory access patterns for each operation type.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Blocking async may cause stuttering | Document that GPU ops block; future async API |
| GPU memory limits | Fall back to CPU for large tensors; document limits |
| Shader compilation overhead | Cache compiled pipelines; warm-up on first use |
| WebGPU browser compatibility | Feature detection; graceful degradation message |

## Open Questions

- Should we expose async variants of operations? (Future API consideration)
- What's the minimum tensor size where GPU is faster than CPU? (Benchmark needed)
- How to handle out-of-memory on GPU? (Error type vs panic)
