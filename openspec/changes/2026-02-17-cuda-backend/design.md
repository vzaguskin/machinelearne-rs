## Context

Rust-CUDA is an emerging project that compiles Rust directly to PTX code for NVIDIA GPUs. It consists of:
- `rustc_codegen_nvvm`: rustc backend targeting NVVM IR → PTX
- `cuda_std`: GPU-side intrinsics, thread indexing, warp operations
- `cust`: CPU-side CUDA (kernel launching, memory, streams)
- `cudnn`: Bindings to cuDNN for ML primitives

This allows writing GPU kernels in Rust rather than CUDA C++.

## Goals / Non-Goals

**Goals:**
- Implement `CudaBackend` satisfying the `Backend` trait
- Use cuDNN for optimized matmul and matvec operations
- Write custom kernels in Rust for non-cuDNN operations
- Support CUDA streams for async operations
- Multi-GPU support via device selection

**Non-Goals:**
- AMD GPU support (use ROCm backend separately)
- Automatic kernel optimization
- Training framework features (keep it tensor ops only)

## Decisions

### Decision 1: Use cuDNN where available

Use cuDNN for matmul, matvec, and other operations it supports.

**Rationale:** cuDNN is highly optimized by NVIDIA. Using it gives us the best performance for ML workloads without writing custom kernels.

### Decision 2: Custom Rust kernels for other operations

Write custom kernels in Rust using `cuda_std` for operations not in cuDNN.

**Rationale:** Gives us full coverage of Backend trait while staying in pure Rust.

### Decision 3: Stream-based async model

Support CUDA streams for overlapping compute and memory transfer.

**Rationale:** Essential for performance in real workloads. Default stream is synchronous.

### Decision 4: Build-time kernel compilation

Compile Rust → PTX at build time, not runtime.

**Rationale:** Simpler deployment, no JIT compilation overhead at runtime.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Rust-CUDA is early stage | Test extensively; have fallback to cudarc raw CUDA |
| Requires NVIDIA GPU + CUDA toolkit | Clear documentation of requirements |
| Build complexity | Provide Dockerfile with build environment |
| PTX compatibility across architectures | Build multiple PTX versions for different SM versions |

## Open Questions

- Minimum CUDA version required? (Aim for CUDA 11.x)
- Support for older GPU architectures (Pascal, Volta)?
- How to handle missing cuDNN gracefully?
