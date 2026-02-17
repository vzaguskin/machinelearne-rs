# ADR-0008: Backend Research and Prioritization

## Status

**Proposed** - 2026-02-17

## Context

The machinelearne-rs library currently supports two backends:
- **CpuBackend** (default, pure Rust)
- **NdarrayBackend** (ndarray ecosystem)

The Backend trait requires approximately 60 methods including:
- Tensor constructors (zeros, from_vec)
- Element-wise ops (add, sub, mul, div)
- Reductions (mean, sum)
- Math functions (exp, log, sigmoid, sqrt, abs, sign)
- Linear algebra (matvec, matmul, transpose)
- Column/row operations (col_mean, col_std, row_sum)
- Broadcasting operations

For future growth, particularly neural network training with GPU support, we need to evaluate and prioritize additional backend implementations.

### Design Constraints

- **No wrapper frameworks**: We want direct control over backend implementations, not frameworks like Burn or Candle
- **Pure Rust preferred**: Minimize external C/C++ dependencies where possible
- **Cross-platform**: Support for multiple GPU APIs (Vulkan, Metal, CUDA, WebGPU)

## Backend Options Research

### 1. WGPU (Cross-platform GPU)

**Source**: https://github.com/gfx-rs/wgpu

**Description**: Pure Rust, cross-platform GPU compute framework that targets Vulkan, Metal, D3D12, OpenGL, WebGL2, and WebGPU.

**Pros**:
- Cross-platform GPU compute
- Pure Rust, no external dependencies
- Runs in browser (WebGPU/WASM)
- First-class Metal support for Apple Silicon
- First-class Vulkan support
- Mature project (used in Firefox, Bevy, Servo)

**Cons**:
- Graphics-focused API, compute is secondary
- More verbose for ML operations
- No built-in BLAS/cuBLAS equivalents
- Requires writing WGSL compute shaders

**Best for**: Cross-platform GPU support, web deployment

### 2. BLAS (CPU Performance)

**Libraries**: OpenBLAS, Intel MKL, Apple Accelerate

**Description**: Industry-standard Basic Linear Algebra Subprograms for optimized CPU matrix operations.

**Pros**:
- Industry standard linear algebra
- Highly optimized CPU implementations
- Available everywhere (OpenBLAS is portable)
- Drop-in performance boost for matmul/matvec operations

**Cons**:
- Only linear algebra (no higher-level ops like sigmoid, exp)
- External C library dependency
- CPU-only

**Best for**: CPU performance, matrix operations

### 3. Rust-CUDA (NVIDIA GPU with Pure Rust Kernels)

**Source**: https://github.com/Rust-GPU/Rust-CUDA

**Description**: Compiles Rust directly to PTX code via `rustc_codegen_nvvm`. Write CUDA kernels in pure Rust.

**Key Crates**:
- `rustc_codegen_nvvm` - rustc backend targeting NVVM IR
- `cuda_std` - GPU-side functions, thread indexing
- `cust` - CPU-side CUDA (kernel launching, memory)
- `cudnn` - GPU-accelerated deep learning primitives

**Pros**:
- Write GPU kernels in pure Rust
- No CUDA C++ needed
- Access to cuDNN for optimized ML primitives
- Full control over kernel optimization

**Cons**:
- Early development (bugs, safety issues expected)
- NVIDIA-only
- Requires CUDA toolkit installation
- Complex build process

**Best for**: Custom CUDA kernels written in Rust, NVIDIA GPU training

### 4. Metal (Apple Silicon)

**Options**: metal-rs (Rust bindings to Metal)

**Description**: Native Apple GPU compute framework with first-class support on macOS and iOS.

**Pros**:
- Native Apple Silicon performance
- First-class on macOS/iOS
- Unified memory architecture (no CPU-GPU copy overhead)
- Metal Performance Shaders for ML ops

**Cons**:
- Apple-only (no cross-platform)
- Limited Rust ecosystem
- External dependency (Metal framework)

**Best for**: Apple Silicon optimization

### 5. rust-gpu (Vulkan Compute via SPIR-V)

**Source**: https://github.com/Rust-GPU/rust-gpu

**Description**: Compiles Rust to SPIR-V for Vulkan compute shaders. Now maintained by community after Embark Studios.

**Pros**:
- Cross-platform via Vulkan/SPIR-V
- Write compute shaders in Rust
- Mature codebase

**Cons**:
- Graphics/shader focused (not ML-specific)
- Requires Vulkan runtime
- Less ML ecosystem than CUDA
- Complex build process

**Best for**: Cross-platform GPU compute via Vulkan

## Decision

We will prioritize backend implementation in the following order:

### Priority 1: WGPU Backend (Cross-platform GPU)

**Rationale**: Single backend covers Vulkan, Metal, D3D12, and WebGPU. Pure Rust implementation with no external dependencies. Best cross-platform coverage with a single codebase.

**Implementation approach**:
1. Use wgpu for compute shaders
2. Implement core operations (matmul, element-wise, reductions)
3. Write WGSL compute shaders for tensor ops
4. Support async operation model required by wgpu

### Priority 2: BLAS Backend (CPU Performance)

**Rationale**: Drop-in CPU performance boost using optimized BLAS implementations (OpenBLAS/MKL/Accelerate). Quick win for CPU performance.

**Implementation approach**:
1. Use `blas-src` crate for BLAS abstraction
2. Implement matrix ops via BLAS calls (GEMM, GEMV)
3. Keep element-wise ops in Rust
4. Support multiple BLAS implementations via feature flags

### Priority 3: Rust-CUDA Backend (NVIDIA GPU)

**Rationale**: Enable NVIDIA GPU training with custom kernels written in pure Rust. Access to cuDNN for optimized ML primitives.

**Implementation approach**:
1. Use `cust` for CUDA driver API
2. Use `cuda_std` for writing kernels in Rust
3. Integrate `cudnn` for optimized ML primitives
4. Custom kernels for non-cuDNN operations

### Priority 4: Metal Backend (Apple Silicon)

**Rationale**: Native performance on Apple hardware (M1/M2/M3). Important for developer experience on macOS.

**Implementation approach**:
1. Use `metal-rs` for Metal bindings
2. Leverage Metal Performance Shaders (MPS) for ML ops
3. Custom compute shaders for ML-specific ops

### Priority 5: rust-gpu Backend (Vulkan Compute)

**Rationale**: Cross-platform GPU via Vulkan/SPIR-V with pure Rust kernels. Lower priority due to complexity and overlap with wgpu.

## Summary Table

| Backend | Platforms | Rust Purity | Effort | Performance | Priority |
|---------|-----------|-------------|--------|-------------|----------|
| WGPU | Cross (Vulkan/Metal/D3D12/WebGPU) | Pure | Medium | Good | **1** |
| BLAS | CPU (OpenBLAS/MKL/Accelerate) | FFI | Low | Excellent | **2** |
| Rust-CUDA | NVIDIA GPU | Pure Rust kernels | High | Excellent | **3** |
| Metal | Apple Silicon | FFI | Medium | Excellent | **4** |
| rust-gpu | Vulkan platforms | Pure Rust kernels | High | Good | **5** |

## Consequences

### Positive
- Clear roadmap for backend development
- WGPU provides maximum platform coverage with single implementation
- BLAS provides immediate CPU performance improvements
- Future-proof design for GPU acceleration

### Negative
- WGPU requires async API which may complicate the current synchronous Backend trait
- BLAS introduces external C library dependencies
- Rust-CUDA has complex build requirements and is NVIDIA-only
- Multiple backends increase maintenance burden

### Neutral
- Backend trait may need extension for async operations
- Documentation will need to cover multiple backend options
- CI/CD will need to handle different backend testing scenarios
