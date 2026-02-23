## Context

The current ONNX implementation (`lib/src/onnx/`) provides basic model export via `OnnxExportable` trait and limited pipeline export via `export_pipeline_to_onnx()`. However:
- Pipeline export is not fully integrated or well-tested
- No inference server exists for model deployment
- No examples demonstrate the complete workflow
- No benchmarks exist to evaluate ONNX performance vs native backends

This design extends the ONNX module to support production deployment scenarios.

## Goals / Non-Goals

**Goals:**
- Enable exporting complete pipelines (preprocessing + model) as single ONNX files
- Provide a lightweight HTTP inference server for model deployment
- Create a complete, runnable example of the deployment workflow
- Add benchmarks comparing native CPU, ONNX CPU, and ONNX GPU performance

**Non-Goals:**
- gRPC support (can be added later if needed)
- Model versioning or A/B testing (out of scope)
- Auto-scaling or orchestration (use external tools like k8s)
- Model training within the server (inference only)

## Decisions

### D1: Pipeline Export Architecture

**Decision**: Extend existing `export_pipeline_to_onnx()` to be the primary pipeline export API, adding `OnnxExportable` implementation for `FittedPipeline<B>`.

**Rationale**: The current approach of separate export functions is correct, but we need to:
1. Make `FittedPipeline` implement `OnnxExportable` directly
2. Ensure all preprocessing transformers have ONNX export implementations
3. Add validation before export to catch unsupported transformers early

**Alternatives considered**:
- Create a separate `PipelineExporter` class - adds unnecessary complexity
- Export each transformer separately - defeats the purpose of single-file deployment

### D2: Inference Server Framework

**Decision**: Use `axum` as the HTTP framework with `tokio` for async runtime.

**Rationale**:
- `axum` is lightweight, well-maintained, and has excellent ergonomics
- `tokio` is the de-facto standard async runtime in Rust
- Both integrate well with the existing crate ecosystem
- Feature flag `onnx-server` allows optional inclusion

**Alternatives considered**:
- `actix-web` - heavier, more complex for our simple use case
- `warp` - less actively maintained
- `rocket` - requires nightly for some features

### D3: ONNX Runtime Execution Providers

**Decision**: Support CPU and CUDA execution providers via feature flags.

**Rationale**:
- CPU provider works everywhere and is the default
- CUDA provider enables GPU acceleration for users with NVIDIA GPUs
- Feature flags (`onnx-cuda`) allow optional GPU support without affecting non-GPU users

**Alternatives considered**:
- Support all providers (TensorRT, OpenVINO, ROCm) - too complex for initial implementation
- CPU-only - misses the performance comparison goal

### D4: Server Binary vs Library

**Decision**: Provide both a library API and a standalone binary.

**Rationale**:
- Library API allows embedding the server in larger applications
- Binary provides quick deployment without writing code
- Same codebase powers both

### D5: Benchmark Structure

**Decision**: Use Criterion.rs with custom benchmark harness comparing backends.

**Rationale**:
- Criterion is the standard Rust benchmarking library
- Custom harness allows fair comparison across different execution providers
- JSON output enables automated analysis

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| ONNX export of some transformers may be incomplete | Audit existing export functions, add missing implementations |
| GPU may not be available in CI | Make GPU tests optional, skip gracefully when unavailable |
| Server performance may not meet production needs | Document as MVP, prioritize correctness over optimization |
| ONNX Runtime version compatibility | Pin versions, test with multiple versions in CI |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ONNX Module (lib/src/onnx/)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ export.rs    │  │server.rs     │  │ inference.rs │      │
│  │ (pipeline)   │  │ (HTTP API)   │  │ (runtime)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ONNX Runtime (ort crate)                  │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ CPU Provider │  │ CUDA Provider│                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Migration Plan

This is a new feature with no breaking changes:
1. Add new modules behind feature flags
2. Update Cargo.toml with new dependencies
3. Add examples and benchmarks
4. Update documentation

No rollback needed as existing functionality is unchanged.

## Open Questions

1. **Model format versioning**: Should we support multiple ONNX opset versions in the server?
   - Recommendation: Start with default (17), add config option if needed

2. **Batch size limits**: Should the server enforce max batch size?
   - Recommendation: Add configurable limit with sensible default (1000)

3. **Authentication**: Should the server support API keys or auth?
   - Recommendation: Out of scope for MVP, document as future enhancement
