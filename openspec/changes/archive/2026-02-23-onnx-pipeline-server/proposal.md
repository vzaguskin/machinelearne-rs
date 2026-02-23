## Why

The current ONNX implementation provides basic model export capabilities, but lacks essential features for production deployment. Users cannot export complete ML pipelines (preprocessing + model) as a single deployable file, have no inference server for model serving, lack comprehensive examples, and cannot evaluate performance trade-offs between backends. This limits the practical value of ONNX support for real-world deployment scenarios.

## What Changes

- **Full Pipeline Export**: Export complete preprocessing pipelines (scalers, transformers, feature engineering) combined with the model as a single ONNX file for end-to-end inference
- **Inference Server**: Add a lightweight HTTP/gRPC inference server for deploying trained models, supporting batch predictions and multiple model loading
- **Complete Example**: Provide a fully working example demonstrating: training a model with preprocessing → exporting to ONNX → deploying with inference server → making predictions
- **Performance Benchmarks**: Add comprehensive benchmarks comparing:
  - Native CPU backend (Rust)
  - ONNX Runtime CPU
  - ONNX Runtime GPU (CUDA)
  - Performance metrics: throughput, latency, memory usage

## Capabilities

### New Capabilities

- `onnx-pipeline-export`: Complete pipeline export combining preprocessing steps and model into single ONNX file
- `onnx-inference-server`: HTTP/gRPC server for model deployment with batch prediction support
- `onnx-deployment-example`: End-to-end example showing training → export → deploy → predict workflow
- `onnx-backend-benchmarks`: Performance comparison suite for CPU/GPU backends

### Modified Capabilities

- `onnx` (from onnx-export change): Extending existing ONNX export capability with pipeline export and inference server support

## Impact

- **New Code**:
  - `lib/src/onnx/pipeline_export.rs` - Pipeline serialization to ONNX
  - `lib/src/onnx/server.rs` - Inference server implementation
  - `lib/examples/onnx_deployment.rs` - Complete deployment example
  - `lib/benches/onnx_backend_comparison.rs` - Backend benchmarks
- **Dependencies**:
  - `tokio` for async server runtime
  - `axum` or `actix-web` for HTTP server
  - `tonic` for gRPC support (optional)
- **Features**: New `onnx-server` feature flag for server functionality
- **Examples**: New example demonstrating full deployment workflow
