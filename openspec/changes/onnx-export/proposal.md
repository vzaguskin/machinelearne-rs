# Proposal: ONNX Export and Inference Support

## Why

The current serialization system uses bincode, which is Rust-specific and not portable. Users cannot deploy trained models in other languages or leverage optimized inference runtimes like ONNX Runtime.

## What Changes

Add ONNX (Open Neural Network Exchange) export capability to enable:
1. **Portable model deployment** - Export models as single ONNX files usable in any language with ONNX Runtime
2. **Optimized inference** - Leverage ONNX Runtime's CPU/GPU acceleration
3. **Industry standard format** - Use the de facto standard for ML model interchange

## Alternatives Considered

1. **PMML** - Older format, less community support
2. **Custom format** - More work, no ecosystem benefits
3. **TensorFlow SavedModel** - Heavier dependency, less portable

## Scope

### In Scope
- ONNX export for LinearModel<Fitted>
- ONNX export for preprocessing transformers
- OnnxExportable trait for consistent API
- Example and benchmark
- Experimental ONNX Runtime inference support

### Out of Scope
- Training in ONNX format
- Full ONNX operator coverage
- GPU execution providers (CUDA, ROCm)

## Success Criteria

- Exported models load successfully in Python ONNX Runtime
- Predictions match native Rust within floating-point tolerance
- All tests pass with 85%+ coverage for new code
- Example demonstrates complete workflow

## Dependencies

- `prost` for protobuf encoding (ONNX format)
- `ort` for ONNX Runtime inference (optional, experimental)

## Risks

| Risk | Mitigation |
|------|------------|
| ort-sys download issues | Mark onnx-inference as experimental |
| ONNX API changes | Pin opset version to 17 |
| Complex transformer exports | Start with basic ops, iterate |
