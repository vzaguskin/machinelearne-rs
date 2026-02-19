# Tasks: ONNX Export Implementation

## Implementation Checklist

### Phase 1: Core Infrastructure

- [x] Add prost, prost-build dependencies to Cargo.toml
- [x] Create onnx module structure (mod.rs, error.rs)
- [x] Define ONNX protobuf types (proto.rs)
- [x] Implement OnnxGraphBuilder (graph.rs)
- [x] Create operator helper functions (operators.rs)
- [x] Define OnnxExportable trait (export.rs)

### Phase 2: Model Exports

- [x] Implement OnnxExportable for LinearModel<Fitted>
- [x] Add Gemm operator support
- [x] Implement StandardScaler export
- [x] Implement MinMaxScaler export
- [x] Implement RobustScaler export
- [x] Implement MaxAbsScaler export
- [x] Implement Normalizer export
- [x] Implement SimpleImputer export
- [x] Add placeholder for OneHotEncoder/OrdinalEncoder/PolynomialFeatures
- [x] Implement FittedPipeline export

### Phase 3: Inference Runtime (Experimental)

- [x] Add ort dependency with feature flag
- [x] Create OnnxInferenceSession skeleton
- [x] Document experimental status and manual setup requirements

### Phase 4: Testing & Documentation

- [x] Add unit tests for graph builder
- [x] Add unit tests for OnnxExportable trait
- [x] Add unit tests for LinearModel export
- [x] Create export_onnx.rs example
- [x] Create onnx_inference.rs benchmark
- [x] Mark onnx-inference as experimental in docs

### Phase 5: Finalization

- [x] Run cargo fmt
- [x] Verify all tests pass
- [x] Archive OpenSpec change
- [ ] Create pull request
- [ ] Merge and close issue
