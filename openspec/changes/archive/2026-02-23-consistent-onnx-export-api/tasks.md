## 1. Trait Definitions

- [x] 1.1 Create `lib/src/onnx/traits.rs` module with `OnnxNodeBuilder` trait definition
- [x] 1.2 Add `OnnxExportable` trait with `build_onnx_graph` method and default `to_onnx`/`save_onnx` implementations
- [x] 1.3 Export traits from `lib/src/onnx/mod.rs`

## 2. OnnxGraphBuilder Enhancements

- [x] 2.1 Add `unique_name(&mut self, prefix: &str) -> String` method for name uniquification
- [x] 2.2 Add `add_initializer(&mut self, name: &str, tensor: TensorProto)` method
- [x] 2.3 Add helper methods for common tensor types (float tensor, int tensor)
- [x] 2.4 Add name collision tracking to prevent duplicate names

## 3. Transformer OnnxNodeBuilder Implementations

- [x] 3.1 Implement `OnnxNodeBuilder` for `FittedStandardScaler` (Sub and Div operations)
- [x] 3.2 Implement `OnnxNodeBuilder` for `FittedMinMaxScaler` (min, scale, feature_range operations)
- [x] 3.3 Implement `OnnxNodeBuilder` for `FittedSimpleImputer` (Where/Replace for NaN handling)
- [x] 3.4 Add unit tests for each transformer's ONNX node generation

## 4. Model OnnxExportable Implementation

- [x] 4.1 Refactor `LinearModel<B, Fitted>` to implement `OnnxExportable` with `build_onnx_graph`
- [x] 4.2 Ensure backward compatibility with existing `to_onnx` and `save_onnx` methods
- [x] 4.3 Add unit tests for model export via new trait

## 5. Pipeline Export Refactoring

- [x] 5.1 Add `OnnxNodeBuilder` implementation for `FittedPipeline` that chains transformer nodes
- [x] 5.2 Replace hardcoded match statement in `from_pipeline.rs` with trait dispatch
- [x] 5.3 Add validation error when pipeline contains non-exportable transformer
- [x] 5.4 Add integration tests for pipeline export with multiple transformers

## 6. Error Handling

- [x] 6.1 Add `UnsupportedTransformer` error variant to `OnnxError`
- [x] 6.2 Add `GraphConstruction` error variant for graph building failures
- [x] 6.3 Update error messages to be descriptive and actionable

## 7. Documentation and Examples

- [x] 7.1 Add doc comments to `OnnxNodeBuilder` trait with example implementation
- [x] 7.2 Add doc comments to `OnnxExportable` trait with usage examples
- [x] 7.3 Update existing ONNX examples to work with new API (if needed)
- [x] 7.4 Update CHANGELOG.md with the new capability
